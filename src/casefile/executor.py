"""Run a case's steps so that a crash anywhere resumes to the same result.

The rules, each of which the crash matrix tests:

* **A completed step never runs again.** Its recorded output is returned instead.
* **A recorded call is never made again.** Inside a step, every model or tool call is appended
  to the log *before* its result is used, keyed by a hash of the request. On replay the recorded
  response is returned. A call that returned but was not yet recorded when the process died is
  made again; nothing can avoid that, and the matrix counts how often it happens.
* **An effect happens once at the destination.** An effect is recorded as intended, then sent
  with an idempotency key, then recorded as confirmed. On resume an unconfirmed intent is sent
  again with the same key, and the destination deduplicates. Delivery is at least once; the
  effect is once. If the request a resumed step builds differs from the recorded intent, the run
  stops: the input changed, and resending the old request or sending a new one blindly would both
  be wrong.

Every append passes the seq the runner last saw, so a second runner on the same case fails with
``AppendConflict`` instead of interleaving its history with this one's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from casefile.canonical import JsonValue, canonical_json, sha256_hex
from casefile.faults import CrashPoint, Faults, NoFaults
from casefile.log import EventLog
from casefile.state import CaseState, EventType, apply, fold

type Call = Callable[[JsonValue], JsonValue]
"""A model or tool: request in, response out. Both must be canonical JSON values."""


class Destination(Protocol):
    """Somewhere outside the system that an effect is sent to, deduplicating by idempotency key."""

    def send(self, key: str, request: JsonValue) -> JsonValue:
        """Perform the effect for ``key`` at most once; return its receipt, the same every time."""
        ...


class ChangedInput(RuntimeError):
    """A resumed step asked for an effect whose request differs from the one already intended."""


def request_hash(request: JsonValue) -> str:
    """Return the hash a call's request is recorded under."""
    return sha256_hex(canonical_json(request))


def idempotency_key(case_id: str, step_id: str, kind: str) -> str:
    """Return the key an effect is sent under: one effect of each kind per step per case."""
    return sha256_hex(canonical_json([case_id, step_id, kind]))


class Runner:
    """Drive one case forward from wherever its log says it is."""

    def __init__(self, log: EventLog, case_id: str, faults: Faults | None = None) -> None:
        """Read the case's log and fold it; a case with no events is opened by ``open``."""
        self._log = log
        self._case_id = case_id
        self._faults: Faults = faults if faults is not None else NoFaults()
        events = log.read(case_id)
        self._state: CaseState | None = fold(events) if events else None

    @property
    def state(self) -> CaseState:
        """The case's current state. Raises if the case has not been opened."""
        if self._state is None:
            raise RuntimeError(f"case {self._case_id!r} has not been opened")
        return self._state

    def open(self, inputs: Mapping[str, JsonValue]) -> CaseState:
        """Open the case with ``inputs`` unless it is already open, and return its state."""
        if self._state is None:
            self._append(EventType.CASE_OPENED, {"inputs": dict(inputs)})
        return self.state

    def step(self, step_id: str, body: Callable[[StepContext], JsonValue]) -> JsonValue:
        """Run ``body`` as step ``step_id`` unless it already completed; return its output.

        ``body`` must be deterministic given the case's state and the responses of the calls it
        makes through its context. That is what makes replay give the same answer.
        """
        if step_id in self.state.outputs:
            return self.state.outputs[step_id]
        self._faults.hit(CrashPoint.BEFORE_STEP)
        if step_id not in self.state.started:
            self._append(EventType.STEP_STARTED, {}, step_id)
        output = body(StepContext(step_id, lambda: self.state, self._append, self._faults))
        self._faults.hit(CrashPoint.BEFORE_STEP_COMPLETED)
        self._append(EventType.STEP_COMPLETED, {"output": output}, step_id)
        return output

    def complete(self, outcome: JsonValue) -> CaseState:
        """Close the case with ``outcome`` unless it is already closed, and return its state."""
        if not self.state.completed:
            self._faults.hit(CrashPoint.BEFORE_CASE_COMPLETED)
            self._append(EventType.CASE_COMPLETED, {"outcome": outcome})
        return self.state

    def _append(
        self, kind: EventType, payload: Mapping[str, JsonValue], step_id: str | None = None
    ) -> None:
        """Append an event fenced on the last seq seen, then advance the in-memory state."""
        expected = 0 if self._state is None else self._state.last_seq
        event = self._log.append(
            self._case_id, kind, payload, step_id=step_id, expected_seq=expected
        )
        self._state = apply(self._state, event)


class StepContext:
    """What a step body uses to reach outside: recorded calls and outbox effects."""

    def __init__(
        self,
        step_id: str,
        state: Callable[[], CaseState],
        append: Callable[[EventType, Mapping[str, JsonValue], str | None], None],
        faults: Faults,
    ) -> None:
        """Bind the context to one running step of a runner, through the runner's own hooks."""
        self._step_id = step_id
        self._state = state
        self._append = append
        self._faults = faults

    def call(self, kind: str, request: JsonValue, call: Call) -> JsonValue:
        """Return the recorded response to ``request``, making and recording the call if needed."""
        key = (self._step_id, request_hash(request))
        calls = self._state().calls
        if key in calls:
            return calls[key]
        response = call(request)
        self._faults.hit(CrashPoint.AFTER_CALL_BEFORE_RECORD)
        self._append(
            EventType.CALL_RECORDED,
            {"kind": kind, "request_hash": key[1], "response": response},
            self._step_id,
        )
        self._faults.hit(CrashPoint.AFTER_RECORD)
        return response

    def effect(self, kind: str, request: JsonValue, destination: Destination) -> JsonValue:
        """Perform an effect once at ``destination`` and return its receipt.

        Raises ``ChangedInput`` if an earlier, unconfirmed attempt intended a different request.
        """
        state = self._state()
        key = idempotency_key(state.case_id, self._step_id, kind)
        if key in state.confirmed:
            return state.confirmed[key]
        if key in state.intended:
            if canonical_json(state.intended[key]) != canonical_json(request):
                raise ChangedInput(f"step {self._step_id!r}: {kind} request changed since intent")
        else:
            intent: dict[str, JsonValue] = {"key": key, "kind": kind, "request": request}
            self._append(EventType.EFFECT_INTENDED, intent, self._step_id)
            self._faults.hit(CrashPoint.AFTER_INTENT_BEFORE_SEND)
        receipt = destination.send(key, request)
        self._faults.hit(CrashPoint.AFTER_SEND_BEFORE_CONFIRM)
        self._append(EventType.EFFECT_CONFIRMED, {"key": key, "receipt": receipt}, self._step_id)
        return receipt
