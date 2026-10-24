"""A case's state is a fold over its events.

``fold`` is pure: no clock, no I/O, no randomness, so the same events always give the same state,
and recovery after a crash is just reading the log again. It is also strict. An event that could
not have been written by a correct run (a step completed before it started, an effect confirmed
that was never intended, anything after the case closed, a type or schema version this code does
not know) raises ``FoldError`` rather than being skipped. Skipping would let a damaged or forged log
fold into a plausible-looking state.

The vocabulary here is the durable core's: opening a case, steps, recorded calls, outbox effects
and completion. Review decisions and budgets join it in later work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from casefile.canonical import JsonValue
from casefile.events import Event, first_broken_link


class EventType(StrEnum):
    """Every event type the fold understands, as stored in the ``type`` column."""

    CASE_OPENED = "CaseOpened"
    STEP_STARTED = "StepStarted"
    CALL_RECORDED = "CallRecorded"
    STEP_COMPLETED = "StepCompleted"
    EFFECT_INTENDED = "EffectIntended"
    EFFECT_CONFIRMED = "EffectConfirmed"
    CASE_COMPLETED = "CaseCompleted"


SCHEMA_VERSION = 1
"""The only payload schema version this code reads. A newer one fails closed."""


class FoldError(ValueError):
    """The events do not describe a history a correct run could have written."""


@dataclass(frozen=True)
class CaseState:
    """Everything a run needs to know to carry on, and nothing that depends on the wall clock.

    ``calls`` is keyed by ``(step_id, request_hash)`` so a step that makes the same request twice
    on purpose gets one recorded answer, the same one on replay.
    """

    case_id: str
    last_seq: int
    inputs: Mapping[str, JsonValue]
    started: frozenset[str] = frozenset()
    outputs: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    calls: Mapping[tuple[str, str], JsonValue] = field(
        default_factory=dict[tuple[str, str], JsonValue]
    )
    intended: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    confirmed: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    completed: bool = False
    outcome: JsonValue = None

    def in_progress(self, step_id: str) -> bool:
        """Say whether ``step_id`` has started and not yet completed."""
        return step_id in self.started and step_id not in self.outputs


def fold(events: Iterable[Event]) -> CaseState:
    """Fold a case's events, oldest first, into its state; raise ``FoldError`` if they cannot be.

    The hash chain is checked first, so a tampered log is reported as tampering, not as whatever
    odd sequence of events the tampering happened to produce.
    """
    history = list(events)
    if not history:
        raise FoldError("a case needs at least a CaseOpened event")
    broken = first_broken_link(history)
    if broken is not None:
        raise FoldError(f"hash chain broken at seq {broken}")

    state = apply(None, history[0])
    for event in history[1:]:
        state = apply(state, event)
    return state


def apply(state: CaseState | None, event: Event) -> CaseState:
    """Return the state after one more event. ``None`` is the state before the case was opened.

    This does not check the hash chain; ``fold`` does. The executor uses ``apply`` to advance its
    in-memory state after an append it has just made itself.
    """
    kind = _event_type(event)
    if state is None:
        if kind is not EventType.CASE_OPENED:
            raise FoldError(f"seq {event.seq}: a case must start with CaseOpened, not {kind}")
        return CaseState(case_id=event.case_id, last_seq=event.seq, inputs=_object(event, "inputs"))
    if event.case_id != state.case_id:
        raise FoldError(f"seq {event.seq}: event for case {event.case_id!r} in {state.case_id!r}")
    if state.completed:
        raise FoldError(f"seq {event.seq}: {kind} after the case completed")
    return replace(_APPLY[kind](state, event), last_seq=event.seq)


def _event_type(event: Event) -> EventType:
    """Return the event's type if this code understands it at this schema version."""
    try:
        kind = EventType(event.type)
    except ValueError:
        raise FoldError(f"seq {event.seq}: unknown event type {event.type!r}") from None
    if event.schema_version != SCHEMA_VERSION:
        raise FoldError(
            f"seq {event.seq}: {kind} schema version {event.schema_version} unsupported"
        )
    return kind


def _opened_twice(state: CaseState, event: Event) -> CaseState:
    raise FoldError(f"seq {event.seq}: case opened twice")


def _step_started(state: CaseState, event: Event) -> CaseState:
    step = _step(event)
    if step in state.started:
        raise FoldError(f"seq {event.seq}: step {step!r} started twice")
    return replace(state, started=state.started | {step})


def _call_recorded(state: CaseState, event: Event) -> CaseState:
    step = _running_step(state, event)
    key = (step, _text(event, "request_hash"))
    if key in state.calls:
        raise FoldError(f"seq {event.seq}: call {key[1][:12]} in step {step!r} recorded twice")
    _text(event, "kind")
    return replace(state, calls={**state.calls, key: _value(event, "response")})


def _step_completed(state: CaseState, event: Event) -> CaseState:
    step = _running_step(state, event)
    return replace(state, outputs={**state.outputs, step: _value(event, "output")})


def _effect_intended(state: CaseState, event: Event) -> CaseState:
    _running_step(state, event)
    key = _text(event, "key")
    if key in state.intended:
        raise FoldError(f"seq {event.seq}: effect {key[:12]} intended twice")
    _text(event, "kind")
    return replace(state, intended={**state.intended, key: _value(event, "request")})


def _effect_confirmed(state: CaseState, event: Event) -> CaseState:
    key = _text(event, "key")
    if key not in state.intended:
        raise FoldError(f"seq {event.seq}: effect {key[:12]} confirmed but never intended")
    if key in state.confirmed:
        raise FoldError(f"seq {event.seq}: effect {key[:12]} confirmed twice")
    return replace(state, confirmed={**state.confirmed, key: _value(event, "receipt")})


def _case_completed(state: CaseState, event: Event) -> CaseState:
    unfinished = sorted(state.started - state.outputs.keys())
    if unfinished:
        raise FoldError(f"seq {event.seq}: case completed with steps still running: {unfinished}")
    unconfirmed = sorted(state.intended.keys() - state.confirmed.keys())
    if unconfirmed:
        raise FoldError(f"seq {event.seq}: case completed with {len(unconfirmed)} unsent effect(s)")
    return replace(state, completed=True, outcome=_value(event, "outcome"))


_APPLY = {
    EventType.CASE_OPENED: _opened_twice,
    EventType.STEP_STARTED: _step_started,
    EventType.CALL_RECORDED: _call_recorded,
    EventType.STEP_COMPLETED: _step_completed,
    EventType.EFFECT_INTENDED: _effect_intended,
    EventType.EFFECT_CONFIRMED: _effect_confirmed,
    EventType.CASE_COMPLETED: _case_completed,
}


def _step(event: Event) -> str:
    """Return the event's step id, which step events must carry."""
    if not event.step_id:
        raise FoldError(f"seq {event.seq}: {event.type} needs a step_id")
    return event.step_id


def _running_step(state: CaseState, event: Event) -> str:
    """Return the event's step id if that step has started and not completed."""
    step = _step(event)
    if not state.in_progress(step):
        raise FoldError(f"seq {event.seq}: {event.type} for step {step!r}, which is not running")
    return step


def _value(event: Event, name: str) -> JsonValue:
    """Return a required payload field of any JSON type."""
    if name not in event.payload:
        raise FoldError(f"seq {event.seq}: {event.type} payload lacks {name!r}")
    return event.payload[name]


def _text(event: Event, name: str) -> str:
    """Return a required, non-empty string payload field."""
    value = _value(event, name)
    if not isinstance(value, str) or not value:
        raise FoldError(f"seq {event.seq}: {event.type} field {name!r} must be a non-empty string")
    return value


def _object(event: Event, name: str) -> Mapping[str, JsonValue]:
    """Return a required JSON-object payload field."""
    value = _value(event, name)
    if not isinstance(value, dict):
        raise FoldError(f"seq {event.seq}: {event.type} field {name!r} must be an object")
    return value
