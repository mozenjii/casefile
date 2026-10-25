"""Tests for the fold: the state a correct history produces, and every history it refuses."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from casefile.canonical import CanonicalJsonError, JsonValue
from casefile.events import Event
from casefile.log import EventLog
from casefile.state import CaseState, EventType, FoldError, apply, fold

E = EventType
NOW = datetime(2026, 1, 2, tzinfo=UTC)

type Spec = tuple[EventType | str, str | None, dict[str, JsonValue]]


@pytest.fixture
def log(tmp_path: Path) -> Iterator[EventLog]:
    """Yield an empty log with a fixed clock."""
    with EventLog(tmp_path / "events.sqlite", clock=lambda: NOW) as opened:
        yield opened


def write(log: EventLog, specs: list[Spec]) -> list[Event]:
    """Append ``(type, step_id, payload)`` specs to case ``c`` and return the stored events."""
    for kind, step, payload in specs:
        log.append("c", str(kind), payload, step_id=step)
    return log.read("c")


OPEN: Spec = (E.CASE_OPENED, None, {"inputs": {"member": "M-1"}})
START: Spec = (E.STEP_STARTED, "s/1", {})
CALL: Spec = (E.CALL_RECORDED, "s/1", {"kind": "model", "request_hash": "h1", "response": "r"})
INTEND: Spec = (E.EFFECT_INTENDED, "s/1", {"key": "k1", "kind": "rfi", "request": {"to": "p"}})
CONFIRM: Spec = (E.EFFECT_CONFIRMED, "s/1", {"key": "k1", "receipt": {"id": 7}})
DONE: Spec = (E.STEP_COMPLETED, "s/1", {"output": {"ok": True}})
CLOSE: Spec = (E.CASE_COMPLETED, None, {"outcome": "approve"})


def test_a_full_history_folds_to_the_expected_state(log: EventLog) -> None:
    state = fold(write(log, [OPEN, START, CALL, INTEND, CONFIRM, DONE, CLOSE]))
    assert state == CaseState(
        case_id="c",
        last_seq=7,
        inputs={"member": "M-1"},
        started=frozenset({"s/1"}),
        outputs={"s/1": {"ok": True}},
        calls={("s/1", "h1"): "r"},
        intended={"k1": {"to": "p"}},
        confirmed={"k1": {"id": 7}},
        completed=True,
        outcome="approve",
        cost_unknown=True,  # a call recorded with no price and no estimate
        revised_at=7,  # the dossier was written by CaseCompleted at seq 7
    )


def test_a_partial_history_shows_the_step_in_progress(log: EventLog) -> None:
    state = fold(write(log, [OPEN, START, CALL]))
    assert state.in_progress("s/1")
    assert not state.completed
    assert state.last_seq == 3


def test_folding_twice_gives_equal_states(log: EventLog) -> None:
    events = write(log, [OPEN, START, CALL, DONE])
    assert fold(events) == fold(list(events))


REFUSED: list[tuple[list[Spec], str]] = [
    ([START], "must start with CaseOpened"),
    ([OPEN, OPEN], "opened twice"),
    ([OPEN, START, START], "started twice"),
    ([OPEN, CALL], "not running"),
    ([OPEN, DONE], "not running"),
    ([OPEN, START, DONE, DONE], "not running"),
    ([OPEN, START, CALL, CALL], "recorded twice"),
    ([OPEN, INTEND], "not running"),
    ([OPEN, START, INTEND, INTEND], "intended twice"),
    ([OPEN, START, CONFIRM], "never intended"),
    ([OPEN, START, INTEND, CONFIRM, CONFIRM], "confirmed twice"),
    ([OPEN, START, CLOSE], "steps still running"),
    ([OPEN, START, INTEND, DONE, CLOSE], "unsent effect"),
    ([OPEN, CLOSE, START], "after the case completed"),
    ([OPEN, ("Mystery", None, {})], "unknown event type"),
    ([OPEN, (E.STEP_STARTED, None, {})], "needs a step_id"),
    ([OPEN, START, (E.STEP_COMPLETED, "s/1", {})], "lacks 'output'"),
    ([OPEN, START, (E.CALL_RECORDED, "s/1", {"request_hash": "", "response": 1})], "non-empty"),
    ([(E.CASE_OPENED, None, {"inputs": []})], "must be an object"),
]


@pytest.mark.parametrize(("specs", "message"), REFUSED)
def test_impossible_histories_are_refused(log: EventLog, specs: list[Spec], message: str) -> None:
    with pytest.raises(FoldError, match=message):
        fold(write(log, specs))


def test_an_empty_history_is_refused() -> None:
    with pytest.raises(FoldError, match="at least a CaseOpened"):
        fold([])


def test_an_unsupported_schema_version_fails_closed(log: EventLog) -> None:
    log.append("c", E.CASE_OPENED, {"inputs": {}}, schema_version=2)
    with pytest.raises(FoldError, match="schema version 2 unsupported"):
        fold(log.read("c"))


def test_a_tampered_history_is_reported_as_tampering(log: EventLog) -> None:
    events = write(log, [OPEN, START, DONE])
    events[1] = dataclasses.replace(events[1], step_id="s/2")
    with pytest.raises(FoldError, match="hash chain broken at seq 2"):
        fold(events)


def test_apply_refuses_another_cases_event(log: EventLog, tmp_path: Path) -> None:
    state = fold(write(log, [OPEN]))
    stranger = log.append("other", E.CASE_OPENED, {"inputs": {}})
    with pytest.raises(FoldError, match="event for case 'other'"):
        apply(state, dataclasses.replace(stranger, seq=2))


# Review and release. CLOSE's outcome, the string "approve", is the dossier; its hash is computed
# here independently of the fold.
DOSSIER_HASH = hashlib.sha256(b'"approve"').hexdigest()
REVISED_HASH = hashlib.sha256(b'{"note":"edited"}').hexdigest()
REVISE: Spec = (E.DOSSIER_REVISED, None, {"dossier": {"note": "edited"}, "editor": "human:ana"})
UNREVISE: Spec = (E.DOSSIER_REVISED, None, {"dossier": "approve", "editor": "human:ana"})
BASE: list[Spec] = [OPEN, START, DONE, CLOSE]


def decide(decision: str, seen: str = DOSSIER_HASH, reviewer: str = "human:ana") -> Spec:
    """A reviewer's decision on the dossier with hash ``seen``."""
    payload: dict[str, JsonValue] = {
        "decision": decision,
        "dossier_hash": seen,
        "reviewer": reviewer,
    }
    return (E.REVIEWER_DECISION, None, payload)


def release(named: str = DOSSIER_HASH, releaser: str = "human:ben") -> Spec:
    """A release of the dossier with hash ``named``."""
    return (E.CASE_RELEASED, None, {"dossier_hash": named, "releaser": releaser})


def test_an_approved_dossier_can_be_released(log: EventLog) -> None:
    state = fold(write(log, [*BASE, decide("approve"), release()]))
    assert state.dossier_hash == DOSSIER_HASH
    assert state.approved
    assert state.released


def test_there_is_no_dossier_hash_before_the_run_completes(log: EventLog) -> None:
    assert fold(write(log, [OPEN, START])).dossier_hash is None


def test_revising_back_to_approved_content_does_not_bring_the_approval_back(
    log: EventLog,
) -> None:
    with pytest.raises(FoldError, match="without an approval"):
        fold(write(log, [*BASE, decide("approve"), REVISE, UNREVISE, release()]))


def test_a_stale_decision_is_kept_but_authorizes_nothing(log: EventLog) -> None:
    state = fold(write(log, [*BASE, REVISE, decide("approve")]))
    assert state.decisions == {DOSSIER_HASH: "approve"}
    assert not state.approved


REFUSED_REVIEWS: list[tuple[list[Spec], str]] = [
    ([*BASE, release()], "without an approval"),
    ([*BASE, decide("approve"), release(releaser="agent:drafter")], "must be a human"),
    ([*BASE, decide("approve"), (E.CASE_RELEASED, None, {"dossier_hash": DOSSIER_HASH})], "lacks"),
    ([*BASE, decide("approve"), REVISE, release()], "not the current dossier"),
    ([*BASE, decide("approve"), REVISE, release(REVISED_HASH)], "without an approval"),
    ([*BASE, decide("approve"), decide("reject"), release()], "without an approval"),
    ([*BASE, decide("approve"), decide("request_info"), release()], "without an approval"),
    ([*BASE, decide("approve", reviewer="agent:drafter")], "must be a human"),
    ([*BASE, decide("approve", reviewer="human:")], "must be a human"),
    ([*BASE, decide("rubber-stamp")], "unknown decision"),
    ([OPEN, START, decide("approve")], "before there is a dossier"),
    ([*BASE, decide("approve"), release(), REVISE], "after the case was released"),
    ([*BASE, decide("approve"), release(), release()], "after the case was released"),
    ([*BASE, START], "after the case completed"),
]


@pytest.mark.parametrize(("specs", "message"), REFUSED_REVIEWS)
def test_impossible_reviews_are_refused(log: EventLog, specs: list[Spec], message: str) -> None:
    with pytest.raises(FoldError, match=message):
        fold(write(log, specs))


STARTED: Spec = (
    E.CALL_STARTED,
    "s/1",
    {"kind": "model", "request_hash": "h1", "estimate_microusd": 300},
)
PRICED: Spec = (
    E.CALL_RECORDED,
    "s/1",
    {"kind": "model", "request_hash": "h1", "response": {"usage": {"cost_microusd": 410}}},
)


def test_spend_is_the_real_cost_once_recorded_and_the_estimate_while_lost(log: EventLog) -> None:
    lost = fold(write(log, [OPEN, START, STARTED, STARTED]))
    assert (lost.model_calls, lost.lost_calls, lost.spent_microusd) == (2, 2, 600)
    log.append("c", E.CALL_RECORDED, PRICED[2], step_id="s/1")
    recorded = fold(log.read("c"))
    assert (recorded.model_calls, recorded.lost_calls, recorded.spent_microusd) == (2, 1, 710)
    assert not recorded.cost_unknown


def test_a_recorded_call_without_a_price_is_charged_its_estimate(log: EventLog) -> None:
    unpriced: Spec = (
        E.CALL_RECORDED,
        "s/1",
        {"kind": "model", "request_hash": "h1", "response": "r"},
    )
    state = fold(write(log, [OPEN, START, STARTED, unpriced]))
    assert (state.spent_microusd, state.cost_unknown) == (300, False)


def test_a_budget_stop_is_recorded_with_its_reason(log: EventLog) -> None:
    stop: Spec = (E.BUDGET_EXHAUSTED, None, {"reason": "model call limit reached (2 of 2)"})
    state = fold(write(log, [OPEN, stop, CLOSE]))
    assert state.stopped == "model call limit reached (2 of 2)"


def test_a_malformed_estimate_is_refused(log: EventLog) -> None:
    floating = cast(JsonValue, 2.5)  # a float is exactly what the log must refuse
    bad: Spec = (
        E.CALL_STARTED,
        "s/1",
        {"kind": "model", "request_hash": "h1", "estimate_microusd": floating},
    )
    with pytest.raises(CanonicalJsonError):
        write(log, [OPEN, START, bad])
    # The refused float left OPEN and START in the log; append the next bad start after them.
    words: dict[str, JsonValue] = {
        "kind": "model",
        "request_hash": "h1",
        "estimate_microusd": "lots",
    }
    log.append("c", E.CALL_STARTED, words, step_id="s/1")
    with pytest.raises(FoldError, match="must be an integer or null"):
        fold(log.read("c"))
