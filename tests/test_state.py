"""Tests for the fold: the state a correct history produces, and every history it refuses."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.canonical import JsonValue
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
