"""Tests for the three-design comparison."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.canonical import JsonValue
from casefile.compare import DESIGNS, Result, run_design, summarise, wilson
from casefile.fakes import Ledger
from casefile.generator import DEFECTS, generate_set
from casefile.log import EventLog
from casefile.reference_model import reference_model

CASES = generate_set(len(DEFECTS), seed=900)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[tuple[EventLog, Ledger]]:
    """An empty log and ledger."""
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(tmp_path / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        yield log, ledger
    ledger.close()


def run_all(
    world: tuple[EventLog, Ledger], model: Callable[[JsonValue], JsonValue], design: str
) -> list[Result]:
    """Run one design over every test case."""
    log, ledger = world
    return [run_design(design, case, model, log, ledger, lambda _: 100) for case in CASES]


@pytest.mark.parametrize(("design", "calls"), [("single_agent", 1.0), ("fixed_workflow", 1.0)])
def test_the_one_call_designs_make_one_call(
    world: tuple[EventLog, Ledger], design: str, calls: float
) -> None:
    results = run_all(world, reference_model, design)
    assert all(r.correct for r in results)
    assert sum(r.calls for r in results) / len(results) == calls


def test_the_specialists_make_one_call_per_document_and_a_draft(
    world: tuple[EventLog, Ledger],
) -> None:
    results = run_all(world, reference_model, "specialists")
    assert all(r.correct for r in results)
    assert [r.calls for r in results] == [len(c.documents) + 1 for c in CASES]
    assert all(r.cost_microusd == 100 * r.calls for r in results)


def test_an_agent_that_always_approves_is_caught_approving_unsafely(
    world: tuple[EventLog, Ledger],
) -> None:
    def reckless(request: JsonValue) -> JsonValue:
        reply = reference_model(request)
        if (
            isinstance(request, dict)
            and request.get("task") == "decide_case"
            and isinstance(reply, dict)
        ):
            return {**reply, "recommendation": "approve"}
        return reply

    results = run_all(world, reckless, "single_agent")
    unsafe = [r for r in results if r.unsafe]
    assert len(unsafe) == sum(1 for c in CASES if c.label.recommendation.value != "approve")
    assert "| single_agent | 9 | 22% " in summarise(results)


def test_omissions_lower_recall_while_the_sweep_keeps_decisions_right(
    world: tuple[EventLog, Ledger],
) -> None:
    def forgetful(request: JsonValue) -> JsonValue:
        reply = reference_model(request)
        if (
            isinstance(request, dict)
            and request.get("task") == "extract_case"
            and isinstance(reply, dict)
        ):
            observations = reply["observations"]
            assert isinstance(observations, list)
            kept = [o for o in observations if isinstance(o, dict) and o["fact"] != "therapy_weeks"]
            return {"observations": list[JsonValue](kept)}
        return reply

    results = run_all(world, forgetful, "fixed_workflow")
    assert all(r.correct for r in results)
    assert all(r.evidence_recall < 1.0 for r in results)


def test_an_answer_outside_the_allowed_three_counts_as_wrong(
    world: tuple[EventLog, Ledger],
) -> None:
    def creative(request: JsonValue) -> JsonValue:
        reply = reference_model(request)
        if isinstance(reply, dict) and "recommendation" in reply:
            return {**reply, "recommendation": "probably fine"}
        return reply

    results = run_all(world, creative, "single_agent")
    assert {r.recommendation for r in results} == {"invalid"}
    assert not any(r.correct for r in results)


@pytest.mark.parametrize(
    ("successes", "trials", "low", "high"),
    [
        (27, 27, 0.8754, 1.0),  # all succeed: low = n / (n + z^2) = 27 / 30.8416
        (0, 10, 0.0, 0.2775),  # none succeed: high = z^2 / (n + z^2) = 3.8416 / 13.8416
        (5, 10, 0.2366, 0.7634),  # the usual textbook example
        (0, 0, 0.0, 1.0),
    ],
)
def test_the_wilson_interval_matches_values_worked_out_by_hand(
    successes: int, trials: int, low: float, high: float
) -> None:
    got_low, got_high = wilson(successes, trials)
    assert (round(got_low, 4), round(got_high, 4)) == (low, high)


def test_an_unknown_design_is_refused(world: tuple[EventLog, Ledger]) -> None:
    log, ledger = world
    with pytest.raises(ValueError, match="unknown design"):
        run_design("committee", CASES[0], reference_model, log, ledger, lambda _: None)
    assert DESIGNS == ("single_agent", "fixed_workflow", "specialists")
