"""In-process tests for the executor: replay, every crash point, and the refusal paths.

``RaiseAt`` stands in for a crash here so each path runs in milliseconds and counts toward
coverage. The subprocess crash matrix repeats the exercise with real process kills.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.canonical import JsonValue
from casefile.executor import ChangedInput, Runner, StepContext
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.faults import CrashPoint, Faults, HitRecorder, RaiseAt, SimulatedCrash
from casefile.log import AppendConflict, EventLog
from casefile.state import CaseState, FoldError

NOW = datetime(2026, 1, 2, tzinfo=UTC)


@dataclass
class World:
    """A log and an outside world, both on disk, as a crashed process would leave them."""

    log: EventLog
    ledger: Ledger

    def run(self, faults: Faults | None = None, member: str = "M-1") -> CaseState:
        """Run (or resume) the two-step test workflow for case ``c``."""
        runner = Runner(self.log, "c", faults)
        model = FakeModel(self.ledger, lambda request: {"echo": request})
        destination = FakeDestination(self.ledger)
        runner.open({"member": member})

        def ask(ctx: StepContext) -> JsonValue:
            return ctx.call("model", {"q": "facts"}, model)

        def notify(ctx: StepContext) -> JsonValue:
            return ctx.effect("request_info", {"member": member}, destination)

        facts = runner.step("ask/1", ask)
        receipt = runner.step("notify/1", notify)
        return runner.complete({"facts": facts, "receipt": receipt})

    def content(self) -> list[tuple[str, str | None, dict[str, JsonValue]]]:
        """The log without timestamps, hashes or call starts: what a resume must reproduce.

        Call starts are left out because a crash mid-call leaves an extra one on purpose: it is
        the record that gets the lost call charged.
        """
        events = self.log.read("c")
        return [(e.type, e.step_id, dict(e.payload)) for e in events if e.type != "CallStarted"]


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    """Yield a fresh world; closed after the test."""
    log = EventLog(tmp_path / "events.sqlite", clock=lambda: NOW)
    ledger = Ledger(tmp_path / "world.sqlite")
    yield World(log, ledger)
    log.close()
    ledger.close()


@pytest.fixture
def golden(tmp_path: Path) -> Iterator[World]:
    """Yield a second world that has run the workflow once, uninterrupted."""
    log = EventLog(tmp_path / "golden.sqlite", clock=lambda: NOW)
    ledger = Ledger(tmp_path / "golden-world.sqlite")
    reference = World(log, ledger)
    reference.run()
    yield reference
    log.close()
    ledger.close()


def test_a_clean_run_completes_with_one_call_and_one_effect(world: World) -> None:
    state = world.run()
    assert state.completed
    assert world.ledger.model_calls() == 1
    assert list(world.ledger.sends().values()) == [1]
    assert len(world.ledger.applied()) == 1


def test_rerunning_a_completed_case_does_nothing(world: World) -> None:
    first = world.run()
    before = world.content()
    assert world.run() == first
    assert world.content() == before
    assert world.ledger.model_calls() == 1
    assert list(world.ledger.sends().values()) == [1]


def test_a_clean_run_passes_every_crash_point_it_should(world: World) -> None:
    recorder = HitRecorder()
    world.run(recorder)
    assert [point for point, _ in recorder.hits] == [
        CrashPoint.BEFORE_STEP,
        CrashPoint.AFTER_CALL_STARTED,
        CrashPoint.AFTER_CALL_BEFORE_RECORD,
        CrashPoint.AFTER_RECORD,
        CrashPoint.BEFORE_STEP_COMPLETED,
        CrashPoint.BEFORE_STEP,
        CrashPoint.AFTER_INTENT_BEFORE_SEND,
        CrashPoint.AFTER_SEND_BEFORE_CONFIRM,
        CrashPoint.BEFORE_STEP_COMPLETED,
        CrashPoint.BEFORE_CASE_COMPLETED,
    ]


CRASHES = [
    (CrashPoint.BEFORE_STEP, 1),
    (CrashPoint.AFTER_CALL_STARTED, 1),
    (CrashPoint.AFTER_CALL_BEFORE_RECORD, 1),
    (CrashPoint.AFTER_RECORD, 1),
    (CrashPoint.BEFORE_STEP_COMPLETED, 1),
    (CrashPoint.BEFORE_STEP, 2),
    (CrashPoint.AFTER_INTENT_BEFORE_SEND, 1),
    (CrashPoint.AFTER_SEND_BEFORE_CONFIRM, 1),
    (CrashPoint.BEFORE_STEP_COMPLETED, 2),
    (CrashPoint.BEFORE_CASE_COMPLETED, 1),
]


@pytest.mark.parametrize(("point", "occurrence"), CRASHES)
def test_a_crash_anywhere_resumes_to_the_uninterrupted_result(
    world: World, golden: World, point: CrashPoint, occurrence: int
) -> None:
    with pytest.raises(SimulatedCrash):
        world.run(RaiseAt(point, occurrence))
    resumed, expected = world.run(), golden.run()

    assert (resumed.outputs, resumed.outcome, resumed.confirmed) == (
        expected.outputs,
        expected.outcome,
        expected.confirmed,
    )
    assert world.content() == golden.content()
    mid_call = point in (CrashPoint.AFTER_CALL_STARTED, CrashPoint.AFTER_CALL_BEFORE_RECORD)
    assert resumed.lost_calls == (1 if mid_call else 0)
    assert world.ledger.applied() == golden.ledger.applied()
    assert world.log.verify("c") is None
    repeated = world.ledger.model_calls() - golden.ledger.model_calls()
    assert repeated == (1 if point is CrashPoint.AFTER_CALL_BEFORE_RECORD else 0)


def test_a_crash_between_send_and_confirm_resends_under_the_same_key(world: World) -> None:
    with pytest.raises(SimulatedCrash):
        world.run(RaiseAt(CrashPoint.AFTER_SEND_BEFORE_CONFIRM))
    world.run()
    assert list(world.ledger.sends().values()) == [2]
    assert len(world.ledger.applied()) == 1


def test_a_changed_request_after_an_unconfirmed_intent_stops_the_run(world: World) -> None:
    with pytest.raises(SimulatedCrash):
        world.run(RaiseAt(CrashPoint.AFTER_INTENT_BEFORE_SEND))
    with pytest.raises(ChangedInput, match="request changed since intent"):
        world.run(member="M-2")
    assert world.ledger.sends() == {}


def test_a_second_runner_on_the_same_case_cannot_interleave(world: World) -> None:
    first = Runner(world.log, "c")
    second = Runner(world.log, "c")
    first.open({"member": "M-1"})
    with pytest.raises(AppendConflict):
        second.open({"member": "M-1"})


def test_state_before_opening_is_an_error(world: World) -> None:
    with pytest.raises(RuntimeError, match="has not been opened"):
        _ = Runner(world.log, "c").state


def test_an_event_the_fold_would_refuse_is_never_written(world: World) -> None:
    world.run()
    before = world.content()
    runner = Runner(world.log, "c")
    with pytest.raises(FoldError, match="after the case completed"):
        runner.step("late/1", lambda _: "too late")
    assert world.content() == before
    assert Runner(world.log, "c").state.completed


def test_a_lost_call_is_charged_at_its_estimate(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.sqlite", clock=lambda: NOW)
    ledger = Ledger(tmp_path / "world.sqlite")
    model = FakeModel(ledger, lambda request: {"echo": request, "usage": {"cost_microusd": 400}})

    def ask(ctx: StepContext) -> JsonValue:
        return ctx.call("model", {"q": "facts"}, model)

    def run(faults: Faults | None = None) -> CaseState:
        runner = Runner(log, "c", faults, estimate=lambda _: 250)
        runner.open({})
        runner.step("ask/1", ask)
        return runner.complete("done")

    with pytest.raises(SimulatedCrash):
        run(RaiseAt(CrashPoint.AFTER_CALL_BEFORE_RECORD))
    state = run()
    assert (state.model_calls, state.lost_calls) == (2, 1)
    assert state.recorded_cost_microusd == 400
    assert state.spent_microusd == 400 + 250
    assert not state.cost_unknown
    log.close()
    ledger.close()


def test_a_lost_call_with_no_price_makes_the_total_unknown(world: World) -> None:
    with pytest.raises(SimulatedCrash):
        world.run(RaiseAt(CrashPoint.AFTER_CALL_STARTED))
    state = world.run()
    assert state.lost_calls == 1
    assert state.cost_unknown
