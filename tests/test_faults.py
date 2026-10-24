"""Tests for the fault plans: each fires on exactly the chosen hit and on no other."""

from __future__ import annotations

import pytest

from casefile.faults import (
    CRASH_EXIT_CODE,
    CrashAt,
    CrashPoint,
    HitRecorder,
    NoFaults,
    RaiseAt,
    SimulatedCrash,
)

STEP = CrashPoint.BEFORE_STEP
RECORD = CrashPoint.AFTER_RECORD


def test_no_faults_ignores_every_point() -> None:
    faults = NoFaults()
    for point in CrashPoint:
        faults.hit(point)


def test_the_recorder_numbers_occurrences_per_point() -> None:
    recorder = HitRecorder()
    for point in (STEP, RECORD, STEP, STEP, RECORD):
        recorder.hit(point)
    assert recorder.hits == [(STEP, 1), (RECORD, 1), (STEP, 2), (STEP, 3), (RECORD, 2)]


def test_raise_at_fires_on_the_chosen_occurrence_only() -> None:
    faults = RaiseAt(STEP, occurrence=2)
    faults.hit(RECORD)
    faults.hit(STEP)
    with pytest.raises(SimulatedCrash, match="before_step #2"):
        faults.hit(STEP)
    faults.hit(STEP)


def test_simulated_crash_is_not_swallowed_by_except_exception() -> None:
    assert not issubclass(SimulatedCrash, Exception)


def test_crash_at_exits_with_the_crash_code_on_the_chosen_hit() -> None:
    exits: list[int] = []
    faults = CrashAt(RECORD, occurrence=2, exit=exits.append)
    for point in (STEP, RECORD, STEP):
        faults.hit(point)
    assert exits == []
    faults.hit(RECORD)
    assert exits == [CRASH_EXIT_CODE]
