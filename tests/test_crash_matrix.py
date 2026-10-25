"""The crash matrix itself: real child processes, killed at every crash point and resumed.

This is the test behind the claim that a case resumes correctly after a kill at any step, on the
toy workflow and on the real review workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from casefile.crash_matrix import Cell, format_table, run_child, run_matrix
from casefile.faults import CrashPoint

MID_CALL = {CrashPoint.AFTER_CALL_STARTED, CrashPoint.AFTER_CALL_BEFORE_RECORD}


def check(cells: list[Cell]) -> None:
    """All cells pass; only a crash after the answer repeats a call; mid-call crashes charge one."""
    assert [cell for cell in cells if not cell.passed] == [], format_table(cells)
    for cell in cells:
        repeated = 1 if cell.point is CrashPoint.AFTER_CALL_BEFORE_RECORD else 0
        assert cell.repeated_model_calls == repeated, cell
        assert cell.lost_calls_charged == (1 if cell.point in MID_CALL else 0), cell


def test_every_toy_cell_resumes_to_the_uninterrupted_result(tmp_path: Path) -> None:
    cells = run_matrix(tmp_path, "toy")
    assert len(cells) == 17
    assert {cell.point for cell in cells} == set(CrashPoint)
    check(cells)


def test_every_cell_of_the_real_review_workflow_resumes(tmp_path: Path) -> None:
    cells = run_matrix(tmp_path, "review")
    assert len(cells) == 32
    assert {cell.point for cell in cells} == set(CrashPoint)
    check(cells)


def test_a_failed_cell_is_impossible_to_miss_in_the_table() -> None:
    good = Cell(CrashPoint.BEFORE_STEP, 1, True, True, True, True, 0)
    bad = Cell(CrashPoint.AFTER_RECORD, 2, True, False, True, True, 0, 1)
    table = format_table([good, bad])
    assert "| after_record | 2 | yes | **NO** | yes | yes | 0 | 1 |" in table
    assert table.endswith("1 of 2 cells resumed to the uninterrupted result.")
    assert not bad.passed


def test_a_child_that_fails_for_another_reason_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="toy child failed"):
        run_child(tmp_path, "--crash-at", "nowhere")
