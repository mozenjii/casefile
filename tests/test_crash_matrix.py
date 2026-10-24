"""The crash matrix itself: real child processes, killed at every crash point and resumed.

This is the test behind the claim that a case resumes correctly after a kill at any step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from casefile.crash_matrix import Cell, format_table, run_child, run_matrix
from casefile.faults import CrashPoint


def test_every_cell_resumes_to_the_uninterrupted_result(tmp_path: Path) -> None:
    cells = run_matrix(tmp_path)
    assert len(cells) == 15
    assert {cell.point for cell in cells} == set(CrashPoint)
    failed = [cell for cell in cells if not cell.passed]
    assert failed == [], format_table(cells)
    repeats = {(cell.point, cell.repeated_model_calls) for cell in cells}
    assert repeats == {(point, 0) for point in CrashPoint} - {
        (CrashPoint.AFTER_CALL_BEFORE_RECORD, 0)
    } | {(CrashPoint.AFTER_CALL_BEFORE_RECORD, 1)}


def test_a_failed_cell_is_impossible_to_miss_in_the_table() -> None:
    good = Cell(CrashPoint.BEFORE_STEP, 1, True, True, True, True, 0)
    bad = Cell(CrashPoint.AFTER_RECORD, 2, True, False, True, True, 0)
    table = format_table([good, bad])
    assert "| after_record | 2 | yes | **NO** | yes | yes | 0 |" in table
    assert table.endswith("1 of 2 cells resumed to the uninterrupted result.")
    assert not bad.passed


def test_a_child_that_fails_for_another_reason_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="toy child failed"):
        run_child(tmp_path, "--crash-at", "nowhere")
