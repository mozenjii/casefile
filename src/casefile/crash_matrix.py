"""The crash matrix: kill the toy case at every crash point, resume it, compare with a clean run.

1. A golden run in a child process records every crash point it passes. That list is the matrix,
   so adding a step to the workflow grows the matrix without touching this code.
2. For each ``(point, occurrence)`` a fresh child runs with that crash planned and must die with
   ``CRASH_EXIT_CODE``. A second child, with no crash planned, resumes the case to completion.
3. The resumed case is compared with the golden one: the log's content (everything except
   timestamps and the hashes that cover them), the effects applied at the destination, and the
   hash chain. The number of model calls repeated because of the crash is reported, not judged.

Children are real processes ended with ``os._exit``, so nothing in memory survives the crash and
no ``finally`` block tidies up. Each cell gets its own directory, so cells run in parallel.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from casefile.canonical import JsonValue
from casefile.fakes import Ledger
from casefile.faults import CRASH_EXIT_CODE, CrashPoint
from casefile.log import EventLog
from casefile.toy import CASE_ID

_CHILD = "import sys; from casefile.toy import main; sys.exit(main(sys.argv[1:]))"

type LogContent = list[tuple[str, str | None, dict[str, JsonValue]]]


@dataclass(frozen=True)
class Outcome:
    """What one finished run left behind, in the terms the matrix compares."""

    content: LogContent
    applied: list[str]
    model_calls: int
    chain_intact: bool


@dataclass(frozen=True)
class Cell:
    """One row of the matrix: where the crash was planned and what the resume produced."""

    point: CrashPoint
    occurrence: int
    crashed: bool
    resumed_equal: bool
    effects_once: bool
    chain_intact: bool
    repeated_model_calls: int

    @property
    def passed(self) -> bool:
        """The crash happened and the resumed case is indistinguishable from the golden one."""
        return self.crashed and self.resumed_equal and self.effects_once and self.chain_intact


def run_child(directory: Path, *extra: str) -> int:
    """Run the toy case in a child process against files in ``directory``; return its exit code."""
    directory.mkdir(parents=True, exist_ok=True)
    files = ("--log", str(directory / "events.sqlite"), "--ledger", str(directory / "world.sqlite"))
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, *files, *extra], check=False, capture_output=True
    )
    if completed.returncode not in (0, CRASH_EXIT_CODE):
        raise RuntimeError(f"toy child failed ({completed.returncode}): {completed.stderr!r}")
    return completed.returncode


def read_outcome(directory: Path) -> Outcome:
    """Read the log and ledger a finished run left in ``directory``."""
    with EventLog(directory / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        events = log.read(CASE_ID)
        intact = log.verify(CASE_ID) is None
    ledger = Ledger(directory / "world.sqlite")
    applied, calls = ledger.applied(), ledger.model_calls()
    ledger.close()
    content = [(event.type, event.step_id, dict(event.payload)) for event in events]
    return Outcome(content, applied, calls, intact)


def golden_run(directory: Path) -> tuple[Outcome, list[tuple[CrashPoint, int]]]:
    """Run the case once without crashing; return its outcome and every crash point it passed."""
    hits_file = directory / "hits.json"
    run_child(directory, "--hits", str(hits_file))
    hits = [(CrashPoint(point), int(n)) for point, n in json.loads(hits_file.read_text("utf-8"))]
    return read_outcome(directory), hits


def run_cell(directory: Path, golden: Outcome, point: CrashPoint, occurrence: int) -> Cell:
    """Crash a fresh run at ``(point, occurrence)``, resume it, and compare it with ``golden``."""
    code = run_child(directory, "--crash-at", str(point), "--occurrence", str(occurrence))
    run_child(directory)
    resumed = read_outcome(directory)
    return Cell(
        point=point,
        occurrence=occurrence,
        crashed=code == CRASH_EXIT_CODE,
        resumed_equal=resumed.content == golden.content,
        effects_once=resumed.applied == golden.applied,
        chain_intact=resumed.chain_intact,
        repeated_model_calls=resumed.model_calls - golden.model_calls,
    )


def run_matrix(directory: Path, workers: int = 4) -> list[Cell]:
    """Run the golden case and every cell of the matrix under ``directory``, in matrix order."""
    golden, hits = golden_run(directory / "golden")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_cell, directory / f"{index:02d}-{point}-{n}", golden, point, n)
            for index, (point, n) in enumerate(hits)
        ]
        return [future.result() for future in futures]


def format_table(cells: Sequence[Cell]) -> str:
    """Render the matrix as a Markdown table with a one-line total underneath."""
    rows = [
        "| crash point | # | crashed | resumed = golden | effects once | chain intact "
        "| repeated model calls |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        marks = (cell.crashed, cell.resumed_equal, cell.effects_once, cell.chain_intact)
        flags = " | ".join("yes" if mark else "**NO**" for mark in marks)
        rows.append(f"| {cell.point} | {cell.occurrence} | {flags} | {cell.repeated_model_calls} |")
    passed = sum(cell.passed for cell in cells)
    rows.append("")
    rows.append(f"{passed} of {len(cells)} cells resumed to the uninterrupted result.")
    return "\n".join(rows)
