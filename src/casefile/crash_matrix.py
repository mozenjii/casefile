"""The crash matrix: kill a workflow at every crash point, resume it, compare with a clean run.

Two workflows can be crashed: ``toy`` (four small steps) and ``review`` (the real one: a generated
case with four documents, a conflict and a request for information, through every specialist).

1. A golden run in a child process records every crash point it passes. That list is the matrix,
   so adding a step to the workflow grows the matrix without touching this code.
2. For each ``(point, occurrence)`` a fresh child runs with that crash planned and must die with
   ``CRASH_EXIT_CODE``. A second child, with no crash planned, resumes the case to completion.
3. The resumed case is compared with the golden one: the log's content (everything except
   timestamps, the hashes that cover them, and ``CallStarted`` records), the effects applied at the
   destination, and the hash chain. ``CallStarted`` records are left out of the comparison on
   purpose: a crash mid-call leaves an extra one, and that extra record is how the lost call gets
   charged. The matrix reports how many lost calls were charged and how many model calls were
   really repeated, rather than judging them.

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
from casefile.state import EventType, fold
from casefile.toy import CASE_ID
from casefile.workflow_child import case_id as review_case_id

type LogContent = list[tuple[str, str | None, dict[str, JsonValue]]]


@dataclass(frozen=True)
class Workflow:
    """A workflow the matrix can crash: the child process to run and the case it works on."""

    name: str
    child: str
    case_id: str


def _child(module: str) -> str:
    """The ``python -c`` program that runs ``module.main`` with the command-line arguments."""
    return f"import sys; from {module} import main; sys.exit(main(sys.argv[1:]))"


WORKFLOWS: dict[str, Workflow] = {
    "toy": Workflow("toy", _child("casefile.toy"), CASE_ID),
    "review": Workflow("review", _child("casefile.workflow_child"), review_case_id()),
}
"""The workflows the matrix knows how to crash."""


@dataclass(frozen=True)
class Outcome:
    """What one finished run left behind, in the terms the matrix compares."""

    content: LogContent
    applied: list[str]
    model_calls: int
    chain_intact: bool
    lost_calls: int


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
    lost_calls_charged: int = 0

    @property
    def passed(self) -> bool:
        """The crash happened and the resumed case is indistinguishable from the golden one."""
        return self.crashed and self.resumed_equal and self.effects_once and self.chain_intact


def run_child(directory: Path, *extra: str, workflow: Workflow = WORKFLOWS["toy"]) -> int:
    """Run ``workflow`` in a child process against files in ``directory``; return its exit code."""
    directory.mkdir(parents=True, exist_ok=True)
    files = ("--log", str(directory / "events.sqlite"), "--ledger", str(directory / "world.sqlite"))
    completed = subprocess.run(
        [sys.executable, "-c", workflow.child, *files, *extra], check=False, capture_output=True
    )
    if completed.returncode not in (0, CRASH_EXIT_CODE):
        message = f"{workflow.name} child failed ({completed.returncode}): {completed.stderr!r}"
        raise RuntimeError(message)
    return completed.returncode


def read_outcome(directory: Path, workflow: Workflow = WORKFLOWS["toy"]) -> Outcome:
    """Read the log and ledger a finished run left in ``directory``."""
    with EventLog(directory / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        events = log.read(workflow.case_id)
        intact = log.verify(workflow.case_id) is None
    ledger = Ledger(directory / "world.sqlite")
    applied, calls = ledger.applied(), ledger.model_calls()
    ledger.close()
    content = [
        (event.type, event.step_id, dict(event.payload))
        for event in events
        if event.type != EventType.CALL_STARTED
    ]
    return Outcome(content, applied, calls, intact, fold(events).lost_calls)


def golden_run(
    directory: Path, workflow: Workflow = WORKFLOWS["toy"]
) -> tuple[Outcome, list[tuple[CrashPoint, int]]]:
    """Run the case once without crashing; return its outcome and every crash point it passed."""
    hits_file = directory / "hits.json"
    run_child(directory, "--hits", str(hits_file), workflow=workflow)
    hits = [(CrashPoint(point), int(n)) for point, n in json.loads(hits_file.read_text("utf-8"))]
    return read_outcome(directory, workflow), hits


def run_cell(
    directory: Path,
    golden: Outcome,
    point: CrashPoint,
    occurrence: int,
    workflow: Workflow = WORKFLOWS["toy"],
) -> Cell:
    """Crash a fresh run at ``(point, occurrence)``, resume it, and compare it with ``golden``."""
    crash = ("--crash-at", str(point), "--occurrence", str(occurrence))
    code = run_child(directory, *crash, workflow=workflow)
    run_child(directory, workflow=workflow)
    resumed = read_outcome(directory, workflow)
    return Cell(
        point=point,
        occurrence=occurrence,
        crashed=code == CRASH_EXIT_CODE,
        resumed_equal=resumed.content == golden.content,
        effects_once=resumed.applied == golden.applied,
        chain_intact=resumed.chain_intact,
        repeated_model_calls=resumed.model_calls - golden.model_calls,
        lost_calls_charged=resumed.lost_calls - golden.lost_calls,
    )


def run_matrix(directory: Path, workflow: str = "toy", workers: int = 4) -> list[Cell]:
    """Run the golden case and every cell of the matrix under ``directory``, in matrix order."""
    chosen = WORKFLOWS[workflow]
    golden, hits = golden_run(directory / "golden", chosen)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_cell, directory / f"{index:02d}-{point}-{n}", golden, point, n, chosen)
            for index, (point, n) in enumerate(hits)
        ]
        return [future.result() for future in futures]


def format_table(cells: Sequence[Cell]) -> str:
    """Render the matrix as a Markdown table with a one-line total underneath."""
    rows = [
        "| crash point | # | crashed | resumed = golden | effects once | chain intact "
        "| model calls repeated | lost calls charged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        marks = (cell.crashed, cell.resumed_equal, cell.effects_once, cell.chain_intact)
        flags = " | ".join("yes" if mark else "**NO**" for mark in marks)
        counts = f"{cell.repeated_model_calls} | {cell.lost_calls_charged}"
        rows.append(f"| {cell.point} | {cell.occurrence} | {flags} | {counts} |")
    passed = sum(cell.passed for cell in cells)
    rows.append("")
    rows.append(f"{passed} of {len(cells)} cells resumed to the uninterrupted result.")
    return "\n".join(rows)
