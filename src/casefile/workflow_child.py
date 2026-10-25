"""Run the real review workflow on one generated case, as a child process the crash matrix can kill.

The same entry point serves the golden run (``--hits`` records every crash point passed), the crash
run (``--crash-at`` ends the process on one hit) and the resume (neither). The model is the
deterministic stand-in counted through the ledger, so a repeated call is visible afterwards.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from casefile.agents import SourceDocument, run_review
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.faults import CrashAt, CrashPoint, Faults, HitRecorder
from casefile.generator import generate
from casefile.log import EventLog
from casefile.reference_model import reference_model

CRASH_CASE = (5, "therapy_conflict")
"""The case the matrix crashes: four documents, a conflict, and a request for information."""


def case_id() -> str:
    """The id of the case the matrix crashes."""
    return generate(*CRASH_CASE).case_id


def main(argv: Sequence[str]) -> int:
    """Run (or resume) the review of the crash case against the files named in ``argv``."""
    parser = argparse.ArgumentParser(prog="casefile.workflow_child")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--crash-at", type=CrashPoint, choices=list(CrashPoint))
    parser.add_argument("--occurrence", type=int, default=1)
    parser.add_argument("--hits", type=Path)
    arguments = parser.parse_args(argv)

    recorder = HitRecorder()
    faults: Faults = recorder
    if arguments.crash_at is not None:
        faults = CrashAt(arguments.crash_at, arguments.occurrence)

    case = generate(*CRASH_CASE)
    documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
    ledger = Ledger(arguments.ledger)
    with EventLog(arguments.log, clock=lambda: datetime.now(UTC)) as log:
        runner = Runner(log, case.case_id, faults, estimate=lambda _: 250)
        model = FakeModel(ledger, reference_model)
        run_review(runner, documents, model, FakeDestination(ledger))
    ledger.close()

    if arguments.hits is not None:
        hits = [[str(point), occurrence] for point, occurrence in recorder.hits]
        arguments.hits.write_text(json.dumps(hits), encoding="utf-8")
    return 0
