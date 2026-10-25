"""Generate a synthetic case per defect, run each through the workflow, serve the review screen.

Usage, from the repository root::

    uv run python scripts/demo.py            # then open http://127.0.0.1:8000
    uv run python scripts/demo.py --fresh    # start over with new case files

Cases are written under ``runs/demo/``. Running again reuses them, so decisions made in the browser
survive a restart, the same way a crashed run resumes.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from casefile.agents import SourceDocument, run_review
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.generator import DEFECTS, generate_set
from casefile.log import EventLog
from casefile.reference_model import reference_model
from casefile.web import create_app

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "runs" / "demo"


def seed(directory: Path) -> None:
    """Run (or resume) the automated review of one case per defect."""
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(directory / "world.sqlite")
    with EventLog(directory / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        for case in generate_set(len(DEFECTS), seed=1):
            documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
            model, destination = FakeModel(ledger, reference_model), FakeDestination(ledger)
            run_review(Runner(log, case.case_id), documents, model, destination)
    ledger.close()


def main() -> None:
    """Seed the demo cases and serve the review screen until interrupted."""
    parser = argparse.ArgumentParser(description="Serve the casefile review screen on demo cases.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fresh", action="store_true", help="delete earlier demo cases first")
    arguments = parser.parse_args()
    if arguments.fresh and DEMO.exists():
        shutil.rmtree(DEMO)
    seed(DEMO)
    uvicorn.run(create_app(DEMO / "events.sqlite"), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
