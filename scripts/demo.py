"""Generate a synthetic case per defect, run each through the workflow, serve the review screen.

Usage, from the repository root::

    uv run python scripts/demo.py            # then open http://127.0.0.1:8000
    uv run python scripts/demo.py --fresh    # start over with new case files

Cases are written under ``runs/demo/``. Running again reuses them, so decisions made in the browser
survive a restart, the same way a crashed run resumes.

The specialists use whichever model the environment configures (see ``.env.example``); with nothing
set, that is the offline stand-in. Model answers are cached per provider and model under
``runs/demo/``, so ``--fresh`` regenerates cases without paying for the same answers twice.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from casefile.agents import SourceDocument, run_review
from casefile.cache import ResponseCache, cached
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.generator import DEFECTS, generate_set
from casefile.llm import ConfigError, Settings, build_model, describe, settings_for_repository
from casefile.log import EventLog
from casefile.web import create_app

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "runs" / "demo"
CACHE = ROOT / "runs" / "model-cache.sqlite"


def seed(directory: Path, settings: Settings) -> None:
    """Run (or resume) the automated review of one case per defect with the configured model."""
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(directory / "world.sqlite")
    cache = ResponseCache(CACHE)
    model = cached(build_model(settings), cache, f"{settings.provider}/{settings.model}")
    with EventLog(directory / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        for case in generate_set(len(DEFECTS), seed=1):
            documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
            counted, destination = FakeModel(ledger, model), FakeDestination(ledger)
            run_review(Runner(log, case.case_id), documents, counted, destination)
    cache.close()
    ledger.close()


def main() -> None:
    """Seed the demo cases and serve the review screen until interrupted."""
    parser = argparse.ArgumentParser(description="Serve the casefile review screen on demo cases.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fresh", action="store_true", help="delete earlier demo cases first")
    arguments = parser.parse_args()
    try:
        settings = settings_for_repository(ROOT)
    except ConfigError as error:
        raise SystemExit(f"provider settings: {error}") from None
    print(f"specialists use: {describe(settings)}")
    if arguments.fresh:
        shutil.rmtree(DEMO / "cases", ignore_errors=True)
    seed(DEMO / "cases", settings)
    uvicorn.run(create_app(DEMO / "cases" / "events.sqlite"), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
