"""Compare the single-agent, fixed-workflow and specialist designs on generated cases.

Usage, from the repository root::

    uv run python scripts/compare.py                 # offline stand-in: proves the harness
    CASEFILE_PROVIDER=openrouter OPENROUTER_API_KEY=... uv run python scripts/compare.py --yes

With a real provider this spends money, so it prints the provider and an estimate and stops unless
``--yes`` is given. Answers are cached under ``runs/``, so running it again costs nothing.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from casefile.cache import ResponseCache, cached
from casefile.compare import DESIGNS, Result, run_design, summarise
from casefile.estimate import dollars, estimate_cases, estimator
from casefile.fakes import Ledger
from casefile.generator import DEFECTS, generate_set
from casefile.llm import ConfigError, build_model, describe, settings_for_repository
from casefile.log import EventLog

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "model-cache.sqlite"


def main() -> int:
    """Run the comparison and print the table; return the exit code."""
    parser = argparse.ArgumentParser(description="Compare three workflow designs.")
    parser.add_argument("--cases", type=int, default=3 * len(DEFECTS), help="default 27")
    parser.add_argument("--seed", type=int, default=500, help="first case seed")
    parser.add_argument("--yes", action="store_true", help="allow spending on a paid provider")
    arguments = parser.parse_args()
    try:
        settings = settings_for_repository(ROOT)
    except ConfigError as error:
        print(f"provider settings: {error}", file=sys.stderr)
        return 2

    cases = generate_set(arguments.cases, seed=arguments.seed)
    print(f"provider: {describe(settings)}")
    if settings.backend == "reference":
        print("The offline stand-in already knows the document layout, so every design scores")
        print("perfectly here. This run proves the harness; real numbers need a real model.")
    else:
        sample = estimate_cases(cases[: min(len(cases), 9)])
        per_case = settings.cost(
            sample.input_tokens // sample.cases, sample.output_tokens // sample.cases
        )
        rough = None if per_case is None else per_case * len(cases) * len(DESIGNS)
        runs = f"{len(cases)} cases x {len(DESIGNS)} designs"
        print(f"estimated cost: about {dollars(rough)} for {runs}")
        if not arguments.yes:
            print("Not run: this provider costs money. Re-run with --yes to go ahead.")
            return 1

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(CACHE)
    model = cached(build_model(settings), cache, f"{settings.provider}/{settings.model}")
    price = estimator(settings)
    results: list[Result] = []
    with tempfile.TemporaryDirectory() as scratch:
        ledger = Ledger(Path(scratch) / "world.sqlite")
        with EventLog(Path(scratch) / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
            for case in cases:
                for design in DESIGNS:
                    results.append(run_design(design, case, model, log, ledger, price))
        ledger.close()
    cache.close()
    print()
    print(summarise(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
