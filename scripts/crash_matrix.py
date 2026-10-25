"""Print the crash matrix for a workflow (the real review workflow by default) as a table.

Usage, from the repository root::

    uv run python scripts/crash_matrix.py                   # the real review workflow
    uv run python scripts/crash_matrix.py --workflow toy    # the four-step toy case

Exits 1 if any cell fails, so the same command can gate a release. The README's resume figure is
copied from this output.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from casefile.crash_matrix import WORKFLOWS, format_table, run_matrix


def main() -> int:
    """Run the matrix in a scratch directory, print the table, and return the exit code."""
    parser = argparse.ArgumentParser(description="Crash a workflow at every point and resume it.")
    parser.add_argument("--workflow", choices=sorted(WORKFLOWS), default="review")
    workflow = parser.parse_args().workflow
    with tempfile.TemporaryDirectory() as scratch:
        cells = run_matrix(Path(scratch), workflow)
    print(f"workflow: {workflow}")
    print(format_table(cells))
    return 0 if all(cell.passed for cell in cells) else 1


if __name__ == "__main__":
    sys.exit(main())
