"""Print the crash matrix for the toy case as a Markdown table.

Usage, from the repository root::

    uv run python scripts/crash_matrix.py

Exits 1 if any cell fails, so the same command can gate a release. The README's resume figure is
copied from this output.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from casefile.crash_matrix import format_table, run_matrix


def main() -> int:
    """Run the matrix in a scratch directory, print the table, and return the exit code."""
    with tempfile.TemporaryDirectory() as scratch:
        cells = run_matrix(Path(scratch))
    print(format_table(cells))
    return 0 if all(cell.passed for cell in cells) else 1


if __name__ == "__main__":
    sys.exit(main())
