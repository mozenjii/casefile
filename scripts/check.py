"""Run every gate this repository enforces, cheapest first, and stop at the first failure.

Usage, from the repository root::

    uv run python scripts/check.py           # the local gates
    uv run python scripts/check.py --audit   # plus the dependency audit, which needs the network
    uv run python scripts/check.py --browser # plus the browser tests (needs: playwright install)

CI runs the second form, so a change that passes locally fails in CI only for a dependency
advisory or a platform difference, never because the two run different checks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from repo_checks import Finding, find_broken_links, find_real_identifiers, find_suppressions

ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIRS = ("src", "tests", "scripts")
PIP_AUDIT = "pip-audit==2.10.1"


def public_markdown() -> list[Path]:
    """Return the Markdown files that are published: the README and anything under ``docs/``."""
    files = [ROOT / "README.md"]
    docs = ROOT / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
    return [path for path in files if path.is_file()]


def python_sources() -> list[Path]:
    """Return every Python file under the directories the gates cover."""
    return sorted(path for name in PYTHON_DIRS for path in (ROOT / name).rglob("*.py"))


def publishable_files() -> list[Path]:
    """Return every file git would publish: tracked ones and new ones that are not ignored."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ROOT / name for name in listed.split("\0") if name and (ROOT / name).is_file()]


def report(findings: Iterable[Finding]) -> bool:
    """Print findings relative to the repository root and return True when there were none."""
    found = list(findings)
    for finding in found:
        relative = Finding(finding.path.relative_to(ROOT), finding.line, finding.message)
        print(f"  {relative}")
    return not found


def run(*argv: str) -> bool:
    """Run one tool from the repository root, streaming its output, and return True on exit 0."""
    return subprocess.run(argv, cwd=ROOT, check=False).returncode == 0


def audit_dependencies() -> bool:
    """Check every locked dependency, with its hash, against the public advisory databases.

    The auditor runs in its own throwaway environment so it never lands in the project's lockfile,
    and its version is pinned so the gate does not change underneath us between two runs.
    """
    uv = os.environ.get("UV", "uv")
    with tempfile.TemporaryDirectory() as scratch:
        requirements = str(Path(scratch) / "requirements.txt")
        export = ("export", "--frozen", "--no-emit-project", "--quiet", "--format")
        audit = ("--strict", "--require-hashes", "--disable-pip", "--requirement")
        return run(uv, *export, "requirements-txt", "--output-file", requirements) and run(
            uv, "tool", "run", PIP_AUDIT, *audit, requirements
        )


def gates(include_audit: bool, include_browser: bool) -> list[tuple[str, Callable[[], bool]]]:
    """Return the gates in the order they run: fastest and most common failures first."""
    python = sys.executable
    selected: list[tuple[str, Callable[[], bool]]] = [
        ("format", lambda: run(python, "-m", "ruff", "format", "--check", *PYTHON_DIRS)),
        ("lint", lambda: run(python, "-m", "ruff", "check", *PYTHON_DIRS)),
        (
            "suppressions",
            lambda: report(f for path in python_sources() for f in find_suppressions(path)),
        ),
        (
            "doc links",
            lambda: report(f for path in public_markdown() for f in find_broken_links(path)),
        ),
        (
            "real-data scan",
            lambda: report(f for path in publishable_files() for f in find_real_identifiers(path)),
        ),
        ("types", lambda: run(python, "-m", "pyright")),
        ("tests", lambda: run(python, "-m", "pytest", "--cov", "--cov-report=term-missing")),
    ]
    if include_browser:
        browser = ("-m", "browser", "--no-cov", "-p", "no:cacheprovider")
        selected.append(("browser", lambda: run(python, "-m", "pytest", *browser)))
    if include_audit:
        selected.append(("dependency audit", audit_dependencies))
    return selected


def main() -> int:
    """Run the gates and return the process exit code: 0 when every gate passed, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Run every gate this repository enforces.")
    parser.add_argument("--audit", action="store_true", help="also audit dependencies (network)")
    parser.add_argument("--browser", action="store_true", help="also drive the UI in Chromium")
    arguments = parser.parse_args()

    for name, gate in gates(arguments.audit, arguments.browser):
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        passed = gate()
        elapsed = time.perf_counter() - started
        if not passed:
            print(f"FAILED: {name} ({elapsed:.1f}s)")
            return 1
        print(f"ok: {name} ({elapsed:.1f}s)", flush=True)

    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
