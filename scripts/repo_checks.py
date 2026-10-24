"""Repository checks that no off-the-shelf tool runs the way this project needs.

Two rules from the quality bar are enforced here:

* No check is silenced. A suppression comment (``noqa``, a type-checker ignore, a coverage pragma)
  or a skipped or expected-to-fail test turns a red gate green without fixing anything, so the
  presence of one fails the build. Comments are found with ``tokenize`` and skips with ``ast``, so
  the same text inside a string literal (a test fixture, say) is not flagged.
* Relative links in the public Markdown resolve. A README that points at a moved file is the first
  thing a reader notices.

Both functions are pure over the files they are given and return findings instead of printing, so
the runner decides how to report and the tests can assert on exact results.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard
from urllib.parse import unquote

_SUPPRESSION_COMMENTS = re.compile(
    r"#\s*(noqa\b|type:\s*ignore\b|pyright:\s*ignore\b|pragma:\s*no\s*cover\b)", re.IGNORECASE
)
_SKIP_MARKERS = frozenset({"skip", "skipif", "xfail"})
_SKIP_CALLS = frozenset({"skip", "xfail", "importorskip"})

_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


@dataclass(frozen=True)
class Finding:
    """One problem at one place in one file, ready to print as ``path:line: message``."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        """Format the finding the way editors and CI logs turn into a clickable location."""
        return f"{self.path.as_posix()}:{self.line}: {self.message}"


def find_suppressions(path: Path) -> list[Finding]:
    """Return every suppression comment and every skipped or xfailed test in one Python file.

    A file that does not parse is reported as a single finding rather than raised, because the lint
    and type gates report syntax errors properly and this gate should not hide behind them.
    """
    source = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, tokenize.TokenError) as error:
        line = getattr(error, "lineno", None) or 1
        return [Finding(path, line, f"could not parse: {error}")]

    for token in tokens:
        if token.type == tokenize.COMMENT and (match := _SUPPRESSION_COMMENTS.search(token.string)):
            findings.append(
                Finding(path, token.start[0], f"suppression comment '{match.group(1)}'")
            )

    for node in ast.walk(tree):
        if _is_pytest_attribute(node, "mark", _SKIP_MARKERS):
            findings.append(Finding(path, node.lineno, "skipped or xfailed test marker"))
        elif isinstance(node, ast.Call) and _is_pytest_attribute(node.func, None, _SKIP_CALLS):
            findings.append(Finding(path, node.lineno, "test skipped at run time"))

    return sorted(findings, key=lambda finding: finding.line)


def _is_pytest_attribute(
    node: ast.AST, middle: str | None, names: frozenset[str]
) -> TypeGuard[ast.Attribute]:
    """Match ``pytest.<name>`` when ``middle`` is None, else ``pytest.<middle>.<name>``."""
    if not isinstance(node, ast.Attribute) or node.attr not in names:
        return False
    owner = node.value
    if middle is not None:
        if not isinstance(owner, ast.Attribute) or owner.attr != middle:
            return False
        owner = owner.value
    return isinstance(owner, ast.Name) and owner.id == "pytest"


def find_broken_links(path: Path) -> list[Finding]:
    """Return every relative link in one Markdown file whose target does not exist on disk.

    External links (anything with a scheme) and same-page anchors are skipped: checking them needs
    the network or a Markdown renderer's heading slugs, and a flaky gate gets switched off. Links
    inside fenced code blocks are examples, not links, and are skipped too.
    """
    findings: list[Finding] = []
    in_fence = False

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        targets = [match.group(1) for match in _INLINE_LINK.finditer(line)]
        if reference := _REFERENCE_LINK.match(line):
            targets.append(reference.group(1))

        for target in targets:
            if _SCHEME.match(target) or target.startswith("#"):
                continue
            relative = unquote(target.split("#", 1)[0])
            if not (path.parent / relative).exists():
                findings.append(Finding(path, number, f"broken link '{target}'"))

    return findings
