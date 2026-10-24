"""Tests for the suppression and doc-link gates.

Each rule gets a positive case (the gate fires) and a look-alike that must not fire, because a gate
that flags innocent text is the kind that gets disabled.
"""

from __future__ import annotations

from pathlib import Path

from repo_checks import Finding, find_broken_links, find_real_identifiers, find_suppressions


def write(directory: Path, name: str, text: str) -> Path:
    """Write ``text`` to ``directory/name`` and return the path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def messages(findings: list[Finding]) -> list[str]:
    """Reduce findings to ``line: message`` strings so expectations read as plain text."""
    return [f"{finding.line}: {finding.message}" for finding in findings]


def test_suppression_comments_are_found(tmp_path: Path) -> None:
    source = write(
        tmp_path,
        "bad.py",
        "import os  # noqa: F401\n"
        "x: int = 'a'  # type: ignore[assignment]\n"
        "y = 1  # pyright: ignore\n"
        "def f() -> None:  # pragma: no cover\n"
        "    pass\n",
    )
    assert messages(find_suppressions(source)) == [
        "1: suppression comment 'noqa'",
        "2: suppression comment 'type: ignore'",
        "3: suppression comment 'pyright: ignore'",
        "4: suppression comment 'pragma: no cover'",
    ]


def test_skipped_and_xfailed_tests_are_found(tmp_path: Path) -> None:
    source = write(
        tmp_path,
        "test_bad.py",
        "import pytest\n"
        "@pytest.mark.skip(reason='later')\n"
        "def test_a() -> None: ...\n"
        "@pytest.mark.skipif(True, reason='never')\n"
        "def test_b() -> None: ...\n"
        "@pytest.mark.xfail\n"
        "def test_c() -> None: ...\n"
        "def test_d() -> None:\n"
        "    pytest.skip('not today')\n"
        "def test_e() -> None:\n"
        "    pytest.importorskip('numpy')\n",
    )
    assert messages(find_suppressions(source)) == [
        "2: skipped or xfailed test marker",
        "4: skipped or xfailed test marker",
        "6: skipped or xfailed test marker",
        "9: test skipped at run time",
        "11: test skipped at run time",
    ]


def test_suppression_text_inside_strings_is_not_flagged(tmp_path: Path) -> None:
    source = write(
        tmp_path,
        "fixture.py",
        'NOTE = "# noqa and # type: ignore are banned here"\n'
        "MARK = 'pytest.mark.skip'\n"
        "# an ordinary comment mentioning nothing in particular\n",
    )
    assert find_suppressions(source) == []


def test_unparseable_python_is_reported_not_raised(tmp_path: Path) -> None:
    source = write(tmp_path, "broken.py", "def f(:\n")
    findings = find_suppressions(source)
    assert len(findings) == 1
    assert findings[0].message.startswith("could not parse")


def test_broken_relative_links_are_found(tmp_path: Path) -> None:
    write(tmp_path, "docs/exists.md", "# here\n")
    readme = write(
        tmp_path,
        "README.md",
        "See [the design](docs/exists.md) and [the plan](docs/missing.md).\n"
        "![diagram](img/flow.png)\n"
        "[ref]: docs/gone.md\n",
    )
    assert messages(find_broken_links(readme)) == [
        "1: broken link 'docs/missing.md'",
        "2: broken link 'img/flow.png'",
        "3: broken link 'docs/gone.md'",
    ]


def test_external_anchor_encoded_and_fenced_links_are_not_flagged(tmp_path: Path) -> None:
    write(tmp_path, "docs/event log.md", "# Events\n")
    readme = write(
        tmp_path,
        "README.md",
        "[site](https://example.com/x) [mail](mailto:a@example.com) [top](#status)\n"
        "[events](docs/event%20log.md#schema)\n"
        "```markdown\n"
        "[example](not/a/real/file.md)\n"
        "```\n",
    )
    assert find_broken_links(readme) == []


def test_finding_prints_as_a_clickable_location(tmp_path: Path) -> None:
    source = write(tmp_path, "bad.py", "import os  # noqa\n")
    (finding,) = find_suppressions(source)
    assert str(finding).endswith("bad.py:1: suppression comment 'noqa'")


def test_possibly_real_identifiers_are_found_without_echoing_them(tmp_path: Path) -> None:
    real = "-".join(("123", "45", "6789"))
    fictional = "-".join(("212", "555", "0142"))
    case = write(tmp_path, "case.json", f'{{"phone": "{fictional}",\n "ssn": "{real}"}}\n')
    (finding,) = find_real_identifiers(case)
    assert (finding.line, finding.message) == (2, "possibly real ssn")
    assert real not in str(finding)


def test_binary_files_are_skipped_by_the_identifier_scan(tmp_path: Path) -> None:
    image = tmp_path / "logo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    assert find_real_identifiers(image) == []
