"""Smoke test that the package installs and imports from the locked environment."""

import casefile


def test_package_exposes_a_version() -> None:
    assert casefile.__version__ == "0.0.0"
