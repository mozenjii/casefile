"""Tests for the outside-world ledger's refusal path."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from casefile.canonical import CanonicalJsonError, JsonValue
from casefile.fakes import FakeDestination, Ledger


def test_a_send_that_cannot_be_recorded_leaves_no_trace(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "world.sqlite")
    destination = FakeDestination(ledger)
    with pytest.raises(CanonicalJsonError):
        destination.send("k", cast(JsonValue, {"amount": 1.5}))
    assert ledger.sends() == {}
    assert ledger.applied() == []
    assert destination.send("k", {"amount": 2}) == destination.send("k", {"amount": 2})
    assert ledger.sends() == {"k": 2}
    ledger.close()
