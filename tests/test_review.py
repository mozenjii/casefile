"""Tests for the review service, on the synthetic case run to completion by the toy workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.canonical import JsonValue
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.log import EventLog
from casefile.review import NotReviewable, Review, StaleView
from casefile.state import FoldError
from casefile.toy import CASE_ID, CASE_INPUTS, run_case, toy_model

ANA, BEN = "human:ana", "human:ben"


@pytest.fixture
def log(tmp_path: Path) -> Iterator[EventLog]:
    """Yield a log holding the synthetic case after its automated run."""
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(tmp_path / "events.sqlite", clock=lambda: datetime.now(UTC)) as opened:
        model, destination = FakeModel(ledger, toy_model), FakeDestination(ledger)
        run_case(Runner(opened, CASE_ID), model, destination, CASE_INPUTS)
        yield opened
    ledger.close()


def test_the_reviewer_sees_the_dossier_and_its_hash(log: EventLog) -> None:
    view = Review(log).show(CASE_ID)
    canonical = json.dumps(view.dossier, sort_keys=True, separators=(",", ":")).encode()
    assert view.dossier_hash == hashlib.sha256(canonical).hexdigest()
    assert (view.approved, view.released, view.decisions) == (False, False, {})


def test_an_approved_dossier_is_released(log: EventLog) -> None:
    review = Review(log)
    seen = review.show(CASE_ID).dossier_hash
    assert review.decide(CASE_ID, "approve", reviewer=ANA, seen_hash=seen).approved
    assert review.release(CASE_ID, releaser=BEN).released


def test_release_without_approval_is_refused_and_nothing_written(log: EventLog) -> None:
    before = log.read(CASE_ID)
    with pytest.raises(FoldError, match="without an approval"):
        Review(log).release(CASE_ID, releaser=ANA)
    assert log.read(CASE_ID) == before


def test_an_edit_after_approval_needs_a_fresh_approval(log: EventLog) -> None:
    review = Review(log)
    seen = review.show(CASE_ID).dossier_hash
    review.decide(CASE_ID, "approve", reviewer=ANA, seen_hash=seen)
    summary: JsonValue = {"summary": "Pend: imaging report missing."}
    edited = review.revise(CASE_ID, summary, editor=BEN, seen_hash=seen)
    assert not edited.approved
    with pytest.raises(FoldError, match="without an approval"):
        review.release(CASE_ID, releaser=ANA)
    review.decide(CASE_ID, "approve", reviewer=ANA, seen_hash=edited.dossier_hash)
    assert review.release(CASE_ID, releaser=ANA).released


def test_an_approval_of_an_outdated_view_is_recorded_but_does_not_count(log: EventLog) -> None:
    review = Review(log)
    anas_view = review.show(CASE_ID)
    review.revise(CASE_ID, {"summary": "changed"}, editor=BEN, seen_hash=anas_view.dossier_hash)
    after = review.decide(CASE_ID, "approve", reviewer=ANA, seen_hash=anas_view.dossier_hash)
    assert after.decisions == {anas_view.dossier_hash: "approve"}
    assert not after.approved


def test_two_editors_cannot_overwrite_each_other(log: EventLog) -> None:
    review = Review(log)
    seen = review.show(CASE_ID).dossier_hash
    review.revise(CASE_ID, {"summary": "first"}, editor=ANA, seen_hash=seen)
    with pytest.raises(StaleView):
        review.revise(CASE_ID, {"summary": "second"}, editor=BEN, seen_hash=seen)


def test_an_agent_cannot_approve(log: EventLog) -> None:
    review = Review(log)
    seen = review.show(CASE_ID).dossier_hash
    with pytest.raises(FoldError, match="must be a human"):
        review.decide(CASE_ID, "approve", reviewer="agent:drafter", seen_hash=seen)


def test_a_case_without_a_dossier_cannot_be_reviewed(tmp_path: Path) -> None:
    with EventLog(tmp_path / "empty.sqlite", clock=lambda: datetime.now(UTC)) as empty:
        with pytest.raises(NotReviewable):
            Review(empty).show("nobody")
        Runner(empty, "half").open({"member": "SYN"})
        with pytest.raises(NotReviewable):
            Review(empty).release("half", releaser=ANA)
