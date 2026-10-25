"""Tests for the review screen over HTTP: rendering, the decision flow, and refusals."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from casefile.agents import Budget, SourceDocument, run_review
from casefile.canonical import JsonValue
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.generator import generate
from casefile.log import EventLog
from casefile.reference_model import reference_model
from casefile.review import Review
from casefile.web import anchor, create_app

APPROVABLE = generate(21, "none")
PENDING = generate(22, "missing_xray")


def seed(path: Path, case_id: str, documents: list[SourceDocument]) -> None:
    """Run the automated review of one case into the log at ``path``."""
    ledger = Ledger(path.with_name("world.sqlite"))
    with EventLog(path, clock=lambda: datetime.now(UTC)) as log:
        model, destination = FakeModel(ledger, reference_model), FakeDestination(ledger)
        run_review(Runner(log, case_id), documents, model, destination)
    ledger.close()


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    """A log holding two generated cases after their automated runs."""
    path = tmp_path / "events.sqlite"
    for case in (APPROVABLE, PENDING):
        seed(path, case.case_id, [SourceDocument(d.id, d.title, d.text) for d in case.documents])
    return path


@pytest.fixture
def client(log_path: Path) -> Iterator[TestClient]:
    """A test client for the app over that log."""
    with TestClient(create_app(log_path)) as test_client:
        yield test_client


def current_hash(client: TestClient, case_id: str) -> str:
    """Read the dossier hash the page asks the reviewer to decide on."""
    page = client.get(f"/cases/{case_id}").text
    match = re.search(r'name="seen_hash" value="([0-9a-f]{64})"', page)
    assert match is not None
    return match.group(1)


def decide(client: TestClient, case_id: str, decision: str, seen: str, reviewer: str = "ana"):
    """Post a decision without following the redirect."""
    form = {"decision": decision, "seen_hash": seen, "reviewer": reviewer}
    return client.post(f"/cases/{case_id}/decision", data=form, follow_redirects=False)


def test_the_case_list_shows_each_case_and_where_it_stands(client: TestClient) -> None:
    page = client.get("/").text
    assert APPROVABLE.case_id in page
    assert PENDING.case_id in page
    assert page.count("awaiting review") == 2
    assert "rec-request_info" in page
    assert "rec-approve" in page


def test_the_review_page_links_every_criterion_to_marked_evidence(client: TestClient) -> None:
    page = client.get(f"/cases/{APPROVABLE.case_id}").text
    assert "Recommended: approve" in page
    assert page.count('<td class="id">') == 5
    for observation in APPROVABLE.observations:
        target = anchor(observation.evidence)
        assert f'<mark id="{target}"' in page
        assert f'href="#{target}"' in page
    assert "Release unlocks once this exact version is approved." in page


def test_a_pending_case_shows_what_was_requested(client: TestClient) -> None:
    page = client.get(f"/cases/{PENDING.case_id}").text
    assert "Requested from the provider" in page
    assert "not documented: xray_date" in page


def test_approve_then_release(client: TestClient) -> None:
    seen = current_hash(client, APPROVABLE.case_id)
    response = decide(client, APPROVABLE.case_id, "approve", seen)
    assert response.status_code == 303
    page = client.get(response.headers["location"]).text
    assert "Approval recorded for this exact version" in page
    assert "Approved, ready to release" in page
    released = client.post(f"/cases/{APPROVABLE.case_id}/release", follow_redirects=True)
    assert "This dossier has been released." in released.text
    assert "approve by human:ana" in released.text


def test_an_approval_of_an_outdated_version_is_flagged(client: TestClient, log_path: Path) -> None:
    seen = current_hash(client, APPROVABLE.case_id)
    with EventLog(log_path, clock=lambda: datetime.now(UTC)) as log:
        Review(log).revise(
            APPROVABLE.case_id, {"summary": "edited"}, editor="human:ben", seen_hash=seen
        )
    response = decide(client, APPROVABLE.case_id, "approve", seen)
    assert response.headers["location"].endswith("notice=stale")
    page = client.get(response.headers["location"]).text
    assert "does not count" in page
    assert "Awaiting review" in page


def test_release_without_approval_is_refused(client: TestClient) -> None:
    response = client.post(f"/cases/{PENDING.case_id}/release")
    assert response.status_code == 409
    assert "has no approval" in response.text


@pytest.mark.parametrize(
    ("decision", "reviewer"), [("rubber-stamp", "ana"), ("approve", ""), ("approve", "<b>x</b>")]
)
def test_bad_input_is_refused(client: TestClient, decision: str, reviewer: str) -> None:
    seen = current_hash(client, APPROVABLE.case_id)
    response = decide(client, APPROVABLE.case_id, decision, seen, reviewer)
    assert response.status_code == 400
    assert "Choose a decision" in response.text


def test_an_unknown_case_is_not_found(client: TestClient) -> None:
    assert client.get("/cases/SYN-NOPE").status_code == 404
    assert decide(client, "SYN-NOPE", "approve", "0" * 64).status_code == 404
    assert client.post("/cases/SYN-NOPE/release").status_code == 404


def test_document_text_is_escaped(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite"
    hostile = SourceDocument(
        "note", "Clinical note", "Red flags: none.\n<script>alert(1)</script>\n"
    )
    seed(path, "SYN-HTML", [hostile])
    with TestClient(create_app(path)) as client:
        page = client.get("/cases/SYN-HTML").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def priced_model(request: JsonValue) -> JsonValue:
    """The stand-in's answer, with usage as a priced provider would report it."""
    answer = reference_model(request)
    assert isinstance(answer, dict)
    usage: JsonValue = {
        "model": "m",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cost_microusd": 250,
    }
    return {**answer, "usage": usage}


def test_the_timeline_shows_what_the_model_calls_cost(client: TestClient, tmp_path: Path) -> None:
    page = client.get(f"/cases/{APPROVABLE.case_id}").text
    assert "answered by the offline stand-in at no cost" in page

    path = tmp_path / "priced.sqlite"
    ledger = Ledger(tmp_path / "priced-world.sqlite")
    documents = [SourceDocument(d.id, d.title, d.text) for d in APPROVABLE.documents]
    with EventLog(path, clock=lambda: datetime.now(UTC)) as log:
        run_review(Runner(log, "SYN-PRICED"), documents, priced_model, FakeDestination(ledger))
    ledger.close()
    with TestClient(create_app(path)) as priced:
        page = priced.get("/cases/SYN-PRICED").text
    assert "5 model calls, $0.0013 in total" in page
    assert "m · 1000 in / 100 out · $0.0003" in page


def test_a_stopped_case_says_why_and_needs_a_human(tmp_path: Path) -> None:
    path = tmp_path / "stopped.sqlite"
    ledger = Ledger(tmp_path / "stopped-world.sqlite")
    documents = [SourceDocument(d.id, d.title, d.text) for d in APPROVABLE.documents]
    with EventLog(path, clock=lambda: datetime.now(UTC)) as log:
        model = FakeModel(ledger, reference_model)
        runner = Runner(log, "SYN-STOPPED")
        run_review(runner, documents, model, FakeDestination(ledger), Budget(max_model_calls=1))
    ledger.close()
    with TestClient(create_app(path)) as stopped:
        page = stopped.get("/cases/SYN-STOPPED").text
    assert "Stopped early: model call limit reached (0 of 1)" in page
    assert "Recommended: needs human" in page
