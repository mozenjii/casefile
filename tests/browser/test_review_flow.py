"""A real browser drives the review screen: follow evidence, approve, release.

These run as their own gate (``scripts/check.py --browser``) because they need a browser binary; the
default test run deselects them by marker rather than skipping them.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from casefile.agents import SourceDocument, run_review
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.generator import generate
from casefile.log import EventLog
from casefile.reference_model import reference_model
from casefile.web import anchor, create_app

pytestmark = pytest.mark.browser

CASE = generate(31, "none")


def free_port() -> int:
    """Ask the operating system for a port nobody is using."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    """Serve the review screen over a freshly reviewed case, in a background thread."""
    log_path = tmp_path / "events.sqlite"
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(log_path, clock=lambda: datetime.now(UTC)) as log:
        documents = [SourceDocument(d.id, d.title, d.text) for d in CASE.documents]
        model, destination = FakeModel(ledger, reference_model), FakeDestination(ledger)
        run_review(Runner(log, CASE.case_id), documents, model, destination)
    ledger.close()

    port = free_port()
    config = uvicorn.Config(create_app(log_path), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the review server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page() -> Iterator[Page]:
    """A fresh headless Chromium page."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser.new_page(viewport={"width": 1400, "height": 900})
        browser.close()


def test_a_reviewer_follows_evidence_approves_and_releases(base_url: str, page: Page) -> None:
    page.goto(f"{base_url}/cases/{CASE.case_id}")
    expect(page.locator("#review-status")).to_have_text("Awaiting review")
    expect(page.get_by_role("button", name="Release")).to_be_disabled()

    therapy = next(o.evidence for o in CASE.observations if o.fact == "therapy_weeks")
    page.locator(f'a.cite[data-target="{anchor(therapy)}"]').first.click()
    expect(page.locator(f"#{anchor(therapy)}")).to_have_class("active")

    page.get_by_label("Reviewer").fill("ana")
    page.get_by_role("button", name="Approve").click()
    expect(page.get_by_role("status")).to_contain_text("Approval recorded for this exact version")
    expect(page.locator("#review-status")).to_have_text("Approved, ready to release")

    page.get_by_role("button", name="Release").click()
    expect(page.locator("#review-status")).to_have_text("Released")
    expect(page.get_by_text("approve by human:ana")).to_be_visible()


def test_the_approve_key_asks_for_a_name_first(base_url: str, page: Page) -> None:
    page.goto(f"{base_url}/cases/{CASE.case_id}")
    page.evaluate("localStorage.clear()")
    page.reload()
    page.keyboard.press("a")
    expect(page.get_by_label("Reviewer")).to_be_focused()
    expect(page.locator("#review-status")).to_have_text("Awaiting review")
