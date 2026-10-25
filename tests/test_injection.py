"""A hijacked model must never be able to turn a case into an approval.

These reproduce attacks found in a security review: a model obeying an instruction planted in a
document can claim facts by quoting a fragment of a word, or simply leave out the facts that would
deny the case. Neither may produce an approval.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.agents import SourceDocument, ground, run_review
from casefile.canonical import JsonValue
from casefile.executor import Runner
from casefile.fakes import FakeDestination, Ledger
from casefile.generator import Case, generate
from casefile.log import EventLog
from casefile.reference_model import reference_model

type Rewrite = Callable[[str, list[dict[str, JsonValue]]], list[JsonValue]]


@pytest.fixture
def world(tmp_path: Path) -> Iterator[tuple[EventLog, Ledger]]:
    """An empty log and ledger."""
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(tmp_path / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        yield log, ledger
    ledger.close()


def hijacked(rewrite: Rewrite) -> Callable[[JsonValue], JsonValue]:
    """A model that answers like the stand-in, then rewrites its extraction as an attacker would."""

    def answer(request: JsonValue) -> JsonValue:
        reply = reference_model(request)
        if not isinstance(request, dict) or request.get("task") != "extract":
            return reply
        assert isinstance(reply, dict)
        assert isinstance(reply["observations"], list)
        honest = [item for item in reply["observations"] if isinstance(item, dict)]
        return {"observations": rewrite(str(request["text"]), honest)}

    return answer


def outcome(
    world: tuple[EventLog, Ledger], case: Case, model: Callable[[JsonValue], JsonValue]
) -> str:
    """Run the review with ``model`` and return the dossier's recommendation."""
    log, ledger = world
    documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
    state = run_review(Runner(log, case.case_id), documents, model, FakeDestination(ledger))
    assert isinstance(state.outcome, dict)
    return str(state.outcome["recommendation"])


def test_a_fragment_of_a_word_cannot_claim_a_red_flag(world: tuple[EventLog, Ledger]) -> None:
    case = generate(3, "injected_instruction")
    assert case.label.recommendation.value == "deny"

    def forge(text: str, honest: list[dict[str, JsonValue]]) -> list[JsonValue]:
        kept: list[JsonValue] = [
            o for o in honest if o["fact"] not in ("therapy_weeks", "red_flag")
        ]
        if "Red flags: none" in text:
            start = text.index("Red flags: none") + len("Red flags: ")
            kept.append({"fact": "red_flag", "value": True, "start": start, "end": start + 1})
        return kept

    assert outcome(world, case, hijacked(forge)) != "approve"


def test_leaving_out_the_conflicting_fact_cannot_win_an_approval(
    world: tuple[EventLog, Ledger],
) -> None:
    case = generate(4, "therapy_conflict")

    def omit(text: str, honest: list[dict[str, JsonValue]]) -> list[JsonValue]:
        is_form = "PRIOR AUTHORIZATION REQUEST" in text
        return [o for o in honest if not (is_form and o["fact"] == "therapy_weeks")]

    assert outcome(world, case, hijacked(omit)) == "request_info"


@pytest.mark.parametrize(
    ("fact", "value", "quote"),
    [
        ("red_flag", True, "n"),
        ("red_flag", True, "non"),
        ("therapy_weeks", 2026, "2026"),
        ("referral", True, "Referred by"),
    ],
)
def test_words_that_are_not_the_whole_field_are_rejected(
    fact: str, value: JsonValue, quote: str
) -> None:
    note = SourceDocument(
        "note", "Clinical note", "Seen 2026-03-01. Red flags: none.\nReferred by mistake? No.\n"
    )
    kept, rejected = ground(
        note, {"observations": [{"fact": fact, "value": value, "quote": quote}]}
    )
    assert kept == [], kept
    assert rejected
