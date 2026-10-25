"""End-to-end tests: generated cases through the supervisor, and a model that lies.

Every generated case is run through the real workflow with the reference model and must reach the
generator's label, with exactly the ground-truth observations at exactly the ground-truth spans.
The grounding check is then tested against models that misreport what a span says.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.agents import Budget, SourceDocument, ground, run_review
from casefile.canonical import JsonValue
from casefile.executor import Call, Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.generator import DEFECTS, Case, generate
from casefile.log import EventLog
from casefile.quotes import coerce, read_quote
from casefile.reference_model import reference_model
from casefile.state import CaseState


@pytest.fixture
def world(tmp_path: Path) -> Iterator[tuple[EventLog, Ledger]]:
    """Yield an empty log and ledger."""
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(tmp_path / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        yield log, ledger
    ledger.close()


def sources(case: Case) -> list[SourceDocument]:
    """The generated documents as the workflow receives them."""
    return [SourceDocument(d.id, d.title, d.text) for d in case.documents]


def review(world: tuple[EventLog, Ledger], case: Case, model: Call = reference_model) -> CaseState:
    """Run the workflow on ``case`` with ``model`` (the reference model by default)."""
    log, ledger = world
    answer = model
    runner = Runner(log, case.case_id)
    return run_review(runner, sources(case), FakeModel(ledger, answer), FakeDestination(ledger))


def dossier(state: CaseState) -> dict[str, JsonValue]:
    """The dossier the run ended with."""
    assert isinstance(state.outcome, dict)
    return state.outcome


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_every_defect_reaches_its_label_from_the_documents_alone(
    world: tuple[EventLog, Ledger], defect: str
) -> None:
    case = generate(11, defect)
    result = dossier(review(world, case))
    assert result["recommendation"] == case.label.recommendation.value

    truth = {
        (o.fact, o.value, span.document, span.start, span.end)
        for o in case.observations
        for span in case.evidence
        if span.id == o.evidence
    }
    ledger = result["evidence"]
    assert isinstance(ledger, dict)
    found = {
        (item["fact"], item["value"], item["document"], item["start"], item["end"])
        for item in ledger.values()
        if isinstance(item, dict)
    }
    assert found == truth
    assert result["rejected"] == {"observations": [], "claims": []}


def test_the_planted_instruction_changes_nothing(world: tuple[EventLog, Ledger]) -> None:
    case = generate(5, "injected_instruction")
    assert dossier(review(world, case))["recommendation"] == "deny"


def test_a_request_for_information_goes_out_once_with_what_is_missing(
    world: tuple[EventLog, Ledger],
) -> None:
    case = generate(3, "missing_xray")
    result = dossier(review(world, case))
    assert result["requests"] == [
        "Knee X-ray within 90 days before the request (not documented: xray_date)"
    ]
    assert list(world[1].sends().values()) == [1]


def test_every_claim_cites_evidence_in_the_ledger(world: tuple[EventLog, Ledger]) -> None:
    result = dossier(review(world, generate(4, "therapy_conflict")))
    ledger, claims = result["evidence"], result["claims"]
    assert isinstance(ledger, dict)
    assert isinstance(claims, list)
    assert len(claims) == 5
    for claim in claims:
        assert isinstance(claim, dict)
        assert isinstance(claim["evidence"], list)
        assert all(str(ref) in ledger for ref in claim["evidence"])


def test_reviewing_twice_replays_without_calling_the_model(world: tuple[EventLog, Ledger]) -> None:
    case = generate(6, "none")
    first = review(world, case)
    calls = world[1].model_calls()
    assert review(world, case) == first
    assert world[1].model_calls() == calls


DOC = SourceDocument("xray", "X-ray report", "Date: 3 March 2026\nFindings: none.")
NOTE = SourceDocument("note", "Clinical note", "Plan: rest.\nRed flags: suspected fracture.\n")


def fact(name: str, value: JsonValue, start: JsonValue, end: JsonValue) -> JsonValue:
    """One extracted fact as a model would return it."""
    return {"fact": name, "value": value, "start": start, "end": end}


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (fact("xray_date", "2026-03-04", 6, 18), "the span says '3 March 2026', not '2026-03-04'"),
        (fact("xray_date", "2026-03-03", 0, 4), "the span says 'Date', not '2026-03-03'"),
        (fact("xray_date", "2026-03-03", 30, 99), "span is outside the document"),
        (fact("favourite_colour", "blue", 6, 18), "unknown fact"),
        (fact("red_flag", True, 29, 33), "the span says 'none', not True"),
        (fact("red_flag", True, 19, 27), "the span says 'Findings', not True"),
    ],
)
def test_ungrounded_observations_are_rejected(observation: JsonValue, reason: str) -> None:
    kept, rejected = ground(DOC, {"observations": [observation]})
    assert kept == []
    assert len(rejected) == 1
    assert rejected[0].endswith(reason)


def test_a_grounded_observation_carries_its_quote() -> None:
    kept, rejected = ground(DOC, {"observations": [fact("xray_date", "2026-03-03", 6, 18)]})
    assert rejected == []
    assert kept == [
        {
            "fact": "xray_date",
            "value": "2026-03-03",
            "evidence": "xray:xray_date",
            "document": "xray",
            "start": 6,
            "end": 18,
            "quote": "3 March 2026",
        }
    ]


@pytest.mark.parametrize(
    "response",
    [None, {"observations": "all of them"}, {"observations": [{"fact": "x"}]}, {"extra": 1}],
)
def test_a_malformed_extraction_is_rejected_whole(response: JsonValue) -> None:
    kept, rejected = ground(DOC, response)
    assert kept == []
    assert rejected[0].startswith("xray: malformed response")


def test_strict_parsing_does_not_coerce_types() -> None:
    kept, rejected = ground(DOC, {"observations": [fact("xray_date", "2026-03-03", "6", 18)]})
    assert kept == []
    assert "malformed" in rejected[0]


def misreport_therapy(answer: JsonValue) -> JsonValue:
    """Return an extraction with every therapy_weeks value replaced by 99."""
    assert isinstance(answer, dict)
    observations = answer["observations"]
    assert isinstance(observations, list)
    changed: list[JsonValue] = []
    for item in observations:
        assert isinstance(item, dict)
        changed.append({**item, "value": 99} if item["fact"] == "therapy_weeks" else item)
    return {"observations": changed}


def test_a_misreading_model_is_asked_again_and_both_answers_are_on_record(
    world: tuple[EventLog, Ledger],
) -> None:
    def careless(request: JsonValue) -> JsonValue:
        answer = reference_model(request)
        if (
            isinstance(request, dict)
            and request.get("task") == "extract"
            and "rejected" not in request
        ):
            return misreport_therapy(answer)
        return answer

    case = generate(8, "none")
    result = dossier(review(world, case, careless))
    assert result["recommendation"] == "approve"
    rejected = result["rejected"]
    assert isinstance(rejected, dict)
    observations = rejected["observations"]
    assert isinstance(observations, list)
    (only,) = observations
    assert str(only).startswith("note therapy_weeks [")
    assert str(only).endswith(", not 99")
    four_documents_plus_one_retry_plus_one_draft = 6
    assert world[1].model_calls() == four_documents_plus_one_retry_plus_one_draft


def test_a_claim_citing_missing_evidence_is_dropped(world: tuple[EventLog, Ledger]) -> None:
    def inventive(request: JsonValue) -> JsonValue:
        answer = reference_model(request)
        if isinstance(request, dict) and request.get("task") == "write":
            assert isinstance(answer, dict)
            assert isinstance(answer["claims"], list)
            invented: JsonValue = {"text": "Looks fine.", "evidence": ["note:vibes"]}
            return {**answer, "claims": [*answer["claims"], invented]}
        return answer

    result = dossier(review(world, generate(9, "none"), inventive))
    rejected = result["rejected"]
    assert isinstance(rejected, dict)
    assert rejected["claims"] == ["claim cites evidence that does not exist: note:vibes"]
    claims = result["claims"]
    assert isinstance(claims, list)
    assert len(claims) == 5


def test_a_malformed_draft_leaves_an_empty_dossier_text_not_a_crash(
    world: tuple[EventLog, Ledger],
) -> None:
    def mute(request: JsonValue) -> JsonValue:
        if isinstance(request, dict) and request.get("task") == "write":
            return {"summary": 42}
        return reference_model(request)

    result = dossier(review(world, generate(10, "none"), mute))
    assert result["summary"] == ""
    assert result["claims"] == []
    assert result["recommendation"] == "approve"


def test_a_yes_no_fact_counts_only_after_its_label() -> None:
    kept, rejected = ground(NOTE, {"observations": [fact("red_flag", True, 23, 41)]})
    assert rejected == []
    (grounded,) = kept
    assert isinstance(grounded, dict)
    assert grounded["quote"] == "suspected fracture"
    kept, rejected = ground(NOTE, {"observations": [fact("red_flag", True, 6, 10)]})
    assert kept == []
    assert "not True" in rejected[0]


@pytest.mark.parametrize(
    ("request_", "message"), [("hello", "JSON object"), ({"task": "sing"}, "no task")]
)
def test_the_reference_model_refuses_what_it_was_not_built_for(
    request_: JsonValue, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reference_model(request_)


def test_a_quote_for_an_unknown_fact_states_nothing() -> None:
    assert read_quote("favourite_colour", "blue") is None
    assert read_quote("therapy_weeks", "several weeks") is None


TWICE = SourceDocument(
    "note", "Clinical note", "Seen 2 weeks ago. They completed 7 weeks of therapy.\n"
)


def quoted(name: str, value: JsonValue, quote: str) -> JsonValue:
    """One extracted fact as a language model returns it: the value as text, and the quote."""
    return {"fact": name, "value": value, "quote": quote}


def test_a_quoted_fact_is_located_in_the_document() -> None:
    kept, rejected = ground(
        DOC, {"observations": [quoted("xray_date", "2026-03-03", "3 March 2026")]}
    )
    assert rejected == []
    (item,) = kept
    assert isinstance(item, dict)
    assert (item["value"], item["start"], item["end"]) == ("2026-03-03", 6, 18)


def test_a_repeated_quote_resolves_to_the_occurrence_that_states_the_value() -> None:
    kept, rejected = ground(TWICE, {"observations": [quoted("therapy_weeks", "7", "weeks")]})
    assert kept == []
    assert "not '7'" in rejected[0]
    kept, rejected = ground(TWICE, {"observations": [quoted("therapy_weeks", "7", "7 weeks")]})
    assert rejected == []
    (item,) = kept
    assert isinstance(item, dict)
    assert item["value"] == 7
    assert TWICE.text[int(str(item["start"])) : int(str(item["end"]))] == "7 weeks"


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (quoted("xray_date", "2026-03-03", "4 March 2026"), "is not in the document"),
        (quoted("therapy_weeks", "six", "6 weeks"), "is not a valid value for this fact"),
        ({"fact": "xray_date", "value": "2026-03-03"}, "no span and no quote given"),
    ],
)
def test_bad_quotes_are_rejected(observation: JsonValue, reason: str) -> None:
    kept, rejected = ground(DOC, {"observations": [observation]})
    assert kept == []
    assert rejected[0].endswith(reason)


@pytest.mark.parametrize(
    ("fact", "value", "expected"),
    [
        ("therapy_weeks", "6", 6),
        ("therapy_weeks", 6, 6),
        ("therapy_weeks", True, None),
        ("red_flag", "Yes", True),
        ("red_flag", "false", False),
        ("red_flag", "maybe", None),
        ("xray_date", "3 March 2026", "2026-03-03"),
        ("xray_date", 20260303, None),
        ("favourite_colour", "blue", None),
    ],
)
def test_reported_values_are_normalised_to_the_fact_type(
    fact: str, value: str | int | bool, expected: object
) -> None:
    assert coerce(fact, value) == expected


def budgeted(
    world: tuple[EventLog, Ledger], case: Case, budget: Budget, estimate: int | None = 250
) -> CaseState:
    """Run ``case`` under ``budget``, pricing every model request at ``estimate``."""
    log, ledger = world
    runner = Runner(log, case.case_id, estimate=lambda _: estimate)
    model = FakeModel(ledger, reference_model)
    return run_review(runner, sources(case), model, FakeDestination(ledger), budget)


def test_a_call_limit_stops_the_run_with_what_was_found(world: tuple[EventLog, Ledger]) -> None:
    case = generate(12, "none")
    state = budgeted(world, case, Budget(max_model_calls=2))
    result = dossier(state)
    assert result["recommendation"] == "needs_human"
    assert result["stopped"] == "model call limit reached (2 of 2)"
    assert state.stopped == "model call limit reached (2 of 2)"
    ledger = result["evidence"]
    assert isinstance(ledger, dict)
    assert {str(item["document"]) for item in ledger.values() if isinstance(item, dict)} == {
        "form",
        "note",
    }
    assert world[1].model_calls() == 2
    assert [e.type for e in world[0].read(case.case_id)][-2:] == [
        "BudgetExhausted",
        "CaseCompleted",
    ]


def test_a_cost_limit_counts_what_was_spent(world: tuple[EventLog, Ledger]) -> None:
    case = generate(13, "none")
    state = budgeted(world, case, Budget(max_cost_microusd=600), estimate=250)
    assert state.stopped is not None
    assert state.stopped.startswith("cost limit reached")
    assert state.model_calls == 2


def test_a_cost_limit_without_a_price_fails_closed(world: tuple[EventLog, Ledger]) -> None:
    state = budgeted(world, generate(14, "none"), Budget(max_cost_microusd=10_000), estimate=None)
    assert state.stopped == "a cost limit is set but the model's price is unknown"
    assert world[1].model_calls() == 0


def test_a_budget_that_is_never_reached_changes_nothing(world: tuple[EventLog, Ledger]) -> None:
    case = generate(15, "none")
    state = budgeted(world, case, Budget(max_model_calls=50, max_cost_microusd=1_000_000))
    assert state.stopped is None
    assert dossier(state)["recommendation"] == "approve"


def test_a_stopped_case_resumes_to_the_same_stop(world: tuple[EventLog, Ledger]) -> None:
    case = generate(16, "none")
    first = budgeted(world, case, Budget(max_model_calls=1))
    assert budgeted(world, case, Budget(max_model_calls=1)) == first
    assert budgeted(world, case, Budget(max_model_calls=99)) == first


def test_a_limit_can_stop_the_draft_after_the_rules_ran(world: tuple[EventLog, Ledger]) -> None:
    case = generate(17, "missing_xray")
    result = dossier(budgeted(world, case, Budget(max_model_calls=3)))
    assert result["recommendation"] == "needs_human"
    criteria = result["criteria"]
    assert isinstance(criteria, list)
    assert len(criteria) == 5
    assert result["requests"] == [
        "Knee X-ray within 90 days before the request (not documented: xray_date)"
    ]
