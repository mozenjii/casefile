"""Tests for the case generator: labels, determinism, and that every span states its fact."""

from __future__ import annotations

from datetime import date

import pytest

from casefile.generator import DEFECTS, INJECTED_INSTRUCTION, Case, generate, generate_set
from casefile.identifiers import find_identifiers
from casefile.rules import Observation, Recommendation

MONTHS = "January February March April May June July August September October November December"


def stated(quote: str, observation: Observation) -> bool:
    """Say whether ``quote`` states the observation's value in a form the documents use."""
    value = observation.value
    if isinstance(value, bool):
        affirmative = quote not in ("No", "none")
        return affirmative is value
    if isinstance(value, int):
        return quote.split()[0] == str(value)
    day = date.fromisoformat(value)
    long_form = f"{day.day} {MONTHS.split()[day.month - 1]} {day.year}"
    return quote in (value, long_form)


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_every_defect_produces_its_declared_recommendation(defect: str) -> None:
    for seed in range(25):
        assert generate(seed, defect).label.recommendation is DEFECTS[defect]


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_every_span_states_the_fact_it_is_labelled_with(defect: str) -> None:
    case = generate(7, defect)
    for observation in case.observations:
        assert stated(case.quote(observation.evidence), observation), observation


def test_the_same_seed_gives_the_same_case() -> None:
    assert generate(3, "stale_xray") == generate(3, "stale_xray")
    assert generate(3, "stale_xray") != generate(4, "stale_xray")


def test_generated_cases_contain_no_possibly_real_identifier() -> None:
    for case in generate_set(36):
        found = [item for document in case.documents for item in find_identifiers(document.text)]
        assert {item.kind for item in found} >= {"phone", "email"}, "fictional ones are expected"
        assert not [item for item in found if item.possibly_real], case.case_id


def documents(case: Case) -> set[str]:
    """The ids of a case's documents."""
    return {document.id for document in case.documents}


def test_defects_shape_the_paperwork() -> None:
    assert documents(generate(1, "none")) == {"form", "note", "xray", "referral"}
    assert "xray" not in documents(generate(1, "missing_xray"))
    assert "referral" not in documents(generate(1, "missing_referral"))
    conflict = generate(1, "therapy_conflict")
    assert {o.evidence for o in conflict.observations if o.fact == "therapy_weeks"} == {
        "form:therapy_weeks",
        "note:therapy_weeks",
    }


def test_the_planted_instruction_appears_only_in_its_own_defect() -> None:
    note = next(d.text for d in generate(2, "injected_instruction").documents if d.id == "note")
    assert INJECTED_INSTRUCTION in note
    for case in generate_set(9):
        if case.defect != "injected_instruction":
            assert all(INJECTED_INSTRUCTION not in d.text for d in case.documents)


def test_a_set_cycles_through_every_defect_with_unique_ids() -> None:
    cases = generate_set(18, seed=100)
    assert [c.defect for c in cases[:9]] == list(DEFECTS)
    assert len({c.case_id for c in cases}) == 18


def test_an_unknown_defect_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown defect"):
        generate(1, "cosmic_rays")


def test_a_label_that_contradicts_its_defect_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(DEFECTS, "none", Recommendation.DENY)
    with pytest.raises(AssertionError, match="rules say approve, the defect declares deny"):
        generate(1, "none")


def test_a_case_id_does_not_give_away_the_defect() -> None:
    for case in generate_set(9):
        assert case.defect.replace("_", "-").upper() not in case.case_id
        assert len(case.case_id) == len("SYN-") + 8
