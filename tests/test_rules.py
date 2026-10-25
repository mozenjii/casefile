"""Tests for the rules engine's explanations and its handling of bad input.

Boundaries and the full three-valued tables are pinned by the independent oracle; these tests
cover what a reviewer reads: the reason attached to each result.
"""

from __future__ import annotations

import pytest

from casefile.rules import CRITERION_TEXT, Observation, Recommendation, Status, evaluate

COMPLETE = [
    Observation("request_date", "2026-03-01", "form:date"),
    Observation("prior_mri", False, "form:prior"),
    Observation("symptom_onset", "2026-01-01", "note:onset"),
    Observation("therapy_weeks", 6, "note:therapy"),
    Observation("red_flag", False, "note:flags"),
    Observation("xray_date", "2026-02-10", "xray:date"),
    Observation("referral", True, "referral:letter"),
]


def reasons(observations: list[Observation]) -> dict[str, tuple[Status, str]]:
    """Map each criterion id to its status and reason."""
    return {r.id: (r.status, r.reason) for r in evaluate(observations).results}


def test_a_complete_case_explains_every_criterion() -> None:
    decision = evaluate(COMPLETE)
    assert decision.recommendation is Recommendation.APPROVE
    assert reasons(COMPLETE) == {
        "C1": (Status.MET, "symptoms for 8 weeks before the request (needs 6)"),
        "C2": (Status.MET, "6 weeks of conservative therapy (needs 6)"),
        "C3": (Status.MET, "X-ray taken 19 days before the request (needs 90 or fewer)"),
        "C4": (Status.MET, "referral on file"),
        "X1": (Status.MET, "no prior MRI of this knee"),
    }
    assert set(CRITERION_TEXT) == {result.id for result in decision.results}


def test_uncertain_results_say_what_is_missing_or_disagrees() -> None:
    observations = [o for o in COMPLETE if o.fact != "xray_date"]
    observations.append(Observation("therapy_weeks", 4, "form:therapy"))
    found = reasons(observations)
    assert found["C3"] == (Status.UNKNOWN, "not documented: xray_date")
    assert found["C2"] == (Status.CONFLICT, "documents disagree on therapy_weeks")


def test_a_failed_or_explains_both_sides() -> None:
    observations = [o for o in COMPLETE if o.fact != "therapy_weeks"]
    observations.append(Observation("therapy_weeks", 2, "note:therapy"))
    assert reasons(observations)["C2"] == (
        Status.NOT_MET,
        "no red flag; 2 weeks of conservative therapy (needs 6)",
    )


def test_a_recent_prior_mri_explains_the_gap() -> None:
    observations = [o for o in COMPLETE if o.fact != "prior_mri"]
    observations += [
        Observation("prior_mri", True, "form:prior"),
        Observation("prior_mri_date", "2025-09-01", "form:prior-date"),
    ]
    assert reasons(observations)["X1"] == (
        Status.NOT_MET,
        "prior MRI 181 days before the request (needs over 365)",
    )


def test_a_malformed_date_is_refused() -> None:
    with pytest.raises(ValueError, match="Invalid isoformat"):
        evaluate([Observation("xray_date", "last spring", "xray:date")])


def test_an_unknown_fact_is_refused_before_anything_else() -> None:
    with pytest.raises(ValueError, match="unknown fact 'favourite_colour'"):
        evaluate([Observation("favourite_colour", "blue", "note:1")])
