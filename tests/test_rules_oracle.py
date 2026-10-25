"""Independent oracle for the synthetic knee-MRI criteria, written before their implementation.

Every expected value here is worked out by hand from the criteria specification: the fact table and
its unknown/conflict semantics, the five criteria with their propagation rules, the severity order
of the recommendation, and the ``casefile.rules`` interface. Nothing was taken from the code under
test. Where the specification leaves a choice open, this file does not assert it.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import pytest

from casefile.rules import (
    CriterionResult,
    Decision,
    Observation,
    Recommendation,
    Status,
    evaluate,
)

MET = Status.MET
NOT_MET = Status.NOT_MET
UNKNOWN = Status.UNKNOWN
CONFLICT = Status.CONFLICT
APPROVE = Recommendation.APPROVE
REQUEST_INFO = Recommendation.REQUEST_INFO
DENY = Recommendation.DENY

ORDER = ("C1", "C2", "C3", "C4", "X1")

# The specification fixes the exception type for a bad fact name or value, not its message, so the
# match pattern accepts any message, including an empty one.
ANY_MESSAGE = r"(?s).*"

# A fixed request date. 2026 is not a leap year, so February has 28 days.
REQUEST = date(2026, 3, 1)


def days_before(days: int) -> str:
    """Return the ISO date ``days`` days before the request date; negative means after it."""
    return (REQUEST - timedelta(days=days)).isoformat()


def obs(fact: str, value: str | int | bool, evidence: str) -> Observation:
    """Build one observation of ``fact`` with ``value``, cited by ``evidence``."""
    return Observation(fact=fact, value=value, evidence=evidence)


# One evidence id per fact for the baseline case. They are chosen so that sorting reorders them.
EVIDENCE: dict[str, str] = {
    "request_date": "form:date",
    "prior_mri": "form:prior",
    "prior_mri_date": "form:prior-date",
    "symptom_onset": "note:onset",
    "therapy_weeks": "note:therapy",
    "red_flag": "note:flags",
    "xray_date": "xray:date",
    "referral": "referral:letter",
}

# A case that meets every criterion: 70 days of symptoms (10 weeks), 8 weeks of therapy and no red
# flag, an X-ray 30 days old, a referral letter, and no prior MRI.
BASELINE: dict[str, str | int | bool] = {
    "request_date": REQUEST.isoformat(),
    "prior_mri": False,
    "symptom_onset": days_before(70),
    "therapy_weeks": 8,
    "red_flag": False,
    "xray_date": days_before(30),
    "referral": True,
}


def case(
    *extra: Observation, drop: tuple[str, ...] = (), **values: str | int | bool
) -> list[Observation]:
    """Return the baseline case with ``values`` replaced, ``drop`` removed and ``extra`` appended.

    A replaced or added fact keeps its baseline evidence id from ``EVIDENCE``.
    """
    merged = {**BASELINE, **values}
    kept = [obs(fact, value, EVIDENCE[fact]) for fact, value in merged.items() if fact not in drop]
    return [*kept, *extra]


def result(decision: Decision, criterion: str) -> CriterionResult:
    """Return the result for ``criterion`` from ``decision``."""
    by_id = {each.id: each for each in decision.results}
    return by_id[criterion]


def status_of(observations: list[Observation], criterion: str) -> Status:
    """Evaluate ``observations`` and return the status of ``criterion``."""
    return result(evaluate(observations), criterion).status


def statuses(decision: Decision) -> tuple[Status, ...]:
    """Return the five statuses of ``decision`` in result order."""
    return tuple(each.status for each in decision.results)


# --- The arithmetic the boundary tests rely on -------------------------------------------------


def test_the_boundary_dates_are_the_ones_worked_out_by_hand() -> None:
    # 1 Mar 2026 - 28 days = 1 Feb; - 14 more = 18 Jan (42 days in all).
    assert days_before(42) == "2026-01-18"
    # 1 Mar - 28 = 1 Feb; - 31 = 1 Jan; - 31 = 1 Dec 2025: 90 days.
    assert days_before(90) == "2025-12-01"
    assert days_before(91) == "2025-11-30"
    # 2025 is not a leap year, so one calendar year back is exactly 365 days.
    assert days_before(365) == "2025-03-01"
    assert days_before(366) == "2025-02-28"


# --- Shape of the decision ---------------------------------------------------------------------


def test_the_enumerations_carry_the_specified_values() -> None:
    assert [s.value for s in Status] == ["met", "not_met", "unknown", "conflict"]
    assert [r.value for r in Recommendation] == ["approve", "request_info", "deny"]


def test_results_come_back_in_the_specified_order() -> None:
    decision = evaluate(case())
    assert tuple(each.id for each in decision.results) == ORDER


def test_results_come_back_in_order_even_when_nothing_is_observed() -> None:
    decision = evaluate([])
    assert tuple(each.id for each in decision.results) == ORDER


def test_any_iterable_of_observations_is_accepted() -> None:
    assert evaluate(iter(case())) == evaluate(case())


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(evaluate(case()), id="Decision"),
        pytest.param(evaluate(case()).results[0], id="CriterionResult"),
        pytest.param(obs("referral", True, "referral:letter"), id="Observation"),
    ],
)
def test_the_value_types_are_frozen_dataclasses(instance: object) -> None:
    assert dataclasses.is_dataclass(instance)
    for field in dataclasses.fields(instance):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field.name, getattr(instance, field.name))


# --- C1: symptoms for at least 6 weeks ---------------------------------------------------------


@pytest.mark.parametrize(
    ("onset_days_before", "expected"),
    [
        # 42 // 7 = 6 >= 6.
        pytest.param(42, MET, id="exactly-6-weeks"),
        # 41 // 7 = 5 < 6: 5.86 weeks floors to 5, so rounding would be wrong here.
        pytest.param(41, NOT_MET, id="one-day-short"),
        # 48 // 7 = 6: 6.86 weeks floors to 6.
        pytest.param(48, MET, id="floored-to-6"),
        # 70 // 7 = 10.
        pytest.param(70, MET, id="10-weeks"),
        # Onset on the request date: 0 // 7 = 0.
        pytest.param(0, NOT_MET, id="same-day"),
        # Onset after the request: -1 // 7 = -1.
        pytest.param(-1, NOT_MET, id="onset-after-request"),
    ],
)
def test_c1_compares_whole_weeks_of_symptoms(onset_days_before: int, expected: Status) -> None:
    assert status_of(case(symptom_onset=days_before(onset_days_before)), "C1") == expected


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        pytest.param(case(drop=("symptom_onset",)), UNKNOWN, id="onset-missing"),
        pytest.param(case(drop=("request_date",)), UNKNOWN, id="request-date-missing"),
        pytest.param(case(drop=("request_date", "symptom_onset")), UNKNOWN, id="both-missing"),
        pytest.param(
            case(obs("symptom_onset", days_before(20), "form:onset")),
            CONFLICT,
            id="onset-conflict",
        ),
        pytest.param(
            case(obs("request_date", days_before(1), "email:date")),
            CONFLICT,
            id="request-date-conflict",
        ),
        pytest.param(
            case(obs("request_date", days_before(1), "email:date"), drop=("symptom_onset",)),
            CONFLICT,
            id="request-conflict-beats-onset-unknown",
        ),
        pytest.param(
            case(obs("symptom_onset", days_before(20), "form:onset"), drop=("request_date",)),
            CONFLICT,
            id="onset-conflict-beats-request-unknown",
        ),
    ],
)
def test_c1_propagates_unknown_and_conflict(
    observations: list[Observation], expected: Status
) -> None:
    assert status_of(observations, "C1") == expected


# --- C2: conservative care tried, or a red flag (Kleene OR) ------------------------------------


def red_flag_observations(state: str) -> list[Observation]:
    """Observations that give ``red_flag`` the value state true, false, unknown or conflict."""
    match state:
        case "true":
            return [obs("red_flag", True, "note:flags")]
        case "false":
            return [obs("red_flag", False, "note:flags")]
        case "unknown":
            return []
        case _:
            return [obs("red_flag", True, "note:flags"), obs("red_flag", False, "form:flags")]


def therapy_observations(state: str) -> list[Observation]:
    """Observations that give ``therapy_weeks`` the state ge6, lt6, unknown or conflict."""
    match state:
        case "ge6":
            return [obs("therapy_weeks", 6, "note:therapy")]
        case "lt6":
            return [obs("therapy_weeks", 5, "note:therapy")]
        case "unknown":
            return []
        case _:
            return [
                obs("therapy_weeks", 8, "note:therapy"),
                obs("therapy_weeks", 4, "form:therapy"),
            ]


# Rows: red_flag. Columns: therapy_weeks >= 6 (6), < 6 (5), unknown (absent), conflict (8 vs 4).
# met if either side is met; not_met only if both are known and not met; otherwise conflict if
# either side is a conflict, else unknown.
KLEENE_OR: list[tuple[str, str, Status]] = [
    ("true", "ge6", MET),
    ("true", "lt6", MET),
    ("true", "unknown", MET),
    ("true", "conflict", MET),
    ("false", "ge6", MET),
    ("false", "lt6", NOT_MET),
    ("false", "unknown", UNKNOWN),
    ("false", "conflict", CONFLICT),
    ("unknown", "ge6", MET),
    ("unknown", "lt6", UNKNOWN),
    ("unknown", "unknown", UNKNOWN),
    ("unknown", "conflict", CONFLICT),
    ("conflict", "ge6", MET),
    ("conflict", "lt6", CONFLICT),
    ("conflict", "unknown", CONFLICT),
    ("conflict", "conflict", CONFLICT),
]


def test_the_kleene_table_covers_every_pair_once() -> None:
    pairs = {(red, therapy) for red, therapy, _ in KLEENE_OR}
    assert len(KLEENE_OR) == len(pairs) == 16


@pytest.mark.parametrize(("red_flag", "therapy", "expected"), KLEENE_OR)
def test_c2_is_a_kleene_or_of_red_flag_and_therapy(
    red_flag: str, therapy: str, expected: Status
) -> None:
    observations = [*red_flag_observations(red_flag), *therapy_observations(therapy)]
    assert status_of(observations, "C2") == expected


@pytest.mark.parametrize(
    ("weeks", "expected"),
    [
        pytest.param(6, MET, id="exactly-6"),
        pytest.param(5, NOT_MET, id="5"),
        pytest.param(0, NOT_MET, id="none"),
        pytest.param(12, MET, id="12"),
    ],
)
def test_c2_therapy_boundary_without_a_red_flag(weeks: int, expected: Status) -> None:
    assert status_of(case(therapy_weeks=weeks, red_flag=False), "C2") == expected


def test_c2_a_red_flag_makes_short_therapy_irrelevant() -> None:
    assert status_of(case(therapy_weeks=2, red_flag=True), "C2") == MET


# --- C3: recent X-ray --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xray_days_before", "expected"),
    [
        # 0 <= 0 <= 90.
        pytest.param(0, MET, id="same-day"),
        # 0 <= 90 <= 90.
        pytest.param(90, MET, id="exactly-90-days"),
        # 91 > 90.
        pytest.param(91, NOT_MET, id="91-days"),
        # Taken after the request: -1 < 0.
        pytest.param(-1, NOT_MET, id="after-request"),
    ],
)
def test_c3_requires_an_xray_within_90_days_before_the_request(
    xray_days_before: int, expected: Status
) -> None:
    assert status_of(case(xray_date=days_before(xray_days_before)), "C3") == expected


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        pytest.param(case(drop=("xray_date",)), UNKNOWN, id="xray-missing"),
        pytest.param(case(drop=("request_date",)), UNKNOWN, id="request-date-missing"),
        pytest.param(
            case(obs("xray_date", days_before(200), "xray:old")),
            CONFLICT,
            id="xray-conflict",
        ),
        pytest.param(
            case(obs("request_date", days_before(1), "email:date")),
            CONFLICT,
            id="request-date-conflict",
        ),
        pytest.param(
            case(obs("xray_date", days_before(200), "xray:old"), drop=("request_date",)),
            CONFLICT,
            id="xray-conflict-beats-request-unknown",
        ),
        pytest.param(
            case(obs("request_date", days_before(1), "email:date"), drop=("xray_date",)),
            CONFLICT,
            id="request-conflict-beats-xray-unknown",
        ),
    ],
)
def test_c3_propagates_unknown_and_conflict(
    observations: list[Observation], expected: Status
) -> None:
    assert status_of(observations, "C3") == expected


# --- C4: referral on file ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        pytest.param(case(referral=True), MET, id="true"),
        pytest.param(case(referral=False), NOT_MET, id="false"),
        pytest.param(case(drop=("referral",)), UNKNOWN, id="no-observation"),
        pytest.param(case(obs("referral", False, "form:referral")), CONFLICT, id="true-and-false"),
    ],
)
def test_c4_reads_the_referral(observations: list[Observation], expected: Status) -> None:
    assert status_of(observations, "C4") == expected


# --- X1: no MRI of this knee in the last year --------------------------------------------------


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        # 365 > 365 is false.
        pytest.param(
            case(prior_mri=True, prior_mri_date=days_before(365)), NOT_MET, id="exactly-365-days"
        ),
        # 366 > 365.
        pytest.param(case(prior_mri=True, prior_mri_date=days_before(366)), MET, id="366-days"),
        pytest.param(case(prior_mri=True, prior_mri_date=days_before(10)), NOT_MET, id="recent"),
        pytest.param(case(prior_mri=False), MET, id="no-prior-no-date"),
        # prior_mri false gives met whatever the date, even a recent one.
        pytest.param(
            case(prior_mri=False, prior_mri_date=days_before(10)), MET, id="no-prior-with-date"
        ),
    ],
)
def test_x1_boundaries(observations: list[Observation], expected: Status) -> None:
    assert status_of(observations, "X1") == expected


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        pytest.param(case(prior_mri=True), UNKNOWN, id="prior-but-no-date"),
        pytest.param(
            case(
                obs("prior_mri_date", days_before(400), "form:prior-date-2"),
                prior_mri=True,
                prior_mri_date=days_before(10),
            ),
            CONFLICT,
            id="prior-with-conflicting-dates",
        ),
        pytest.param(
            case(prior_mri=True, prior_mri_date=days_before(400), drop=("request_date",)),
            UNKNOWN,
            id="prior-but-no-request-date",
        ),
        pytest.param(
            case(
                obs("request_date", days_before(1), "email:date"),
                prior_mri=True,
                prior_mri_date=days_before(400),
            ),
            CONFLICT,
            id="prior-with-conflicting-request-dates",
        ),
        pytest.param(case(drop=("prior_mri",)), UNKNOWN, id="prior-unknown"),
        pytest.param(
            case(prior_mri_date=days_before(400), drop=("prior_mri",)),
            UNKNOWN,
            id="prior-unknown-even-with-an-old-date",
        ),
        pytest.param(
            case(obs("prior_mri", True, "note:prior")),
            CONFLICT,
            id="prior-conflict",
        ),
        pytest.param(
            case(obs("prior_mri", True, "note:prior"), prior_mri_date=days_before(400)),
            CONFLICT,
            id="prior-conflict-even-with-an-old-date",
        ),
    ],
)
def test_x1_propagates_unknown_and_conflict(
    observations: list[Observation], expected: Status
) -> None:
    assert status_of(observations, "X1") == expected


# --- Repeated observations and evidence --------------------------------------------------------


def test_the_same_value_observed_twice_is_not_a_conflict_and_cites_both() -> None:
    decision = evaluate(case(obs("therapy_weeks", 8, "form:therapy")))
    c2 = result(decision, "C2")
    assert c2.status == MET
    # red_flag false and therapy 8 from both documents: both facts decide, so all three are cited,
    # sorted.
    assert c2.evidence == ("form:therapy", "note:flags", "note:therapy")


def test_the_same_date_observed_twice_is_not_a_conflict() -> None:
    decision = evaluate(case(obs("request_date", REQUEST.isoformat(), "email:date")))
    c1 = result(decision, "C1")
    assert c1.status == MET
    assert c1.evidence == ("email:date", "form:date", "note:onset")


def test_a_repeated_observation_is_cited_once() -> None:
    letter = obs("referral", True, "referral:letter")
    decision = evaluate([*case(drop=("referral",)), letter, letter])
    c4 = result(decision, "C4")
    assert c4.status == MET
    assert c4.evidence == ("referral:letter",)


def test_each_criterion_cites_only_the_facts_it_read() -> None:
    decision = evaluate(case())
    assert result(decision, "C1").evidence == ("form:date", "note:onset")
    assert result(decision, "C2").evidence == ("note:flags", "note:therapy")
    assert result(decision, "C3").evidence == ("form:date", "xray:date")
    assert result(decision, "C4").evidence == ("referral:letter",)


def test_x1_cites_the_prior_mri_its_date_and_the_request_date() -> None:
    decision = evaluate(case(prior_mri=True, prior_mri_date=days_before(400)))
    x1 = result(decision, "X1")
    assert x1.status == MET
    assert x1.evidence == ("form:date", "form:prior", "form:prior-date")


def test_a_not_met_result_cites_its_evidence_too() -> None:
    decision = evaluate(case(therapy_weeks=5, xray_date=days_before(91)))
    assert result(decision, "C2").evidence == ("note:flags", "note:therapy")
    assert result(decision, "C3").evidence == ("form:date", "xray:date")


def test_evidence_is_sorted_not_in_input_order() -> None:
    observations = [
        obs("xray_date", days_before(30), "z:xray"),
        obs("request_date", REQUEST.isoformat(), "a:form"),
    ]
    assert result(evaluate(observations), "C3").evidence == ("a:form", "z:xray")


def test_nothing_observed_cites_nothing() -> None:
    for each in evaluate([]).results:
        assert each.evidence == ()


# --- Recommendation ----------------------------------------------------------------------------

# Each row: observations, the five statuses in order, the recommendation.
RECOMMENDATIONS = [
    pytest.param(case(), (MET, MET, MET, MET, MET), APPROVE, id="all-met"),
    pytest.param(
        case(drop=("referral",)), (MET, MET, MET, UNKNOWN, MET), REQUEST_INFO, id="one-unknown"
    ),
    pytest.param(
        case(obs("therapy_weeks", 4, "form:therapy")),
        (MET, CONFLICT, MET, MET, MET),
        REQUEST_INFO,
        id="one-conflict",
    ),
    pytest.param(
        case(xray_date=days_before(91)),
        (MET, MET, NOT_MET, MET, MET),
        REQUEST_INFO,
        id="c3-not-met-only",
    ),
    pytest.param(
        case(referral=False),
        (MET, MET, MET, NOT_MET, MET),
        REQUEST_INFO,
        id="c4-not-met-only",
    ),
    pytest.param(
        case(symptom_onset=days_before(41), drop=("referral",)),
        (NOT_MET, MET, MET, UNKNOWN, MET),
        DENY,
        id="c1-not-met-with-missing-referral",
    ),
    pytest.param(
        case(therapy_weeks=5),
        (MET, NOT_MET, MET, MET, MET),
        DENY,
        id="c2-not-met",
    ),
    pytest.param(
        case(prior_mri=True, prior_mri_date=days_before(100)),
        (MET, MET, MET, MET, NOT_MET),
        DENY,
        id="x1-not-met",
    ),
    pytest.param(
        case(prior_mri=True, prior_mri_date=days_before(100), drop=("xray_date", "referral")),
        (MET, MET, UNKNOWN, UNKNOWN, NOT_MET),
        DENY,
        id="deny-not-softened-by-missing-documents",
    ),
    pytest.param(
        [],
        (UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN),
        REQUEST_INFO,
        id="nothing-observed",
    ),
]


@pytest.mark.parametrize(("observations", "expected", "recommendation"), RECOMMENDATIONS)
def test_the_recommendation_is_the_most_severe_result(
    observations: list[Observation],
    expected: tuple[Status, ...],
    recommendation: Recommendation,
) -> None:
    decision = evaluate(observations)
    assert statuses(decision) == expected
    assert decision.recommendation == recommendation


# --- Refused input -----------------------------------------------------------------------------


def test_an_unknown_fact_name_is_refused() -> None:
    with pytest.raises(ValueError, match=ANY_MESSAGE):
        evaluate([*case(), obs("patient_age", 40, "form:age")])


WRONG_TYPES: list[tuple[str, str | int | bool]] = [
    ("therapy_weeks", "six"),
    ("therapy_weeks", "6"),
    # bool is a subclass of int in Python, but the fact is an int and True is a bool.
    ("therapy_weeks", True),
    ("red_flag", 1),
    ("red_flag", "yes"),
    ("prior_mri", 0),
    ("referral", "present"),
    ("request_date", 20260101),
    ("symptom_onset", True),
    ("xray_date", 20251201),
    ("prior_mri_date", False),
]


@pytest.mark.parametrize(("fact", "value"), WRONG_TYPES)
def test_a_value_of_the_wrong_type_is_refused(fact: str, value: str | int | bool) -> None:
    with pytest.raises(ValueError, match=ANY_MESSAGE):
        evaluate([*case(drop=(fact,)), obs(fact, value, "bad:value")])
