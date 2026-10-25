"""The criteria for the synthetic procedure, evaluated with unknowns and conflicts kept visible.

The procedure (an MRI of one knee) and its criteria are invented for this project and are **not
clinical guidance**. What matters is how they are evaluated:

* A fact nobody documented is **unknown**, and two documents that disagree make it a **conflict**.
  Neither is ever quietly turned into a pass or a fail.
* Unknowns and conflicts flow through each criterion. A comparison over several facts is a conflict
  if any input is, else unknown if any input is. "Red flag *or* six weeks of therapy" follows
  three-valued logic: a documented red flag satisfies it even when therapy is unknown, but it only
  fails when both sides are known to fail.
* Each outcome has a severity: met means approve; not met means the criterion's own consequence
  (deny, or ask for more information); unknown and conflict mean ask for more information. The
  recommendation is the most severe of them. A definite deny is never softened by a missing
  document elsewhere, and nothing unknown is ever approved.

The engine is pure and deterministic. It is both how cases are labelled and the policy checker the
agents call, so its expected results are pinned by tests written from the specification alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class Status(StrEnum):
    """The outcome of one criterion."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class Recommendation(StrEnum):
    """What the case should get, in increasing severity."""

    APPROVE = "approve"
    REQUEST_INFO = "request_info"
    DENY = "deny"


_SEVERITY = {Recommendation.APPROVE: 0, Recommendation.REQUEST_INFO: 1, Recommendation.DENY: 2}

FACT_TYPES: Mapping[str, type] = {
    "request_date": date,
    "prior_mri": bool,
    "prior_mri_date": date,
    "symptom_onset": date,
    "therapy_weeks": int,
    "red_flag": bool,
    "xray_date": date,
    "referral": bool,
}
"""Every fact the criteria read, with its type. Dates travel as ISO strings."""


@dataclass(frozen=True)
class Observation:
    """One fact as one document states it, with the id of the span that states it."""

    fact: str
    value: str | int | bool
    evidence: str


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's outcome, the evidence it read, and a sentence a reviewer can check."""

    id: str
    status: Status
    evidence: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Decision:
    """The recommendation and every criterion's result, in the order the criteria are listed."""

    recommendation: Recommendation
    results: tuple[CriterionResult, ...]


@dataclass(frozen=True)
class _Fact:
    """A fact after reading every observation of it: a value, or why there is none."""

    name: str
    status: Status  # MET stands for "known" here; UNKNOWN and CONFLICT mean what they say
    value: object
    evidence: tuple[str, ...]


type _Facts = Mapping[str, _Fact]
type _Outcome = tuple[Status, str]


@dataclass(frozen=True)
class _Criterion:
    """A criterion as data: what it reads, what failing it costs, and how it is decided."""

    id: str
    reads: tuple[str, ...]
    if_not_met: Recommendation
    decide: Callable[[_Facts], _Outcome]


def evaluate(observations: Iterable[Observation]) -> Decision:
    """Evaluate every criterion over the observations and return the recommendation."""
    facts = _resolve(list(observations))
    results = tuple(_result(criterion, facts) for criterion in CRITERIA)
    severities = [
        _severity(criterion, result) for criterion, result in zip(CRITERIA, results, strict=True)
    ]
    return Decision(max(severities, key=_SEVERITY.__getitem__), results)


def _result(criterion: _Criterion, facts: _Facts) -> CriterionResult:
    """Decide one criterion and attach the evidence of every fact it read."""
    status, reason = criterion.decide(facts)
    cited = sorted({ref for name in criterion.reads for ref in facts[name].evidence})
    return CriterionResult(criterion.id, status, tuple(cited), reason)


def _severity(criterion: _Criterion, result: CriterionResult) -> Recommendation:
    """Map a result to what it means for the case."""
    if result.status is Status.MET:
        return Recommendation.APPROVE
    if result.status is Status.NOT_MET:
        return criterion.if_not_met
    return Recommendation.REQUEST_INFO


def _resolve(observations: list[Observation]) -> dict[str, _Fact]:
    """Group observations by fact: none is unknown, disagreeing values are a conflict."""
    values: dict[str, set[object]] = {name: set() for name in FACT_TYPES}
    evidence: dict[str, list[str]] = {name: [] for name in FACT_TYPES}
    for observation in observations:
        parsed = _parse(observation)  # rejects an unknown fact before it is used as a key
        values[observation.fact].add(parsed)
        evidence[observation.fact].append(observation.evidence)

    facts: dict[str, _Fact] = {}
    for name in FACT_TYPES:
        refs = tuple(sorted(set(evidence[name])))
        if not values[name]:
            facts[name] = _Fact(name, Status.UNKNOWN, None, refs)
        elif len(values[name]) > 1:
            facts[name] = _Fact(name, Status.CONFLICT, None, refs)
        else:
            facts[name] = _Fact(name, Status.MET, next(iter(values[name])), refs)
    return facts


def _parse(observation: Observation) -> object:
    """Check an observation's fact name and type, and turn ISO dates into dates."""
    expected = FACT_TYPES.get(observation.fact)
    if expected is None:
        raise ValueError(f"unknown fact {observation.fact!r}")
    value = observation.value
    if expected is date:
        if not isinstance(value, str):
            raise ValueError(f"{observation.fact} must be an ISO date string, got {value!r}")
        return date.fromisoformat(value)
    # bool is a subclass of int, so check the exact type: True is not a number of weeks.
    if type(value) is not expected:
        raise ValueError(f"{observation.fact} must be {expected.__name__}, got {value!r}")
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        raise ValueError(f"{observation.fact} cannot be negative, got {value}")
    return value


def _compare(facts: _Facts, names: tuple[str, ...], test: Callable[..., _Outcome]) -> _Outcome:
    """Run ``test`` on the facts' values if all are known; otherwise say which are not."""
    conflicts = [name for name in names if facts[name].status is Status.CONFLICT]
    if conflicts:
        return Status.CONFLICT, f"documents disagree on {', '.join(conflicts)}"
    missing = [name for name in names if facts[name].status is Status.UNKNOWN]
    if missing:
        return Status.UNKNOWN, f"not documented: {', '.join(missing)}"
    return test(*(facts[name].value for name in names))


def _either(first: _Outcome, second: _Outcome) -> _Outcome:
    """Three-valued OR: met if either is met, not met only if both are, otherwise uncertain."""
    statuses = (first[0], second[0])
    if Status.MET in statuses:
        return first if first[0] is Status.MET else second
    if statuses == (Status.NOT_MET, Status.NOT_MET):
        return Status.NOT_MET, f"{first[1]}; {second[1]}"
    uncertain = Status.CONFLICT if Status.CONFLICT in statuses else Status.UNKNOWN
    return uncertain, "; ".join(reason for status, reason in (first, second) if status is uncertain)


def _outcome(passed: bool, reason: str) -> _Outcome:
    """Turn a boolean test and its explanation into an outcome."""
    return (Status.MET if passed else Status.NOT_MET), reason


def _symptoms(facts: _Facts) -> _Outcome:
    def test(requested: date, onset: date) -> _Outcome:
        weeks = (requested - onset).days // 7
        return _outcome(weeks >= 6, f"symptoms for {weeks} weeks before the request (needs 6)")

    return _compare(facts, ("request_date", "symptom_onset"), test)


def _conservative_care(facts: _Facts) -> _Outcome:
    def flagged(flag: bool) -> _Outcome:
        return _outcome(flag, "red flag documented" if flag else "no red flag")

    def therapy(weeks: int) -> _Outcome:
        return _outcome(weeks >= 6, f"{weeks} weeks of conservative therapy (needs 6)")

    return _either(
        _compare(facts, ("red_flag",), flagged), _compare(facts, ("therapy_weeks",), therapy)
    )


def _recent_xray(facts: _Facts) -> _Outcome:
    def test(requested: date, taken: date) -> _Outcome:
        age = (requested - taken).days
        if age < 0:
            return Status.NOT_MET, "X-ray is dated after the request"
        return _outcome(age <= 90, f"X-ray taken {age} days before the request (needs 90 or fewer)")

    return _compare(facts, ("request_date", "xray_date"), test)


def _referral(facts: _Facts) -> _Outcome:
    def test(present: bool) -> _Outcome:
        return _outcome(present, "referral on file" if present else "no referral")

    return _compare(facts, ("referral",), test)


def _no_recent_mri(facts: _Facts) -> _Outcome:
    def test(requested: date, previous: date) -> _Outcome:
        days = (requested - previous).days
        return _outcome(days > 365, f"prior MRI {days} days before the request (needs over 365)")

    prior = facts["prior_mri"]
    if prior.status is Status.MET:
        if prior.value is False:
            return Status.MET, "no prior MRI of this knee"
        return _compare(facts, ("request_date", "prior_mri_date"), test)
    # Whether there was a prior MRI is itself unknown or disputed; _compare reports which.
    return _compare(facts, ("prior_mri",), test)


CRITERIA: tuple[_Criterion, ...] = (
    _Criterion("C1", ("request_date", "symptom_onset"), Recommendation.DENY, _symptoms),
    _Criterion("C2", ("red_flag", "therapy_weeks"), Recommendation.DENY, _conservative_care),
    _Criterion("C3", ("request_date", "xray_date"), Recommendation.REQUEST_INFO, _recent_xray),
    _Criterion("C4", ("referral",), Recommendation.REQUEST_INFO, _referral),
    _Criterion(
        "X1", ("prior_mri", "prior_mri_date", "request_date"), Recommendation.DENY, _no_recent_mri
    ),
)
"""The criteria in the order results are reported."""

CRITERION_TEXT: Mapping[str, str] = {
    "C1": "Symptoms for at least 6 weeks",
    "C2": "Conservative care for at least 6 weeks, or a red flag",
    "C3": "Knee X-ray within 90 days before the request",
    "C4": "Referral on file",
    "X1": "No MRI of this knee in the last year",
}
"""What each criterion asks, for the dossier and the review screen."""
