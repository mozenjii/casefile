"""The supervisor and its specialists: from documents to a dossier a person can check.

Each specialist does one job and runs as a durable step, so a crash anywhere resumes cleanly:

1. **Extractor**, once per document: the model returns facts, each with the character span it read.
   The **grounding check** re-reads every span with ``read_quote`` and keeps an observation only if
   those words state the value the model reported. Anything else is rejected with a reason. If a
   document had rejections, the supervisor asks once more, telling the model what was wrong.
2. **Policy checker**: the rules engine, deterministic. The recommendation comes from here, never
   from the model.
3. **Gap finder**: every criterion that is unknown, in conflict, or failed with "ask for more"
   becomes a request. If the recommendation is to ask, one request-for-information effect goes out
   through the outbox, once.
4. **Dossier writer**: the model drafts a summary and claims; a claim citing evidence that does not
   exist is dropped and counted.

The automated run ends with the dossier, which is what a reviewer sees and approves by hash. Model
output is untrusted input: it is parsed with strict schemas, and nothing it says can change the
recommendation or skip the review.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from casefile.canonical import JsonValue
from casefile.executor import Call, Destination, Runner, StepContext
from casefile.quotes import read_in_context
from casefile.rules import (
    CRITERION_TEXT,
    FACT_TYPES,
    IF_NOT_MET,
    Observation,
    Recommendation,
    Status,
    evaluate,
)
from casefile.state import CaseState


class _Strict(BaseModel):
    """Model output is parsed strictly: no coercion of "6" into 6 or "true" into True."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class _Fact(_Strict):
    fact: str
    value: str | int | bool
    start: int
    end: int


class _Extraction(_Strict):
    observations: list[_Fact]


class _Claim(_Strict):
    text: str
    evidence: list[str]


class _Draft(_Strict):
    summary: str
    claims: list[_Claim]


@dataclass(frozen=True)
class SourceDocument:
    """A document handed to the workflow: an id, a title and its text."""

    id: str
    title: str
    text: str


def ground(document: SourceDocument, response: JsonValue) -> tuple[list[JsonValue], list[str]]:
    """Split a model's extraction into grounded observations and reasons for rejecting the rest.

    A grounded observation carries an evidence id (``document:fact``, numbered if a fact appears
    twice), its span and the quoted words, so the reviewer can see exactly what it rests on.
    """
    try:
        extraction = _Extraction.model_validate(response)
    except ValidationError as error:
        return [], [f"{document.id}: malformed response ({error.error_count()} problem(s))"]

    kept: list[JsonValue] = []
    rejected: list[str] = []
    used: set[str] = set()
    for item in extraction.observations:
        where = f"{document.id} {item.fact} [{item.start}:{item.end}]"
        if item.fact not in FACT_TYPES:
            rejected.append(f"{where}: unknown fact")
            continue
        if not 0 <= item.start < item.end <= len(document.text):
            rejected.append(f"{where}: span is outside the document")
            continue
        quote = document.text[item.start : item.end]
        stated = read_in_context(item.fact, document.text, item.start, item.end)
        if stated is None or stated != item.value or type(stated) is not type(item.value):
            rejected.append(f"{where}: the span says {quote!r}, not {item.value!r}")
            continue
        evidence = _fresh_id(f"{document.id}:{item.fact}", used)
        kept.append(
            {
                "fact": item.fact,
                "value": item.value,
                "evidence": evidence,
                "document": document.id,
                "start": item.start,
                "end": item.end,
                "quote": quote,
            }
        )
    return kept, rejected


def run_review(
    runner: Runner,
    documents: Sequence[SourceDocument],
    model: Call,
    destination: Destination,
) -> CaseState:
    """Run (or resume) the automated review of one case to its dossier and return the state."""
    runner.open({"documents": [{"id": d.id, "title": d.title, "text": d.text} for d in documents]})

    extracted: list[JsonValue] = []
    rejected: list[JsonValue] = []
    for document in documents:
        output = runner.step(f"extract/{document.id}", _extractor(document, model))
        extracted += _list(output, "observations")
        rejected += _list(output, "rejected")

    policy = runner.step("policy/1", lambda _: _policy(extracted))
    gaps = runner.step("gaps/1", lambda ctx: _gaps(ctx, runner.state.case_id, policy, destination))
    draft = runner.step("draft/1", lambda ctx: _writer(ctx, policy, gaps, extracted, model))

    ledger: dict[str, JsonValue] = {}
    for item in extracted:
        if isinstance(item, dict):
            ledger[str(item["evidence"])] = {
                key: item[key] for key in ("fact", "value", "document", "start", "end", "quote")
            }
    dossier: dict[str, JsonValue] = {
        "recommendation": _field(policy, "recommendation"),
        "summary": _field(draft, "summary"),
        "criteria": _field(policy, "criteria"),
        "claims": _field(draft, "claims"),
        "requests": _field(gaps, "requests"),
        "evidence": ledger,
        "rejected": {"observations": rejected, "claims": _field(draft, "rejected")},
    }
    return runner.complete(dossier)


def _extractor(document: SourceDocument, model: Call) -> Callable[[StepContext], JsonValue]:
    """Build the step body that extracts one document, retrying once after a rejection."""

    def extract(ctx: StepContext) -> JsonValue:
        request: dict[str, JsonValue] = {
            "task": "extract",
            "document": document.id,
            "text": document.text,
        }
        kept, rejected = ground(document, ctx.call("model", request, model))
        if rejected:
            # One retry, told what was wrong. Both attempts are recorded, so replay is exact.
            retry: dict[str, JsonValue] = {**request, "rejected": list[JsonValue](rejected)}
            kept, second = ground(document, ctx.call("model", retry, model))
            rejected = [*rejected, *second]
        return {"observations": kept, "rejected": list[JsonValue](rejected)}

    return extract


def _policy(extracted: list[JsonValue]) -> JsonValue:
    """Evaluate the criteria over the grounded observations."""
    observations = [
        Observation(str(item["fact"]), _scalar(item["value"]), str(item["evidence"]))
        for item in extracted
        if isinstance(item, dict)
    ]
    decision = evaluate(observations)
    criteria: list[JsonValue] = [
        {
            "id": result.id,
            "text": CRITERION_TEXT[result.id],
            "status": result.status.value,
            "reason": result.reason,
            "evidence": list[JsonValue](result.evidence),
        }
        for result in decision.results
    ]
    return {"recommendation": decision.recommendation.value, "criteria": criteria}


def _gaps(ctx: StepContext, case_id: str, policy: JsonValue, destination: Destination) -> JsonValue:
    """List what is missing or disputed, and ask for it once if the recommendation is to ask."""
    requests: list[JsonValue] = []
    for row in _list(policy, "criteria"):
        status = _field(row, "status")
        if status in (Status.UNKNOWN, Status.CONFLICT) or (
            status == Status.NOT_MET
            and IF_NOT_MET[str(_field(row, "id"))] is Recommendation.REQUEST_INFO
        ):
            requests.append(f"{_field(row, 'text')} ({_field(row, 'reason')})")
    receipt: JsonValue = None
    if _field(policy, "recommendation") == Recommendation.REQUEST_INFO:
        receipt = ctx.effect("request_info", {"case": case_id, "requests": requests}, destination)
    return {"requests": requests, "receipt": receipt}


def _writer(
    ctx: StepContext,
    policy: JsonValue,
    gaps: JsonValue,
    extracted: list[JsonValue],
    model: Call,
) -> JsonValue:
    """Draft the summary and claims, dropping any claim that cites evidence that does not exist."""
    request: dict[str, JsonValue] = {
        "task": "write",
        "recommendation": _field(policy, "recommendation"),
        "criteria": _field(policy, "criteria"),
        "requests": _field(gaps, "requests"),
    }
    try:
        draft = _Draft.model_validate(ctx.call("model", request, model))
    except ValidationError as error:
        return {
            "summary": "",
            "claims": [],
            "rejected": [f"malformed draft: {error.error_count()}"],
        }
    known = {str(item["evidence"]) for item in extracted if isinstance(item, dict)}
    claims: list[JsonValue] = []
    rejected: list[JsonValue] = []
    for claim in draft.claims:
        unknown = sorted(set(claim.evidence) - known)
        if unknown:
            rejected.append(f"claim cites evidence that does not exist: {', '.join(unknown)}")
        else:
            claims.append({"text": claim.text, "evidence": list[JsonValue](claim.evidence)})
    return {"summary": draft.summary, "claims": claims, "rejected": rejected}


def _fresh_id(base: str, used: set[str]) -> str:
    """Return ``base``, or ``base#2``, ``base#3`` … if it is taken, and mark it used."""
    candidate, count = base, 1
    while candidate in used:
        count += 1
        candidate = f"{base}#{count}"
    used.add(candidate)
    return candidate


def _scalar(value: JsonValue) -> str | int | bool:
    """Narrow a recorded observation value to the types the rules engine reads."""
    if isinstance(value, str | int | bool):
        return value
    raise ValueError(f"an observation value must be a string, number or boolean, got {value!r}")


def _field(value: JsonValue, name: str) -> JsonValue:
    """Return a field of a recorded JSON object."""
    if not isinstance(value, dict) or name not in value:
        raise ValueError(f"expected an object with {name!r}")
    return value[name]


def _list(value: JsonValue, name: str) -> list[JsonValue]:
    """Return a list field of a recorded JSON object."""
    field = _field(value, name)
    if not isinstance(field, list):
        raise ValueError(f"expected {name!r} to be a list")
    return field
