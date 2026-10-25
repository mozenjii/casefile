"""A deterministic stand-in for the language model, with the same JSON interface.

It answers the two prompts the specialists send:

* ``extract``: read one document and return observations, each with the character span it read;
* ``write``: turn the policy result into a short summary and one claim per criterion, each citing
  the evidence ids it rests on.

It reads documents by their wording, never by their ids, and it never follows instructions found in
a document, so a planted "approve this" changes nothing. A real model behind the same interface is
judged by the same grounding check; this one exists so the workflow, its durability and its review
screen can be built and tested without a model on the machine.
"""

from __future__ import annotations

from casefile.canonical import JsonValue
from casefile.quotes import fields, read_quote
from casefile.rules import FACT_TYPES, Observation, evaluate


def reference_model(request: JsonValue) -> JsonValue:
    """Answer an ``extract`` or ``write`` request."""
    if not isinstance(request, dict):
        raise ValueError("a model request is a JSON object")
    task = request.get("task")
    if task == "extract":
        return {"observations": _extract(str(request.get("text", "")))}
    if task == "write":
        return _write(request)
    if task == "extract_case":
        return {"observations": _extract_case(request)}
    if task == "decide_case":
        return _decide(request)
    raise ValueError(f"the reference model has no task {task!r}")


def _extract(text: str) -> list[JsonValue]:
    """Read every labelled field in the document, with the span of its value."""
    found: list[JsonValue] = []
    for fact in FACT_TYPES:
        for start, end in fields(fact, text):
            value = read_quote(fact, text[start:end])
            if value is not None:
                found.append({"fact": fact, "value": value, "start": start, "end": end})
    return found


def _extract_case(request: dict[str, JsonValue]) -> list[JsonValue]:
    """Extract every document of a case, tagging each fact with its document id."""
    found: list[JsonValue] = []
    documents = request.get("documents")
    for document in documents if isinstance(documents, list) else []:
        if not isinstance(document, dict):
            continue
        for item in _extract(str(document.get("text", ""))):
            if isinstance(item, dict):
                found.append({"document": document.get("id"), **item})
    return found


def _decide(request: dict[str, JsonValue]) -> JsonValue:
    """Decide a case the way the rules would; a real model has to reason its way there."""
    observations = _extract_case(request)
    decision = evaluate(
        Observation(str(item["fact"]), _typed(item["value"]), f"{item['document']}:{item['fact']}")
        for item in observations
        if isinstance(item, dict)
    )
    reasons: list[JsonValue] = [f"{r.id}: {r.reason}" for r in decision.results]
    return {
        "recommendation": decision.recommendation.value,
        "reasons": reasons,
        "observations": observations,
    }


def _typed(value: JsonValue) -> str | int | bool:
    """Narrow an extracted value to what the rules read."""
    if isinstance(value, str | int | bool):
        return value
    raise ValueError(f"unexpected value {value!r}")


def _write(request: dict[str, JsonValue]) -> JsonValue:
    """Summarise the policy result and state each criterion's outcome as a cited claim."""
    criteria = request.get("criteria")
    rows = [row for row in criteria if isinstance(row, dict)] if isinstance(criteria, list) else []
    met = sum(1 for row in rows if row.get("status") == "met")
    recommendation = str(request.get("recommendation", "")).replace("_", " ")
    summary = f"Recommendation: {recommendation}. {met} of {len(rows)} criteria met."
    requests = request.get("requests")
    if isinstance(requests, list) and requests:
        summary += f" Requesting: {'; '.join(str(item) for item in requests)}."
    claims: list[JsonValue] = [
        {"text": f"{row.get('text')}: {row.get('reason')}.", "evidence": row.get("evidence", [])}
        for row in rows
    ]
    return {"summary": summary, "claims": claims}
