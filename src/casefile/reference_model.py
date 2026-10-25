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

import re

from casefile.canonical import JsonValue
from casefile.quotes import read_quote

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("request_date", re.compile(r"Request date: (?P<v>\d{4}-\d{2}-\d{2})")),
    ("prior_mri", re.compile(r"Prior MRI of this knee: (?P<v>Yes|No)")),
    ("prior_mri_date", re.compile(r"Prior MRI of this knee: Yes, on (?P<v>\d{4}-\d{2}-\d{2})")),
    ("therapy_weeks", re.compile(r"Weeks of conservative therapy: (?P<v>\d+)")),
    ("symptom_onset", re.compile(r"pain since (?P<v>\d{1,2} [A-Z][a-z]+ \d{4})")),
    ("therapy_weeks", re.compile(r"completed (?P<v>\d+ weeks)")),
    ("red_flag", re.compile(r"Red flags: (?P<v>[^.\n]+)")),
    ("xray_date", re.compile(r"Examination: X-ray[^\n]*\nDate: (?P<v>\d{1,2} [A-Z][a-z]+ \d{4})")),
    ("referral", re.compile(r"(?P<v>Referred by)")),
)


def reference_model(request: JsonValue) -> JsonValue:
    """Answer an ``extract`` or ``write`` request."""
    if not isinstance(request, dict):
        raise ValueError("a model request is a JSON object")
    task = request.get("task")
    if task == "extract":
        return {"observations": _extract(str(request.get("text", "")))}
    if task == "write":
        return _write(request)
    raise ValueError(f"the reference model has no task {task!r}")


def _extract(text: str) -> list[JsonValue]:
    """Find every fact the patterns know, with the span of its value."""
    found: list[JsonValue] = []
    for fact, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span("v")
            value = read_quote(fact, text[start:end])
            if value is not None:
                found.append({"fact": fact, "value": value, "start": start, "end": end})
    return found


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
