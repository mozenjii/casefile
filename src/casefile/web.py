"""The review screen: one case's dossier, its evidence in the source documents, and the decision.

Server-rendered HTML with no build step and no client framework. Every evidence span is marked in
the document text, and every criterion and claim links to the spans it rests on, so the links work
without JavaScript; a few lines of script add keyboard shortcuts and a highlight on the cited span.

The decision form posts back the dossier hash the page was rendered with. The service records the
decision against that hash, so if the dossier changed while the reviewer was reading, their click
is kept on record but authorizes nothing, and the page says so.

This is a local tool over synthetic data. It has no login: the reviewer types a name, which is
recorded as ``human:<name>``. Real users would need authentication before anything else.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from casefile.canonical import JsonValue
from casefile.cases import load
from casefile.events import Event
from casefile.log import AppendConflict, EventLog
from casefile.money import dollars
from casefile.review import NotReviewable, Review, ReviewView
from casefile.state import DECISIONS, FoldError

# Markup.format escapes every argument, so document text can never inject markup.
_MARK = Markup('<mark id="{}" tabindex="-1" title="{}">{}</mark>')
_TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,39}")


def create_app(log_path: Path, clock: Callable[[], datetime] | None = None) -> FastAPI:
    """Build the review app over the event log at ``log_path``."""
    app = FastAPI(title="casefile review", docs_url=None, redoc_url=None, openapi_url=None)
    now = clock or (lambda: datetime.now(UTC))

    def open_log() -> EventLog:
        return EventLog(log_path, clock=now)

    @app.get("/", response_class=HTMLResponse)
    def cases(request: Request) -> HTMLResponse:
        with open_log() as log:
            rows = [_summary(log, case_id) for case_id in log.case_ids()]
        return _TEMPLATES.TemplateResponse(request, "cases.html", {"cases": rows})

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case(request: Request, case_id: str, notice: str = "") -> HTMLResponse:
        return _page(request, open_log, case_id, notice=notice)

    @app.post("/cases/{case_id}/decision")
    def decide(
        request: Request,
        case_id: str,
        decision: Annotated[str, Form()],
        seen_hash: Annotated[str, Form()],
        reviewer: Annotated[str, Form()] = "",
    ) -> Response:
        name = reviewer.strip().lower()
        if decision not in DECISIONS or not _NAME.fullmatch(name):
            return _page(request, open_log, case_id, error=_INVALID, status=400)
        with open_log() as log:
            try:
                view = Review(log).decide(
                    case_id, decision, reviewer=f"human:{name}", seen_hash=seen_hash
                )
            except NotReviewable as error:
                raise HTTPException(404, str(error)) from error
            except AppendConflict:
                return _page(request, open_log, case_id, error=_RACE, status=409)
        stale = decision == "approve" and not view.approved
        notice = "stale" if stale else f"recorded-{decision}"
        return RedirectResponse(f"/cases/{case_id}?notice={notice}", status_code=303)

    @app.post("/cases/{case_id}/release")
    def release(request: Request, case_id: str) -> Response:
        with open_log() as log:
            try:
                Review(log).release(case_id)
            except NotReviewable as error:
                raise HTTPException(404, str(error)) from error
            except FoldError:
                return _page(request, open_log, case_id, error=_UNAPPROVED, status=409)
        return RedirectResponse(f"/cases/{case_id}?notice=released", status_code=303)

    return app


_EMPTY_DOSSIER: dict[str, JsonValue] = {
    "recommendation": "",
    "summary": "",
    "criteria": [],
    "claims": [],
    "requests": [],
    "evidence": {},
    "rejected": {"observations": [], "claims": []},
}

_INVALID = "Choose a decision and enter your reviewer name (letters, digits, dots or dashes)."
_RACE = "Someone else acted on this case at the same moment. Reload and try again."
_UNAPPROVED = "This dossier cannot be released: its current version has no approval."
_NOTICES = {
    "stale": "Your approval was recorded, but the dossier changed after you loaded it, so it "
    "does not count. Review the current version.",
    "recorded-approve": "Approval recorded for this exact version of the dossier.",
    "recorded-reject": "Rejection recorded.",
    "recorded-request_info": "Request for more information recorded.",
    "released": "Released.",
}


def _page(
    request: Request,
    open_log: Callable[[], EventLog],
    case_id: str,
    *,
    notice: str = "",
    error: str = "",
    status: int = 200,
) -> HTMLResponse:
    """Render one case's review page."""
    with open_log() as log:
        try:
            view = Review(log).show(case_id)
        except NotReviewable as missing:
            raise HTTPException(404, str(missing)) from missing
        events = log.read(case_id)
    state = _inputs(events)
    # A human revision may replace the dossier with a partial one; show what is there.
    dossier = {**_EMPTY_DOSSIER, **_object(view.dossier)}
    evidence = _object(dossier.get("evidence"))
    documents = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "html": _mark(str(doc["text"]), str(doc["id"]), evidence),
        }
        for doc in _documents(state)
    ]
    context: dict[str, object] = {
        "view": view,
        "dossier": dossier,
        "documents": documents,
        "timeline": [_moment(event) for event in events],
        "spend": _spend(events),
        "anchor": anchor,
        "notice": _NOTICES.get(notice, ""),
        "error": error,
        "status": _status(view),
    }
    return _TEMPLATES.TemplateResponse(request, "review.html", context, status_code=status)


def _inputs(events: list[Event]) -> dict[str, JsonValue]:
    """Return the case's opening inputs (the documents) from its first event."""
    inputs = events[0].payload.get("inputs") if events else None
    return _object(inputs)


def anchor(evidence_id: str) -> str:
    """Turn an evidence id into an HTML id: ``note:therapy_weeks`` → ``ev-note-therapy_weeks``."""
    return "ev-" + re.sub(r"[^A-Za-z0-9_-]", "-", evidence_id)


def _mark(text: str, document_id: str, evidence: dict[str, JsonValue]) -> Markup:
    """Escape a document's text and wrap each of its evidence spans in a linked ``<mark>``."""
    spans = sorted(
        (_number(item, "start"), _number(item, "end"), key)
        for key, raw in evidence.items()
        if (item := _object(raw)).get("document") == document_id
    )
    pieces: list[Markup] = []
    cursor = 0
    for start, end, key in spans:
        if start < cursor:
            continue  # overlapping spans are not produced; never render one inside another
        pieces.append(escape(text[cursor:start]))
        pieces.append(_MARK.format(anchor(key), key, text[start:end]))
        cursor = end
    pieces.append(escape(text[cursor:]))
    return Markup("").join(pieces)


def _summary(log: EventLog, case_id: str) -> dict[str, object]:
    """One row of the case list."""
    state = load(log, case_id)
    if state is None or not state.completed:
        return {"id": case_id, "recommendation": "", "status": "running"}
    outcome = _object(state.outcome)
    stage = "released" if state.released else ("approved" if state.approved else "awaiting review")
    return {"id": case_id, "recommendation": outcome.get("recommendation", ""), "status": stage}


def _status(view: ReviewView) -> str:
    """Where the review stands, in words."""
    if view.released:
        return "Released"
    if view.approved:
        return "Approved, ready to release"
    return "Awaiting review"


def _usage(event: Event) -> dict[str, JsonValue] | None:
    """The token usage a recorded model call carries, if the model reported any."""
    response = _object(event.payload.get("response"))
    usage = response.get("usage")
    return _object(usage) if isinstance(usage, dict) else None


def _spend(events: list[Event]) -> str:
    """What the case's model calls cost, in words: a total, unknown, or none (the stand-in)."""
    calls = [event for event in events if event.type == "CallRecorded"]
    usages = [usage for event in calls if (usage := _usage(event)) is not None]
    if not usages:
        return f"{len(calls)} model calls, answered by the offline stand-in at no cost"
    costs = [usage.get("cost_microusd") for usage in usages]
    if any(not isinstance(cost, int) or isinstance(cost, bool) for cost in costs):
        return f"{len(calls)} model calls, cost unknown (no price configured)"
    total = sum(cost for cost in costs if isinstance(cost, int))
    return f"{len(calls)} model calls, {dollars(total)} in total"


def _call_detail(event: Event) -> str:
    """Describe one recorded model call, with its tokens and cost when known."""
    usage = _usage(event)
    if usage is None:
        return f"{event.payload.get('kind')} call recorded"
    cost = usage.get("cost_microusd")
    priced = (
        dollars(cost) if isinstance(cost, int) and not isinstance(cost, bool) else "cost unknown"
    )
    tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    return f"{usage.get('model')} · {tokens_in} in / {tokens_out} out · {priced}"


def _moment(event: Event) -> dict[str, str]:
    """One line of the timeline."""
    payload = event.payload
    detail = {
        "CallRecorded": _call_detail(event),
        "EffectIntended": f"{payload.get('kind')} intended",
        "EffectConfirmed": "delivery confirmed",
        "ReviewerDecision": f"{payload.get('decision')} by {payload.get('reviewer')}",
        "DossierRevised": f"revised by {payload.get('editor')}",
    }.get(event.type, "")
    return {
        "seq": str(event.seq),
        "time": event.recorded_at[11:19],
        "type": event.type,
        "step": event.step_id or "",
        "detail": detail,
    }


def _documents(inputs: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """The case's documents as recorded when it was opened."""
    documents = inputs.get("documents")
    return [_object(doc) for doc in documents] if isinstance(documents, list) else []


def _object(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a recorded value to a JSON object, or an empty one."""
    return cast(dict[str, JsonValue], value) if isinstance(value, dict) else {}


def _number(item: dict[str, JsonValue], name: str) -> int:
    """Read an integer field of a ledger entry."""
    value = item.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
