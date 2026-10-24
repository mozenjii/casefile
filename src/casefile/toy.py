"""A four-step toy review of one synthetic case, small enough to crash at every point.

The steps are the shape of the real workflow with the intelligence taken out: extract facts with a
(fake) model, check them against fixed criteria, ask for whatever is missing, draft a summary. The
criteria are invented for a made-up procedure and are not clinical guidance. The point of this
module is to give the crash matrix something with model calls, a deterministic decision and an
external effect to break.

``main`` is the entry point the crash matrix runs in a child process.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from casefile.canonical import JsonValue
from casefile.executor import Call, Destination, Runner, StepContext
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.faults import CrashAt, CrashPoint, Faults, HitRecorder
from casefile.log import EventLog
from casefile.state import CaseState

CASE_ID = "SYN-0001"

CASE_INPUTS: dict[str, JsonValue] = {
    "member": "SYN-MEMBER-0001",
    "procedure": "SYN-PROC-A",
    "notes": "Synthetic case. Knee pain for 8 weeks. Supervised therapy for 6 weeks.",
    "documents": ["referral"],
}
"""The one synthetic case. Everything in it is made up."""

REQUIRED_DOCUMENTS = ("imaging_report", "referral")
MIN_THERAPY_WEEKS = 6


def toy_model(request: JsonValue) -> JsonValue:
    """Answer the two prompts the toy workflow sends, deterministically.

    ``extract`` pulls week counts out of the notes; ``draft`` writes a one-line summary.
    """
    task, body = _field(request, "task"), _field(request, "input")
    if task == "extract":
        notes = str(body)
        return {
            "symptom_weeks": _weeks(r"pain for (\d+) weeks", notes),
            "therapy_weeks": _weeks(r"therapy for (\d+) weeks", notes),
        }
    if task == "draft":
        return {"summary": f"Recommendation: {_field(body, 'recommendation')}."}
    raise ValueError(f"toy model has no task {task!r}")


def check_policy(facts: JsonValue, documents: JsonValue) -> dict[str, JsonValue]:
    """Apply the toy criteria: required documents present and enough therapy.

    The recommendation is ``pend`` while anything is missing, else ``approve`` or ``deny``.
    """
    have = {str(item) for item in documents} if isinstance(documents, list) else set[str]()
    missing: list[JsonValue] = [name for name in REQUIRED_DOCUMENTS if name not in have]
    therapy_ok = _int(facts, "therapy_weeks") >= MIN_THERAPY_WEEKS
    recommendation = "pend" if missing else ("approve" if therapy_ok else "deny")
    return {"missing": missing, "therapy_ok": therapy_ok, "recommendation": recommendation}


def run_case(
    runner: Runner, model: Call, destination: Destination, inputs: Mapping[str, JsonValue]
) -> CaseState:
    """Run (or resume) the toy review of one case to completion and return its final state."""
    runner.open(inputs)
    state = runner.state

    def extract(ctx: StepContext) -> JsonValue:
        return ctx.call("model", {"task": "extract", "input": state.inputs["notes"]}, model)

    facts = runner.step("extract/1", extract)
    check = runner.step("policy/1", lambda _: check_policy(facts, runner.state.inputs["documents"]))

    missing = _field(check, "missing")
    if missing:
        request: JsonValue = {"member": state.inputs["member"], "missing": missing}
        runner.step("request-info/1", lambda ctx: ctx.effect("request_info", request, destination))

    def draft(ctx: StepContext) -> JsonValue:
        return ctx.call("model", {"task": "draft", "input": check}, model)

    summary = runner.step("draft/1", draft)
    return runner.complete({"recommendation": _field(check, "recommendation"), "draft": summary})


def main(argv: Sequence[str]) -> int:
    """Run the toy case against the log and ledger files named in ``argv``; return 0 when done.

    ``--crash-at POINT --occurrence N`` ends the process on that hit; ``--hits FILE`` writes every
    crash point passed to ``FILE`` as JSON, which is how the matrix learns what to try.
    """
    parser = argparse.ArgumentParser(prog="casefile.toy")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--crash-at", type=CrashPoint, choices=list(CrashPoint))
    parser.add_argument("--occurrence", type=int, default=1)
    parser.add_argument("--hits", type=Path)
    arguments = parser.parse_args(argv)

    recorder = HitRecorder()
    faults: Faults = recorder
    if arguments.crash_at is not None:
        faults = CrashAt(arguments.crash_at, arguments.occurrence)

    ledger = Ledger(arguments.ledger)
    with EventLog(arguments.log, clock=lambda: datetime.now(UTC)) as log:
        runner = Runner(log, CASE_ID, faults)
        run_case(runner, FakeModel(ledger, toy_model), FakeDestination(ledger), CASE_INPUTS)
    ledger.close()

    if arguments.hits is not None:
        hits = [[str(point), occurrence] for point, occurrence in recorder.hits]
        arguments.hits.write_text(json.dumps(hits), encoding="utf-8")
    return 0


def _field(value: JsonValue, name: str) -> JsonValue:
    """Return ``value[name]`` for a JSON object, or raise naming what was expected."""
    if not isinstance(value, dict) or name not in value:
        raise ValueError(f"expected an object with {name!r}, got {value!r}")
    return value[name]


def _int(value: JsonValue, name: str) -> int:
    """Return an integer field of a JSON object."""
    field = _field(value, name)
    if not isinstance(field, int) or isinstance(field, bool):
        raise ValueError(f"expected an integer {name!r}, got {field!r}")
    return field


def _weeks(pattern: str, notes: str) -> int:
    """Return the first number ``pattern`` captures in ``notes``, or 0 if it does not match."""
    match = re.search(pattern, notes)
    return int(match.group(1)) if match else 0
