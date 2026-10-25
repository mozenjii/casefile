"""Compare three ways of building the workflow on the same cases, with the same model.

More agents is not automatically better, so the design this project ships is measured against two
simpler ones:

* ``single_agent``: one call reads every document and decides. The model is the decision-maker.
* ``fixed_workflow``: one call extracts facts from every document; the rules decide; no writer.
* ``specialists``: the shipped design. One extraction per document with a retry, the grounding
  check and recall sweep, the rules, a request for missing information, and a writer.

Every design runs through the same durable executor, so calls, tokens and cost are counted the same
way, and its result is scored against the generated label. The figures that matter:

* **accuracy**, with a 95% Wilson interval, because a few dozen cases cannot give a point estimate;
* **unsafe approvals**: approving a case whose correct answer was not approve, counted apart;
* **evidence recall**: of the ground-truth facts, how many the model itself cited correctly (the
  recall sweep's additions do not count; they measure the check, not the model);
* **cost per case** and **calls per case**.

With the offline stand-in every design scores perfectly, because the stand-in already knows the
document layout. The numbers mean something only with a real model behind the specialists.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from casefile.agents import SourceDocument, ground, run_review, sweep
from casefile.canonical import JsonValue
from casefile.executor import Call, Runner, StepContext
from casefile.fakes import FakeDestination, Ledger
from casefile.generator import Case
from casefile.log import EventLog
from casefile.money import dollars
from casefile.rules import Observation, evaluate
from casefile.state import CaseState

DESIGNS = ("single_agent", "fixed_workflow", "specialists")
"""The designs compared, simplest first."""


@dataclass(frozen=True)
class Result:
    """One design on one case."""

    design: str
    case_id: str
    label: str
    recommendation: str
    evidence_recall: float
    calls: int
    input_tokens: int
    output_tokens: int
    cost_microusd: int | None
    seconds: float

    @property
    def correct(self) -> bool:
        """The design reached the generated label."""
        return self.recommendation == self.label

    @property
    def unsafe(self) -> bool:
        """The design approved a case whose correct answer was not approve."""
        return self.recommendation == "approve" and self.label != "approve"


def run_design(
    design: str,
    case: Case,
    model: Call,
    log: EventLog,
    ledger: Ledger,
    estimate: Callable[[JsonValue], int | None],
) -> Result:
    """Run one design on one case and score it."""
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")
    documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
    runner = Runner(log, f"{design}/{case.case_id}", estimate=estimate)
    started = time.perf_counter()
    if design == "specialists":
        state = run_review(runner, documents, model, FakeDestination(ledger))
    else:
        state = _single_call(runner, documents, model, decide=design == "single_agent")
    seconds = time.perf_counter() - started

    outcome = state.outcome if isinstance(state.outcome, dict) else {}
    tokens_in, tokens_out = _tokens(state)
    return Result(
        design=design,
        case_id=case.case_id,
        label=case.label.recommendation.value,
        recommendation=str(outcome.get("recommendation", "")),
        evidence_recall=_recall(case, outcome.get("evidence")),
        calls=state.model_calls,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_microusd=None if state.cost_unknown else state.spent_microusd,
        seconds=seconds,
    )


def _single_call(
    runner: Runner, documents: Sequence[SourceDocument], model: Call, *, decide: bool
) -> CaseState:
    """One model call over every document; the model decides, or the rules do."""
    runner.open({"documents": [{"id": d.id, "title": d.title, "text": d.text} for d in documents]})
    task = "decide_case" if decide else "extract_case"
    request: dict[str, JsonValue] = {
        "task": task,
        "documents": [{"id": d.id, "text": d.text} for d in documents],
    }

    def body(ctx: StepContext) -> JsonValue:
        return ctx.call("model", request, model)

    reply = runner.step(f"{task}/1", body)
    observations = _ground_case(documents, reply)
    if decide:
        chosen = reply.get("recommendation") if isinstance(reply, dict) else None
        recommendation = str(chosen) if chosen in ("approve", "request_info", "deny") else "invalid"
    else:
        rows = [o for o in observations if isinstance(o, dict)]
        decision = evaluate(
            Observation(str(o["fact"]), _scalar(o["value"]), str(o["evidence"])) for o in rows
        )
        recommendation = decision.recommendation.value
    evidence: dict[str, JsonValue] = {
        str(o["evidence"]): o for o in observations if isinstance(o, dict)
    }
    dossier: dict[str, JsonValue] = {"recommendation": recommendation, "evidence": evidence}
    return runner.complete(dossier)


def _ground_case(documents: Sequence[SourceDocument], reply: JsonValue) -> list[JsonValue]:
    """Ground a whole-case extraction one document at a time, then sweep for omissions."""
    items = reply.get("observations") if isinstance(reply, dict) else None
    rows = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    grounded: list[JsonValue] = []
    for document in documents:
        mine: list[JsonValue] = [
            {key: value for key, value in row.items() if key != "document"}
            for row in rows
            if row.get("document") == document.id
        ]
        kept, _ = ground(document, {"observations": mine})
        grounded += kept + [
            {**o, "missed": True} for o in sweep(document, kept) if isinstance(o, dict)
        ]
    return grounded


def _recall(case: Case, evidence: JsonValue) -> float:
    """Share of ground-truth facts the model itself cited at exactly the right span."""
    truth = {
        (o.fact, span.document, span.start, span.end)
        for o in case.observations
        for span in case.evidence
        if span.id == o.evidence
    }
    if not truth:
        return 1.0
    cited: set[tuple[object, ...]] = set()
    if isinstance(evidence, dict):
        for value in evidence.values():
            if isinstance(value, dict) and not value.get("missed"):
                cited.add(
                    (value.get("fact"), value.get("document"), value.get("start"), value.get("end"))
                )
    return len(truth & cited) / len(truth)


def _tokens(state: CaseState) -> tuple[int, int]:
    """Input and output tokens the case's recorded calls report."""
    total_in = total_out = 0
    for response in state.calls.values():
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            total_in += _int(usage.get("input_tokens"))
            total_out += _int(usage.get("output_tokens"))
    return total_in, total_out


def _int(value: object) -> int:
    """A token count, or zero when the provider did not report one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _scalar(value: JsonValue) -> str | int | bool:
    """Narrow a recorded value to what the rules read."""
    if isinstance(value, str | int | bool):
        return value
    raise ValueError(f"unexpected value {value!r}")


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a proportion: honest at small samples and near 0 or 1."""
    if trials == 0:
        return 0.0, 1.0
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    spread = z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials))
    spread /= 1 + z * z / trials
    return max(0.0, centre - spread), min(1.0, centre + spread)


def summarise(results: Sequence[Result]) -> str:
    """Render one Markdown row per design."""
    rows = [
        "| design | cases | accuracy (95% interval) | unsafe approvals | evidence recall "
        "| calls per case | cost per case |",
        "|---|---|---|---|---|---|---|",
    ]
    for design in DESIGNS:
        mine = [r for r in results if r.design == design]
        if not mine:
            continue
        right = sum(r.correct for r in mine)
        low, high = wilson(right, len(mine))
        costs = [r.cost_microusd for r in mine]
        per_case = None if None in costs else round(sum(c or 0 for c in costs) / len(mine))
        recall = sum(r.evidence_recall for r in mine) / len(mine)
        calls = sum(r.calls for r in mine) / len(mine)
        rows.append(
            f"| {design} | {len(mine)} | {right / len(mine):.0%} ({low:.0%} to {high:.0%}) "
            f"| {sum(r.unsafe for r in mine)} | {recall:.0%} | {calls:.1f} | {dollars(per_case)} |"
        )
    return "\n".join(rows)
