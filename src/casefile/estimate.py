"""Estimate what running cases through a real model would cost, without calling one.

The estimate runs the real workflow offline with the deterministic stand-in, captures every request
the specialists would send, builds the exact prompts a real provider would receive, and counts them.
Output size is taken from the stand-in's answers rewritten in the quote form a language model
returns. Characters are turned into tokens at about four per token, the usual rule of thumb for
English text; that is an estimate, and the report says so.

What it cannot see: hidden reasoning. A model that thinks before answering (Claude Opus 5.5 always
does) bills those thinking tokens as output, so its real cost is higher than this figure.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from casefile.agents import SourceDocument, run_review
from casefile.canonical import JsonValue
from casefile.executor import Runner
from casefile.fakes import FakeDestination, Ledger
from casefile.generator import Case
from casefile.llm import Settings
from casefile.log import EventLog
from casefile.money import dollars
from casefile.prompts import PROMPTS
from casefile.reference_model import reference_model

__all__ = ["CHARS_PER_TOKEN", "Estimate", "dollars", "estimate_cases", "estimator"]

CHARS_PER_TOKEN = 4
"""Rule-of-thumb characters per token for English text."""


@dataclass(frozen=True)
class Estimate:
    """Estimated model traffic for some number of cases."""

    cases: int
    calls: int
    input_tokens: int
    output_tokens: int

    def cost_microusd(self, settings: Settings) -> int | None:
        """The estimated cost in micro-dollars at ``settings``' prices, or None without prices."""
        return settings.cost(self.input_tokens, self.output_tokens)

    def __add__(self, other: Estimate) -> Estimate:
        """Combine two estimates."""
        return Estimate(
            self.cases + other.cases,
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


def estimate_cases(cases: list[Case]) -> Estimate:
    """Run each case offline and count the tokens a real model would see and write."""
    total = Estimate(0, 0, 0, 0)
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        ledger = Ledger(directory / "world.sqlite")
        with EventLog(directory / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
            for case in cases:
                recorder = _Recorder()
                documents = [SourceDocument(d.id, d.title, d.text) for d in case.documents]
                run_review(Runner(log, case.case_id), documents, recorder, FakeDestination(ledger))
                total = total + recorder.estimate()
        ledger.close()
    return total


class _Recorder:
    """Answers like the stand-in while measuring the prompts a real provider would receive."""

    def __init__(self) -> None:
        """Start with nothing measured."""
        self.calls = 0
        self.input_chars = 0
        self.output_chars = 0

    def __call__(self, request: JsonValue) -> JsonValue:
        """Answer the request and add its prompt and answer sizes."""
        if not isinstance(request, dict):
            raise ValueError("a model request is a JSON object")
        build = PROMPTS[str(request.get("task"))]
        system, user, schema = build(request)
        answer = reference_model(request)
        self.calls += 1
        self.input_chars += len(system) + len(user) + len(json.dumps(schema))
        self.output_chars += len(json.dumps(_as_quotes(request, answer)))
        return answer

    def estimate(self) -> Estimate:
        """What this case's calls add up to, in tokens."""
        return Estimate(
            1,
            self.calls,
            -(-self.input_chars // CHARS_PER_TOKEN),
            -(-self.output_chars // CHARS_PER_TOKEN),
        )


def _as_quotes(request: dict[str, JsonValue], answer: JsonValue) -> JsonValue:
    """Rewrite an offset-based extraction the way a language model answers: values and quotes."""
    if request.get("task") != "extract" or not isinstance(answer, dict):
        return answer
    text = str(request.get("text", ""))
    items = answer.get("observations")
    rows = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    quoted: list[JsonValue] = [
        {
            "fact": row["fact"],
            "value": str(row["value"]),
            "quote": text[int(str(row["start"])) : int(str(row["end"]))],
        }
        for row in rows
    ]
    return {"observations": quoted}


ASSUMED_OUTPUT_TOKENS = 400
"""Output a single call is assumed to produce when pricing it before it is made."""


def estimator(settings: Settings) -> Callable[[JsonValue], int | None]:
    """Price one model request before it is sent, for budgets and the write-ahead of each call.

    Input tokens come from the exact prompt the provider would receive; output is assumed at
    ``ASSUMED_OUTPUT_TOKENS``. None when the provider has no price, which a cost limit treats as a
    reason to stop rather than guess.
    """

    def price(request: JsonValue) -> int | None:
        if not isinstance(request, dict):
            return None
        build = PROMPTS.get(str(request.get("task")))
        if build is None:
            return None
        system, user, schema = build(request)
        chars = len(system) + len(user) + len(json.dumps(schema))
        return settings.cost(-(-chars // CHARS_PER_TOKEN), ASSUMED_OUTPUT_TOKENS)

    return price
