"""Tests for the toy workflow: its model, its criteria, and a full run in process."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from casefile.canonical import JsonValue
from casefile.executor import Runner
from casefile.fakes import FakeDestination, FakeModel, Ledger
from casefile.log import EventLog
from casefile.toy import CASE_ID, CASE_INPUTS, check_policy, main, run_case, toy_model


def test_the_model_extracts_week_counts_from_the_notes() -> None:
    notes = "Knee pain for 12 weeks. Supervised therapy for 3 weeks."
    assert toy_model({"task": "extract", "input": notes}) == {
        "symptom_weeks": 12,
        "therapy_weeks": 3,
    }
    assert toy_model({"task": "extract", "input": "nothing useful"}) == {
        "symptom_weeks": 0,
        "therapy_weeks": 0,
    }


def test_the_model_drafts_from_the_recommendation() -> None:
    draft = toy_model({"task": "draft", "input": {"recommendation": "deny"}})
    assert draft == {"summary": "Recommendation: deny."}


@pytest.mark.parametrize(
    "request_",
    [{"task": "poem", "input": ""}, {"input": "no task"}, "not an object"],
)
def test_the_model_refuses_what_it_was_not_built_for(request_: JsonValue) -> None:
    with pytest.raises(ValueError, match=r"task|expected an object"):
        toy_model(request_)


@pytest.mark.parametrize(
    ("weeks", "documents", "recommendation", "missing"),
    [
        (6, ["referral", "imaging_report"], "approve", []),
        (5, ["referral", "imaging_report"], "deny", []),
        (6, ["referral"], "pend", ["imaging_report"]),
        (9, "not a list", "pend", ["imaging_report", "referral"]),
    ],
)
def test_the_criteria(
    weeks: int, documents: JsonValue, recommendation: str, missing: list[str]
) -> None:
    result = check_policy({"therapy_weeks": weeks}, documents)
    assert result["recommendation"] == recommendation
    assert result["missing"] == missing


def test_the_criteria_refuse_a_non_integer_week_count() -> None:
    with pytest.raises(ValueError, match="integer"):
        check_policy({"therapy_weeks": True}, ["referral"])


def test_the_synthetic_case_runs_to_a_pended_recommendation(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "world.sqlite")
    with EventLog(tmp_path / "events.sqlite", clock=lambda: datetime.now(UTC)) as log:
        runner = Runner(log, CASE_ID)
        state = run_case(runner, FakeModel(ledger, toy_model), FakeDestination(ledger), CASE_INPUTS)
    assert state.completed
    assert state.outcome == {
        "recommendation": "pend",
        "draft": {"summary": "Recommendation: pend."},
    }
    assert ledger.model_calls() == 2
    assert len(ledger.applied()) == 1
    ledger.close()


def test_main_records_its_crash_points_and_is_idempotent(tmp_path: Path) -> None:
    files = ["--log", str(tmp_path / "e.sqlite"), "--ledger", str(tmp_path / "w.sqlite")]
    assert main([*files, "--hits", str(tmp_path / "hits.json")]) == 0
    hits = json.loads((tmp_path / "hits.json").read_text("utf-8"))
    assert len(hits) == 17
    assert hits[0] == ["before_step", 1]
    assert hits[-1] == ["before_case_completed", 1]
    assert main(files) == 0
    ledger = Ledger(tmp_path / "w.sqlite")
    assert ledger.model_calls() == 2
    ledger.close()
