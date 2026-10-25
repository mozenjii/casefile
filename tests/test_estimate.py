"""Tests for the offline cost estimate."""

from __future__ import annotations

from casefile.estimate import Estimate, dollars, estimate_cases
from casefile.generator import generate, generate_set
from casefile.llm import settings_from_env


def test_each_case_is_one_call_per_document_plus_the_draft() -> None:
    cases = [generate(1, "none"), generate(1, "missing_xray")]
    result = estimate_cases(cases)
    assert result.cases == 2
    assert result.calls == (4 + 1) + (3 + 1)
    assert result.input_tokens > 1000
    assert 0 < result.output_tokens < result.input_tokens


def test_the_estimate_is_priced_with_the_run_settings() -> None:
    result = Estimate(cases=1, calls=5, input_tokens=2_000, output_tokens=300)
    openrouter = settings_from_env({"CASEFILE_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "x"})
    assert result.cost_microusd(openrouter) == round(2_000 * 0.209 + 300 * 0.418)
    assert result.cost_microusd(settings_from_env({})) == 0  # the offline stand-in is free


def test_estimates_add_up() -> None:
    one = estimate_cases(generate_set(2))
    assert one + one == Estimate(4, one.calls * 2, one.input_tokens * 2, one.output_tokens * 2)


def test_money_is_written_for_people() -> None:
    assert dollars(627) == "$0.0006"
    assert dollars(2_420_000) == "$2.42"
    assert dollars(12_400_000_000) == "$12,400.00"
    assert dollars(None) == "unknown"
