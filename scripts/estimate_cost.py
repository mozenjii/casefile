"""Estimate what reviewing generated cases with the configured model would cost. Calls no model.

Usage, from the repository root::

    uv run python scripts/estimate_cost.py                      # the configured provider
    CASEFILE_PROVIDER=openrouter OPENROUTER_API_KEY=x uv run python scripts/estimate_cost.py
    uv run python scripts/estimate_cost.py --cases 300 --designs 3

The provider settings are read exactly as a real run reads them, so the prices are the ones that
run would use. A key is only needed because the settings require one; nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import sys

from casefile.estimate import dollars, estimate_cases
from casefile.generator import generate_set
from casefile.llm import ConfigError, describe, settings_from_env


def main() -> int:
    """Print the estimate and return the exit code."""
    parser = argparse.ArgumentParser(description="Estimate model cost for generated cases.")
    parser.add_argument("--cases", type=int, default=150, help="cases per design (default 150)")
    parser.add_argument("--designs", type=int, default=1, help="workflow designs to run each case")
    parser.add_argument("--sample", type=int, default=18, help="cases to measure (default 18)")
    arguments = parser.parse_args()
    try:
        settings = settings_from_env()
    except ConfigError as error:
        print(f"provider settings: {error}", file=sys.stderr)
        return 2

    sample = estimate_cases(generate_set(arguments.sample))
    runs = arguments.cases * arguments.designs
    per_case_in = sample.input_tokens / sample.cases
    per_case_out = sample.output_tokens / sample.cases
    total_in, total_out = round(per_case_in * runs), round(per_case_out * runs)
    per_case = settings.cost(round(per_case_in), round(per_case_out))
    total = settings.cost(total_in, total_out)

    print(f"provider:       {describe(settings)}")
    print(f"prices:         {_price(settings.price_input)} in, {_price(settings.price_output)} out")
    calls = sample.calls / sample.cases
    print(f"measured:       {sample.cases} generated cases, {calls:.1f} calls each")
    tokens = f"~{per_case_in:,.0f} tokens in, ~{per_case_out:,.0f} out"
    print(f"per case:       {tokens}, {dollars(per_case)}")
    print(f"for {runs:,} runs: ~{total_in:,} tokens in, ~{total_out:,} out, {dollars(total)}")
    print()
    print("An estimate, not a quote: about four characters per token, and no allowance for hidden")
    print("reasoning tokens, which models that think before answering bill as output.")
    if settings.price_input is None:
        print(
            "No price is known for this model: set CASEFILE_PRICE_INPUT and CASEFILE_PRICE_OUTPUT."
        )
    return 0


def _price(per_million: float | None) -> str:
    """Format a price per million tokens."""
    return "unknown" if per_million is None else f"${per_million:g}/M tokens"


if __name__ == "__main__":
    sys.exit(main())
