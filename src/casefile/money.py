"""Money as the system keeps it: integer micro-dollars, written for people only at the edge.

Costs are integers (millionths of a dollar) everywhere inside, because the event log refuses floats:
two serialisers disagree about them and a hash chain cannot. This module is the one place they
become text a person reads.
"""

from __future__ import annotations


def dollars(microusd: int | None) -> str:
    """Format micro-dollars for people: ``$0.0021``, ``$12.40``, or ``unknown``."""
    if microusd is None:
        return "unknown"
    value = microusd / 1_000_000
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"
