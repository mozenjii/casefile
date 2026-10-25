"""Read a fact's value back out of the words that state it.

This is the heart of the grounding check. A model that extracts a fact must say where in the
document it read it; ``read_quote`` takes the quoted span and decides, without the model's help,
what value those words actually state. If that is not the value the model reported, the
observation is not grounded and is thrown away.

The forms understood are the ones the documents use: ISO dates and "3 March 2026", a number of
weeks, yes/no, "none" or a named red flag, and the "Referred by" line of a referral letter.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from casefile.rules import FACT_TYPES

_LONG_DATE = "%d %B %Y"
_WEEKS = re.compile(r"(\d+)(?: weeks?)?")
_NEGATIVE = frozenset({"no", "none"})


def read_quote(fact: str, quote: str) -> str | int | bool | None:
    """Return the value ``quote`` states for ``fact``, or None if it states no such value."""
    kind = FACT_TYPES.get(fact)
    text = quote.strip()
    if kind is date:
        return _date(text)
    if kind is int:
        match = _WEEKS.fullmatch(text)
        return int(match.group(1)) if match else None
    if kind is bool and text:
        if fact == "referral":
            return True if text == "Referred by" else None
        return text.lower() not in _NEGATIVE
    return None


def _long_date(text: str) -> date:
    """Parse "3 March 2026"."""
    return datetime.strptime(text, _LONG_DATE).date()


def _date(text: str) -> str | None:
    """Parse an ISO or long-form date into ISO, or return None."""
    for parse in (date.fromisoformat, _long_date):
        try:
            return parse(text).isoformat()
        except ValueError:
            continue
    return None


_LABELS = {"red_flag": "Red flags:", "prior_mri": "Prior MRI of this knee:"}
"""Yes/no facts need context: "none" or a flag's name only counts after its field's label."""


def read_in_context(fact: str, text: str, start: int, end: int) -> str | int | bool | None:
    """Return the value ``text[start:end]`` states for ``fact``, checking its label if it needs one.

    For a yes/no fact, almost any words could be read as "yes", so the span must follow the
    field's own label on the same line. Without that, a model could point at "Findings" and have
    it accepted as a red flag.
    """
    label = _LABELS.get(fact)
    if label is not None:
        line_start = text.rfind("\n", 0, start) + 1
        if not text[line_start:start].rstrip().endswith(label):
            return None
    return read_quote(fact, text[start:end])
