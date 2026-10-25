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


def read_in_context(fact: str, text: str, start: int, end: int) -> str | int | bool | None:
    """Return the value ``text[start:end]`` states for ``fact``, or None if it is not a field value.

    The span must be the whole value of one of the fact's fields (``FIELD_PATTERNS``). Reading the
    words alone is not enough: almost any fragment can be read as "yes", and any digits as a count.
    """
    if (start, end) not in fields(fact, text):
        return None
    return read_quote(fact, text[start:end])


_TRUE = frozenset({"true", "yes"})
_FALSE = frozenset({"false", "no"})


def coerce(fact: str, value: str | int | bool) -> str | int | bool | None:
    """Normalise a reported value to the fact's type, or return None if it cannot be one.

    Language models report every value as text ("6", "yes", "2026-03-01"); the reference model
    reports typed values. Both end up as what the rules engine reads: an ISO date, an int or a bool.
    """
    kind = FACT_TYPES.get(fact)
    if kind is date:
        return _date(value) if isinstance(value, str) else None
    if kind is int:
        if type(value) is int:
            return value
        return int(value) if isinstance(value, str) and value.strip().isdigit() else None
    if kind is bool:
        if type(value) is bool:
            return value
        word = value.strip().lower() if isinstance(value, str) else ""
        return True if word in _TRUE else False if word in _FALSE else None
    return None


FIELD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "request_date": (re.compile(r"^Request date: (?P<v>[^\n]+)$", re.M),),
    "prior_mri": (re.compile(r"^Prior MRI of this knee: (?P<v>Yes|No)\b", re.M),),
    "prior_mri_date": (re.compile(r"^Prior MRI of this knee: Yes, on (?P<v>[^\n]+)$", re.M),),
    "therapy_weeks": (
        re.compile(r"^Weeks of conservative therapy: (?P<v>\d+)$", re.M),
        re.compile(r"\bcompleted (?P<v>\d+ weeks?)\b"),
    ),
    "symptom_onset": (re.compile(r"\bpain since (?P<v>\d{1,2} [A-Z][a-z]+ \d{4})\b"),),
    "red_flag": (re.compile(r"^Red flags: (?P<v>[^.\n]+)", re.M),),
    "xray_date": (re.compile(r"^Examination: X-ray[^\n]*\nDate: (?P<v>[^\n]+)$", re.M),),
    "referral": (re.compile(r"\AREFERRAL\n.*?\b(?P<v>Referred by)\b", re.S),),
}
"""Where each fact is stated in the documents: the labelled field and the exact value span.

A fact is grounded only if the cited words are the *whole* value of one of its fields. A fragment
("n" out of "none", "2026" out of a date) never is, and a referral only counts inside a referral
letter. These patterns describe the synthetic documents; real documents would need their own.
"""


def fields(fact: str, text: str) -> list[tuple[int, int]]:
    """Return the spans in ``text`` that are the whole value of a ``fact`` field."""
    spans: list[tuple[int, int]] = []
    for pattern in FIELD_PATTERNS.get(fact, ()):
        spans.extend(match.span("v") for match in pattern.finditer(text))
    return sorted(set(spans))
