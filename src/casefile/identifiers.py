"""Find personal identifiers in text, tell possibly real ones from fictional ones, redact them.

A scanner that flags every nine-digit number is switched off within a week. This one leans on the
ranges that registries reserve and never assign, so synthetic data can look realistic and still
be provably fake:

* **SSN**: an area of 000 or 666, a group of 00 or a serial of 0000 is never issued, so it cannot
  be anyone's. Areas 900-999 are not SSNs but can be real taxpayer numbers (ITINs), so they are
  flagged.
* **US phone**: 555-0100 to 555-0199 is reserved for fiction. Anything else with a valid area code
  and exchange could ring someone.
* **Email**: ``example.com``, ``.net`` and ``.org``, and the ``.test``, ``.example``, ``.invalid``
  and ``.localhost`` top-level domains are reserved and never delivered.
* **Card number**: only digit runs that pass the Luhn check are cards at all; the networks'
  published test numbers are fictional.
* **Medicare Beneficiary Identifier**: matched by its exact character layout. No range is reserved
  for fiction, so every match is possibly real, and synthetic cases must not contain one.

``find_identifiers`` reports every match with ``possibly_real`` set; the repository scan fails on
the possibly real ones. ``redact`` replaces every match, fictional or not, because a log line is no
place for anything shaped like an identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SSN = re.compile(r"\b(\d{3})-(\d{2})-(\d{4})\b")
_PHONE = re.compile(r"(?<!\d)\(?([2-9]\d{2})\)?[-. ]?([2-9]\d{2})[-. ](\d{4})(?!\d)")
# The lookbehind anchors the local part at its true start, so a long run of dots and letters
# with no @ is scanned once instead of once per starting position.
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_MBI_LETTER = "AC-HJKMNP-RT-Y"  # A to Z without B, I, L, O, S, Z
_MBI = re.compile(
    rf"\b[1-9][{_MBI_LETTER}][{_MBI_LETTER}0-9]\d-?[{_MBI_LETTER}][{_MBI_LETTER}0-9]\d-?"
    rf"[{_MBI_LETTER}]{{2}}\d{{2}}\b"
)

_FICTIONAL_DOMAINS = ("example.com", "example.net", "example.org")
_FICTIONAL_TLDS = ("test", "example", "invalid", "localhost")
_TEST_CARDS = frozenset(
    {
        "4111111111111111",
        "4242424242424242",
        "4012888888881881",
        "5555555555554444",
        "5105105105105100",
        "378282246310005",
        "371449635398431",
        "6011111111111117",
    }
)


@dataclass(frozen=True)
class Identifier:
    """One identifier-shaped match: what it is, where it is, and whether it could be someone's."""

    kind: str
    start: int
    end: int
    text: str
    possibly_real: bool


def find_identifiers(text: str) -> list[Identifier]:
    """Return every identifier-shaped match in ``text``, in order of position.

    Where two kinds overlap (a card number contains digit groups a phone pattern would also match),
    the longer match wins, so each span is reported once.
    """
    found: list[Identifier] = []
    for match in _SSN.finditer(text):
        area, group, serial = match.groups()
        never_issued = area in ("000", "666") or group == "00" or serial == "0000"
        found.append(_identifier("ssn", match, possibly_real=not never_issued))
    for match in _PHONE.finditer(text):
        _, exchange, line = match.groups()
        fictional = exchange == "555" and 100 <= int(line) <= 199
        found.append(_identifier("phone", match, possibly_real=not fictional))
    for match in _EMAIL.finditer(text):
        domain = match.group(1).lower()
        fictional = _reserved_domain(domain)
        found.append(_identifier("email", match, possibly_real=not fictional))
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if luhn_valid(digits):
            found.append(_identifier("card", match, possibly_real=digits not in _TEST_CARDS))
    for match in _MBI.finditer(text):
        found.append(_identifier("mbi", match, possibly_real=True))
    return _longest_first(found)


def redact(text: str) -> str:
    """Replace every identifier-shaped match in ``text`` with its kind in brackets."""
    pieces: list[str] = []
    cursor = 0
    for identifier in find_identifiers(text):
        pieces.append(text[cursor : identifier.start])
        pieces.append(f"[{identifier.kind}]")
        cursor = identifier.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum that every payment card number carries."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _reserved_domain(domain: str) -> bool:
    """Say whether ``domain`` is a documentation domain or under a reserved top-level domain."""
    if domain.rsplit(".", 1)[-1] in _FICTIONAL_TLDS:
        return True
    return any(domain == name or domain.endswith("." + name) for name in _FICTIONAL_DOMAINS)


def _identifier(kind: str, match: re.Match[str], *, possibly_real: bool) -> Identifier:
    """Wrap a regex match."""
    return Identifier(kind, match.start(), match.end(), match.group(), possibly_real)


def _longest_first(found: list[Identifier]) -> list[Identifier]:
    """Drop matches that overlap a longer one, and return the rest in order of position."""
    kept: list[Identifier] = []
    for candidate in sorted(found, key=lambda item: (item.start - item.end, item.start)):
        if all(candidate.end <= other.start or candidate.start >= other.end for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.start)
