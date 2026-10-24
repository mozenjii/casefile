"""Tests for identifier detection and redaction.

Values that look real are assembled from pieces at run time, so this file itself never contains a
possibly real identifier for the repository scan to find.
"""

from __future__ import annotations

import pytest

from casefile.identifiers import Identifier, find_identifiers, luhn_valid, redact


def ssn(area: str, group: str, serial: str) -> str:
    """Join SSN parts with dashes."""
    return "-".join((area, group, serial))


def phone(area: str, exchange: str, line: str) -> str:
    """Join phone parts with dashes."""
    return "-".join((area, exchange, line))


def email(user: str, domain: str) -> str:
    """Join an address."""
    return user + "@" + domain


def with_check_digit(body: str) -> str:
    """Append the digit that makes ``body`` pass the Luhn check."""
    return next(body + d for d in "0123456789" if luhn_valid(body + d))


def kinds(text: str) -> list[tuple[str, bool]]:
    """Reduce matches to ``(kind, possibly_real)`` pairs."""
    return [(found.kind, found.possibly_real) for found in find_identifiers(text)]


@pytest.mark.parametrize(
    ("value", "possibly_real"),
    [
        (ssn("123", "45", "6789"), True),
        (ssn("912", "70", "1234"), True),  # an ITIN-shaped number can be a real taxpayer's
        (ssn("000", "12", "3456"), False),
        (ssn("666", "12", "3456"), False),
        (ssn("123", "00", "4567"), False),
        (ssn("123", "45", "0000"), False),
    ],
)
def test_ssns_are_real_unless_never_issued(value: str, possibly_real: bool) -> None:
    assert kinds(f"member {value} on file") == [("ssn", possibly_real)]


@pytest.mark.parametrize(
    ("value", "possibly_real"),
    [
        (phone("212", "867", "5309"), True),
        ("(" + "212" + ") " + "868-5309", True),
        (phone("212", "555", "0142"), False),
        (phone("212", "555", "0200"), True),  # only 0100-0199 is reserved for fiction
    ],
)
def test_phones_are_real_unless_in_the_fiction_range(value: str, possibly_real: bool) -> None:
    assert kinds(f"call {value} today") == [("phone", possibly_real)]


@pytest.mark.parametrize(
    ("value", "possibly_real"),
    [
        (email("jane.doe", "gmail.com"), True),
        (email("jane.doe", "clinic.example.com"), False),
        (email("ops", "example.org"), False),
        (email("ops", "casefile.test"), False),
        (email("ops", "notexample.com"), True),
    ],
)
def test_emails_are_real_unless_at_a_reserved_domain(value: str, possibly_real: bool) -> None:
    assert kinds(f"write to {value}.") == [("email", possibly_real)]


def test_cards_must_pass_luhn_and_test_numbers_are_fictional() -> None:
    real_looking = with_check_digit("453201511283036")
    assert kinds(f"card {real_looking}") == [("card", True)]
    assert kinds("card 4111 1111 1111 1111") == [("card", False)]
    one_off = real_looking[:-1] + str((int(real_looking[-1]) + 1) % 10)
    assert kinds(f"order {one_off}") == []


def test_medicare_identifiers_are_always_possibly_real() -> None:
    mbi = "1EG4" + "-TE5-" + "MK73"
    assert kinds(f"MBI {mbi}") == [("mbi", True)]
    assert kinds(f"MBI {mbi.replace('-', '')}") == [("mbi", True)]
    assert kinds("token 1EB4TE5MK73 uses a letter the layout excludes") == []


def test_ordinary_text_has_no_identifiers() -> None:
    text = "Case SYN-0001, step extract/1, 2026-10-25, version 6.170.9, 15 of 15 cells."
    assert find_identifiers(text) == []


def test_overlapping_matches_report_the_longest_once() -> None:
    card = with_check_digit("453201511283036")
    spaced = " ".join(card[i : i + 4] for i in range(0, 16, 4))
    assert kinds(spaced) == [("card", True)]


def test_redact_replaces_every_identifier_fictional_or_not() -> None:
    address, number = email("ops", "example.com"), ssn("123", "45", "6789")
    text = f"{address} and {number}, call {phone('212', '555', '0142')}."
    assert redact(text) == "[email] and [ssn], call [phone]."
    assert redact("nothing to hide") == "nothing to hide"


def test_identifiers_carry_their_span() -> None:
    value = ssn("123", "45", "6789")
    (found,) = find_identifiers(f"id={value}")
    assert found == Identifier("ssn", 3, 3 + len(value), value, True)


def test_luhn_matches_the_published_example() -> None:
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")
