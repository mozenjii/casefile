"""Tests for canonical JSON: one byte string per value, and refusal of anything ambiguous."""

from __future__ import annotations

import pytest

from casefile.canonical import CanonicalJsonError, canonical_json, sha256_hex


def test_keys_are_sorted_and_whitespace_dropped() -> None:
    assert canonical_json({"b": 1, "a": [True, None, "x"]}) == b'{"a":[true,null,"x"],"b":1}'


def test_nested_keys_are_sorted_at_every_level() -> None:
    assert canonical_json({"z": {"y": 1, "x": 2}}) == b'{"z":{"x":2,"y":1}}'


def test_non_ascii_text_is_kept_as_utf8() -> None:
    assert canonical_json({"name": "Zoë"}) == '{"name":"Zoë"}'.encode()


def test_insertion_order_does_not_change_the_bytes() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


@pytest.mark.parametrize(
    "value",
    [1.5, float("nan"), float("inf"), {"cost": 0.1}, [1, 2.0]],
    ids=["float", "nan", "inf", "nested float", "float in list"],
)
def test_floats_are_refused(value: object) -> None:
    with pytest.raises(CanonicalJsonError, match="float"):
        canonical_json(value)


@pytest.mark.parametrize(
    "value",
    [(1, 2), {1, 2}, b"bytes", object(), {1: "int key"}],
    ids=["tuple", "set", "bytes", "object", "int key"],
)
def test_other_types_are_refused(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json(value)


def test_booleans_are_not_mistaken_for_integers() -> None:
    assert canonical_json([True, 1, False, 0]) == b"[true,1,false,0]"


def test_sha256_hex_matches_a_published_vector() -> None:
    # The SHA-256 of "abc" from FIPS 180-2, appendix B.1.
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_hex(b"abc") == expected
