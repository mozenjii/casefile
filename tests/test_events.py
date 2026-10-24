"""Tests for the event envelope hash and the chain check over a list of events."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from casefile.canonical import JsonValue
from casefile.events import GENESIS_HASH, Event, event_hash, first_broken_link

AT = "2026-01-02T03:04:05+00:00"


def link(seq: int, payload: dict[str, JsonValue], prev_hash: str) -> Event:
    """Build event ``seq`` of case ``c1`` with a correct hash over its fields."""
    unhashed = Event("c1", seq, "T", 1, None, payload, AT, prev_hash, hash="")
    return dataclasses.replace(unhashed, hash=unhashed.expected_hash())


def make_chain(payloads: list[dict[str, JsonValue]]) -> list[Event]:
    """Build a correctly linked chain for case ``c1`` with one event per payload."""
    events: list[Event] = []
    for seq, payload in enumerate(payloads, start=1):
        events.append(link(seq, payload, events[-1].hash if events else GENESIS_HASH))
    return events


def test_event_hash_is_sha256_of_the_sorted_compact_envelope() -> None:
    written_by_hand = (
        '{"case_id":"c1","payload":{"n":1},"prev_hash":"' + GENESIS_HASH + '",'
        '"recorded_at":"' + AT + '","schema_version":1,"seq":1,"step_id":"s/1","type":"T"}'
    )
    actual = event_hash(
        case_id="c1",
        seq=1,
        type="T",
        schema_version=1,
        step_id="s/1",
        payload={"n": 1},
        recorded_at=AT,
        prev_hash=GENESIS_HASH,
    )
    assert actual == hashlib.sha256(written_by_hand.encode()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "c2"),
        ("seq", 2),
        ("type", "U"),
        ("schema_version", 2),
        ("step_id", "s"),
        ("payload", {"x": 0}),
        ("recorded_at", "2026-01-02T03:04:06+00:00"),
        ("prev_hash", "1" * 64),
    ],
)
def test_every_field_changes_the_hash(field: str, value: object) -> None:
    original = link(1, {}, GENESIS_HASH)
    assert dataclasses.replace(original, **{field: value}).expected_hash() != original.hash


def test_an_intact_chain_has_no_broken_link() -> None:
    assert first_broken_link(make_chain([{"i": 1}, {"i": 2}, {"i": 3}])) is None
    assert first_broken_link([]) is None


def test_a_changed_payload_is_found_at_its_seq() -> None:
    events = make_chain([{"i": 1}, {"i": 2}, {"i": 3}])
    events[1] = dataclasses.replace(events[1], payload={"i": 20})
    assert first_broken_link(events) == 2


def test_a_rehashed_forgery_is_found_where_the_next_link_breaks() -> None:
    events = make_chain([{"i": 1}, {"i": 2}, {"i": 3}])
    events[1] = link(2, {"i": 20}, events[0].hash)
    assert first_broken_link(events) == 3


def test_a_missing_or_reordered_event_is_found() -> None:
    events = make_chain([{"i": 1}, {"i": 2}, {"i": 3}])
    assert first_broken_link([events[0], events[2]]) == 3
    assert first_broken_link([events[1], events[0], events[2]]) == 2


def test_events_are_immutable() -> None:
    (event,) = make_chain([{"i": 1}])
    field = "seq"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(event, field, 2)
