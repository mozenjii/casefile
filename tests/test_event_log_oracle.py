"""Independent oracle for canonical JSON, event hashing and the append-only event log.

These tests were written from the interface contract and the event-log design (the envelope
and the verifier) before any implementation existed, without reading the implementation. Every
expected byte string and hash is computed here with ``json`` and ``hashlib`` or written out by
hand; the code under test never produces its own expectation. A disagreement between this file and
the implementation is resolved by finding the cause, never by editing the oracle to agree.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from casefile.canonical import CanonicalJsonError, canonical_json, sha256_hex
from casefile.events import GENESIS_HASH, Event, event_hash
from casefile.log import EventLog

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_ISO = "2026-01-02T03:04:05+00:00"
ZEROS = "0" * 64

# The contract fixes the exception type for a naive clock but says nothing about its message, so
# the match pattern accepts any message, including an empty one.
ANY_MESSAGE = r"(?s).*"


def fixed_clock() -> datetime:
    """Return the same timezone-aware instant on every call."""
    return FIXED_TIME


@dataclass
class HandClock:
    """A clock the test sets by hand, so each append's timestamp is known exactly."""

    now: datetime

    def __call__(self) -> datetime:
        """Return the time most recently set by the test."""
        return self.now


@dataclass(frozen=True)
class Spec:
    """One event a test appends: its type, payload, step and schema version."""

    type: str
    payload: dict[str, Any]
    step_id: str | None = None
    schema_version: int = 1


THREE_EVENTS: tuple[Spec, ...] = (
    Spec("CaseOpened", {"source": "synthetic", "n": 1}),
    Spec("StepStarted", {"note": "alpha", "tokens": 120}, step_id="extract/1"),
    Spec(
        "StepCompleted",
        {"note": "Zoë ✓", "nested": {"b": [1, 2], "a": None}, "ok": True},
        step_id="extract/1",
        schema_version=2,
    ),
)


def reference_canonical(value: object) -> bytes:
    """Serialise ``value`` the way the contract describes, using only the standard library."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def reference_event_hash(
    *,
    case_id: str,
    seq: int,
    event_type: str,
    schema_version: int,
    step_id: str | None,
    payload: dict[str, Any],
    recorded_at: str,
    prev_hash: str,
) -> str:
    """Compute an event hash independently: SHA-256 hex of the canonical JSON of every field."""
    fields: dict[str, Any] = {
        "case_id": case_id,
        "payload": payload,
        "prev_hash": prev_hash,
        "recorded_at": recorded_at,
        "schema_version": schema_version,
        "seq": seq,
        "step_id": step_id,
        "type": event_type,
    }
    return hashlib.sha256(reference_canonical(fields)).hexdigest()


def reference_chain(
    case_id: str, specs: Sequence[Spec], recorded_at: str = FIXED_ISO
) -> list[tuple[str, str]]:
    """Return the expected ``(prev_hash, hash)`` pair for each spec appended to a fresh case."""
    pairs: list[tuple[str, str]] = []
    prev = ZEROS
    for seq, spec in enumerate(specs, start=1):
        digest = reference_event_hash(
            case_id=case_id,
            seq=seq,
            event_type=spec.type,
            schema_version=spec.schema_version,
            step_id=spec.step_id,
            payload=spec.payload,
            recorded_at=recorded_at,
            prev_hash=prev,
        )
        pairs.append((prev, digest))
        prev = digest
    return pairs


def append_spec(log: EventLog, case_id: str, spec: Spec) -> Event:
    """Append one spec to ``log`` and return the event the log reports."""
    return log.append(
        case_id,
        spec.type,
        spec.payload,
        step_id=spec.step_id,
        schema_version=spec.schema_version,
    )


def raw_execute(path: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    """Run one statement on the database file through a plain sqlite3 connection and commit."""
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(sql, params)


def raw_rows(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[Any, ...]]:
    """Fetch every row for one query through a plain sqlite3 connection."""
    with closing(sqlite3.connect(path)) as connection:
        rows: list[tuple[Any, ...]] = connection.execute(sql, params).fetchall()
    return rows


def all_rows(path: Path) -> list[tuple[Any, ...]]:
    """Return every stored event row, all contract columns, in a stable order."""
    return raw_rows(
        path,
        "SELECT case_id, seq, type, schema_version, step_id, payload, recorded_at, prev_hash, hash"
        " FROM events ORDER BY case_id, seq",
    )


def write_three(path: Path, case_id: str = "case-a") -> list[Event]:
    """Create a log at ``path`` holding the three standard events, close it, return the events."""
    with EventLog(path, fixed_clock) as log:
        return [append_spec(log, case_id, spec) for spec in THREE_EVENTS]


def verify_after_reopen(path: Path, case_id: str = "case-a") -> int | None:
    """Reopen the log at ``path`` and return what ``verify`` reports for ``case_id``."""
    with EventLog(path, fixed_clock) as log:
        return log.verify(case_id)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Give each test its own database file location."""
    return tmp_path / "events.sqlite3"


# --- canonical_json -----------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_drops_whitespace() -> None:
    value = {"b": 1, "a": [True, False, None], "c": "x"}
    assert canonical_json(value) == b'{"a":[true,false,null],"b":1,"c":"x"}'


def test_canonical_json_sorts_keys_at_every_depth() -> None:
    value: dict[str, Any] = {"z": {"y": [{"b": 2, "a": 1}], "x": {}}, "a": {"d": [], "c": ""}}
    assert canonical_json(value) == b'{"a":{"c":"","d":[]},"z":{"x":{},"y":[{"a":1,"b":2}]}}'


def test_canonical_json_keeps_non_ascii_as_utf8() -> None:
    value = {"name": "Zoë", "emoji": "\U0001f600", "é": "中"}
    expected = '{"emoji":"\U0001f600","name":"Zoë","é":"中"}'.encode()
    assert canonical_json(value) == expected
    assert b"\\u" not in canonical_json(value)


SCALAR_CASES: list[tuple[object, bytes]] = [
    (None, b"null"),
    (True, b"true"),
    (False, b"false"),
    (0, b"0"),
    (-5, b"-5"),
    (2**70, b"1180591620717411303424"),
    ("", b'""'),
    ([], b"[]"),
    ({}, b"{}"),
    ([[], {}, [None]], b"[[],{},[null]]"),
]


@pytest.mark.parametrize(("value", "expected"), SCALAR_CASES)
def test_canonical_json_scalars_and_empty_containers(value: object, expected: bytes) -> None:
    assert canonical_json(value) == expected


def test_canonical_json_escapes_quotes_backslashes_and_control_characters() -> None:
    value = 'q"b\\n\nt\t'
    assert canonical_json(value) == b'"q\\"b\\\\n\\nt\\t"'


@pytest.mark.parametrize(
    "value",
    [
        {"case_id": "c-1", "payload": {"tokens": 1200, "micro_usd": 3_400_000}},
        [1, "two", None, {"k": [False]}],
        {"é": 1, "e": 2, "E": 3, "": 4},
        {"deep": {"deeper": {"deepest": ["\u2028", "\U0001f600"]}}},
    ],
)
def test_canonical_json_matches_reference_serialiser(value: object) -> None:
    assert canonical_json(value) == reference_canonical(value)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.5,
        -2.0,
        math.nan,
        math.inf,
        -math.inf,
        [1, 2.5],
        {"a": {"b": [0.1]}},
        {"micro_usd": 1.0},
    ],
)
def test_canonical_json_rejects_floats(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json(value)


@pytest.mark.parametrize(
    "value",
    [
        {1: "a"},
        {None: "a"},
        {True: "a"},
        {(1, 2): "a"},
        {1.5: "a"},
        {"outer": {2: "inner"}},
        [{"ok": 1}, {3: "bad"}],
    ],
)
def test_canonical_json_rejects_non_string_keys(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json(value)


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        (),
        {1, 2},
        frozenset({"a"}),
        b"bytes",
        bytearray(b"x"),
        object(),
        Decimal("1"),
        FIXED_TIME,
        {"a": (1,)},
        [b"x"],
        {"a": {"b": {1}}},
    ],
)
def test_canonical_json_rejects_other_types(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json(value)


def test_canonical_json_error_is_a_value_error() -> None:
    assert issubclass(CanonicalJsonError, ValueError)


# --- sha256_hex ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ],
)
def test_sha256_hex_known_vectors(data: bytes, expected: str) -> None:
    assert sha256_hex(data) == expected


def test_sha256_hex_matches_hashlib_and_is_lowercase() -> None:
    data = bytes(range(256)) * 3
    digest = sha256_hex(data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert digest == digest.lower()
    assert len(digest) == 64


# --- events -------------------------------------------------------------------------------------


def test_genesis_hash_is_sixty_four_zeros() -> None:
    assert GENESIS_HASH == "0" * 64


def test_event_hash_matches_hand_written_canonical_bytes() -> None:
    hand_written = (
        b'{"case_id":"c1","payload":{"a":1},"prev_hash":"' + ZEROS.encode() + b'",'
        b'"recorded_at":"2026-01-02T03:04:05+00:00","schema_version":1,"seq":1,'
        b'"step_id":null,"type":"CaseOpened"}'
    )
    expected = hashlib.sha256(hand_written).hexdigest()
    actual = event_hash(
        case_id="c1",
        seq=1,
        type="CaseOpened",
        schema_version=1,
        step_id=None,
        payload={"a": 1},
        recorded_at=FIXED_ISO,
        prev_hash=ZEROS,
    )
    assert actual == expected


def test_event_hash_matches_independent_computation() -> None:
    payload: dict[str, Any] = {"note": "Zoë", "tokens": 7, "list": [{"b": 1, "a": None}]}
    prev = hashlib.sha256(b"previous").hexdigest()
    expected = reference_event_hash(
        case_id="case-é",
        seq=12,
        event_type="ModelCallRecorded",
        schema_version=3,
        step_id="policy/2",
        payload=payload,
        recorded_at="2026-03-04T05:06:07+00:00",
        prev_hash=prev,
    )
    actual = event_hash(
        case_id="case-é",
        seq=12,
        type="ModelCallRecorded",
        schema_version=3,
        step_id="policy/2",
        payload=payload,
        recorded_at="2026-03-04T05:06:07+00:00",
        prev_hash=prev,
    )
    assert actual == expected


BASE_FIELDS: dict[str, Any] = {
    "case_id": "c1",
    "seq": 2,
    "type": "StepStarted",
    "schema_version": 1,
    "step_id": "extract/1",
    "payload": {"a": 1},
    "recorded_at": FIXED_ISO,
    "prev_hash": ZEROS,
}


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("case_id", "c2"),
        ("seq", 3),
        ("type", "StepCompleted"),
        ("schema_version", 2),
        ("step_id", None),
        ("payload", {"a": 2}),
        ("recorded_at", "2026-01-02T03:04:06+00:00"),
        ("prev_hash", "1" * 64),
    ],
)
def test_event_hash_changes_when_any_field_changes(field: str, changed: object) -> None:
    original = event_hash(**BASE_FIELDS)
    altered_fields: dict[str, Any] = {**BASE_FIELDS, field: changed}
    altered = event_hash(**altered_fields)
    assert altered != original


def test_event_hash_rejects_float_payload() -> None:
    fields: dict[str, Any] = {**BASE_FIELDS, "payload": {"cost": 0.5}}
    with pytest.raises(CanonicalJsonError):
        event_hash(**fields)


def test_event_is_frozen_dataclass_with_contract_fields(db_path: Path) -> None:
    with EventLog(db_path, fixed_clock) as log:
        event = log.append("case-a", "CaseOpened", {"n": 1})
    assert isinstance(event, Event)
    assert dataclasses.is_dataclass(event)
    names = {field.name for field in dataclasses.fields(event)}
    assert names == {
        "case_id",
        "seq",
        "type",
        "schema_version",
        "step_id",
        "payload",
        "recorded_at",
        "prev_hash",
        "hash",
    }
    for field in dataclasses.fields(event):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(event, field.name, "changed")


# --- EventLog: append and read ------------------------------------------------------------------


def test_first_append_starts_a_case_at_seq_one_from_genesis(db_path: Path) -> None:
    with EventLog(db_path, fixed_clock) as log:
        event = log.append("case-a", "CaseOpened", {"source": "synthetic"})
    assert event.case_id == "case-a"
    assert event.seq == 1
    assert event.type == "CaseOpened"
    assert event.schema_version == 1
    assert event.step_id is None
    assert event.payload == {"source": "synthetic"}
    assert event.recorded_at == FIXED_ISO
    assert event.prev_hash == ZEROS
    assert event.hash == reference_event_hash(
        case_id="case-a",
        seq=1,
        event_type="CaseOpened",
        schema_version=1,
        step_id=None,
        payload={"source": "synthetic"},
        recorded_at=FIXED_ISO,
        prev_hash=ZEROS,
    )


def test_appends_link_three_events_into_a_chain(db_path: Path) -> None:
    with EventLog(db_path, fixed_clock) as log:
        events = [append_spec(log, "case-a", spec) for spec in THREE_EVENTS]
        read_back = log.read("case-a")
    expected = reference_chain("case-a", THREE_EVENTS)
    assert [event.seq for event in events] == [1, 2, 3]
    assert [(event.prev_hash, event.hash) for event in events] == expected
    assert events[1].prev_hash == events[0].hash
    assert events[2].prev_hash == events[1].hash
    for event, spec in zip(events, THREE_EVENTS, strict=True):
        assert event.type == spec.type
        assert event.payload == spec.payload
        assert event.step_id == spec.step_id
        assert event.schema_version == spec.schema_version
    assert read_back == events


def test_interleaved_cases_keep_independent_sequences_and_chains(db_path: Path) -> None:
    case_b_specs = (
        Spec("CaseOpened", {"source": "other"}),
        Spec("StepStarted", {"note": "beta"}, step_id="extract/1"),
    )
    with EventLog(db_path, fixed_clock) as log:
        a1 = append_spec(log, "case-a", THREE_EVENTS[0])
        b1 = append_spec(log, "case-b", case_b_specs[0])
        a2 = append_spec(log, "case-a", THREE_EVENTS[1])
        b2 = append_spec(log, "case-b", case_b_specs[1])
        a3 = append_spec(log, "case-a", THREE_EVENTS[2])
        read_a = log.read("case-a")
        read_b = log.read("case-b")
        assert log.verify("case-a") is None
        assert log.verify("case-b") is None
    assert [a1.seq, a2.seq, a3.seq] == [1, 2, 3]
    assert [b1.seq, b2.seq] == [1, 2]
    assert b1.prev_hash == ZEROS
    assert [(e.prev_hash, e.hash) for e in (a1, a2, a3)] == reference_chain("case-a", THREE_EVENTS)
    assert [(e.prev_hash, e.hash) for e in (b1, b2)] == reference_chain("case-b", case_b_specs)
    assert read_a == [a1, a2, a3]
    assert read_b == [b1, b2]


def test_read_unknown_case_is_empty(db_path: Path) -> None:
    with EventLog(db_path, fixed_clock) as log:
        assert log.read("never-seen") == []
        log.append("case-a", "CaseOpened", {"n": 1})
        assert log.read("never-seen") == []


def test_recorded_at_is_the_clock_isoformat_at_each_append(db_path: Path) -> None:
    clock = HandClock(datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC))
    with EventLog(db_path, clock) as log:
        first = log.append("case-a", "CaseOpened", {"n": 1})
        clock.now = datetime(2026, 5, 6, 7, 8, 10, 123456, tzinfo=UTC)
        second = log.append("case-a", "StepStarted", {"n": 2})
    assert first.recorded_at == "2026-05-06T07:08:09+00:00"
    assert second.recorded_at == "2026-05-06T07:08:10.123456+00:00"
    assert second.hash == reference_event_hash(
        case_id="case-a",
        seq=2,
        event_type="StepStarted",
        schema_version=1,
        step_id=None,
        payload={"n": 2},
        recorded_at="2026-05-06T07:08:10.123456+00:00",
        prev_hash=first.hash,
    )


def test_events_survive_close_and_reopen(db_path: Path) -> None:
    written = write_three(db_path)
    assert db_path.exists()
    with EventLog(db_path, fixed_clock) as log:
        assert log.read("case-a") == written
        fourth = log.append("case-a", "CaseClosed", {"outcome": "approved"})
        assert log.verify("case-a") is None
    assert fourth.seq == 4
    assert fourth.prev_hash == written[2].hash
    with EventLog(db_path, fixed_clock) as log:
        assert log.read("case-a") == [*written, fourth]


def test_explicit_close_without_context_manager(db_path: Path) -> None:
    log = EventLog(db_path, fixed_clock)
    try:
        event = log.append("case-a", "CaseOpened", {"n": 1})
    finally:
        log.close()
    reopened = EventLog(db_path, fixed_clock)
    try:
        assert reopened.read("case-a") == [event]
    finally:
        reopened.close()


def test_storage_rows_match_returned_events(db_path: Path) -> None:
    events = write_three(db_path)
    rows = all_rows(db_path)
    expected = [
        (
            event.case_id,
            event.seq,
            event.type,
            event.schema_version,
            event.step_id,
            reference_canonical(spec.payload).decode("utf-8"),
            event.recorded_at,
            event.prev_hash,
            event.hash,
        )
        for event, spec in zip(events, THREE_EVENTS, strict=True)
    ]
    assert rows == expected


def test_naive_clock_is_rejected_and_nothing_is_appended(db_path: Path) -> None:
    write_three(db_path)
    before = all_rows(db_path)
    log = EventLog(db_path, lambda: datetime(2026, 1, 2, 3, 4, 5))
    try:
        with pytest.raises(ValueError, match=ANY_MESSAGE):
            log.append("case-a", "StepStarted", {"n": 4})
        with pytest.raises(ValueError, match=ANY_MESSAGE):
            log.append("case-new", "CaseOpened", {"n": 1})
        assert log.read("case-new") == []
    finally:
        log.close()
    assert all_rows(db_path) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"cost": 1.5},
        {"cost": math.nan},
        {"tokens": [1, 2, 3.0]},
        {"deep": {"deeper": {"x": math.inf}}},
        {"pair": (1, 2)},
        {"blob": b"x"},
        {"nested": {7: "int key"}},
    ],
)
def test_rejected_payload_appends_nothing(db_path: Path, payload: dict[str, Any]) -> None:
    with EventLog(db_path, fixed_clock) as log:
        first = log.append("case-a", "CaseOpened", {"n": 1})
        with pytest.raises(CanonicalJsonError):
            log.append("case-a", "StepStarted", payload)
        with pytest.raises(CanonicalJsonError):
            log.append("case-fresh", "CaseOpened", payload)
        assert log.read("case-a") == [first]
        assert log.read("case-fresh") == []
        second = log.append("case-a", "StepStarted", {"n": 2})
    assert second.seq == 2
    assert second.prev_hash == first.hash


# --- EventLog: append-only enforcement ----------------------------------------------------------


def test_append_only_triggers_are_installed(db_path: Path) -> None:
    write_three(db_path)
    names = {
        row[0] for row in raw_rows(db_path, "SELECT name FROM sqlite_master WHERE type = 'trigger'")
    }
    assert {"events_no_update", "events_no_delete"} <= names


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE events SET type = 'Tampered' WHERE case_id = 'case-a' AND seq = 2",
        "UPDATE events SET payload = '{}' WHERE case_id = 'case-a'",
        "UPDATE events SET hash = prev_hash",
        "DELETE FROM events WHERE case_id = 'case-a' AND seq = 3",
        "DELETE FROM events",
    ],
)
def test_raw_update_and_delete_are_blocked(db_path: Path, sql: str) -> None:
    write_three(db_path)
    before = all_rows(db_path)
    with pytest.raises(sqlite3.DatabaseError):
        raw_execute(db_path, sql)
    assert all_rows(db_path) == before
    assert verify_after_reopen(db_path) is None


# --- EventLog: verify ---------------------------------------------------------------------------


def test_verify_reports_intact_chain_and_empty_case_as_none(db_path: Path) -> None:
    with EventLog(db_path, fixed_clock) as log:
        assert log.verify("case-a") is None
        for spec in THREE_EVENTS:
            append_spec(log, "case-a", spec)
        assert log.verify("case-a") is None
        assert log.verify("never-seen") is None


def test_verify_names_the_seq_whose_payload_lost_one_byte(db_path: Path) -> None:
    write_three(db_path)
    raw_execute(db_path, "DROP TRIGGER events_no_update")
    stored = raw_rows(db_path, "SELECT payload FROM events WHERE case_id = 'case-a' AND seq = 2")
    original: str = stored[0][0]
    tampered = original.replace("alpha", "alphb")
    original_bytes, tampered_bytes = original.encode(), tampered.encode()
    assert len(original_bytes) == len(tampered_bytes)
    assert sum(a != b for a, b in zip(original_bytes, tampered_bytes, strict=True)) == 1
    raw_execute(
        db_path,
        "UPDATE events SET payload = ? WHERE case_id = 'case-a' AND seq = 2",
        (tampered,),
    )
    assert verify_after_reopen(db_path) == 2


@pytest.mark.parametrize("seq", [1, 2, 3])
def test_verify_names_the_seq_whose_prev_hash_was_changed(db_path: Path, seq: int) -> None:
    write_three(db_path)
    raw_execute(db_path, "DROP TRIGGER events_no_update")
    raw_execute(
        db_path,
        "UPDATE events SET prev_hash = ? WHERE case_id = 'case-a' AND seq = ?",
        ("f" * 64, seq),
    )
    assert verify_after_reopen(db_path) == seq


@pytest.mark.parametrize(
    ("sql", "params", "bad_seq"),
    [
        ("UPDATE events SET type = ? WHERE case_id = 'case-a' AND seq = ?", ("Forged", 2), 2),
        ("UPDATE events SET schema_version = ? WHERE case_id = 'case-a' AND seq = ?", (9, 2), 2),
        ("UPDATE events SET step_id = ? WHERE case_id = 'case-a' AND seq = ?", ("x/9", 1), 1),
        ("UPDATE events SET step_id = ? WHERE case_id = 'case-a' AND seq = ?", (None, 2), 2),
        (
            "UPDATE events SET recorded_at = ? WHERE case_id = 'case-a' AND seq = ?",
            ("2030-01-01T00:00:00+00:00", 3),
            3,
        ),
        ("UPDATE events SET hash = ? WHERE case_id = 'case-a' AND seq = ?", ("a" * 64, 2), 2),
        ("UPDATE events SET hash = ? WHERE case_id = 'case-a' AND seq = ?", ("a" * 64, 3), 3),
    ],
)
def test_verify_names_the_seq_when_a_hashed_column_is_edited(
    db_path: Path, sql: str, params: tuple[object, ...], bad_seq: int
) -> None:
    write_three(db_path)
    raw_execute(db_path, "DROP TRIGGER events_no_update")
    raw_execute(db_path, sql, params)
    assert verify_after_reopen(db_path) == bad_seq


def test_verify_catches_a_rewritten_event_by_the_next_prev_hash(db_path: Path) -> None:
    events = write_three(db_path)
    forged_payload: dict[str, Any] = {"note": "forged", "tokens": 1}
    forged_hash = reference_event_hash(
        case_id="case-a",
        seq=2,
        event_type=THREE_EVENTS[1].type,
        schema_version=THREE_EVENTS[1].schema_version,
        step_id=THREE_EVENTS[1].step_id,
        payload=forged_payload,
        recorded_at=FIXED_ISO,
        prev_hash=events[0].hash,
    )
    raw_execute(db_path, "DROP TRIGGER events_no_update")
    raw_execute(
        db_path,
        "UPDATE events SET payload = ?, hash = ? WHERE case_id = 'case-a' AND seq = 2",
        (reference_canonical(forged_payload).decode("utf-8"), forged_hash),
    )
    assert verify_after_reopen(db_path) == 3


def test_tampering_one_case_does_not_flag_another(db_path: Path) -> None:
    write_three(db_path, "case-a")
    write_three(db_path, "case-b")
    raw_execute(db_path, "DROP TRIGGER events_no_update")
    raw_execute(
        db_path,
        "UPDATE events SET type = 'Forged' WHERE case_id = 'case-a' AND seq = 2",
    )
    with EventLog(db_path, fixed_clock) as log:
        assert log.verify("case-a") == 2
        assert log.verify("case-b") is None
