"""Tests for the SQLite event log: linking, persistence, refusal paths and tamper evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from casefile.canonical import CanonicalJsonError, JsonValue
from casefile.events import GENESIS_HASH
from casefile.log import AppendConflict, EventLog

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a path for a fresh log file."""
    return tmp_path / "events.sqlite"


@pytest.fixture
def log(db_path: Path) -> Iterator[EventLog]:
    """Yield an open log with a fixed clock, closed after the test."""
    with EventLog(db_path, clock=lambda: NOW) as opened:
        yield opened


def raw(db_path: Path) -> sqlite3.Connection:
    """Open the log's file directly, the way an attacker with file access would."""
    return sqlite3.connect(db_path, isolation_level=None)


def test_events_are_numbered_and_linked_per_case(log: EventLog) -> None:
    a1 = log.append("a", "Opened", {"n": 1})
    b1 = log.append("b", "Opened", {"n": 1})
    a2 = log.append("a", "Stepped", {"n": 2}, step_id="s/1")
    assert (a1.seq, a2.seq, b1.seq) == (1, 2, 1)
    assert a1.prev_hash == GENESIS_HASH == b1.prev_hash
    assert a2.prev_hash == a1.hash
    assert a2.recorded_at == "2026-01-02T03:04:05+00:00"
    assert log.read("a") == [a1, a2]
    assert log.case_ids() == ["a", "b"]


def test_events_survive_closing_and_reopening(db_path: Path) -> None:
    with EventLog(db_path, clock=lambda: NOW) as first:
        written = [first.append("a", "T", {"i": i, "who": "Zoë"}) for i in range(3)]
    with EventLog(db_path, clock=lambda: NOW) as second:
        assert second.read("a") == written
        assert second.verify("a") is None


def test_an_unknown_case_reads_empty_and_verifies(log: EventLog) -> None:
    assert log.read("nobody") == []
    assert log.verify("nobody") is None


def test_a_naive_clock_is_refused_and_nothing_written(db_path: Path) -> None:
    with EventLog(db_path, clock=lambda: datetime(2026, 1, 1)) as naive:
        with pytest.raises(ValueError, match="naive"):
            naive.append("a", "T", {})
        assert naive.read("a") == []


def test_a_float_payload_is_refused_and_nothing_written(log: EventLog) -> None:
    with pytest.raises(CanonicalJsonError):
        log.append("a", "T", cast(dict[str, JsonValue], {"cost": 0.25}))
    assert log.read("a") == []


def test_a_stale_writer_gets_a_conflict(log: EventLog) -> None:
    log.append("a", "T", {}, expected_seq=0)
    log.append("a", "T", {}, expected_seq=1)
    with pytest.raises(AppendConflict, match="expected last seq 1"):
        log.append("a", "T", {}, expected_seq=1)
    assert [event.seq for event in log.read("a")] == [1, 2]


def test_update_and_delete_are_refused_by_the_database(log: EventLog, db_path: Path) -> None:
    log.append("a", "T", {"i": 1})
    with raw(db_path) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("UPDATE events SET payload = '{}'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("DELETE FROM events")
    assert len(log.read("a")) == 1


def test_a_tampered_payload_is_found_at_its_seq(log: EventLog, db_path: Path) -> None:
    for i in range(3):
        log.append("a", "T", {"amount": 100 + i})
    with raw(db_path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET payload = replace(payload, '101', '901') WHERE seq = 2"
        )
    assert log.verify("a") == 2


def test_a_relinked_chain_is_found_where_the_forgery_starts(log: EventLog, db_path: Path) -> None:
    for i in range(3):
        log.append("a", "T", {"i": i})
    with raw(db_path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET prev_hash = ? WHERE seq = 3", ("f" * 64,))
    assert log.verify("a") == 3


def test_a_stored_payload_that_is_not_an_object_is_refused_on_read(
    log: EventLog, db_path: Path
) -> None:
    log.append("a", "T", {"i": 1})
    with raw(db_path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET payload = '[1]'")
    with pytest.raises(ValueError, match="not a JSON object"):
        log.read("a")


def test_recorded_at_is_stored_in_utc_whatever_the_clock_zone(db_path: Path) -> None:
    karachi = timezone(timedelta(hours=5))
    with EventLog(db_path, clock=lambda: datetime(2026, 1, 2, 8, 4, 5, tzinfo=karachi)) as zoned:
        assert zoned.append("a", "T", {}).recorded_at == "2026-01-02T03:04:05+00:00"


def test_a_saved_head_catches_events_cut_off_the_end(log: EventLog, db_path: Path) -> None:
    for i in range(3):
        log.append("a", "T", {"i": i})
    head = log.head("a")
    assert head is not None
    assert head[0] == 3
    assert log.verify("a", head) is None
    with raw(db_path) as connection:
        connection.execute("DROP TRIGGER events_no_delete")
        connection.execute("DELETE FROM events WHERE seq = 3")
    assert log.verify("a") is None  # the chain alone cannot see a cut tail
    assert log.verify("a", head) == 3
    assert log.head("nobody") is None


def test_a_saved_head_catches_a_rewritten_event(log: EventLog, db_path: Path) -> None:
    for i in range(2):
        log.append("a", "T", {"i": i})
    head = log.head("a")
    assert head is not None
    with raw(db_path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET hash = ? WHERE seq = 2", ("0" * 64,))
    assert log.verify("a", head) == 2
