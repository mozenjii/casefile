"""The append-only event log, stored in SQLite.

Three properties matter and each is enforced by the database, not by convention:

* **Durable on return.** WAL mode with ``synchronous=FULL``: once ``append`` returns, the event
  survives the process being killed, which is what the crash matrix relies on.
* **Append-only.** Triggers abort any ``UPDATE`` or ``DELETE`` on the table. Tampering needs someone
  to drop the triggers first, and the hash chain then shows where they changed something.
* **No forked history.** ``(case_id, seq)`` is the primary key and ``seq`` is assigned under the
  write lock, so two writers cannot both append event 5. A writer that decided on stale state says
  so with ``expected_seq`` and gets ``AppendConflict``; it must re-read, never retry blindly.

The clock is injected so tests are deterministic. ``recorded_at`` is stored in UTC whatever zone the
clock reports, and a naive clock is refused.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from casefile.canonical import JsonValue, canonical_json
from casefile.events import GENESIS_HASH, Event, first_broken_link

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    case_id        TEXT    NOT NULL,
    seq            INTEGER NOT NULL CHECK (seq >= 1),
    type           TEXT    NOT NULL,
    schema_version INTEGER NOT NULL,
    step_id        TEXT,
    payload        TEXT    NOT NULL,
    recorded_at    TEXT    NOT NULL,
    prev_hash      TEXT    NOT NULL,
    hash           TEXT    NOT NULL,
    PRIMARY KEY (case_id, seq)
) WITHOUT ROWID;

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;

CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""

_INSERT = """
INSERT INTO events
    (case_id, seq, type, schema_version, step_id, payload, recorded_at, prev_hash, hash)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_LAST = "SELECT seq, hash FROM events WHERE case_id = ? ORDER BY seq DESC LIMIT 1"

_SELECT_CASE = """
SELECT case_id, seq, type, schema_version, step_id, payload, recorded_at, prev_hash, hash
FROM events WHERE case_id = ? ORDER BY seq
"""


class AppendConflict(Exception):
    """The log moved past the seq the caller decided on; re-read it before deciding again."""


class EventLog:
    """An append-only, hash-chained log of events for many cases in one SQLite file."""

    def __init__(self, path: Path, clock: Callable[[], datetime]) -> None:
        """Open (creating if needed) the log at ``path``; ``clock`` must return aware datetimes."""
        self._clock = clock
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        """Use the log as a context manager that closes it on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection, whether or not the block raised."""
        self.close()

    def close(self) -> None:
        """Close the database connection. Appended events are already durable."""
        self._db.close()

    def append(
        self,
        case_id: str,
        type: str,
        payload: Mapping[str, JsonValue],
        *,
        step_id: str | None = None,
        schema_version: int = 1,
        expected_seq: int | None = None,
    ) -> Event:
        """Append one event to ``case_id`` and return it as stored.

        ``expected_seq`` is the last seq the caller had read when it decided to append (0 for a new
        case). If the log has moved on since, the decision was made on stale state and
        ``AppendConflict`` is raised. ``CanonicalJsonError`` is raised for a payload with no
        canonical form (a float, say) and ``ValueError`` for a naive clock. Nothing is written in
        any of those cases.
        """
        payload_text = canonical_json(dict(payload)).decode()
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the clock returned a naive datetime; recorded_at must carry a zone")

        # IMMEDIATE takes the write lock before reading the last seq, so two writers serialise here
        # instead of both computing the same next seq.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            last = self._db.execute(_SELECT_LAST, (case_id,)).fetchone()
            last_seq, prev_hash = (0, GENESIS_HASH) if last is None else (last[0], last[1])
            if expected_seq is not None and expected_seq != last_seq:
                raise AppendConflict(
                    f"case {case_id!r}: expected last seq {expected_seq}, the log is at {last_seq}"
                )
            unhashed = Event(
                case_id=case_id,
                seq=last_seq + 1,
                type=type,
                schema_version=schema_version,
                step_id=step_id,
                payload=json.loads(payload_text),
                recorded_at=now.astimezone(UTC).isoformat(),
                prev_hash=prev_hash,
                hash="",
            )
            event = dataclasses.replace(unhashed, hash=unhashed.expected_hash())
            self._db.execute(
                _INSERT,
                (
                    event.case_id,
                    event.seq,
                    event.type,
                    event.schema_version,
                    event.step_id,
                    payload_text,
                    event.recorded_at,
                    event.prev_hash,
                    event.hash,
                ),
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return event

    def read(self, case_id: str) -> list[Event]:
        """Return every event of ``case_id`` in ``seq`` order, or ``[]`` for an unknown case."""
        rows = self._db.execute(_SELECT_CASE, (case_id,)).fetchall()
        return [_row_to_event(row) for row in rows]

    def verify(self, case_id: str) -> int | None:
        """Return the ``seq`` of the first event that breaks the hash chain, or None if intact."""
        return first_broken_link(self.read(case_id))

    def case_ids(self) -> list[str]:
        """Return every case that has at least one event, sorted."""
        return [
            row[0] for row in self._db.execute("SELECT DISTINCT case_id FROM events ORDER BY 1")
        ]


def _row_to_event(row: tuple[object, ...]) -> Event:
    """Rebuild an ``Event`` from a stored row. The hash check, not this, catches a tampered row."""
    case_id, seq, type_, version, step_id, payload, recorded_at, prev_hash, digest = row
    decoded = json.loads(cast(str, payload))
    if not isinstance(decoded, dict):
        raise ValueError(f"case {case_id!r} seq {seq!r}: payload is not a JSON object")
    return Event(
        case_id=cast(str, case_id),
        seq=cast(int, seq),
        type=cast(str, type_),
        schema_version=cast(int, version),
        step_id=cast(str | None, step_id),
        payload=cast(dict[str, JsonValue], decoded),
        recorded_at=cast(str, recorded_at),
        prev_hash=cast(str, prev_hash),
        hash=cast(str, digest),
    )
