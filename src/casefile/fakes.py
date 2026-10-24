"""Stand-ins for the outside world, with a ledger that outlives the process that called them.

The crash matrix kills the process that runs a case, so what the outside world saw has to be
recorded somewhere the kill cannot reach: a SQLite file of its own, written with
``synchronous=FULL``. After a crash and a resume the ledger answers the questions the matrix asks.
How many times was the model called? How many times was each effect sent, and was it applied
exactly once?

``FakeModel`` answers with a caller-supplied pure function, so its answers are deterministic.
``FakeDestination`` applies an effect the first time it sees a key and returns the same receipt
for every repeat, which is the contract a real destination must offer for the outbox to be safe.
These fakes prove our handling, not any real counterparty's behaviour.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from casefile.canonical import JsonValue, canonical_json
from casefile.executor import Call, request_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls (n INTEGER PRIMARY KEY, request_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sends (n INTEGER PRIMARY KEY, key TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS applied (key TEXT PRIMARY KEY, request TEXT NOT NULL, receipt TEXT);
"""


class Ledger:
    """What the outside world saw, kept in its own SQLite file."""

    def __init__(self, path: Path) -> None:
        """Open (creating if needed) the ledger at ``path``."""
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the ledger. Everything recorded is already durable."""
        self._db.close()

    def record_model_call(self, digest: str) -> None:
        """Record that the model was asked the request with hash ``digest``."""
        self._db.execute("INSERT INTO model_calls (request_hash) VALUES (?)", (digest,))

    def record_send(self, key: str, request: JsonValue) -> str:
        """Record one delivery of ``key`` and return its receipt, applying it on first sight."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("INSERT INTO sends (key) VALUES (?)", (key,))
            receipt = f"RFI-{key[:10]}"
            self._db.execute(
                "INSERT OR IGNORE INTO applied (key, request, receipt) VALUES (?, ?, ?)",
                (key, canonical_json(request).decode(), receipt),
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return receipt

    def model_calls(self) -> int:
        """Return how many times the model was called, repeats included."""
        return cast(int, self._db.execute("SELECT count(*) FROM model_calls").fetchone()[0])

    def sends(self) -> dict[str, int]:
        """Return how many times each key was delivered, repeats included."""
        rows = self._db.execute("SELECT key, count(*) FROM sends GROUP BY key ORDER BY key")
        return {cast(str, key): cast(int, count) for key, count in rows}

    def applied(self) -> list[str]:
        """Return the keys whose effect was applied, each listed once by construction."""
        return [cast(str, row[0]) for row in self._db.execute("SELECT key FROM applied ORDER BY 1")]


class FakeModel:
    """A deterministic model: ``respond`` maps each request to its response; the ledger counts."""

    def __init__(self, ledger: Ledger, respond: Call) -> None:
        """Answer with ``respond`` and record each call in ``ledger``."""
        self._ledger = ledger
        self._respond = respond

    def __call__(self, request: JsonValue) -> JsonValue:
        """Record the call, then answer it."""
        self._ledger.record_model_call(request_hash(request))
        return self._respond(request)


class FakeDestination:
    """A destination that applies each idempotency key once and repeats its receipt after that."""

    def __init__(self, ledger: Ledger) -> None:
        """Record deliveries in ``ledger``."""
        self._ledger = ledger

    def send(self, key: str, request: JsonValue) -> JsonValue:
        """Deliver ``request`` under ``key`` and return the receipt."""
        return {"receipt": self._ledger.record_send(key, request)}
