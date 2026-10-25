"""A response cache for model calls, so running the same cases again costs nothing.

The event log already makes each case replay without calling the model. This cache covers what the
log cannot: a fresh run of the same cases (a reset demo, the same cases under a second workflow
design, a rerun of the whole comparison). An answer is keyed by the provider and model plus the
exact request, so changing the model or a single character of a prompt asks the model again, and a
retry that carries the earlier rejections is a different request and is never served a stale
answer.

A refusal or a network failure raises before anything is stored, so only real answers are cached.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from casefile.canonical import JsonValue, canonical_json, sha256_hex
from casefile.llm import Call


class ResponseCache:
    """Model answers on disk, keyed by a hash of who was asked and exactly what."""

    def __init__(self, path: Path) -> None:
        """Open (creating if needed) the cache at ``path``."""
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT NOT NULL)"
        )

    def close(self) -> None:
        """Close the cache."""
        self._db.close()

    def lookup(self, key: str) -> tuple[bool, JsonValue]:
        """Return ``(True, answer)`` for a stored key, else ``(False, None)``."""
        row = self._db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
        if row is None:
            return False, None
        answer: JsonValue = json.loads(row[0])
        return True, answer

    def store(self, key: str, answer: JsonValue) -> None:
        """Keep an answer; storing the same key twice keeps the first."""
        text = canonical_json(answer).decode()
        self._db.execute("INSERT OR IGNORE INTO answers (key, answer) VALUES (?, ?)", (key, text))

    def size(self) -> int:
        """How many answers are stored."""
        return int(self._db.execute("SELECT count(*) FROM answers").fetchone()[0])


def cached(model: Call, cache: ResponseCache, namespace: str) -> Call:
    """Wrap ``model`` so an identical request under ``namespace`` is answered from ``cache``."""

    def answer(request: JsonValue) -> JsonValue:
        key = sha256_hex(canonical_json([namespace, request]))
        found, stored = cache.lookup(key)
        if found:
            return stored
        fresh = model(request)
        cache.store(key, fresh)
        return fresh

    return answer
