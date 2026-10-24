"""The event envelope and the hash that chains one event to the one before it.

A case's history is a sequence of events, numbered 1, 2, 3 … per case. Each event stores the hash of
its predecessor and its own hash over every field, so changing any stored byte, reordering events or
dropping one breaks the chain at a place a verifier can name. The first event links to
``GENESIS_HASH``.

This module is pure: it builds and hashes values and never touches storage. ``casefile.log`` owns
persistence; ``casefile.state`` owns what the events mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from casefile.canonical import JsonValue, canonical_json, sha256_hex

GENESIS_HASH = "0" * 64
"""The ``prev_hash`` of every case's first event."""


@dataclass(frozen=True)
class Event:
    """One recorded fact about a case, as stored in the log.

    ``recorded_at`` is part of the hash, because it is part of what was recorded, but it is never
    part of the folded state, so a replayed case equals an uninterrupted one whatever the clock
    said.
    """

    case_id: str
    seq: int
    type: str
    schema_version: int
    step_id: str | None
    payload: Mapping[str, JsonValue]
    recorded_at: str
    prev_hash: str
    hash: str

    def expected_hash(self) -> str:
        """Recompute the hash from the fields; it differs from ``hash`` if either was changed."""
        return event_hash(
            case_id=self.case_id,
            seq=self.seq,
            type=self.type,
            schema_version=self.schema_version,
            step_id=self.step_id,
            payload=self.payload,
            recorded_at=self.recorded_at,
            prev_hash=self.prev_hash,
        )


def event_hash(
    *,
    case_id: str,
    seq: int,
    type: str,
    schema_version: int,
    step_id: str | None,
    payload: Mapping[str, JsonValue],
    recorded_at: str,
    prev_hash: str,
) -> str:
    """Return the SHA-256 of the canonical JSON of every envelope field except the hash itself."""
    return sha256_hex(
        canonical_json(
            {
                "case_id": case_id,
                "payload": dict(payload),
                "prev_hash": prev_hash,
                "recorded_at": recorded_at,
                "schema_version": schema_version,
                "seq": seq,
                "step_id": step_id,
                "type": type,
            }
        )
    )


def first_broken_link(events: list[Event]) -> int | None:
    """Return the ``seq`` of the first event that breaks the chain, or None if it is intact.

    An event breaks the chain if its stored hash is not the hash of its fields, if its ``prev_hash``
    is not its predecessor's hash, or if its ``seq`` is not one more than its predecessor's.
    """
    previous_hash, previous_seq = GENESIS_HASH, 0
    for event in events:
        if (
            event.seq != previous_seq + 1
            or event.prev_hash != previous_hash
            or event.hash != event.expected_hash()
        ):
            return event.seq
        previous_hash, previous_seq = event.hash, event.seq
    return None
