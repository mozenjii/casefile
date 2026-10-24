"""Reading and writing one case: the fold judges every event before the log stores it.

The log accepts any well-formed event; it knows nothing about what events mean. If code appended an
event the fold would refuse (a step after completion, a release without approval), the case could
never be read again. So every writer goes through ``append_checked``: it applies the event to the
current state first, on a provisional copy, and writes only if the fold accepts it. The fold stays
the single definition of what a valid history is, and no invalid event reaches the disk.
"""

from __future__ import annotations

from collections.abc import Mapping

from casefile.canonical import JsonValue
from casefile.events import Event
from casefile.log import EventLog
from casefile.state import SCHEMA_VERSION, CaseState, EventType, apply, fold


def load(log: EventLog, case_id: str) -> CaseState | None:
    """Fold a case from the log, or return None if it has no events yet."""
    events = log.read(case_id)
    return fold(events) if events else None


def append_checked(
    log: EventLog,
    case_id: str,
    state: CaseState | None,
    kind: EventType,
    payload: Mapping[str, JsonValue],
    step_id: str | None = None,
) -> CaseState:
    """Append one event if the fold accepts it after ``state``, and return the new state.

    Raises ``FoldError`` without writing anything if the event is not a valid next step, and
    ``AppendConflict`` if the log has moved on since ``state`` was read.
    """
    last_seq = 0 if state is None else state.last_seq
    provisional = Event(
        case_id=case_id,
        seq=last_seq + 1,
        type=kind,
        schema_version=SCHEMA_VERSION,
        step_id=step_id,
        payload=dict(payload),
        recorded_at="",
        prev_hash="",
        hash="",
    )
    apply(state, provisional)
    event = log.append(case_id, kind, payload, step_id=step_id, expected_seq=last_seq)
    return apply(state, event)
