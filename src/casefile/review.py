"""The human side of a case: show the dossier, revise it, decide on it, release it.

The dossier's hash does two jobs here:

* **It is what a decision is about.** ``show`` hands the reviewer the dossier together with its
  hash, and ``decide`` records the hash the reviewer passes back, not whatever the dossier is by
  the time the click arrives. A decision on an outdated view is recorded, and authorizes nothing.
* **It is the precondition for an edit.** ``revise`` refuses unless the editor's hash is still the
  current one, so two people editing the same view cannot silently overwrite each other.

Release is refused by the fold itself unless the latest decision on the current dossier is an
approval; this module only gives that rule a name callers can catch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from casefile.canonical import JsonValue
from casefile.cases import append_checked, load
from casefile.log import EventLog
from casefile.state import CaseState, EventType


class NotReviewable(RuntimeError):
    """The case does not exist or its automated run has not produced a dossier yet."""


class StaleView(RuntimeError):
    """The dossier changed after the editor loaded it; reload and edit the current one."""


@dataclass(frozen=True)
class ReviewView:
    """What a reviewer is shown: the dossier, its hash, and where the review stands."""

    case_id: str
    dossier: JsonValue
    dossier_hash: str
    decisions: Mapping[str, str]
    approved: bool
    released: bool


class Review:
    """Review operations on the cases in one log."""

    def __init__(self, log: EventLog) -> None:
        """Work on the cases stored in ``log``."""
        self._log = log

    def show(self, case_id: str) -> ReviewView:
        """Return the current dossier of ``case_id`` and its hash, to display and decide on."""
        return _view(self._reviewable(case_id))

    def revise(
        self, case_id: str, dossier: JsonValue, *, editor: str, seen_hash: str
    ) -> ReviewView:
        """Replace the dossier, if ``seen_hash`` is still current, and return the new view.

        Any earlier approval stays filed under the old hash, so it no longer applies.
        """
        state = self._reviewable(case_id)
        if seen_hash != state.dossier_hash:
            raise StaleView(f"case {case_id!r}: the dossier changed since it was loaded")
        payload: dict[str, JsonValue] = {"dossier": dossier, "editor": editor}
        return _view(self._append(state, EventType.DOSSIER_REVISED, payload))

    def decide(self, case_id: str, decision: str, *, reviewer: str, seen_hash: str) -> ReviewView:
        """Record ``reviewer``'s decision on the dossier they saw, and return the new view.

        ``view.approved`` is True only if the decision was an approval of the current dossier.
        """
        state = self._reviewable(case_id)
        payload: dict[str, JsonValue] = {
            "decision": decision,
            "dossier_hash": seen_hash,
            "reviewer": reviewer,
        }
        return _view(self._append(state, EventType.REVIEWER_DECISION, payload))

    def release(self, case_id: str) -> ReviewView:
        """Release the current dossier; the fold refuses unless it is approved as it stands."""
        state = self._reviewable(case_id)
        payload: dict[str, JsonValue] = {"dossier_hash": _hash(state)}
        return _view(self._append(state, EventType.CASE_RELEASED, payload))

    def _reviewable(self, case_id: str) -> CaseState:
        """Load a case that has a dossier, or raise ``NotReviewable``."""
        state = load(self._log, case_id)
        if state is None or not state.completed:
            raise NotReviewable(f"case {case_id!r} has no dossier to review yet")
        return state

    def _append(
        self, state: CaseState, kind: EventType, payload: Mapping[str, JsonValue]
    ) -> CaseState:
        """Append a review event the fold accepts, fenced on the state just loaded."""
        return append_checked(self._log, state.case_id, state, kind, payload)


def _hash(state: CaseState) -> str:
    """Return the dossier hash of a completed case."""
    if state.dossier_hash is None:
        raise NotReviewable(f"case {state.case_id!r} has no dossier")
    return state.dossier_hash


def _view(state: CaseState) -> ReviewView:
    """Build what the reviewer sees from a completed case's state."""
    return ReviewView(
        case_id=state.case_id,
        dossier=state.outcome,
        dossier_hash=_hash(state),
        decisions=state.decisions,
        approved=state.approved,
        released=state.released,
    )
