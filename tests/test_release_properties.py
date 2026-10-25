"""Property tests: no sequence of review events releases a dossier without a matching approval.

Hypothesis drives random sequences of revisions, decisions and release attempts at the fold:
decisions on the current dossier, on stale ones and on hashes that never existed, by people and by
agents, with valid and invalid verdicts. Each attempt is judged twice, by the fold and by
``release_is_justified`` below, a deliberately plain re-statement of the rule written without
reusing the fold's code. They must agree both ways: the fold never allows a release the rule
forbids (safety) and never refuses one the rule allows (no rejecting legitimate work).
"""

from __future__ import annotations

import hashlib
import json

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from casefile.canonical import JsonValue
from casefile.events import Event
from casefile.state import CaseState, EventType, FoldError, apply

DOSSIERS: list[JsonValue] = ["first draft", "second draft", {"summary": "pend", "missing": [1]}]
ACTORS = ["human:ana", "human:ben", "agent:drafter", "human:", ""]
VERDICTS = ["approve", "reject", "request_info", "rubber-stamp"]
NEVER_A_DOSSIER = hashlib.sha256(b"never shown to anyone").hexdigest()


def digest(dossier: JsonValue) -> str:
    """Hash a dossier the way the rule states it: SHA-256 of sorted, compact JSON."""
    text = json.dumps(dossier, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def release_is_justified(history: list[Event], named: str) -> bool:
    """Decide from scratch whether releasing the dossier hashed ``named`` is allowed now.

    Allowed when the case is not already released, ``named`` is the current dossier's hash, and the
    most recent valid decision by a person on that exact hash is an approval made after the dossier
    last changed.
    """
    current: str | None = None
    changed_at = 0
    latest: dict[str, tuple[str, int]] = {}
    for position, event in enumerate(history):
        payload = event.payload
        if event.type == EventType.CASE_RELEASED:
            return False
        if event.type == EventType.CASE_COMPLETED:
            current, changed_at = digest(payload["outcome"]), position
        elif event.type == EventType.DOSSIER_REVISED:
            current, changed_at = digest(payload["dossier"]), position
        elif event.type == EventType.REVIEWER_DECISION:
            latest[str(payload["dossier_hash"])] = (str(payload["decision"]), position)
    decision, made_at = latest.get(named, ("none", -1))
    return named == current and decision == "approve" and made_at > changed_at


def event(seq: int, kind: EventType, payload: dict[str, JsonValue]) -> Event:
    """Build an event as the fold sees it; ``apply`` does not check hashes, ``fold`` does."""
    return Event("c", seq, kind, 1, None, payload, "", "", "")


class ReviewMachine(RuleBasedStateMachine):
    """A completed case under random review traffic; only accepted events enter its history."""

    def __init__(self) -> None:
        """Start from a case whose automated run produced the first dossier."""
        super().__init__()
        self.history: list[Event] = []
        self.state: CaseState | None = None
        self.seen: list[str] = []
        self.offer(EventType.CASE_OPENED, {"inputs": {}})
        self.offer(EventType.CASE_COMPLETED, {"outcome": DOSSIERS[0]})

    def offer(self, kind: EventType, payload: dict[str, JsonValue]) -> bool:
        """Apply one event if the fold accepts it; return whether it did."""
        candidate = event(len(self.history) + 1, kind, payload)
        try:
            self.state = apply(self.state, candidate)
        except FoldError:
            return False
        self.history.append(candidate)
        if self.state.dossier_hash is not None:
            self.seen.append(self.state.dossier_hash)
        return True

    def pick_hash(self, which: str) -> str:
        """The current dossier's hash, one seen earlier, or one that never existed."""
        assert self.state is not None
        if which == "stale" and self.seen:
            return self.seen[0]
        if which == "never":
            return NEVER_A_DOSSIER
        return self.state.dossier_hash or NEVER_A_DOSSIER

    @rule(dossier=st.sampled_from(DOSSIERS), editor=st.sampled_from(ACTORS))
    def revise(self, dossier: JsonValue, editor: str) -> None:
        """Someone (possibly not a person) replaces the dossier."""
        self.offer(EventType.DOSSIER_REVISED, {"dossier": dossier, "editor": editor})

    @rule(
        verdict=st.sampled_from(VERDICTS),
        reviewer=st.sampled_from(ACTORS),
        which=st.sampled_from(["current", "stale", "never"]),
    )
    def decide(self, verdict: str, reviewer: str, which: str) -> None:
        """Someone records a verdict on the current, a stale or a made-up dossier hash."""
        payload: dict[str, JsonValue] = {
            "decision": verdict,
            "dossier_hash": self.pick_hash(which),
            "reviewer": reviewer,
        }
        accepted = self.offer(EventType.REVIEWER_DECISION, payload)
        is_valid = (
            verdict != "rubber-stamp" and reviewer.startswith("human:") and reviewer != "human:"
        )
        assert self.state is not None
        assert accepted == (is_valid and not self.state.released)

    @rule(which=st.sampled_from(["current", "stale", "never"]))
    def release(self, which: str) -> None:
        """Try to release a dossier; the fold must accept exactly when the rule allows it."""
        named = self.pick_hash(which)
        expected = release_is_justified(self.history, named)
        payload: dict[str, JsonValue] = {"dossier_hash": named, "releaser": "human:ben"}
        assert self.offer(EventType.CASE_RELEASED, payload) == expected

    @invariant()
    def a_released_case_was_released_by_a_justified_event(self) -> None:
        """Once released, the release event was justified by the history before it."""
        assert self.state is not None
        if self.state.released:
            release = self.history[-1]
            assert release.type == EventType.CASE_RELEASED
            assert release_is_justified(self.history[:-1], str(release.payload["dossier_hash"]))

    @invariant()
    def the_fold_and_the_rule_agree_on_approval(self) -> None:
        """While unreleased, the fold's ``approved`` matches the rule's verdict."""
        assert self.state is not None
        current = self.state.dossier_hash
        assert current is not None
        if not self.state.released:
            assert self.state.approved == release_is_justified(self.history, current)


ReviewMachine.TestCase.settings = settings(max_examples=300, stateful_step_count=25, deadline=None)
test_review_properties = ReviewMachine.TestCase
