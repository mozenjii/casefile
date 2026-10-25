"""Named crash points and the fault plans that act on them.

The executor calls ``faults.hit(point)`` at each place where a crash would leave the log in an
interesting half-done state. Production passes ``NoFaults``. Tests pass something else:

* ``CrashAt`` ends the process on the n-th hit of one point with ``os._exit``: no ``finally``
  blocks, no buffered writes flushed, the nearest thing to a power cut a test can produce;
* ``RaiseAt`` raises ``SimulatedCrash`` instead, for fast in-process tests of the same paths;
* ``HitRecorder`` counts every hit, so a clean run lists exactly which crashes are worth trying.

The plan is passed in as an object, never read from the environment, so no test switch can leak
into a real run.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

CRASH_EXIT_CODE = 137
"""The exit status ``CrashAt`` uses, the same one a process killed by SIGKILL reports on Linux."""


class CrashPoint(StrEnum):
    """Every place the executor offers a crash. The design document explains each one."""

    BEFORE_STEP = "before_step"
    AFTER_CALL_STARTED = "after_call_started"
    AFTER_CALL_BEFORE_RECORD = "after_call_before_record"
    AFTER_RECORD = "after_record"
    BEFORE_STEP_COMPLETED = "before_step_completed"
    AFTER_INTENT_BEFORE_SEND = "after_intent_before_send"
    AFTER_SEND_BEFORE_CONFIRM = "after_send_before_confirm"
    BEFORE_CASE_COMPLETED = "before_case_completed"


class Faults(Protocol):
    """Something the executor tells about each crash point it passes."""

    def hit(self, point: CrashPoint) -> None:
        """React to passing ``point``: usually nothing, sometimes end the run."""
        ...


class NoFaults:
    """The production plan: passing a crash point does nothing."""

    def hit(self, point: CrashPoint) -> None:
        """Do nothing."""


class SimulatedCrash(BaseException):
    """Raised by ``RaiseAt``; a ``BaseException`` so ``except Exception`` cannot swallow it."""


@dataclass
class HitRecorder:
    """Record every hit in order, as ``(point, occurrence)`` with occurrences counted from 1."""

    hits: list[tuple[CrashPoint, int]] = field(default_factory=list[tuple[CrashPoint, int]])

    def hit(self, point: CrashPoint) -> None:
        """Append this hit with its occurrence number for its point."""
        occurrence = 1 + sum(1 for seen, _ in self.hits if seen is point)
        self.hits.append((point, occurrence))


@dataclass
class _Countdown:
    """Count hits of one point and report True exactly on the chosen occurrence."""

    point: CrashPoint
    occurrence: int
    seen: int = 0

    def due(self, point: CrashPoint) -> bool:
        """Count ``point`` if it is ours and say whether this is the hit to crash on."""
        if point is not self.point:
            return False
        self.seen += 1
        return self.seen == self.occurrence


@dataclass
class RaiseAt:
    """Raise ``SimulatedCrash`` on the ``occurrence``-th hit of ``point``."""

    point: CrashPoint
    occurrence: int = 1
    _countdown: _Countdown = field(init=False)

    def __post_init__(self) -> None:
        """Start counting from zero."""
        self._countdown = _Countdown(self.point, self.occurrence)

    def hit(self, point: CrashPoint) -> None:
        """Raise when the chosen hit arrives."""
        if self._countdown.due(point):
            raise SimulatedCrash(f"{point} #{self.occurrence}")


@dataclass
class CrashAt:
    """End the process with ``CRASH_EXIT_CODE`` on the ``occurrence``-th hit of ``point``.

    ``exit`` is injectable only so a test can prove the right hit triggers it without dying.
    """

    point: CrashPoint
    occurrence: int = 1
    exit: Callable[[int], object] = os._exit
    _countdown: _Countdown = field(init=False)

    def __post_init__(self) -> None:
        """Start counting from zero."""
        self._countdown = _Countdown(self.point, self.occurrence)

    def hit(self, point: CrashPoint) -> None:
        """Exit immediately when the chosen hit arrives."""
        if self._countdown.due(point):
            self.exit(CRASH_EXIT_CODE)
