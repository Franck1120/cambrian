"""Zeitgeber — Technique 62.

A Zeitgeber (German: "time-giver") is an external cue that synchronises
a biological clock.  Here we model a synthetic circadian rhythm that
modulates the evolutionary process:

* **Exploration phase** (early in the oscillation cycle): lower
  acceptance threshold, higher mutation rate multiplier.
* **Exploitation phase** (late in the cycle): stricter selection,
  lower mutation rate.

The rhythm is a sinusoid over ``period`` generations.  This creates
natural diversity without manual scheduling.

Components
----------
ZeitgeberClock
    Tracks the generation and computes the current phase ∈ [0, 2π] and
    the resulting ``exploration_factor`` ∈ [0, 1].

ZeitgeberScheduler
    Uses the clock to adjust ``mutation_rate`` and ``acceptance_threshold``
    dynamically at each generation.

Usage::

    from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler

    clock = ZeitgeberClock(period=20)
    scheduler = ZeitgeberScheduler(clock=clock)
    mutation_rate, threshold = scheduler.tick()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ZeitgeberState:
    """State snapshot at one tick."""

    generation: int
    phase: float               # 0 .. 2π
    exploration_factor: float  # 0 = full exploitation, 1 = full exploration
    mutation_rate: float
    acceptance_threshold: float


# ---------------------------------------------------------------------------
# ZeitgeberClock
# ---------------------------------------------------------------------------


class ZeitgeberClock:
    """Synthetic circadian oscillator.

    Parameters
    ----------
    period:
        Number of generations per full oscillation (default 20).
    amplitude:
        Strength of the oscillation [0, 1] (default 0.5).
    phase_offset:
        Phase offset in radians (default 0 — exploration starts first).
    """

    def __init__(
        self,
        period: int = 20,
        amplitude: float = 0.5,
        phase_offset: float = 0.0,
    ) -> None:
        if period < 1:
            raise ValueError("period must be ≥ 1")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        self._period = period
        self._amplitude = amplitude
        self._phase_offset = phase_offset
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        """Current generation counter."""
        return self._generation

    def advance(self) -> None:
        """Increment the generation counter."""
        self._generation += 1

    def phase(self) -> float:
        """Current phase ∈ [0, 2π]."""
        return (
            2 * math.pi * self._generation / self._period + self._phase_offset
        ) % (2 * math.pi)

    def exploration_factor(self) -> float:
        """Return exploration factor ∈ [0, 1].

        Peak = 1 (maximum exploration) at phase=0, trough = 0 (exploitation)
        at phase=π.  The amplitude parameter scales the oscillation around 0.5.
        """
        # sin ∈ [-1, 1] → scale to [0.5 - amplitude/2, 0.5 + amplitude/2]
        raw = math.sin(self.phase())
        return 0.5 + self._amplitude / 2 * raw

    def reset(self) -> None:
        """Reset generation counter to 0."""
        self._generation = 0


# ---------------------------------------------------------------------------
# ZeitgeberScheduler
# ---------------------------------------------------------------------------


class ZeitgeberScheduler:
    """Dynamically adjust evolution parameters based on the Zeitgeber clock.

    Parameters
    ----------
    clock:
        ``ZeitgeberClock`` instance.
    base_mutation_rate:
        Baseline mutation rate at exploration_factor=0.5 (default 0.3).
    mutation_range:
        Amount to add/subtract from base_mutation_rate at extremes
        (default 0.2, so rate ∈ [0.1, 0.5]).
    base_threshold:
        Baseline acceptance threshold (default 0.5).
    threshold_range:
        Amount to add/subtract from base_threshold (default 0.2).
    auto_advance:
        If True, ``tick()`` automatically advances the clock (default True).
    """

    def __init__(
        self,
        clock: ZeitgeberClock,
        base_mutation_rate: float = 0.3,
        mutation_range: float = 0.2,
        base_threshold: float = 0.5,
        threshold_range: float = 0.2,
        auto_advance: bool = True,
    ) -> None:
        self._clock = clock
        self._base_mr = base_mutation_rate
        self._mr_range = mutation_range
        self._base_th = base_threshold
        self._th_range = threshold_range
        self._auto_advance = auto_advance
        self._history: list[ZeitgeberState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ZeitgeberState]:
        """Return a copy of all scheduler states."""
        return list(self._history)

    def tick(self) -> tuple[float, float]:
        """Advance the clock (if auto_advance) and return ``(mutation_rate, threshold)``.

        High exploration_factor → high mutation rate, low acceptance threshold.
        """
        ef = self._clock.exploration_factor()
        # ef ∈ [0, 1]: 1 = full exploration
        mutation_rate = max(
            0.0, min(1.0, self._base_mr + self._mr_range * (ef - 0.5) * 2)
        )
        # High exploration → lower acceptance threshold (accept more freely)
        threshold = max(
            0.0, min(1.0, self._base_th - self._th_range * (ef - 0.5) * 2)
        )

        state = ZeitgeberState(
            generation=self._clock.generation,
            phase=self._clock.phase(),
            exploration_factor=ef,
            mutation_rate=mutation_rate,
            acceptance_threshold=threshold,
        )
        self._history.append(state)

        if self._auto_advance:
            self._clock.advance()

        return mutation_rate, threshold

    def current_state(self) -> Optional[ZeitgeberState]:
        """Return the most recent state, or None if no tick yet."""
        return self._history[-1] if self._history else None
