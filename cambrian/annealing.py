# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Simulated Annealing — Technique 60.

Simulated Annealing (Kirkpatrick 1983) accepts worse solutions with a
probability that decreases as a "temperature" schedule cools.  Applied to
agent evolution, this prevents premature convergence: early in evolution
the system freely accepts worse mutations (exploration); later it becomes
selective (exploitation).

Components
----------
AnnealingSchedule
    Computes the current temperature T(t) from a given ``schedule_type``:
    * ``"linear"``  — T = T_max - (T_max - T_min) * t/n_steps
    * ``"exponential"``  — T = T_max * (T_min/T_max)^(t/n_steps)
    * ``"cosine"``  — T = T_min + 0.5*(T_max-T_min)*(1+cos(π*t/n_steps))

AnnealingSelector
    Wraps a population update step.  Given the current agent and a mutant
    candidate, it accepts the mutant if it is better **or** with probability
    exp(-(fitness_curr - fitness_new) / T).  Uses ``AnnealingSchedule``
    to get T at the current step.

Usage::

    from cambrian.annealing import AnnealingSchedule, AnnealingSelector

    schedule = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=100)
    selector = AnnealingSelector(schedule=schedule)
    accepted = selector.step(current_agent, candidate_agent)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ScheduleType = Literal["linear", "exponential", "cosine"]


# ---------------------------------------------------------------------------
# AnnealingSchedule
# ---------------------------------------------------------------------------


class AnnealingSchedule:
    """Compute the annealing temperature at step t.

    Parameters
    ----------
    T_max:
        Initial (hot) temperature (default 2.0).
    T_min:
        Final (cold) temperature, must be > 0 (default 0.01).
    n_steps:
        Total number of steps over which to cool (default 100).
    schedule_type:
        Cooling curve: ``"linear"`` | ``"exponential"`` | ``"cosine"``.
    """

    def __init__(
        self,
        T_max: float = 2.0,
        T_min: float = 0.01,
        n_steps: int = 100,
        schedule_type: ScheduleType = "exponential",
    ) -> None:
        if T_min <= 0:
            raise ValueError("T_min must be > 0")
        if T_max <= T_min:
            raise ValueError("T_max must be > T_min")
        if n_steps < 1:
            raise ValueError("n_steps must be ≥ 1")
        self.T_max = T_max
        self.T_min = T_min
        self.n_steps = n_steps
        self.schedule_type = schedule_type

    def temperature(self, t: int) -> float:
        """Return the temperature at step *t* ∈ [0, n_steps]."""
        t = max(0, min(t, self.n_steps))
        frac = t / self.n_steps if self.n_steps > 0 else 1.0

        if self.schedule_type == "linear":
            return self.T_max - (self.T_max - self.T_min) * frac

        if self.schedule_type == "exponential":
            return float(self.T_max * (self.T_min / self.T_max) ** frac)

        # cosine
        return self.T_min + 0.5 * (self.T_max - self.T_min) * (
            1.0 + math.cos(math.pi * frac)
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnnealingEvent:
    """Record of one acceptance/rejection decision."""

    step: int
    temperature: float
    current_fitness: float
    candidate_fitness: float
    accepted: bool
    acceptance_prob: float


# ---------------------------------------------------------------------------
# AnnealingSelector
# ---------------------------------------------------------------------------


class AnnealingSelector:
    """Accept or reject a candidate mutation using SA acceptance criterion.

    Parameters
    ----------
    schedule:
        ``AnnealingSchedule`` that provides the temperature at each step.
    rng:
        Optional ``random.Random`` instance for reproducibility.
    """

    def __init__(
        self,
        schedule: AnnealingSchedule,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._schedule = schedule
        self._rng = rng or random.Random()
        self._step: int = 0
        self._history: list[AnnealingEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """Current annealing step."""
        return self._step

    @property
    def current_temperature(self) -> float:
        """Temperature at the current step."""
        return self._schedule.temperature(self._step)

    @property
    def history(self) -> list[AnnealingEvent]:
        """Return a copy of all acceptance events."""
        return list(self._history)

    def step(
        self,
        current_fitness: float,
        candidate_fitness: float,
    ) -> bool:
        """Decide whether to accept the candidate.

        Returns True (accept) or False (reject).  Always accepts
        improvements.  Accepts regressions with probability
        exp(-delta / T) where delta = current - candidate (always ≥ 0).
        """
        T = self._schedule.temperature(self._step)
        self._step += 1

        if candidate_fitness >= current_fitness:
            accepted = True
            prob = 1.0
        else:
            delta = current_fitness - candidate_fitness
            prob = math.exp(-delta / max(T, 1e-10))
            accepted = self._rng.random() < prob

        self._history.append(
            AnnealingEvent(
                step=self._step - 1,
                temperature=T,
                current_fitness=current_fitness,
                candidate_fitness=candidate_fitness,
                accepted=accepted,
                acceptance_prob=prob,
            )
        )
        return accepted

    def reset(self) -> None:
        """Reset step counter and history."""
        self._step = 0
        self._history.clear()

    def acceptance_rate(self) -> float:
        """Return the fraction of steps that resulted in acceptance."""
        if not self._history:
            return 0.0
        return sum(1 for e in self._history if e.accepted) / len(self._history)
