# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Neuromodulation — Technique 66.

Inspired by neuromodulatory systems (dopamine, serotonin, acetylcholine,
noradrenaline) that globally regulate learning and plasticity in the brain.

Each *neuromodulator* monitors a population-level signal (fitness trend,
diversity, surprise) and emits a scalar *modulation level* ∈ [0, 1].
A ``NeuromodulatorBank`` combines multiple modulators and computes
an aggregate effect on two downstream hyperparameters:

* **mutation_rate** — how aggressively agents are mutated.
* **selection_pressure** — how strictly only top performers are kept
  (higher → fewer survivors).

Components
----------
NeuromodulatorBase
    Abstract base for all modulators.

DopamineModulator
    Increases when average fitness is rising (reward signal).
    High dopamine → reduce mutation_rate (exploit good solutions).

SerotoninModulator
    Tracks population diversity (distinct prompt fingerprints).
    Low diversity (groupthink) → high serotonin → increase mutation_rate.

AcetylcholineModulator
    Tracks surprise: large fitness variance → high Ach → lower
    selection_pressure to preserve diverse candidates.

NoradrenalineModulator
    Novelty / stagnation detector. If best fitness hasn't improved in
    ``patience`` generations, noradrenaline spikes → raise mutation_rate.

NeuromodulatorBank
    Aggregates all four modulators and exposes ``modulate()`` which
    returns a ``NeuroState`` with the adjusted hyperparameter values.

Usage::

    from cambrian.neuromodulation import NeuromodulatorBank

    bank = NeuromodulatorBank(base_mutation_rate=0.3, base_selection_pressure=0.5)
    state = bank.modulate(population, generation=1)
    print(state.mutation_rate, state.selection_pressure)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from cambrian.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _diversity(population: list[Agent]) -> float:
    """Word-set diversity: fraction of distinct prompts (approx)."""
    fingerprints: set[frozenset[str]] = set()
    for agent in population:
        words = frozenset(re.split(r"\W+", agent.genome.system_prompt.lower())) - {""}
        fingerprints.add(words)
    if not population:
        return 0.0
    return len(fingerprints) / len(population)


# ---------------------------------------------------------------------------
# NeuroState — output of NeuromodulatorBank
# ---------------------------------------------------------------------------


@dataclass
class NeuroState:
    """Current neuromodulatory state."""

    mutation_rate: float
    selection_pressure: float
    dopamine: float = 0.0
    serotonin: float = 0.0
    acetylcholine: float = 0.0
    noradrenaline: float = 0.0
    generation: int = 0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class NeuromodulatorBase(ABC):
    """Abstract base class for all neuromodulators."""

    @abstractmethod
    def level(self, population: list[Agent], generation: int) -> float:
        """Return modulation level ∈ [0, 1]."""


# ---------------------------------------------------------------------------
# Dopamine — reward / fitness improvement
# ---------------------------------------------------------------------------


class DopamineModulator(NeuromodulatorBase):
    """High dopamine when fitness is improving → reduce mutation rate (exploit).

    Parameters
    ----------
    window:
        Number of generations to track for trend (default 3).
    """

    def __init__(self, window: int = 3) -> None:
        self._window = window
        self._history: list[float] = []

    def level(self, population: list[Agent], generation: int) -> float:
        fitnesses = [a.fitness for a in population if a.fitness is not None]
        avg = _mean(fitnesses)
        self._history.append(avg)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < 2:
            return 0.5
        # Positive trend → high dopamine
        trend = self._history[-1] - self._history[0]
        # Normalise: trend in [-1, 1] → level in [0, 1]
        return float(max(0.0, min(1.0, 0.5 + trend)))


# ---------------------------------------------------------------------------
# Serotonin — diversity / groupthink prevention
# ---------------------------------------------------------------------------


class SerotoninModulator(NeuromodulatorBase):
    """High serotonin when diversity is LOW → increase mutation to break groupthink.

    Parameters
    ----------
    diversity_floor:
        Diversity fraction below which serotonin is maximal (default 0.3).
    """

    def __init__(self, diversity_floor: float = 0.3) -> None:
        self._floor = diversity_floor

    def level(self, population: list[Agent], generation: int) -> float:
        d = _diversity(population)
        if d <= self._floor:
            return 1.0
        return float(max(0.0, 1.0 - (d - self._floor) / (1.0 - self._floor)))


# ---------------------------------------------------------------------------
# Acetylcholine — surprise / variance modulator
# ---------------------------------------------------------------------------


class AcetylcholineModulator(NeuromodulatorBase):
    """High Ach when fitness variance is high → lower selection pressure.

    Parameters
    ----------
    variance_cap:
        Maximum expected variance for normalisation (default 0.1).
    """

    def __init__(self, variance_cap: float = 0.1) -> None:
        self._cap = variance_cap

    def level(self, population: list[Agent], generation: int) -> float:
        fitnesses = [a.fitness for a in population if a.fitness is not None]
        var = _variance(fitnesses)
        return float(min(1.0, var / self._cap))


# ---------------------------------------------------------------------------
# Noradrenaline — stagnation / novelty
# ---------------------------------------------------------------------------


class NoradrenalineModulator(NeuromodulatorBase):
    """Spikes when best fitness stagnates → raise mutation (explore).

    Parameters
    ----------
    patience:
        Generations without improvement before spike (default 3).
    epsilon:
        Minimum improvement to reset stagnation counter (default 1e-4).
    """

    def __init__(self, patience: int = 3, epsilon: float = 1e-4) -> None:
        self._patience = patience
        self._epsilon = epsilon
        self._best: Optional[float] = None
        self._stagnant_gens: int = 0

    def level(self, population: list[Agent], generation: int) -> float:
        fitnesses = [a.fitness for a in population if a.fitness is not None]
        current_best = max(fitnesses) if fitnesses else 0.0
        if self._best is None or current_best >= self._best + self._epsilon:
            self._best = current_best
            self._stagnant_gens = 0
        else:
            self._stagnant_gens += 1
        if self._stagnant_gens >= self._patience:
            return 1.0
        return float(self._stagnant_gens / max(1, self._patience))


# ---------------------------------------------------------------------------
# NeuromodulatorBank
# ---------------------------------------------------------------------------


class NeuromodulatorBank:
    """Aggregates all four modulators and computes adjusted hyperparameters.

    Downstream effects
    ------------------
    mutation_rate
        Increased by serotonin (diversity loss) and noradrenaline (stagnation).
        Decreased by dopamine (fitness improving).

    selection_pressure
        Decreased by acetylcholine (high surprise/variance).
        Increased by dopamine (reward consolidation).

    Parameters
    ----------
    base_mutation_rate:
        Baseline mutation rate before modulation (default 0.2).
    base_selection_pressure:
        Baseline selection pressure before modulation (default 0.5).
    mr_range:
        Maximum deviation of mutation_rate from baseline (default 0.3).
    sp_range:
        Maximum deviation of selection_pressure from baseline (default 0.3).
    """

    def __init__(
        self,
        base_mutation_rate: float = 0.2,
        base_selection_pressure: float = 0.5,
        mr_range: float = 0.3,
        sp_range: float = 0.3,
    ) -> None:
        self._base_mr = base_mutation_rate
        self._base_sp = base_selection_pressure
        self._mr_range = mr_range
        self._sp_range = sp_range
        self._dopamine = DopamineModulator()
        self._serotonin = SerotoninModulator()
        self._acetylcholine = AcetylcholineModulator()
        self._noradrenaline = NoradrenalineModulator()
        self._history: list[NeuroState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[NeuroState]:
        """Return a copy of all NeuroState records."""
        return list(self._history)

    def modulate(self, population: list[Agent], generation: int = 0) -> NeuroState:
        """Compute the current neuromodulatory state given the population.

        Returns
        -------
        NeuroState
            Adjusted mutation_rate and selection_pressure plus raw modulator
            levels for inspection.
        """
        dopa = self._dopamine.level(population, generation)
        sero = self._serotonin.level(population, generation)
        ach = self._acetylcholine.level(population, generation)
        nora = self._noradrenaline.level(population, generation)

        # mutation_rate: boosted by serotonin + noradrenaline, reduced by dopamine
        mr_delta = self._mr_range * (0.5 * sero + 0.5 * nora - dopa * 0.5)
        mr = float(max(0.0, min(1.0, self._base_mr + mr_delta)))

        # selection_pressure: boosted by dopamine, reduced by acetylcholine
        sp_delta = self._sp_range * (0.5 * dopa - 0.5 * ach)
        sp = float(max(0.0, min(1.0, self._base_sp + sp_delta)))

        state = NeuroState(
            mutation_rate=mr,
            selection_pressure=sp,
            dopamine=dopa,
            serotonin=sero,
            acetylcholine=ach,
            noradrenaline=nora,
            generation=generation,
        )
        self._history.append(state)
        return state

    def reset(self) -> None:
        """Reset all modulator state and history."""
        self._dopamine = DopamineModulator()
        self._serotonin = SerotoninModulator()
        self._acetylcholine = AcetylcholineModulator()
        self._noradrenaline = NoradrenalineModulator()
        self._history.clear()
