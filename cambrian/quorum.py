"""Quorum Sensing — population diversity auto-regulation.

Inspired by bacterial quorum sensing, this module monitors the diversity of
the evolution population (via Shannon entropy of fitness/strategy distributions)
and auto-adjusts ``mutation_rate`` and ``elitism`` to maintain a healthy balance
between exploration and exploitation.

Rules
-----
- **High entropy** (diverse population) → reduce mutation rate (exploit good solutions).
- **Low entropy** (converged population) → increase mutation rate (explore).
- Entropy is computed over discretised fitness bins.

Usage::

    from cambrian.quorum import QuorumSensor

    sensor = QuorumSensor(n_bins=10, target_entropy=0.7)
    new_rate = sensor.update(population)   # returns adjusted mutation_rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent

logger = get_logger(__name__)


# ── QuorumState ────────────────────────────────────────────────────────────────


@dataclass
class QuorumState:
    """Snapshot of the quorum sensor at a single generation.

    Attributes:
        generation: The generation number this state was recorded at.
        entropy: Shannon entropy of the population fitness distribution.
        mutation_rate: The mutation rate recommended by the sensor.
        elite_n: The elite count recommended by the sensor.
        population_size: Number of agents in the population.
        diversity_ratio: Normalised diversity score (0 = no diversity, 1 = max).
    """

    generation: int
    entropy: float
    mutation_rate: float
    elite_n: int
    population_size: int
    diversity_ratio: float = 0.0


# ── QuorumSensor ───────────────────────────────────────────────────────────────


class QuorumSensor:
    """Monitors population diversity and auto-adjusts mutation parameters.

    Args:
        n_bins: Number of bins for discretising the fitness histogram.
            More bins = finer diversity measurement. Default 10.
        target_entropy: Desired normalised entropy (0–1). Sensor tries to
            keep actual entropy close to this value. Default 0.6.
        min_mutation_rate: Hard lower bound on mutation rate. Default 0.2.
        max_mutation_rate: Hard upper bound on mutation rate. Default 1.0.
        min_elite_ratio: Hard lower bound on elite fraction. Default 0.1.
        max_elite_ratio: Hard upper bound on elite fraction. Default 0.5.
        lr: Learning rate for mutation rate adjustment. Default 0.05.
            Higher values = faster adaptation.
        history_size: Number of past states to retain. Default 50.
    """

    def __init__(
        self,
        n_bins: int = 10,
        target_entropy: float = 0.6,
        min_mutation_rate: float = 0.2,
        max_mutation_rate: float = 1.0,
        min_elite_ratio: float = 0.1,
        max_elite_ratio: float = 0.5,
        lr: float = 0.05,
        history_size: int = 50,
    ) -> None:
        self._n_bins = max(2, n_bins)
        self._target = max(0.0, min(1.0, target_entropy))
        self._min_mut = min_mutation_rate
        self._max_mut = max_mutation_rate
        self._min_elite_ratio = min_elite_ratio
        self._max_elite_ratio = max_elite_ratio
        self._lr = lr
        self._history_size = history_size
        self._history: list[QuorumState] = []
        self._current_mut_rate: float = 0.8  # default starting rate
        self._generation: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def history(self) -> list[QuorumState]:
        """History of quorum states (most recent last)."""
        return list(self._history)

    @property
    def current_mutation_rate(self) -> float:
        """Most recently computed mutation rate."""
        return self._current_mut_rate

    def update(
        self,
        population: "list[Agent]",
        current_mutation_rate: "float | None" = None,
        population_size: "int | None" = None,
    ) -> QuorumState:
        """Compute diversity metrics and return recommended parameters.

        Args:
            population: Current list of agents (must have fitness set).
            current_mutation_rate: Seed rate for this update. If ``None``,
                uses the rate from the previous :meth:`update`.
            population_size: Override for population size (used for elite_n
                calculation). If ``None``, uses ``len(population)``.

        Returns:
            A :class:`QuorumState` with recommended mutation_rate and elite_n.
        """
        if current_mutation_rate is not None:
            self._current_mut_rate = current_mutation_rate

        pop_size = population_size if population_size is not None else len(population)
        fitnesses = [
            a.fitness for a in population if a.fitness is not None
        ]

        entropy = self.compute_entropy(fitnesses)
        diversity_ratio = entropy  # already normalised

        # Adjust mutation rate: if below target, increase; if above, decrease
        delta = self._target - entropy
        new_rate = self._current_mut_rate + self._lr * delta
        new_rate = max(self._min_mut, min(self._max_mut, new_rate))
        self._current_mut_rate = new_rate

        # Adjust elite count: low diversity → fewer elites (more variety in offspring)
        elite_ratio = self._min_elite_ratio + (1.0 - diversity_ratio) * (
            self._max_elite_ratio - self._min_elite_ratio
        )
        # Invert: low diversity → more mutation, so fewer elites
        elite_ratio = self._max_elite_ratio - (1.0 - diversity_ratio) * (
            self._max_elite_ratio - self._min_elite_ratio
        )
        elite_ratio = max(self._min_elite_ratio, min(self._max_elite_ratio, elite_ratio))
        elite_n = max(1, int(pop_size * elite_ratio))

        state = QuorumState(
            generation=self._generation,
            entropy=entropy,
            mutation_rate=new_rate,
            elite_n=elite_n,
            population_size=pop_size,
            diversity_ratio=diversity_ratio,
        )
        self._generation += 1
        self._history.append(state)
        # Trim history
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        logger.debug(
            "QuorumSensor gen=%d entropy=%.4f mut_rate=%.4f elite_n=%d",
            state.generation,
            entropy,
            new_rate,
            elite_n,
        )
        return state

    # ── Entropy calculation ────────────────────────────────────────────────────

    def compute_entropy(self, fitnesses: list[float]) -> float:
        """Compute normalised Shannon entropy of a fitness distribution.

        Bins the fitness values into :attr:`n_bins` uniform bins, then computes
        Shannon entropy H = -Σ p_i log2(p_i), normalised to [0, 1] by dividing
        by log2(n_bins).

        Args:
            fitnesses: List of fitness values (any float, will be clamped to [0, 1]).

        Returns:
            Normalised entropy in [0, 1]. Returns 0.0 if fewer than 2 values.
        """
        if len(fitnesses) < 2:
            return 0.0

        # Clamp to [0, 1] and bin
        clamped = [max(0.0, min(1.0, f)) for f in fitnesses]
        bins: list[int] = [0] * self._n_bins

        for f in clamped:
            idx = min(self._n_bins - 1, int(f * self._n_bins))
            bins[idx] += 1

        n = len(clamped)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        max_entropy = math.log2(self._n_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def stagnation_detected(self, window: int = 5, threshold: float = 0.01) -> bool:
        """Return True if entropy has been below threshold for *window* generations.

        Args:
            window: Number of recent generations to check. Default 5.
            threshold: Minimum normalised entropy. Default 0.01.

        Returns:
            True if the last *window* states all have entropy < *threshold*.
        """
        recent = self._history[-window:]
        if len(recent) < window:
            return False
        return all(s.entropy < threshold for s in recent)

    def should_inject_diversity(
        self, low_threshold: float = 0.2, window: int = 3
    ) -> bool:
        """Return True if entropy has been persistently low and diversity injection is warranted.

        Args:
            low_threshold: Entropy below which diversity is considered low.
            window: Number of consecutive low-entropy generations to trigger.

        Returns:
            True if the last *window* states all have entropy < *low_threshold*.
        """
        recent = self._history[-window:]
        if len(recent) < window:
            return False
        return all(s.entropy < low_threshold for s in recent)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of quorum sensor state."""
        if not self._history:
            return {"generations": 0}
        last = self._history[-1]
        entropies = [s.entropy for s in self._history]
        return {
            "generations": len(self._history),
            "current_mutation_rate": last.mutation_rate,
            "current_elite_n": last.elite_n,
            "current_entropy": last.entropy,
            "mean_entropy": sum(entropies) / len(entropies),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "stagnation_detected": self.stagnation_detected(),
        }
