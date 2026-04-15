# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Quorum sensing — Shannon entropy-based adaptive mutation control.

In social insect colonies, *quorum sensing* is the ability of a population to
collectively detect when a threshold has been reached and switch behaviour
accordingly.  Cambrian adapts this metaphor to evolutionary dynamics:

The population's **diversity** is measured by the Shannon entropy of the fitness
distribution.  When entropy is low (most agents cluster around the same fitness),
the population has *converged* — like ants all following the same pheromone trail.
The quorum sensor responds by **increasing** the mutation rate to reintroduce
exploration.  When entropy is high (scores are spread across the range) the sensor
**decreases** the mutation rate to allow exploitation of promising regions.

Usage::

    from cambrian.quorum import QuorumSensor

    sensor = QuorumSensor(low_entropy_threshold=1.0, high_entropy_threshold=2.5)

    for gen in range(1, n_generations + 1):
        scores = [a.fitness or 0.0 for a in population]
        new_rate = sensor.update(scores, current_mutation_rate)
        engine.mutation_rate = new_rate
        population = engine._next_generation(population, task)
"""

from __future__ import annotations

import math

from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QuorumSensor
# ─────────────────────────────────────────────────────────────────────────────


class QuorumSensor:
    """Adapts mutation rate based on population fitness diversity.

    Uses Shannon entropy of a discretised fitness histogram to measure
    diversity.  Mutation rate is nudged up when diversity is low (population
    has converged) and nudged down when diversity is high (exploration already
    happening).

    Args:
        low_entropy_threshold: Entropy below this triggers a boost.
            Default ``1.0`` (low diversity).
        high_entropy_threshold: Entropy above this triggers a reduction.
            Default ``2.5`` (high diversity).
        boost_factor: Multiplier applied to mutation rate when entropy is low.
            Default ``1.3`` (30 % boost).
        decay_factor: Multiplier applied to mutation rate when entropy is high.
            Default ``0.85`` (15 % decay).
        min_rate: Lower bound for the mutation rate. Default ``0.1``.
        max_rate: Upper bound for the mutation rate. Default ``1.0``.
        n_bins: Number of histogram bins for entropy computation. Default ``10``.
    """

    def __init__(
        self,
        low_entropy_threshold: float = 1.0,
        high_entropy_threshold: float = 2.5,
        boost_factor: float = 1.3,
        decay_factor: float = 0.85,
        min_rate: float = 0.1,
        max_rate: float = 1.0,
        n_bins: int = 10,
    ) -> None:
        if low_entropy_threshold >= high_entropy_threshold:
            raise ValueError(
                "low_entropy_threshold must be strictly less than high_entropy_threshold"
            )
        self._low_thresh = low_entropy_threshold
        self._high_thresh = high_entropy_threshold
        self._boost = boost_factor
        self._decay = decay_factor
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._n_bins = n_bins
        self._last_entropy: float | None = None
        self._history: list[tuple[float, float]] = []  # (entropy, mutation_rate)

    @property
    def last_entropy(self) -> float | None:
        """Shannon entropy computed in the most recent :meth:`update` call."""
        return self._last_entropy

    @property
    def history(self) -> list[tuple[float, float]]:
        """List of ``(entropy, mutation_rate)`` tuples, one per :meth:`update` call."""
        return list(self._history)

    def compute_entropy(self, scores: list[float]) -> float:
        """Compute Shannon entropy of the fitness distribution.

        Fitness values are discretised into :attr:`n_bins` equal-width bins
        covering ``[0.0, 1.0]``.  Empty bins contribute 0 to the sum.

        Args:
            scores: List of fitness values.  Values outside ``[0.0, 1.0]`` are
                clamped before binning.

        Returns:
            Shannon entropy in nats (log base e).  Returns ``0.0`` for
            empty or uniform populations.
        """
        if not scores:
            return 0.0

        # Build histogram
        bins = [0] * self._n_bins
        for s in scores:
            s = max(0.0, min(1.0, s))
            idx = min(int(s * self._n_bins), self._n_bins - 1)
            bins[idx] += 1

        n = len(scores)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / n
                entropy -= p * math.log(p)
        return entropy

    def update(self, scores: list[float], current_rate: float) -> float:
        """Compute new mutation rate based on population diversity.

        Args:
            scores: Current generation fitness values.
            current_rate: The mutation rate in use before this call.

        Returns:
            New mutation rate, clamped to ``[min_rate, max_rate]``.
        """
        entropy = self.compute_entropy(scores)
        self._last_entropy = entropy

        if entropy < self._low_thresh:
            # Low diversity → boost mutation to re-explore
            new_rate = current_rate * self._boost
            logger.debug(
                "QuorumSensor: low diversity (H=%.3f < %.3f) → boosting rate %.3f→%.3f",
                entropy, self._low_thresh, current_rate, new_rate,
            )
        elif entropy > self._high_thresh:
            # High diversity → reduce mutation to exploit
            new_rate = current_rate * self._decay
            logger.debug(
                "QuorumSensor: high diversity (H=%.3f > %.3f) → decaying rate %.3f→%.3f",
                entropy, self._high_thresh, current_rate, new_rate,
            )
        else:
            new_rate = current_rate
            logger.debug(
                "QuorumSensor: balanced diversity (H=%.3f) → keeping rate %.3f",
                entropy, current_rate,
            )

        new_rate = max(self._min_rate, min(self._max_rate, new_rate))
        self._history.append((entropy, new_rate))
        return new_rate

    def reset(self) -> None:
        """Clear history and last_entropy."""
        self._last_entropy = None
        self._history.clear()
