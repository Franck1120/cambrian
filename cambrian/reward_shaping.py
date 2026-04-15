# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Reward shaping for Cambrian — transform raw fitness scores to guide evolution.

Raw fitness values from evaluators are often noisy, non-stationary, or
improperly scaled for evolutionary selection.  This module provides a
collection of *reward shaping* transforms that wrap any evaluator and
post-process its output before it reaches the selection operator.

Transforms are composable: pass the output of one shaper as the input
evaluator of the next.

Available shapers
-----------------

:class:`ClipShaper`
    Clamp fitness to ``[min_val, max_val]``.

:class:`NormalisationShaper`
    Z-score or min-max normalisation over a sliding window of recent scores.

:class:`PotentialShaper`
    Potential-based reward shaping [Ng1999].  Adds a shaping bonus
    ``γ·Φ(s') − Φ(s)`` where Φ is the agent's current prompt length.

:class:`RankShaper`
    Transform raw scores to their rank within the population (avoids
    outlier fitness domination).

:class:`CuriosityShaper`
    Intrinsic motivation bonus proportional to genome novelty.

:func:`build_shaped_evaluator`
    Convenience factory: ``build_shaped_evaluator(base, "clip+normalise")``

References
----------
[Ng1999] Ng, A.Y., Harada, D., Russell, S. (1999). Policy Invariance Under
    Reward Transformations.  ICML.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable

from cambrian.agent import Agent
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

# Evaluator type alias
_Evaluator = Callable[[Agent, str], float]


# ── Base class ─────────────────────────────────────────────────────────────────


class RewardShaper:
    """Base class for reward shapers.

    Subclasses override :meth:`shape`.  All shapers are callable:
    ``shaper(agent, task) → float``.

    Args:
        base_evaluator: The upstream evaluator to wrap.
    """

    def __init__(self, base_evaluator: _Evaluator) -> None:
        self._base = base_evaluator

    def shape(self, raw_fitness: float, agent: Agent, task: str) -> float:
        """Transform *raw_fitness* and return the shaped value.

        Args:
            raw_fitness: Score from the base evaluator.
            agent: Agent being evaluated.
            task: Task description.

        Returns:
            Shaped fitness value.
        """
        return raw_fitness

    def __call__(self, agent: Agent, task: str) -> float:
        raw = self._base(agent, task)
        shaped = self.shape(raw, agent, task)
        logger.debug(
            "%s: raw=%.4f → shaped=%.4f", type(self).__name__, raw, shaped
        )
        return shaped


# ── ClipShaper ─────────────────────────────────────────────────────────────────


class ClipShaper(RewardShaper):
    """Clamp fitness to ``[min_val, max_val]``.

    Args:
        base_evaluator: Upstream evaluator.
        min_val: Lower bound (default ``0.0``).
        max_val: Upper bound (default ``1.0``).
    """

    def __init__(
        self,
        base_evaluator: _Evaluator,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        super().__init__(base_evaluator)
        self._min = min_val
        self._max = max_val

    def shape(self, raw_fitness: float, agent: Agent, task: str) -> float:
        return max(self._min, min(self._max, raw_fitness))


# ── NormalisationShaper ────────────────────────────────────────────────────────


class NormalisationShaper(RewardShaper):
    """Online normalisation of fitness scores.

    Keeps a sliding window of recent raw scores and normalises the current
    score relative to that history.

    Args:
        base_evaluator: Upstream evaluator.
        method: ``"minmax"`` (scale to [0,1]) or ``"zscore"`` (standardise).
        window_size: Number of recent scores to keep.  Default ``100``.
        eps: Small constant to avoid division by zero.  Default ``1e-8``.
    """

    def __init__(
        self,
        base_evaluator: _Evaluator,
        method: str = "minmax",
        window_size: int = 100,
        eps: float = 1e-8,
    ) -> None:
        if method not in {"minmax", "zscore"}:
            raise ValueError(f"method must be 'minmax' or 'zscore', got {method!r}")
        super().__init__(base_evaluator)
        self._method = method
        self._window: deque[float] = deque(maxlen=window_size)
        self._eps = eps

    def shape(self, raw_fitness: float, agent: Agent, task: str) -> float:
        self._window.append(raw_fitness)
        if len(self._window) < 2:
            return raw_fitness

        data = list(self._window)
        if self._method == "minmax":
            lo, hi = min(data), max(data)
            span = hi - lo
            if span < self._eps:
                return 0.5
            return (raw_fitness - lo) / span
        else:  # zscore
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std = math.sqrt(variance + self._eps)
            return (raw_fitness - mean) / std

    def reset(self) -> None:
        """Clear the normalisation window."""
        self._window.clear()


# ── PotentialShaper ────────────────────────────────────────────────────────────


class PotentialShaper(RewardShaper):
    """Potential-based reward shaping.

    Adds a shaping bonus ``γ·Φ(agent) − Φ(agent)`` where Φ maps an agent to
    its potential.  The default potential is the *negative* normalised system
    prompt length (shorter = higher potential = bonus for compression).

    Args:
        base_evaluator: Upstream evaluator.
        gamma: Discount factor (0 < γ ≤ 1).  Default ``0.99``.
        potential_fn: Callable ``(agent) → float``.  If ``None``, the default
            brevity potential is used.
        scale: Multiply the shaping bonus by this factor.  Default ``0.1``.
    """

    def __init__(
        self,
        base_evaluator: _Evaluator,
        gamma: float = 0.99,
        potential_fn: Callable[[Agent], float] | None = None,
        scale: float = 0.1,
    ) -> None:
        super().__init__(base_evaluator)
        self._gamma = gamma
        self._phi = potential_fn or self._default_potential
        self._scale = scale
        self._prev_potentials: dict[str, float] = {}

    @staticmethod
    def _default_potential(agent: Agent) -> float:
        """Negative normalised prompt length (shorter prompt = higher Φ)."""
        max_chars = 8000.0
        length = min(len(agent.genome.system_prompt), max_chars)
        return 1.0 - length / max_chars

    def shape(self, raw_fitness: float, agent: Agent, task: str) -> float:
        phi_s_prime = self._phi(agent)
        phi_s = self._prev_potentials.get(agent.id, phi_s_prime)
        self._prev_potentials[agent.id] = phi_s_prime
        bonus = self._scale * (self._gamma * phi_s_prime - phi_s)
        return raw_fitness + bonus


# ── RankShaper ─────────────────────────────────────────────────────────────────


class RankShaper:
    """Population-level rank-based fitness transform.

    Converts absolute fitness values to fractional ranks within the
    population.  This prevents superfit outliers from dominating selection
    and stabilises evolution in multi-modal landscapes.

    Unlike other shapers, ``RankShaper`` operates on the entire population at
    once via :meth:`rank_population`.  It also exposes a ``__call__`` that
    ranks single agents against a stored snapshot of recent scores.

    Args:
        window_size: History of recent scores used for single-agent ranking.
        base_evaluator: Optional upstream evaluator.
    """

    def __init__(
        self,
        base_evaluator: _Evaluator | None = None,
        window_size: int = 200,
    ) -> None:
        self._base = base_evaluator
        self._history: deque[float] = deque(maxlen=window_size)

    def rank_population(
        self, agents: list[Agent], scores: list[float]
    ) -> list[float]:
        """Convert *scores* to fractional ranks in ``[0, 1]``.

        Args:
            agents: Population (same order as *scores*).
            scores: Raw fitness values.

        Returns:
            Ranked fitness values in the same order.
        """
        n = len(scores)
        if n == 0:
            return []
        self._history.extend(scores)
        order = sorted(range(n), key=lambda i: scores[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(order):
            ranks[idx] = rank / max(n - 1, 1)
        return ranks

    def __call__(self, agent: Agent, task: str) -> float:
        if self._base is None:
            raise RuntimeError("RankShaper requires a base_evaluator for single-agent mode.")
        raw = self._base(agent, task)
        self._history.append(raw)
        history = list(self._history)
        history.sort()
        # Binary search position
        lo, hi = 0, len(history)
        while lo < hi:
            mid = (lo + hi) // 2
            if history[mid] < raw:
                lo = mid + 1
            else:
                hi = mid
        return lo / max(len(history) - 1, 1)


# ── CuriosityShaper ────────────────────────────────────────────────────────────


class CuriosityShaper(RewardShaper):
    """Intrinsic motivation bonus proportional to genome novelty.

    Agents with prompts distant from recently seen prompts receive a bonus,
    encouraging exploration of novel prompt strategies.

    Args:
        base_evaluator: Upstream evaluator.
        scale: Novelty bonus multiplier.  Default ``0.1``.
        memory_size: Number of recent agent fingerprints to track.
    """

    def __init__(
        self,
        base_evaluator: _Evaluator,
        scale: float = 0.1,
        memory_size: int = 50,
    ) -> None:
        super().__init__(base_evaluator)
        self._scale = scale
        self._seen: deque[str] = deque(maxlen=memory_size)

    def _novelty(self, prompt: str) -> float:
        """Fraction of remembered prompts that differ from *prompt*."""
        if not self._seen:
            return 1.0
        # Use character trigram overlap as a similarity proxy
        def trigrams(s: str) -> set[str]:
            return {s[i:i+3] for i in range(len(s) - 2)}

        tg = trigrams(prompt)
        diffs = sum(
            1 for p in self._seen
            if len(tg | trigrams(p)) == 0 or len(tg & trigrams(p)) / len(tg | trigrams(p)) < 0.5
        )
        return diffs / len(self._seen)

    def shape(self, raw_fitness: float, agent: Agent, task: str) -> float:
        novelty = self._novelty(agent.genome.system_prompt)
        self._seen.append(agent.genome.system_prompt)
        return raw_fitness + self._scale * novelty


# ── Convenience factory ────────────────────────────────────────────────────────


def build_shaped_evaluator(
    base_evaluator: _Evaluator,
    spec: str = "clip",
    **kwargs: Any,
) -> _Evaluator:
    """Build a shaped evaluator from a simple spec string.

    Spec is a ``+``-separated list of shaper names applied left to right:

    - ``"clip"`` → :class:`ClipShaper`
    - ``"normalise"`` or ``"normalize"`` → :class:`NormalisationShaper`
    - ``"potential"`` → :class:`PotentialShaper`
    - ``"curiosity"`` → :class:`CuriosityShaper`

    Args:
        base_evaluator: The raw evaluator to wrap.
        spec: Shaper pipeline string (e.g. ``"clip+normalise"``).
        **kwargs: Forwarded to each shaper constructor.

    Returns:
        Composed evaluator callable.

    Example::

        shaped = build_shaped_evaluator(my_eval, "clip+normalise+curiosity")
    """
    evaluator: _Evaluator = base_evaluator
    for name in spec.split("+"):
        name = name.strip().lower()
        if name == "clip":
            evaluator = ClipShaper(
                evaluator,
                min_val=kwargs.get("min_val", 0.0),
                max_val=kwargs.get("max_val", 1.0),
            )
        elif name in {"normalise", "normalize"}:
            evaluator = NormalisationShaper(
                evaluator,
                method=kwargs.get("method", "minmax"),
                window_size=kwargs.get("window_size", 100),
            )
        elif name == "potential":
            evaluator = PotentialShaper(
                evaluator,
                gamma=kwargs.get("gamma", 0.99),
                scale=kwargs.get("scale", 0.1),
            )
        elif name == "curiosity":
            evaluator = CuriosityShaper(
                evaluator,
                scale=kwargs.get("curiosity_scale", 0.1),
            )
        else:
            raise ValueError(f"Unknown shaper {name!r}. Valid: clip, normalise, potential, curiosity")
    return evaluator
