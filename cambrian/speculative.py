# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Speculative execution for Cambrian — async parallel mutation sampling.

In standard evolution, each agent is mutated sequentially and the single
offspring is kept.  *Speculative execution* generates **K candidate mutations
in parallel** (using ``asyncio``), evaluates them, and keeps only the best.
This trades LLM API cost for higher offspring quality per generation.

Architecture
------------

:class:`SpeculativeResult`
    Outcome of one speculative run: the winning candidate agent plus metadata
    about how many candidates were generated and their fitness values.

:func:`speculate`
    Given an agent, produce K mutations concurrently and return the best.
    Works with any ``LLMMutator``-compatible object.

:class:`SpeculativeMutator`
    Drop-in replacement for :class:`~cambrian.mutator.LLMMutator` that
    transparently applies speculative execution during ``mutate()``.

Usage::

    from cambrian.speculative import SpeculativeMutator

    # Replace the standard mutator in your EvolutionEngine
    mutator = SpeculativeMutator(backend=backend, k_candidates=3)
    engine = EvolutionEngine(evaluator=my_eval, mutator=mutator, ...)

    # Or call speculate() directly
    from cambrian.speculative import speculate
    result = await speculate(agent, task, mutator=mutator, evaluator=my_eval, k=4)
    best = result.winner
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from cambrian.agent import Agent
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── SpeculativeResult ─────────────────────────────────────────────────────────


@dataclass
class SpeculativeResult:
    """Outcome of a speculative mutation run.

    Attributes:
        winner: The best candidate agent (highest fitness after evaluation).
        candidates: All K generated candidates (including winner).
        fitness_values: Fitness scores in the same order as *candidates*.
            ``None`` entries mean evaluation raised an exception.
        k: Number of candidates requested.
    """

    winner: Agent
    candidates: list[Agent] = field(default_factory=list)
    fitness_values: list[float | None] = field(default_factory=list)
    k: int = 1

    @property
    def best_fitness(self) -> float:
        """Fitness of the winning candidate."""
        return self.winner.fitness or 0.0

    @property
    def mean_fitness(self) -> float:
        """Mean fitness across all candidates (ignoring evaluation failures)."""
        vals = [v for v in self.fitness_values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def improvement_over_mean(self) -> float:
        """How much the winner beats the mean (reward for speculative sampling)."""
        return self.best_fitness - self.mean_fitness


# ── Core speculate() function ─────────────────────────────────────────────────


async def speculate(
    agent: Agent,
    task: str,
    mutator: LLMMutator,
    evaluator: Callable[[Agent, str], float],
    k: int = 3,
) -> SpeculativeResult:
    """Generate K mutations in parallel and return the best.

    Args:
        agent: Parent agent to mutate.
        task: Task description (passed to mutator and evaluator).
        mutator: Mutation operator — must implement ``mutate(agent, task)``.
        evaluator: Callable that returns a fitness score in [0, 1].
        k: Number of candidate mutations to generate (minimum 1).

    Returns:
        :class:`SpeculativeResult` with the winning agent and all candidates.
    """
    k = max(1, k)

    async def _one_candidate(idx: int) -> tuple[Agent, float | None]:
        loop = asyncio.get_running_loop()
        # Run blocking mutator call in a thread pool
        try:
            candidate = await loop.run_in_executor(None, mutator.mutate, agent, task)
        except Exception as exc:
            logger.warning("Speculative candidate %d mutation failed: %s", idx, exc)
            return agent.clone(), None

        try:
            fitness = await loop.run_in_executor(None, evaluator, candidate, task)
            candidate.fitness = fitness
        except Exception as exc:
            logger.warning("Speculative candidate %d evaluation failed: %s", idx, exc)
            return candidate, None

        return candidate, fitness

    # Launch all K candidates concurrently
    tasks = [_one_candidate(i) for i in range(k)]
    results = await asyncio.gather(*tasks)

    candidates: list[Agent] = []
    fitness_values: list[float | None] = []
    for cand, fit in results:
        candidates.append(cand)
        fitness_values.append(fit)

    # Select winner — best fitness; fall back to parent if all failed
    best_candidate = agent
    best_fitness = agent.fitness or -1.0
    for cand, fit in zip(candidates, fitness_values):
        if fit is not None and fit > best_fitness:
            best_fitness = fit
            best_candidate = cand

    logger.debug(
        "Speculate: k=%d, best=%.4f, mean=%.4f",
        k,
        best_fitness,
        sum(f for f in fitness_values if f is not None)
        / max(1, sum(1 for f in fitness_values if f is not None)),
    )
    return SpeculativeResult(
        winner=best_candidate,
        candidates=candidates,
        fitness_values=fitness_values,
        k=k,
    )


# ── SpeculativeMutator ────────────────────────────────────────────────────────


class SpeculativeMutator(LLMMutator):
    """Drop-in mutator with speculative execution.

    Extends :class:`~cambrian.mutator.LLMMutator` so that each call to
    ``mutate()`` internally generates ``k_candidates`` mutations (in parallel
    via ``asyncio``) and returns only the best.

    Args:
        backend: LLM backend for mutation (same as ``LLMMutator``).
        k_candidates: Number of parallel mutations to sample.  Default ``3``.
        evaluator: Optional evaluator used to select the best candidate.
            If ``None``, the candidate with the highest token diversity
            (prompt length) is selected as a cheap proxy.
        **kwargs: Additional keyword arguments forwarded to ``LLMMutator``.
    """

    def __init__(
        self,
        backend: Any,
        k_candidates: int = 3,
        evaluator: Callable[[Agent, str], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend=backend, **kwargs)
        self._k = max(1, k_candidates)
        self._evaluator = evaluator
        self._total_saved: int = 0  # count of times winner > mean

    def mutate(self, agent: Agent, task: str = "") -> Agent:
        """Generate *k* mutations and return the best.

        If no external evaluator is provided, selects by prompt length as a
        cheap diversity proxy.

        Args:
            agent: Parent agent.
            task: Task description.

        Returns:
            Best offspring agent.
        """
        if self._k == 1 or self._evaluator is None:
            # Generate candidates sequentially without async overhead
            candidates = []
            for _ in range(self._k):
                try:
                    c = super().mutate(agent, task)
                    candidates.append(c)
                except Exception:
                    pass
            if not candidates:
                return super().mutate(agent, task)
            if self._evaluator is not None:
                for c in candidates:
                    try:
                        c.fitness = self._evaluator(c, task)
                    except Exception:
                        pass
                winner = max(candidates, key=lambda c: c.fitness or 0.0)
            else:
                # Diversity proxy: longest prompt
                winner = max(candidates, key=lambda c: len(c.genome.system_prompt))
            return winner

        # Use async path when evaluator is available
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                speculate(
                    agent=agent,
                    task=task,
                    mutator=self,
                    evaluator=self._evaluator,
                    k=self._k,
                )
            )
        finally:
            loop.close()

        if result.improvement_over_mean > 0:
            self._total_saved += 1
        return result.winner

    @property
    def k_candidates(self) -> int:
        """Number of speculative candidates per mutation call."""
        return self._k

    @property
    def total_saved(self) -> int:
        """How many times the winner beat the mean (selection saved quality)."""
        return self._total_saved

    def __repr__(self) -> str:
        return f"SpeculativeMutator(k={self._k}, total_saved={self._total_saved})"
