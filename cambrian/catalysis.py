# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Catalysis — Technique 53.

In biochemistry a catalyst lowers the activation energy of a reaction
without itself being consumed.  Here the "catalyst" is a high-fitness
agent whose strategy and system-prompt fragments are extracted and injected
as contextual seeds into the mutation prompt of lower-fitness target agents,
accelerating their improvement without changing the catalyst itself.

Components
----------
CatalystSelector
    Picks the most catalytically active agent from a population
    (highest fitness, largest vocabulary, most distinct strategy keywords).

CatalysisEngine
    Wraps a ``LLMMutator`` (or any mutator exposing ``mutate(agent, task)``)
    and injects catalyst context into each mutation call by prepending
    strategic hints to the target genome's system prompt before mutation,
    then restoring the original prompt afterwards so the mutation sees
    richer context without permanently altering the base genome.

Usage::

    from cambrian.catalysis import CatalysisEngine, CatalystSelector

    selector = CatalystSelector()
    catalyst = selector.select(population)
    engine = CatalysisEngine(mutator=mutator, inject_n_sentences=3)
    improved = engine.catalyse(target_agent, catalyst, task)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from cambrian.agent import Agent

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Mutator protocol — compatible with LLMMutator and any duck-typed mutator
# ---------------------------------------------------------------------------


class MutatorProtocol(Protocol):
    """Minimal interface required by CatalysisEngine."""

    def mutate(self, agent: Agent, task: str) -> Agent:
        """Return a mutated copy of *agent* for *task*."""
        ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CatalysisEvent:
    """Record of a single catalytic mutation event."""

    target_id: str
    catalyst_id: str
    catalyst_fitness: float
    injected_sentences: int


# ---------------------------------------------------------------------------
# CatalystSelector
# ---------------------------------------------------------------------------


class CatalystSelector:
    """Select the best catalyst from a population.

    The "catalytic activity" score combines:
    * ``fitness_weight`` × agent.fitness
    * ``vocab_weight`` × normalised prompt vocabulary size
    * ``strategy_weight`` × normalised count of strategy keywords

    Parameters
    ----------
    fitness_weight, vocab_weight, strategy_weight:
        Weights for the composite score (default 0.6, 0.2, 0.2).
    min_fitness:
        Agents below this fitness are excluded from catalyst candidacy
        (default 0.3).
    """

    def __init__(
        self,
        fitness_weight: float = 0.6,
        vocab_weight: float = 0.2,
        strategy_weight: float = 0.2,
        min_fitness: float = 0.3,
    ) -> None:
        self._fw = fitness_weight
        self._vw = vocab_weight
        self._sw = strategy_weight
        self._min_fitness = min_fitness

    def select(self, population: list[Agent]) -> Optional[Agent]:
        """Return the agent with the highest catalytic score, or None."""
        candidates = [a for a in population if (a.fitness or 0.0) >= self._min_fitness]
        if not candidates:
            return None

        scores = [(self._score(a), a) for a in candidates]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _score(self, agent: Agent) -> float:
        fitness = agent.fitness or 0.0
        vocab = len(set(re.split(r"\W+", agent.genome.system_prompt.lower()))) / 200
        kw = len(agent.genome.strategy.split()) / 20
        return (
            self._fw * fitness
            + self._vw * min(vocab, 1.0)
            + self._sw * min(kw, 1.0)
        )


# ---------------------------------------------------------------------------
# CatalysisEngine
# ---------------------------------------------------------------------------


class CatalysisEngine:
    """Inject catalyst context into mutation calls.

    Parameters
    ----------
    mutator:
        Any mutator with a ``mutate(agent, task) -> Agent`` method.
    inject_n_sentences:
        Number of leading sentences from the catalyst's system prompt to
        inject as context (default 3).
    context_header:
        Header prepended to the injected snippet (default "[CATALYST]").
    """

    def __init__(
        self,
        mutator: MutatorProtocol,
        inject_n_sentences: int = 3,
        context_header: str = "[CATALYST]",
    ) -> None:
        self._mutator = mutator
        self._n = inject_n_sentences
        self._header = context_header
        self._events: list[CatalysisEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[CatalysisEvent]:
        """Return a copy of all catalysis events."""
        return list(self._events)

    def catalyse(
        self,
        target: Agent,
        catalyst: Agent,
        task: str,
    ) -> Agent:
        """Mutate *target* using context from *catalyst*.

        The catalyst's leading sentences are prepended to the target's
        system prompt for the duration of the mutation call, then the
        original prompt is restored.  The catalyst itself is not modified.
        """
        snippet = self._extract_snippet(catalyst.genome.system_prompt)
        augmented_prompt = (
            f"{self._header}\n{snippet}\n\n"
            f"[TARGET]\n{target.genome.system_prompt}"
        )

        # Temporarily patch the target's genome
        original_prompt = target.genome.system_prompt
        target.genome.system_prompt = augmented_prompt

        try:
            mutated = self._mutator.mutate(target, task)
        finally:
            # Always restore — even if mutator raises
            target.genome.system_prompt = original_prompt

        catalyst_fitness = catalyst.fitness if catalyst.fitness is not None else 0.0
        self._events.append(
            CatalysisEvent(
                target_id=target.agent_id,
                catalyst_id=catalyst.agent_id,
                catalyst_fitness=catalyst_fitness,
                injected_sentences=self._count_sentences(snippet),
            )
        )
        return mutated

    def catalyse_population(
        self,
        population: list[Agent],
        catalyst: Agent,
        task: str,
    ) -> list[Agent]:
        """Apply ``catalyse()`` to every agent in *population*."""
        return [self.catalyse(a, catalyst, task) for a in population]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_snippet(self, text: str) -> str:
        """Return the first *n* sentences from *text*."""
        # Split on sentence-ending punctuation followed by whitespace or end
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sentences[: self._n])

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in *text*."""
        if not text.strip():
            return 0
        return len(re.split(r"(?<=[.!?])\s+", text.strip()))
