"""Symbiotic Fusion — Technique 50.

Two agents that perform well on complementary task facets merge into a
single genome that inherits traits from both.  Inspired by endosymbiosis
(mitochondria / chloroplast origin) and bacterial conjugation.

Key components
--------------
SymbioticPair
    Lightweight record describing the two partners and the merge result.

SymbioticFuser
    Selects partner agents from a population, checks *compatibility*
    (fitness above threshold + prompt-distance above min_distance so we
    don't fuse near-identical genomes), calls an LLM to synthesise a
    merged system_prompt, and returns the fused ``Agent``.

Usage::

    from cambrian.symbiosis import SymbioticFuser

    fuser = SymbioticFuser(backend=backend, fitness_threshold=0.5,
                           min_distance=0.2)
    fused = fuser.fuse(host, donor, task)
    if fused is not None:
        population.append(fused)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cambrian.agent import Agent, Genome

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FUSE_SYSTEM = (
    "You are an expert at synthesising two AI agent system prompts into a "
    "single, coherent, superior prompt.  Preserve the best strategies from "
    "both and remove redundancy.  Return ONLY the merged system prompt text."
)

_FUSE_TEMPLATE = """\
TASK: {task}

HOST AGENT (fitness={host_fitness:.3f}):
System prompt:
{host_prompt}
Strategy: {host_strategy}

DONOR AGENT (fitness={donor_fitness:.3f}):
System prompt:
{donor_prompt}
Strategy: {donor_strategy}

Synthesise a single system prompt that inherits the complementary strengths
of both agents.  Return ONLY the merged system prompt — no commentary.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbioticPair:
    """Record of a fusion event."""

    host_id: str
    donor_id: str
    host_fitness: float
    donor_fitness: float
    offspring_id: str
    merged_prompt_preview: str  # first 120 chars


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SymbioticFuser:
    """Select compatible partners and fuse them into a hybrid genome.

    Parameters
    ----------
    backend:
        LLM backend used to synthesise the merged prompt.
    fitness_threshold:
        Both agents must have fitness ≥ this value to qualify.
    min_distance:
        Minimum normalised edit-distance between prompts (0–1).
        Prevents fusing near-identical agents (no gain).
    temperature:
        Sampling temperature for the fusion LLM call.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        fitness_threshold: float = 0.4,
        min_distance: float = 0.15,
        temperature: float = 0.7,
    ) -> None:
        self._backend = backend
        self._fitness_threshold = fitness_threshold
        self._min_distance = min_distance
        self._temperature = temperature
        self._history: list[SymbioticPair] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[SymbioticPair]:
        """Return a copy of all fusion events."""
        return list(self._history)

    def fuse(
        self,
        host: Agent,
        donor: Agent,
        task: str,
    ) -> Optional[Agent]:
        """Attempt to fuse *host* and *donor* into a new hybrid ``Agent``.

        Returns ``None`` when the pair is not compatible (fitness too low
        or prompts too similar).
        """
        if not self._compatible(host, donor):
            return None

        host_fitness = host.fitness if host.fitness is not None else 0.0
        donor_fitness = donor.fitness if donor.fitness is not None else 0.0

        merged_prompt = self._call_llm(host, donor, host_fitness, donor_fitness, task)
        merged_genome = Genome(
            system_prompt=merged_prompt,
            temperature=(host.genome.temperature + donor.genome.temperature) / 2,
            strategy=f"symbiosis({host.genome.strategy},{donor.genome.strategy})",
        )
        offspring = Agent(genome=merged_genome)

        self._history.append(
            SymbioticPair(
                host_id=host.agent_id,
                donor_id=donor.agent_id,
                host_fitness=host_fitness,
                donor_fitness=donor_fitness,
                offspring_id=offspring.agent_id,
                merged_prompt_preview=merged_prompt[:120],
            )
        )
        return offspring

    def fuse_best_pair(
        self,
        population: list[Agent],
        task: str,
        rng: Optional[random.Random] = None,
    ) -> Optional[Agent]:
        """Select the best compatible pair from *population* and fuse them.

        Selection strategy: pick the top-fitness agent as host, then find
        the most distant compatible donor.  Returns ``None`` if no valid
        pair exists.
        """
        candidates = [
            a for a in population if (a.fitness or 0.0) >= self._fitness_threshold
        ]
        if len(candidates) < 2:
            return None

        rng = rng or random.Random()
        candidates_sorted = sorted(
            candidates, key=lambda a: a.fitness or 0.0, reverse=True
        )
        host = candidates_sorted[0]

        # Find donor: furthest prompt from host that is also compatible
        best_donor: Optional[Agent] = None
        best_dist = -1.0
        for agent in candidates_sorted[1:]:
            d = self._prompt_distance(host.genome.system_prompt, agent.genome.system_prompt)
            if d >= self._min_distance and d > best_dist:
                best_dist = d
                best_donor = agent

        if best_donor is None:
            return None
        return self.fuse(host, best_donor, task)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compatible(self, host: Agent, donor: Agent) -> bool:
        """Return True if the pair meets fitness and distance thresholds."""
        if (host.fitness or 0.0) < self._fitness_threshold:
            return False
        if (donor.fitness or 0.0) < self._fitness_threshold:
            return False
        dist = self._prompt_distance(
            host.genome.system_prompt, donor.genome.system_prompt
        )
        return dist >= self._min_distance

    @staticmethod
    def _prompt_distance(a: str, b: str) -> float:
        """Approximate normalised word-level Jaccard distance ∈ [0, 1]."""
        if not a and not b:
            return 0.0
        tokens_a = set(re.split(r"\W+", a.lower()))
        tokens_b = set(re.split(r"\W+", b.lower()))
        tokens_a.discard("")
        tokens_b.discard("")
        if not tokens_a and not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union == 0:
            return 0.0
        return 1.0 - intersection / union

    def _call_llm(
        self,
        host: Agent,
        donor: Agent,
        host_fitness: float,
        donor_fitness: float,
        task: str,
    ) -> str:
        """Call the LLM to produce a merged system prompt."""
        user_msg = _FUSE_TEMPLATE.format(
            task=task,
            host_fitness=host_fitness,
            host_prompt=host.genome.system_prompt,
            host_strategy=host.genome.strategy,
            donor_fitness=donor_fitness,
            donor_prompt=donor.genome.system_prompt,
            donor_strategy=donor.genome.strategy,
        )
        try:
            return str(
                self._backend.generate(
                    f"{_FUSE_SYSTEM}\n\n{user_msg}",
                    temperature=self._temperature,
                )
            )
        except Exception:  # noqa: BLE001
            # Fallback: naive concatenation
            return (
                f"{host.genome.system_prompt}\n\n"
                f"[SYMBIOTIC ADDON]\n{donor.genome.system_prompt}"
            )
