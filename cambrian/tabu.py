"""Tabu Search — Technique 59.

Tabu Search (Glover 1986) maintains a short-term "tabu list" of recently
visited solutions, preventing the search from revisiting them.  Applied to
agent evolution, this prevents the mutator from re-generating system prompts
that are "too similar" to recently explored ones, forcing broader exploration.

Components
----------
TabuList
    Stores hashed genome fingerprints.  Fingerprints use a fast bi-gram
    hash of the system_prompt so that near-identical prompts are blocked.

TabuMutator
    Wraps any mutator and rejects mutations whose genome fingerprint is in
    the tabu list, re-trying up to ``max_retries`` times.  Accepted
    mutations are added to the tabu list.

Usage::

    from cambrian.tabu import TabuMutator, TabuList

    tabu_list = TabuList(max_size=20)
    mutator = TabuMutator(base_mutator=llm_mutator, tabu_list=tabu_list)
    new_agent = mutator.mutate(agent, task)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from cambrian.agent import Agent

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Mutator protocol (same as in catalysis.py)
# ---------------------------------------------------------------------------


class MutatorProtocol(Protocol):
    """Minimal interface required by TabuMutator."""

    def mutate(self, agent: Agent, task: str) -> Agent:
        """Return a mutated copy of *agent* for *task*."""
        ...


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _fingerprint(prompt: str, strategy: str = "") -> str:
    """Compute a fast fingerprint of a genome for tabu comparison.

    Uses SHA-256 of the normalised bi-gram set (lowercased, punctuation
    stripped) so that near-identical prompts produce the same fingerprint.
    """
    tokens = re.split(r"\W+", (prompt + " " + strategy).lower())
    tokens = [t for t in tokens if t]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    canonical = " ".join(sorted(set(bigrams)))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TabuEntry:
    """A single entry in the tabu list."""

    fingerprint: str
    agent_id: str
    generation_added: int


# ---------------------------------------------------------------------------
# TabuList
# ---------------------------------------------------------------------------


class TabuList:
    """Fixed-size tabu list with FIFO eviction.

    Parameters
    ----------
    max_size:
        Maximum number of entries before oldest is evicted (default 20).
    """

    def __init__(self, max_size: int = 20) -> None:
        self._max = max_size
        self._entries: list[TabuEntry] = []
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._entries)

    @property
    def entries(self) -> list[TabuEntry]:
        """Return a copy of all entries."""
        return list(self._entries)

    def advance_generation(self) -> None:
        """Increment the internal generation counter."""
        self._generation += 1

    def is_tabu(self, agent: Agent) -> bool:
        """Return True if *agent*'s genome fingerprint is in the tabu list."""
        fp = _fingerprint(agent.genome.system_prompt, agent.genome.strategy)
        return any(e.fingerprint == fp for e in self._entries)

    def add(self, agent: Agent) -> None:
        """Add *agent*'s fingerprint to the tabu list (FIFO eviction)."""
        fp = _fingerprint(agent.genome.system_prompt, agent.genome.strategy)
        # Skip duplicates
        if any(e.fingerprint == fp for e in self._entries):
            return
        self._entries.append(
            TabuEntry(
                fingerprint=fp,
                agent_id=agent.agent_id,
                generation_added=self._generation,
            )
        )
        if len(self._entries) > self._max:
            self._entries.pop(0)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# TabuMutator
# ---------------------------------------------------------------------------


class TabuMutator:
    """Mutator wrapper that skips tabu mutations.

    Parameters
    ----------
    base_mutator:
        Any mutator exposing ``mutate(agent, task) -> Agent``.
    tabu_list:
        Shared ``TabuList`` instance.
    max_retries:
        How many times to retry before giving up and returning the
        first (tabu) mutation anyway (default 3).
    """

    def __init__(
        self,
        base_mutator: MutatorProtocol,
        tabu_list: TabuList,
        max_retries: int = 3,
    ) -> None:
        self._mutator = base_mutator
        self._tabu = tabu_list
        self._max_retries = max_retries
        self._tabu_hits: int = 0
        self._total_mutations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tabu_hit_rate(self) -> float:
        """Fraction of mutations that were blocked by the tabu list."""
        if self._total_mutations == 0:
            return 0.0
        return self._tabu_hits / self._total_mutations

    def mutate(self, agent: Agent, task: str) -> Agent:
        """Mutate *agent*, retrying if the result is tabu."""
        first_candidate: Agent | None = None
        self._total_mutations += 1

        for attempt in range(self._max_retries + 1):
            candidate = self._mutator.mutate(agent, task)
            if first_candidate is None:
                first_candidate = candidate

            if not self._tabu.is_tabu(candidate):
                self._tabu.add(candidate)
                return candidate

            self._tabu_hits += 1

        # All retries exhausted — return first candidate despite tabu
        assert first_candidate is not None
        self._tabu.add(first_candidate)
        return first_candidate
