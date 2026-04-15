# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""B/T-cell Immune Memory — Technique 65.

Inspired by adaptive immunity:
* **B-cells** produce antibodies for known antigens (fast recall for
  previously-seen task patterns → return stored high-fitness response).
* **T-cells** coordinate adaptive responses for novel threats (novel task
  patterns → select the most generalisable stored agent to seed evolution).

Components
----------
MemoryCell
    A stored (task_fingerprint, genome, fitness) record.

BCellMemory
    Exact/near-exact match lookup.  When a task is similar to one in memory,
    instantly returns the stored response without re-querying the LLM.

TCellMemory
    Adaptive lookup.  For novel tasks, finds the stored cell with the highest
    semantic similarity to seed evolution from a good starting point.

ImmuneCortex
    Coordinates both memories.  Records new cells when fitness surpasses
    a threshold; answers queries by checking B-cell first, then T-cell.

Usage::

    from cambrian.immune_memory import ImmuneCortex

    cortex = ImmuneCortex(b_threshold=0.9, t_threshold=0.6)
    cortex.record(agent, task="Solve 2+2")
    result = cortex.recall(agent, task="What is 2+2?")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Fingerprinting (word-Jaccard)
# ---------------------------------------------------------------------------


def _task_similarity(a: str, b: str) -> float:
    """Return word-level Jaccard similarity ∈ [0, 1]."""
    ta = set(re.split(r"\W+", a.lower())) - {""}
    tb = set(re.split(r"\W+", b.lower())) - {""}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MemoryCell:
    """A stored immune memory record."""

    task: str
    genome: Genome
    agent_id: str
    fitness: float
    recall_count: int = 0


@dataclass
class RecallResult:
    """Result from a cortex recall query."""

    recalled: bool
    agent: Optional[Agent]
    cell_type: str   # "b_cell" | "t_cell" | "none"
    similarity: float
    source_task: str


# ---------------------------------------------------------------------------
# BCellMemory
# ---------------------------------------------------------------------------


class BCellMemory:
    """Fast exact/near-exact match memory (B-cell response).

    Parameters
    ----------
    similarity_threshold:
        Jaccard similarity above which a task is considered "known" (default 0.8).
    max_cells:
        Maximum stored cells (FIFO eviction, default 100).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_cells: int = 100,
    ) -> None:
        self._threshold = similarity_threshold
        self._max = max_cells
        self._cells: list[MemoryCell] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of stored B-cells."""
        return len(self._cells)

    def store(self, agent: Agent, task: str) -> None:
        """Store a new B-cell memory.  Evicts oldest if over capacity."""
        if agent.fitness is None:
            return
        cell = MemoryCell(
            task=task,
            genome=Genome.from_dict(agent.genome.to_dict()),
            agent_id=agent.agent_id,
            fitness=agent.fitness,
        )
        self._cells.append(cell)
        if len(self._cells) > self._max:
            self._cells.pop(0)

    def recall(self, task: str) -> Optional[MemoryCell]:
        """Return the best-matching B-cell for *task*, or None."""
        best: Optional[MemoryCell] = None
        best_sim = 0.0
        for cell in self._cells:
            sim = _task_similarity(task, cell.task)
            if sim >= self._threshold and sim > best_sim:
                best_sim = sim
                best = cell
        if best is not None:
            best.recall_count += 1
        return best


# ---------------------------------------------------------------------------
# TCellMemory
# ---------------------------------------------------------------------------


class TCellMemory:
    """Adaptive lookup for novel tasks (T-cell response).

    Finds the stored cell with the highest task similarity even if below
    the B-cell threshold — useful for seeding evolution.

    Parameters
    ----------
    min_similarity:
        Minimum similarity to consider a T-cell response meaningful (default 0.2).
    max_cells:
        Maximum stored cells (FIFO eviction, default 50).
    """

    def __init__(
        self,
        min_similarity: float = 0.2,
        max_cells: int = 50,
    ) -> None:
        self._min_sim = min_similarity
        self._max = max_cells
        self._cells: list[MemoryCell] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of stored T-cells."""
        return len(self._cells)

    def store(self, agent: Agent, task: str) -> None:
        """Store a new T-cell memory."""
        if agent.fitness is None:
            return
        cell = MemoryCell(
            task=task,
            genome=Genome.from_dict(agent.genome.to_dict()),
            agent_id=agent.agent_id,
            fitness=agent.fitness,
        )
        self._cells.append(cell)
        if len(self._cells) > self._max:
            self._cells.pop(0)

    def best_match(self, task: str) -> Optional[MemoryCell]:
        """Return the most similar T-cell, or None if no match above min_similarity."""
        best: Optional[MemoryCell] = None
        best_sim = 0.0
        for cell in self._cells:
            sim = _task_similarity(task, cell.task)
            if sim > best_sim:
                best_sim = sim
                best = cell
        if best is not None and best_sim >= self._min_sim:
            best.recall_count += 1
            return best
        return None


# ---------------------------------------------------------------------------
# ImmuneCortex
# ---------------------------------------------------------------------------


class ImmuneCortex:
    """Coordinates B-cell (fast recall) and T-cell (adaptive seed) memories.

    Parameters
    ----------
    b_threshold:
        Fitness threshold for B-cell storage (default 0.8).
    t_threshold:
        Fitness threshold for T-cell storage (default 0.5).
    b_similarity:
        Jaccard similarity threshold for B-cell recall (default 0.75).
    t_min_similarity:
        Minimum similarity for T-cell match (default 0.2).
    """

    def __init__(
        self,
        b_threshold: float = 0.8,
        t_threshold: float = 0.5,
        b_similarity: float = 0.75,
        t_min_similarity: float = 0.2,
    ) -> None:
        self._b_threshold = b_threshold
        self._t_threshold = t_threshold
        self._b_memory = BCellMemory(similarity_threshold=b_similarity)
        self._t_memory = TCellMemory(min_similarity=t_min_similarity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, agent: Agent, task: str) -> None:
        """Store agent in B-cell and/or T-cell memory based on fitness."""
        fitness = agent.fitness or 0.0
        if fitness >= self._b_threshold:
            self._b_memory.store(agent, task)
        if fitness >= self._t_threshold:
            self._t_memory.store(agent, task)

    def recall(self, task: str) -> RecallResult:
        """Try B-cell first, then T-cell, return the best match."""
        # B-cell check
        b_cell = self._b_memory.recall(task)
        if b_cell is not None:
            new_genome = Genome.from_dict(b_cell.genome.to_dict())
            recalled_agent = Agent(genome=new_genome)
            recalled_agent.fitness = b_cell.fitness
            return RecallResult(
                recalled=True,
                agent=recalled_agent,
                cell_type="b_cell",
                similarity=_task_similarity(task, b_cell.task),
                source_task=b_cell.task,
            )

        # T-cell check
        t_cell = self._t_memory.best_match(task)
        if t_cell is not None:
            new_genome = Genome.from_dict(t_cell.genome.to_dict())
            recalled_agent = Agent(genome=new_genome)
            recalled_agent.fitness = t_cell.fitness
            return RecallResult(
                recalled=True,
                agent=recalled_agent,
                cell_type="t_cell",
                similarity=_task_similarity(task, t_cell.task),
                source_task=t_cell.task,
            )

        return RecallResult(
            recalled=False,
            agent=None,
            cell_type="none",
            similarity=0.0,
            source_task="",
        )

    @property
    def b_cell_count(self) -> int:
        """Number of B-cell memories."""
        return self._b_memory.size

    @property
    def t_cell_count(self) -> int:
        """Number of T-cell memories."""
        return self._t_memory.size
