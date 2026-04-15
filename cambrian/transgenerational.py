"""Transgenerational Epigenetics — Technique 64.

In biology, transgenerational epigenetics refers to heritable changes in
gene expression that do not involve changes to the DNA sequence itself.
Certain environmental experiences alter epigenetic marks that can persist
across multiple generations.

Applied to agent evolution:
* **EpigeneMark**: a named "experience annotation" (e.g. "prefers step-by-step
  reasoning") attached to an agent.
* **TransgenerationalRegistry**: stores marks that were observed to improve
  fitness.  During reproduction, offspring inherit a decaying subset of their
  parent's marks.  Marks fade over ``max_generations`` generations.
* **TransgenerationalAdapter**: applies inherited marks to the offspring's
  genome as contextual injections into the system prompt.

Usage::

    from cambrian.transgenerational import TransgenerationalRegistry

    registry = TransgenerationalRegistry(max_generations=5)
    registry.record_mark(parent, mark_name="step-by-step", strength=0.9)
    registry.inherit(parent, offspring)  # offspring receives decayed marks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpigeneMark:
    """An epigenetic annotation attached to an agent."""

    name: str              # e.g. "step-by-step", "verify-output"
    strength: float        # 0.0 – 1.0
    generation_born: int   # generation when the mark was first recorded
    source_agent_id: str


@dataclass
class InheritanceRecord:
    """Record of a transgenerational inheritance event."""

    parent_id: str
    offspring_id: str
    marks_transferred: int
    generation: int


# ---------------------------------------------------------------------------
# TransgenerationalRegistry
# ---------------------------------------------------------------------------


class TransgenerationalRegistry:
    """Manage epigenetic marks and their transgenerational inheritance.

    Parameters
    ----------
    max_generations:
        How many generations a mark persists before fully decaying
        (strength decays by 1/max_generations per generation).  Default 5.
    strength_threshold:
        Marks with strength below this are pruned (default 0.1).
    inherit_top_n:
        Maximum number of marks an offspring inherits (default 5).
    """

    def __init__(
        self,
        max_generations: int = 5,
        strength_threshold: float = 0.1,
        inherit_top_n: int = 5,
    ) -> None:
        self._max_gen = max_generations
        self._threshold = strength_threshold
        self._top_n = inherit_top_n
        self._marks: dict[str, list[EpigeneMark]] = {}  # agent_id → marks
        self._records: list[InheritanceRecord] = []
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[InheritanceRecord]:
        """Return a copy of all inheritance records."""
        return list(self._records)

    def advance_generation(self) -> None:
        """Increment the generation counter and decay all marks."""
        self._generation += 1
        decay = 1.0 / self._max_gen
        for agent_id in list(self._marks.keys()):
            surviving: list[EpigeneMark] = []
            for m in self._marks[agent_id]:
                m.strength -= decay
                if m.strength >= self._threshold:
                    surviving.append(m)
            if surviving:
                self._marks[agent_id] = surviving
            else:
                del self._marks[agent_id]

    def record_mark(
        self,
        agent: Agent,
        mark_name: str,
        strength: float = 0.8,
    ) -> EpigeneMark:
        """Attach a new epigenetic mark to *agent*.

        If a mark with the same name already exists, its strength is boosted.
        """
        strength = max(0.0, min(1.0, strength))
        pool = self._marks.setdefault(agent.agent_id, [])
        for existing in pool:
            if existing.name == mark_name:
                existing.strength = min(1.0, existing.strength + strength * 0.5)
                return existing
        mark = EpigeneMark(
            name=mark_name,
            strength=strength,
            generation_born=self._generation,
            source_agent_id=agent.agent_id,
        )
        pool.append(mark)
        return mark

    def get_marks(self, agent: Agent) -> list[EpigeneMark]:
        """Return the marks attached to *agent* (sorted by strength desc)."""
        pool = self._marks.get(agent.agent_id, [])
        return sorted(pool, key=lambda m: m.strength, reverse=True)

    def inherit(
        self,
        parent: Agent,
        offspring: Agent,
    ) -> int:
        """Transfer the strongest marks from *parent* to *offspring*.

        Marks are copied with strength decayed by 1 / max_generations.
        Returns the number of marks transferred.
        """
        parent_marks = self.get_marks(parent)
        top = parent_marks[: self._top_n]
        decay = 1.0 / self._max_gen
        transferred = 0
        for m in top:
            inherited_strength = m.strength - decay
            if inherited_strength >= self._threshold:
                self.record_mark(offspring, m.name, inherited_strength)
                transferred += 1

        self._records.append(
            InheritanceRecord(
                parent_id=parent.agent_id,
                offspring_id=offspring.agent_id,
                marks_transferred=transferred,
                generation=self._generation,
            )
        )
        return transferred

    def inject_context(self, agent: Agent, max_marks: int = 3) -> str:
        """Return a context block summarising *agent*'s strongest marks."""
        marks = self.get_marks(agent)[:max_marks]
        if not marks:
            return ""
        lines = ["[EPIGENETIC INHERITANCE] Inherited tendencies (strength):"]
        for m in marks:
            lines.append(f"  • {m.name} ({m.strength:.2f})")
        return "\n".join(lines)

    def apply_to_genome(
        self,
        agent: Agent,
        genome: Optional[Genome] = None,
    ) -> Genome:
        """Return a new Genome with epigenetic context injected into prompt."""
        target = genome if genome is not None else agent.genome
        ctx = self.inject_context(agent)
        if not ctx:
            return Genome.from_dict(target.to_dict())
        new_prompt = f"{target.system_prompt}\n\n{ctx}"
        new = Genome.from_dict(target.to_dict())
        new.system_prompt = new_prompt
        return new
