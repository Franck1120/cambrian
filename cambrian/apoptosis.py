# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Apoptosis — Technique 52.

Programmed cell death applied to agent populations: agents that are
chronically underperforming are removed from the population to make room
for fresher, more diverse variants.

Unlike simple truncation selection (which only looks at the current fitness),
``ApoptosisController`` tracks a *fitness history* per agent and triggers
removal when:

1. **Fitness stagnation**: the agent's fitness has not improved by
   more than ``improvement_epsilon`` over the last ``stagnation_window``
   evaluations.
2. **Absolute floor**: the agent's current fitness is below ``min_fitness``
   after ``grace_period`` evaluations.

When an agent is marked for apoptosis the controller optionally replaces it
with a freshly initialised clone of the current best agent (with a blank
fitness record).

Usage::

    from cambrian.apoptosis import ApoptosisController

    ctrl = ApoptosisController(
        stagnation_window=5, improvement_epsilon=0.02,
        min_fitness=0.1, grace_period=3,
    )
    # Call after every evaluation round:
    population = ctrl.apply(population, best_agent=best)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ApoptosisEvent:
    """Record of a single apoptosis (removal) event."""

    agent_id: str
    reason: str          # "stagnation" | "floor"
    fitness_at_death: float
    generation: int


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ApoptosisController:
    """Track agent fitness histories and remove chronically poor agents.

    Parameters
    ----------
    stagnation_window:
        Number of recent evaluations to examine for improvement (default 5).
    improvement_epsilon:
        Minimum fitness improvement considered "real" progress (default 0.01).
    min_fitness:
        Absolute fitness floor; agents below this after grace_period
        evaluations are removed (default 0.05).
    grace_period:
        Minimum number of evaluations before floor-based removal fires
        (default 3).
    replace_with_clone:
        If True, dead agents are replaced with a randomised clone of the
        current best (default True).
    """

    def __init__(
        self,
        stagnation_window: int = 5,
        improvement_epsilon: float = 0.01,
        min_fitness: float = 0.05,
        grace_period: int = 3,
        replace_with_clone: bool = True,
    ) -> None:
        self._stagnation_window = stagnation_window
        self._improvement_epsilon = improvement_epsilon
        self._min_fitness = min_fitness
        self._grace_period = grace_period
        self._replace_with_clone = replace_with_clone
        # agent_id → list of recorded fitness values (chronological)
        self._histories: dict[str, list[float]] = {}
        self._events: list[ApoptosisEvent] = []
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[ApoptosisEvent]:
        """Return a copy of all apoptosis events."""
        return list(self._events)

    def record(self, agent: Agent) -> None:
        """Record the current fitness of *agent* in its history."""
        if agent.fitness is None:
            return
        hist = self._histories.setdefault(agent.agent_id, [])
        hist.append(agent.fitness)

    def record_population(self, population: list[Agent]) -> None:
        """Call ``record()`` for every agent in *population*."""
        for a in population:
            self.record(a)

    def is_stagnant(self, agent: Agent) -> bool:
        """Return True if the agent shows no improvement over recent history."""
        hist = self._histories.get(agent.agent_id, [])
        if len(hist) < self._stagnation_window:
            return False
        window = hist[-self._stagnation_window :]
        return (max(window) - min(window)) <= self._improvement_epsilon

    def is_below_floor(self, agent: Agent) -> bool:
        """Return True if the agent is stuck below ``min_fitness``."""
        hist = self._histories.get(agent.agent_id, [])
        if len(hist) < self._grace_period:
            return False
        return (agent.fitness or 0.0) < self._min_fitness

    def should_die(self, agent: Agent) -> tuple[bool, str]:
        """Return ``(True, reason)`` if the agent should be removed."""
        if self.is_stagnant(agent):
            return True, "stagnation"
        if self.is_below_floor(agent):
            return True, "floor"
        return False, ""

    def apply(
        self,
        population: list[Agent],
        best_agent: Optional[Agent] = None,
    ) -> list[Agent]:
        """Remove doomed agents and optionally replace them.

        Parameters
        ----------
        population:
            Current population.  Each agent should have had ``record()``
            called at least once before this call.
        best_agent:
            The current best agent.  Used as template for replacements when
            ``replace_with_clone=True``.

        Returns
        -------
        list[Agent]
            Updated population (same length if replacement is enabled).
        """
        self._generation += 1
        survivors: list[Agent] = []
        replacements: list[Agent] = []

        for agent in population:
            should, reason = self.should_die(agent)
            if should:
                fitness = agent.fitness if agent.fitness is not None else 0.0
                self._events.append(
                    ApoptosisEvent(
                        agent_id=agent.agent_id,
                        reason=reason,
                        fitness_at_death=fitness,
                        generation=self._generation,
                    )
                )
                if self._replace_with_clone and best_agent is not None:
                    replacements.append(self._make_replacement(best_agent))
            else:
                survivors.append(agent)

        return survivors + replacements

    def reset_history(self, agent_id: str) -> None:
        """Clear the fitness history for a specific agent (e.g. after reinit)."""
        self._histories.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_replacement(template: Agent) -> Agent:
        """Create a fresh agent cloned from *template* with blank history."""
        new_genome = Genome.from_dict(template.genome.to_dict())
        return Agent(genome=new_genome)
