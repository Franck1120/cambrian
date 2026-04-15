# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Hormesis — Technique 51.

Biological hormesis: low-dose stressor triggers an adaptive, beneficial
response; high-dose is harmful.  Translated to agent evolution:

* An agent with *low* fitness (below ``stress_threshold``) is identified as
  "stressed".
* The stress intensity ``s = 1 - fitness / stress_threshold`` (∈ [0, 1])
  determines the mutation boost.
* Mild stress  (s < ``mild_cutoff``)  → small boost to mutation temperature.
* Moderate stress (s ∈ [mild, severe))→ medium boost + strategy hint injection.
* Severe stress  (s ≥ ``severe_cutoff``) → aggressive full re-prompt via LLM.

Agents above ``stress_threshold`` are *not* stressed and pass through unchanged.

Usage::

    from cambrian.hormesis import HormesisAdapter

    adapter = HormesisAdapter(backend=backend, stress_threshold=0.5)
    stimulated = adapter.stimulate(agent, task)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cambrian.agent import Agent, Genome

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Stress levels
# ---------------------------------------------------------------------------

_STRESS_NONE = "none"
_STRESS_MILD = "mild"
_STRESS_MODERATE = "moderate"
_STRESS_SEVERE = "severe"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REPROGRAM_SYSTEM = (
    "You are an adaptive AI coach.  An agent is performing very poorly. "
    "Rewrite its system prompt to fundamentally change its approach. "
    "Return ONLY the new system prompt — no commentary."
)

_REPROGRAM_TEMPLATE = """\
TASK: {task}

STRUGGLING AGENT (fitness={fitness:.3f}, stress={stress:.3f}):
Current system prompt:
{prompt}

Current strategy: {strategy}

This agent is severely stressed.  Write a completely new system prompt that
takes a radically different approach to improve performance.
Return ONLY the new system prompt.
"""

_MODERATE_HINT = (
    "\n\n[HORMESIS] You are underperforming.  Try a different reasoning "
    "strategy: break the task into sub-problems, verify each step, "
    "and double-check your final answer."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HormesisEvent:
    """Record of a hormetic stimulation event."""

    agent_id: str
    fitness_before: float
    stress_level: str
    stress_intensity: float
    temp_before: float
    temp_after: float


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class HormesisAdapter:
    """Apply hormetic stimulation to under-performing agents.

    Parameters
    ----------
    backend:
        LLM backend used for severe-stress re-prompting.
    stress_threshold:
        Fitness below which an agent is considered stressed (default 0.5).
    mild_cutoff:
        Stress intensity below this → mild response (default 0.33).
    severe_cutoff:
        Stress intensity above this → severe response (default 0.66).
    temp_mild_boost:
        Temperature delta added on mild stress (default 0.1).
    temp_moderate_boost:
        Temperature delta added on moderate stress (default 0.2).
    temp_severe_boost:
        Temperature delta added on severe stress (default 0.3).
    max_temperature:
        Temperature ceiling after boost (default 1.8).
    """

    def __init__(
        self,
        backend: "LLMBackend",
        stress_threshold: float = 0.5,
        mild_cutoff: float = 0.33,
        severe_cutoff: float = 0.66,
        temp_mild_boost: float = 0.1,
        temp_moderate_boost: float = 0.2,
        temp_severe_boost: float = 0.3,
        max_temperature: float = 1.8,
    ) -> None:
        self._backend = backend
        self._stress_threshold = stress_threshold
        self._mild_cutoff = mild_cutoff
        self._severe_cutoff = severe_cutoff
        self._temp_mild_boost = temp_mild_boost
        self._temp_moderate_boost = temp_moderate_boost
        self._temp_severe_boost = temp_severe_boost
        self._max_temperature = max_temperature
        self._history: list[HormesisEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[HormesisEvent]:
        """Return a copy of all stimulation events."""
        return list(self._history)

    def stress_level(self, agent: Agent) -> str:
        """Return the stress level label for *agent*."""
        fitness = agent.fitness if agent.fitness is not None else 0.0
        if fitness >= self._stress_threshold:
            return _STRESS_NONE
        s = self._stress_intensity(fitness)
        if s < self._mild_cutoff:
            return _STRESS_MILD
        if s < self._severe_cutoff:
            return _STRESS_MODERATE
        return _STRESS_SEVERE

    def stimulate(self, agent: Agent, task: str) -> Agent:
        """Apply hormetic stimulation; return a (possibly modified) Agent.

        Agents above ``stress_threshold`` are returned unchanged (no copy).
        Stressed agents are cloned and modified in-place.
        """
        fitness = agent.fitness if agent.fitness is not None else 0.0
        level = self.stress_level(agent)

        if level == _STRESS_NONE:
            return agent

        s = self._stress_intensity(fitness)
        new_agent = self._clone_agent(agent)
        temp_before = new_agent.genome.temperature

        if level == _STRESS_MILD:
            new_agent.genome.temperature = min(
                new_agent.genome.temperature + self._temp_mild_boost,
                self._max_temperature,
            )
        elif level == _STRESS_MODERATE:
            new_agent.genome.temperature = min(
                new_agent.genome.temperature + self._temp_moderate_boost,
                self._max_temperature,
            )
            new_agent.genome.system_prompt = (
                new_agent.genome.system_prompt + _MODERATE_HINT
            )
        else:  # severe
            new_agent.genome.temperature = min(
                new_agent.genome.temperature + self._temp_severe_boost,
                self._max_temperature,
            )
            new_prompt = self._reprogram(agent, fitness, s, task)
            new_agent.genome.system_prompt = new_prompt

        self._history.append(
            HormesisEvent(
                agent_id=agent.agent_id,
                fitness_before=fitness,
                stress_level=level,
                stress_intensity=s,
                temp_before=temp_before,
                temp_after=new_agent.genome.temperature,
            )
        )
        return new_agent

    def stimulate_population(
        self, population: list[Agent], task: str
    ) -> list[Agent]:
        """Apply stimulate() to every agent in *population*."""
        return [self.stimulate(a, task) for a in population]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stress_intensity(self, fitness: float) -> float:
        """Compute stress intensity ∈ [0, 1].  0 = barely stressed, 1 = worst."""
        if self._stress_threshold <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - fitness / self._stress_threshold))

    @staticmethod
    def _clone_agent(agent: Agent) -> Agent:
        new_genome = Genome.from_dict(agent.genome.to_dict())
        new_agent = Agent(genome=new_genome)
        if agent.fitness is not None:
            new_agent.fitness = agent.fitness
        return new_agent

    def _reprogram(
        self, agent: Agent, fitness: float, stress: float, task: str
    ) -> str:
        """Ask LLM for a completely new system prompt."""
        msg = _REPROGRAM_TEMPLATE.format(
            task=task,
            fitness=fitness,
            stress=stress,
            prompt=agent.genome.system_prompt,
            strategy=agent.genome.strategy,
        )
        try:
            return str(self._backend.generate(msg))
        except Exception:  # noqa: BLE001
            return agent.genome.system_prompt + "\n[HORMESIS] Try a fresh approach."
