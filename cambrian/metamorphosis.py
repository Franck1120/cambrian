# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""cambrian/metamorphosis.py — Discrete metamorphic phase transitions for agents.

Implements a holometabolous-inspired lifecycle:
    LARVA -> CHRYSALIS -> IMAGO

Each phase imposes different evolutionary pressures:
- LARVA: high exploration, broad mutation, no strategy constraints.
- CHRYSALIS: frozen mutation; LLM-driven internal reorganisation.
- IMAGO: low mutation, high specialisation, strategy locked to 'exploit'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend

logger = logging.getLogger(__name__)


# ── Phase Enum ────────────────────────────────────────────────────────────────


class MetamorphicPhase(str, Enum):
    """Discrete lifecycle phases inspired by holometabolous insect development.

    Values correspond to stages of metamorphosis:
    - LARVA: juvenile, broad exploration.
    - CHRYSALIS: transition, internal reorganisation.
    - IMAGO: adult, specialised exploitation.
    """

    LARVA = "larva"
    CHRYSALIS = "chrysalis"
    IMAGO = "imago"


# ── PhaseConfig ───────────────────────────────────────────────────────────────


@dataclass
class PhaseConfig:
    """Configuration parameters for a single metamorphic phase.

    Attributes:
        phase: The phase this config applies to.
        min_generations: Minimum number of generations an agent must spend in
            this phase before it is eligible to advance.
        fitness_threshold: Minimum fitness score required to advance to the
            next phase. Defaults to 0.0 (IMAGO has no successor).
        mutation_rate_multiplier: Scalar multiplied by the base mutation rate
            while the agent is in this phase. 0.0 freezes mutation entirely.
        description: Human-readable description of the phase. Optional.
    """

    phase: MetamorphicPhase
    min_generations: int
    fitness_threshold: float = 0.0
    mutation_rate_multiplier: float = 1.0
    description: str = ""


# ── MorphEvent ────────────────────────────────────────────────────────────────


@dataclass
class MorphEvent:
    """Record of a phase transition for one agent.

    Attributes:
        agent_id: ID of the agent that transitioned.
        from_phase: The phase the agent left.
        to_phase: The phase the agent entered.
        generation: Evolution generation at which the transition occurred.
        fitness_at_transition: Agent fitness score at the moment of transition.
    """

    agent_id: str
    from_phase: MetamorphicPhase
    to_phase: MetamorphicPhase
    generation: int
    fitness_at_transition: float


# ── MetamorphosisController ───────────────────────────────────────────────────

# Successor map for linear phase progression
_NEXT_PHASE: dict[MetamorphicPhase, Optional[MetamorphicPhase]] = {
    MetamorphicPhase.LARVA: MetamorphicPhase.CHRYSALIS,
    MetamorphicPhase.CHRYSALIS: MetamorphicPhase.IMAGO,
    MetamorphicPhase.IMAGO: None,
}


class MetamorphosisController:
    """Manages the metamorphic lifecycle of agents across a population.

    Tracks each agent's current phase, generations spent in that phase, and
    records all phase-transition events. Provides the chrysalis reorganisation
    step where the LLM rewrites the agent's genome based on its larval history.

    Args:
        backend: LLM backend used during the chrysalis reorganisation prompt.
        larva_config: Phase config for LARVA. Defaults to
            ``PhaseConfig(LARVA, min_generations=3, fitness_threshold=0.4,
            mutation_rate_multiplier=1.5)``.
        chrysalis_config: Phase config for CHRYSALIS. Defaults to
            ``PhaseConfig(CHRYSALIS, min_generations=1, fitness_threshold=0.6,
            mutation_rate_multiplier=0.0)``.
        imago_config: Phase config for IMAGO. Defaults to
            ``PhaseConfig(IMAGO, min_generations=0, fitness_threshold=0.0,
            mutation_rate_multiplier=0.5)``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        larva_config: Optional[PhaseConfig] = None,
        chrysalis_config: Optional[PhaseConfig] = None,
        imago_config: Optional[PhaseConfig] = None,
    ) -> None:
        self._backend: LLMBackend = backend

        self._configs: dict[MetamorphicPhase, PhaseConfig] = {
            MetamorphicPhase.LARVA: larva_config or PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=3,
                fitness_threshold=0.4,
                mutation_rate_multiplier=1.5,
                description="High exploration, broad mutation, no strategy constraints.",
            ),
            MetamorphicPhase.CHRYSALIS: chrysalis_config or PhaseConfig(
                phase=MetamorphicPhase.CHRYSALIS,
                min_generations=1,
                fitness_threshold=0.6,
                mutation_rate_multiplier=0.0,
                description="Frozen mutation; LLM-driven internal reorganisation.",
            ),
            MetamorphicPhase.IMAGO: imago_config or PhaseConfig(
                phase=MetamorphicPhase.IMAGO,
                min_generations=0,
                fitness_threshold=0.0,
                mutation_rate_multiplier=0.5,
                description="Low mutation, high specialisation, strategy locked to exploit.",
            ),
        }

        self._agent_phase: dict[str, MetamorphicPhase] = {}
        self._agent_gen_in_phase: dict[str, int] = {}
        self._events: list[MorphEvent] = []

    # ── Phase queries ──────────────────────────────────────────────────────────

    def current_phase(self, agent: Agent) -> MetamorphicPhase:
        """Return the current metamorphic phase for *agent*.

        Agents not yet registered default to LARVA.

        Args:
            agent: The agent to query.

        Returns:
            The agent's current :class:`MetamorphicPhase`.
        """
        return self._agent_phase.get(agent.agent_id, MetamorphicPhase.LARVA)

    def mutation_rate_multiplier(self, agent: Agent) -> float:
        """Return the mutation-rate multiplier appropriate for *agent*'s phase.

        Args:
            agent: The agent to query.

        Returns:
            Scalar multiplier. CHRYSALIS returns 0.0 (no mutation).
        """
        phase = self.current_phase(agent)
        return self._configs[phase].mutation_rate_multiplier

    # ── Phase advancement ──────────────────────────────────────────────────────

    def advance(
        self,
        agent: Agent,
        generation: int,
        fitness: float,
    ) -> Optional[MorphEvent]:
        """Attempt to advance *agent* to the next phase.

        Checks both the minimum-generations and fitness-threshold criteria for
        the agent's current phase config. Increments the generation counter
        regardless of whether a transition occurs.

        Args:
            agent: The agent to evaluate.
            generation: Current evolution generation number.
            fitness: The agent's fitness score this generation.

        Returns:
            A :class:`MorphEvent` if a transition occurred, otherwise ``None``.
        """
        aid = agent.agent_id

        # Ensure agent is registered
        if aid not in self._agent_phase:
            self._agent_phase[aid] = MetamorphicPhase.LARVA
            self._agent_gen_in_phase[aid] = 0

        current = self._agent_phase[aid]
        next_phase = _NEXT_PHASE[current]

        if next_phase is None:
            # IMAGO — terminal phase, nothing to advance to
            self._agent_gen_in_phase[aid] += 1
            return None

        cfg = self._configs[current]
        gens_in_phase = self._agent_gen_in_phase[aid]

        # Increment BEFORE checking so the count represents "after this gen"
        self._agent_gen_in_phase[aid] += 1
        gens_in_phase += 1

        if gens_in_phase < cfg.min_generations:
            return None

        if fitness < cfg.fitness_threshold:
            return None

        # Criteria met — transition
        event = MorphEvent(
            agent_id=aid,
            from_phase=current,
            to_phase=next_phase,
            generation=generation,
            fitness_at_transition=fitness,
        )
        self._agent_phase[aid] = next_phase
        self._agent_gen_in_phase[aid] = 0
        self._events.append(event)
        logger.info(
            "Agent %s: %s -> %s at gen %d (fitness=%.4f)",
            aid,
            current.value,
            next_phase.value,
            generation,
            fitness,
        )
        return event

    # ── Chrysalis reorganisation ───────────────────────────────────────────────

    def metamorphose(self, agent: Agent, task: str) -> Agent:
        """Perform the chrysalis reorganisation for *agent*.

        Calls the LLM backend to restructure the agent's system prompt into a
        mature, specialised version. Returns a *new* Agent with the reorganised
        genome; the original is not modified.

        Falls back to concatenating the original prompt with a maturity
        annotation if the LLM call fails.

        Args:
            agent: The agent in CHRYSALIS phase to reorganise.
            task: The task context used to guide specialisation.

        Returns:
            A new :class:`Agent` with the restructured genome.
        """
        genome = agent.genome
        prompt = (
            "Reorganize this agent's system prompt into a mature, specialized version. "
            f"Current genome: {genome.system_prompt}. "
            f"Task: {task}. "
            "Return only the new system prompt."
        )

        try:
            new_system_prompt = self._backend.generate(prompt)
        except Exception:  # noqa: BLE001
            logger.warning(
                "MetamorphosisController.metamorphose: backend call failed for agent %s; "
                "using fallback prompt.",
                agent.agent_id,
            )
            new_system_prompt = (
                f"{genome.system_prompt} [metamorphosed: matured for task: {task}]"
            )

        new_genome = Genome.from_dict(genome.to_dict())
        new_genome.system_prompt = new_system_prompt

        new_agent = Agent(
            genome=new_genome,
            backend=agent.backend,
            agent_id=agent.agent_id,
        )
        new_agent.generation = agent.generation
        return new_agent

    # ── Phase pressure ─────────────────────────────────────────────────────────

    def apply_phase_pressure(self, genome: Genome, phase: MetamorphicPhase) -> Genome:
        """Return a modified copy of *genome* reflecting the constraints of *phase*.

        - LARVA: raises temperature to encourage exploration.
        - CHRYSALIS: no change (reorganisation handled separately).
        - IMAGO: sets strategy to ``"chain-of-thought"`` and lowers temperature.

        Args:
            genome: Source genome (not mutated).
            phase: The phase whose pressure to apply.

        Returns:
            A new :class:`Genome` with phase-appropriate modifications.
        """
        new_genome = Genome.from_dict(genome.to_dict())

        if phase is MetamorphicPhase.LARVA:
            new_genome.temperature = min(2.0, genome.temperature + 0.3)
        elif phase is MetamorphicPhase.CHRYSALIS:
            pass  # No structural change; reorganisation is handled by metamorphose()
        elif phase is MetamorphicPhase.IMAGO:
            new_genome.strategy = "chain-of-thought"
            new_genome.temperature = max(0.0, genome.temperature - 0.3)

        return new_genome

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def events(self) -> list[MorphEvent]:
        """All phase-transition events recorded so far, in chronological order."""
        return list(self._events)

    def phase_distribution(self) -> dict[str, int]:
        """Return a count of registered agents per phase.

        Returns:
            Dict mapping phase value strings to agent counts.
            Phases with zero agents are omitted.
        """
        dist: dict[str, int] = {}
        for phase in self._agent_phase.values():
            dist[phase.value] = dist.get(phase.value, 0) + 1
        return dist


# ── MetamorphicPopulation ─────────────────────────────────────────────────────


class MetamorphicPopulation:
    """Convenience wrapper that applies metamorphic logic to a whole population.

    Delegates per-agent logic to a :class:`MetamorphosisController` and
    orchestrates the chrysalis reorganisation step when agents transition
    through CHRYSALIS.

    Args:
        controller: The :class:`MetamorphosisController` to use.
    """

    def __init__(self, controller: MetamorphosisController) -> None:
        self._controller: MetamorphosisController = controller
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register *agent* as a LARVA in the population.

        Subsequent calls to :meth:`tick` will include this agent.

        Args:
            agent: The agent to register.
        """
        aid = agent.agent_id
        self._agents[aid] = agent
        # Ensure the controller knows about this agent
        if aid not in self._controller._agent_phase:
            self._controller._agent_phase[aid] = MetamorphicPhase.LARVA
            self._controller._agent_gen_in_phase[aid] = 0

    def tick(
        self,
        population: list[Agent],
        generation: int,
        task: str,
    ) -> list[MorphEvent]:
        """Advance all agents in *population* by one generation.

        For each agent:
        1. Calls :meth:`MetamorphosisController.advance` to check transitions.
        2. If the agent just entered CHRYSALIS, calls
           :meth:`MetamorphosisController.metamorphose` and updates the stored
           agent reference.

        Args:
            population: Agents to process this generation.
            generation: Current evolution generation number.
            task: Task context (used for chrysalis reorganisation).

        Returns:
            All :class:`MorphEvent` objects produced during this tick.
        """
        tick_events: list[MorphEvent] = []

        for agent in population:
            fitness = agent.fitness if agent.fitness is not None else 0.0
            event = self._controller.advance(agent, generation, fitness)

            if event is not None:
                tick_events.append(event)

                # If agent just entered CHRYSALIS, trigger reorganisation
                if event.to_phase is MetamorphicPhase.CHRYSALIS:
                    cfg = self._controller._configs[MetamorphicPhase.CHRYSALIS]
                    if fitness >= cfg.fitness_threshold:
                        new_agent = self._controller.metamorphose(agent, task)
                        self._agents[agent.agent_id] = new_agent

        return tick_events
