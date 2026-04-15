"""cambrian/ecosystem.py — Ecological role-based evolutionary pressures.

Implements a 4-role ecosystem that shapes fitness dynamics:

    Herbivore  — exploits environmental rewards; high diversity preference
    Predator   — hunts weak agents; fitness bonus from defeating prey
    Decomposer — recycles failed agents; extracts fragments from low-fitness genomes
    Parasite   — latches onto strong agents; copies and mutates successful strategies

Roles are assigned to agents and affect how their fitness is modified
during the ecosystem interaction step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator

if TYPE_CHECKING:
    from cambrian.agent import Agent


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class EcologicalRole(str, Enum):
    """The four ecological roles an agent can occupy."""

    HERBIVORE = "herbivore"
    PREDATOR = "predator"
    DECOMPOSER = "decomposer"
    PARASITE = "parasite"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class EcosystemConfig:
    """Tunable parameters that govern ecological interaction dynamics.

    Attributes:
        herbivore_diversity_bonus: Fitness bonus per unique strategy in the
            population (excluding the herbivore's own strategy).
        predator_hunt_threshold: Prey agent must have fitness strictly below
            this value to be counted as huntable.
        predator_hunt_bonus: Fitness bonus added to a predator for each prey
            agent it hunts in one interaction round.
        decomposer_recycle_threshold: Agent must have fitness strictly below
            this value to be considered recyclable by a decomposer.
        decomposer_bonus: Fitness bonus added to a decomposer per recyclable
            agent it identifies.
        parasite_host_threshold: Agent must have fitness strictly above this
            value to be eligible as a parasite host.
        parasite_drain: Fitness drained from the strongest eligible host per
            interaction round.
        parasite_gain: Fitness gained by the parasite when it successfully
            latches onto a host.
    """

    herbivore_diversity_bonus: float = 0.05
    predator_hunt_threshold: float = 0.3
    predator_hunt_bonus: float = 0.1
    decomposer_recycle_threshold: float = 0.25
    decomposer_bonus: float = 0.08
    parasite_host_threshold: float = 0.7
    parasite_drain: float = 0.03
    parasite_gain: float = 0.06


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class EcosystemEvent:
    """A single ecological interaction event.

    Attributes:
        role: The ecological role of the acting agent.
        agent_id: ID of the agent whose fitness is being adjusted.
        target_id: ID of the agent that was interacted with, or ``None`` for
            herbivore foraging (no direct target).
        delta: Fitness change applied to the acting agent (positive = gain).
        event_type: Human-readable label, e.g. ``"hunt"``, ``"recycle"``,
            ``"parasite"``, ``"forage"``.
    """

    role: EcologicalRole
    agent_id: str
    target_id: str | None
    delta: float
    event_type: str


# ---------------------------------------------------------------------------
# Core interaction engine
# ---------------------------------------------------------------------------


class EcosystemInteraction:
    """Manages ecological role assignments and runs one interaction round.

    Interaction logic collects all fitness deltas first, then applies them
    atomically so that mid-loop mutations do not corrupt ongoing calculations.

    Args:
        config: Tunable ecosystem parameters. Defaults to
            :class:`EcosystemConfig` with default values.
    """

    def __init__(self, config: EcosystemConfig | None = None) -> None:
        self._config: EcosystemConfig = config if config is not None else EcosystemConfig()
        self._roles: dict[str, EcologicalRole] = {}
        self._events: list[EcosystemEvent] = []

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def assign_role(self, agent: Agent, role: EcologicalRole) -> None:
        """Assign an ecological *role* to *agent*.

        Args:
            agent: The agent to assign a role to.
            role: The :class:`EcologicalRole` to assign.
        """
        self._roles[agent.agent_id] = role

    def get_role(self, agent: Agent) -> EcologicalRole | None:
        """Return the ecological role of *agent*, or ``None`` if unassigned.

        Args:
            agent: The agent to look up.

        Returns:
            The agent's :class:`EcologicalRole`, or ``None``.
        """
        return self._roles.get(agent.agent_id)

    def auto_assign(self, population: list[Agent]) -> None:
        """Automatically assign ecological roles based on fitness distribution.

        Assignment logic:
        - Top 20 % by fitness → :attr:`~EcologicalRole.PREDATOR`
        - Bottom 20 % by fitness → :attr:`~EcologicalRole.DECOMPOSER`
        - Random 20 % of the remaining agents → :attr:`~EcologicalRole.PARASITE`
        - Everyone else → :attr:`~EcologicalRole.HERBIVORE`

        Agents with ``fitness = None`` are treated as having fitness ``0.0``
        for ranking purposes.

        Args:
            population: All agents participating in this generation.
        """
        if not population:
            return

        n = len(population)

        def _fitness_key(a: Agent) -> float:
            return a.fitness if a.fitness is not None else 0.0

        sorted_pop = sorted(population, key=_fitness_key, reverse=True)

        top_n = max(1, round(n * 0.2))
        bottom_n = max(1, round(n * 0.2))

        # Avoid double-assigning the same agent when population is tiny.
        top_n = min(top_n, n)
        bottom_n = min(bottom_n, n - top_n)

        predators = set(a.agent_id for a in sorted_pop[:top_n])
        decomposers = set(a.agent_id for a in sorted_pop[n - bottom_n :])

        # Middle tier — agents not yet assigned
        middle = [a for a in sorted_pop if a.agent_id not in predators and a.agent_id not in decomposers]
        parasite_n = max(0, round(n * 0.2))
        parasite_n = min(parasite_n, len(middle))
        parasites = set(a.agent_id for a in random.sample(middle, parasite_n))

        for agent in population:
            aid = agent.agent_id
            if aid in predators:
                self._roles[aid] = EcologicalRole.PREDATOR
            elif aid in decomposers:
                self._roles[aid] = EcologicalRole.DECOMPOSER
            elif aid in parasites:
                self._roles[aid] = EcologicalRole.PARASITE
            else:
                self._roles[aid] = EcologicalRole.HERBIVORE

    # ------------------------------------------------------------------
    # Interaction round
    # ------------------------------------------------------------------

    def interact(self, population: list[Agent], task: str) -> list[EcosystemEvent]:
        """Run one round of ecological interactions.

        All fitness deltas are collected first, then applied via
        :meth:`apply_events` to avoid mid-loop mutation artifacts.
        The accumulated events are stored and accessible via the
        :attr:`events` property.

        Args:
            population: All agents in the current generation.
            task: The task description (passed through for context; not used
                in the default implementation but available for subclasses).

        Returns:
            List of :class:`EcosystemEvent` objects generated this round.
        """
        if not population:
            return []

        round_events: list[EcosystemEvent] = []
        cfg = self._config

        # Pre-compute values we need without modifying fitness mid-loop.
        # fitness_snapshot maps agent_id → fitness (None → 0.0)
        fitness_snapshot: dict[str, float] = {
            a.agent_id: (a.fitness if a.fitness is not None else 0.0)
            for a in population
        }

        for agent in population:
            role = self._roles.get(agent.agent_id)
            if role is None:
                # Unassigned agents participate passively (no event generated)
                continue

            aid = agent.agent_id

            if role is EcologicalRole.HERBIVORE:
                # Count unique strategies in the rest of the population
                other_strategies = [
                    a.genome.strategy for a in population if a.agent_id != aid
                ]
                unique_count = len(set(other_strategies))
                delta = cfg.herbivore_diversity_bonus * unique_count
                round_events.append(
                    EcosystemEvent(
                        role=role,
                        agent_id=aid,
                        target_id=None,
                        delta=delta,
                        event_type="forage",
                    )
                )

            elif role is EcologicalRole.PREDATOR:
                # Collect all prey (other agents with fitness < threshold)
                prey = [
                    a for a in population
                    if a.agent_id != aid
                    and fitness_snapshot[a.agent_id] < cfg.predator_hunt_threshold
                ]
                if prey:
                    weakest = min(prey, key=lambda a: fitness_snapshot[a.agent_id])
                    delta = cfg.predator_hunt_bonus * len(prey)
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=aid,
                            target_id=weakest.agent_id,
                            delta=delta,
                            event_type="hunt",
                        )
                    )
                else:
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=aid,
                            target_id=None,
                            delta=0.0,
                            event_type="hunt",
                        )
                    )

            elif role is EcologicalRole.DECOMPOSER:
                # Collect recyclable agents (fitness < threshold, excluding self)
                recyclable = [
                    a for a in population
                    if a.agent_id != aid
                    and fitness_snapshot[a.agent_id] < cfg.decomposer_recycle_threshold
                ]
                if recyclable:
                    first_target = recyclable[0]
                    delta = cfg.decomposer_bonus * len(recyclable)
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=aid,
                            target_id=first_target.agent_id,
                            delta=delta,
                            event_type="recycle",
                        )
                    )
                else:
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=aid,
                            target_id=None,
                            delta=0.0,
                            event_type="recycle",
                        )
                    )

            elif role is EcologicalRole.PARASITE:
                # Find eligible hosts (fitness > threshold, excluding self)
                hosts = [
                    a for a in population
                    if a.agent_id != aid
                    and fitness_snapshot[a.agent_id] > cfg.parasite_host_threshold
                ]
                if hosts:
                    strongest_host = max(hosts, key=lambda a: fitness_snapshot[a.agent_id])
                    # Parasite gains
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=aid,
                            target_id=strongest_host.agent_id,
                            delta=cfg.parasite_gain,
                            event_type="parasite",
                        )
                    )
                    # Host loses — stored as a separate event attributed to the host
                    round_events.append(
                        EcosystemEvent(
                            role=role,
                            agent_id=strongest_host.agent_id,
                            target_id=aid,
                            delta=-cfg.parasite_drain,
                            event_type="parasite",
                        )
                    )
                # No event when there are no eligible hosts

        self._events.extend(round_events)
        return round_events

    # ------------------------------------------------------------------
    # Applying events
    # ------------------------------------------------------------------

    def apply_events(self, events: list[EcosystemEvent], population: list[Agent]) -> None:
        """Apply fitness deltas from *events* to agents in *population*.

        Fitness values are clamped to ``[0.0, 1.0]`` after adjustment. Agents
        referenced in events but absent from *population* are silently ignored.

        Args:
            events: Events produced by :meth:`interact`.
            population: The agents whose fitness values will be updated.
        """
        agent_by_id: dict[str, Agent] = {a.agent_id: a for a in population}

        # Accumulate all deltas per agent before writing
        delta_map: dict[str, float] = {}
        for event in events:
            delta_map[event.agent_id] = delta_map.get(event.agent_id, 0.0) + event.delta

        for aid, total_delta in delta_map.items():
            agent = agent_by_id.get(aid)
            if agent is None:
                continue
            current = agent.fitness if agent.fitness is not None else 0.0
            agent.fitness = max(0.0, min(1.0, current + total_delta))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def role_counts(self) -> dict[str, int]:
        """Return a mapping from role name to the number of agents with that role.

        Returns:
            Dictionary with role names as keys and counts as values.
            Only roles with at least one agent are included.
        """
        counts: dict[str, int] = {}
        for role in self._roles.values():
            counts[role.value] = counts.get(role.value, 0) + 1
        return counts

    @property
    def events(self) -> list[EcosystemEvent]:
        """All :class:`EcosystemEvent` objects accumulated across all rounds."""
        return list(self._events)


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


class EcosystemEvaluator(Evaluator):
    """Wraps a base evaluator and blends ecological fitness adjustments.

    The final score is a weighted blend::

        score = (1 - w) * base_score + w * ecological_score

    where ``ecological_score`` is the agent's fitness after one interaction
    round (clamped to ``[0, 1]``), and ``w`` is *interaction_weight*.

    If the agent has no fitness (``None``) before evaluation, it defaults to
    ``0.5`` for the ecological component.

    Args:
        base_evaluator: The primary evaluator that scores agent responses.
        interaction: The :class:`EcosystemInteraction` used to compute the
            ecological component.
        interaction_weight: Weight ``w ∈ [0, 1]`` for the ecological score.
            Default ``0.2`` (20 % ecological, 80 % base).
    """

    def __init__(
        self,
        base_evaluator: Evaluator,
        interaction: EcosystemInteraction,
        interaction_weight: float = 0.2,
    ) -> None:
        self._base_evaluator = base_evaluator
        self._interaction = interaction
        self._interaction_weight = interaction_weight

    def evaluate(self, agent: Agent, task: str) -> float:
        """Evaluate *agent* on *task* and return an ecologically-blended score.

        Args:
            agent: The agent to evaluate.
            task: Natural-language task description.

        Returns:
            Blended fitness score in ``[0.0, 1.0]``.
        """
        base_score = self._base_evaluator.evaluate(agent, task)

        # Use existing fitness as ecological signal; default to 0.5 if None
        eco_score = agent.fitness if agent.fitness is not None else 0.5
        eco_score = max(0.0, min(1.0, eco_score))

        w = self._interaction_weight
        blended = (1.0 - w) * base_score + w * eco_score
        return max(0.0, min(1.0, blended))
