"""Mixture of Agents (MoA) — ensemble reasoning and Quantum Tunneling.

This module provides two related capabilities:

1. **Mixture of Agents** (``MixtureOfAgents``): N agents independently answer a task,
   then an aggregator LLM combines their responses into a single higher-quality answer.
   This reduces variance and improves quality via diversity of reasoning.

2. **Quantum Tunneling** (``QuantumTunneler``): Occasionally makes large random jumps
   in the genome space to escape local optima — inspired by quantum tunneling where
   a particle can pass through a barrier it classically cannot.

References
----------
- Wang et al. (2024) "Mixture-of-Agents Enhances Large Language Model Capabilities"
- Evolutionary algorithms literature on large mutation / random restart strategies
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ── MixtureOfAgents ────────────────────────────────────────────────────────────


_MOA_AGGREGATE_SYSTEM = """You are a synthesis expert.
You receive multiple independent answers to the same question.
Your task: produce a single, improved answer by:
1. Taking the best elements from each response
2. Resolving contradictions using your own judgement
3. Removing redundancy
4. Ensuring accuracy and completeness
Return ONLY the synthesised answer, no meta-commentary."""

_MOA_AGGREGATE_TEMPLATE = """Task: {task}

Here are {n} independent responses from different agents:

{responses_block}

Synthesise these into one optimal response."""


@dataclass
class MoAResult:
    """Result of a Mixture of Agents inference.

    Attributes:
        final_answer: The aggregated answer from the synthesiser.
        individual_answers: Raw answers from each agent before aggregation.
        n_agents: Number of agents that contributed.
    """

    final_answer: str
    individual_answers: list[str] = field(default_factory=list)
    n_agents: int = 0


class MixtureOfAgents:
    """Ensemble inference: N agents answer independently, one aggregates.

    Args:
        agents: List of :class:`~cambrian.agent.Agent` objects to form the ensemble.
        aggregator_backend: LLM backend used for the aggregation step.
            If ``None``, the backend of the first agent is used.
        aggregator_temperature: Temperature for the aggregation step. Default 0.3.
        n_agents: If set, randomly sample this many agents from the ensemble for
            each call. If ``None`` (default), use all agents.
    """

    def __init__(
        self,
        agents: list[Agent],
        aggregator_backend: "LLMBackend | None" = None,
        aggregator_temperature: float = 0.3,
        n_agents: "int | None" = None,
    ) -> None:
        if not agents:
            raise ValueError("agents must not be empty")
        self._agents = agents
        _backend = aggregator_backend if aggregator_backend is not None else agents[0].backend
        if _backend is None:
            raise ValueError(
                "MixtureOfAgents requires an aggregator_backend or an agent with a backend."
            )
        self._agg_backend: LLMBackend = _backend
        self._agg_temp = aggregator_temperature
        self._n_agents = n_agents

    def run(self, task: str, rng: "random.Random | None" = None) -> MoAResult:
        """Run the mixture inference on *task*.

        Each agent generates an answer independently.  The aggregator
        combines them into the final answer.

        Args:
            task: The task/question to answer.
            rng: Optional random generator for agent sampling.

        Returns:
            A :class:`MoAResult` with the final synthesised answer.
        """
        _rng = rng or random.Random()
        agents = self._agents
        if self._n_agents is not None and self._n_agents < len(self._agents):
            agents = _rng.sample(self._agents, self._n_agents)

        individual_answers: list[str] = []
        for agent in agents:
            try:
                answer = agent.run(task)
                individual_answers.append(answer)
            except Exception as exc:
                logger.warning("MoA agent %s failed: %s", agent.id, exc)

        if not individual_answers:
            return MoAResult(
                final_answer="",
                individual_answers=[],
                n_agents=len(agents),
            )

        if len(individual_answers) == 1:
            # No need to aggregate a single answer
            return MoAResult(
                final_answer=individual_answers[0],
                individual_answers=individual_answers,
                n_agents=len(agents),
            )

        final_answer = self._aggregate(task, individual_answers)
        return MoAResult(
            final_answer=final_answer,
            individual_answers=individual_answers,
            n_agents=len(agents),
        )

    def _aggregate(self, task: str, answers: list[str]) -> str:
        """Combine individual answers into a single synthesised response."""
        responses_block = "\n\n".join(
            f"[Agent {i+1}]:\n{answer[:500]}"
            for i, answer in enumerate(answers)
        )
        prompt = _MOA_AGGREGATE_TEMPLATE.format(
            task=task,
            n=len(answers),
            responses_block=responses_block,
        )
        try:
            return self._agg_backend.generate(
                prompt,
                system=_MOA_AGGREGATE_SYSTEM,
                temperature=self._agg_temp,
            )
        except Exception as exc:
            logger.warning("MoA aggregation failed: %s", exc)
            # Fallback: return longest individual answer
            return max(answers, key=len)

    @classmethod
    def from_population(
        cls,
        population: list[Agent],
        aggregator_backend: "LLMBackend | None" = None,
        **kwargs: Any,
    ) -> "MixtureOfAgents":
        """Convenience constructor from a list of agents.

        Args:
            population: Full population of agents.
            aggregator_backend: Backend for aggregation.
            **kwargs: Additional kwargs forwarded to :class:`MixtureOfAgents`.
        """
        return cls(
            agents=population,
            aggregator_backend=aggregator_backend,
            **kwargs,
        )


# ── QuantumTunneler ────────────────────────────────────────────────────────────


@dataclass
class TunnelEvent:
    """Record of a quantum tunneling event.

    Attributes:
        generation: Generation when tunneling occurred.
        agent_id: ID of the agent that tunneled.
        old_fitness: Fitness before tunneling.
        genome_id: Genome ID of the new genome.
    """

    generation: int
    agent_id: str
    old_fitness: float
    genome_id: str


class QuantumTunneler:
    """Occasionally makes large random genome jumps to escape local optima.

    At each generation, each agent has a ``tunnel_prob`` chance of being
    replaced by a fresh random genome (a "quantum tunneling" event).

    Higher ``tunnel_prob`` = more exploration but destabilises good solutions.
    Lower ``tunnel_prob`` = more stable but may get stuck.

    Typical use: tunnel_prob=0.05 to 0.15 (5%–15% chance per agent per gen).

    Args:
        tunnel_prob: Probability of tunneling per agent per generation. Default 0.1.
        protect_elites: If True, the top ``n_elites`` agents are never tunneled.
            Default True.
        n_elites: Number of elite agents to protect. Default 1.
        seed: Optional random seed.
    """

    def __init__(
        self,
        tunnel_prob: float = 0.1,
        protect_elites: bool = True,
        n_elites: int = 1,
        seed: "int | None" = None,
    ) -> None:
        self._prob = max(0.0, min(1.0, tunnel_prob))
        self._protect_elites = protect_elites
        self._n_elites = n_elites
        self._rng = random.Random(seed)
        self._events: list[TunnelEvent] = []
        self._generation: int = 0

    @property
    def events(self) -> list[TunnelEvent]:
        """History of all tunneling events."""
        return list(self._events)

    @property
    def tunnel_count(self) -> int:
        """Total number of tunneling events that have occurred."""
        return len(self._events)

    def apply(
        self,
        population: list[Agent],
        model: str = "gpt-4o-mini",
    ) -> list[Agent]:
        """Apply quantum tunneling to a population.

        Args:
            population: Current list of agents (sorted by fitness descending,
                if ``protect_elites=True``).
            model: Model name for freshly generated genomes.

        Returns:
            The same list with some agents replaced by fresh random genomes.
        """
        n_protect = self._n_elites if self._protect_elites else 0
        result: list[Agent] = []

        for i, agent in enumerate(population):
            if i < n_protect:
                # Protected elite — never tunnel
                result.append(agent)
                continue

            if self._rng.random() < self._prob:
                # Quantum tunnel: replace with fresh random genome
                new_genome = self._random_genome(model)
                old_fitness = agent.fitness or 0.0
                new_agent = Agent(
                    genome=new_genome,
                    backend=agent.backend,
                    agent_id=str(uuid.uuid4())[:8],
                )
                event = TunnelEvent(
                    generation=self._generation,
                    agent_id=new_agent.id,
                    old_fitness=old_fitness,
                    genome_id=new_genome.genome_id,
                )
                self._events.append(event)
                logger.debug(
                    "QuantumTunnel gen=%d agent=%s old_fitness=%.4f → fresh genome",
                    self._generation,
                    agent.id,
                    old_fitness,
                )
                result.append(new_agent)
            else:
                result.append(agent)

        self._generation += 1
        return result

    @staticmethod
    def _random_genome(model: str) -> Genome:
        """Generate a random genome with diverse parameters."""
        strategies = [
            "chain-of-thought",
            "step-by-step",
            "concise",
            "socratic",
            "direct",
            "exploratory",
            "systematic",
            "creative",
        ]
        prompts = [
            "You are a precise and analytical AI assistant.",
            "You are a creative problem solver who thinks outside the box.",
            "You are an expert who gives direct, concise answers.",
            "You reason step by step, checking each step carefully.",
            "You break problems into smaller parts and solve each methodically.",
            "You use analogies and examples to explain complex ideas.",
        ]
        rng = random.Random()
        return Genome(
            system_prompt=rng.choice(prompts),
            strategy=rng.choice(strategies),
            temperature=round(rng.uniform(0.3, 1.2), 2),
            model=model,
            genome_id=str(uuid.uuid4())[:8],
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of tunneling activity."""
        return {
            "tunnel_prob": self._prob,
            "total_events": self.tunnel_count,
            "generations_run": self._generation,
            "events_per_gen": (
                self.tunnel_count / self._generation if self._generation > 0 else 0.0
            ),
        }
