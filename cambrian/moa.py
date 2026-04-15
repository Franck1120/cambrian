"""Mixture of Agents (MoA) and Quantum Tunneling for Cambrian.

Mixture of Agents (Technique 36)
---------------------------------
A *mixture of agents* runs N independent LLM agents on the same task and then
uses an **aggregator** to synthesise the best answer from all responses.  This
reduces variance, improves reliability on complex tasks, and naturally produces
diverse answer sets for the evaluator to score.

:class:`MixtureOfAgents`:
    Given a list of :class:`~cambrian.agent.Agent` objects (possibly with different
    genomes), runs each one on the task and calls an aggregator LLM to combine
    the answers.

Quantum Tunneling (Technique 17)
---------------------------------
In quantum mechanics, particles occasionally tunnel through energy barriers they
classically cannot cross.  Analogously, :class:`QuantumTunneler` occasionally
replaces an agent with a completely randomised genome, allowing the evolutionary
search to escape deep local optima.

:class:`QuantumTunneler`:
    Given a probability ``tunnel_prob``, each call to :meth:`maybe_tunnel`
    either returns the original agent unchanged or replaces its genome with a
    fresh randomised one.  The randomisation perturbs temperature, strategy,
    and system prompt — producing a large jump in genome space.

Usage::

    from cambrian.moa import MixtureOfAgents, QuantumTunneler
    from cambrian.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    moa = MixtureOfAgents(agents=[a1, a2, a3], backend=backend)
    answer = moa.run("Explain how neural networks work")

    tunneler = QuantumTunneler(tunnel_prob=0.05)
    for agent in population:
        agent = tunneler.maybe_tunnel(agent, task)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent
    from cambrian.backends.base import LLMBackend

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_AGGREGATOR_SYSTEM = """\
You are an expert synthesiser.  You will receive multiple answers to the same
question produced by different AI agents.  Your task is to combine the strongest
elements from all responses into a single, authoritative, high-quality answer.

- Preserve correct information from any source.
- Discard incorrect or redundant content.
- Produce a coherent, well-structured final answer.
- Do not mention that you are synthesising multiple answers.
"""

_AGGREGATOR_TEMPLATE = """\
Question / Task:
{task}

Agent responses:
{responses}

Synthesise the best answer from all responses above.
"""

_TUNNEL_SYSTEM = """\
You are a creative AI prompt engineer.  Generate a completely novel system
prompt for an AI agent that will attempt to solve the task below.  The prompt
should be surprising, unconventional, and explore an approach very different
from standard methods.

Return ONLY the system prompt text — no explanation, no headers.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MixtureOfAgents
# ─────────────────────────────────────────────────────────────────────────────


class MixtureOfAgents:
    """Run N agents on a task and aggregate their answers.

    Args:
        agents: List of :class:`~cambrian.agent.Agent` objects to query.
        backend: LLM backend used for the aggregator call.
        aggregator_temperature: Temperature for the aggregation call. Default ``0.2``.
        max_response_chars: Truncate each agent response to this many characters
            before aggregation to keep the context manageable. Default ``2000``.
    """

    def __init__(
        self,
        agents: "list[Agent]",
        backend: "LLMBackend",
        aggregator_temperature: float = 0.2,
        max_response_chars: int = 2000,
    ) -> None:
        if not agents:
            raise ValueError("MixtureOfAgents requires at least one agent")
        self._agents = agents
        self._backend = backend
        self._agg_temp = aggregator_temperature
        self._max_chars = max_response_chars
        self._last_responses: list[str] = []

    @property
    def last_responses(self) -> list[str]:
        """Individual agent responses from the most recent :meth:`run` call."""
        return list(self._last_responses)

    def run(self, task: str) -> str:
        """Run all agents on *task* and return the aggregated answer.

        Each agent uses its genome's ``system_prompt`` and ``temperature``.
        The aggregator combines all responses into a single answer.

        Args:
            task: The task / question to send to all agents.

        Returns:
            Aggregated answer string.
        """
        responses: list[str] = []
        for i, agent in enumerate(self._agents):
            try:
                raw = self._backend.generate(
                    task,
                    system=agent.genome.system_prompt,
                    temperature=agent.genome.temperature,
                )
                response = raw[: self._max_chars]
            except Exception as exc:
                logger.warning("MoA agent %d failed: %s", i, exc)
                response = ""
            responses.append(response)
            logger.debug("MoA agent %d responded (%d chars)", i, len(response))

        self._last_responses = responses

        if not any(responses):
            logger.warning("MoA: all agents returned empty responses")
            return ""

        formatted = "\n\n".join(
            f"[Agent {i + 1}]:\n{r}" for i, r in enumerate(responses) if r
        )
        agg_prompt = _AGGREGATOR_TEMPLATE.format(task=task, responses=formatted)

        try:
            answer = self._backend.generate(
                agg_prompt,
                system=_AGGREGATOR_SYSTEM,
                temperature=self._agg_temp,
            )
        except Exception as exc:
            logger.warning("MoA aggregator failed: %s — returning best individual response", exc)
            answer = max(responses, key=len)  # longest ≈ most detailed

        return answer


# ─────────────────────────────────────────────────────────────────────────────
# QuantumTunneler
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGIES = [
    "chain-of-thought",
    "step-by-step",
    "adversarial",
    "Socratic",
    "analogical",
    "first-principles",
    "contrarian",
    "socratic",
    "lateral-thinking",
    "devil's-advocate",
]


class QuantumTunneler:
    """Stochastic large-jump mutation for escaping local optima.

    With probability ``tunnel_prob``, replaces an agent's genome with a
    completely randomised variant — a large jump in genome space inspired by
    quantum tunneling through energy barriers.

    Args:
        tunnel_prob: Probability of tunneling per agent per call. Default ``0.05``.
        backend: Optional LLM backend for generating a novel system prompt.
            If ``None``, the tunneled prompt is a generic placeholder.
        temperature_range: Range for the randomised temperature. Default ``(0.2, 1.8)``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        tunnel_prob: float = 0.05,
        backend: "LLMBackend | None" = None,
        temperature_range: tuple[float, float] = (0.2, 1.8),
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= tunnel_prob <= 1.0:
            raise ValueError(f"tunnel_prob must be in [0.0, 1.0], got {tunnel_prob}")
        self._prob = tunnel_prob
        self._backend = backend
        self._t_min, self._t_max = temperature_range
        self._tunnel_count = 0
        if seed is not None:
            random.seed(seed)

    @property
    def tunnel_count(self) -> int:
        """Total number of tunneling events that have occurred."""
        return self._tunnel_count

    def maybe_tunnel(self, agent: "Agent", task: str = "") -> "Agent":
        """Possibly replace *agent*'s genome with a randomised variant.

        Args:
            agent: Agent to potentially tunnel.
            task: Task description used for LLM-generated prompt (if backend set).

        Returns:
            Either the original *agent* (unchanged) or a new :class:`~cambrian.agent.Agent`
            with a randomised genome.
        """
        if random.random() > self._prob:
            return agent  # No tunneling

        return self._tunnel(agent, task)

    def tunnel_all(self, population: "list[Agent]", task: str = "") -> "list[Agent]":
        """Apply :meth:`maybe_tunnel` to every agent in *population*.

        Args:
            population: List of agents.
            task: Task description.

        Returns:
            New list where each agent may have been tunneled.
        """
        return [self.maybe_tunnel(a, task) for a in population]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _tunnel(self, agent: "Agent", task: str) -> "Agent":
        """Produce a tunneled clone of *agent* with a randomised genome."""
        from cambrian.agent import Agent as _Agent, Genome as _Genome

        new_genome = _Genome.from_dict(agent.genome.to_dict())
        new_genome.temperature = random.uniform(self._t_min, self._t_max)
        new_genome.strategy = random.choice(_STRATEGIES)

        if self._backend is not None and task:
            try:
                new_prompt = self._backend.generate(
                    f"Task: {task}",
                    system=_TUNNEL_SYSTEM,
                    temperature=1.2,  # High temp for creativity
                )
                new_genome.system_prompt = new_prompt.strip()
            except Exception as exc:
                logger.warning("QuantumTunneler LLM prompt failed: %s — using random strategy", exc)
                new_genome.system_prompt = (
                    f"Approach this task using {new_genome.strategy} reasoning."
                )
        else:
            new_genome.system_prompt = (
                f"Approach this task using {new_genome.strategy} reasoning."
            )

        tunneled = _Agent(genome=new_genome)
        tunneled._generation = agent._generation
        self._tunnel_count += 1
        logger.debug(
            "QuantumTunneler: tunneled agent %s → %s (temp=%.2f, strategy=%s)",
            agent.id[:8], tunneled.id[:8], new_genome.temperature, new_genome.strategy,
        )
        return tunneled
