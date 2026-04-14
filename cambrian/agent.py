"""Agent and Genome — the core evolutionary unit of Cambrian.

A :class:`Genome` encodes the full specification of an AI agent (prompt,
tools, strategy, hyperparameters). An :class:`Agent` wraps a Genome with a
backend and exposes :meth:`~Agent.run` + :attr:`~Agent.fitness`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from cambrian.backends.base import LLMBackend


@dataclass
class Genome:
    """Evolvable specification of an AI agent.

    All fields are mutable and may be altered by the mutator or crossover
    operators to produce offspring.

    Attributes:
        system_prompt: System-level instructions that shape the agent's
            behaviour, persona, and output style.
        tools: List of tool/function names the agent is allowed to use.
            Currently used as metadata for the mutation prompt; function
            calling is future work.
        strategy: High-level reasoning strategy hint (e.g. ``"chain-of-thought"``,
            ``"step-by-step"``, ``"concise"``, ``"socratic"``).
        temperature: Sampling temperature (0.0–2.0). Lower = deterministic.
        model: Model identifier passed to the backend (e.g. ``"gpt-4o-mini"``).
        genome_id: Unique identifier, auto-generated if not supplied.
    """

    system_prompt: str = "You are a helpful AI assistant."
    tools: list[str] = field(default_factory=list)
    strategy: str = "step-by-step"
    temperature: float = 0.7
    model: str = "gpt-4o-mini"
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        """Serialise the genome to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        """Deserialise a genome from a plain dictionary."""
        return cls(
            system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
            tools=data.get("tools", []),
            strategy=data.get("strategy", "step-by-step"),
            temperature=float(data.get("temperature", 0.7)),
            model=data.get("model", "gpt-4o-mini"),
            genome_id=data.get("genome_id", str(uuid.uuid4())[:8]),
        )

    def token_count(self) -> int:
        """Rough token estimate for the system prompt (4 chars ≈ 1 token)."""
        return len(self.system_prompt) // 4

    def __str__(self) -> str:
        return (
            f"Genome(id={self.genome_id}, model={self.model}, "
            f"temp={self.temperature}, strategy={self.strategy!r})"
        )


class Agent:
    """Wraps a :class:`Genome` with an LLM backend and tracks fitness.

    Args:
        genome: The agent's genetic specification.
        backend: LLM backend used to generate responses. Optional — if ``None``
            the agent can still be inspected and mutated, but :meth:`run` will
            raise :exc:`RuntimeError`.
        agent_id: Unique identifier, auto-generated if not supplied.
    """

    def __init__(
        self,
        genome: Genome,
        backend: "LLMBackend | None" = None,
        agent_id: str | None = None,
    ) -> None:
        self.genome = genome
        self.backend = backend
        self.agent_id: str = agent_id or str(uuid.uuid4())[:8]
        self._fitness: float | None = None
        self._generation: int = 0

    # Convenience alias used throughout the codebase
    @property
    def id(self) -> str:
        """Alias for :attr:`agent_id`."""
        return self.agent_id

    def run(self, task: str) -> str:
        """Generate a response to *task* using this agent's genome + backend.

        Incorporates the strategy hint into the user message if the genome's
        strategy is not "default".

        Returns:
            The LLM's text response.

        Raises:
            RuntimeError: If no backend was provided at construction time.
        """
        if self.backend is None:
            raise RuntimeError(
                "Agent has no backend — pass a LLMBackend to Agent() or "
                "EvolutionEngine(backend=...)"
            )
        prompt = task
        if self.genome.strategy and self.genome.strategy != "default":
            prompt = f"{task}\n\n[Approach: {self.genome.strategy}]"

        return self.backend.generate(
            prompt,
            system=self.genome.system_prompt,
            temperature=self.genome.temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise agent state (id, fitness, genome) to a plain dict."""
        return {
            "id": self.agent_id,
            "generation": self._generation,
            "fitness": self._fitness,
            "genome": self.genome.to_dict(),
        }

    @property
    def fitness(self) -> float | None:
        """Last recorded fitness score, or ``None`` if not yet evaluated."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = float(value)

    @property
    def generation(self) -> int:
        """The evolution generation in which this agent was created."""
        return self._generation

    @generation.setter
    def generation(self, value: int) -> None:
        self._generation = int(value)

    def clone(self) -> "Agent":
        """Return a deep copy of this agent with a fresh agent_id and no fitness."""
        new_genome = Genome.from_dict(self.genome.to_dict())
        new_genome.genome_id = str(uuid.uuid4())[:8]
        clone = Agent(new_genome, self.backend)
        clone._generation = self._generation
        # fitness intentionally NOT copied — clone must be re-evaluated
        return clone

    def __repr__(self) -> str:
        fitness_str = f"{self._fitness:.4f}" if self._fitness is not None else "None"
        return (
            f"Agent(id={self.agent_id}, gen={self._generation}, "
            f"fitness={fitness_str}, genome={self.genome})"
        )
