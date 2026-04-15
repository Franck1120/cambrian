# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Agent and Genome — the core evolutionary unit of Cambrian.

A :class:`Genome` encodes the full specification of an AI agent (prompt,
tools, strategy, hyperparameters). An :class:`Agent` wraps a Genome with a
backend and exposes :meth:`~Agent.run` + :attr:`~Agent.fitness`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

from cambrian.backends.base import LLMBackend

if TYPE_CHECKING:
    from cambrian.cli_tools import CLITool


@dataclass
class ToolSpec:
    """Specification for an agent-invented tool.

    Attributes:
        name: Unique tool identifier (alphanumeric + underscores).
        description: What the tool does (shown in system prompt).
        command_template: Shell command with {input} placeholder.
        shell: Whether to run in a shell. Default False.
        timeout: Max seconds. Default 10.0.
        author_genome_id: ID of the genome that invented this tool.
    """

    name: str
    description: str
    command_template: str
    shell: bool = False
    timeout: float = 10.0
    author_genome_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "command_template": self.command_template,
            "shell": self.shell,
            "timeout": self.timeout,
            "author_genome_id": self.author_genome_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSpec":
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            command_template=str(data["command_template"]),
            shell=bool(data.get("shell", False)),
            timeout=float(data.get("timeout", 10.0)),
            author_genome_id=str(data.get("author_genome_id", "")),
        )

    def to_cli_tool(self) -> "CLITool":
        from cambrian.cli_tools import CLITool
        return CLITool(
            name=self.name,
            command_template=self.command_template,
            description=self.description,
            timeout=self.timeout,
            shell=self.shell,
        )


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
    few_shot_examples: list[dict[str, Any]] = field(default_factory=list)
    tool_specs: list[ToolSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the genome to a plain dictionary."""
        base = asdict(self)
        # asdict recurses into dataclasses — replace tool_specs with our own
        # serialisation to keep the shape consistent and explicit.
        base["tool_specs"] = [ts.to_dict() for ts in self.tool_specs]
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        """Deserialise a genome from a plain dictionary."""
        raw_specs = data.get("tool_specs", [])
        tool_specs = [ToolSpec.from_dict(ts) for ts in raw_specs if isinstance(ts, dict)]
        return cls(
            system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
            tools=data.get("tools", []),
            strategy=data.get("strategy", "step-by-step"),
            temperature=float(data.get("temperature", 0.7)),
            model=data.get("model", "gpt-4o-mini"),
            genome_id=data.get("genome_id", str(uuid.uuid4())[:8]),
            few_shot_examples=list(data.get("few_shot_examples", [])),
            tool_specs=tool_specs,
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

        system = self.genome.system_prompt
        # Inject Lamarckian few-shot examples into the system context
        if self.genome.few_shot_examples:
            examples_text = "\n".join(
                f"Example {i+1} (score {ex.get('score', '?')}):\n"
                f"  Task: {ex.get('task', '')}\n"
                f"  Response: {ex.get('response', '')}"
                for i, ex in enumerate(self.genome.few_shot_examples[:3])
                if ex.get("response")
            )
            if examples_text:
                system = f"{system}\n\n--- Successful examples ---\n{examples_text}"

        return self.backend.generate(
            prompt,
            system=system,
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
