# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Dynamic tool creation — agents invent new CLI tools during evolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cambrian.agent import Agent, Genome, ToolSpec
from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── ToolInventionResult ───────────────────────────────────────────────────────


@dataclass
class ToolInventionResult:
    """Outcome of a single tool-invention attempt.

    Attributes:
        tool_spec: The invented :class:`~cambrian.agent.ToolSpec`.
        test_input: Sample input the LLM suggested for testing.
        success: Whether the tool ran without error (exit_code == 0).
        test_output: Captured output from the test run.
    """

    tool_spec: ToolSpec
    test_input: str
    success: bool
    test_output: str


# ── ToolInventor ──────────────────────────────────────────────────────────────


class ToolInventor:
    """Uses an LLM to invent new CLI tool specs for a given task.

    Args:
        backend: LLM backend used to generate tool specs.
        max_tools_per_agent: Maximum number of tool specs an agent may carry.
            When the limit is exceeded the oldest tools are dropped (newest
            kept).  Default ``5``.
    """

    _NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    # Field-extraction regexes — case-insensitive, tolerate surrounding whitespace
    _FIELD_RE: dict[str, re.Pattern[str]] = {
        "name": re.compile(r"^\s*NAME\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
        "description": re.compile(
            r"^\s*DESCRIPTION\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE
        ),
        "command": re.compile(
            r"^\s*COMMAND\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE
        ),
        "shell": re.compile(r"^\s*SHELL\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
        "test_input": re.compile(
            r"^\s*TEST_INPUT\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE
        ),
    }

    _INVENTION_PROMPT = """\
You are helping design a new command-line tool that an AI agent can invoke.

The agent is working on the following task:
{task}

The agent already has these tool specs registered:
{existing_tools}

Design ONE new CLI tool that would be genuinely useful for the task above and
is not already covered by the existing tools.

Respond with ONLY the following fields (no extra text, no markdown fences):

NAME: <tool_name — alphanumeric and underscores only, starts with a letter>
DESCRIPTION: <one-line description of what the tool does>
COMMAND: <shell command with {{input}} as the placeholder for user input>
SHELL: <true or false — set true only if the command requires shell features>
TEST_INPUT: <a realistic sample input to test the tool>
"""

    def __init__(
        self,
        backend: LLMBackend,
        max_tools_per_agent: int = 5,
    ) -> None:
        self._backend = backend
        self._max_tools = max_tools_per_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invent_tool(self, agent: Agent, task: str) -> ToolInventionResult | None:
        """Ask the LLM to invent a CLI tool useful for *task*.

        Prompts the LLM, parses the structured response, validates the tool
        name, instantiates a :class:`~cambrian.agent.ToolSpec`, and tests it
        by running the spec's CLI tool with the suggested test input.

        Args:
            agent: The agent requesting a new tool.  Its existing tool specs
                are listed in the prompt so the LLM avoids duplicates.
            task: Natural-language description of the task the tool should
                support.

        Returns:
            :class:`ToolInventionResult` on success, ``None`` if the LLM
            response could not be parsed or was otherwise invalid.
        """
        existing = (
            "\n".join(
                f"  - {ts.name}: {ts.description}"
                for ts in agent.genome.tool_specs
            )
            or "  (none)"
        )
        prompt = self._INVENTION_PROMPT.format(
            task=task,
            existing_tools=existing,
        )

        try:
            raw = self._backend.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ToolInventor: LLM call failed: %s", exc)
            return None

        spec = self._parse_spec(raw, author_genome_id=agent.genome.genome_id)
        if spec is None:
            logger.warning("ToolInventor: could not parse LLM response")
            return None

        # Extract TEST_INPUT from the raw response
        test_input_match = self._FIELD_RE["test_input"].search(raw)
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        # Attempt to run the tool to verify it works
        cli_tool = spec.to_cli_tool()
        result = cli_tool.run(test_input)

        logger.debug(
            "ToolInventor: invented %r (ok=%s) for genome %s",
            spec.name,
            result.ok,
            spec.author_genome_id,
        )

        return ToolInventionResult(
            tool_spec=spec,
            test_input=test_input,
            success=result.ok,
            test_output=result.output,
        )

    def inject_tools(self, agent: Agent, invented: list[ToolSpec]) -> Agent:
        """Return a new Agent whose genome carries *invented* tool specs.

        Appends the new specs to the existing ``tool_specs`` list, then
        truncates to :attr:`_max_tools` (keeping the newest entries).

        Args:
            agent: Source agent (not mutated).
            invented: Tool specs to add.

        Returns:
            A fresh :class:`~cambrian.agent.Agent` with an updated genome.
            The original agent is not modified.
        """
        new_genome = Genome.from_dict(agent.genome.to_dict())
        combined = new_genome.tool_specs + invented
        new_genome.tool_specs = combined[-self._max_tools :]
        new_agent = Agent(new_genome, backend=agent.backend)
        new_agent._generation = agent._generation
        return new_agent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_spec(self, text: str, author_genome_id: str) -> ToolSpec | None:
        """Parse LLM output into a :class:`~cambrian.agent.ToolSpec`.

        Args:
            text: Raw LLM response text.
            author_genome_id: Genome ID to embed in the spec.

        Returns:
            Parsed :class:`~cambrian.agent.ToolSpec`, or ``None`` if any
            required field is missing or the name fails validation.
        """
        fields: dict[str, str] = {}
        for key, pattern in self._FIELD_RE.items():
            match = pattern.search(text)
            if match:
                fields[key] = match.group(1).strip()

        name = fields.get("name", "")
        description = fields.get("description", "")
        command = fields.get("command", "")

        if not name or not description or not command:
            logger.debug(
                "ToolInventor._parse_spec: missing required fields "
                "(name=%r, description=%r, command=%r)",
                name,
                description,
                command,
            )
            return None

        if not self._NAME_RE.match(name):
            logger.debug(
                "ToolInventor._parse_spec: invalid tool name %r", name
            )
            return None

        shell_raw = fields.get("shell", "false").lower()
        shell = shell_raw in {"true", "yes", "1"}

        return ToolSpec(
            name=name,
            description=description,
            command_template=command,
            shell=shell,
            author_genome_id=author_genome_id,
        )


# ── ToolPopulationRegistry ────────────────────────────────────────────────────


class ToolPopulationRegistry:
    """Shared registry of all tools invented across the population.

    Thread-safety is *not* guaranteed — intended for single-threaded
    evolutionary loops.  Wrap with a lock if using concurrent workers.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Add a tool to the registry.

        Silently ignores duplicates (same ``name``).

        Args:
            spec: Tool specification to register.
        """
        if spec.name not in self._tools:
            self._tools[spec.name] = spec
            logger.debug(
                "ToolPopulationRegistry: registered %r (author=%s)",
                spec.name,
                spec.author_genome_id,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """Return the tool named *name*, or ``None`` if not registered.

        Args:
            name: Tool identifier.
        """
        return self._tools.get(name)

    def all_tools(self) -> list[ToolSpec]:
        """Return all registered tools in insertion order."""
        return list(self._tools.values())

    def top_tools(self, n: int = 5) -> list[ToolSpec]:
        """Return up to *n* tools, sorted alphabetically by author genome ID.

        The alphabetical sort on ``author_genome_id`` is a lightweight proxy
        for ranking until a proper fitness-based sort is available.

        Args:
            n: Maximum number of tools to return.
        """
        sorted_tools = sorted(
            self._tools.values(), key=lambda ts: ts.author_genome_id
        )
        return sorted_tools[:n]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_toolkit_block(self) -> str:
        """Format registered tools as a system prompt block.

        Returns:
            Multi-line string listing all registered tools, ready to prepend
            to an agent's system prompt.  Returns an empty string when there
            are no registered tools.
        """
        if not self._tools:
            return ""
        lines = [
            "=== Population Tool Registry ===",
            "The following tools were invented by agents in this population.",
            "To use a tool: [TOOL: tool_name | your input]",
            "",
        ]
        for spec in self._tools.values():
            lines.append(f"  • {spec.name}: {spec.description}")
        lines.append("=================================")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the registry to a plain dictionary."""
        return {
            "tools": {name: spec.to_dict() for name, spec in self._tools.items()}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPopulationRegistry":
        """Deserialise a registry from a plain dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.
        """
        registry = cls()
        for spec_data in data.get("tools", {}).values():
            if isinstance(spec_data, dict):
                registry.register(ToolSpec.from_dict(spec_data))
        return registry

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolPopulationRegistry(tools={list(self._tools.keys())})"
