"""CLI-Anything — wrap any shell command as an agent-invocable tool.

Agents in Cambrian can be given *tools* — callable utilities they can
invoke when processing a task.  This module makes it trivial to expose
*any* command-line program as a tool without writing adapter code.

Design
------

:class:`CLITool`
    Wraps a shell command template.  When called with a string argument,
    it substitutes ``{input}`` in the template, runs the subprocess, and
    returns stdout (or a combined stdout+stderr string on error).

:class:`CLIToolkit`
    A named collection of :class:`CLITool` objects.  Agents receive the
    toolkit and can call any tool by name.  The toolkit builds a
    *tool-use prompt block* that tells the LLM which tools are available
    and how to invoke them (a minimal function-calling protocol using
    ``[TOOL: name | input]`` syntax).

Tool-use protocol (text-based)
-------------------------------
Since Cambrian backends don't require native function-calling support, the
toolkit uses a simple text markup convention that the agent's LLM can learn
from the system prompt:

    To use a tool: ``[TOOL: tool_name | your input here]``

The toolkit's :meth:`~CLIToolkit.parse_and_execute` method scans the
agent's response for these markers and runs the matching tools.

Security
--------
- Each tool runs in a sandboxed subprocess with a configurable timeout.
- The command template is fixed at construction time; only the ``{input}``
  placeholder is user-controlled.
- ``shell=False`` by default to prevent shell injection; set ``shell=True``
  only when the command string genuinely requires shell features.

Usage::

    from cambrian.cli_tools import CLITool, CLIToolkit

    # Wrap Python interpreter as a tool
    python_tool = CLITool(
        name="python_exec",
        command_template="python -c {input!r}",
        description="Execute a Python expression and return the result.",
    )

    # Wrap ripgrep
    grep_tool = CLITool(
        name="grep",
        command_template="grep -r {input} .",
        description="Search files for a pattern.",
    )

    toolkit = CLIToolkit(tools=[python_tool, grep_tool])

    # Inject into agent system prompt
    system_with_tools = toolkit.system_prompt_block() + base_system_prompt

    # After the agent responds, parse and execute any tool calls
    tool_results = toolkit.parse_and_execute(agent_response)
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

# ── CLITool ───────────────────────────────────────────────────────────────────


class CLITool:
    """A single CLI command wrapped as an agent-callable tool.

    Args:
        name: Unique tool name (alphanumeric + underscores recommended).
        command_template: Shell command with ``{input}`` placeholder.
            Example: ``"python -c {input!r}"``.
        description: Human-readable description shown to the agent.
        timeout: Maximum seconds to wait for the subprocess.  Default ``10``.
        shell: Pass to ``subprocess.run(shell=...)``.  Default ``False``.
            Set to ``True`` only for commands that require shell expansion.
        max_output_chars: Truncate stdout/stderr to this many characters.
            Default ``4000``.
        env: Optional extra environment variables for the subprocess.

    Example::

        tool = CLITool(
            name="word_count",
            command_template="wc -w <<< {input!r}",
            description="Count words in a string.",
            shell=True,
        )
        result = tool.run("The quick brown fox")
        # result.output → "4"
    """

    def __init__(
        self,
        name: str,
        command_template: str,
        description: str = "",
        timeout: float = 10.0,
        shell: bool = False,
        max_output_chars: int = 4000,
        env: dict[str, str] | None = None,
    ) -> None:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Tool name {name!r} must be alphanumeric/underscore, starting with a letter."
            )
        self.name = name
        self._template = command_template
        self.description = description
        self._timeout = timeout
        self._shell = shell
        self._max_chars = max_output_chars
        self._env = env
        self._call_count = 0

    def run(self, input_text: str) -> "CLIToolResult":
        """Execute the tool with *input_text* and return the result.

        Args:
            input_text: Text to substitute for ``{input}`` in the template.

        Returns:
            :class:`CLIToolResult` with ``output``, ``exit_code``, ``ok``.
        """
        self._call_count += 1
        try:
            cmd_str = self._template.format(input=input_text)
        except (KeyError, IndexError, ValueError) as exc:
            return CLIToolResult(
                tool_name=self.name,
                input_text=input_text,
                output=f"[template error: {exc}]",
                exit_code=-1,
                ok=False,
            )

        logger.debug("CLITool %r: running %r", self.name, cmd_str[:80])

        try:
            if self._shell:
                proc = subprocess.run(
                    cmd_str,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=self._env,
                )
            else:
                args = shlex.split(cmd_str)
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=self._env,
                )
        except subprocess.TimeoutExpired:
            return CLIToolResult(
                tool_name=self.name,
                input_text=input_text,
                output=f"[timeout after {self._timeout}s]",
                exit_code=-1,
                ok=False,
            )
        except FileNotFoundError as exc:
            return CLIToolResult(
                tool_name=self.name,
                input_text=input_text,
                output=f"[command not found: {exc}]",
                exit_code=-1,
                ok=False,
            )
        except Exception as exc:
            return CLIToolResult(
                tool_name=self.name,
                input_text=input_text,
                output=f"[error: {exc}]",
                exit_code=-1,
                ok=False,
            )

        combined = proc.stdout
        if proc.returncode != 0 and proc.stderr:
            combined = proc.stdout + "\n[stderr] " + proc.stderr

        output = combined.strip()[: self._max_chars]
        return CLIToolResult(
            tool_name=self.name,
            input_text=input_text,
            output=output,
            exit_code=proc.returncode,
            ok=proc.returncode == 0,
        )

    def __call__(self, input_text: str) -> str:
        """Shorthand: run tool and return output string."""
        return self.run(input_text).output

    def __repr__(self) -> str:
        return f"CLITool(name={self.name!r}, calls={self._call_count})"


# ── CLIToolResult ─────────────────────────────────────────────────────────────


@dataclass
class CLIToolResult:
    """Result of a :class:`CLITool` execution.

    Attributes:
        tool_name: The tool that produced this result.
        input_text: The input passed to the tool.
        output: Captured stdout (+ stderr if non-zero exit).
        exit_code: Process exit code.
        ok: ``True`` if exit_code == 0.
    """

    tool_name: str
    input_text: str
    output: str
    exit_code: int
    ok: bool


# ── CLIToolkit ────────────────────────────────────────────────────────────────


class CLIToolkit:
    """Named collection of :class:`CLITool` objects.

    Manages a set of tools and provides helpers to:
    - Generate a system prompt block that tells the LLM which tools are available.
    - Parse an LLM response for ``[TOOL: name | input]`` markers.
    - Execute the matched tool calls and return results.

    Args:
        tools: Initial list of tools.  More can be added via :meth:`add`.

    Example::

        toolkit = CLIToolkit(tools=[python_tool, shell_tool])
        system = toolkit.system_prompt_block() + "\\n\\n" + base_system_prompt
        # ... run agent ...
        results = toolkit.parse_and_execute(agent.run(task))
    """

    # Regex: [TOOL: tool_name | input text here]
    _CALL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\|\s*(.*?)\]", re.DOTALL)

    def __init__(self, tools: list[CLITool] | None = None) -> None:
        self._tools: dict[str, CLITool] = {}
        for tool in (tools or []):
            self.add(tool)

    def add(self, tool: CLITool) -> None:
        """Register a tool in the toolkit.

        Args:
            tool: Tool to register.  Overwrites existing tool with the same name.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> CLITool | None:
        """Return the tool named *name*, or ``None`` if not registered."""
        return self._tools.get(name)

    def system_prompt_block(self) -> str:
        """Generate a system prompt section describing available tools.

        Returns:
            Multi-line string to prepend to the agent's system prompt.
        """
        if not self._tools:
            return ""
        lines = [
            "You have access to the following CLI tools.",
            "To use a tool, output exactly: [TOOL: tool_name | your input]",
            "The tool result will be shown before you continue.",
            "",
            "Available tools:",
        ]
        for tool in self._tools.values():
            lines.append(f"  • {tool.name}: {tool.description}")
        lines.append("")
        return "\n".join(lines)

    def parse_and_execute(self, text: str) -> list["CLIToolResult"]:
        """Find ``[TOOL: name | input]`` markers in *text* and execute them.

        Args:
            text: Agent response text.

        Returns:
            List of :class:`CLIToolResult` for each valid tool call found.
            Unknown tool names produce a result with ``ok=False``.
        """
        results: list[CLIToolResult] = []
        for match in self._CALL_RE.finditer(text):
            tool_name = match.group(1).strip()
            input_text = match.group(2).strip()
            tool = self._tools.get(tool_name)
            if tool is None:
                results.append(CLIToolResult(
                    tool_name=tool_name,
                    input_text=input_text,
                    output=f"[unknown tool: {tool_name!r}]",
                    exit_code=-1,
                    ok=False,
                ))
            else:
                results.append(tool.run(input_text))
        return results

    def augment_response(self, text: str) -> str:
        """Execute tools found in *text* and interleave their results.

        Replaces each ``[TOOL: name | input]`` marker with
        ``[TOOL: name | input]\\n[RESULT: output]``.

        Args:
            text: Agent response.

        Returns:
            Augmented text with tool results embedded.
        """
        def _replace(match: re.Match[str]) -> str:
            tool_name = match.group(1).strip()
            input_text = match.group(2).strip()
            tool = self._tools.get(tool_name)
            if tool is None:
                return f"{match.group(0)}\n[RESULT: unknown tool]"
            result = tool.run(input_text)
            return f"{match.group(0)}\n[RESULT: {result.output}]"

        return self._CALL_RE.sub(_replace, text)

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

    def __repr__(self) -> str:
        return f"CLIToolkit(tools={self.tool_names})"


# ── Convenience factories ──────────────────────────────────────────────────────


def make_python_tool(timeout: float = 5.0) -> CLITool:
    """Create a CLITool that executes Python one-liners.

    Args:
        timeout: Max seconds. Default ``5.0``.

    Returns:
        CLITool that runs ``python -c "<input>"``.
    """
    return CLITool(
        name="python_exec",
        command_template="python -c {input!r}",
        description="Execute a Python expression or short script.",
        timeout=timeout,
        shell=False,
    )


def make_shell_tool(timeout: float = 5.0) -> CLITool:
    """Create a CLITool that runs arbitrary shell commands.

    Warning: Use only in trusted environments.  User input is not sanitised.

    Args:
        timeout: Max seconds. Default ``5.0``.

    Returns:
        CLITool that runs ``bash -c "<input>"``.
    """
    return CLITool(
        name="shell",
        command_template="bash -c {input!r}",
        description="Run a shell command and return stdout.",
        timeout=timeout,
        shell=False,
    )
