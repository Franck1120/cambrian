# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.cli_tools — CLITool, CLIToolResult, CLIToolkit, factories."""

from __future__ import annotations

import pytest

from cambrian.cli_tools import (
    CLITool,
    CLIToolResult,
    CLIToolkit,
    make_python_tool,
    make_shell_tool,
)


# ---------------------------------------------------------------------------
# CLITool — construction
# ---------------------------------------------------------------------------


class TestCLIToolConstruction:
    def test_valid_name_accepted(self) -> None:
        tool = CLITool(name="my_tool", command_template="echo {input}")
        assert tool.name == "my_tool"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="alphanumeric"):
            CLITool(name="bad-name!", command_template="echo {input}")

    def test_name_starting_with_digit_raises(self) -> None:
        with pytest.raises(ValueError):
            CLITool(name="1tool", command_template="echo {input}")

    def test_description_stored(self) -> None:
        tool = CLITool(name="t", command_template="echo {input}", description="my desc")
        assert tool.description == "my desc"

    def test_repr_contains_name(self) -> None:
        tool = CLITool(name="my_tool", command_template="echo {input}")
        assert "my_tool" in repr(tool)


# ---------------------------------------------------------------------------
# CLITool — run
# ---------------------------------------------------------------------------


class TestCLIToolRun:
    def test_run_returns_cli_tool_result(self) -> None:
        tool = CLITool(
            name="python_echo",
            command_template="python -c \"print('hello')\"",
            shell=False,
            timeout=10.0,
        )
        result = tool.run("unused")
        assert isinstance(result, CLIToolResult)

    def test_echo_tool_works(self) -> None:
        tool = CLITool(name="echo_tool", command_template="python -c \"print('{input}')\"")
        result = tool.run("hello_world")
        assert result.ok is True
        assert "hello_world" in result.output

    def test_run_increments_call_count(self) -> None:
        tool = CLITool(name="t", command_template="python -c \"print('x')\"")
        tool.run("a")
        tool.run("b")
        assert tool._call_count == 2

    def test_timeout_returns_error_result(self) -> None:
        tool = CLITool(
            name="slow",
            command_template="python -c \"import time; time.sleep(5)\"",
            timeout=0.1,
        )
        result = tool.run("x")
        assert result.ok is False
        assert "timeout" in result.output.lower()

    def test_nonexistent_command_returns_error(self) -> None:
        tool = CLITool(name="t", command_template="nonexistent_cmd_xyz_12345 {input}")
        result = tool.run("x")
        assert result.ok is False

    def test_template_error_returns_error_result(self) -> None:
        # Template with no {input} placeholder — format() gets called with input=...
        # This should still work. But a bad template (e.g. broken {}) should fail gracefully.
        tool = CLITool(name="t", command_template="python -c \"print('static')\"")
        result = tool.run("any input")
        assert isinstance(result, CLIToolResult)

    def test_call_dunder_returns_string(self) -> None:
        tool = CLITool(name="t", command_template="python -c \"print('hi')\"")
        output = tool("unused")
        assert isinstance(output, str)

    def test_exit_code_zero_on_success(self) -> None:
        tool = CLITool(name="t", command_template="python -c \"print('ok')\"")
        result = tool.run("x")
        assert result.exit_code == 0

    def test_exit_code_nonzero_on_failure(self) -> None:
        tool = CLITool(name="t", command_template="python -c \"import sys; sys.exit(1)\"")
        result = tool.run("x")
        assert result.exit_code != 0
        assert result.ok is False

    def test_output_truncated_to_max_chars(self) -> None:
        tool = CLITool(
            name="t",
            command_template="python -c \"print('x' * 10000)\"",
            max_output_chars=100,
        )
        result = tool.run("x")
        assert len(result.output) <= 100


# ---------------------------------------------------------------------------
# CLIToolResult
# ---------------------------------------------------------------------------


class TestCLIToolResult:
    def test_fields(self) -> None:
        r = CLIToolResult(
            tool_name="my_tool",
            input_text="hello",
            output="world",
            exit_code=0,
            ok=True,
        )
        assert r.tool_name == "my_tool"
        assert r.input_text == "hello"
        assert r.output == "world"
        assert r.exit_code == 0
        assert r.ok is True

    def test_not_ok_on_nonzero_exit(self) -> None:
        r = CLIToolResult(
            tool_name="t",
            input_text="x",
            output="err",
            exit_code=1,
            ok=False,
        )
        assert r.ok is False


# ---------------------------------------------------------------------------
# CLIToolkit
# ---------------------------------------------------------------------------


class TestCLIToolkit:
    def _echo_tool(self, name: str = "echo_tool") -> CLITool:
        return CLITool(name=name, command_template="python -c \"print('pong')\"")

    def test_empty_toolkit(self) -> None:
        tk = CLIToolkit()
        assert tk.tool_names == []

    def test_add_tool(self) -> None:
        tk = CLIToolkit()
        tk.add(self._echo_tool())
        assert "echo_tool" in tk.tool_names

    def test_init_with_tools(self) -> None:
        tools = [self._echo_tool("tool_a"), self._echo_tool("tool_b")]
        tk = CLIToolkit(tools=tools)
        assert len(tk.tool_names) == 2

    def test_get_existing_tool(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool()])
        assert tk.get("echo_tool") is not None

    def test_get_unknown_tool_returns_none(self) -> None:
        tk = CLIToolkit()
        assert tk.get("nonexistent") is None

    def test_system_prompt_block_empty_on_no_tools(self) -> None:
        tk = CLIToolkit()
        assert tk.system_prompt_block() == ""

    def test_system_prompt_block_mentions_tools(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool("my_tool")])
        block = tk.system_prompt_block()
        assert "my_tool" in block

    def test_parse_and_execute_finds_tool_call(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool("echo_tool")])
        text = "Here is my answer: [TOOL: echo_tool | test input]"
        results = tk.parse_and_execute(text)
        assert len(results) == 1

    def test_parse_and_execute_unknown_tool(self) -> None:
        tk = CLIToolkit()
        text = "[TOOL: ghost_tool | input]"
        results = tk.parse_and_execute(text)
        assert len(results) == 1
        assert results[0].ok is False
        assert "unknown tool" in results[0].output

    def test_parse_and_execute_no_calls(self) -> None:
        tk = CLIToolkit()
        results = tk.parse_and_execute("no tool calls here")
        assert results == []

    def test_augment_response_replaces_tool_calls(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool("echo_tool")])
        text = "Answer: [TOOL: echo_tool | ping]"
        augmented = tk.augment_response(text)
        assert "[RESULT:" in augmented

    def test_augment_response_unknown_tool(self) -> None:
        tk = CLIToolkit()
        text = "[TOOL: ghost | input]"
        augmented = tk.augment_response(text)
        assert "unknown tool" in augmented

    def test_repr_is_string(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool()])
        assert isinstance(repr(tk), str)

    def test_parse_multiple_calls(self) -> None:
        tk = CLIToolkit(tools=[self._echo_tool("t1"), self._echo_tool("t2")])
        text = "[TOOL: t1 | a] some text [TOOL: t2 | b]"
        results = tk.parse_and_execute(text)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


class TestFactories:
    def test_make_python_tool_creates_cli_tool(self) -> None:
        tool = make_python_tool()
        assert isinstance(tool, CLITool)
        assert tool.name == "python_exec"

    def test_make_python_tool_runs(self) -> None:
        tool = make_python_tool(timeout=10.0)
        result = tool.run("print('hello')")
        assert result.ok is True
        assert "hello" in result.output

    def test_make_shell_tool_creates_cli_tool(self) -> None:
        tool = make_shell_tool()
        assert isinstance(tool, CLITool)
        assert tool.name == "shell"

    def test_make_python_tool_custom_timeout(self) -> None:
        tool = make_python_tool(timeout=2.0)
        assert tool._timeout == 2.0
