# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""tests/test_sandbox_security.py — Security tests for the subprocess sandbox.

Verifies that:
1. API keys and sensitive env vars are NOT forwarded to sandbox subprocesses.
2. Only whitelisted env vars pass through.
3. extra_env can explicitly inject additional vars.
4. The sandbox still executes code correctly after the env restriction.
"""

from __future__ import annotations

import os

import pytest

from cambrian.utils.sandbox import (
    SandboxResult,
    _SANDBOX_SAFE_KEYS,
    extract_python_code,
    run_in_sandbox,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _code_print_env(var: str) -> str:
    """Python snippet that prints the value of env var or 'NOT_SET'."""
    return f"import os; print(os.environ.get('{var}', 'NOT_SET'))"


# ─────────────────────────────────────────────────────────────────────────────
# API key isolation
# ─────────────────────────────────────────────────────────────────────────────


class TestApiKeyIsolation:
    """Critical: API keys must never reach sandbox subprocesses."""

    @pytest.fixture(autouse=True)
    def _inject_fake_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject fake API keys into the parent process environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-OPENAI-KEY-12345")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-ANTHROPIC-KEY")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKE-GEMINI-KEY")
        monkeypatch.setenv("CAMBRIAN_API_KEY", "cambrian-FAKE-KEY-99")

    def test_openai_key_not_in_sandbox(self) -> None:
        result = run_in_sandbox(_code_print_env("OPENAI_API_KEY"))
        assert result.success
        assert "FAKE-OPENAI" not in result.stdout
        assert result.stdout.strip() == "NOT_SET"

    def test_anthropic_key_not_in_sandbox(self) -> None:
        result = run_in_sandbox(_code_print_env("ANTHROPIC_API_KEY"))
        assert result.success
        assert "FAKE-ANTHROPIC" not in result.stdout
        assert result.stdout.strip() == "NOT_SET"

    def test_gemini_key_not_in_sandbox(self) -> None:
        result = run_in_sandbox(_code_print_env("GEMINI_API_KEY"))
        assert result.success
        assert "FAKE-GEMINI" not in result.stdout
        assert result.stdout.strip() == "NOT_SET"

    def test_cambrian_key_not_in_sandbox(self) -> None:
        result = run_in_sandbox(_code_print_env("CAMBRIAN_API_KEY"))
        assert result.success
        assert "FAKE-KEY" not in result.stdout
        assert result.stdout.strip() == "NOT_SET"

    def test_arbitrary_secret_not_in_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_SECRET_TOKEN", "super-secret-value-xyz")
        result = run_in_sandbox(_code_print_env("MY_SECRET_TOKEN"))
        assert result.success
        assert "super-secret-value-xyz" not in result.stdout
        assert result.stdout.strip() == "NOT_SET"


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist enforcement
# ─────────────────────────────────────────────────────────────────────────────


class TestWhitelistEnforcement:
    """Only _SANDBOX_SAFE_KEYS should be forwarded."""

    def test_safe_keys_constant_is_frozenset(self) -> None:
        assert isinstance(_SANDBOX_SAFE_KEYS, frozenset)

    def test_safe_keys_contains_path(self) -> None:
        assert "PATH" in _SANDBOX_SAFE_KEYS

    def test_safe_keys_contains_pythonpath(self) -> None:
        assert "PYTHONPATH" in _SANDBOX_SAFE_KEYS

    def test_safe_keys_does_not_contain_api_keys(self) -> None:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                    "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "CAMBRIAN_API_KEY"):
            assert key not in _SANDBOX_SAFE_KEYS, f"{key} must not be in safe keys"

    def test_dump_all_env_cannot_leak_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sandboxed code that dumps ALL env vars must not expose API keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-LEAK-ME")
        code = "import os; print('\\n'.join(os.environ.keys()))"
        result = run_in_sandbox(code)
        assert result.success
        assert "OPENAI_API_KEY" not in result.stdout

    def test_env_key_count_is_small(self) -> None:
        """Sandbox should forward only a handful of vars, not the whole env."""
        code = "import os; print(len(os.environ))"
        result = run_in_sandbox(code)
        assert result.success
        forwarded_count = int(result.stdout.strip())
        # Parent process typically has 50+ vars; sandbox should have far fewer
        parent_count = len(os.environ)
        assert forwarded_count < parent_count


# ─────────────────────────────────────────────────────────────────────────────
# extra_env injection
# ─────────────────────────────────────────────────────────────────────────────


class TestExtraEnv:
    """extra_env must be explicitly injectable while keeping secrets stripped."""

    def test_extra_env_is_forwarded(self) -> None:
        result = run_in_sandbox(
            _code_print_env("MY_EXTRA_VAR"),
            extra_env={"MY_EXTRA_VAR": "hello-from-extra"},
        )
        assert result.success
        assert result.stdout.strip() == "hello-from-extra"

    def test_extra_env_cannot_override_strip_of_api_keys_via_parent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extra_env adds NEW vars; parent's API keys are still stripped."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-SHOULD-NOT-LEAK")
        result = run_in_sandbox(
            _code_print_env("OPENAI_API_KEY"),
            extra_env={"SAFE_EXTRA": "ok"},
        )
        assert result.success
        assert "SHOULD-NOT-LEAK" not in result.stdout

    def test_extra_env_injected_api_key_is_intentional(self) -> None:
        """Caller CAN explicitly inject a key via extra_env — that's intentional."""
        result = run_in_sandbox(
            _code_print_env("EXPLICIT_KEY"),
            extra_env={"EXPLICIT_KEY": "deliberate-value"},
        )
        assert result.success
        assert result.stdout.strip() == "deliberate-value"


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox still works correctly after env restriction
# ─────────────────────────────────────────────────────────────────────────────


class TestSandboxFunctionality:
    """Verify the sandbox executes code correctly despite env stripping."""

    def test_hello_world(self) -> None:
        result = run_in_sandbox('print("hello, world")')
        assert result.success
        assert "hello, world" in result.stdout

    def test_arithmetic(self) -> None:
        result = run_in_sandbox("print(2 + 2)")
        assert result.success
        assert "4" in result.stdout

    def test_timeout_returns_timed_out(self) -> None:
        result = run_in_sandbox("while True: pass", timeout=0.5)
        assert result.timed_out
        assert not result.success

    def test_syntax_error_returns_nonzero(self) -> None:
        result = run_in_sandbox("def f(: pass")
        assert not result.success
        assert result.returncode != 0

    def test_sandbox_result_success_property(self) -> None:
        ok = SandboxResult(stdout="ok", stderr="", returncode=0, timed_out=False)
        assert ok.success
        fail = SandboxResult(stdout="", stderr="err", returncode=1, timed_out=False)
        assert not fail.success
        timeout = SandboxResult(stdout="", stderr="", returncode=0, timed_out=True)
        assert not timeout.success


# ─────────────────────────────────────────────────────────────────────────────
# extract_python_code (unchanged but worth regression testing)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractPythonCode:
    def test_fenced_python_block(self) -> None:
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert extract_python_code(text) == "print('hello')"

    def test_fenced_py_block(self) -> None:
        text = "```py\nx = 1\n```"
        assert extract_python_code(text) == "x = 1"

    def test_no_fence_returns_stripped(self) -> None:
        text = "  print(42)  "
        assert extract_python_code(text) == "print(42)"

    def test_empty_string(self) -> None:
        assert extract_python_code("") == ""
