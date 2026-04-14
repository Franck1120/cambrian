"""Subprocess sandbox for executing untrusted Python code safely.

Runs code in an isolated subprocess with a hard timeout so that runaway
loops or infinite recursion cannot hang the evaluation loop.

Security model
--------------
Only a minimal whitelist of environment variables is forwarded to the
subprocess.  This prevents agent-generated code from reading API keys,
credentials, or other secrets present in the parent process environment.
The set ``_SANDBOX_SAFE_KEYS`` defines exactly which variables are passed.
``extra_env`` can add further variables explicitly if needed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tempfile
import os
from dataclasses import dataclass

# Environment variables forwarded to the sandbox subprocess.
# Everything NOT in this set (including all API keys) is stripped.
_SANDBOX_SAFE_KEYS: frozenset[str] = frozenset({
    "PATH",
    "PYTHONPATH",
    "SYSTEMROOT",   # Windows: required for subprocess to start
    "TEMP",
    "TMP",
    "TMPDIR",       # Unix fallback
    "HOME",         # some libraries need this
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
})


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool

    @property
    def success(self) -> bool:
        """True when the process exited with returncode 0 and did not time out."""
        return self.returncode == 0 and not self.timed_out


def run_in_sandbox(
    code: str,
    timeout: float = 10.0,
    stdin: str = "",
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute *code* in an isolated subprocess.

    Args:
        code: Python source code to execute.
        timeout: Maximum wall-clock seconds before the process is killed.
        stdin: Optional text to pass on stdin.
        extra_env: Additional environment variables for the subprocess.

    Returns:
        :class:`SandboxResult` with stdout, stderr, returncode, timed_out.
    """
    # Write code to a temp file to avoid shell-injection via command args
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(textwrap.dedent(code))
        tmp_path = tmp.name

    # Strip all secrets from the environment — only forward safe vars.
    # This prevents agent code from reading OPENAI_API_KEY etc.
    env: dict[str, str] = {
        k: v for k, v in os.environ.items() if k in _SANDBOX_SAFE_KEYS
    }
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return SandboxResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            stdout="",
            stderr=f"TimeoutExpired: execution exceeded {timeout}s",
            returncode=-1,
            timed_out=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_python_code(text: str) -> str:
    """Extract the first fenced Python code block from *text*.

    If no fenced block is found, returns *text* stripped of leading/trailing
    whitespace (assumes the whole string is code).
    """
    lines = text.splitlines()
    in_block = False
    collected: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not in_block and (
            stripped.startswith("```python") or stripped.startswith("```py")
        ):
            in_block = True
            continue
        if in_block:
            if stripped == "```":
                break
            collected.append(line)

    if collected:
        return "\n".join(collected)

    # Fallback: no fenced block found
    return text.strip()
