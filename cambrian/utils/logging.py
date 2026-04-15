# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Structured logging helpers for Cambrian.

Uses Python's stdlib ``logging`` with a rich handler for pretty terminal output
and falls back to plain text when ``rich`` is not available or stdout is not
a TTY.

Also provides :class:`JSONLogger` for machine-readable structured JSON logs —
each generation produces a single JSON line suitable for streaming analytics
pipelines or post-run comparison tools (``cambrian compare``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

_CAMBRIAN_LOG_LEVEL = os.getenv("CAMBRIAN_LOG_LEVEL", "INFO").upper()

try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


def get_logger(name: str = "cambrian") -> logging.Logger:
    """Return a named logger configured for Cambrian.

    The first call creates and configures the root ``cambrian`` logger;
    subsequent calls with child names reuse the existing root handler.

    Args:
        name: Logger name, e.g. ``"cambrian.evolution"``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Only configure the root cambrian logger once
    root = logging.getLogger("cambrian")
    if root.handlers:
        return logger

    level = getattr(logging, _CAMBRIAN_LOG_LEVEL, logging.INFO)
    root.setLevel(level)

    if _RICH_AVAILABLE and sys.stdout.isatty():
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root.addHandler(handler)
    return logger


def log_generation_summary(
    logger: logging.Logger,
    generation: int,
    best_fitness: float,
    avg_fitness: float,
    diversity: float,
    **extra: Any,
) -> None:
    """Log a one-line summary for a completed evolution generation.

    Args:
        logger: Logger instance to write to.
        generation: Current generation number (0-indexed).
        best_fitness: Highest fitness score in this generation.
        avg_fitness: Mean fitness across the population.
        diversity: Behavioural diversity metric (e.g. MAP-Elites coverage).
        **extra: Additional key=value pairs appended to the log line.
    """
    parts = [
        f"gen={generation}",
        f"best={best_fitness:.4f}",
        f"avg={avg_fitness:.4f}",
        f"diversity={diversity:.4f}",
    ]
    parts += [f"{k}={v}" for k, v in extra.items()]
    logger.info("  ".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# JSONLogger — structured generation logs
# ─────────────────────────────────────────────────────────────────────────────


class JSONLogger:
    """Write structured per-generation JSON log entries to a file or stream.

    Each :meth:`log_generation` call appends one JSON object (newline-delimited
    NDJSON) to the output.  The file can later be consumed by
    ``cambrian compare`` or any streaming analytics tool.

    Log entry schema::

        {
            "ts": 1714000000.123,        # Unix timestamp
            "run_id": "my_run",          # User-defined run identifier
            "generation": 3,
            "best_fitness": 0.8750,
            "mean_fitness": 0.7200,
            "min_fitness": 0.5100,
            "std_fitness": 0.1123,
            "population_size": 8,
            "best_agent_id": "abc12345",
            "best_prompt_len": 320,
            "extra": {"diversity": 0.42}  # any additional kwargs
        }

    Args:
        output: File path (``str`` or :class:`pathlib.Path`) or an open text
            stream.  When a path is given the file is opened in append mode so
            multiple runs can be stored in the same file.
        run_id: Human-readable label for this run.  Included in every log
            entry for filtering. Default ``"default"``.
        flush: Whether to flush the output after each entry. Default ``True``.
    """

    def __init__(
        self,
        output: str | Path | TextIO,
        run_id: str = "default",
        flush: bool = True,
    ) -> None:
        self._run_id = run_id
        self._flush = flush
        self._owned = False

        if isinstance(output, (str, Path)):
            self._file: TextIO = open(Path(output), "a", encoding="utf-8")  # noqa: SIM115
            self._owned = True
        else:
            self._file = output

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_generation(
        self,
        generation: int,
        fitnesses: list[float],
        best_agent_id: str = "",
        best_prompt_len: int = 0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append a JSON log entry for *generation*.

        Args:
            generation: Zero-based generation index.
            fitnesses: List of fitness scores for all agents in the population.
            best_agent_id: ID of the best agent (for traceability).
            best_prompt_len: Character length of the best agent's system prompt.
            **extra: Any additional key-value pairs to include under ``"extra"``.

        Returns:
            The log entry dict (also written to the output).
        """
        valid = [f for f in fitnesses if f is not None]
        n = len(valid)

        best_f = max(valid) if valid else 0.0
        mean_f = sum(valid) / n if n else 0.0
        min_f = min(valid) if valid else 0.0
        std_f = (
            (sum((f - mean_f) ** 2 for f in valid) / n) ** 0.5
            if n > 1
            else 0.0
        )

        entry: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self._run_id,
            "generation": generation,
            "best_fitness": round(best_f, 6),
            "mean_fitness": round(mean_f, 6),
            "min_fitness": round(min_f, 6),
            "std_fitness": round(std_f, 6),
            "population_size": len(fitnesses),
            "best_agent_id": best_agent_id,
            "best_prompt_len": best_prompt_len,
            "extra": extra,
        }

        self._file.write(json.dumps(entry) + "\n")
        if self._flush:
            self._file.flush()

        return entry

    def log_run_summary(
        self,
        n_generations: int,
        best_fitness: float,
        best_agent_id: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        """Append a final run-summary entry.

        Args:
            n_generations: Total generations completed.
            best_fitness: Best fitness achieved across the entire run.
            best_agent_id: ID of the overall best agent.
            **extra: Additional metadata.

        Returns:
            The summary entry dict.
        """
        entry: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self._run_id,
            "event": "run_complete",
            "n_generations": n_generations,
            "best_fitness": round(best_fitness, 6),
            "best_agent_id": best_agent_id,
            "extra": extra,
        }
        self._file.write(json.dumps(entry) + "\n")
        if self._flush:
            self._file.flush()
        return entry

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the output (only if this logger owns the file)."""
        if self._owned:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "JSONLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"JSONLogger(run_id={self._run_id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for compare
# ─────────────────────────────────────────────────────────────────────────────


def load_json_log(path: str | Path) -> list[dict[str, Any]]:
    """Load all generation entries from an NDJSON log file.

    Args:
        path: Path to the log file written by :class:`JSONLogger`.

    Returns:
        List of log entry dicts, in file order.
    """
    entries: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines
    return entries
