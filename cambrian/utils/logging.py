"""Structured logging helpers for Cambrian.

Uses Python's stdlib ``logging`` with a rich handler for pretty terminal output
and falls back to plain text when ``rich`` is not available or stdout is not
a TTY.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

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
