"""LLM Cascade — Technique 54.

A two-level (or N-level) cascade routes inference calls to progressively
more powerful (and expensive) models based on confidence signals from
cheaper models.

Design
------
Each level is a ``CascadeLevel`` pairing an ``LLMBackend`` with a
*confidence extractor*.  The extractor reads the raw completion and returns
a float in [0, 1].  If the score is above ``confidence_threshold`` the
cascade returns the answer immediately; otherwise it passes the task to the
next (stronger) model.

Built-in confidence extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``hedging_confidence(response)``
    Lowers confidence when the response contains hedging phrases
    ("I'm not sure", "may", "might", "possibly", "unclear", etc.)

``length_confidence(response, min_len=50)``
    Very short responses are typically incomplete → low confidence.

Usage::

    from cambrian.llm_cascade import LLMCascade, CascadeLevel, hedging_confidence

    cascade = LLMCascade(
        levels=[
            CascadeLevel(fast_backend, confidence_threshold=0.8),
            CascadeLevel(powerful_backend, confidence_threshold=0.0),
        ]
    )
    answer, level_used = cascade.query(system_prompt, user_message)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend


# ---------------------------------------------------------------------------
# Built-in confidence extractors
# ---------------------------------------------------------------------------

_HEDGING = re.compile(
    r"\b(i(?:'m| am) not sure|i(?:'m| am) uncertain|may|might|possibly|perhaps"
    r"|unclear|not certain|it depends|hard to say|difficult to determine)\b",
    re.IGNORECASE,
)


def hedging_confidence(response: str) -> float:
    """Return confidence ∈ [0, 1] reduced by hedging phrases."""
    hedges = len(_HEDGING.findall(response))
    return max(0.0, 1.0 - hedges * 0.25)


def length_confidence(response: str, min_len: int = 50) -> float:
    """Return confidence based on response length."""
    n = len(response.strip())
    if n == 0:
        return 0.0
    return min(1.0, n / min_len)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ConfidenceExtractor = Callable[[str], float]


@dataclass
class CascadeLevel:
    """One level in the cascade."""

    backend: "LLMBackend"
    confidence_threshold: float = 0.7
    confidence_extractor: ConfidenceExtractor = field(
        default_factory=lambda: hedging_confidence
    )
    temperature: float = 0.7


@dataclass
class CascadeResult:
    """Result from a cascade query."""

    response: str
    level_index: int        # 0-based index of the level that produced the answer
    confidence: float
    attempts: int           # total number of levels tried


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class LLMCascade:
    """Route LLM calls through progressively more powerful models.

    Parameters
    ----------
    levels:
        Ordered list of ``CascadeLevel`` objects, from cheapest to most
        powerful.  The last level always returns its answer regardless of
        confidence (it is the final fallback).
    """

    def __init__(self, levels: list[CascadeLevel]) -> None:
        if not levels:
            raise ValueError("LLMCascade requires at least one level.")
        self._levels = levels
        self._stats: list[CascadeResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> list[CascadeResult]:
        """Return a copy of all query results."""
        return list(self._stats)

    def query(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> tuple[str, int]:
        """Route *user* through the cascade.

        Parameters
        ----------
        system:
            System prompt forwarded to every level.
        user:
            User message / task.
        temperature:
            If set, overrides each level's default temperature.

        Returns
        -------
        (response, level_index)
            The answer and the 0-based index of the level that produced it.
        """
        for idx, level in enumerate(self._levels):
            temp = temperature if temperature is not None else level.temperature
            try:
                response = str(
                    level.backend.generate(
                        f"{system}\n\n{user}",
                        temperature=temp,
                    )
                )
            except Exception:  # noqa: BLE001
                response = ""

            confidence = level.confidence_extractor(response)
            is_last = idx == len(self._levels) - 1

            if confidence >= level.confidence_threshold or is_last:
                result = CascadeResult(
                    response=response,
                    level_index=idx,
                    confidence=confidence,
                    attempts=idx + 1,
                )
                self._stats.append(result)
                return response, idx

        # Unreachable — loop always exits on last level
        raise RuntimeError("Cascade loop exited without returning.")  # pragma: no cover

    def level_usage_counts(self) -> dict[int, int]:
        """Return a dict mapping level index → number of times it answered."""
        counts: dict[int, int] = {}
        for r in self._stats:
            counts[r.level_index] = counts.get(r.level_index, 0) + 1
        return counts
