# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""ModelRouter — selects the appropriate LLM model for a given task.

Routes tasks to cheap fast models when complexity is low and to premium
models when the task demands more capability. This reduces API costs
without sacrificing quality on simple subtasks.
"""

from __future__ import annotations

import re


# Token thresholds for routing tiers
_CHEAP_MAX_TOKENS = 100
_MEDIUM_MAX_TOKENS = 500

# Default model tiers — override via constructor
_DEFAULT_CHEAP_MODEL = "gpt-4o-mini"
_DEFAULT_MEDIUM_MODEL = "gpt-4o-mini"
_DEFAULT_PREMIUM_MODEL = "gpt-4o"


def _estimate_tokens(text: str) -> int:
    """Rough token count: split on whitespace + punctuation."""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


class ModelRouter:
    """Routes tasks to the cheapest model that can handle their complexity.

    Complexity is estimated from the task's token count and the presence of
    difficulty signals (code fences, multi-step instructions, etc.).

    Args:
        cheap_model: Model to use for simple tasks (< 100 tokens, no complexity
            signals). Default ``"gpt-4o-mini"``.
        medium_model: Model for medium tasks (100–500 tokens). Default same
            as *cheap_model*.
        premium_model: Model for complex tasks (> 500 tokens or complexity
            signals detected). Default ``"gpt-4o"``.
    """

    _COMPLEXITY_SIGNALS = re.compile(
        r"(```|step[- ]by[- ]step|multi[- ]?step|implement|architect|design|"
        r"algorithm|complex|difficult|advanced|explain.*detail)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        cheap_model: str = _DEFAULT_CHEAP_MODEL,
        medium_model: str = _DEFAULT_MEDIUM_MODEL,
        premium_model: str = _DEFAULT_PREMIUM_MODEL,
    ) -> None:
        self._cheap = cheap_model
        self._medium = medium_model
        self._premium = premium_model
        self._routing_log: list[tuple[str, str]] = []  # (task_snippet, model)

    def route(self, task: str) -> str:
        """Return the model name best suited for *task*.

        Args:
            task: The task description / prompt text.

        Returns:
            Model identifier string (e.g. ``"gpt-4o-mini"``).
        """
        token_count = _estimate_tokens(task)
        has_complexity = bool(self._COMPLEXITY_SIGNALS.search(task))

        if has_complexity or token_count > _MEDIUM_MAX_TOKENS:
            model = self._premium
        elif token_count > _CHEAP_MAX_TOKENS:
            model = self._medium
        else:
            model = self._cheap

        self._routing_log.append((task[:60], model))
        return model

    @property
    def routing_log(self) -> list[tuple[str, str]]:
        """History of (task_snippet, routed_model) pairs."""
        return list(self._routing_log)

    def routing_stats(self) -> dict[str, int]:
        """Count how many tasks were routed to each model tier."""
        stats: dict[str, int] = {}
        for _, model in self._routing_log:
            stats[model] = stats.get(model, 0) + 1
        return stats

    def __repr__(self) -> str:
        return (
            f"ModelRouter(cheap={self._cheap!r}, "
            f"medium={self._medium!r}, premium={self._premium!r})"
        )
