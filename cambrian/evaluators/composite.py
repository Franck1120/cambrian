"""CompositeEvaluator — combines multiple evaluators to resist reward hacking.

When a single evaluator is used as the fitness signal, agents can overfit to
its specific scoring quirks (reward hacking). CompositeEvaluator mitigates
this by averaging scores across heterogeneous evaluators — an agent must
genuinely improve across all dimensions to raise its composite fitness.

Example::

    evaluator = CompositeEvaluator([
        CodeEvaluator(expected_output="FizzBuzz\\n..."),
        LLMJudgeEvaluator(judge_backend=backend, rubric_extension="Prefer clean code."),
    ], weights=[0.7, 0.3])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator

if TYPE_CHECKING:
    from cambrian.agent import Agent


class CompositeEvaluator(Evaluator):
    """Weighted average of multiple :class:`~cambrian.evaluator.Evaluator` instances.

    Args:
        evaluators: List of evaluators to combine.
        weights: Optional list of floats that must match the length of
            *evaluators*. Weights are normalised internally so they need not
            sum to 1. If ``None``, all evaluators are weighted equally.
        aggregate: Aggregation function name — ``"mean"`` (default) or
            ``"min"``.  ``"min"`` forces the agent to satisfy all evaluators
            simultaneously (strictest anti-reward-hacking guarantee).
    """

    def __init__(
        self,
        evaluators: list[Evaluator],
        weights: list[float] | None = None,
        aggregate: str = "mean",
    ) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator.")
        if weights is not None and len(weights) != len(evaluators):
            raise ValueError(
                f"weights length ({len(weights)}) must match evaluators "
                f"length ({len(evaluators)})."
            )
        self._evaluators = evaluators
        if weights is None:
            total = len(evaluators)
            self._weights = [1.0 / total] * total
        else:
            total = sum(weights)
            self._weights = [w / total for w in weights]
        if aggregate not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'")
        self._aggregate = aggregate

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Run all sub-evaluators and return the composite fitness score."""
        scores: list[float] = []
        for evaluator in self._evaluators:
            try:
                score = evaluator.evaluate(agent, task)
            except Exception:
                score = 0.0
            scores.append(min(max(score, 0.0), 1.0))

        if self._aggregate == "min":
            return min(scores)

        # Weighted mean
        return sum(s * w for s, w in zip(scores, self._weights))

    @property
    def evaluators(self) -> list[Evaluator]:
        """The constituent evaluators."""
        return list(self._evaluators)

    def __repr__(self) -> str:
        names = [type(e).__name__ for e in self._evaluators]
        return f"CompositeEvaluator({names}, aggregate={self._aggregate!r})"
