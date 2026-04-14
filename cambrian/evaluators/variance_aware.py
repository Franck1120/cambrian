"""VarianceAwareEvaluator — anti-reward-hacking via diversified evaluation.

Reward hacking occurs when an agent learns to game a single metric while
failing on others.  A classic defence is to evaluate with *multiple diverse
evaluators* and penalise agents whose scores have high variance across them —
because high variance is the fingerprint of gaming: excelling on one metric
while tanking on others.

This module provides two complementary tools:

``VarianceAwareEvaluator``
    Runs N sub-evaluators, computes their mean score, then subtracts a
    variance penalty scaled by *penalty_weight*.  If any sub-evaluator is
    being gamed (score ≫ others), the penalty keeps the composite score
    honest.

``build_diversified_evaluator``
    Convenience factory that wires a ``CodeEvaluator``, an
    ``LLMJudgeEvaluator``, and a ``CompositeEvaluator`` together under a
    ``VarianceAwareEvaluator`` — the recommended three-pillar setup.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

from cambrian.agent import Agent
from cambrian.evaluator import Evaluator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


class VarianceAwareEvaluator(Evaluator):
    """Evaluate agents with multiple sub-evaluators and penalise reward hacking.

    The composite score is::

        score = mean(sub_scores) - penalty_weight * variance(sub_scores)

    A low-variance agent that performs consistently well receives a score
    close to ``mean(sub_scores)``.  An agent that games one evaluator but
    fails others will have high variance and be penalised accordingly.

    Args:
        evaluators: List of ``(agent, task) -> float`` callables (or
            :class:`~cambrian.evaluator.Evaluator` subclasses).
        weights: Per-evaluator weights (same length as *evaluators*).
            Normalised internally so they sum to 1. ``None`` means equal
            weights.
        penalty_weight: Multiplier on the score variance before subtracting.
            ``0.0`` disables the penalty (pure weighted mean).  ``1.0``
            applies a full-variance penalty.  Default ``0.5``.
        aggregate: ``"mean"`` (default) or ``"min"``.  When ``"min"``, the
            score is the *minimum* sub-score minus the variance penalty —
            even harsher on reward hackers.

    Example::

        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator
        from cambrian.evaluators.code import CodeEvaluator
        from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

        ev = VarianceAwareEvaluator(
            evaluators=[code_ev, judge_ev, style_ev],
            weights=[0.5, 0.3, 0.2],
            penalty_weight=0.5,
        )
    """

    def __init__(
        self,
        evaluators: list[Callable[[Agent, str], float]],
        weights: list[float] | None = None,
        penalty_weight: float = 0.5,
        aggregate: str = "mean",
    ) -> None:
        if len(evaluators) < 2:
            raise ValueError("VarianceAwareEvaluator requires at least 2 sub-evaluators.")
        if weights is not None and len(weights) != len(evaluators):
            raise ValueError("weights and evaluators must have the same length.")
        if aggregate not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'.")

        self._evaluators = evaluators
        self._weights = self._normalise(weights or [1.0] * len(evaluators))
        self._penalty = penalty_weight
        self._aggregate = aggregate

    # ── Evaluator API ─────────────────────────────────────────────────────────

    def evaluate(self, agent: Agent, task: str) -> float:
        """Run all sub-evaluators and return a variance-penalised composite score.

        Args:
            agent: Agent to score.
            task: Task description.

        Returns:
            Float in ``[0.0, 1.0]`` (clipped after penalty application).
        """
        sub_scores: list[float] = []
        for i, ev in enumerate(self._evaluators):
            try:
                s = float(ev(agent, task))
            except Exception as exc:
                logger.warning(
                    "Sub-evaluator %d raised %s: %s — using 0.0",
                    i, type(exc).__name__, exc,
                )
                s = 0.0
            sub_scores.append(s)
            logger.debug("Sub-evaluator %d score=%.4f", i, s)

        weighted = sum(s * w for s, w in zip(sub_scores, self._weights))

        if self._aggregate == "min":
            base = min(sub_scores)
        else:
            base = weighted

        var = statistics.variance(sub_scores) if len(sub_scores) > 1 else 0.0
        penalised = base - self._penalty * var

        result = max(0.0, min(1.0, penalised))
        logger.debug(
            "VarianceAware: sub=%s weighted=%.4f var=%.4f penalty=%.4f result=%.4f",
            [f"{s:.3f}" for s in sub_scores], weighted, var,
            self._penalty * var, result,
        )
        return result

    @property
    def sub_scores_last(self) -> list[float]:
        """Access pattern: unused in hot path; kept for introspection."""
        return []

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(weights: list[float]) -> list[float]:
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number.")
        return [w / total for w in weights]

    def __repr__(self) -> str:
        return (
            f"VarianceAwareEvaluator(n={len(self._evaluators)}, "
            f"penalty={self._penalty}, aggregate={self._aggregate!r})"
        )


def build_diversified_evaluator(
    backend: Any,
    expected_output: str,
    task_description: str,
    weights: list[float] | None = None,
    penalty_weight: float = 0.5,
) -> "VarianceAwareEvaluator":
    """Build a three-pillar diversified evaluator for anti-reward-hacking.

    Combines three perspectives:
    - **Correctness** (``CodeEvaluator``): did the agent produce the right output?
    - **Quality** (``LLMJudgeEvaluator``): is the response high quality?
    - **Conciseness** (length-based heuristic): is the response not bloated?

    Args:
        backend: LLM backend for the judge evaluator.
        expected_output: Expected code output for the ``CodeEvaluator``.
        task_description: Task description for the judge.
        weights: Weights for [correctness, quality, conciseness]. Default [0.5, 0.3, 0.2].
        penalty_weight: Variance penalty multiplier. Default ``0.5``.

    Returns:
        A configured :class:`VarianceAwareEvaluator`.
    """
    from cambrian.evaluators.code import CodeEvaluator
    from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

    code_ev = CodeEvaluator(expected_output=expected_output)
    judge_ev = LLMJudgeEvaluator(judge_backend=backend)

    def _conciseness_ev(agent: Agent, task: str) -> float:
        """Score based on prompt conciseness (shorter = better, up to a point)."""
        tokens = agent.genome.token_count()
        if tokens <= 0:
            return 0.0
        # Sweet spot: 50–150 tokens → score 1.0; penalty above 300
        if tokens <= 50:
            return 0.7
        if tokens <= 150:
            return 1.0
        if tokens <= 300:
            return max(0.4, 1.0 - (tokens - 150) / 300)
        return max(0.1, 1.0 - tokens / 600)

    return VarianceAwareEvaluator(
        evaluators=[code_ev, judge_ev, _conciseness_ev],
        weights=weights or [0.5, 0.3, 0.2],
        penalty_weight=penalty_weight,
    )
