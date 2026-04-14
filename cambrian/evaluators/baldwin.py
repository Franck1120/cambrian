"""BaldwinEvaluator — in-context learning bonus (Baldwin Effect).

The Baldwin Effect (1896) describes how an organism's capacity for
*within-lifetime learning* can accelerate evolutionary search without
Lamarckian inheritance.  Organisms that learn faster find fit regions of the
phenotype space more quickly, and selection then favours the genetic
predisposition to learn — even though the learned behaviour itself is not
inherited.

In Cambrian's setting, *learning* means: the agent improves its responses
when given feedback within a single evaluation episode.

:class:`BaldwinEvaluator` implements this as a **multi-trial wrapper**:

1. **Trial 1** — evaluate the agent on the raw task.  Record the base score.
2. **Trials 2..N** — prepend a *feedback hint* derived from the previous trial
   to the task and re-evaluate.  Record each score.
3. **Improvement score** — the difference between the last trial's score and
   the first trial's score (clamped to ``[0, 1]``).
4. **Final fitness** — ``base_score + baldwin_bonus × improvement_score``
   (clamped to ``[0, 1]``).

Agents whose prompts make them inherently *responsive* to feedback will
improve more across trials and thus receive a higher composite fitness,
creating selection pressure for learnability.

Usage::

    from cambrian.evaluators.baldwin import BaldwinEvaluator

    inner = CodeEvaluator(expected_output="hello")
    baldwin_ev = BaldwinEvaluator(
        base_evaluator=inner,
        n_trials=3,
        baldwin_bonus=0.15,
        feedback_template="Your previous attempt scored {score:.2f}/1.0. "
                          "Improve your response: {task}",
    )

    engine = EvolutionEngine(evaluator=baldwin_ev, ...)
"""

from __future__ import annotations

from typing import Callable

from cambrian.agent import Agent
from cambrian.evaluator import Evaluator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


class BaldwinEvaluator(Evaluator):
    """Multi-trial evaluator that rewards agents able to improve from feedback.

    Args:
        base_evaluator: Any ``(Agent, str) → float`` callable.
        n_trials: Number of evaluation trials per agent.  Must be ≥ 2.
            Default ``3``.
        baldwin_bonus: Maximum fitness bonus awarded for perfect improvement
            (score going from 0 → 1 across trials).  Default ``0.1``.
        feedback_template: Template string for the feedback-augmented task.
            Available variables: ``{task}`` (original task), ``{score}``
            (previous trial score), ``{trial}`` (current trial index, 1-based).
            Default provides a generic "your previous attempt scored X" hint.
        aggregate_base: If ``"last"``, use only the last trial's score as the
            base.  If ``"best"``, use the best score across all trials.
            If ``"first"`` (default), use the first trial's score as the
            base to ensure the bonus is purely additive.

    Raises:
        ValueError: If *n_trials* < 2.
    """

    DEFAULT_FEEDBACK_TEMPLATE = (
        "Your previous attempt scored {score:.2f}/1.0. "
        "Use this feedback to improve your response.\n\nTask: {task}"
    )

    def __init__(
        self,
        base_evaluator: Callable[[Agent, str], float],
        n_trials: int = 3,
        baldwin_bonus: float = 0.1,
        feedback_template: str | None = None,
        aggregate_base: str = "first",
    ) -> None:
        if n_trials < 2:
            raise ValueError("BaldwinEvaluator requires at least n_trials=2.")
        if not (0.0 <= baldwin_bonus <= 1.0):
            raise ValueError("baldwin_bonus must be in [0, 1].")
        if aggregate_base not in ("first", "best", "last"):
            raise ValueError("aggregate_base must be 'first', 'best', or 'last'.")

        self._base = base_evaluator
        self._n_trials = n_trials
        self._bonus = baldwin_bonus
        self._template = feedback_template or self.DEFAULT_FEEDBACK_TEMPLATE
        self._aggregate_base = aggregate_base

    def evaluate(self, agent: Agent, task: str) -> float:
        """Run *n_trials* evaluations with escalating feedback, return Baldwin fitness.

        Args:
            agent: Agent to evaluate.
            task: Original task string.

        Returns:
            Composite fitness: base_score + baldwin_bonus × improvement_score,
            clamped to ``[0.0, 1.0]``.
        """
        scores: list[float] = []
        current_task = task

        for trial in range(self._n_trials):
            try:
                score = float(self._base(agent, current_task))
            except Exception as exc:
                logger.warning(
                    "BaldwinEvaluator trial %d raised %s: %s — using 0.0",
                    trial + 1, type(exc).__name__, exc,
                )
                score = 0.0

            scores.append(score)
            logger.debug(
                "Baldwin trial %d/%d score=%.4f agent=%s",
                trial + 1, self._n_trials, score, agent.id[:8],
            )

            # Build feedback task for next trial (not needed after last trial)
            if trial < self._n_trials - 1:
                current_task = self._template.format(
                    task=task,
                    score=score,
                    trial=trial + 1,
                )

        if not scores:
            return 0.0

        # Base score according to aggregation strategy
        if self._aggregate_base == "first":
            base_score = scores[0]
        elif self._aggregate_base == "best":
            base_score = max(scores)
        else:  # "last"
            base_score = scores[-1]

        # Improvement: how much did the agent improve from trial 1 to trial N?
        improvement = max(0.0, scores[-1] - scores[0])

        fitness = base_score + self._bonus * improvement
        return min(1.0, max(0.0, fitness))

    def improvement_stats(
        self, agent: Agent, task: str
    ) -> dict[str, float | list[float]]:
        """Run the full trial sequence and return detailed statistics.

        Useful for analysis and debugging, not called during normal evolution.

        Args:
            agent: Agent to probe.
            task: Task string.

        Returns:
            Dict with keys ``"scores"``, ``"base"``, ``"improvement"``,
            ``"final_fitness"``, ``"learnable"`` (bool-as-float).
        """
        scores: list[float] = []
        current_task = task

        for trial in range(self._n_trials):
            try:
                score = float(self._base(agent, current_task))
            except Exception:
                score = 0.0
            scores.append(score)
            if trial < self._n_trials - 1:
                current_task = self._template.format(
                    task=task, score=score, trial=trial + 1
                )

        base = scores[0] if scores else 0.0
        improvement = max(0.0, scores[-1] - scores[0]) if len(scores) > 1 else 0.0
        final = min(1.0, max(0.0, base + self._bonus * improvement))

        return {
            "scores": scores,
            "base": base,
            "improvement": improvement,
            "final_fitness": final,
            "learnable": float(improvement > 0),
        }

    def __repr__(self) -> str:
        return (
            f"BaldwinEvaluator(n_trials={self._n_trials}, "
            f"bonus={self._bonus}, base={self._aggregate_base!r})"
        )
