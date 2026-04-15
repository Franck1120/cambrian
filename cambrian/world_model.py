# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""World model — agents predict action outcomes before executing them.

Inspired by Dyna-Q / model-based reinforcement learning and "world models"
(Ha & Schmidhuber 2018).  Each agent carries a lightweight *world model*:
a prediction of how well a given action (prompt + strategy) will score on
a task.  Before evaluating an agent for real, the engine queries its world
model and only runs the expensive base evaluation when the prediction is
sufficiently uncertain or optimistic.

This produces two selective pressures:
1. Agents that *perform* well on the base evaluator.
2. Agents whose world model *predicts accurately* — i.e., agents that
   understand their own capabilities.

Architecture
------------

:class:`WorldModelPrediction`
    A single prediction: expected score + confidence.

:class:`WorldModel`
    Per-agent predictor.  Accumulates experience from past evaluations and
    predicts scores for new tasks via a simple weighted nearest-neighbour
    approach over the agent's experience buffer.

:class:`WorldModelEvaluator`
    Wraps a base :class:`~cambrian.evaluator.Evaluator`.  For each agent it:
    (a) queries the world model for a predicted score,
    (b) runs the real evaluator,
    (c) updates the world model with the observed outcome, and
    (d) computes a blended score that rewards both performance and prediction
        accuracy.

:func:`world_model_fitness`
    Helper: combine raw score and prediction accuracy into a single metric.

Usage::

    from cambrian.world_model import WorldModelEvaluator

    wm_eval = WorldModelEvaluator(
        base_evaluator=my_evaluator,
        accuracy_weight=0.2,    # 20% of fitness from prediction accuracy
        buffer_size=20,
    )
    score = wm_eval.evaluate(agent, task)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelPrediction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WorldModelPrediction:
    """A single world-model prediction.

    Attributes:
        predicted_score: Expected fitness score in ``[0.0, 1.0]``.
        confidence: Confidence of the prediction in ``[0.0, 1.0]``.
            Low confidence ≡ sparse experience; high confidence ≡ many similar
            past observations.
        n_similar: Number of past experiences used to form this prediction.
    """

    predicted_score: float
    confidence: float
    n_similar: int = 0

    @property
    def is_uncertain(self) -> bool:
        """``True`` when confidence is below 0.5."""
        return self.confidence < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# WorldModel
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Experience:
    """A single stored (task, score) observation."""

    task: str
    score: float
    weight: float = 1.0


class WorldModel:
    """Lightweight per-agent world model.

    Predicts the score an agent will receive on a task based on its
    experience buffer.  Similarity is computed as normalised character overlap
    between task strings.

    Args:
        buffer_size: Maximum number of past experiences to retain. Oldest
            entries are evicted first. Default ``20``.
        default_score: Score returned when the buffer is empty. Default ``0.5``.
        decay: Temporal decay factor applied to experience weights each
            time a new entry is added. ``1.0`` = no decay. Default ``0.95``.
    """

    def __init__(
        self,
        buffer_size: int = 20,
        default_score: float = 0.5,
        decay: float = 0.95,
    ) -> None:
        self._buffer: list[_Experience] = []
        self._buffer_size = buffer_size
        self._default_score = default_score
        self._decay = decay

    # ------------------------------------------------------------------
    # Experience management
    # ------------------------------------------------------------------

    def update(self, task: str, score: float) -> None:
        """Add a new (task, score) observation to the experience buffer.

        Args:
            task: Task description.
            score: Observed fitness score.
        """
        # Decay existing weights
        for exp in self._buffer:
            exp.weight *= self._decay

        self._buffer.append(_Experience(task=task, score=score))

        # Evict oldest if over capacity
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Character-level Jaccard similarity between *a* and *b*."""
        if not a and not b:
            return 1.0
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def predict(self, task: str) -> WorldModelPrediction:
        """Predict the fitness score for *task*.

        Uses weighted nearest-neighbour over the experience buffer, with
        similarity as the weight.  Confidence is the fraction of the buffer
        capacity used (more experience → higher confidence).

        Args:
            task: Task description to predict for.

        Returns:
            :class:`WorldModelPrediction` with predicted score and confidence.
        """
        if not self._buffer:
            return WorldModelPrediction(
                predicted_score=self._default_score,
                confidence=0.0,
                n_similar=0,
            )

        similarities = [
            (self._similarity(task, exp.task) * exp.weight, exp.score)
            for exp in self._buffer
        ]
        total_weight = sum(w for w, _ in similarities)

        if total_weight < 1e-10:
            # No similar experiences — fall back to mean
            mean_score = sum(exp.score for exp in self._buffer) / len(self._buffer)
            return WorldModelPrediction(
                predicted_score=mean_score,
                confidence=len(self._buffer) / self._buffer_size * 0.3,
                n_similar=0,
            )

        predicted = sum(w * s for w, s in similarities) / total_weight
        n_similar = sum(1 for w, _ in similarities if w > 0.1)
        confidence = min(1.0, len(self._buffer) / self._buffer_size)

        return WorldModelPrediction(
            predicted_score=min(1.0, max(0.0, predicted)),
            confidence=confidence,
            n_similar=n_similar,
        )

    def experience_count(self) -> int:
        """Number of stored experiences."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorldModel(experiences={self.experience_count()}, capacity={self._buffer_size})"


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelEvaluator
# ─────────────────────────────────────────────────────────────────────────────


def world_model_fitness(
    raw_score: float,
    prediction_error: float,
    accuracy_weight: float = 0.2,
) -> float:
    """Blend raw performance score with prediction accuracy.

    Args:
        raw_score: Actual fitness from the base evaluator.
        prediction_error: Absolute error ``|predicted - actual|``.
        accuracy_weight: Fraction of fitness attributed to prediction accuracy.
            ``0.0`` = pure performance; ``1.0`` = pure accuracy. Default ``0.2``.

    Returns:
        Blended fitness score in ``[0.0, 1.0]``.
    """
    accuracy_bonus = max(0.0, 1.0 - prediction_error)
    perf_weight = 1.0 - accuracy_weight
    return perf_weight * raw_score + accuracy_weight * accuracy_bonus


class WorldModelEvaluator(Evaluator):
    """Wraps a base evaluator and rewards agents for accurate self-prediction.

    Each agent gets its own :class:`WorldModel` that accumulates experience
    across evaluation calls.  After each real evaluation, the world model is
    updated and the blended fitness (performance + accuracy) is returned.

    Args:
        base_evaluator: Underlying evaluator for real fitness.
        accuracy_weight: Weight given to prediction accuracy in the final
            score. ``0.0`` = ignore world model, ``1.0`` = only accuracy matters.
            Default ``0.2``.
        buffer_size: Experience buffer capacity per agent. Default ``20``.
        decay: Temporal decay for experience weights. Default ``0.95``.
        min_confidence_for_blend: Only apply accuracy blending when world model
            confidence exceeds this threshold.  Below it, returns raw score.
            Default ``0.1``.
    """

    def __init__(
        self,
        base_evaluator: Evaluator,
        accuracy_weight: float = 0.2,
        buffer_size: int = 20,
        decay: float = 0.95,
        min_confidence_for_blend: float = 0.1,
    ) -> None:
        self._base = base_evaluator
        self._accuracy_weight = accuracy_weight
        self._buffer_size = buffer_size
        self._decay = decay
        self._min_conf = min_confidence_for_blend
        # Per-agent world models keyed by agent ID
        self._models: dict[str, WorldModel] = {}

    def _get_model(self, agent_id: str) -> WorldModel:
        """Return (or lazily create) the world model for *agent_id*."""
        if agent_id not in self._models:
            self._models[agent_id] = WorldModel(
                buffer_size=self._buffer_size,
                decay=self._decay,
            )
        return self._models[agent_id]

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Evaluate *agent*, update its world model, return blended fitness.

        Args:
            agent: Agent to evaluate.
            task: Task description.

        Returns:
            Blended fitness in ``[0.0, 1.0]``.
        """
        model = self._get_model(agent.id)

        # Query world model before running the real evaluator
        prediction = model.predict(task)

        # Run real evaluation
        raw_score = self._base.evaluate(agent, task)

        # Update world model with observation
        model.update(task, raw_score)

        # Only blend if we have enough confidence
        if prediction.confidence < self._min_conf:
            logger.debug(
                "Agent %s: world model confidence=%.2f < threshold, returning raw score=%.4f",
                agent.id[:8],
                prediction.confidence,
                raw_score,
            )
            return raw_score

        prediction_error = abs(prediction.predicted_score - raw_score)
        blended = world_model_fitness(raw_score, prediction_error, self._accuracy_weight)

        logger.debug(
            "Agent %s: raw=%.4f predicted=%.4f error=%.4f blended=%.4f",
            agent.id[:8],
            raw_score,
            prediction.predicted_score,
            prediction_error,
            blended,
        )
        return blended

    def get_model(self, agent_id: str) -> WorldModel | None:
        """Return the world model for *agent_id*, or ``None`` if it hasn't been evaluated."""
        return self._models.get(agent_id)

    def model_count(self) -> int:
        """Number of agent world models currently tracked."""
        return len(self._models)

    def prediction_errors(self) -> dict[str, float]:
        """Latest prediction error per agent (requires at least 2 evaluations).

        Returns a mapping of agent_id → mean absolute prediction error
        across all experiences in the world model buffer.  Agents with only
        one observation return 0.0.
        """
        errors: dict[str, float] = {}
        for aid, model in self._models.items():
            if model.experience_count() < 2:
                errors[aid] = 0.0
                continue
            # Reconstruct from buffer: compare each entry to the prediction
            # made on the preceding set (approximate — for diagnostics only)
            scores = [exp.score for exp in model._buffer]
            if len(scores) < 2:
                errors[aid] = 0.0
            else:
                diffs = [abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))]
                errors[aid] = sum(diffs) / len(diffs)
        return errors
