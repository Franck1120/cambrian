# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Curriculum learning — progressive task difficulty.

Instead of exposing agents to the hardest task from the start (which can
stall early evolution), a curriculum schedules a sequence of tasks from
easy to hard.  The population advances to the next stage only when its
fitness meets a configurable threshold.

This implements a simple *mastery-based* curriculum:

1. The scheduler starts at stage 0.
2. After each generation, ``advance(population_fitness)`` is called.
3. If the fitness criterion is met, the scheduler advances.
4. ``current_task()`` always returns the appropriate task for this stage.

Example::

    from cambrian.curriculum import CurriculumScheduler, CurriculumStage

    curriculum = CurriculumScheduler(stages=[
        CurriculumStage(task="Reverse a string.", difficulty=0.1, threshold=0.6),
        CurriculumStage(task="Sort a list.", difficulty=0.4, threshold=0.7),
        CurriculumStage(task="Implement quicksort with edge cases.", difficulty=0.9, threshold=0.8),
    ])

    def on_gen(gen, pop):
        fitness_values = [a.fitness or 0 for a in pop]
        mean_fitness = sum(fitness_values) / len(fitness_values)
        advanced = curriculum.advance(mean_fitness)
        if advanced:
            print(f"Advanced to stage {curriculum.stage_index}")

    engine.evolve(seeds, task=curriculum.current_task(), n_generations=30,
                  on_generation=on_gen)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumStage:
    """One stage in a curriculum schedule.

    Attributes:
        task: Natural-language task description for this difficulty level.
        difficulty: Numeric difficulty indicator in ``[0.0, 1.0]``.
            Used for logging and statistics only — does not affect scheduling.
        threshold: Minimum population fitness metric needed to advance.
            The metric used is supplied by the caller to :meth:`CurriculumScheduler.advance`.
        max_generations: Maximum generations to spend on this stage before
            forcibly advancing.  ``None`` means no limit.
        metadata: Optional dict of extra stage annotations (tags, topic, etc.).
    """

    task: str
    difficulty: float = 0.0
    threshold: float = 0.7
    max_generations: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.difficulty <= 1.0):
            raise ValueError(f"difficulty must be in [0, 1], got {self.difficulty}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


class CurriculumScheduler:
    """Manages progressive task difficulty for an evolution run.

    Args:
        stages: Ordered list of :class:`CurriculumStage` objects from
            easiest to hardest.
        metric: Which statistic of the population fitness to compare against
            the stage threshold.  One of ``"mean"``, ``"best"``, ``"median"``.
            Default ``"mean"``.
        advance_patience: Number of consecutive generations where the
            metric exceeds the threshold before advancing.  Default ``1``
            (advance immediately on first qualifying generation).
        loop: If ``True``, restart from stage 0 after completing the last
            stage.  Default ``False`` (stay on last stage).

    Raises:
        ValueError: If *stages* is empty.
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        metric: str = "mean",
        advance_patience: int = 1,
        loop: bool = False,
    ) -> None:
        if not stages:
            raise ValueError("CurriculumScheduler requires at least one stage.")
        if metric not in ("mean", "best", "median"):
            raise ValueError("metric must be 'mean', 'best', or 'median'.")
        if advance_patience < 1:
            raise ValueError("advance_patience must be >= 1.")

        self._stages = list(stages)
        self._metric = metric
        self._patience = advance_patience
        self._loop = loop

        self._stage_idx: int = 0
        self._generations_on_stage: int = 0
        self._consecutive_passing: int = 0
        self._history: list[dict[str, Any]] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def stage_index(self) -> int:
        """Current stage index (0-based)."""
        return self._stage_idx

    @property
    def current_stage(self) -> CurriculumStage:
        """The currently active :class:`CurriculumStage`."""
        return self._stages[self._stage_idx]

    @property
    def is_complete(self) -> bool:
        """``True`` if the final stage has been completed (and *loop* is off)."""
        return (
            not self._loop
            and self._stage_idx == len(self._stages) - 1
            and self._consecutive_passing >= self._patience
        )

    @property
    def progress(self) -> float:
        """Overall progress through the curriculum in ``[0.0, 1.0]``."""
        return self._stage_idx / len(self._stages)

    # ── Core API ──────────────────────────────────────────────────────────────

    def current_task(self) -> str:
        """Return the task string for the current stage.

        Returns:
            Task description to pass to the evolution engine.
        """
        return self._stages[self._stage_idx].task

    def advance(self, fitness_values: list[float]) -> bool:
        """Update internal state and advance the stage if criteria are met.

        Call this at the end of each generation with the fitness values of
        the current population.

        Args:
            fitness_values: List of fitness scores for the current population.

        Returns:
            ``True`` if the curriculum advanced to the next stage, ``False``
            otherwise.
        """
        if not fitness_values:
            return False

        metric_value = self._compute_metric(fitness_values)
        stage = self._stages[self._stage_idx]
        self._generations_on_stage += 1

        passing = metric_value >= stage.threshold

        # Check forced advance (max_generations exceeded)
        forced = (
            stage.max_generations is not None
            and self._generations_on_stage >= stage.max_generations
        )

        if passing:
            self._consecutive_passing += 1
        else:
            self._consecutive_passing = 0

        should_advance = (self._consecutive_passing >= self._patience) or forced

        self._history.append({
            "stage": self._stage_idx,
            "metric": round(metric_value, 4),
            "threshold": stage.threshold,
            "passing": passing,
            "consecutive": self._consecutive_passing,
            "advanced": False,
        })

        if should_advance and self._stage_idx < len(self._stages) - 1:
            self._stage_idx += 1
            self._generations_on_stage = 0
            self._consecutive_passing = 0
            self._history[-1]["advanced"] = True
            return True

        if should_advance and self._loop:
            self._stage_idx = 0
            self._generations_on_stage = 0
            self._consecutive_passing = 0
            self._history[-1]["advanced"] = True
            return True

        return False

    def reset(self) -> None:
        """Reset the scheduler back to stage 0."""
        self._stage_idx = 0
        self._generations_on_stage = 0
        self._consecutive_passing = 0
        self._history.clear()

    # ── Statistics ────────────────────────────────────────────────────────────

    def stage_summary(self) -> list[dict[str, Any]]:
        """Per-stage aggregated statistics from the history log.

        Returns:
            List of dicts with ``stage``, ``name``, ``difficulty``,
            ``threshold``, ``generations_spent``, ``best_metric``.
        """
        from collections import defaultdict

        by_stage: dict[int, list[float]] = defaultdict(list)
        for entry in self._history:
            by_stage[entry["stage"]].append(entry["metric"])

        result: list[dict[str, Any]] = []
        for i, stage in enumerate(self._stages):
            metrics = by_stage.get(i, [])
            result.append({
                "stage": i,
                "task": stage.task[:50],
                "difficulty": stage.difficulty,
                "threshold": stage.threshold,
                "generations_spent": len(metrics),
                "best_metric": max(metrics) if metrics else 0.0,
                "mean_metric": sum(metrics) / len(metrics) if metrics else 0.0,
            })
        return result

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full per-generation history log (read-only view)."""
        return list(self._history)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_metric(self, fitness_values: list[float]) -> float:
        """Compute the configured metric from a list of fitness values."""
        if self._metric == "best":
            return max(fitness_values)
        if self._metric == "median":
            s = sorted(fitness_values)
            mid = len(s) // 2
            return (s[mid - 1] + s[mid]) / 2.0 if len(s) % 2 == 0 else float(s[mid])
        # default: mean
        return sum(fitness_values) / len(fitness_values)

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(stage={self._stage_idx}/{len(self._stages)}, "
            f"metric={self._metric!r}, is_complete={self.is_complete})"
        )


# ── Convenience factories ──────────────────────────────────────────────────────


def make_coding_curriculum() -> CurriculumScheduler:
    """Return a four-stage coding curriculum from trivial to advanced.

    Stages:
        0. Print "Hello, World!"
        1. Reverse a string
        2. FizzBuzz (1–30)
        3. Binary search implementation

    Returns:
        A :class:`CurriculumScheduler` with ``metric="best"`` and
        ``advance_patience=2``.
    """
    stages = [
        CurriculumStage(
            task='Print "Hello, World!" to stdout.',
            difficulty=0.05,
            threshold=0.8,
            max_generations=5,
        ),
        CurriculumStage(
            task="Write a Python function that reverses a string without using slicing.",
            difficulty=0.25,
            threshold=0.75,
        ),
        CurriculumStage(
            task=(
                "Write a Python script that prints FizzBuzz for numbers 1-30: "
                "print 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
                "'FizzBuzz' for both, the number otherwise."
            ),
            difficulty=0.5,
            threshold=0.7,
        ),
        CurriculumStage(
            task=(
                "Implement binary search in Python. "
                "Function signature: def binary_search(arr: list, target: int) -> int. "
                "Return the index or -1 if not found."
            ),
            difficulty=0.8,
            threshold=0.65,
        ),
    ]
    return CurriculumScheduler(stages=stages, metric="best", advance_patience=2)


def make_reasoning_curriculum() -> CurriculumScheduler:
    """Return a five-stage curriculum focused on reasoning tasks.

    Stages progress from factual recall → multi-step reasoning → logic puzzles.

    Returns:
        A :class:`CurriculumScheduler` with ``metric="mean"`` and
        ``advance_patience=3``.
    """
    stages = [
        CurriculumStage(task="What is 2 + 2? Answer with only the number.", difficulty=0.01, threshold=0.9),
        CurriculumStage(task="What is the capital of France?", difficulty=0.1, threshold=0.85),
        CurriculumStage(task="If all cats are mammals and Whiskers is a cat, is Whiskers a mammal? Explain.", difficulty=0.3, threshold=0.75),
        CurriculumStage(task="A farmer has 17 sheep. All but 9 run away. How many are left? Show your reasoning.", difficulty=0.5, threshold=0.7),
        CurriculumStage(task="Solve the Tower of Hanoi for 3 discs. List the moves.", difficulty=0.85, threshold=0.6),
    ]
    return CurriculumScheduler(stages=stages, metric="mean", advance_patience=3)
