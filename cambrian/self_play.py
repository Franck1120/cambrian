# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Self-play competition between agents.

Two agents compete on the same task.  A judge (LLM or heuristic) determines
the winner; the winner receives a fitness bonus, the loser a penalty.  This
creates a selective pressure toward agents that not only solve tasks well but
also outperform peers directly — complementing the standard single-agent
evaluation loop.

Architecture
------------

:class:`SelfPlayResult`
    Outcome of a single match: scores, winner, margin.

:class:`SelfPlayEvaluator`
    Wraps a base :class:`~cambrian.evaluator.Evaluator` and applies win/loss
    bonuses after head-to-head comparison.

:func:`run_tournament`
    Round-robin tournament across a population — each pair plays once.

Usage::

    from cambrian.self_play import SelfPlayEvaluator, run_tournament
    from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

    base = LLMJudgeEvaluator(judge_backend=backend, rubric="...")
    sp_eval = SelfPlayEvaluator(base_evaluator=base, win_bonus=0.1, loss_penalty=0.05)

    # Evaluate individual agent (no competitor — returns base score)
    score = sp_eval.evaluate(agent, task)

    # Run a full tournament
    results = run_tournament(population, sp_eval, task)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SelfPlayResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SelfPlayResult:
    """Outcome of a single head-to-head match.

    Attributes:
        agent_a_id: ID of the first agent.
        agent_b_id: ID of the second agent.
        score_a: Raw fitness score for agent A.
        score_b: Raw fitness score for agent B.
        winner_id: ID of the winning agent, or ``None`` on a draw.
        margin: Absolute score difference ``|score_a - score_b|``.
        task: Task description used in this match.
    """

    agent_a_id: str
    agent_b_id: str
    score_a: float
    score_b: float
    winner_id: str | None
    margin: float
    task: str

    @property
    def is_draw(self) -> bool:
        """``True`` when both agents scored identically."""
        return self.winner_id is None

    def loser_id(self) -> str | None:
        """ID of the losing agent, or ``None`` on a draw."""
        if self.winner_id == self.agent_a_id:
            return self.agent_b_id
        if self.winner_id == self.agent_b_id:
            return self.agent_a_id
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SelfPlayEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class SelfPlayEvaluator(Evaluator):
    """Wraps a base evaluator and applies win/loss bonuses from head-to-head matches.

    When called via :meth:`evaluate` on a single agent (no opponent), it
    delegates to the base evaluator unchanged.  Use :meth:`compete` for
    head-to-head evaluation, or :func:`run_tournament` for a full population
    round-robin.

    Args:
        base_evaluator: Underlying evaluator that produces raw scores.
        win_bonus: Fitness added to the winner's score. Default ``0.1``.
        loss_penalty: Fitness subtracted from the loser's score. Default ``0.05``.
        draw_delta: Fitness adjustment applied to both agents on a draw.
            Positive values reward draws; negative penalise them. Default ``0.0``.
        min_score: Lower bound for adjusted scores. Default ``0.0``.
        max_score: Upper bound for adjusted scores. Default ``1.0``.
        draw_threshold: Score margin below which the match is declared a draw.
            Default ``1e-6``.
    """

    def __init__(
        self,
        base_evaluator: Evaluator,
        win_bonus: float = 0.1,
        loss_penalty: float = 0.05,
        draw_delta: float = 0.0,
        min_score: float = 0.0,
        max_score: float = 1.0,
        draw_threshold: float = 1e-6,
    ) -> None:
        self._base = base_evaluator
        self._win_bonus = win_bonus
        self._loss_penalty = loss_penalty
        self._draw_delta = draw_delta
        self._min_score = min_score
        self._max_score = max_score
        self._draw_threshold = draw_threshold

    # ------------------------------------------------------------------
    # Evaluator ABC
    # ------------------------------------------------------------------

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Evaluate *agent* on *task* using the base evaluator only.

        No self-play adjustment is applied here; call :meth:`compete` for
        head-to-head scoring.
        """
        return self._base.evaluate(agent, task)

    # ------------------------------------------------------------------
    # Head-to-head
    # ------------------------------------------------------------------

    def compete(
        self,
        agent_a: "Agent",
        agent_b: "Agent",
        task: str,
        score_a: float | None = None,
        score_b: float | None = None,
    ) -> SelfPlayResult:
        """Pit *agent_a* against *agent_b* on *task*.

        If *score_a* / *score_b* are provided (pre-computed), they are reused;
        otherwise the base evaluator is called for each agent.

        Args:
            agent_a: First competitor.
            agent_b: Second competitor.
            task: Task description.
            score_a: Pre-computed score for agent_a (optional).
            score_b: Pre-computed score for agent_b (optional).

        Returns:
            :class:`SelfPlayResult` with winner, margin, and adjusted fitness
            already applied to the agents (via ``agent.fitness``).
        """
        if score_a is None:
            score_a = self._base.evaluate(agent_a, task)
        if score_b is None:
            score_b = self._base.evaluate(agent_b, task)

        margin = abs(score_a - score_b)
        is_draw = margin < self._draw_threshold

        if is_draw:
            winner_id = None
            adj_a = score_a + self._draw_delta
            adj_b = score_b + self._draw_delta
        elif score_a > score_b:
            winner_id = agent_a.id
            adj_a = score_a + self._win_bonus
            adj_b = score_b - self._loss_penalty
        else:
            winner_id = agent_b.id
            adj_a = score_a - self._loss_penalty
            adj_b = score_b + self._win_bonus

        # Clamp to [min_score, max_score]
        adj_a = max(self._min_score, min(self._max_score, adj_a))
        adj_b = max(self._min_score, min(self._max_score, adj_b))

        agent_a.fitness = adj_a
        agent_b.fitness = adj_b

        result = SelfPlayResult(
            agent_a_id=agent_a.id,
            agent_b_id=agent_b.id,
            score_a=score_a,
            score_b=score_b,
            winner_id=winner_id,
            margin=margin,
            task=task,
        )

        logger.debug(
            "Self-play: %s(%.3f) vs %s(%.3f) → winner=%s margin=%.4f",
            agent_a.id[:8],
            score_a,
            agent_b.id[:8],
            score_b,
            winner_id[:8] if winner_id else "draw",
            margin,
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# TournamentRecord
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TournamentRecord:
    """Aggregated results from a full round-robin tournament.

    Attributes:
        matches: All individual :class:`SelfPlayResult` objects.
        wins: Mapping of agent_id → win count.
        losses: Mapping of agent_id → loss count.
        draws: Mapping of agent_id → draw count.
        total_score: Mapping of agent_id → sum of raw scores across all matches.
    """

    matches: list[SelfPlayResult] = field(default_factory=list)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    draws: dict[str, int] = field(default_factory=dict)
    total_score: dict[str, float] = field(default_factory=dict)

    def ranking(self) -> list[str]:
        """Agent IDs sorted by wins (desc), then total score (desc)."""
        return sorted(
            self.wins.keys(),
            key=lambda aid: (self.wins.get(aid, 0), self.total_score.get(aid, 0.0)),
            reverse=True,
        )

    def match_count(self) -> int:
        """Total number of matches played."""
        return len(self.matches)

    def win_rate(self, agent_id: str) -> float:
        """Win rate for *agent_id* in ``[0.0, 1.0]``."""
        total = self.wins.get(agent_id, 0) + self.losses.get(agent_id, 0) + self.draws.get(agent_id, 0)
        if total == 0:
            return 0.0
        return self.wins.get(agent_id, 0) / total


# ─────────────────────────────────────────────────────────────────────────────
# round-robin tournament
# ─────────────────────────────────────────────────────────────────────────────


def run_tournament(
    population: list["Agent"],
    evaluator: SelfPlayEvaluator,
    task: str,
    pre_scores: dict[str, float] | None = None,
) -> TournamentRecord:
    """Run a round-robin tournament across the full *population*.

    Each pair of agents competes exactly once.  Fitness values are updated
    on agents in-place as an accumulation of match adjustments (clipped to
    ``[0.0, 1.0]`` after all matches).

    Args:
        population: All agents in the population.
        evaluator: :class:`SelfPlayEvaluator` used for scoring.
        task: Task description.
        pre_scores: Optional pre-computed raw scores keyed by agent ID.
            When provided, base-evaluator calls are skipped.

    Returns:
        :class:`TournamentRecord` with full match history and win/loss stats.
    """
    record = TournamentRecord()

    # Initialise counters
    for agent in population:
        aid = agent.id
        record.wins[aid] = 0
        record.losses[aid] = 0
        record.draws[aid] = 0
        record.total_score[aid] = 0.0

    n = len(population)
    for i in range(n):
        for j in range(i + 1, n):
            a = population[i]
            b = population[j]
            sa = pre_scores.get(a.id) if pre_scores else None
            sb = pre_scores.get(b.id) if pre_scores else None
            result = evaluator.compete(a, b, task, score_a=sa, score_b=sb)
            record.matches.append(result)

            # Tally
            record.total_score[a.id] = record.total_score.get(a.id, 0.0) + result.score_a
            record.total_score[b.id] = record.total_score.get(b.id, 0.0) + result.score_b

            if result.is_draw:
                record.draws[a.id] = record.draws.get(a.id, 0) + 1
                record.draws[b.id] = record.draws.get(b.id, 0) + 1
            elif result.winner_id == a.id:
                record.wins[a.id] = record.wins.get(a.id, 0) + 1
                record.losses[b.id] = record.losses.get(b.id, 0) + 1
            else:
                record.wins[b.id] = record.wins.get(b.id, 0) + 1
                record.losses[a.id] = record.losses.get(a.id, 0) + 1

    logger.info(
        "Tournament complete: %d agents, %d matches",
        n,
        record.match_count(),
    )
    return record
