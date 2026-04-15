# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Ensemble / Adaptive Boosting — Technique 55.

Combines outputs from multiple agents using adaptive weights derived from
their historical accuracy — conceptually similar to AdaBoost but applied
to LLM agent populations.

Components
----------
AgentEnsemble
    Queries N agents in parallel, aggregates answers via majority vote
    or weighted voting, and tracks per-agent accuracy.

BoostingEnsemble
    Extends AgentEnsemble with AdaBoost-inspired weight updates: agents
    that answer correctly gain weight; agents that answer incorrectly lose
    weight.  Weights are normalised after each round.

Usage::

    from cambrian.ensemble import BoostingEnsemble

    ensemble = BoostingEnsemble(agents)
    answer = ensemble.query(task, system_prompt, correct_answer="42")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from cambrian.agent import Agent


# ---------------------------------------------------------------------------
# Type alias for a simple scorer: answer, correct → float in [0,1]
# ---------------------------------------------------------------------------

Scorer = Callable[[str, str], float]


def exact_match_scorer(answer: str, correct: str) -> float:
    """Return 1.0 if *answer* == *correct* (stripped), else 0.0."""
    return 1.0 if answer.strip() == correct.strip() else 0.0


def substring_scorer(answer: str, correct: str) -> float:
    """Return 1.0 if *correct* is a substring of *answer*, else 0.0."""
    return 1.0 if correct.strip().lower() in answer.lower() else 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Result from one ensemble query."""

    final_answer: str
    individual_responses: list[str]
    weights_used: list[float]
    score: float  # correctness score, 0.0 if no correct_answer provided


# ---------------------------------------------------------------------------
# AgentEnsemble — basic majority-vote ensemble
# ---------------------------------------------------------------------------


class AgentEnsemble:
    """Query N agents and aggregate by majority vote.

    Parameters
    ----------
    agents:
        Agents to include in the ensemble.
    temperature:
        Shared sampling temperature (can be overridden per-query).
    """

    def __init__(
        self,
        agents: list[Agent],
        temperature: float = 0.7,
    ) -> None:
        if not agents:
            raise ValueError("AgentEnsemble requires at least one agent.")
        self._agents = list(agents)
        self._temperature = temperature
        self._weights: list[float] = [1.0] * len(agents)
        self._results: list[EnsembleResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def weights(self) -> list[float]:
        """Current per-agent weights (copy)."""
        return list(self._weights)

    @property
    def results(self) -> list[EnsembleResult]:
        """All stored query results (copy)."""
        return list(self._results)

    def query(
        self,
        task: str,
        correct_answer: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """Query all agents, return the weighted-majority answer."""
        temp = temperature if temperature is not None else self._temperature
        responses = self._gather_responses(task, temp)
        answer = self._aggregate(responses)
        score = 0.0
        if correct_answer:
            score = exact_match_scorer(answer, correct_answer)
        self._results.append(
            EnsembleResult(
                final_answer=answer,
                individual_responses=responses,
                weights_used=list(self._weights),
                score=score,
            )
        )
        return answer

    # ------------------------------------------------------------------
    # Protected helpers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _gather_responses(self, task: str, temperature: float) -> list[str]:
        """Query each agent and return their raw responses."""
        responses: list[str] = []
        for agent in self._agents:
            try:
                resp = agent.run(task)
            except Exception:  # noqa: BLE001
                resp = ""
            responses.append(resp)
        return responses

    def _aggregate(self, responses: list[str]) -> str:
        """Weighted majority vote: return the most-weighted unique answer."""
        if not responses:
            return ""

        # Tally weighted votes per answer
        vote_map: dict[str, float] = {}
        for resp, w in zip(responses, self._weights):
            key = resp.strip()
            vote_map[key] = vote_map.get(key, 0.0) + w

        return max(vote_map, key=lambda k: vote_map[k])


# ---------------------------------------------------------------------------
# BoostingEnsemble — AdaBoost-style adaptive weighting
# ---------------------------------------------------------------------------


class BoostingEnsemble(AgentEnsemble):
    """AdaBoost-inspired adaptive-weight ensemble.

    After each scored query, agent weights are updated:
    * correct answer → weight multiplied by ``boost_factor``
    * wrong answer → weight multiplied by ``decay_factor``
    Weights are then normalised so they sum to 1.

    Parameters
    ----------
    boost_factor:
        Multiplier for correct agents (default 1.5).
    decay_factor:
        Multiplier for incorrect agents (default 0.5).
    scorer:
        Function ``(response, correct_answer) → float ∈ [0,1]``.
        Defaults to ``exact_match_scorer``.
    """

    def __init__(
        self,
        agents: list[Agent],
        temperature: float = 0.7,
        boost_factor: float = 1.5,
        decay_factor: float = 0.5,
        scorer: Scorer = exact_match_scorer,
    ) -> None:
        super().__init__(agents, temperature)
        self._boost = boost_factor
        self._decay = decay_factor
        self._scorer = scorer

    def query(
        self,
        task: str,
        correct_answer: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """Query, aggregate, then update weights if *correct_answer* provided."""
        temp = temperature if temperature is not None else self._temperature
        responses = self._gather_responses(task, temp)
        answer = self._aggregate(responses)

        if correct_answer:
            self._update_weights(responses, correct_answer)

        score = self._scorer(answer, correct_answer) if correct_answer else 0.0
        self._results.append(
            EnsembleResult(
                final_answer=answer,
                individual_responses=responses,
                weights_used=list(self._weights),
                score=score,
            )
        )
        return answer

    def _update_weights(self, responses: list[str], correct_answer: str) -> None:
        """Boost correct agents, decay incorrect agents, normalise."""
        for i, resp in enumerate(responses):
            score = self._scorer(resp, correct_answer)
            if score >= 0.5:
                self._weights[i] *= self._boost
            else:
                self._weights[i] *= self._decay

        total = sum(self._weights)
        if total > 0:
            self._weights = [w / total for w in self._weights]
        else:
            # Reset to uniform if all weights collapse
            n = len(self._weights)
            self._weights = [1.0 / n] * n

    def agent_weights_summary(self) -> list[dict[str, float]]:
        """Return a list of ``{agent_id, weight}`` dicts (sorted by weight desc)."""
        pairs = [
            {"agent_id_hash": hash(a.agent_id) % 10**6, "weight": w}
            for a, w in zip(self._agents, self._weights)
        ]
        return sorted(pairs, key=lambda d: d["weight"], reverse=True)
