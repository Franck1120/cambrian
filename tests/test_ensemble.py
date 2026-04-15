"""Tests for cambrian.ensemble — AgentEnsemble and BoostingEnsemble."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.ensemble import (
    AgentEnsemble,
    BoostingEnsemble,
    EnsembleResult,
    exact_match_scorer,
    substring_scorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(response: str = "answer") -> Agent:
    g = Genome(system_prompt="system")
    a = Agent(genome=g)
    # Patch run() to return deterministic response
    a.run = MagicMock(return_value=response)  # type: ignore[method-assign]
    return a


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class TestScorers:
    def test_exact_match_correct(self) -> None:
        assert exact_match_scorer("42", "42") == 1.0

    def test_exact_match_wrong(self) -> None:
        assert exact_match_scorer("43", "42") == 0.0

    def test_exact_match_strips_whitespace(self) -> None:
        assert exact_match_scorer("  42  ", "42") == 1.0

    def test_substring_correct(self) -> None:
        assert substring_scorer("The answer is 42 degrees", "42") == 1.0

    def test_substring_wrong(self) -> None:
        assert substring_scorer("The answer is 43", "42") == 0.0

    def test_substring_case_insensitive(self) -> None:
        assert substring_scorer("ANSWER IS YES", "yes") == 1.0


# ---------------------------------------------------------------------------
# AgentEnsemble — init
# ---------------------------------------------------------------------------


class TestAgentEnsembleInit:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentEnsemble(agents=[])

    def test_weights_initialised_uniform(self) -> None:
        agents = [_make_agent(), _make_agent()]
        e = AgentEnsemble(agents)
        assert e.weights == [1.0, 1.0]

    def test_results_starts_empty(self) -> None:
        e = AgentEnsemble([_make_agent()])
        assert e.results == []


# ---------------------------------------------------------------------------
# AgentEnsemble — query
# ---------------------------------------------------------------------------


class TestAgentEnsembleQuery:
    def test_majority_answer_returned(self) -> None:
        a1 = _make_agent("yes")
        a2 = _make_agent("yes")
        a3 = _make_agent("no")
        e = AgentEnsemble([a1, a2, a3])
        result = e.query("is the sky blue?")
        assert result == "yes"

    def test_single_agent_returns_its_answer(self) -> None:
        e = AgentEnsemble([_make_agent("unique")])
        assert e.query("task") == "unique"

    def test_result_recorded(self) -> None:
        e = AgentEnsemble([_make_agent("yes")])
        e.query("task", correct_answer="yes")
        assert len(e.results) == 1
        r = e.results[0]
        assert isinstance(r, EnsembleResult)
        assert r.score == 1.0

    def test_results_returns_copy(self) -> None:
        e = AgentEnsemble([_make_agent("x")])
        e.query("t")
        r1 = e.results
        r1.clear()
        assert len(e.results) == 1

    def test_no_score_without_correct_answer(self) -> None:
        e = AgentEnsemble([_make_agent("yes")])
        e.query("task")
        assert e.results[0].score == 0.0

    def test_agent_error_treated_as_empty(self) -> None:
        a = _make_agent()
        a.run = MagicMock(side_effect=RuntimeError("err"))  # type: ignore[method-assign]
        e = AgentEnsemble([a, _make_agent("ok")])
        result = e.query("task")
        # "ok" wins because "" has equal weight but "ok" appears once
        assert result in ("ok", "")

    def test_temperature_passed_to_agents(self) -> None:
        # AgentEnsemble.query accepts temperature kwarg without error
        e = AgentEnsemble([_make_agent("yes")])
        e.query("task", temperature=0.1)  # must not raise


# ---------------------------------------------------------------------------
# BoostingEnsemble
# ---------------------------------------------------------------------------


class TestBoostingEnsembleInit:
    def test_defaults(self) -> None:
        e = BoostingEnsemble([_make_agent()])
        assert e._boost == 1.5
        assert e._decay == 0.5

    def test_custom_params(self) -> None:
        e = BoostingEnsemble([_make_agent()], boost_factor=2.0, decay_factor=0.3)
        assert e._boost == 2.0
        assert e._decay == 0.3


class TestBoostingEnsembleWeights:
    def test_correct_agent_boosted(self) -> None:
        a_correct = _make_agent("42")
        a_wrong = _make_agent("99")
        e = BoostingEnsemble([a_correct, a_wrong], boost_factor=2.0, decay_factor=0.5)
        e.query("task", correct_answer="42")
        w = e.weights
        # Correct agent (idx 0) boosted, wrong (idx 1) decayed
        assert w[0] > w[1]

    def test_weights_normalised(self) -> None:
        a1 = _make_agent("yes")
        a2 = _make_agent("no")
        e = BoostingEnsemble([a1, a2])
        e.query("task", correct_answer="yes")
        assert pytest.approx(sum(e.weights), abs=1e-9) == 1.0

    def test_no_update_without_correct_answer(self) -> None:
        e = BoostingEnsemble([_make_agent("x"), _make_agent("y")])
        before = e.weights[:]
        e.query("task")
        assert e.weights == before

    def test_all_wrong_resets_to_uniform(self) -> None:
        a1 = _make_agent("wrong")
        a2 = _make_agent("wrong")
        e = BoostingEnsemble(
            [a1, a2], boost_factor=1.5, decay_factor=0.0  # decay to 0
        )
        e.query("task", correct_answer="correct")
        # All weights → 0, should reset uniform
        assert all(w == pytest.approx(0.5, abs=1e-9) for w in e.weights)

    def test_multiple_rounds_accumulate(self) -> None:
        a_good = _make_agent("42")
        a_bad = _make_agent("99")
        e = BoostingEnsemble([a_good, a_bad])
        for _ in range(3):
            e.query("task", correct_answer="42")
        # After 3 rounds the good agent should dominate
        assert e.weights[0] > 0.8


class TestAgentWeightsSummary:
    def test_returns_sorted_by_weight_desc(self) -> None:
        a1 = _make_agent("42")
        a2 = _make_agent("wrong")
        e = BoostingEnsemble([a1, a2])
        e.query("task", correct_answer="42")
        summary = e.agent_weights_summary()
        assert summary[0]["weight"] >= summary[1]["weight"]

    def test_length_matches_agents(self) -> None:
        agents = [_make_agent("x") for _ in range(4)]
        e = BoostingEnsemble(agents)
        assert len(e.agent_weights_summary()) == 4
