"""Tests for cambrian.red_team — RedTeamAgent, RobustnessEvaluator, RedTeamSession."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.red_team import (
    AttackResult,
    RedTeamAgent,
    RedTeamSession,
    RobustnessEvaluator,
    RobustnessReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(response: str = '["attack 1", "attack 2"]') -> MagicMock:
    b = MagicMock()
    b.generate.return_value = response
    return b


def _make_agent(response: str = "safe response") -> Agent:
    g = Genome(system_prompt="agent system prompt")
    a = Agent(genome=g)
    a.run = MagicMock(return_value=response)  # type: ignore[method-assign]
    a.fitness = 0.7
    return a


def _base_evaluator(score: float = 0.8) -> MagicMock:
    ev = MagicMock()
    ev.evaluate.return_value = score
    return ev


# ---------------------------------------------------------------------------
# RedTeamAgent
# ---------------------------------------------------------------------------


class TestRedTeamAgentInit:
    def test_defaults(self) -> None:
        rta = RedTeamAgent(backend=_backend())
        assert rta._n == 5
        assert rta._temperature == 1.0


class TestGenerateAttacks:
    def test_parses_json_array(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a1", "a2", "a3"]'), n_attacks=3)
        attacks = rta.generate_attacks("task")
        assert attacks == ["a1", "a2", "a3"]

    def test_limits_to_n_attacks(self) -> None:
        rta = RedTeamAgent(
            backend=_backend('["a", "b", "c", "d", "e"]'), n_attacks=2
        )
        attacks = rta.generate_attacks("task")
        assert len(attacks) == 2

    def test_fallback_on_llm_error(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("API error")
        rta = RedTeamAgent(backend=b)
        attacks = rta.generate_attacks("complete this task")
        assert len(attacks) > 0  # fallback provides something

    def test_fallback_on_invalid_json(self) -> None:
        rta = RedTeamAgent(backend=_backend("not json at all"))
        attacks = rta.generate_attacks("task")
        assert len(attacks) > 0

    def test_strips_markdown_fences(self) -> None:
        rta = RedTeamAgent(backend=_backend('```json\n["a1", "a2"]\n```'))
        attacks = rta.generate_attacks("task")
        assert "a1" in attacks


# ---------------------------------------------------------------------------
# RobustnessEvaluator
# ---------------------------------------------------------------------------


class TestRobustnessEvaluator:
    def test_parses_score(self) -> None:
        ev = RobustnessEvaluator(judge_backend=_backend("0.85"))
        score = ev.score("task", "attack", "response")
        assert pytest.approx(score, abs=1e-9) == 0.85

    def test_clamps_above_one(self) -> None:
        ev = RobustnessEvaluator(judge_backend=_backend("1.5"))
        assert ev.score("t", "a", "r") == 1.0

    def test_clamps_below_zero(self) -> None:
        ev = RobustnessEvaluator(judge_backend=_backend("-0.3"))
        assert ev.score("t", "a", "r") == 0.0

    def test_neutral_on_error(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("err")
        ev = RobustnessEvaluator(judge_backend=b)
        score = ev.score("t", "a", "r")
        assert score == 0.5

    def test_neutral_on_invalid_response(self) -> None:
        ev = RobustnessEvaluator(judge_backend=_backend("no number here"))
        score = ev.score("t", "a", "r")
        assert score == 0.5


# ---------------------------------------------------------------------------
# RedTeamSession
# ---------------------------------------------------------------------------


class TestRedTeamSessionInit:
    def test_reports_starts_empty(self) -> None:
        rta = RedTeamAgent(backend=_backend())
        session = RedTeamSession(red_team_agent=rta, base_evaluator=_base_evaluator())
        assert session.reports == []


class TestRedTeamSessionRun:
    def test_returns_robustness_report(self) -> None:
        rta = RedTeamAgent(backend=_backend('["attack 1", "attack 2"]'), n_attacks=2)
        session = RedTeamSession(
            red_team_agent=rta, base_evaluator=_base_evaluator(0.8)
        )
        report = session.run(_make_agent(), "task")
        assert isinstance(report, RobustnessReport)

    def test_normal_score_used(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a"]'), n_attacks=1)
        session = RedTeamSession(
            red_team_agent=rta,
            base_evaluator=_base_evaluator(0.9),
            normal_weight=1.0,  # only normal score
        )
        report = session.run(_make_agent(), "task")
        assert pytest.approx(report.robustness_score, abs=1e-9) == 0.9

    def test_adversarial_score_combined(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a1"]'), n_attacks=1)
        rob_ev = MagicMock()
        rob_ev.score.return_value = 1.0
        session = RedTeamSession(
            red_team_agent=rta,
            base_evaluator=_base_evaluator(0.0),
            robustness_evaluator=rob_ev,
            normal_weight=0.0,  # only adversarial score
        )
        report = session.run(_make_agent(), "task")
        assert pytest.approx(report.robustness_score, abs=1e-9) == 1.0

    def test_attack_results_stored(self) -> None:
        rta = RedTeamAgent(backend=_backend('["attack 1", "attack 2"]'), n_attacks=2)
        session = RedTeamSession(
            red_team_agent=rta, base_evaluator=_base_evaluator()
        )
        report = session.run(_make_agent(), "task")
        assert len(report.attack_results) == 2
        assert all(isinstance(r, AttackResult) for r in report.attack_results)

    def test_reports_returns_copy(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a"]'), n_attacks=1)
        session = RedTeamSession(
            red_team_agent=rta, base_evaluator=_base_evaluator()
        )
        session.run(_make_agent(), "task")
        r1 = session.reports
        r1.clear()
        assert len(session.reports) == 1

    def test_agent_run_error_handled(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a"]'), n_attacks=1)
        agent = _make_agent()
        agent.run = MagicMock(side_effect=RuntimeError("err"))  # type: ignore[method-assign]
        session = RedTeamSession(
            red_team_agent=rta, base_evaluator=_base_evaluator()
        )
        report = session.run(agent, "task")
        # Empty response → score 0.0
        assert report.attack_results[0].robustness_score == 0.0

    def test_no_robustness_evaluator_defaults_to_neutral(self) -> None:
        rta = RedTeamAgent(backend=_backend('["a"]'), n_attacks=1)
        session = RedTeamSession(
            red_team_agent=rta,
            base_evaluator=_base_evaluator(),
            robustness_evaluator=None,
        )
        report = session.run(_make_agent("non-empty response"), "task")
        assert report.attack_results[0].robustness_score == 0.5
