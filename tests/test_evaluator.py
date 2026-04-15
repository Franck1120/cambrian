# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian evaluators."""

from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evaluators.composite import CompositeEvaluator
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator


# ── Fakes ─────────────────────────────────────────────────────────────────────

def _make_agent(prompt: str = "test agent") -> Agent:
    return Agent(genome=Genome(system_prompt=prompt))


class _FixedBackend:
    """Returns a pre-configured response string."""

    model_name = "fixed"

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._response


# ── LLMJudgeEvaluator ─────────────────────────────────────────────────────────

class TestLLMJudgeEvaluator:
    """LLMJudgeEvaluator tests.

    agent.run() is monkeypatched so no real LLM calls are made.
    The judge backend is replaced with _FixedBackend to control LLM output.
    """

    def _judge(self, response: str) -> LLMJudgeEvaluator:
        return LLMJudgeEvaluator(judge_backend=_FixedBackend(response))  # type: ignore[arg-type]

    def test_parses_json_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr(agent, "run", lambda task: "some answer")
        score = self._judge('{"score": 8}').evaluate(agent, "test")
        assert score == pytest.approx(0.8)

    def test_parses_plain_number_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr(agent, "run", lambda task: "answer")
        score = self._judge("7").evaluate(agent, "test")
        assert score == pytest.approx(0.7)

    def test_returns_zero_on_unparseable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr(agent, "run", lambda task: "answer")
        score = self._judge("I cannot judge this.").evaluate(agent, "test")
        assert score == pytest.approx(0.0)

    def test_score_clamped_to_unit_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr(agent, "run", lambda task: "answer")
        score = self._judge('{"score": 15}').evaluate(agent, "test")
        assert 0.0 <= score <= 1.0

    def test_json_in_markdown_fence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        monkeypatch.setattr(agent, "run", lambda task: "answer")
        score = self._judge('```json\n{"score": 6}\n```').evaluate(agent, "test")
        assert score == pytest.approx(0.6)


# ── CompositeEvaluator ────────────────────────────────────────────────────────

class TestCompositeEvaluator:
    def _fixed_scorer(self, value: float) -> object:
        from cambrian.evaluator import Evaluator

        class _Fixed(Evaluator):
            def evaluate(self_inner, agent: Agent, task: str) -> float:
                return value

        return _Fixed()

    def test_weighted_mean(self) -> None:
        comp = CompositeEvaluator(
            evaluators=[self._fixed_scorer(0.4), self._fixed_scorer(0.8)],  # type: ignore[list-item]
            weights=[1.0, 3.0],
            aggregate="mean",
        )
        agent = _make_agent()
        # weighted mean: (0.4*1 + 0.8*3) / 4 = 2.8/4 = 0.7
        assert comp.evaluate(agent, "task") == pytest.approx(0.7)

    def test_min_aggregate(self) -> None:
        comp = CompositeEvaluator(
            evaluators=[self._fixed_scorer(0.9), self._fixed_scorer(0.3)],  # type: ignore[list-item]
            aggregate="min",
        )
        assert comp.evaluate(_make_agent(), "task") == pytest.approx(0.3)

    def test_equal_weights_by_default(self) -> None:
        comp = CompositeEvaluator(
            evaluators=[self._fixed_scorer(0.4), self._fixed_scorer(0.6)],  # type: ignore[list-item]
            aggregate="mean",
        )
        assert comp.evaluate(_make_agent(), "task") == pytest.approx(0.5)

    def test_single_scorer(self) -> None:
        comp = CompositeEvaluator(
            evaluators=[self._fixed_scorer(0.77)],  # type: ignore[list-item]
        )
        assert comp.evaluate(_make_agent(), "task") == pytest.approx(0.77)

    def test_requires_at_least_one_scorer(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
            comp = CompositeEvaluator(evaluators=[])
            comp.evaluate(_make_agent(), "task")
