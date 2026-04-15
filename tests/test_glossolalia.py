# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.glossolalia — GlossaloliaReasoner and GlossaloliaEvaluator."""
from __future__ import annotations

from unittest.mock import MagicMock


from cambrian.agent import Agent, Genome
from cambrian.glossolalia import (
    GlossaloliaEvaluator,
    GlossaloliaReasoner,
    GlossaloliaResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(latent: str = "raw thoughts", synth: str = "final answer") -> MagicMock:
    b = MagicMock()
    b.generate.side_effect = [latent, synth]
    return b


def _reasoner(latent: str = "thoughts", synth: str = "answer") -> GlossaloliaReasoner:
    return GlossaloliaReasoner(backend=_backend(latent, synth))


def _make_agent(response: str = "agent response") -> Agent:
    g = Genome(system_prompt="system")
    a = Agent(genome=g)
    a.run = MagicMock(return_value=response)  # type: ignore[method-assign]
    return a


# ---------------------------------------------------------------------------
# GlossaloliaReasoner — init
# ---------------------------------------------------------------------------


class TestGlossaloliaReasonerInit:
    def test_defaults(self) -> None:
        r = GlossaloliaReasoner(backend=MagicMock())
        assert r._latent_temp == 1.2
        assert r._synth_temp == 0.6
        assert r._max_latent == 300

    def test_history_starts_empty(self) -> None:
        r = GlossaloliaReasoner(backend=MagicMock())
        assert r.history == []


# ---------------------------------------------------------------------------
# reason() — two-phase pipeline
# ---------------------------------------------------------------------------


class TestReason:
    def test_returns_answer_and_latent(self) -> None:
        r = _reasoner("my thoughts", "final answer")
        answer, latent = r.reason("task")
        assert answer == "final answer"
        assert latent == "my thoughts"

    def test_backend_called_twice(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["latent", "synth"]
        r = GlossaloliaReasoner(backend=b)
        r.reason("task")
        assert b.generate.call_count == 2

    def test_latent_temperature_used(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["lat", "syn"]
        r = GlossaloliaReasoner(backend=b, latent_temperature=1.5, synth_temperature=0.3)
        r.reason("task")
        calls = b.generate.call_args_list
        assert calls[0][1]["temperature"] == 1.5
        assert calls[1][1]["temperature"] == 0.3

    def test_temperature_override_applies_to_both(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["lat", "syn"]
        r = GlossaloliaReasoner(backend=b)
        r.reason("task", temperature_override=0.9)
        calls = b.generate.call_args_list
        assert calls[0][1]["temperature"] == 0.9
        assert calls[1][1]["temperature"] == 0.9

    def test_history_recorded(self) -> None:
        r = _reasoner("t", "a")
        r.reason("my task")
        assert len(r.history) == 1
        result = r.history[0]
        assert isinstance(result, GlossaloliaResult)
        assert result.task == "my task"
        assert result.latent_stream == "t"
        assert result.final_answer == "a"

    def test_history_returns_copy(self) -> None:
        r = _reasoner("t", "a")
        r.reason("task")
        h1 = r.history
        h1.clear()
        assert len(r.history) == 1

    def test_latent_truncated_to_max(self) -> None:
        long_latent = "x" * 10000
        b = MagicMock()
        b.generate.side_effect = [long_latent, "synth"]
        r = GlossaloliaReasoner(backend=b, max_latent_tokens=10)
        _, latent = r.reason("task")
        # max_latent_tokens=10 → 40 chars
        assert len(latent) <= 40

    def test_latent_failure_uses_placeholder(self) -> None:
        b = MagicMock()
        b.generate.side_effect = [RuntimeError("err"), "synth"]
        r = GlossaloliaReasoner(backend=b)
        answer, latent = r.reason("my task")
        assert "latent unavailable" in latent

    def test_synth_failure_falls_back_to_latent(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["latent stream", RuntimeError("err")]
        r = GlossaloliaReasoner(backend=b)
        answer, latent = r.reason("task")
        assert answer == "latent stream"

    def test_system_prompt_prepended_in_latent_phase(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["lat", "syn"]
        r = GlossaloliaReasoner(backend=b)
        r.reason("task", system_prompt="YOU ARE AN EXPERT")
        first_call_prompt = b.generate.call_args_list[0][0][0]
        assert "YOU ARE AN EXPERT" in first_call_prompt


# ---------------------------------------------------------------------------
# GlossaloliaEvaluator
# ---------------------------------------------------------------------------


class TestGlossaloliaEvaluator:
    def test_returns_score_from_base_evaluator(self) -> None:
        reasoner = _reasoner("thoughts", "enhanced")
        base = MagicMock()
        base.evaluate.return_value = 0.9
        evaluator = GlossaloliaEvaluator(base_evaluator=base, reasoner=reasoner)
        agent = _make_agent("initial")
        score = evaluator.evaluate(agent, "task")
        assert score == 0.9

    def test_base_evaluator_called_with_enhanced_response(self) -> None:
        reasoner = _reasoner("thoughts", "ENHANCED ANSWER")
        base = MagicMock()
        base.evaluate.return_value = 1.0
        evaluator = GlossaloliaEvaluator(base_evaluator=base, reasoner=reasoner)
        agent = _make_agent("initial")
        evaluator.evaluate(agent, "task")
        # Check that base received an agent whose run() returns the enhanced answer
        patched_agent = base.evaluate.call_args[0][0]
        assert patched_agent.run("x") == "ENHANCED ANSWER"

    def test_agent_run_failure_gracefully_handled(self) -> None:
        b = MagicMock()
        b.generate.side_effect = ["thoughts", "answer"]
        reasoner = GlossaloliaReasoner(backend=b)
        base = MagicMock()
        base.evaluate.return_value = 0.5
        evaluator = GlossaloliaEvaluator(base_evaluator=base, reasoner=reasoner)
        agent = _make_agent()
        agent.run = MagicMock(side_effect=RuntimeError("run error"))  # type: ignore[method-assign]
        score = evaluator.evaluate(agent, "task")
        assert score == 0.5
