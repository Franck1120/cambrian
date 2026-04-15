"""tests/test_moa_reflexion.py — Unit tests for MoA, QuantumTunneler, and Reflexion.

Covers:
- MixtureOfAgents: run, aggregation, empty response handling, last_responses
- QuantumTunneler: maybe_tunnel probability, tunnel_all, tunnel_count, clone genome
- QuantumTunneler: constructor validation
- ReflexionEvaluator._parse_reflection: format parsing, score clamping
- ReflexionEvaluator.evaluate: full loop, perfect score early exit, failure handling
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from cambrian.agent import Agent, Genome
from cambrian.moa import MixtureOfAgents, QuantumTunneler
from cambrian.reflexion import ReflexionEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_backend(*responses: str) -> MagicMock:
    backend = MagicMock()
    if len(responses) == 1:
        backend.generate.return_value = responses[0]
    else:
        backend.generate.side_effect = list(responses)
    return backend


def _agent(prompt: str = "You are helpful.", temperature: float = 0.7) -> Agent:
    return Agent(genome=Genome(system_prompt=prompt, temperature=temperature))


# ─────────────────────────────────────────────────────────────────────────────
# MixtureOfAgents
# ─────────────────────────────────────────────────────────────────────────────


class TestMixtureOfAgents:
    def test_requires_at_least_one_agent(self) -> None:
        with pytest.raises(ValueError, match="at least one agent"):
            MixtureOfAgents(agents=[], backend=_mock_backend("x"))

    def test_run_calls_each_agent(self) -> None:
        agents = [_agent("sys1"), _agent("sys2")]
        # 2 agent responses + 1 aggregator call
        backend = _mock_backend("resp1", "resp2", "aggregated")
        moa = MixtureOfAgents(agents=agents, backend=backend)
        result = moa.run("task")
        assert result == "aggregated"
        assert backend.generate.call_count == 3

    def test_last_responses_stored(self) -> None:
        agents = [_agent("s1"), _agent("s2")]
        backend = _mock_backend("r1", "r2", "agg")
        moa = MixtureOfAgents(agents=agents, backend=backend)
        moa.run("task")
        assert moa.last_responses == ["r1", "r2"]

    def test_agent_failure_skips_to_aggregator(self) -> None:
        agents = [_agent(), _agent()]
        backend = MagicMock()
        backend.generate.side_effect = [RuntimeError("fail"), "resp2", "aggregated"]
        moa = MixtureOfAgents(agents=agents, backend=backend)
        result = moa.run("task")
        assert result == "aggregated"

    def test_all_agents_fail_returns_empty(self) -> None:
        agents = [_agent()]
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("always fails")
        moa = MixtureOfAgents(agents=agents, backend=backend)
        result = moa.run("task")
        assert result == ""

    def test_responses_truncated_to_max_chars(self) -> None:
        long_response = "x" * 5000
        agents = [_agent()]
        backend = _mock_backend(long_response, "agg")
        moa = MixtureOfAgents(agents=agents, backend=backend, max_response_chars=100)
        moa.run("task")
        # Aggregator prompt should contain only 100 chars of the response
        agg_call_args = backend.generate.call_args_list[1][0][0]
        assert "x" * 101 not in agg_call_args

    def test_aggregator_failure_returns_longest_response(self) -> None:
        agents = [_agent(), _agent()]
        backend = MagicMock()
        backend.generate.side_effect = ["short", "much longer response here", RuntimeError("agg fail")]
        moa = MixtureOfAgents(agents=agents, backend=backend)
        result = moa.run("task")
        assert result == "much longer response here"

    def test_last_responses_is_copy(self) -> None:
        agents = [_agent()]
        backend = _mock_backend("r1", "agg")
        moa = MixtureOfAgents(agents=agents, backend=backend)
        moa.run("task")
        last = moa.last_responses
        last.clear()
        assert len(moa.last_responses) == 1


# ─────────────────────────────────────────────────────────────────────────────
# QuantumTunneler
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantumTunneler:
    def test_constructor_rejects_invalid_prob(self) -> None:
        with pytest.raises(ValueError):
            QuantumTunneler(tunnel_prob=1.5)
        with pytest.raises(ValueError):
            QuantumTunneler(tunnel_prob=-0.1)

    def test_zero_prob_never_tunnels(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.0, seed=42)
        agent = _agent()
        for _ in range(50):
            result = tunneler.maybe_tunnel(agent)
            assert result is agent  # exact same object

    def test_prob_one_always_tunnels(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, seed=42)
        agent = _agent()
        for _ in range(10):
            result = tunneler.maybe_tunnel(agent)
            assert result is not agent

    def test_tunnel_count_increments(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0)
        agent = _agent()
        assert tunneler.tunnel_count == 0
        tunneler.maybe_tunnel(agent)
        assert tunneler.tunnel_count == 1
        tunneler.maybe_tunnel(agent)
        assert tunneler.tunnel_count == 2

    def test_tunnel_count_unchanged_when_no_tunnel(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.0)
        agent = _agent()
        tunneler.maybe_tunnel(agent)
        assert tunneler.tunnel_count == 0

    def test_tunneled_agent_is_new_object(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0)
        agent = _agent("original prompt")
        result = tunneler.maybe_tunnel(agent)
        assert result.id != agent.id

    def test_tunneled_genome_temperature_in_range(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, temperature_range=(0.3, 1.5))
        agent = _agent()
        for _ in range(20):
            result = tunneler.maybe_tunnel(agent)
            assert 0.3 <= result.genome.temperature <= 1.5

    def test_tunneled_genome_strategy_from_list(self) -> None:
        from cambrian.moa import _STRATEGIES
        tunneler = QuantumTunneler(tunnel_prob=1.0)
        agent = _agent()
        for _ in range(20):
            result = tunneler.maybe_tunnel(agent)
            assert result.genome.strategy in _STRATEGIES

    def test_tunnel_all_applies_to_each(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0)
        agents = [_agent() for _ in range(5)]
        results = tunneler.tunnel_all(agents)
        assert len(results) == 5
        assert tunneler.tunnel_count == 5

    def test_tunnel_all_with_zero_prob(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.0)
        agents = [_agent() for _ in range(5)]
        results = tunneler.tunnel_all(agents)
        for orig, res in zip(agents, results):
            assert res is orig

    def test_tunneled_with_backend_uses_llm_prompt(self) -> None:
        backend = _mock_backend("LLM-generated novel prompt")
        tunneler = QuantumTunneler(tunnel_prob=1.0, backend=backend)
        agent = _agent()
        result = tunneler.maybe_tunnel(agent, task="solve the puzzle")
        assert result.genome.system_prompt == "LLM-generated novel prompt"
        backend.generate.assert_called_once()

    def test_tunneled_without_backend_uses_strategy(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, backend=None)
        agent = _agent()
        result = tunneler.maybe_tunnel(agent, task="solve")
        assert "reasoning" in result.genome.system_prompt.lower()

    def test_backend_failure_falls_back_to_strategy(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        tunneler = QuantumTunneler(tunnel_prob=1.0, backend=backend)
        agent = _agent()
        result = tunneler.maybe_tunnel(agent, task="task")
        # Should not raise and should produce a valid agent
        assert isinstance(result, Agent)
        assert "reasoning" in result.genome.system_prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# ReflexionEvaluator._parse_reflection
# ─────────────────────────────────────────────────────────────────────────────


class TestParseReflection:
    def test_parses_score(self) -> None:
        raw = "CRITIQUE:\n- Too vague\n\nSCORE: 0.75"
        critique, score = ReflexionEvaluator._parse_reflection(raw)
        assert score == pytest.approx(0.75)

    def test_parses_critique_section(self) -> None:
        raw = "CRITIQUE:\n- Missing detail\n- Wrong format\n\nSCORE: 0.5"
        critique, score = ReflexionEvaluator._parse_reflection(raw)
        assert "Missing detail" in critique

    def test_clamps_score_above_one(self) -> None:
        raw = "CRITIQUE:\n- ok\n\nSCORE: 1.5"
        _, score = ReflexionEvaluator._parse_reflection(raw)
        assert score == pytest.approx(1.0)

    def test_clamps_score_below_zero(self) -> None:
        raw = "CRITIQUE:\n- terrible\n\nSCORE: -0.3"
        _, score = ReflexionEvaluator._parse_reflection(raw)
        assert score == pytest.approx(0.0)

    def test_fallback_on_missing_score(self) -> None:
        raw = "No score here."
        _, score = ReflexionEvaluator._parse_reflection(raw)
        assert score == pytest.approx(0.5)

    def test_case_insensitive_headers(self) -> None:
        raw = "critique:\n- issue\n\nscore: 0.6"
        _, score = ReflexionEvaluator._parse_reflection(raw)
        assert score == pytest.approx(0.6)

    def test_empty_string_fallback(self) -> None:
        critique, score = ReflexionEvaluator._parse_reflection("")
        assert score == pytest.approx(0.5)
        assert critique == ""


# ─────────────────────────────────────────────────────────────────────────────
# ReflexionEvaluator.evaluate
# ─────────────────────────────────────────────────────────────────────────────


class TestReflexionEvaluator:
    def test_evaluate_returns_response_and_score(self) -> None:
        # Calls: initial gen, reflect (returns CRITIQUE + SCORE)
        reflect_raw = "CRITIQUE:\n- Good\n\nSCORE: 0.9"
        backend = _mock_backend("initial response", reflect_raw)
        ev = ReflexionEvaluator(backend=backend, n_reflections=1)
        response, score = ev.evaluate(_agent(), task="task")
        assert response == "initial response"
        assert score == pytest.approx(0.9)

    def test_perfect_score_stops_loop(self) -> None:
        reflect_raw = "CRITIQUE:\n- Perfect\n\nSCORE: 1.0"
        backend = _mock_backend("initial", reflect_raw)
        # n_reflections=2 but score=1.0 → only 1 reflect, no revise
        ev = ReflexionEvaluator(backend=backend, n_reflections=2)
        ev.evaluate(_agent(), task="task")
        # initial gen + 1 reflect = 2 calls (no revise for perfect score)
        assert backend.generate.call_count == 2

    def test_revision_happens_on_imperfect_score(self) -> None:
        reflect_raw = "CRITIQUE:\n- Needs work\n\nSCORE: 0.6"
        backend = _mock_backend("initial", reflect_raw, "revised response")
        ev = ReflexionEvaluator(backend=backend, n_reflections=1)
        response, _ = ev.evaluate(_agent(), task="task")
        # initial gen + reflect + revise = 3 calls; final response is revised
        assert response == "revised response"
        assert backend.generate.call_count == 3

    def test_initial_gen_failure_returns_empty(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("gen fail")
        ev = ReflexionEvaluator(backend=backend)
        response, score = ev.evaluate(_agent(), task="task")
        assert response == ""
        assert score == pytest.approx(0.0)

    def test_reflection_failure_keeps_current_response(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["initial", RuntimeError("reflect fail")]
        ev = ReflexionEvaluator(backend=backend, n_reflections=1)
        response, score = ev.evaluate(_agent(), task="task")
        assert response == "initial"
        assert score == pytest.approx(0.5)  # neutral fallback

    def test_revision_failure_keeps_current_response(self) -> None:
        reflect_raw = "CRITIQUE:\n- Weak\n\nSCORE: 0.3"
        backend = MagicMock()
        backend.generate.side_effect = ["initial", reflect_raw, RuntimeError("revise fail")]
        ev = ReflexionEvaluator(backend=backend, n_reflections=1)
        response, score = ev.evaluate(_agent(), task="task")
        assert response == "initial"
        assert score == pytest.approx(0.3)

    def test_multiple_reflection_cycles(self) -> None:
        """Two cycles: first imperfect → revise → second perfect → stop."""
        reflect1 = "CRITIQUE:\n- Too short\n\nSCORE: 0.5"
        reflect2 = "CRITIQUE:\n- Great!\n\nSCORE: 1.0"
        backend = _mock_backend("v1", reflect1, "v2", reflect2)
        ev = ReflexionEvaluator(backend=backend, n_reflections=2)
        response, score = ev.evaluate(_agent(), task="task")
        # v1 → reflect1 (0.5) → revise → v2 → reflect2 (1.0) → stop
        assert response == "v2"
        assert score == pytest.approx(1.0)

    def test_uses_agent_genome_system_prompt(self) -> None:
        backend = _mock_backend("response", "CRITIQUE:\n- ok\n\nSCORE: 0.8")
        ev = ReflexionEvaluator(backend=backend, n_reflections=1)
        agent = _agent("Custom system prompt here")
        ev.evaluate(agent, task="task")
        first_call = backend.generate.call_args_list[0]
        assert first_call[1].get("system") == "Custom system prompt here"
