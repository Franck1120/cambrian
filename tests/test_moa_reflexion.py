"""Tests for cambrian.moa (MixtureOfAgents, QuantumTunneler) and cambrian.reflexion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.moa import MixtureOfAgents, MoAResult, QuantumTunneler, TunnelEvent
from cambrian.reflexion import (
    ReflexionAgent,
    ReflexionEvaluator,
    ReflexionResult,
    ReflexionRound,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _agent(fitness: float | None = None, prompt: str = "Agent") -> Agent:
    backend = MagicMock()
    backend.generate.return_value = f"response from {prompt}"
    a = Agent(Genome(system_prompt=prompt), backend=backend)
    if fitness is not None:
        a.fitness = fitness
    return a


def _pop(n: int = 4) -> list[Agent]:
    return [_agent(0.5 + i * 0.1, f"agent{i}") for i in range(n)]


class _ConstEvaluator:
    def __init__(self, score: float = 0.6) -> None:
        self._score = score

    def evaluate(self, agent: Agent, task: str) -> float:
        return self._score


# ── MoAResult ──────────────────────────────────────────────────────────────────


class TestMoAResult:
    def test_fields(self) -> None:
        r = MoAResult(final_answer="ans", individual_answers=["a", "b"], n_agents=2)
        assert r.final_answer == "ans"
        assert r.n_agents == 2

    def test_default_individual_answers(self) -> None:
        r = MoAResult(final_answer="x")
        assert r.individual_answers == []


# ── MixtureOfAgents ────────────────────────────────────────────────────────────


class TestMixtureOfAgents:
    def test_raises_empty_agents(self) -> None:
        with pytest.raises(ValueError, match="agents"):
            MixtureOfAgents(agents=[])

    def test_single_agent_no_aggregation(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "direct answer"
        a = Agent(Genome(), backend=backend)
        moa = MixtureOfAgents([a])
        result = moa.run("task")
        assert result.final_answer == "direct answer"
        assert result.n_agents == 1

    def test_multiple_agents_aggregates(self) -> None:
        agents = []
        for i in range(3):
            b = MagicMock()
            b.generate.side_effect = [f"answer{i}", "aggregated"]
            a = Agent(Genome(), backend=b)
            agents.append(a)

        agg_backend = MagicMock()
        agg_backend.generate.return_value = "aggregated final"

        moa = MixtureOfAgents(agents, aggregator_backend=agg_backend)
        result = moa.run("task")
        assert result.final_answer == "aggregated final"
        assert result.n_agents == 3
        assert len(result.individual_answers) == 3

    def test_fallback_on_aggregation_error(self) -> None:
        agents = [_agent(), _agent()]
        agg_backend = MagicMock()
        agg_backend.generate.side_effect = RuntimeError("API down")
        moa = MixtureOfAgents(agents, aggregator_backend=agg_backend)
        result = moa.run("task")
        # Fallback: returns longest individual answer
        assert result.final_answer in result.individual_answers

    def test_n_agents_sampling(self) -> None:
        agents = _pop(6)
        agg_backend = MagicMock()
        agg_backend.generate.return_value = "aggregated"
        moa = MixtureOfAgents(agents, aggregator_backend=agg_backend, n_agents=3)
        result = moa.run("task")
        assert result.n_agents == 3

    def test_agent_error_skipped(self) -> None:
        good_backend = MagicMock()
        good_backend.generate.return_value = "good answer"
        bad_backend = MagicMock()
        bad_backend.generate.side_effect = [RuntimeError("fail"), "aggregated"]

        agg_backend = MagicMock()
        agg_backend.generate.return_value = "final"

        agents = [
            Agent(Genome(), backend=bad_backend),
            Agent(Genome(), backend=good_backend),
        ]
        moa = MixtureOfAgents(agents, aggregator_backend=agg_backend)
        result = moa.run("task")
        # Should still work with 1 valid answer
        assert isinstance(result, MoAResult)

    def test_all_agents_fail_returns_empty(self) -> None:
        bad_backend = MagicMock()
        bad_backend.generate.side_effect = RuntimeError("all fail")
        agents = [Agent(Genome(), backend=bad_backend)]
        moa = MixtureOfAgents(agents)
        result = moa.run("task")
        assert result.final_answer == ""

    def test_from_population(self) -> None:
        pop = _pop(4)
        moa = MixtureOfAgents.from_population(pop)
        assert isinstance(moa, MixtureOfAgents)

    def test_uses_first_agent_backend_if_no_aggregator(self) -> None:
        # With a single agent, no aggregation needed
        a = _agent()
        moa = MixtureOfAgents([a])
        result = moa.run("task")
        assert isinstance(result.final_answer, str)


# ── QuantumTunneler ────────────────────────────────────────────────────────────


class TestQuantumTunneler:
    def test_no_tunneling_at_zero_prob(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.0)
        pop = _pop(4)
        original_ids = [a.id for a in pop]
        result = tunneler.apply(pop)
        assert [a.id for a in result] == original_ids

    def test_all_tunnel_at_one_prob(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=42)
        pop = _pop(4)
        original_ids = {a.id for a in pop}
        result = tunneler.apply(pop)
        new_ids = {a.id for a in result}
        # All agents should have new IDs
        assert len(new_ids.intersection(original_ids)) == 0

    def test_elites_protected(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=True, n_elites=2, seed=42)
        pop = _pop(4)
        original_ids = [a.id for a in pop[:2]]  # first 2 are elites
        result = tunneler.apply(pop)
        # First 2 should be unchanged
        assert result[0].id == original_ids[0]
        assert result[1].id == original_ids[1]

    def test_events_recorded(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=42)
        pop = _pop(4)
        tunneler.apply(pop)
        assert tunneler.tunnel_count == 4

    def test_event_fields(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=42)
        pop = [_agent(0.7)]
        tunneler.apply(pop)
        events = tunneler.events
        assert len(events) == 1
        assert isinstance(events[0], TunnelEvent)
        assert events[0].old_fitness == pytest.approx(0.7)
        assert events[0].generation == 0

    def test_generation_incremented(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.0)
        tunneler.apply(_pop(2))
        tunneler.apply(_pop(2))
        # After 2 calls, generation should be 2
        assert tunneler.summary()["generations_run"] == 2

    def test_result_same_size(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.5, seed=42)
        pop = _pop(8)
        result = tunneler.apply(pop)
        assert len(result) == len(pop)

    def test_tunneled_agent_has_backend(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=42)
        pop = [_agent()]
        result = tunneler.apply(pop)
        # New agent should inherit backend from original
        assert result[0].backend is not None

    def test_summary_fields(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=0.5, seed=42)
        tunneler.apply(_pop(4))
        s = tunneler.summary()
        assert "tunnel_prob" in s
        assert "total_events" in s
        assert "generations_run" in s
        assert s["generations_run"] == 1

    def test_random_genome_has_strategy(self) -> None:
        genome = QuantumTunneler._random_genome("gpt-4o-mini")
        assert genome.strategy
        assert genome.system_prompt

    def test_no_events_at_start(self) -> None:
        tunneler = QuantumTunneler()
        assert tunneler.tunnel_count == 0
        assert tunneler.events == []


# ── ReflexionRound ─────────────────────────────────────────────────────────────


class TestReflexionRound:
    def test_defaults(self) -> None:
        r = ReflexionRound(round_number=0, response="hello")
        assert r.critique == ""
        assert not r.improved

    def test_fields(self) -> None:
        r = ReflexionRound(round_number=1, response="rev", critique="fix this", improved=True)
        assert r.improved


# ── ReflexionResult ────────────────────────────────────────────────────────────


class TestReflexionResult:
    def test_initial_response(self) -> None:
        rounds = [
            ReflexionRound(0, "initial"),
            ReflexionRound(1, "revised"),
        ]
        r = ReflexionResult(final_response="revised", rounds=rounds)
        assert r.initial_response == "initial"

    def test_initial_response_empty(self) -> None:
        r = ReflexionResult(final_response="x")
        assert r.initial_response == ""

    def test_improved_true(self) -> None:
        rounds = [
            ReflexionRound(0, "old"),
            ReflexionRound(1, "new", critique="fix", improved=True),
        ]
        r = ReflexionResult(final_response="new", rounds=rounds)
        assert r.improved

    def test_improved_false(self) -> None:
        rounds = [ReflexionRound(0, "same")]
        r = ReflexionResult(final_response="same", rounds=rounds)
        assert not r.improved


# ── ReflexionAgent ─────────────────────────────────────────────────────────────


class TestReflexionAgent:
    def _make_agent(
        self,
        initial_response: str = "initial answer",
        critique: str = "needs improvement",
        revision: str = "improved answer",
    ) -> Agent:
        backend = MagicMock()
        # agent.run() calls backend.generate() with the user prompt
        backend.generate.side_effect = [
            initial_response,  # agent.run() → initial generation
            critique,          # _critique()
            revision,          # _revise()
            critique,          # second round critique
            revision,          # second round revision
        ]
        return Agent(Genome(), backend=backend)

    def test_returns_reflexion_result(self) -> None:
        agent = self._make_agent()
        reflexion = ReflexionAgent(agent=agent, n_rounds=1)
        result = reflexion.run("task")
        assert isinstance(result, ReflexionResult)

    def test_rounds_recorded(self) -> None:
        agent = self._make_agent()
        reflexion = ReflexionAgent(agent=agent, n_rounds=2)
        result = reflexion.run("task")
        assert len(result.rounds) >= 1  # at minimum round 0

    def test_final_response_improved(self) -> None:
        agent = self._make_agent(
            initial_response="bad answer",
            critique="This is wrong: fix it",
            revision="better answer",
        )
        reflexion = ReflexionAgent(agent=agent, n_rounds=1)
        result = reflexion.run("task")
        assert result.final_response == "better answer"

    def test_stop_if_excellent(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = [
            "good initial answer",
            "EXCELLENT: No significant issues found.",  # critique → stops early
        ]
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=3, stop_if_excellent=True)
        result = reflexion.run("task")
        # Should stop after 1 critique (excellent), not run 3 rounds
        assert result.n_rounds_used == 1

    def test_zero_rounds(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "original"
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=0)
        result = reflexion.run("task")
        assert result.final_response == "original"
        assert result.n_rounds_used == 1  # round 0 only

    def test_critique_error_stops_gracefully(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = [
            "initial",
            RuntimeError("API down"),  # critique fails
        ]
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=2)
        result = reflexion.run("task")
        assert result.final_response == "initial"

    def test_revision_error_keeps_previous(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = [
            "initial",
            "needs work",          # critique
            RuntimeError("down"),  # revision fails
        ]
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=1)
        result = reflexion.run("task")
        # Should keep original response on revision failure
        assert result.final_response in ("initial", "initial")

    def test_task_stored(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "answer"
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=0)
        result = reflexion.run("my specific task")
        assert result.task == "my specific task"


# ── ReflexionEvaluator ─────────────────────────────────────────────────────────


class TestReflexionEvaluator:
    def test_returns_score(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["initial", "EXCELLENT: fine", "refined"]
        agent = Agent(Genome(), backend=backend)
        ev = ReflexionEvaluator(_ConstEvaluator(0.75), n_rounds=1)
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.75)

    def test_no_backend_falls_through(self) -> None:
        # Agent without backend → base evaluator called directly
        agent = Agent(Genome())  # no backend
        ev = ReflexionEvaluator(_ConstEvaluator(0.55), n_rounds=1)
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.55)

    def test_original_run_restored(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["original", "EXCELLENT: fine"]
        agent = Agent(Genome(), backend=backend)
        original_run = agent.run

        ev = ReflexionEvaluator(_ConstEvaluator(0.6), n_rounds=1)
        ev.evaluate(agent, "task")

        # run method should be restored to original
        assert agent.run == original_run
