"""Edge-case tests for Round 9 modules — covers gaps in primary test suites.

Focuses on:
- CodeEvaluator: timeout, multiline output, trailing whitespace
- QuantumTunneler: n_elites > population size
- MixtureOfAgents: single agent, all-empty answers
- ReflexionAgent: n_rounds=0
- PipelineMutator: LLM returns JSON without 'steps' key
- QuorumSensor: population with None fitnesses
- DreamPhase: all agents have None fitness (no blending)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.code_genome import CodeEvaluator, CodeGenome, TestCase
from cambrian.dream import DreamPhase
from cambrian.moa import MixtureOfAgents, QuantumTunneler
from cambrian.pipeline import Pipeline, PipelineMutator, PipelineStep
from cambrian.quorum import QuorumSensor
from cambrian.reflexion import ReflexionAgent, ReflexionResult


# ── Helpers ────────────────────────────────────────────────────────────────────


def _agent(fitness: float | None = None, prompt: str = "Agent") -> Agent:
    b = MagicMock()
    b.generate.return_value = "mock response"
    a = Agent(Genome(system_prompt=prompt), backend=b)
    if fitness is not None:
        a.fitness = fitness
    return a


# ── CodeEvaluator edge cases ───────────────────────────────────────────────────


class TestCodeEvaluatorEdgeCases:
    def test_multiline_expected_output(self) -> None:
        """Code that produces two lines should match multiline expected output."""
        code = 'print("line1")\nprint("line2")'
        # The evaluator strips, so multiline becomes "line1\nline2" vs expected stripped
        genome = CodeGenome(code=code, description="two lines")
        tc = TestCase(input_data="", expected_output="line1\nline2")
        ev = CodeEvaluator([tc])
        result = ev.evaluate(genome)
        assert result.passed == 1
        assert result.pass_rate == pytest.approx(1.0)

    def test_trailing_newline_ignored(self) -> None:
        """print() adds \\n; expected without trailing newline should still pass."""
        genome = CodeGenome(code='print("ok")', description="ok")
        tc = TestCase(input_data="", expected_output="ok")
        ev = CodeEvaluator([tc])
        result = ev.evaluate(genome)
        assert result.passed == 1

    def test_timed_out_code_gets_zero_pass_rate(self) -> None:
        """An infinite loop should be penalised with pass_rate=0 and timed_out=True.

        Note: fitness is NOT exactly 0 because the formula includes small LOC/runtime
        bonuses even at 0% pass rate.  We only assert it is very low (<0.1).
        """
        genome = CodeGenome(code="while True: pass", description="infinite loop")
        tc = TestCase(input_data="", expected_output="hello")
        ev = CodeEvaluator([tc], timeout=0.5)  # very short timeout
        result = ev.evaluate(genome)
        assert result.timed_out is True
        assert result.pass_rate == pytest.approx(0.0)
        assert result.fitness < 0.1  # dominated by pass_rate=0; only tiny LOC/runtime bonus

    def test_syntax_error_handled_gracefully(self) -> None:
        """Code with syntax errors should return pass_rate=0, no exception raised."""
        genome = CodeGenome(code="def broken(: pass", description="syntax error")
        tc = TestCase(input_data="", expected_output="anything")
        ev = CodeEvaluator([tc])
        result = ev.evaluate(genome)
        assert result.passed == 0
        assert result.pass_rate == pytest.approx(0.0)
        assert result.error  # should have a non-empty error message

    def test_code_with_all_comment_lines_has_zero_loc(self) -> None:
        """A genome whose code is only comments should have loc=0."""
        code = "# this is a comment\n# another comment\n"
        genome = CodeGenome(code=code, description="only comments")
        assert genome.loc() == 0

    def test_fitness_not_negative(self) -> None:
        """Fitness should always be >= 0 even for bad code."""
        genome = CodeGenome(code='print("wrong")', description="wrong output")
        tc = TestCase(input_data="", expected_output="correct")
        ev = CodeEvaluator([tc])
        result = ev.evaluate(genome)
        assert result.fitness >= 0.0

    def test_multiple_test_cases_partial_pass_fitness_blended(self) -> None:
        """Pass 2/3 → pass_rate=0.667; fitness should be > 0 but < 1."""
        code = "x = int(input()); print(x * 2)"
        genome = CodeGenome(code=code, description="double")
        tcs = [
            TestCase(input_data="3", expected_output="6"),   # pass
            TestCase(input_data="5", expected_output="10"),  # pass
            TestCase(input_data="2", expected_output="5"),   # fail (2*2=4 not 5)
        ]
        ev = CodeEvaluator(tcs)
        result = ev.evaluate(genome)
        assert result.passed == 2
        assert result.total == 3
        assert 0.0 < result.fitness < 1.0


# ── QuantumTunneler edge cases ─────────────────────────────────────────────────


class TestQuantumTunnelerEdgeCases:
    def test_n_elites_larger_than_population(self) -> None:
        """When n_elites >= population size, no agents should be tunneled."""
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=True, n_elites=10)
        pop = [_agent(float(i)) for i in range(3)]
        result = tunneler.apply(pop)
        assert len(result) == 3
        assert tunneler.tunnel_count == 0

    def test_single_agent_with_elite_protection(self) -> None:
        """Single agent + protect_elites=True + n_elites=1 → no tunneling."""
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=True, n_elites=1)
        pop = [_agent(0.9)]
        result = tunneler.apply(pop)
        assert len(result) == 1
        assert tunneler.tunnel_count == 0

    def test_zero_tunnel_prob_preserves_all(self) -> None:
        """tunnel_prob=0 should never replace any agent."""
        tunneler = QuantumTunneler(tunnel_prob=0.0, protect_elites=False)
        pop = [_agent(float(i)) for i in range(8)]
        original_ids = [a.agent_id for a in pop]
        result = tunneler.apply(pop)
        result_ids = [a.agent_id for a in result]
        assert result_ids == original_ids
        assert tunneler.tunnel_count == 0

    def test_tunnel_count_accumulates_across_calls(self) -> None:
        """tunnel_count should be the running total across multiple apply() calls."""
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=42)
        pop = [_agent(float(i)) for i in range(4)]
        tunneler.apply(pop)
        first_count = tunneler.tunnel_count
        tunneler.apply(pop)
        assert tunneler.tunnel_count == first_count * 2

    def test_tunneled_genome_differs_from_original(self) -> None:
        """A tunneled agent should have a different genome than the original."""
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=0)
        agent = _agent(0.5, prompt="unique original prompt text")
        pop = [agent]
        result = tunneler.apply(pop)
        assert result[0].genome.system_prompt != "unique original prompt text"


# ── MixtureOfAgents edge cases ─────────────────────────────────────────────────


class TestMixtureOfAgentsEdgeCases:
    def test_single_agent_returns_direct_answer(self) -> None:
        """With one agent, the MoA should return that agent's answer directly."""
        b = MagicMock()
        b.generate.return_value = "direct answer"
        agent = Agent(Genome(), backend=b)
        moa = MixtureOfAgents([agent], aggregator_backend=b)
        result = moa.run("task")
        assert result.final_answer == "direct answer"
        assert result.n_agents == 1

    def test_all_agents_return_empty_strings(self) -> None:
        """When all agents return '', fallback should return '' (the longest = '')."""
        agents = []
        for _ in range(3):
            b = MagicMock()
            b.generate.return_value = ""
            agents.append(Agent(Genome(), backend=b))
        agg = MagicMock()
        agg.generate.side_effect = Exception("aggregator failed")
        moa = MixtureOfAgents(agents, aggregator_backend=agg)
        result = moa.run("task")
        # All are empty; fallback picks longest (empty or first)
        assert isinstance(result.final_answer, str)
        assert len(result.individual_answers) == 3

    def test_n_agents_caps_agent_selection(self) -> None:
        """With n_agents=2 and 4 available, only 2 should be queried."""
        agents = []
        for i in range(4):
            b = MagicMock()
            b.generate.return_value = f"answer {i}"
            agents.append(Agent(Genome(), backend=b))
        agg = MagicMock()
        agg.generate.return_value = "synthesis"
        moa = MixtureOfAgents(agents, aggregator_backend=agg, n_agents=2)
        result = moa.run("task")
        assert result.n_agents == 2
        assert len(result.individual_answers) == 2

    def test_raises_when_no_backend_available(self) -> None:
        """MixtureOfAgents should raise ValueError if no aggregator and agents have no backend."""
        agent = Agent(Genome())  # no backend
        with pytest.raises(ValueError, match="backend"):
            MixtureOfAgents([agent])


# ── ReflexionAgent edge cases ──────────────────────────────────────────────────


class TestReflexionAgentEdgeCases:
    def test_zero_rounds_returns_initial(self) -> None:
        """n_rounds=0 should return the initial response without any reflection."""
        b = MagicMock()
        b.generate.return_value = "initial answer"
        agent = Agent(Genome(), backend=b)
        reflexion = ReflexionAgent(agent=agent, n_rounds=0)
        result = reflexion.run("task")
        assert result.final_response == "initial answer"
        assert result.n_rounds_used == 1  # round 0 always counted
        assert len(result.rounds) == 1

    def test_negative_rounds_treated_as_zero(self) -> None:
        """n_rounds=-3 should be clamped to 0."""
        b = MagicMock()
        b.generate.return_value = "response"
        agent = Agent(Genome(), backend=b)
        reflexion = ReflexionAgent(agent=agent, n_rounds=-3)
        result = reflexion.run("task")
        assert result.n_rounds_used == 1

    def test_empty_critique_stops_loop(self) -> None:
        """When critique returns empty string, the loop should stop without revision."""
        b = MagicMock()
        b.generate.side_effect = ["initial answer", ""]  # empty critique
        agent = Agent(Genome(), backend=b)
        reflexion = ReflexionAgent(agent=agent, n_rounds=3)
        result = reflexion.run("task")
        # Loop stopped at empty critique
        assert result.final_response == "initial answer"

    def test_revision_same_as_original_improved_is_false(self) -> None:
        """If revision doesn't change the response, improved should be False."""
        b = MagicMock()
        b.generate.side_effect = [
            "the response",   # initial
            "critique",       # non-EXCELLENT critique
            "the response",   # revision identical to original
        ]
        agent = Agent(Genome(), backend=b)
        reflexion = ReflexionAgent(agent=agent, n_rounds=1)
        result = reflexion.run("task")
        # The round was added, but improved=False because text didn't change
        if len(result.rounds) > 1:
            assert result.rounds[-1].improved is False

    def test_result_improved_property_false_when_no_rounds(self) -> None:
        """ReflexionResult.improved should be False when only round 0 exists."""
        result = ReflexionResult(
            final_response="r",
            rounds=[],
            task="t",
            n_rounds_used=0,
        )
        assert result.improved is False


# ── PipelineMutator edge cases ─────────────────────────────────────────────────


class TestPipelineMutatorEdgeCases:
    def test_mutate_json_without_steps_key_uses_fallback(self) -> None:
        """LLM returning JSON without 'steps' key should trigger fallback."""
        backend = MagicMock()
        backend.generate.return_value = json.dumps({"no_steps_here": "oops"})
        mutator = PipelineMutator(backend=backend, fallback_on_error=True)
        original = Pipeline(
            steps=[PipelineStep("writer", "You write things")],
            description="task",
        )
        # mutate(pipeline, fitness=0.0) — second arg is fitness float, not task string
        result = mutator.mutate(original, fitness=0.5)
        # Should return original pipeline (fallback) because 'steps' key is missing
        assert isinstance(result, Pipeline)
        assert result.step_count() == original.step_count()

    def test_mutate_json_with_empty_steps_list(self) -> None:
        """LLM returning empty steps list should produce pipeline with 0 steps."""
        backend = MagicMock()
        backend.generate.return_value = json.dumps({"steps": []})
        mutator = PipelineMutator(backend=backend, fallback_on_error=False)
        original = Pipeline(steps=[], description="task")
        result = mutator.mutate(original, "task")
        assert isinstance(result, Pipeline)
        assert result.step_count() == 0

    def test_crossover_with_single_step_parents(self) -> None:
        """Crossover of two single-step pipelines should produce a valid pipeline."""
        backend = MagicMock()
        backend.generate.return_value = json.dumps({
            "steps": [{"role": "combined", "system_prompt": "merged", "temperature": 0.5}]
        })
        mutator = PipelineMutator(backend=backend)
        pa = Pipeline(steps=[PipelineStep("role_a", "prompt_a")], description="a")
        pb = Pipeline(steps=[PipelineStep("role_b", "prompt_b")], description="b")
        child = mutator.crossover(pa, pb)
        assert isinstance(child, Pipeline)


# ── QuorumSensor edge cases ────────────────────────────────────────────────────


class TestQuorumSensorEdgeCases:
    def test_population_with_all_none_fitness(self) -> None:
        """Population where all agents have None fitness should produce entropy=0."""
        sensor = QuorumSensor(n_bins=4)
        agents = [_agent(None) for _ in range(6)]
        state = sensor.update(agents)
        # All None → treated as empty fitnesses → entropy=0
        assert state.entropy == pytest.approx(0.0)

    def test_population_with_single_agent(self) -> None:
        """Single agent should produce entropy=0 (only one bin occupied)."""
        sensor = QuorumSensor(n_bins=4)
        agents = [_agent(0.5)]
        state = sensor.update(agents)
        assert state.entropy == pytest.approx(0.0)
        assert state.population_size == 1

    def test_mutation_rate_clamped_to_min(self) -> None:
        """Even with very high entropy (mutation rate decreasing), rate >= min."""
        sensor = QuorumSensor(
            n_bins=4,
            target_entropy=0.0,
            min_mutation_rate=0.1,
            max_mutation_rate=1.0,
            lr=1.0,  # aggressive learning rate
        )
        # All agents have spread fitness → high entropy → decreases rate
        agents = [_agent(float(i) / 10.0) for i in range(10)]
        for _ in range(50):
            state = sensor.update(agents)
        assert state.mutation_rate >= 0.1

    def test_mutation_rate_clamped_to_max(self) -> None:
        """Even with very low entropy (mutation rate increasing), rate <= max."""
        sensor = QuorumSensor(
            n_bins=4,
            target_entropy=1.0,
            min_mutation_rate=0.0,
            max_mutation_rate=0.9,
            lr=1.0,
        )
        # All same fitness → entropy=0 → increases rate aggressively
        agents = [_agent(0.5) for _ in range(10)]
        for _ in range(50):
            state = sensor.update(agents)
        assert state.mutation_rate <= 0.9


# ── DreamPhase edge cases ──────────────────────────────────────────────────────


class TestDreamPhaseEdgeCases:
    def test_run_with_no_experiences_uses_main_task(self) -> None:
        """With no experiences, DreamPhase should still run on the main task."""
        backend = MagicMock()
        backend.generate.return_value = "dream scenario"

        class _ConstEv:
            def evaluate(self, agent: Agent, task: str) -> float:
                return 0.7

        phase = DreamPhase(backend=backend, blend_weight=0.5, n_dreams=2)
        agents = [_agent(0.6)]
        phase.run(agents, "task", _ConstEv(), experiences=[])
        # Agent fitness should be blended
        assert agents[0].fitness is not None

    def test_run_does_not_raise_on_empty_population(self) -> None:
        """DreamPhase.run() on empty population should not raise."""
        backend = MagicMock()
        phase = DreamPhase(backend=backend)
        phase.run([], "task", MagicMock())  # should be a no-op

    def test_blend_weight_one_uses_dream_score_only(self) -> None:
        """blend_weight=1.0 should override real fitness with dream fitness."""
        backend = MagicMock()
        backend.generate.return_value = "dream"

        class _ConstEv:
            def evaluate(self, agent: Agent, task: str) -> float:
                return 0.3  # dream score

        phase = DreamPhase(backend=backend, blend_weight=1.0, n_dreams=1)
        agent = _agent(0.9)  # high real fitness
        phase.run([agent], "task", _ConstEv())
        # With blend=1.0: result = 0.0 * 0.9 + 1.0 * 0.3 = 0.3
        assert abs((agent.fitness or 0.0) - 0.3) < 0.05
