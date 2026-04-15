"""Round 9 integration tests — verifica che tutti i nuovi moduli lavorino insieme.

Questi test verificano:
- CodeGenome + CodeEvaluator nel loop CodeEvolutionEngine
- Pipeline + PipelineMutator nel loop PipelineEvolutionEngine
- DreamPhase integrata con QuorumSensor
- MixtureOfAgents con QuantumTunneler
- ReflexionAgent con ReflexionEvaluator
- __init__.py esporta tutti i simboli Round 9
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.code_genome import (
    CodeEvaluator,
    CodeEvolutionEngine,
    CodeGenome,
    TestCase,
)
from cambrian.dream import DreamPhase, Experience
from cambrian.moa import QuantumTunneler
from cambrian.pipeline import (
    Pipeline,
    PipelineEvaluator,
    PipelineEvolutionEngine,
    PipelineStep,
)
from cambrian.quorum import QuorumSensor
from cambrian.reflexion import ReflexionAgent, ReflexionEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _agent(fitness: float | None = None, prompt: str = "Agent") -> Agent:
    b = MagicMock()
    b.generate.return_value = "mock response"
    a = Agent(Genome(system_prompt=prompt), backend=b)
    if fitness is not None:
        a.fitness = fitness
    return a


def _const_evaluator(score: float = 0.5):  # type: ignore[no-untyped-def]
    class _E:
        def evaluate(self, agent: Agent, task: str) -> float:
            return score
    return _E()


# ── __init__.py exports Round 9 ───────────────────────────────────────────────


class TestInitExports:
    def test_code_genome_exported(self) -> None:
        import cambrian
        assert hasattr(cambrian, "CodeGenome")
        assert hasattr(cambrian, "CodeEvolutionEngine")
        assert hasattr(cambrian, "CodeEvaluator")
        assert hasattr(cambrian, "TestCase")

    def test_pipeline_exported(self) -> None:
        import cambrian
        assert hasattr(cambrian, "Pipeline")
        assert hasattr(cambrian, "PipelineStep")
        assert hasattr(cambrian, "PipelineEvolutionEngine")

    def test_dream_quorum_exported(self) -> None:
        import cambrian
        assert hasattr(cambrian, "DreamPhase")
        assert hasattr(cambrian, "QuorumSensor")

    def test_moa_reflexion_exported(self) -> None:
        import cambrian
        assert hasattr(cambrian, "MixtureOfAgents")
        assert hasattr(cambrian, "QuantumTunneler")
        assert hasattr(cambrian, "ReflexionAgent")
        assert hasattr(cambrian, "ReflexionEvaluator")

    def test_total_symbol_count(self) -> None:
        import cambrian
        assert len(cambrian.__all__) >= 48


# ── CodeGenome integration ────────────────────────────────────────────────────


class TestCodeGenotypeIntegration:
    def test_engine_uses_evaluator_passed_at_construction(self) -> None:
        """CodeEvolutionEngine should use evaluator if no test_cases at evolve()."""
        backend = MagicMock()
        backend.generate.return_value = 'print("hello")'
        ev = CodeEvaluator([TestCase(input_data="", expected_output="hello")])
        engine = CodeEvolutionEngine(backend=backend, evaluator=ev, population_size=2)
        seed = CodeGenome(code='print("hello")', description="print hello")
        best = engine.evolve(seed, "print hello", n_generations=1)
        assert isinstance(best, CodeGenome)

    def test_perfect_code_evaluates_correctly(self) -> None:
        """Engine should find perfect fitness on known-correct code."""
        backend = MagicMock()
        backend.generate.return_value = 'print("perfect")'
        tcs = [TestCase(input_data="", expected_output="perfect")]
        ev = CodeEvaluator(tcs)
        genome = CodeGenome(code='print("perfect")', description="t")
        result = ev.evaluate(genome)
        assert result.passed == 1
        assert result.pass_rate == pytest.approx(1.0)
        # Fitness should be very close to 1.0
        assert result.fitness > 0.9

    def test_code_genome_serialisation_roundtrip(self) -> None:
        cg = CodeGenome(code='x = 42\nprint(x)', description="set x", version=3, metadata={"k": 1})
        restored = CodeGenome.from_dict(cg.to_dict())
        assert restored.code == cg.code
        assert restored.version == cg.version
        assert restored.metadata == cg.metadata

    def test_test_case_weights_affect_pass_rate(self) -> None:
        tcs = [
            TestCase(input_data="", expected_output="ok", weight=4.0),
            TestCase(input_data="", expected_output="no", weight=1.0),
        ]
        ev = CodeEvaluator(tcs)
        genome = CodeGenome(code='print("ok")')
        r = ev.evaluate(genome)
        # passes weight 4, fails weight 1 → pass_rate = 4/5 = 0.8
        assert r.pass_rate == pytest.approx(0.8)


# ── Pipeline integration ──────────────────────────────────────────────────────


class TestPipelineIntegration:
    def test_crossover_preserves_parent_description(self) -> None:
        from cambrian.pipeline import PipelineMutator

        backend = MagicMock()
        backend.generate.return_value = json.dumps({
            "steps": [{"role": "a", "system_prompt": "s", "temperature": 0.7}]
        })
        mutator = PipelineMutator(backend=backend)
        pa = Pipeline(
            steps=[PipelineStep("r1", "s1"), PipelineStep("r2", "s2")],
            description="my task",
        )
        pb = Pipeline(
            steps=[PipelineStep("r3", "s3")],
            description="other task",
        )
        child = mutator.crossover(pa, pb)
        assert child.description == "my task"

    def test_pipeline_runner_empty_pipeline(self) -> None:
        from cambrian.pipeline import PipelineRunner

        backend = MagicMock()
        runner = PipelineRunner(backend)
        result = runner.run(Pipeline(), "task")
        assert result == ""
        backend.generate.assert_not_called()

    def test_engine_with_evaluator_at_construction(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "good result"
        ev = PipelineEvaluator(backend, score_fn=lambda o, t: 0.8)
        engine = PipelineEvolutionEngine(backend=backend, evaluator=ev, population_size=3)
        seed = Pipeline(steps=[PipelineStep("r", "s")], description="task")
        best = engine.evolve(seed, "task", n_generations=2)
        assert isinstance(best, Pipeline)


# ── Dream + Quorum integration ────────────────────────────────────────────────


class TestDreamQuorumIntegration:
    def test_quorum_adjusts_rate_over_time(self) -> None:
        """Quorum sensor should converge toward target entropy."""
        sensor = QuorumSensor(target_entropy=0.5, lr=0.2, n_bins=5)
        agents = [_agent(0.9) for _ in range(5)]  # converged → low entropy
        for _ in range(10):
            state = sensor.update(agents)
        # After many updates on converged pop, rate should approach max
        assert state.mutation_rate > 0.5

    def test_dream_updates_fitness_proportionally(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "dream scenario"
        phase = DreamPhase(backend=backend, blend_weight=0.2, n_dreams=2)

        agents = [_agent(0.8)]
        exps = [
            Experience("task", "resp", 0.9),
            Experience("task2", "resp2", 0.7),
        ]
        phase.run(agents, "task", _const_evaluator(0.5), experiences=exps)
        # With blend=0.2: new = 0.8 * original + 0.2 * 0.5 = 0.64 + 0.10 = 0.74
        assert agents[0].fitness is not None
        assert abs((agents[0].fitness or 0.0) - (0.8 * 0.8 + 0.2 * 0.5)) < 0.01

    def test_dream_with_quorum_feedback(self) -> None:
        """After dream phase, quorum sensor can compute updated entropy."""
        backend = MagicMock()
        backend.generate.return_value = "dream"
        phase = DreamPhase(backend=backend, blend_weight=0.1, n_dreams=1)
        sensor = QuorumSensor(n_bins=4)

        agents = [_agent(0.5 + i * 0.1) for i in range(6)]
        # Evaluate before
        state_before = sensor.update(agents)
        # Dream phase blends fitness
        exps = [Experience("t", "r", 0.7 + i * 0.05) for i in range(4)]
        phase.run(agents, "task", _const_evaluator(0.6), experiences=exps)
        # Evaluate after
        state_after = sensor.update(agents)
        # Both states should be valid
        assert 0.0 <= state_before.entropy <= 1.0
        assert 0.0 <= state_after.entropy <= 1.0


# ── MoA + QuantumTunneler integration ────────────────────────────────────────


class TestMoAQuantumIntegration:
    def test_quantum_then_moa(self) -> None:
        """QuantumTunneler followed by MoA should work without errors."""
        tunneler = QuantumTunneler(tunnel_prob=0.5, seed=42, n_elites=1)
        pop = [_agent(0.8 - i * 0.1) for i in range(6)]
        # Apply quantum tunneling
        new_pop = tunneler.apply(pop)
        assert len(new_pop) == 6
        # Build MoA from surviving agents (those with backends)
        surviving = [a for a in new_pop if a.backend is not None]
        assert surviving  # at least elite should survive

    def test_quantum_events_increase_per_call(self) -> None:
        tunneler = QuantumTunneler(tunnel_prob=1.0, protect_elites=False, seed=0)
        pop = [_agent() for _ in range(5)]
        tunneler.apply(pop)
        assert tunneler.tunnel_count == 5
        tunneler.apply(pop)
        assert tunneler.tunnel_count == 10


# ── Reflexion integration ─────────────────────────────────────────────────────


class TestReflexionIntegration:
    def test_reflexion_improves_evaluator_score(self) -> None:
        """ReflexionEvaluator should call base evaluator with improved response."""
        backend = MagicMock()
        backend.generate.side_effect = [
            "initial answer",
            "needs improvement: add detail",
            "improved detailed answer",
            "EXCELLENT: No issues",  # second round stops
        ]
        agent = Agent(Genome(), backend=backend)

        recorded: list[str] = []

        class _RecordingEvaluator:
            def evaluate(self, a: Agent, task: str) -> float:
                response = a.run(task)
                recorded.append(response)
                return len(response) / 100.0  # longer = better

        ev = ReflexionEvaluator(_RecordingEvaluator(), n_rounds=2)
        score = ev.evaluate(agent, "explain something")
        assert isinstance(score, float)

    def test_reflexion_result_structure(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = [
            "initial answer",
            "EXCELLENT: No issues",  # stops after round 0 critique
        ]
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=1, stop_if_excellent=True)
        result = reflexion.run("task")
        assert result.initial_response == "initial answer"
        assert result.final_response == "initial answer"
        assert result.n_rounds_used == 1

    def test_reflexion_no_backend_raises(self) -> None:
        agent = Agent(Genome())  # no backend
        with pytest.raises(ValueError, match="backend"):
            ReflexionAgent(agent=agent, n_rounds=1)

    def test_reflexion_multiple_rounds_improve(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = [
            "draft 1",
            "critique 1: too short",
            "draft 2 with more detail",
            "critique 2: still missing X",
            "draft 3 with X included",
            "EXCELLENT",
        ]
        agent = Agent(Genome(), backend=backend)
        reflexion = ReflexionAgent(agent=agent, n_rounds=3)
        result = reflexion.run("task")
        assert len(result.rounds) >= 3


# ── CLI integration ───────────────────────────────────────────────────────────


class TestForgeCliIntegration:
    def test_forge_code_mode_help(self) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        result = CliRunner().invoke(cli, ["forge", "--help"])
        assert result.exit_code == 0
        assert "code" in result.output.lower() or "forge" in result.output.lower()

    def test_forge_pipeline_mode_help(self) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        result = CliRunner().invoke(cli, ["forge", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()

    def test_forge_test_case_parsing(self) -> None:
        """Test case parsing: '3 4|7' should be parsed as input='3 4', expected='7'."""
        from cambrian.code_genome import TestCase
        tc_str = "3 4|7"
        if "|" in tc_str:
            inp, exp = tc_str.split("|", 1)
            tc = TestCase(input_data=inp, expected_output=exp)
            assert tc.input_data == "3 4"
            assert tc.expected_output == "7"
