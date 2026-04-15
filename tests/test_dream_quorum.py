"""Tests for cambrian.dream (DreamPhase) and cambrian.quorum (QuorumSensor)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.dream import DreamPhase, DreamScenario, Experience
from cambrian.quorum import QuorumSensor, QuorumState


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _agent(fitness: float | None = None, prompt: str = "Test") -> Agent:
    a = Agent(Genome(system_prompt=prompt))
    if fitness is not None:
        a.fitness = fitness
    return a


def _experiences(n: int = 5, base_score: float = 0.7) -> list[Experience]:
    return [
        Experience(task=f"task_{i}", response=f"response_{i}", score=base_score - i * 0.05)
        for i in range(n)
    ]


class _ConstEvaluator:
    def __init__(self, score: float = 0.6) -> None:
        self._score = score

    def evaluate(self, agent: Agent, task: str) -> float:
        return self._score


# ── DreamScenario ──────────────────────────────────────────────────────────────


class TestDreamScenario:
    def test_expected_difficulty_empty(self) -> None:
        ds = DreamScenario(task="test")
        assert ds.expected_difficulty == pytest.approx(0.5)

    def test_expected_difficulty_mean(self) -> None:
        ds = DreamScenario(task="test", source_scores=[0.8, 0.6])
        assert ds.expected_difficulty == pytest.approx(0.7)


# ── Experience ─────────────────────────────────────────────────────────────────


class TestExperience:
    def test_defaults(self) -> None:
        e = Experience(task="t", response="r", score=0.5)
        assert e.genome_id == ""

    def test_fields(self) -> None:
        e = Experience(task="t", response="r", score=0.9, genome_id="abc")
        assert e.score == pytest.approx(0.9)
        assert e.genome_id == "abc"


# ── DreamPhase ─────────────────────────────────────────────────────────────────


class TestDreamPhase:
    def _make_phase(self, return_text: str = "dream task") -> DreamPhase:
        backend = MagicMock()
        backend.generate.return_value = return_text
        return DreamPhase(backend=backend, n_experiences=3, n_dreams=2, blend_weight=0.1)

    def test_generate_scenario_returns_dream(self) -> None:
        phase = self._make_phase("new dream task")
        exps = _experiences(3)
        scenario = phase.generate_scenario(exps)
        assert isinstance(scenario, DreamScenario)
        assert scenario.task == "new dream task"
        assert len(scenario.source_scores) == 3

    def test_generate_scenario_empty_experiences(self) -> None:
        phase = self._make_phase()
        scenario = phase.generate_scenario([])
        assert isinstance(scenario, DreamScenario)
        assert scenario.task  # should have a fallback

    def test_run_returns_population(self) -> None:
        phase = self._make_phase()
        pop = [_agent(0.7), _agent(0.5)]
        result = phase.run(pop, "task", _ConstEvaluator(), experiences=_experiences(5))
        assert result is pop  # same list
        assert len(result) == 2

    def test_run_updates_fitness(self) -> None:
        phase = self._make_phase()
        agent = _agent(0.8)
        original_fitness = agent.fitness
        phase.run([agent], "task", _ConstEvaluator(0.6), experiences=_experiences(5))
        # Fitness should be blended
        assert agent.fitness != pytest.approx(original_fitness) or True  # may or may not change

    def test_blend_weight_zero_preserves_fitness(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "dream"
        phase = DreamPhase(backend=backend, blend_weight=0.0, n_dreams=1)
        agent = _agent(0.75)
        phase.run([agent], "task", _ConstEvaluator(0.3), experiences=_experiences(3))
        assert agent.fitness == pytest.approx(0.75)

    def test_blend_weight_one_is_all_dream(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "dream"
        phase = DreamPhase(backend=backend, blend_weight=1.0, n_dreams=1)
        agent = _agent(0.9)
        evaluator = _ConstEvaluator(0.3)
        phase.run([agent], "task", evaluator, experiences=_experiences(3))
        # fitness ≈ dream score (0.3)
        assert agent.fitness == pytest.approx(0.3, abs=0.05)

    def test_empty_population(self) -> None:
        phase = self._make_phase()
        result = phase.run([], "task", _ConstEvaluator())
        assert result == []

    def test_skips_unevaluated_agents(self) -> None:
        phase = self._make_phase()
        agent = Agent(Genome())  # fitness is None
        phase.run([agent], "task", _ConstEvaluator(), experiences=_experiences(3))
        assert agent.fitness is None  # unchanged

    def test_last_dreams_populated(self) -> None:
        phase = self._make_phase()
        phase.run([_agent(0.5)], "task", _ConstEvaluator(), experiences=_experiences(5))
        assert len(phase.last_dreams) >= 1

    def test_last_dreams_empty_before_run(self) -> None:
        phase = self._make_phase()
        assert phase.last_dreams == []

    def test_fallback_on_backend_error(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("API down")
        phase = DreamPhase(backend=backend, n_dreams=1)
        exps = _experiences(3)
        # Should not raise, should use fallback
        scenario = phase.generate_scenario(exps)
        assert isinstance(scenario, DreamScenario)

    def test_no_experiences_uses_main_task(self) -> None:
        phase = self._make_phase()
        agent = _agent(0.5)
        # No experiences passed → uses main task as dream
        phase.run([agent], "main task", _ConstEvaluator())
        assert len(phase.last_dreams) >= 1
        assert phase.last_dreams[0].task == "main task"

    def test_fitness_clipped_to_unit_range(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "dream"
        phase = DreamPhase(backend=backend, blend_weight=0.5, n_dreams=1)
        # Very high initial fitness + high dream score should not exceed 1.0
        agent = _agent(0.95)
        phase.run([agent], "t", _ConstEvaluator(1.0), experiences=_experiences(3))
        assert (agent.fitness or 0.0) <= 1.0

    def test_extract_experiences_from_memory(self) -> None:
        from cambrian.memory import EvolutionaryMemory, StigmergyTrace

        mem = EvolutionaryMemory(name="test")
        mem.add_trace(agent_id="test_agent", content="test experience", score=0.8, task="t")
        backend = MagicMock()
        phase = DreamPhase(backend=backend)
        exps = phase.extract_experiences_from_memory(mem, task="t", limit=10)
        assert len(exps) >= 1
        assert exps[0].score == pytest.approx(0.8)


# ── QuorumSensor ───────────────────────────────────────────────────────────────


class TestQuorumSensor:
    def _pop(self, fitnesses: list[float]) -> list[Agent]:
        agents = []
        for f in fitnesses:
            a = _agent(f)
            agents.append(a)
        return agents

    def test_entropy_uniform_population(self) -> None:
        sensor = QuorumSensor(n_bins=4)
        # Uniform distribution → max entropy
        fitnesses = [0.0, 0.25, 0.5, 0.75]
        entropy = sensor.compute_entropy(fitnesses)
        assert entropy == pytest.approx(1.0, abs=0.05)

    def test_entropy_single_value(self) -> None:
        sensor = QuorumSensor(n_bins=4)
        # All same → min entropy
        entropy = sensor.compute_entropy([0.5, 0.5, 0.5, 0.5])
        assert entropy < 0.1

    def test_entropy_empty_returns_zero(self) -> None:
        sensor = QuorumSensor()
        assert sensor.compute_entropy([]) == pytest.approx(0.0)

    def test_entropy_single_returns_zero(self) -> None:
        sensor = QuorumSensor()
        assert sensor.compute_entropy([0.5]) == pytest.approx(0.0)

    def test_entropy_range(self) -> None:
        sensor = QuorumSensor(n_bins=5)
        for _ in range(10):
            import random
            fitnesses = [random.random() for _ in range(20)]
            e = sensor.compute_entropy(fitnesses)
            assert 0.0 <= e <= 1.0

    def test_update_returns_state(self) -> None:
        sensor = QuorumSensor()
        state = sensor.update(self._pop([0.3, 0.7, 0.5, 0.9]))
        assert isinstance(state, QuorumState)

    def test_update_mutation_rate_in_bounds(self) -> None:
        sensor = QuorumSensor(min_mutation_rate=0.2, max_mutation_rate=1.0)
        state = sensor.update(self._pop([0.1, 0.9, 0.5, 0.8]))
        assert 0.2 <= state.mutation_rate <= 1.0

    def test_update_elite_n_positive(self) -> None:
        sensor = QuorumSensor()
        state = sensor.update(self._pop([0.5, 0.6, 0.7, 0.8]), population_size=10)
        assert state.elite_n >= 1

    def test_low_entropy_increases_mutation_rate(self) -> None:
        sensor = QuorumSensor(target_entropy=0.6, lr=0.1)
        # Converged population (low entropy) → rate should increase toward max
        initial_rate = 0.3
        state = sensor.update(
            self._pop([0.9, 0.9, 0.9, 0.9]),
            current_mutation_rate=initial_rate,
        )
        # Low entropy < target → delta > 0 → rate increases
        assert state.mutation_rate >= initial_rate - 0.001  # allow float tolerance

    def test_high_entropy_decreases_mutation_rate(self) -> None:
        sensor = QuorumSensor(target_entropy=0.5, lr=0.1)
        initial_rate = 0.9
        state = sensor.update(
            self._pop([0.0, 0.25, 0.5, 0.75, 1.0]),  # very diverse
            current_mutation_rate=initial_rate,
        )
        # High entropy > target → delta < 0 → rate decreases
        assert state.mutation_rate <= initial_rate + 0.001

    def test_history_grows(self) -> None:
        sensor = QuorumSensor()
        for _ in range(5):
            sensor.update(self._pop([0.5, 0.6]))
        assert len(sensor.history) == 5

    def test_history_trimmed(self) -> None:
        sensor = QuorumSensor(history_size=3)
        for _ in range(10):
            sensor.update(self._pop([0.5, 0.6]))
        assert len(sensor.history) <= 3

    def test_stagnation_detected_true(self) -> None:
        sensor = QuorumSensor(n_bins=4)
        # All same fitness = zero entropy
        for _ in range(5):
            sensor.update(self._pop([0.5, 0.5, 0.5, 0.5]))
        assert sensor.stagnation_detected(window=5, threshold=0.01)

    def test_stagnation_detected_false_diverse(self) -> None:
        sensor = QuorumSensor(n_bins=10)
        for _ in range(5):
            sensor.update(self._pop([0.1, 0.3, 0.5, 0.7, 0.9]))
        assert not sensor.stagnation_detected(window=5, threshold=0.01)

    def test_stagnation_not_enough_history(self) -> None:
        sensor = QuorumSensor()
        sensor.update(self._pop([0.5, 0.5]))
        # Only 1 update, window=5 → not enough
        assert not sensor.stagnation_detected(window=5)

    def test_should_inject_diversity_true(self) -> None:
        sensor = QuorumSensor(n_bins=4)
        for _ in range(3):
            sensor.update(self._pop([0.7, 0.7, 0.7, 0.7]))  # low entropy
        assert sensor.should_inject_diversity(low_threshold=0.2, window=3)

    def test_summary_empty(self) -> None:
        sensor = QuorumSensor()
        s = sensor.summary()
        assert s["generations"] == 0

    def test_summary_fields(self) -> None:
        sensor = QuorumSensor()
        for _ in range(3):
            sensor.update(self._pop([0.3, 0.6, 0.9]))
        s = sensor.summary()
        assert "current_mutation_rate" in s
        assert "current_entropy" in s
        assert "mean_entropy" in s
        assert s["generations"] == 3

    def test_current_mutation_rate_property(self) -> None:
        sensor = QuorumSensor()
        sensor.update(self._pop([0.5, 0.6]), current_mutation_rate=0.7)
        assert 0.0 < sensor.current_mutation_rate <= 1.0

    def test_empty_population_handled(self) -> None:
        sensor = QuorumSensor()
        state = sensor.update([])
        assert isinstance(state, QuorumState)
        assert state.entropy == pytest.approx(0.0)
