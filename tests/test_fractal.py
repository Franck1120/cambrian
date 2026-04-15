# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.fractal — FractalScale, ScaleConfig, FractalResult,
FractalMutator, FractalPopulation, FractalEvolution.

All LLM/backend calls are mocked; no live network requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.evaluator import Evaluator
from cambrian.fractal import (
    FractalEvolution,
    FractalMutator,
    FractalPopulation,
    FractalResult,
    FractalScale,
    ScaleConfig,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _mock_backend(return_value: str = "improved prompt") -> MagicMock:
    """Return a MagicMock that satisfies the LLMBackend interface."""
    backend = MagicMock(spec=LLMBackend)
    backend.generate.return_value = return_value
    backend.model_name = "mock-model"
    return backend


def _mock_evaluator(score: float = 0.5) -> MagicMock:
    """Return a MagicMock that satisfies the Evaluator interface."""
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate.return_value = score
    return evaluator


def _simple_genome(prompt: str = "You are helpful.") -> Genome:
    return Genome(system_prompt=prompt, temperature=0.7, strategy="step-by-step", model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# FractalScale enum
# ---------------------------------------------------------------------------


class TestFractalScaleEnum:
    def test_macro_value(self) -> None:
        assert FractalScale.MACRO == 0

    def test_meso_value(self) -> None:
        assert FractalScale.MESO == 1

    def test_micro_value(self) -> None:
        assert FractalScale.MICRO == 2

    def test_is_int(self) -> None:
        assert isinstance(FractalScale.MACRO, int)

    def test_three_members(self) -> None:
        assert len(FractalScale) == 3


# ---------------------------------------------------------------------------
# ScaleConfig defaults
# ---------------------------------------------------------------------------


class TestScaleConfigDefaults:
    def test_macro_temperature(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO)
        assert cfg.mutation_temperature == pytest.approx(0.7)

    def test_meso_temperature(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MESO)
        assert cfg.mutation_temperature == pytest.approx(0.5)

    def test_micro_temperature(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MICRO)
        assert cfg.mutation_temperature == pytest.approx(0.3)

    def test_macro_fragment_size(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO)
        assert cfg.fragment_size == 500

    def test_meso_fragment_size(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MESO)
        assert cfg.fragment_size == 100

    def test_micro_fragment_size(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MICRO)
        assert cfg.fragment_size == 20

    def test_default_population_size(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO)
        assert cfg.population_size == 4

    def test_default_n_generations(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO)
        assert cfg.n_generations == 3


# ---------------------------------------------------------------------------
# FractalResult
# ---------------------------------------------------------------------------


class TestFractalResult:
    def test_fields_stored(self) -> None:
        genome = _simple_genome()
        result = FractalResult(scale=FractalScale.MACRO, best_genome=genome, best_fitness=0.8, n_evaluations=12)
        assert result.scale is FractalScale.MACRO
        assert result.best_genome is genome
        assert result.best_fitness == pytest.approx(0.8)
        assert result.n_evaluations == 12

    def test_negative_fitness_clamped_to_zero(self) -> None:
        genome = _simple_genome()
        result = FractalResult(scale=FractalScale.MICRO, best_genome=genome, best_fitness=-0.5, n_evaluations=1)
        assert result.best_fitness >= 0.0

    def test_zero_fitness_accepted(self) -> None:
        genome = _simple_genome()
        result = FractalResult(scale=FractalScale.MESO, best_genome=genome, best_fitness=0.0, n_evaluations=4)
        assert result.best_fitness == 0.0

    def test_n_evaluations_stored(self) -> None:
        genome = _simple_genome()
        result = FractalResult(scale=FractalScale.MICRO, best_genome=genome, best_fitness=0.5, n_evaluations=99)
        assert result.n_evaluations == 99


# ---------------------------------------------------------------------------
# FractalMutator
# ---------------------------------------------------------------------------


class TestFractalMutatorInit:
    def test_init_stores_backend(self) -> None:
        backend = _mock_backend()
        mutator = FractalMutator(backend)
        assert mutator._backend is backend


class TestFractalMutatorMacro:
    def test_returns_genome(self) -> None:
        backend = _mock_backend("New improved prompt")
        mutator = FractalMutator(backend)
        result = mutator.mutate_macro(_simple_genome(), "solve maths")
        assert isinstance(result, Genome)

    def test_calls_backend(self) -> None:
        backend = _mock_backend("Better prompt")
        mutator = FractalMutator(backend)
        mutator.mutate_macro(_simple_genome(), "task")
        backend.generate.assert_called_once()

    def test_fallback_on_error(self) -> None:
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("boom")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_macro(genome, "task")
        assert result is genome

    def test_prompt_updated(self) -> None:
        backend = _mock_backend("Totally new system prompt")
        mutator = FractalMutator(backend)
        result = mutator.mutate_macro(_simple_genome("Old prompt"), "task")
        assert result.system_prompt == "Totally new system prompt"

    def test_model_preserved(self) -> None:
        backend = _mock_backend("New prompt")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_macro(genome, "task")
        assert result.model == genome.model


class TestFractalMutatorMeso:
    def test_returns_genome(self) -> None:
        backend = _mock_backend("New paragraph")
        mutator = FractalMutator(backend)
        genome = _simple_genome("Para one.\n\nPara two.\n\nPara three.")
        result = mutator.mutate_meso(genome, "task")
        assert isinstance(result, Genome)

    def test_fallback_on_error(self) -> None:
        backend = _mock_backend()
        backend.generate.side_effect = ValueError("fail")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_meso(genome, "task")
        assert result is genome

    def test_handles_single_line_genome(self) -> None:
        """No \\n\\n separator — entire prompt treated as one chunk."""
        backend = _mock_backend("Rewritten single line")
        mutator = FractalMutator(backend)
        genome = _simple_genome("Single line without double newlines")
        result = mutator.mutate_meso(genome, "task")
        assert isinstance(result, Genome)
        assert result.system_prompt == "Rewritten single line"

    def test_model_preserved(self) -> None:
        backend = _mock_backend("Para")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_meso(genome, "task")
        assert result.model == genome.model


class TestFractalMutatorMicro:
    def test_returns_genome(self) -> None:
        backend = _mock_backend("helpful")
        mutator = FractalMutator(backend)
        result = mutator.mutate_micro(_simple_genome(), "task")
        assert isinstance(result, Genome)

    def test_fallback_on_error(self) -> None:
        backend = _mock_backend()
        backend.generate.side_effect = Exception("fail")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_micro(genome, "task")
        assert result is genome

    def test_handles_very_short_genome(self) -> None:
        """Genome with a single short word — should not crash."""
        backend = _mock_backend("ok")
        mutator = FractalMutator(backend)
        genome = _simple_genome("Hi")
        result = mutator.mutate_micro(genome, "task")
        assert isinstance(result, Genome)

    def test_model_preserved(self) -> None:
        backend = _mock_backend("word")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        result = mutator.mutate_micro(genome, "task")
        assert result.model == genome.model


class TestFractalMutatorDispatch:
    def test_dispatches_macro(self) -> None:
        mutator = FractalMutator(_mock_backend("m"))
        genome = _simple_genome()
        with patch.object(mutator, "mutate_macro", return_value=genome) as mock_macro:
            mutator.mutate(genome, "task", FractalScale.MACRO)
            mock_macro.assert_called_once_with(genome, "task")

    def test_dispatches_meso(self) -> None:
        mutator = FractalMutator(_mock_backend("m"))
        genome = _simple_genome()
        with patch.object(mutator, "mutate_meso", return_value=genome) as mock_meso:
            mutator.mutate(genome, "task", FractalScale.MESO)
            mock_meso.assert_called_once_with(genome, "task")

    def test_dispatches_micro(self) -> None:
        mutator = FractalMutator(_mock_backend("m"))
        genome = _simple_genome()
        with patch.object(mutator, "mutate_micro", return_value=genome) as mock_micro:
            mutator.mutate(genome, "task", FractalScale.MICRO)
            mock_micro.assert_called_once_with(genome, "task")

    def test_mutate_returns_genome_instance(self) -> None:
        backend = _mock_backend("out")
        mutator = FractalMutator(backend)
        genome = _simple_genome()
        for scale in FractalScale:
            result = mutator.mutate(genome, "task", scale)
            assert isinstance(result, Genome), f"scale={scale}"


# ---------------------------------------------------------------------------
# FractalPopulation
# ---------------------------------------------------------------------------


class TestFractalPopulationSeed:
    def test_best_agent_none_before_seed(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=4)
        pop = FractalPopulation(
            scale=FractalScale.MACRO,
            config=cfg,
            evaluator=_mock_evaluator(),
            mutator=FractalMutator(_mock_backend()),
        )
        assert pop.best_agent() is None

    def test_seed_creates_agents(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=4)
        pop = FractalPopulation(
            scale=FractalScale.MACRO,
            config=cfg,
            evaluator=_mock_evaluator(),
            mutator=FractalMutator(_mock_backend()),
        )
        pop.seed(_simple_genome())
        assert len(pop._agents) == 4

    def test_seed_population_size_respected(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2)
        pop = FractalPopulation(
            scale=FractalScale.MESO,
            config=cfg,
            evaluator=_mock_evaluator(),
            mutator=FractalMutator(_mock_backend()),
        )
        pop.seed(_simple_genome())
        assert len(pop._agents) == 2

    def test_agents_are_agent_instances(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=3)
        pop = FractalPopulation(
            scale=FractalScale.MICRO,
            config=cfg,
            evaluator=_mock_evaluator(),
            mutator=FractalMutator(_mock_backend()),
        )
        pop.seed(_simple_genome())
        for agent in pop._agents:
            assert isinstance(agent, Agent)


class TestFractalPopulationEvolveStep:
    def _make_pop(self, pop_size: int = 4, score: float = 0.6) -> FractalPopulation:
        cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=pop_size)
        evaluator = _mock_evaluator(score)
        mutator = FractalMutator(_mock_backend("new prompt"))
        pop = FractalPopulation(
            scale=FractalScale.MACRO,
            config=cfg,
            evaluator=evaluator,
            mutator=mutator,
        )
        pop.seed(_simple_genome())
        return pop

    def test_returns_agent(self) -> None:
        pop = self._make_pop()
        best = pop.evolve_step("task")
        assert isinstance(best, Agent)

    def test_population_size_constant(self) -> None:
        pop = self._make_pop(pop_size=4)
        pop.evolve_step("task")
        assert len(pop._agents) == 4

    def test_population_size_2_works(self) -> None:
        pop = self._make_pop(pop_size=2)
        best = pop.evolve_step("task")
        assert isinstance(best, Agent)
        assert len(pop._agents) == 2

    def test_evaluator_called(self) -> None:
        cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=4)
        evaluator = _mock_evaluator(0.7)
        mutator = FractalMutator(_mock_backend("x"))
        pop = FractalPopulation(
            scale=FractalScale.MACRO,
            config=cfg,
            evaluator=evaluator,
            mutator=mutator,
        )
        pop.seed(_simple_genome())
        pop.evolve_step("task")
        assert evaluator.evaluate.call_count >= 4

    def test_higher_score_wins(self) -> None:
        """Agent scored 1.0 should be best after evolve_step."""
        cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2)
        scores = iter([1.0, 0.0, 1.0, 0.0])
        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate.side_effect = lambda *_: next(scores, 0.5)
        mutator = FractalMutator(_mock_backend("p"))
        pop = FractalPopulation(
            scale=FractalScale.MACRO,
            config=cfg,
            evaluator=evaluator,
            mutator=mutator,
        )
        pop.seed(_simple_genome())
        best = pop.evolve_step("task")
        assert best.fitness == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FractalEvolution
# ---------------------------------------------------------------------------


class TestFractalEvolutionInit:
    def test_results_initially_empty(self) -> None:
        fe = FractalEvolution(_mock_backend(), _mock_evaluator())
        assert fe.results == []

    def test_default_configs_created(self) -> None:
        fe = FractalEvolution(_mock_backend(), _mock_evaluator())
        # Three scales must be configured.
        assert FractalScale.MACRO in fe._configs
        assert FractalScale.MESO in fe._configs
        assert FractalScale.MICRO in fe._configs

    def test_custom_macro_config(self) -> None:
        custom = ScaleConfig(scale=FractalScale.MACRO, population_size=8)
        fe = FractalEvolution(_mock_backend(), _mock_evaluator(), macro_config=custom)
        assert fe._configs[FractalScale.MACRO].population_size == 8


class TestFractalEvolutionEvolve:
    def _make_fe(self, score: float = 0.6) -> FractalEvolution:
        backend = _mock_backend("improved system prompt text")
        evaluator = _mock_evaluator(score)
        macro_cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2, n_generations=1)
        meso_cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        micro_cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=2, n_generations=1)
        return FractalEvolution(backend, evaluator, macro_cfg, meso_cfg, micro_cfg)

    def test_returns_fractal_result(self) -> None:
        fe = self._make_fe()
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert isinstance(result, FractalResult)

    def test_fitness_in_0_1(self) -> None:
        fe = self._make_fe(score=0.75)
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert 0.0 <= result.best_fitness <= 1.0

    def test_best_fitness_non_negative(self) -> None:
        fe = self._make_fe(score=0.0)
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert result.best_fitness >= 0.0

    def test_evaluator_called_during_evolve(self) -> None:
        backend = _mock_backend("p")
        evaluator = _mock_evaluator(0.5)
        macro_cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2, n_generations=1)
        meso_cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        micro_cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=2, n_generations=1)
        fe = FractalEvolution(backend, evaluator, macro_cfg, meso_cfg, micro_cfg)
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert evaluator.evaluate.call_count > 0

    def test_n_cycles_1(self) -> None:
        fe = self._make_fe()
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert isinstance(result, FractalResult)

    def test_n_cycles_2(self) -> None:
        fe = self._make_fe()
        result = fe.evolve(_simple_genome(), "task", n_cycles=2)
        assert isinstance(result, FractalResult)

    def test_results_grow_after_evolve(self) -> None:
        fe = self._make_fe()
        assert len(fe.results) == 0
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert len(fe.results) == 1
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert len(fe.results) == 2

    def test_n_evaluations_positive(self) -> None:
        fe = self._make_fe()
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert result.n_evaluations > 0

    def test_backend_called_at_least_once(self) -> None:
        backend = _mock_backend("new text")
        evaluator = _mock_evaluator(0.5)
        macro_cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2, n_generations=1)
        meso_cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        micro_cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=2, n_generations=1)
        fe = FractalEvolution(backend, evaluator, macro_cfg, meso_cfg, micro_cfg)
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert backend.generate.call_count >= 1

    def test_uses_all_three_scales(self) -> None:
        """Ensure populations for all three scales are created (via evaluator calls)."""
        backend = _mock_backend("p")
        evaluator = _mock_evaluator(0.5)
        # population_size=2, n_generations=1 → 2 evaluate calls per scale → 6 total per cycle
        macro_cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2, n_generations=1)
        meso_cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        micro_cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=2, n_generations=1)
        fe = FractalEvolution(backend, evaluator, macro_cfg, meso_cfg, micro_cfg)
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        # 2 agents * 1 gen * 3 scales = 6 evaluate calls
        assert evaluator.evaluate.call_count == 6

    def test_result_scale_is_fractalscale(self) -> None:
        fe = self._make_fe()
        result = fe.evolve(_simple_genome(), "task", n_cycles=1)
        assert isinstance(result.scale, FractalScale)

    def test_best_fitness_across_cycles(self) -> None:
        """With n_cycles=2, returned fitness should be the best across all cycles."""
        backend = _mock_backend("better")
        # First cycle low, second cycle higher
        scores = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # cycle 1 (2*3 scales)
                  0.9, 0.9, 0.9, 0.9, 0.9, 0.9]  # cycle 2
        evaluator = MagicMock(spec=Evaluator)
        evaluator.evaluate.side_effect = lambda *_: scores.pop(0) if scores else 0.5
        macro_cfg = ScaleConfig(scale=FractalScale.MACRO, population_size=2, n_generations=1)
        meso_cfg = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        micro_cfg = ScaleConfig(scale=FractalScale.MICRO, population_size=2, n_generations=1)
        fe = FractalEvolution(backend, evaluator, macro_cfg, meso_cfg, micro_cfg)
        result = fe.evolve(_simple_genome(), "task", n_cycles=2)
        # Best result should be from cycle 2 with fitness 0.9
        assert result.best_fitness >= 0.3

    def test_results_property_is_copy(self) -> None:
        """Mutating the returned list should not affect internal state."""
        fe = self._make_fe()
        fe.evolve(_simple_genome(), "task", n_cycles=1)
        results_copy = fe.results
        results_copy.clear()
        assert len(fe.results) == 1
