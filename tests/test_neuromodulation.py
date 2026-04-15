# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.neuromodulation — Neuromodulator*, NeuromodulatorBank."""
from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.neuromodulation import (
    AcetylcholineModulator,
    DopamineModulator,
    NeuromodulatorBank,
    NeuroState,
    NoradrenalineModulator,
    SerotoninModulator,
    _diversity,
    _mean,
    _variance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float = 0.5, prompt: str = "agent prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _pop(fitnesses: list[float], prompts: list[str] | None = None) -> list[Agent]:
    if prompts is None:
        prompts = [f"agent prompt {i}" for i in range(len(fitnesses))]
    return [_make_agent(f, p) for f, p in zip(fitnesses, prompts)]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TestUtils:
    def test_mean_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_mean_values(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_variance_empty(self) -> None:
        assert _variance([]) == 0.0

    def test_variance_single(self) -> None:
        assert _variance([5.0]) == 0.0

    def test_variance_values(self) -> None:
        v = _variance([0.0, 1.0])
        assert v == pytest.approx(0.25)

    def test_diversity_empty(self) -> None:
        assert _diversity([]) == 0.0

    def test_diversity_identical_prompts(self) -> None:
        pop = [_make_agent(prompt="same prompt") for _ in range(3)]
        assert _diversity(pop) == pytest.approx(1 / 3)

    def test_diversity_all_unique(self) -> None:
        pop = [_make_agent(prompt=f"unique prompt {i}") for i in range(4)]
        assert _diversity(pop) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DopamineModulator
# ---------------------------------------------------------------------------


class TestDopamineModulator:
    def test_initial_level_is_neutral(self) -> None:
        mod = DopamineModulator()
        pop = _pop([0.5, 0.6])
        level = mod.level(pop, 0)
        assert 0.0 <= level <= 1.0

    def test_rising_fitness_increases_dopamine(self) -> None:
        mod = DopamineModulator(window=2)
        pop_low = _pop([0.3, 0.3])
        pop_high = _pop([0.9, 0.9])
        mod.level(pop_low, 0)
        level2 = mod.level(pop_high, 1)
        assert level2 > 0.5

    def test_stable_fitness_gives_neutral(self) -> None:
        mod = DopamineModulator(window=2)
        pop = _pop([0.5, 0.5])
        mod.level(pop, 0)
        level2 = mod.level(pop, 1)
        assert level2 == pytest.approx(0.5)

    def test_level_clamped_0_1(self) -> None:
        mod = DopamineModulator(window=2)
        mod.level(_pop([0.0]), 0)
        level = mod.level(_pop([2.0]), 1)  # extreme jump
        assert 0.0 <= level <= 1.0

    def test_empty_population(self) -> None:
        mod = DopamineModulator()
        level = mod.level([], 0)
        assert 0.0 <= level <= 1.0


# ---------------------------------------------------------------------------
# SerotoninModulator
# ---------------------------------------------------------------------------


class TestSerotoninModulator:
    def test_low_diversity_gives_high_serotonin(self) -> None:
        mod = SerotoninModulator(diversity_floor=0.5)
        pop = [_make_agent(prompt="same prompt") for _ in range(5)]
        level = mod.level(pop, 0)
        assert level == pytest.approx(1.0)

    def test_high_diversity_gives_low_serotonin(self) -> None:
        mod = SerotoninModulator(diversity_floor=0.1)
        pop = [_make_agent(prompt=f"unique prompt {i}") for i in range(10)]
        level = mod.level(pop, 0)
        assert level < 0.5

    def test_level_clamped_0_1(self) -> None:
        mod = SerotoninModulator()
        level = mod.level(_pop([0.5] * 3, ["a a", "b b", "c c"]), 0)
        assert 0.0 <= level <= 1.0

    def test_empty_population(self) -> None:
        mod = SerotoninModulator()
        level = mod.level([], 0)
        assert 0.0 <= level <= 1.0


# ---------------------------------------------------------------------------
# AcetylcholineModulator
# ---------------------------------------------------------------------------


class TestAcetylcholineModulator:
    def test_high_variance_gives_high_ach(self) -> None:
        mod = AcetylcholineModulator(variance_cap=0.1)
        pop = _pop([0.0, 0.0, 1.0, 1.0])
        level = mod.level(pop, 0)
        assert level == pytest.approx(1.0)  # variance = 0.25 > cap → clamped to 1.0

    def test_zero_variance_gives_zero_ach(self) -> None:
        mod = AcetylcholineModulator()
        pop = _pop([0.5, 0.5, 0.5])
        level = mod.level(pop, 0)
        assert level == pytest.approx(0.0)

    def test_level_clamped_0_1(self) -> None:
        mod = AcetylcholineModulator(variance_cap=0.001)
        pop = _pop([0.0, 1.0])
        level = mod.level(pop, 0)
        assert level == pytest.approx(1.0)

    def test_empty_population(self) -> None:
        mod = AcetylcholineModulator()
        level = mod.level([], 0)
        assert level == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NoradrenalineModulator
# ---------------------------------------------------------------------------


class TestNoradrenalineModulator:
    def test_no_stagnation_gives_low_nora(self) -> None:
        mod = NoradrenalineModulator(patience=3)
        pop = _pop([0.5])
        level = mod.level(pop, 0)
        assert level == pytest.approx(0.0)

    def test_stagnation_builds_up(self) -> None:
        mod = NoradrenalineModulator(patience=3, epsilon=0.01)
        pop = _pop([0.5])
        mod.level(pop, 0)
        mod.level(pop, 1)
        mod.level(pop, 2)
        level = mod.level(pop, 3)
        assert level == pytest.approx(1.0)

    def test_improvement_resets_stagnation(self) -> None:
        mod = NoradrenalineModulator(patience=3, epsilon=0.01)
        pop_low = _pop([0.5])
        pop_high = _pop([0.9])
        mod.level(pop_low, 0)
        mod.level(pop_low, 1)
        mod.level(pop_high, 2)  # improvement → reset
        level = mod.level(pop_high, 3)
        assert level < 1.0

    def test_empty_population(self) -> None:
        mod = NoradrenalineModulator()
        level = mod.level([], 0)
        assert 0.0 <= level <= 1.0


# ---------------------------------------------------------------------------
# NeuromodulatorBank — init
# ---------------------------------------------------------------------------


class TestNeuromodulatorBankInit:
    def test_defaults(self) -> None:
        bank = NeuromodulatorBank()
        assert bank._base_mr == pytest.approx(0.2)
        assert bank._base_sp == pytest.approx(0.5)
        assert bank._mr_range == pytest.approx(0.3)
        assert bank._sp_range == pytest.approx(0.3)

    def test_history_starts_empty(self) -> None:
        bank = NeuromodulatorBank()
        assert bank.history == []

    def test_custom_params(self) -> None:
        bank = NeuromodulatorBank(
            base_mutation_rate=0.4,
            base_selection_pressure=0.6,
            mr_range=0.2,
            sp_range=0.2,
        )
        assert bank._base_mr == pytest.approx(0.4)
        assert bank._base_sp == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# NeuromodulatorBank — modulate
# ---------------------------------------------------------------------------


class TestNeuromodulatorBankModulate:
    def test_returns_neuro_state(self) -> None:
        bank = NeuromodulatorBank()
        state = bank.modulate(_pop([0.5, 0.6]))
        assert isinstance(state, NeuroState)

    def test_state_appended_to_history(self) -> None:
        bank = NeuromodulatorBank()
        bank.modulate(_pop([0.5]))
        assert len(bank.history) == 1

    def test_mutation_rate_clamped(self) -> None:
        bank = NeuromodulatorBank(base_mutation_rate=0.0, mr_range=5.0)
        state = bank.modulate(_pop([0.5]))
        assert 0.0 <= state.mutation_rate <= 1.0

    def test_selection_pressure_clamped(self) -> None:
        bank = NeuromodulatorBank(base_selection_pressure=1.0, sp_range=5.0)
        state = bank.modulate(_pop([0.5]))
        assert 0.0 <= state.selection_pressure <= 1.0

    def test_all_modulator_levels_in_state(self) -> None:
        bank = NeuromodulatorBank()
        state = bank.modulate(_pop([0.5, 0.6]))
        for attr in ("dopamine", "serotonin", "acetylcholine", "noradrenaline"):
            val = getattr(state, attr)
            assert 0.0 <= val <= 1.0, f"{attr} out of range: {val}"

    def test_generation_recorded(self) -> None:
        bank = NeuromodulatorBank()
        state = bank.modulate(_pop([0.5]), generation=7)
        assert state.generation == 7

    def test_history_returns_copy(self) -> None:
        bank = NeuromodulatorBank()
        bank.modulate(_pop([0.5]))
        h1 = bank.history
        h1.clear()
        assert len(bank.history) == 1

    def test_stagnation_raises_mutation_rate(self) -> None:
        bank = NeuromodulatorBank(base_mutation_rate=0.2)
        pop = _pop([0.5, 0.5, 0.5])
        state0 = bank.modulate(pop, 0)
        # Simulate several stagnant generations
        for i in range(1, 5):
            bank.modulate(pop, i)
        state4 = bank.history[-1]
        assert state4.mutation_rate >= state0.mutation_rate

    def test_empty_population(self) -> None:
        bank = NeuromodulatorBank()
        state = bank.modulate([])
        assert isinstance(state, NeuroState)
        assert 0.0 <= state.mutation_rate <= 1.0


# ---------------------------------------------------------------------------
# NeuromodulatorBank — reset
# ---------------------------------------------------------------------------


class TestNeuromodulatorBankReset:
    def test_reset_clears_history(self) -> None:
        bank = NeuromodulatorBank()
        bank.modulate(_pop([0.5]))
        bank.reset()
        assert bank.history == []

    def test_reset_restores_baseline(self) -> None:
        bank = NeuromodulatorBank(base_mutation_rate=0.3)
        # Create stagnation
        pop = _pop([0.5])
        for i in range(5):
            bank.modulate(pop, i)
        bank.reset()
        # After reset, first call should be close to baseline
        state = bank.modulate(_pop([0.7]), 0)
        assert abs(state.mutation_rate - 0.3) < 0.4  # within mr_range
