"""Tests for cambrian.annealing — AnnealingSchedule and AnnealingSelector."""
from __future__ import annotations

import random

import pytest

from cambrian.annealing import AnnealingEvent, AnnealingSchedule, AnnealingSelector


# ---------------------------------------------------------------------------
# AnnealingSchedule — init validation
# ---------------------------------------------------------------------------


class TestAnnealingScheduleInit:
    def test_valid_defaults(self) -> None:
        s = AnnealingSchedule()
        assert s.T_max == 2.0
        assert s.T_min == 0.01

    def test_T_min_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            AnnealingSchedule(T_min=0)

    def test_T_max_below_T_min_raises(self) -> None:
        with pytest.raises(ValueError):
            AnnealingSchedule(T_max=0.5, T_min=1.0)

    def test_n_steps_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            AnnealingSchedule(n_steps=0)


# ---------------------------------------------------------------------------
# AnnealingSchedule — temperature curves
# ---------------------------------------------------------------------------


class TestTemperatureCurves:
    def test_linear_starts_at_T_max(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.1, n_steps=10, schedule_type="linear")
        assert pytest.approx(s.temperature(0), abs=1e-9) == 2.0

    def test_linear_ends_at_T_min(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.1, n_steps=10, schedule_type="linear")
        assert pytest.approx(s.temperature(10), abs=1e-9) == 0.1

    def test_linear_midpoint(self) -> None:
        s2 = AnnealingSchedule(T_max=2.0, T_min=1.0, n_steps=10, schedule_type="linear")
        assert pytest.approx(s2.temperature(5), abs=1e-9) == 1.5

    def test_exponential_starts_at_T_max(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10, schedule_type="exponential")
        assert pytest.approx(s.temperature(0), abs=1e-9) == 2.0

    def test_exponential_ends_at_T_min(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10, schedule_type="exponential")
        assert pytest.approx(s.temperature(10), abs=1e-6) == 0.01

    def test_exponential_monotone_decreasing(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10, schedule_type="exponential")
        temps = [s.temperature(t) for t in range(11)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_cosine_starts_at_T_max(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10, schedule_type="cosine")
        assert pytest.approx(s.temperature(0), abs=1e-6) == 2.0

    def test_cosine_ends_at_T_min(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10, schedule_type="cosine")
        assert pytest.approx(s.temperature(10), abs=1e-6) == 0.01

    def test_temperature_clamped_above_n_steps(self) -> None:
        s = AnnealingSchedule(T_max=2.0, T_min=0.01, n_steps=10)
        # t > n_steps → clamped to n_steps
        assert pytest.approx(s.temperature(100), abs=1e-6) == s.temperature(10)


# ---------------------------------------------------------------------------
# AnnealingSelector — init
# ---------------------------------------------------------------------------


class TestAnnealingSelectorInit:
    def test_starts_at_step_zero(self) -> None:
        s = AnnealingSelector(schedule=AnnealingSchedule())
        assert s.current_step == 0

    def test_history_starts_empty(self) -> None:
        s = AnnealingSelector(schedule=AnnealingSchedule())
        assert s.history == []


# ---------------------------------------------------------------------------
# AnnealingSelector — step
# ---------------------------------------------------------------------------


class TestAnnealingSelectorStep:
    def test_always_accepts_improvement(self) -> None:
        schedule = AnnealingSchedule(T_max=0.01, T_min=0.001, n_steps=100)
        sel = AnnealingSelector(schedule=schedule, rng=random.Random(42))
        # candidate better → always accept
        accepted = sel.step(current_fitness=0.5, candidate_fitness=0.8)
        assert accepted is True

    def test_always_accepts_equal_fitness(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        accepted = sel.step(current_fitness=0.5, candidate_fitness=0.5)
        assert accepted is True

    def test_worse_candidate_sometimes_accepted(self) -> None:
        # High temperature → high acceptance probability for worse candidates
        schedule = AnnealingSchedule(T_max=100.0, T_min=0.01, n_steps=1000)
        sel = AnnealingSelector(schedule=schedule, rng=random.Random(1))
        results = [sel.step(0.8, 0.1) for _ in range(50)]
        assert any(r is True for r in results)

    def test_worse_candidate_mostly_rejected_at_low_temp(self) -> None:
        # Cold temperature → near-zero acceptance for large regression
        schedule = AnnealingSchedule(T_max=0.001, T_min=0.0001, n_steps=10)
        sel = AnnealingSelector(schedule=schedule, rng=random.Random(0))
        # Jump to last step
        sel._step = 10
        accepted = sel.step(0.9, 0.1)
        # With T ~0.0001 and delta=0.8, prob = exp(-0.8/0.0001) ≈ 0
        assert accepted is False

    def test_step_increments(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.5, 0.6)
        assert sel.current_step == 1

    def test_history_recorded(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.5, 0.7)
        assert len(sel.history) == 1
        ev = sel.history[0]
        assert isinstance(ev, AnnealingEvent)
        assert ev.current_fitness == 0.5
        assert ev.candidate_fitness == 0.7
        assert ev.accepted is True

    def test_history_returns_copy(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.5, 0.7)
        h1 = sel.history
        h1.clear()
        assert len(sel.history) == 1

    def test_reset_clears_state(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.5, 0.7)
        sel.reset()
        assert sel.current_step == 0
        assert sel.history == []

    def test_acceptance_rate(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.5, 0.9)  # accepted
        sel.step(0.5, 0.9)  # accepted
        assert pytest.approx(sel.acceptance_rate(), abs=1e-9) == 1.0

    def test_acceptance_rate_zero_steps(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        assert sel.acceptance_rate() == 0.0

    def test_acceptance_prob_for_improvement_is_one(self) -> None:
        sel = AnnealingSelector(schedule=AnnealingSchedule())
        sel.step(0.3, 0.9)
        assert sel.history[0].acceptance_prob == 1.0
