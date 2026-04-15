"""Tests for cambrian.zeitgeber — ZeitgeberClock and ZeitgeberScheduler."""
from __future__ import annotations

import math

import pytest

from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler, ZeitgeberState


# ---------------------------------------------------------------------------
# ZeitgeberClock — init
# ---------------------------------------------------------------------------


class TestZeitgeberClockInit:
    def test_defaults(self) -> None:
        clk = ZeitgeberClock()
        assert clk._period == 20
        assert clk._amplitude == 0.5

    def test_period_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            ZeitgeberClock(period=0)

    def test_amplitude_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            ZeitgeberClock(amplitude=1.5)

    def test_starts_at_generation_zero(self) -> None:
        clk = ZeitgeberClock()
        assert clk.generation == 0


# ---------------------------------------------------------------------------
# ZeitgeberClock — phase and exploration_factor
# ---------------------------------------------------------------------------


class TestZeitgeberClockPhase:
    def test_phase_at_gen_zero(self) -> None:
        clk = ZeitgeberClock(period=20, phase_offset=0.0)
        assert pytest.approx(clk.phase(), abs=1e-9) == 0.0

    def test_phase_full_cycle_returns_to_zero(self) -> None:
        clk = ZeitgeberClock(period=20)
        for _ in range(20):
            clk.advance()
        assert pytest.approx(clk.phase(), abs=1e-9) == 0.0

    def test_phase_half_cycle(self) -> None:
        clk = ZeitgeberClock(period=20)
        for _ in range(10):
            clk.advance()
        assert pytest.approx(clk.phase(), abs=1e-6) == math.pi

    def test_exploration_factor_at_zero_phase(self) -> None:
        clk = ZeitgeberClock(period=20, amplitude=1.0)
        # phase=0 → sin=0 → ef = 0.5
        assert pytest.approx(clk.exploration_factor(), abs=1e-9) == 0.5

    def test_exploration_factor_at_quarter_cycle(self) -> None:
        clk = ZeitgeberClock(period=20, amplitude=1.0)
        for _ in range(5):  # quarter cycle → phase = π/2 → sin = 1
            clk.advance()
        assert pytest.approx(clk.exploration_factor(), abs=1e-6) == 1.0

    def test_exploration_factor_bounded(self) -> None:
        clk = ZeitgeberClock(period=4, amplitude=0.8)
        for _ in range(50):
            ef = clk.exploration_factor()
            assert 0.0 <= ef <= 1.0
            clk.advance()

    def test_reset_returns_to_zero(self) -> None:
        clk = ZeitgeberClock()
        for _ in range(5):
            clk.advance()
        clk.reset()
        assert clk.generation == 0


# ---------------------------------------------------------------------------
# ZeitgeberScheduler — init
# ---------------------------------------------------------------------------


class TestZeitgeberSchedulerInit:
    def test_defaults(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        assert sched._base_mr == 0.3
        assert sched._base_th == 0.5

    def test_history_starts_empty(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        assert sched.history == []

    def test_current_state_before_tick_is_none(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        assert sched.current_state() is None


# ---------------------------------------------------------------------------
# ZeitgeberScheduler — tick
# ---------------------------------------------------------------------------


class TestZeitgeberSchedulerTick:
    def test_returns_tuple(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        result = sched.tick()
        assert len(result) == 2
        mutation_rate, threshold = result
        assert 0.0 <= mutation_rate <= 1.0
        assert 0.0 <= threshold <= 1.0

    def test_history_grows_on_tick(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        sched.tick()
        sched.tick()
        assert len(sched.history) == 2

    def test_auto_advance_increments_generation(self) -> None:
        clk = ZeitgeberClock()
        sched = ZeitgeberScheduler(clock=clk, auto_advance=True)
        sched.tick()
        assert clk.generation == 1

    def test_no_auto_advance(self) -> None:
        clk = ZeitgeberClock()
        sched = ZeitgeberScheduler(clock=clk, auto_advance=False)
        sched.tick()
        assert clk.generation == 0

    def test_history_returns_copy(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        sched.tick()
        h1 = sched.history
        h1.clear()
        assert len(sched.history) == 1

    def test_current_state_after_tick(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock())
        sched.tick()
        state = sched.current_state()
        assert isinstance(state, ZeitgeberState)

    def test_state_contains_generation(self) -> None:
        clk = ZeitgeberClock()
        sched = ZeitgeberScheduler(clock=clk, auto_advance=False)
        sched.tick()
        assert sched.current_state().generation == 0  # type: ignore[union-attr]

    def test_oscillation_over_full_cycle(self) -> None:
        clk = ZeitgeberClock(period=20)
        sched = ZeitgeberScheduler(clock=clk, base_mutation_rate=0.5, mutation_range=0.4)
        rates = []
        for _ in range(20):
            mr, _ = sched.tick()
            rates.append(mr)
        # Should not be constant — there should be variation
        assert max(rates) - min(rates) > 0.1

    def test_rates_bounded(self) -> None:
        sched = ZeitgeberScheduler(clock=ZeitgeberClock(period=8))
        for _ in range(40):
            mr, th = sched.tick()
            assert 0.0 <= mr <= 1.0
            assert 0.0 <= th <= 1.0
