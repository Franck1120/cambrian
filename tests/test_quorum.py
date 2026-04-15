"""tests/test_quorum.py — Unit tests for QuorumSensor (quorum sensing).

Covers:
- compute_entropy: empty list, uniform scores, varied scores, out-of-range clamping
- update: boost when low entropy, decay when high entropy, no-op in middle
- update: rate clamping to [min_rate, max_rate]
- history tracking
- reset()
- Constructor validation
"""

from __future__ import annotations

import math

import pytest

from cambrian.quorum import QuorumSensor


# ─────────────────────────────────────────────────────────────────────────────
# compute_entropy
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeEntropy:
    def test_empty_list_returns_zero(self) -> None:
        sensor = QuorumSensor()
        assert sensor.compute_entropy([]) == pytest.approx(0.0)

    def test_single_score_returns_zero(self) -> None:
        sensor = QuorumSensor()
        assert sensor.compute_entropy([0.5]) == pytest.approx(0.0)

    def test_uniform_population_high_entropy(self) -> None:
        """All different bins → maximum entropy."""
        sensor = QuorumSensor(n_bins=5)
        # Spread scores across 5 bins: 0.1, 0.3, 0.5, 0.7, 0.9
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        h = sensor.compute_entropy(scores)
        # Perfect spread across 5 bins → H = log(5) ≈ 1.609
        assert h == pytest.approx(math.log(5), rel=0.01)

    def test_converged_population_low_entropy(self) -> None:
        """All scores identical → single bin → entropy = 0."""
        sensor = QuorumSensor()
        scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        h = sensor.compute_entropy(scores)
        assert h == pytest.approx(0.0)

    def test_out_of_range_scores_clamped(self) -> None:
        sensor = QuorumSensor()
        # Should not raise; values clamped to [0, 1]
        h = sensor.compute_entropy([-1.0, 2.0, 0.5])
        assert h >= 0.0

    def test_entropy_increases_with_diversity(self) -> None:
        sensor = QuorumSensor(n_bins=10)
        # All same → low entropy
        same = [0.5] * 10
        # Spread → higher entropy
        varied = [i / 10 for i in range(10)]
        assert sensor.compute_entropy(same) < sensor.compute_entropy(varied)

    def test_returns_non_negative(self) -> None:
        sensor = QuorumSensor()
        for _ in range(10):
            scores = [float(i) / 10 for i in range(10)]
            assert sensor.compute_entropy(scores) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# update — rate adjustments
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_low_entropy_boosts_rate(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=1.5,
            high_entropy_threshold=2.5,
            boost_factor=1.3,
        )
        # All same score → H=0 < 1.5 → boost
        scores = [0.8, 0.8, 0.8, 0.8]
        new_rate = sensor.update(scores, current_rate=0.5)
        assert new_rate > 0.5

    def test_high_entropy_decays_rate(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=0.5,
            high_entropy_threshold=1.2,
            decay_factor=0.8,
            n_bins=5,
        )
        # Perfectly spread scores → high entropy
        scores = [0.0, 0.2, 0.4, 0.6, 0.8]
        new_rate = sensor.update(scores, current_rate=0.8)
        assert new_rate < 0.8

    def test_balanced_entropy_unchanged(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=0.0,
            high_entropy_threshold=10.0,
        )
        # Thresholds are extreme → never triggers
        scores = [0.3, 0.5, 0.7]
        new_rate = sensor.update(scores, current_rate=0.6)
        assert new_rate == pytest.approx(0.6)

    def test_rate_clamped_to_max(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=1.0,
            high_entropy_threshold=2.0,
            boost_factor=10.0,  # huge boost
            max_rate=1.0,
        )
        scores = [0.5, 0.5, 0.5]  # H=0 → boost
        new_rate = sensor.update(scores, current_rate=0.9)
        assert new_rate <= 1.0

    def test_rate_clamped_to_min(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=0.5,
            high_entropy_threshold=0.8,
            decay_factor=0.01,  # huge decay
            min_rate=0.1,
            n_bins=5,
        )
        scores = [0.0, 0.2, 0.4, 0.6, 0.8]  # high entropy
        new_rate = sensor.update(scores, current_rate=0.5)
        assert new_rate >= 0.1

    def test_last_entropy_set_after_update(self) -> None:
        sensor = QuorumSensor()
        assert sensor.last_entropy is None
        sensor.update([0.5, 0.5], current_rate=0.8)
        assert sensor.last_entropy is not None

    def test_history_appended_per_call(self) -> None:
        sensor = QuorumSensor()
        assert sensor.history == []
        sensor.update([0.5], current_rate=0.8)
        assert len(sensor.history) == 1
        sensor.update([0.3, 0.7], current_rate=0.8)
        assert len(sensor.history) == 2

    def test_history_contains_entropy_and_rate(self) -> None:
        sensor = QuorumSensor()
        sensor.update([0.5, 0.5], current_rate=0.8)
        entropy, rate = sensor.history[0]
        assert entropy >= 0.0
        assert 0.0 <= rate <= 1.0

    def test_multiple_updates_track_correctly(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=0.5,
            high_entropy_threshold=3.0,
            boost_factor=1.2,
            decay_factor=0.9,
        )
        rate = 0.8
        for i in range(5):
            rate = sensor.update([0.5] * 5, current_rate=rate)  # converged → boost each time
        assert len(sensor.history) == 5
        # Rate should have grown (or hit max)
        assert rate >= 0.8


# ─────────────────────────────────────────────────────────────────────────────
# reset
# ─────────────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_history(self) -> None:
        sensor = QuorumSensor()
        sensor.update([0.5], current_rate=0.8)
        sensor.reset()
        assert sensor.history == []

    def test_reset_clears_last_entropy(self) -> None:
        sensor = QuorumSensor()
        sensor.update([0.5], current_rate=0.8)
        sensor.reset()
        assert sensor.last_entropy is None

    def test_can_use_after_reset(self) -> None:
        sensor = QuorumSensor()
        sensor.update([0.5, 0.5], current_rate=0.8)
        sensor.reset()
        rate = sensor.update([0.3, 0.7], current_rate=0.5)
        assert 0.0 < rate <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructor:
    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="low_entropy_threshold"):
            QuorumSensor(low_entropy_threshold=2.0, high_entropy_threshold=1.0)

    def test_equal_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            QuorumSensor(low_entropy_threshold=1.5, high_entropy_threshold=1.5)

    def test_valid_construction(self) -> None:
        sensor = QuorumSensor(
            low_entropy_threshold=0.5,
            high_entropy_threshold=2.0,
            boost_factor=1.4,
            decay_factor=0.9,
            min_rate=0.05,
            max_rate=0.99,
            n_bins=20,
        )
        assert sensor.last_entropy is None
        assert sensor.history == []

    def test_history_is_copy(self) -> None:
        """history property returns a copy, not the internal list."""
        sensor = QuorumSensor()
        sensor.update([0.5], 0.8)
        h = sensor.history
        h.clear()
        assert len(sensor.history) == 1
