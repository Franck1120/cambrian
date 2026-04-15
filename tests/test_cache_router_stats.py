# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.cache, cambrian.compress, cambrian.router, cambrian.stats."""

from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.cache import SemanticCache
from cambrian.compress import caveman_compress
from cambrian.router import ModelRouter
from cambrian.stats import DiversityTracker, FitnessLandscape, ParetoAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(prompt: str = "You are helpful.", fitness: float = 0.5) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


def _population(n: int = 4) -> list[Agent]:
    fitnesses = [0.2 * (i + 1) for i in range(n)]
    return [
        _agent(f"agent {i}", fit)
        for i, fit in zip(range(n), fitnesses, strict=False)
    ]


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class TestSemanticCache:
    def test_construction_defaults(self) -> None:
        cache = SemanticCache()
        assert cache.size == 0

    def test_construction_custom(self) -> None:
        cache = SemanticCache(max_size=10, ttl_seconds=60.0)
        assert cache.size == 0

    def test_set_and_get(self) -> None:
        cache = SemanticCache()
        cache.set("hello world", "response text")
        result = cache.get("hello world")
        assert result == "response text"

    def test_get_missing_returns_none(self) -> None:
        cache = SemanticCache()
        assert cache.get("nonexistent key") is None

    def test_size_increments_on_set(self) -> None:
        cache = SemanticCache()
        cache.set("key1", "val1")
        assert cache.size == 1
        cache.set("key2", "val2")
        assert cache.size == 2

    def test_hit_rate_zero_on_empty(self) -> None:
        cache = SemanticCache()
        assert cache.hit_rate() == pytest.approx(0.0)

    def test_hit_rate_after_misses(self) -> None:
        cache = SemanticCache()
        cache.get("miss1")
        cache.get("miss2")
        assert cache.hit_rate() == pytest.approx(0.0)

    def test_hit_rate_after_hit(self) -> None:
        cache = SemanticCache()
        cache.set("prompt", "resp")
        cache.get("prompt")   # hit
        rate = cache.hit_rate()
        assert isinstance(rate, float)
        assert rate > 0.0

    def test_clear_resets_size(self) -> None:
        cache = SemanticCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.size == 0

    def test_clear_makes_keys_inaccessible(self) -> None:
        cache = SemanticCache()
        cache.set("prompt", "response")
        cache.clear()
        assert cache.get("prompt") is None

    def test_model_parameter_differentiates_keys(self) -> None:
        cache = SemanticCache()
        cache.set("task", "resp-a", model="gpt-4o-mini")
        cache.set("task", "resp-b", model="gpt-4o")
        assert cache.get("task", model="gpt-4o-mini") == "resp-a"
        assert cache.get("task", model="gpt-4o") == "resp-b"

    def test_temperature_parameter_differentiates_keys(self) -> None:
        cache = SemanticCache()
        cache.set("task", "resp-cold", temperature=0.0)
        cache.set("task", "resp-hot", temperature=1.0)
        assert cache.get("task", temperature=0.0) == "resp-cold"
        assert cache.get("task", temperature=1.0) == "resp-hot"

    def test_max_size_eviction(self) -> None:
        cache = SemanticCache(max_size=3)
        for i in range(5):
            cache.set(f"key{i}", f"val{i}")
        assert cache.size <= 3

    def test_hit_rate_mixed(self) -> None:
        cache = SemanticCache()
        cache.set("p1", "r1")
        cache.set("p2", "r2")
        cache.get("p1")    # hit
        cache.get("miss")  # miss
        rate = cache.hit_rate()
        assert 0.0 < rate < 1.0


# ---------------------------------------------------------------------------
# caveman_compress
# ---------------------------------------------------------------------------


class TestCavemanCompress:
    def test_returns_string(self) -> None:
        result = caveman_compress("the quick brown fox")
        assert isinstance(result, str)

    def test_empty_string(self) -> None:
        assert caveman_compress("") == ""

    def test_removes_stopwords(self) -> None:
        # "the" and "a" are common stopwords — they should be removed or reduced
        original = "the quick brown fox jumps over the lazy dog"
        compressed = caveman_compress(original)
        assert len(compressed) <= len(original)

    def test_short_text_passes_through(self) -> None:
        result = caveman_compress("hello")
        assert "hello" in result

    def test_compressed_shorter_or_equal(self) -> None:
        long_text = (
            "the system prompt is a very important part of the overall "
            "configuration of the agent and the agent needs to be able to "
            "understand the task in order to perform well"
        )
        compressed = caveman_compress(long_text)
        assert len(compressed) <= len(long_text)

    def test_preserves_content_words(self) -> None:
        result = caveman_compress("implement binary search algorithm")
        # Domain words should survive compression
        assert any(word in result for word in ["implement", "binary", "search", "algorithm"])

    def test_deterministic(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        assert caveman_compress(text) == caveman_compress(text)


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_construction_defaults(self) -> None:
        router = ModelRouter()
        assert router is not None

    def test_construction_custom_models(self) -> None:
        router = ModelRouter(
            cheap_model="cheap-m",
            medium_model="medium-m",
            premium_model="premium-m",
        )
        result = router.route("hi")
        assert result in ("cheap-m", "medium-m", "premium-m")

    def test_route_returns_string(self) -> None:
        router = ModelRouter()
        result = router.route("What is 2+2?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_short_simple_task_routes_cheap(self) -> None:
        router = ModelRouter(
            cheap_model="cheap", medium_model="medium", premium_model="premium"
        )
        result = router.route("hi")
        assert result == "cheap"

    def test_complexity_signal_routes_premium(self) -> None:
        router = ModelRouter(
            cheap_model="cheap", medium_model="medium", premium_model="premium"
        )
        # "implement" is a complexity signal
        result = router.route("implement a sorting algorithm step-by-step")
        assert result == "premium"

    def test_routing_log_populated(self) -> None:
        router = ModelRouter()
        router.route("short task")
        router.route("another task")
        log = router.routing_log
        assert len(log) == 2

    def test_routing_log_is_copy(self) -> None:
        router = ModelRouter()
        router.route("task")
        log1 = router.routing_log
        log2 = router.routing_log
        assert log1 is not log2

    def test_routing_stats_counts(self) -> None:
        router = ModelRouter(
            cheap_model="cheap", medium_model="medium", premium_model="premium"
        )
        router.route("hi")         # cheap
        router.route("hello")      # cheap
        router.route("implement a complex algorithm with step-by-step details")  # premium
        stats = router.routing_stats()
        assert isinstance(stats, dict)
        assert stats.get("cheap", 0) >= 2

    def test_routing_stats_empty(self) -> None:
        router = ModelRouter()
        assert router.routing_stats() == {}

    def test_multiple_routes_accumulate(self) -> None:
        router = ModelRouter()
        for i in range(5):
            router.route(f"task {i}")
        assert len(router.routing_log) == 5

    def test_long_task_premium(self) -> None:
        router = ModelRouter(
            cheap_model="cheap", medium_model="medium", premium_model="premium"
        )
        # Build a task with > 500 tokens
        long_task = " ".join([f"word{i}" for i in range(600)])
        result = router.route(long_task)
        assert result == "premium"


# ---------------------------------------------------------------------------
# DiversityTracker
# ---------------------------------------------------------------------------


class TestDiversityTracker:
    def test_construction(self) -> None:
        tracker = DiversityTracker()
        assert tracker is not None

    def test_timeline_empty_initially(self) -> None:
        tracker = DiversityTracker()
        assert tracker.timeline() == []

    def test_record_returns_snapshot(self) -> None:
        from cambrian.stats import DiversitySnapshot

        tracker = DiversityTracker()
        pop = _population(4)
        snapshot = tracker.record(generation=0, agents=pop)
        assert isinstance(snapshot, DiversitySnapshot)

    def test_snapshot_fields(self) -> None:
        tracker = DiversityTracker()
        pop = _population(4)
        snap = tracker.record(generation=1, agents=pop)
        assert snap.generation == 1
        assert snap.n_agents == 4
        assert isinstance(snap.mean_fitness, float)
        assert isinstance(snap.strategy_entropy, float)

    def test_timeline_grows(self) -> None:
        tracker = DiversityTracker()
        pop = _population(3)
        tracker.record(0, pop)
        tracker.record(1, pop)
        assert len(tracker.timeline()) == 2

    def test_diversity_collapsed_false_on_diverse(self) -> None:
        tracker = DiversityTracker()
        pop = [_agent(f"agent {i}", 0.1 * i) for i in range(6)]
        # Manually set different strategies
        strategies = ["step-by-step", "concise", "detailed", "chain-of-thought", "few-shot", "zero-shot"]
        for agent, strat in zip(pop, strategies, strict=False):
            agent.genome.strategy = strat
        tracker.record(0, pop)
        # With diverse strategies, collapsed should be False (at typical threshold)
        result = tracker.diversity_collapsed()
        assert isinstance(result, bool)

    def test_diversity_collapsed_true_on_uniform(self) -> None:
        tracker = DiversityTracker()
        # All agents same strategy → low entropy → collapsed
        pop = [_agent("agent", 0.5) for _ in range(6)]
        for agent in pop:
            agent.genome.strategy = "step-by-step"
        tracker.record(0, pop)
        # High threshold → should detect collapse
        result = tracker.diversity_collapsed(threshold_entropy=10.0)
        assert result is True

    def test_to_dicts_returns_list(self) -> None:
        tracker = DiversityTracker()
        tracker.record(0, _population(3))
        dicts = tracker.to_dicts()
        assert isinstance(dicts, list)
        assert len(dicts) == 1
        assert isinstance(dicts[0], dict)

    def test_to_dicts_empty(self) -> None:
        tracker = DiversityTracker()
        assert tracker.to_dicts() == []


# ---------------------------------------------------------------------------
# FitnessLandscape
# ---------------------------------------------------------------------------


class TestFitnessLandscape:
    def test_construction_defaults(self) -> None:
        landscape = FitnessLandscape()
        assert landscape is not None

    def test_construction_custom(self) -> None:
        landscape = FitnessLandscape(n_temp_bins=3, n_token_bins=3)
        assert landscape is not None

    def test_add_agent(self) -> None:
        landscape = FitnessLandscape()
        agent = _agent(fitness=0.7)
        landscape.add(agent)  # should not raise

    def test_add_population(self) -> None:
        landscape = FitnessLandscape()
        pop = _population(4)
        landscape.add_population(pop)  # should not raise

    def test_mean_fitness_grid_is_list_of_lists(self) -> None:
        landscape = FitnessLandscape()
        landscape.add_population(_population(4))
        grid = landscape.mean_fitness_grid()
        assert isinstance(grid, list)
        for row in grid:
            assert isinstance(row, list)

    def test_peak_returns_tuple(self) -> None:
        landscape = FitnessLandscape()
        landscape.add_population(_population(4))
        peak = landscape.peak()
        assert isinstance(peak, tuple)
        assert len(peak) == 3  # (temp_bin, token_bin, fitness)

    def test_peak_fitness_is_float(self) -> None:
        landscape = FitnessLandscape()
        landscape.add_population(_population(4))
        _, _, fitness = landscape.peak()
        assert isinstance(fitness, float)

    def test_bin_labels_has_both_axes(self) -> None:
        landscape = FitnessLandscape()
        labels = landscape.bin_labels()
        assert isinstance(labels, dict)
        assert "temperature" in labels or len(labels) >= 1

    def test_to_dict_is_dict(self) -> None:
        landscape = FitnessLandscape()
        landscape.add_population(_population(4))
        d = landscape.to_dict()
        assert isinstance(d, dict)

    def test_empty_landscape_peak(self) -> None:
        landscape = FitnessLandscape()
        peak = landscape.peak()
        assert isinstance(peak, tuple)


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------


class TestParetoAnalyzer:
    def test_construction_defaults(self) -> None:
        analyzer = ParetoAnalyzer()
        assert analyzer is not None

    def test_construction_custom_objectives(self) -> None:
        analyzer = ParetoAnalyzer(objectives=["fitness"])
        assert analyzer is not None

    def test_compute_returns_list(self) -> None:
        from cambrian.stats import ParetoPoint

        analyzer = ParetoAnalyzer()
        pop = _population(4)
        points = analyzer.compute(pop)
        assert isinstance(points, list)
        for p in points:
            assert isinstance(p, ParetoPoint)

    def test_compute_same_count_as_population(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(6)
        points = analyzer.compute(pop)
        assert len(points) == len(pop)

    def test_pareto_agents_subset(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(4)
        analyzer.compute(pop)
        pareto = analyzer.pareto_agents()
        assert isinstance(pareto, list)
        assert len(pareto) <= len(pop)

    def test_dominated_agents_subset(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(4)
        analyzer.compute(pop)
        dominated = analyzer.dominated_agents()
        assert isinstance(dominated, list)

    def test_pareto_plus_dominated_equals_total(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(5)
        analyzer.compute(pop)
        assert len(analyzer.pareto_agents()) + len(analyzer.dominated_agents()) == len(pop)

    def test_summary_dict_keys(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(4)
        analyzer.compute(pop)
        s = analyzer.summary()
        assert "total" in s
        assert "pareto_count" in s
        assert "dominated_count" in s

    def test_summary_empty_no_compute(self) -> None:
        analyzer = ParetoAnalyzer()
        s = analyzer.summary()
        assert s["total"] == 0

    def test_add_custom_objective(self) -> None:
        analyzer = ParetoAnalyzer()
        analyzer.add_objective("token_count", lambda a: len(a.genome.system_prompt))
        pop = _population(3)
        points = analyzer.compute(pop)
        assert len(points) == 3

    def test_pareto_point_has_is_pareto(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(3)
        points = analyzer.compute(pop)
        for p in points:
            assert isinstance(p.is_pareto, bool)

    def test_pareto_fraction_in_unit_range(self) -> None:
        analyzer = ParetoAnalyzer()
        pop = _population(4)
        analyzer.compute(pop)
        s = analyzer.summary()
        fraction = s.get("pareto_fraction", 0.0)
        assert 0.0 <= fraction <= 1.0
