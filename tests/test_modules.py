"""Unit tests for cache, compress, router, diversity, and memory modules."""

from __future__ import annotations

import time

import pytest

from cambrian.agent import Agent, Genome
from cambrian.cache import SemanticCache
from cambrian.compress import caveman_compress, procut_prune
from cambrian.diversity import MAPElites
from cambrian.memory import EvolutionaryMemory
from cambrian.router import ModelRouter


# ── helpers ───────────────────────────────────────────────────────────────────

def _agent(prompt: str = "x", temperature: float = 0.5, fitness: float = 0.5) -> Agent:
    a = Agent(genome=Genome(system_prompt=prompt, temperature=temperature))
    a._fitness = fitness
    return a


# ── SemanticCache ─────────────────────────────────────────────────────────────

class TestSemanticCache:
    def test_miss_returns_none(self) -> None:
        c = SemanticCache()
        assert c.get("hello", "gpt-4o", 0.7) is None

    def test_hit_returns_cached(self) -> None:
        c = SemanticCache()
        c.set("hello", "world", model="gpt-4o", temperature=0.7)
        assert c.get("hello", "gpt-4o", 0.7) == "world"

    def test_different_keys_no_collision(self) -> None:
        c = SemanticCache()
        c.set("p1", "r1", model="m", temperature=0.5)
        c.set("p2", "r2", model="m", temperature=0.5)
        assert c.get("p1", "m", 0.5) == "r1"
        assert c.get("p2", "m", 0.5) == "r2"

    def test_different_temperature_different_key(self) -> None:
        c = SemanticCache()
        c.set("p", "hot", model="m", temperature=1.0)
        assert c.get("p", "m", 0.0) is None  # different temp → miss

    def test_lru_eviction(self) -> None:
        c = SemanticCache(max_size=2)
        c.set("a", "va", model="m", temperature=0.0)
        c.set("b", "vb", model="m", temperature=0.0)
        c.set("c", "vc", model="m", temperature=0.0)  # evicts "a" (LRU)
        assert c.get("a", "m", 0.0) is None
        assert c.get("b", "m", 0.0) == "vb"
        assert c.get("c", "m", 0.0) == "vc"
        assert c.size == 2

    def test_lru_promotes_on_get(self) -> None:
        c = SemanticCache(max_size=2)
        c.set("a", "va", model="m", temperature=0.0)
        c.set("b", "vb", model="m", temperature=0.0)
        c.get("a", "m", 0.0)  # promote "a"
        c.set("c", "vc", model="m", temperature=0.0)  # should evict "b" now
        assert c.get("a", "m", 0.0) == "va"
        assert c.get("b", "m", 0.0) is None

    def test_ttl_expires(self) -> None:
        c = SemanticCache(ttl_seconds=0.01)
        c.set("p", "r", model="m", temperature=0.0)
        time.sleep(0.02)
        assert c.get("p", "m", 0.0) is None

    def test_hit_rate(self) -> None:
        c = SemanticCache()
        c.set("p", "r")
        c.get("p")          # hit
        c.get("missing")    # miss
        assert c.hit_rate() == pytest.approx(0.5)

    def test_hit_rate_zero_when_empty(self) -> None:
        c = SemanticCache()
        assert c.hit_rate() == pytest.approx(0.0)

    def test_clear_resets(self) -> None:
        c = SemanticCache()
        c.set("p", "r")
        c.get("p")
        c.clear()
        assert c.size == 0
        assert c.hit_rate() == pytest.approx(0.0)


# ── caveman_compress ──────────────────────────────────────────────────────────

class TestCavemanCompress:
    def test_removes_stopwords(self) -> None:
        result = caveman_compress("implement a very simple function")
        assert "a" not in result.split()
        assert "very" not in result.split()
        assert "implement" in result
        assert "simple" in result
        assert "function" in result

    def test_removes_filler_phrases(self) -> None:
        result = caveman_compress("Please note that you should be careful")
        assert "please note that" not in result.lower()

    def test_preserves_important_words(self) -> None:
        result = caveman_compress("calculate derivative integral")
        assert "calculate" in result
        assert "derivative" in result
        assert "integral" in result

    def test_empty_string(self) -> None:
        assert caveman_compress("") == ""

    def test_returns_string(self) -> None:
        assert isinstance(caveman_compress("hello world"), str)


# ── procut_prune ──────────────────────────────────────────────────────────────

class TestProcutPrune:
    def _genome(self, prompt: str) -> Genome:
        return Genome(system_prompt=prompt)

    def test_already_short_unchanged(self) -> None:
        g = self._genome("Short prompt.")
        result = procut_prune(g, max_tokens=256)
        assert result.system_prompt == "Short prompt."

    def test_prunes_long_prompt(self) -> None:
        # Create prompt with many paragraphs > 256 tokens (~1024 chars)
        para = "This is a paragraph with meaningful content about the task."
        long_prompt = "\n\n".join([para] * 30)
        g = self._genome(long_prompt)
        result = procut_prune(g, max_tokens=256)
        assert len(result.system_prompt) < len(long_prompt)
        assert len(result.system_prompt) // 4 <= 256 + 20  # approximate

    def test_preserves_at_least_one_paragraph(self) -> None:
        single = "Only paragraph here."
        g = self._genome(single)
        result = procut_prune(g, max_tokens=1)  # very tight budget
        assert result.system_prompt == single

    def test_returns_new_genome_object(self) -> None:
        g = self._genome("Some content.")
        result = procut_prune(g, max_tokens=256)
        assert result is not g

    def test_original_not_modified(self) -> None:
        original_text = "Original text."
        g = self._genome(original_text)
        procut_prune(g, max_tokens=1)
        assert g.system_prompt == original_text


# ── ModelRouter ───────────────────────────────────────────────────────────────

class TestModelRouter:
    def test_short_simple_task_routes_cheap(self) -> None:
        r = ModelRouter()
        model = r.route("What is 2+2?")
        assert model == r._cheap

    def test_long_task_routes_premium(self) -> None:
        r = ModelRouter()
        long_task = " ".join(["word"] * 600)
        model = r.route(long_task)
        assert model == r._premium

    def test_complexity_signal_routes_premium(self) -> None:
        r = ModelRouter()
        model = r.route("Implement a complex algorithm")
        assert model == r._premium

    def test_routing_log_populated(self) -> None:
        r = ModelRouter()
        r.route("hello")
        r.route("world")
        assert len(r.routing_log) == 2

    def test_routing_stats_counts(self) -> None:
        r = ModelRouter(cheap_model="cheap", medium_model="medium", premium_model="premium")
        r.route("hi")          # cheap
        r.route("implement a complex algorithm")  # premium
        stats = r.routing_stats()
        assert stats.get("cheap", 0) >= 1

    def test_custom_models(self) -> None:
        r = ModelRouter(cheap_model="fast", medium_model="mid", premium_model="big")
        assert r.route("hello") == "fast"
        assert r.route("implement complex step-by-step algorithm design") == "big"


# ── MAPElites ─────────────────────────────────────────────────────────────────

class TestMAPElites:
    def test_empty_archive(self) -> None:
        me = MAPElites()
        assert me.occupancy == 0
        assert me.coverage() == 0.0
        assert me.best() is None

    def test_agent_without_fitness_not_inserted(self) -> None:
        me = MAPElites()
        a = Agent(genome=Genome(system_prompt="x"))
        inserted = me.add(a)
        assert not inserted
        assert me.occupancy == 0

    def test_inserts_agent_with_fitness(self) -> None:
        me = MAPElites()
        a = _agent("hello", temperature=0.3, fitness=0.7)
        inserted = me.add(a)
        assert inserted
        assert me.occupancy == 1

    def test_better_agent_replaces_worse(self) -> None:
        me = MAPElites()
        a_low = _agent("x", temperature=0.3, fitness=0.3)
        a_high = _agent("x " * 50, temperature=0.3, fitness=0.9)  # same bucket
        me.add(a_low)
        me.add(a_high)
        best = me.best()
        assert best is not None
        assert best.fitness == pytest.approx(0.9)

    def test_worse_agent_does_not_replace(self) -> None:
        me = MAPElites()
        a_high = _agent("x", temperature=0.3, fitness=0.9)
        a_low = _agent("y", temperature=0.3, fitness=0.1)
        me.add(a_high)
        inserted = me.add(a_low)
        assert not inserted

    def test_diverse_population_multiple_cells(self) -> None:
        me = MAPElites()
        # short prompt, cold temp
        me.add(_agent("Hi.", temperature=0.1, fitness=0.5))
        # long prompt, hot temp
        me.add(_agent("x " * 200, temperature=1.2, fitness=0.6))
        assert me.occupancy >= 2

    def test_get_diverse_population_sorted(self) -> None:
        me = MAPElites()
        me.add(_agent("a", temperature=0.1, fitness=0.3))
        me.add(_agent("b " * 100, temperature=1.0, fitness=0.9))
        pop = me.get_diverse_population()
        assert pop[0].fitness >= pop[-1].fitness  # descending

    def test_coverage_fraction(self) -> None:
        me = MAPElites(n_prompt_buckets=2, n_temp_buckets=2)
        me.add(_agent("short", temperature=0.1, fitness=0.5))
        assert 0.0 < me.coverage() <= 1.0


# ── EvolutionaryMemory ────────────────────────────────────────────────────────

class TestEvolutionaryMemory:
    def test_add_and_count(self) -> None:
        m = EvolutionaryMemory(name="test")
        m.add_agent("a1", generation=0, fitness=0.5, genome_snapshot={})
        assert m.total_agents == 1

    def test_update_fitness(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("a1", generation=0, fitness=None, genome_snapshot={})
        m.update_fitness("a1", 0.8)
        top = m.get_top_ancestors(n=1)
        assert top[0]["fitness"] == pytest.approx(0.8)

    def test_get_top_ancestors_sorted(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("a1", generation=0, fitness=0.2, genome_snapshot={})
        m.add_agent("a2", generation=0, fitness=0.9, genome_snapshot={})
        m.add_agent("a3", generation=0, fitness=0.5, genome_snapshot={})
        top = m.get_top_ancestors(n=2)
        assert len(top) == 2
        assert top[0]["fitness"] >= top[1]["fitness"]

    def test_min_fitness_filter(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("a1", generation=0, fitness=0.1, genome_snapshot={})
        m.add_agent("a2", generation=0, fitness=0.8, genome_snapshot={})
        top = m.get_top_ancestors(n=10, min_fitness=0.5)
        assert all(a.get("fitness", 0) >= 0.5 for a in top)

    def test_lineage_single_agent(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("root", generation=0, fitness=0.5, genome_snapshot={})
        lineage = m.get_lineage("root")
        assert "root" in lineage

    def test_lineage_with_parent(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("parent", generation=0, fitness=0.5, genome_snapshot={})
        m.add_agent("child", generation=1, fitness=0.7, genome_snapshot={}, parents=["parent"])
        lineage = m.get_lineage("child")
        assert "parent" in lineage
        assert "child" in lineage

    def test_generation_stats(self) -> None:
        m = EvolutionaryMemory()
        m.add_agent("a1", generation=0, fitness=0.4, genome_snapshot={})
        m.add_agent("a2", generation=0, fitness=0.6, genome_snapshot={})
        m.add_agent("a3", generation=1, fitness=0.8, genome_snapshot={})
        stats = m.generation_stats()
        assert stats[0]["best"] == pytest.approx(0.6)
        assert stats[0]["mean"] == pytest.approx(0.5)
        assert stats[1]["best"] == pytest.approx(0.8)

    def test_json_round_trip(self) -> None:
        m = EvolutionaryMemory(name="roundtrip")
        m.add_agent("a1", generation=0, fitness=0.5, genome_snapshot={"x": 1})
        m.add_agent("a2", generation=1, fitness=0.7, genome_snapshot={}, parents=["a1"])
        json_str = m.to_json()
        m2 = EvolutionaryMemory.from_json(json_str)
        assert m2.name == "roundtrip"
        assert m2.total_agents == 2
