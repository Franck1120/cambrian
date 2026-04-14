"""Edge case and boundary condition tests for Cambrian.

Covers: genome serialisation corner cases, cache boundary conditions,
compress on unusual inputs, mutator fallbacks, evolution with extreme
parameters, memory graph edge cases, and sandbox error handling.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cambrian.agent import Agent, Genome
from cambrian.cache import SemanticCache
from cambrian.compress import caveman_compress, procut_prune
from cambrian.diversity import MAPElites
from cambrian.memory import EvolutionaryMemory
from cambrian.mutator import LLMMutator
from cambrian.router import ModelRouter
from cambrian.utils.sandbox import SandboxResult, run_in_sandbox, extract_python_code


# ── Genome edge cases ─────────────────────────────────────────────────────────

class TestGenomeEdgeCases:
    def test_empty_system_prompt(self) -> None:
        g = Genome(system_prompt="")
        assert g.system_prompt == ""
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.system_prompt == ""

    def test_very_long_system_prompt(self) -> None:
        long = "word " * 5000
        g = Genome(system_prompt=long)
        assert g.token_count() == len(long) // 4

    def test_unicode_system_prompt_round_trip(self) -> None:
        g = Genome(system_prompt="Ciao! こんにちは 🌍")
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.system_prompt == g.system_prompt

    def test_temperature_boundary_values(self) -> None:
        for temp in (0.0, 0.1, 1.0, 1.5, 2.0):
            g = Genome(system_prompt="x", temperature=temp)
            assert g.temperature == temp

    def test_empty_tools_list(self) -> None:
        g = Genome(system_prompt="x", tools=[])
        assert g.tools == []
        assert g.to_dict()["tools"] == []

    def test_many_tools(self) -> None:
        tools = [f"tool_{i}" for i in range(50)]
        g = Genome(system_prompt="x", tools=tools)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.tools == tools

    def test_from_dict_preserves_genome_id(self) -> None:
        g = Genome(system_prompt="x", genome_id="abc12345")
        g2 = Genome.from_dict(g.to_dict())
        assert g2.genome_id == "abc12345"

    def test_json_serialisation_valid_json(self) -> None:
        g = Genome(system_prompt="Check JSON: {\"key\": \"value\"}")
        json.dumps(g.to_dict())  # must not raise


# ── Agent edge cases ──────────────────────────────────────────────────────────

class TestAgentEdgeCases:
    def test_clone_chain(self) -> None:
        """Cloning a clone should produce independent objects."""
        a = Agent(genome=Genome(system_prompt="original"))
        b = a.clone()
        c = b.clone()
        assert a.id != b.id != c.id
        c.genome = Genome(system_prompt="modified")
        assert a.genome.system_prompt == "original"
        assert b.genome.system_prompt == "original"

    def test_fitness_setter_coerces_to_float(self) -> None:
        # Use the public property setter which coerces to float
        a = Agent(genome=Genome(system_prompt="x"))
        a.fitness = 1  # int passed through the setter
        assert isinstance(a.fitness, float)
        assert a.fitness == pytest.approx(1.0)

    def test_to_dict_fitness_none(self) -> None:
        a = Agent(genome=Genome(system_prompt="x"))
        d = a.to_dict()
        assert d["fitness"] is None

    def test_to_dict_round_trip(self) -> None:
        a = Agent(genome=Genome(system_prompt="hello"))
        a._fitness = 0.75
        d = a.to_dict()
        assert d["id"] == a.agent_id
        assert d["fitness"] == pytest.approx(0.75)
        assert "genome" in d


# ── SemanticCache edge cases ──────────────────────────────────────────────────

class TestCacheEdgeCases:
    def test_max_size_one(self) -> None:
        c = SemanticCache(max_size=1)
        c.set("a", "va")
        c.set("b", "vb")  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == "vb"

    def test_empty_prompt_key(self) -> None:
        c = SemanticCache()
        c.set("", "empty prompt response")
        assert c.get("") == "empty prompt response"

    def test_same_prompt_different_models(self) -> None:
        c = SemanticCache()
        c.set("hello", "from mini", model="gpt-4o-mini")
        c.set("hello", "from big", model="gpt-4o")
        assert c.get("hello", model="gpt-4o-mini") == "from mini"
        assert c.get("hello", model="gpt-4o") == "from big"

    def test_overwrite_existing_key(self) -> None:
        c = SemanticCache()
        c.set("p", "v1")
        c.set("p", "v2")  # overwrite
        assert c.get("p") == "v2"
        assert c.size == 1

    def test_hit_rate_all_misses(self) -> None:
        c = SemanticCache()
        c.get("a")
        c.get("b")
        assert c.hit_rate() == pytest.approx(0.0)

    def test_hit_rate_all_hits(self) -> None:
        c = SemanticCache()
        c.set("p", "r")
        c.get("p")
        c.get("p")
        assert c.hit_rate() == pytest.approx(1.0)


# ── Compress edge cases ───────────────────────────────────────────────────────

class TestCompressEdgeCases:
    def test_caveman_only_stopwords(self) -> None:
        result = caveman_compress("the a an is are")
        assert result == ""

    def test_caveman_preserves_numbers(self) -> None:
        result = caveman_compress("calculate 42 items")
        assert "42" in result

    def test_caveman_punctuation_adjacent_stopword(self) -> None:
        result = caveman_compress("analyze, the data")
        # "the" should be stripped; "analyze" and "data" should remain
        words = result.split()
        assert "the" not in words

    def test_procut_empty_prompt(self) -> None:
        g = Genome(system_prompt="")
        result = procut_prune(g, max_tokens=256)
        assert result.system_prompt == ""

    def test_procut_single_word(self) -> None:
        g = Genome(system_prompt="word")
        result = procut_prune(g, max_tokens=1)
        assert result.system_prompt == "word"

    def test_procut_exact_budget(self) -> None:
        # Prompt that is exactly at the token budget
        prompt = "x" * (256 * 4)  # exactly 256 tokens
        g = Genome(system_prompt=prompt)
        result = procut_prune(g, max_tokens=256)
        assert result.system_prompt == prompt  # no pruning needed


# ── Router edge cases ─────────────────────────────────────────────────────────

class TestRouterEdgeCases:
    def test_empty_task_routes_cheap(self) -> None:
        r = ModelRouter()
        model = r.route("")
        assert model == r._cheap

    def test_exactly_at_cheap_threshold(self) -> None:
        r = ModelRouter()
        # 100 tokens → boundary between cheap and medium
        task = "word " * 100
        # May route to cheap or medium; just must not raise
        model = r.route(task)
        assert model in (r._cheap, r._medium, r._premium)

    def test_routing_log_truncates_task(self) -> None:
        r = ModelRouter()
        long_task = "x" * 200
        r.route(long_task)
        snippet, _ = r.routing_log[0]
        assert len(snippet) <= 60

    def test_all_three_tiers_reachable(self) -> None:
        r = ModelRouter(cheap_model="c", medium_model="m", premium_model="p")
        # cheap: < 100 tokens, no complexity signals
        r.route("hi")
        # medium: 100–500 tokens (200 tokens)
        r.route("word " * 200)
        # premium: > 500 tokens (600 tokens)
        r.route("word " * 600)
        stats = r.routing_stats()
        assert "c" in stats
        assert "m" in stats
        assert "p" in stats


# ── MAPElites edge cases ──────────────────────────────────────────────────────

class TestMAPElitesEdgeCases:
    def _agent(self, prompt: str = "x", temp: float = 0.5, fitness: float = 0.5) -> Agent:
        a = Agent(genome=Genome(system_prompt=prompt, temperature=temp))
        a._fitness = fitness
        return a

    def test_1x1_grid(self) -> None:
        me = MAPElites(n_prompt_buckets=1, n_temp_buckets=1)
        me.add(self._agent(fitness=0.3))
        me.add(self._agent(fitness=0.9))
        me.add(self._agent(fitness=0.1))
        assert me.occupancy == 1
        assert me.best().fitness == pytest.approx(0.9)  # type: ignore[union-attr]

    def test_same_fitness_no_replacement(self) -> None:
        me = MAPElites()
        a1 = self._agent("a", fitness=0.5)
        a2 = self._agent("a", fitness=0.5)
        me.add(a1)
        inserted = me.add(a2)
        # Same fitness → not strictly better → should not replace
        assert not inserted

    def test_coverage_full_grid(self) -> None:
        # _PROMPT_LENGTH_BUCKETS: < 100 tokens → bucket 0, 100-300 → bucket 1
        # "s " * 500 = 1000 chars ≈ 250 tokens → bucket 1 (clamped to min(1, 1))
        # _TEMPERATURE_BUCKETS: < 0.4 → cold (0), >= 0.8 → hot (clamped to 1 in 2-bucket grid)
        me = MAPElites(n_prompt_buckets=2, n_temp_buckets=2)
        me.add(self._agent("s", temp=0.1, fitness=0.5))           # short, cold → (0, 0)
        me.add(self._agent("s " * 500, temp=0.1, fitness=0.5))    # long, cold  → (1, 0)
        me.add(self._agent("s", temp=1.0, fitness=0.5))           # short, hot  → (0, 1)
        me.add(self._agent("s " * 500, temp=1.0, fitness=0.5))    # long, hot   → (1, 1)
        assert me.coverage() == pytest.approx(1.0)


# ── EvolutionaryMemory edge cases ─────────────────────────────────────────────

class TestMemoryEdgeCases:
    def test_update_nonexistent_agent(self) -> None:
        m = EvolutionaryMemory()
        # Should not raise — just silently ignored
        m.update_fitness("ghost", 0.9)

    def test_lineage_nonexistent_agent(self) -> None:
        m = EvolutionaryMemory()
        lineage = m.get_lineage("ghost")
        assert lineage == ["ghost"]

    def test_get_top_ancestors_empty(self) -> None:
        m = EvolutionaryMemory()
        assert m.get_top_ancestors() == []

    def test_generation_stats_empty(self) -> None:
        m = EvolutionaryMemory()
        assert m.generation_stats() == {}

    def test_deep_lineage(self) -> None:
        m = EvolutionaryMemory()
        prev = "a0"
        m.add_agent(prev, generation=0, fitness=0.1, genome_snapshot={})
        for i in range(1, 10):
            agent_id = f"a{i}"
            m.add_agent(agent_id, generation=i, fitness=float(i) / 10,
                        genome_snapshot={}, parents=[prev])
            prev = agent_id
        lineage = m.get_lineage("a9")
        assert "a0" in lineage
        assert "a9" in lineage

    def test_json_empty_memory(self) -> None:
        m = EvolutionaryMemory(name="empty")
        j = m.to_json()
        m2 = EvolutionaryMemory.from_json(j)
        assert m2.name == "empty"
        assert m2.total_agents == 0


# ── Sandbox edge cases ────────────────────────────────────────────────────────

class TestSandboxEdgeCases:
    def test_successful_execution(self) -> None:
        result = run_in_sandbox("print('hello')", timeout=5)
        assert result.success
        assert "hello" in result.stdout

    def test_syntax_error(self) -> None:
        result = run_in_sandbox("def broken(:", timeout=5)
        assert not result.success
        assert result.returncode != 0

    def test_runtime_error(self) -> None:
        result = run_in_sandbox("1 / 0", timeout=5)
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    def test_timeout(self) -> None:
        result = run_in_sandbox("while True: pass", timeout=1.0)
        assert result.timed_out
        assert not result.success
        assert result.returncode == -1

    def test_multiline_output(self) -> None:
        code = "\n".join(f"print({i})" for i in range(5))
        result = run_in_sandbox(code, timeout=5)
        assert result.success
        assert result.stdout.strip() == "0\n1\n2\n3\n4"

    def test_extract_python_code_with_fence(self) -> None:
        text = "Here is the code:\n```python\nprint('hi')\n```\nDone."
        extracted = extract_python_code(text)
        assert extracted.strip() == "print('hi')"

    def test_extract_python_code_no_fence(self) -> None:
        text = "print('hello world')"
        extracted = extract_python_code(text)
        assert "print" in extracted

    def test_extract_python_code_py_shorthand(self) -> None:
        text = "```py\nx = 1\n```"
        extracted = extract_python_code(text)
        assert "x = 1" in extracted

    def test_sandbox_result_success_property(self) -> None:
        ok = SandboxResult(stdout="ok", stderr="", returncode=0, timed_out=False)
        assert ok.success
        fail = SandboxResult(stdout="", stderr="err", returncode=1, timed_out=False)
        assert not fail.success
        timeout = SandboxResult(stdout="", stderr="", returncode=-1, timed_out=True)
        assert not timeout.success


# ── LLMMutator edge cases ─────────────────────────────────────────────────────

class _BadBackend:
    """Backend that always raises."""
    model_name = "bad"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("simulated network failure")


class _JsonBackend:
    """Backend that returns valid genome JSON."""
    model_name = "json"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import re
        m = re.search(r"\{[\s\S]+\}", prompt)
        return m.group(0) if m else json.dumps(Genome(system_prompt="fallback").to_dict())


class TestMutatorEdgeCases:
    def _genome(self, prompt: str = "test") -> Genome:
        return Genome(system_prompt=prompt, temperature=0.7)

    def test_fallback_on_error_true(self) -> None:
        mutator = LLMMutator(backend=_BadBackend(), fallback_on_error=True)  # type: ignore
        agent = Agent(genome=self._genome("original"))
        mutated = mutator.mutate(agent, task="test")
        # Should not raise; returns genome with tweaked temperature
        assert mutated.genome is not None

    def test_fallback_on_error_false_raises(self) -> None:
        mutator = LLMMutator(backend=_BadBackend(), fallback_on_error=False)  # type: ignore
        agent = Agent(genome=self._genome("original"))
        with pytest.raises(RuntimeError):
            mutator.mutate(agent, task="test")

    def test_mutate_resets_fitness(self) -> None:
        mutator = LLMMutator(backend=_JsonBackend(), fallback_on_error=True)  # type: ignore
        agent = Agent(genome=self._genome())
        agent._fitness = 0.9
        mutated = mutator.mutate(agent, task="test")
        assert mutated.fitness is None

    def test_crossover_fallback_on_error(self) -> None:
        mutator = LLMMutator(backend=_BadBackend(), fallback_on_error=True)  # type: ignore
        a = Agent(genome=self._genome("parent A"))
        a._fitness = 0.7
        b = Agent(genome=self._genome("parent B with more text here"))
        b._fitness = 0.4
        child = mutator.crossover(a, b, task="test")
        assert child.genome is not None
        assert child.fitness is None

    def test_parse_genome_invalid_json_returns_fallback(self) -> None:
        fallback = self._genome("fallback prompt")
        result = LLMMutator._parse_genome("not json at all {{{{", fallback)
        assert result.system_prompt == fallback.system_prompt

    def test_parse_genome_from_markdown_fence(self) -> None:
        g = self._genome("improved prompt")
        json_str = json.dumps(g.to_dict())
        raw = f"Here's the improved genome:\n```json\n{json_str}\n```"
        result = LLMMutator._parse_genome(raw, self._genome("old"))
        assert result.system_prompt == "improved prompt"

    def test_random_tweak_clamps_temperature(self) -> None:
        g = Genome(system_prompt="x", temperature=0.1)
        for _ in range(50):
            tweaked = LLMMutator._random_tweak(g)
            assert 0.1 <= tweaked.temperature <= 1.5

    def test_deterministic_crossover_combines_prompts(self) -> None:
        g_a = Genome(system_prompt="Sentence one. Sentence two.")
        g_b = Genome(system_prompt="Alt one. Alt two.")
        child = LLMMutator._deterministic_crossover(g_a, g_b)
        assert child.system_prompt != ""
        assert child.temperature == pytest.approx((g_a.temperature + g_b.temperature) / 2)
