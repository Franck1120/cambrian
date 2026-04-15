# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""tests/test_code_genome.py — Unit tests for Forge mode code evolution.

Tests cover:
- CodeGenome: serialisation, clone, __str__
- CodeAgent: fitness tracking, clone, to_dict, repr
- CodeMutator: _extract_code static method (no LLM needed)
- CodeEvaluator: scoring logic against real sandbox execution
- CodeEvolutionEngine: initialization, tournament, _update_best
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.code_genome import (
    CodeAgent,
    CodeEvaluator,
    CodeEvolutionEngine,
    CodeGenome,
    CodeMutator,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _reverse_genome() -> CodeGenome:
    """A CodeGenome that already has a correct reverse() implementation."""
    return CodeGenome(
        code="def solution(s: str) -> str:\n    return s[::-1]\n",
        entry_point="solution",
        description="reverse a string",
        test_cases=[
            {"input": "hello", "expected": "olleh"},
            {"input": "abc", "expected": "cba"},
            {"input": "", "expected": ""},
        ],
    )


def _mock_backend(response: str = "def solution(s):\n    return s[::-1]") -> MagicMock:
    """Return a MagicMock LLM backend that always returns *response*."""
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# CodeGenome
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeGenome:
    def test_defaults(self) -> None:
        g = CodeGenome()
        assert g.code == ""
        assert g.entry_point == "solution"
        assert g.language == "python"
        assert g.version == 0
        assert g.test_cases == []
        assert len(g.genome_id) == 8

    def test_genome_id_is_unique(self) -> None:
        ids = {CodeGenome().genome_id for _ in range(20)}
        assert len(ids) == 20

    def test_to_dict_round_trip(self) -> None:
        g = CodeGenome(
            code="def f(): pass",
            entry_point="f",
            description="noop",
            version=3,
            test_cases=[{"input": "x", "expected": "y"}],
        )
        restored = CodeGenome.from_dict(g.to_dict())
        assert restored.code == g.code
        assert restored.entry_point == g.entry_point
        assert restored.description == g.description
        assert restored.version == g.version
        assert restored.test_cases == g.test_cases
        assert restored.genome_id == g.genome_id

    def test_from_dict_defaults_on_missing_keys(self) -> None:
        g = CodeGenome.from_dict({})
        assert g.code == ""
        assert g.entry_point == "solution"
        assert g.version == 0

    def test_clone_has_fresh_id(self) -> None:
        g = CodeGenome(code="x = 1")
        clone = g.clone()
        assert clone.genome_id != g.genome_id
        assert clone.code == g.code

    def test_clone_is_deep(self) -> None:
        g = CodeGenome(test_cases=[{"input": "a", "expected": "b"}])
        clone = g.clone()
        clone.test_cases.append({"input": "c", "expected": "d"})
        assert len(g.test_cases) == 1

    def test_str_shows_line_count(self) -> None:
        g = CodeGenome(code="a\nb\nc")
        s = str(g)
        assert "lines=3" in s

    def test_str_empty_code(self) -> None:
        g = CodeGenome()
        assert "lines=0" in str(g)


# ─────────────────────────────────────────────────────────────────────────────
# CodeAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAgent:
    def test_initial_fitness_is_none(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        assert a.fitness is None

    def test_fitness_setter(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        a.fitness = 0.75
        assert a.fitness == pytest.approx(0.75)

    def test_generation_setter(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        a.generation = 5
        assert a.generation == 5

    def test_id_alias(self) -> None:
        a = CodeAgent(genome=CodeGenome(), agent_id="abc12345")
        assert a.id == "abc12345"

    def test_auto_agent_id(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        assert len(a.agent_id) == 8

    def test_clone_clears_fitness(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        a.fitness = 0.9
        a.generation = 3
        clone = a.clone()
        assert clone.fitness is None
        assert clone.generation == 3

    def test_clone_different_id(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        clone = a.clone()
        assert clone.agent_id != a.agent_id

    def test_to_dict_structure(self) -> None:
        a = CodeAgent(genome=CodeGenome(code="x=1"), agent_id="test1234")
        a.fitness = 0.5
        a.generation = 2
        d = a.to_dict()
        assert d["id"] == "test1234"
        assert d["generation"] == 2
        assert d["fitness"] == pytest.approx(0.5)
        assert "genome" in d
        assert d["genome"]["code"] == "x=1"

    def test_repr_contains_fitness(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        a.fitness = 0.1234
        assert "0.1234" in repr(a)

    def test_repr_none_fitness(self) -> None:
        a = CodeAgent(genome=CodeGenome())
        assert "None" in repr(a)


# ─────────────────────────────────────────────────────────────────────────────
# CodeMutator._extract_code
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeMutatorExtract:
    def test_fenced_python_block(self) -> None:
        raw = "Here is the solution:\n```python\ndef f(): pass\n```\nDone."
        result = CodeMutator._extract_code(raw)
        assert result == "def f(): pass"

    def test_fenced_py_block(self) -> None:
        raw = "```py\nx = 1 + 2\n```"
        result = CodeMutator._extract_code(raw)
        assert result == "x = 1 + 2"

    def test_no_fence_returns_stripped(self) -> None:
        raw = "  def g(): return 42  "
        result = CodeMutator._extract_code(raw)
        assert result == "def g(): return 42"

    def test_empty_string(self) -> None:
        assert CodeMutator._extract_code("") == ""


class TestCodeMutatorWithMock:
    def test_mutate_increments_version(self) -> None:
        backend = _mock_backend("def solution(s): return s[::-1]")
        mutator = CodeMutator(backend)
        genome = CodeGenome(code="def solution(s): return s", version=2)
        agent = CodeAgent(genome=genome)
        agent.fitness = 0.5
        child = mutator.mutate(agent, task="reverse string")
        assert child.genome.version == 3
        assert child.fitness is None
        assert child.agent_id != agent.agent_id

    def test_mutate_strips_fence(self) -> None:
        backend = _mock_backend("```python\ndef solution(s): return s[::-1]\n```")
        mutator = CodeMutator(backend)
        agent = CodeAgent(genome=CodeGenome(code="def solution(s): return s"))
        child = mutator.mutate(agent, "reverse")
        assert "```" not in child.genome.code

    def test_mutate_fallback_on_error(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = CodeMutator(backend, fallback_on_error=True)
        agent = CodeAgent(genome=CodeGenome(code="def solution(s): return s"))
        child = mutator.mutate(agent, "task")
        assert child.genome.code == "def solution(s): return s"

    def test_mutate_raises_when_no_fallback(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = CodeMutator(backend, fallback_on_error=False)
        agent = CodeAgent(genome=CodeGenome(code="x=1"))
        with pytest.raises(RuntimeError):
            mutator.mutate(agent, "task")

    def test_crossover_takes_best_parent_as_base(self) -> None:
        backend = _mock_backend("def solution(s): return s[::-1]")
        mutator = CodeMutator(backend)
        g_a = CodeGenome(code="def solution(s): return s", version=1)
        g_b = CodeGenome(code="def solution(s): return s+'!'", version=2)
        a = CodeAgent(genome=g_a)
        a.fitness = 0.3
        b = CodeAgent(genome=g_b)
        b.fitness = 0.9
        child = mutator.crossover(a, b, "reverse string")
        assert child.genome.version == 3  # max(2,1)+1
        assert child.fitness is None

    def test_crossover_version_is_max_plus_one(self) -> None:
        backend = _mock_backend("def solution(s): pass")
        mutator = CodeMutator(backend)
        a = CodeAgent(genome=CodeGenome(code="x=1", version=5))
        a.fitness = 0.6
        b = CodeAgent(genome=CodeGenome(code="y=2", version=3))
        b.fitness = 0.4
        child = mutator.crossover(a, b, "")
        assert child.genome.version == 6


# ─────────────────────────────────────────────────────────────────────────────
# CodeEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeEvaluator:
    def test_empty_code_scores_zero(self) -> None:
        ev = CodeEvaluator()
        agent = CodeAgent(genome=CodeGenome(code=""))
        assert ev.evaluate(agent) == pytest.approx(0.0)

    def test_whitespace_only_scores_zero(self) -> None:
        ev = CodeEvaluator()
        agent = CodeAgent(genome=CodeGenome(code="   \n  "))
        assert ev.evaluate(agent) == pytest.approx(0.0)

    def test_correct_code_all_pass_scores_one(self) -> None:
        ev = CodeEvaluator()
        agent = CodeAgent(genome=_reverse_genome())
        score = ev.evaluate(agent)
        assert score == pytest.approx(1.0)

    def test_partial_pass_in_range(self) -> None:
        # Code that always returns empty string — fails all non-empty inputs
        code = "def solution(s: str) -> str:\n    return ''\n"
        genome = CodeGenome(
            code=code,
            entry_point="solution",
            test_cases=[
                {"input": "hello", "expected": "olleh"},  # fail
                {"input": "", "expected": ""},             # pass
            ],
        )
        ev = CodeEvaluator()
        agent = CodeAgent(genome=genome)
        score = ev.evaluate(agent)
        # 1/2 pass → 0.1 + 0.9 * 0.5 = 0.55
        assert 0.5 < score < 0.7

    def test_syntax_error_scores_low(self) -> None:
        genome = CodeGenome(
            code="def solution(s:\n    pass",
            entry_point="solution",
            test_cases=[{"input": "x", "expected": "x"}],
        )
        ev = CodeEvaluator()
        agent = CodeAgent(genome=genome)
        score = ev.evaluate(agent)
        assert score < 0.2

    def test_no_test_cases_compilable_scores_high(self) -> None:
        """No test cases: compilable code should score ~0.8."""
        genome = CodeGenome(
            code="x = 1 + 2\n",
            entry_point="solution",
            test_cases=[],
        )
        ev = CodeEvaluator()
        agent = CodeAgent(genome=genome)
        score = ev.evaluate(agent)
        assert score == pytest.approx(0.8)

    def test_no_test_cases_broken_code_scores_low(self) -> None:
        genome = CodeGenome(
            code="raise RuntimeError('always fails')",
            entry_point="solution",
            test_cases=[],
        )
        ev = CodeEvaluator()
        agent = CodeAgent(genome=genome)
        score = ev.evaluate(agent)
        assert score < 0.2

    def test_timeout_handled_gracefully(self) -> None:
        genome = CodeGenome(
            code="def solution(s): \n    while True: pass\n",
            entry_point="solution",
            test_cases=[{"input": "x", "expected": "x"}],
        )
        ev = CodeEvaluator(timeout=0.5)
        agent = CodeAgent(genome=genome)
        score = ev.evaluate(agent)
        # Timed-out test case should not count as passed
        assert score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# CodeEvolutionEngine internals
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeEvolutionEngine:
    def _engine(self, pop_size: int = 4, seed: int = 42) -> CodeEvolutionEngine:
        backend = _mock_backend("def solution(s): return s[::-1]")
        return CodeEvolutionEngine(
            backend=backend,
            population_size=pop_size,
            elite_ratio=0.25,
            tournament_k=2,
            timeout=5.0,
            seed=seed,
        )

    def test_initialize_creates_correct_population_size(self) -> None:
        engine = self._engine(pop_size=6)
        genome = CodeGenome(code="def solution(s): return s")
        population = engine._initialize(genome)
        assert len(population) == 6

    def test_initialize_all_have_different_agent_ids(self) -> None:
        engine = self._engine(pop_size=5)
        genome = CodeGenome(code="def solution(s): return s")
        population = engine._initialize(genome)
        ids = [a.agent_id for a in population]
        assert len(set(ids)) == 5

    def test_tournament_returns_best(self) -> None:
        engine = self._engine()
        pop = [CodeAgent(genome=CodeGenome()) for _ in range(5)]
        for i, a in enumerate(pop):
            a.fitness = float(i) * 0.1
        best = engine._tournament(pop)
        assert best.fitness == pytest.approx(0.4)

    def test_update_best_tracks_highest_fitness(self) -> None:
        engine = self._engine()
        pop = [CodeAgent(genome=CodeGenome()) for _ in range(3)]
        pop[0].fitness = 0.2
        pop[1].fitness = 0.9
        pop[2].fitness = 0.5
        engine._update_best(pop)
        assert engine.best is not None
        assert engine.best.fitness == pytest.approx(0.9)

    def test_update_best_does_not_downgrade(self) -> None:
        engine = self._engine()
        pop = [CodeAgent(genome=CodeGenome())]
        pop[0].fitness = 0.8
        engine._update_best(pop)
        pop2 = [CodeAgent(genome=CodeGenome())]
        pop2[0].fitness = 0.3
        engine._update_best(pop2)
        assert engine.best is not None
        assert engine.best.fitness == pytest.approx(0.8)

    def test_update_best_skips_none_fitness(self) -> None:
        engine = self._engine()
        pop = [CodeAgent(genome=CodeGenome())]
        # fitness is None — should not crash
        engine._update_best(pop)
        assert engine.best is None

    def test_elite_n_minimum_one(self) -> None:
        backend = _mock_backend()
        engine = CodeEvolutionEngine(backend=backend, population_size=3, elite_ratio=0.0)
        assert engine._elite_n == 1

    def test_best_property_initially_none(self) -> None:
        engine = self._engine()
        assert engine.best is None

    def test_evolve_returns_code_agent(self) -> None:
        """Integration: evolve runs for 2 gens with correct code — returns best."""
        backend = _mock_backend("def solution(s): return s[::-1]")
        engine = CodeEvolutionEngine(
            backend=backend,
            population_size=3,
            elite_ratio=0.33,
            timeout=5.0,
            seed=0,
        )
        seed = CodeGenome(
            code="def solution(s): return s[::-1]",
            entry_point="solution",
            description="reverse a string",
            test_cases=[
                {"input": "hello", "expected": "olleh"},
                {"input": "ab", "expected": "ba"},
            ],
        )
        best = engine.evolve(seed=seed, task="reverse string", n_generations=2)
        assert isinstance(best, CodeAgent)
        assert best.fitness is not None
        assert best.fitness >= 0.0

    def test_on_generation_callback(self) -> None:
        backend = _mock_backend("def solution(s): return s[::-1]")
        engine = CodeEvolutionEngine(
            backend=backend, population_size=3, timeout=5.0, seed=0
        )
        seed = CodeGenome(
            code="def solution(s): return s[::-1]",
            entry_point="solution",
            test_cases=[{"input": "hi", "expected": "ih"}],
        )
        calls: list[tuple[int, int]] = []

        def cb(gen: int, pop: list[CodeAgent]) -> None:
            calls.append((gen, len(pop)))

        engine.evolve(seed=seed, task="reverse", n_generations=2, on_generation=cb)
        assert len(calls) == 3  # gen 0, 1, 2
        assert calls[0][0] == 0
        assert calls[2][0] == 2

    def test_test_cases_override_seed(self) -> None:
        backend = _mock_backend("def solution(s): return s[::-1]")
        engine = CodeEvolutionEngine(backend=backend, population_size=2, timeout=5.0, seed=0)
        seed = CodeGenome(
            code="def solution(s): return s[::-1]",
            entry_point="solution",
            test_cases=[],
        )
        new_cases = [{"input": "x", "expected": "x"}]
        engine.evolve(seed=seed, task="identity", test_cases=new_cases, n_generations=1)
        # After evolve, seed.test_cases should be overridden
        assert seed.test_cases == new_cases
