"""Tests for cambrian.code_genome — CodeGenome, CodeEvaluator, CodeMutator, CodeEvolutionEngine."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from cambrian.code_genome import (
    CodeEvaluationResult,
    CodeEvaluator,
    CodeEvolutionEngine,
    CodeGenome,
    CodeMutator,
    TestCase,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _hello_genome() -> CodeGenome:
    return CodeGenome(code='print("hello")', description="print hello")


def _add_genome() -> CodeGenome:
    return CodeGenome(
        code="a, b = map(int, input().split()); print(a + b)",
        description="sum two integers from stdin",
    )


def _simple_test_cases() -> list[TestCase]:
    return [TestCase(input_data="", expected_output="hello")]


def _add_test_cases() -> list[TestCase]:
    return [
        TestCase(input_data="3 4", expected_output="7"),
        TestCase(input_data="10 20", expected_output="30"),
        TestCase(input_data="-1 1", expected_output="0"),
    ]


# ── TestCase ───────────────────────────────────────────────────────────────────


class TestTestCase:
    def test_auto_label(self) -> None:
        tc = TestCase(input_data="x", expected_output="y")
        assert tc.label.startswith("test_")

    def test_explicit_label(self) -> None:
        tc = TestCase(input_data="x", expected_output="y", label="my_label")
        assert tc.label == "my_label"

    def test_default_weight(self) -> None:
        tc = TestCase(input_data="", expected_output="")
        assert tc.weight == pytest.approx(1.0)


# ── CodeGenome ─────────────────────────────────────────────────────────────────


class TestCodeGenome:
    def test_loc_counts_code_lines(self) -> None:
        cg = CodeGenome(code="a = 1\n# comment\nb = 2\n\nc = 3")
        assert cg.loc() == 3

    def test_loc_empty(self) -> None:
        assert CodeGenome(code="").loc() == 0

    def test_roundtrip(self) -> None:
        cg = CodeGenome(code="x=1", description="set x", version=2, metadata={"k": "v"})
        r = CodeGenome.from_dict(cg.to_dict())
        assert r.code == "x=1"
        assert r.description == "set x"
        assert r.version == 2
        assert r.metadata == {"k": "v"}

    def test_genome_id_generated(self) -> None:
        g1 = CodeGenome()
        g2 = CodeGenome()
        assert g1.genome_id != g2.genome_id

    def test_str(self) -> None:
        cg = CodeGenome(code="print(1)", description="test")
        assert "CodeGenome" in str(cg)

    def test_parent_id_default_empty(self) -> None:
        assert CodeGenome().parent_id == ""


# ── CodeEvaluationResult ───────────────────────────────────────────────────────


class TestCodeEvaluationResult:
    def test_success_true(self) -> None:
        r = CodeEvaluationResult(
            pass_rate=1.0, runtime_s=0.1, loc=5, passed=3, total=3, fitness=1.0
        )
        assert r.success

    def test_success_false_not_all_pass(self) -> None:
        r = CodeEvaluationResult(
            pass_rate=0.5, runtime_s=0.1, loc=5, passed=1, total=2, fitness=0.5
        )
        assert not r.success

    def test_success_false_zero_total(self) -> None:
        r = CodeEvaluationResult(
            pass_rate=0.0, runtime_s=0.0, loc=0, passed=0, total=0, fitness=0.0
        )
        assert not r.success

    def test_repr(self) -> None:
        r = CodeEvaluationResult(
            pass_rate=0.8, runtime_s=0.1, loc=10, passed=2, total=3, fitness=0.8
        )
        assert "CodeEvaluationResult" in repr(r)


# ── CodeEvaluator ──────────────────────────────────────────────────────────────


class TestCodeEvaluator:
    def test_hello_passes(self) -> None:
        ev = CodeEvaluator(_simple_test_cases())
        result = ev.evaluate(_hello_genome())
        assert result.passed == 1
        assert result.pass_rate == pytest.approx(1.0)
        assert result.fitness > 0.9

    def test_wrong_output_fails(self) -> None:
        ev = CodeEvaluator(_simple_test_cases())
        genome = CodeGenome(code='print("world")', description="test")
        result = ev.evaluate(genome)
        assert result.passed == 0
        assert result.pass_rate == pytest.approx(0.0)

    def test_add_program(self) -> None:
        ev = CodeEvaluator(_add_test_cases())
        result = ev.evaluate(_add_genome())
        assert result.passed == 3
        assert result.pass_rate == pytest.approx(1.0)

    def test_partial_pass(self) -> None:
        test_cases = [
            TestCase(input_data="3 4", expected_output="7"),
            TestCase(input_data="a b", expected_output="10"),  # will fail
        ]
        ev = CodeEvaluator(test_cases)
        result = ev.evaluate(_add_genome())
        assert result.passed == 1
        assert 0.0 < result.pass_rate < 1.0

    def test_empty_code_fitness_zero(self) -> None:
        ev = CodeEvaluator(_simple_test_cases())
        result = ev.evaluate(CodeGenome(code="", description="test"))
        assert result.fitness == pytest.approx(0.0)
        assert result.error == "Empty code"

    def test_crash_is_handled(self) -> None:
        ev = CodeEvaluator(_simple_test_cases())
        genome = CodeGenome(code="raise ValueError('boom')", description="test")
        result = ev.evaluate(genome)
        assert result.passed == 0
        assert result.error != ""

    def test_raises_on_empty_test_cases(self) -> None:
        with pytest.raises(ValueError, match="test_cases"):
            CodeEvaluator([])

    def test_fitness_in_range(self) -> None:
        ev = CodeEvaluator(_add_test_cases())
        result = ev.evaluate(_add_genome())
        assert 0.0 <= result.fitness <= 1.0

    def test_weighted_test_cases(self) -> None:
        tcs = [
            TestCase(input_data="", expected_output="hello", weight=3.0),
            TestCase(input_data="", expected_output="world", weight=1.0),
        ]
        ev = CodeEvaluator(tcs)
        # 'hello' genome passes first (weight 3) but not second (weight 1)
        result = ev.evaluate(_hello_genome())
        assert result.pass_rate == pytest.approx(3.0 / 4.0)

    def test_loc_reported(self) -> None:
        ev = CodeEvaluator(_simple_test_cases())
        result = ev.evaluate(_hello_genome())
        assert result.loc >= 1


# ── CodeMutator ────────────────────────────────────────────────────────────────


class TestCodeMutator:
    def _mock_backend(self, return_code: str) -> MagicMock:
        backend = MagicMock()
        backend.generate.return_value = return_code
        return backend

    def test_seed_empty_genome(self) -> None:
        backend = self._mock_backend('print("seeded")')
        mutator = CodeMutator(backend=backend)
        child = mutator.mutate(CodeGenome(code="", description="print something"))
        assert child.code == 'print("seeded")'
        assert child.version == 1

    def test_rewrite_existing(self) -> None:
        backend = self._mock_backend('print("improved")')
        mutator = CodeMutator(backend=backend)
        parent = CodeGenome(code='print("old")', description="test", version=1)
        child = mutator.mutate(parent)
        assert child.code == 'print("improved")'
        assert child.version == 2
        assert child.parent_id == parent.genome_id

    def test_version_incremented(self) -> None:
        backend = self._mock_backend("x = 1")
        mutator = CodeMutator(backend=backend)
        g = CodeGenome(code="y = 2", description="t", version=5)
        assert mutator.mutate(g).version == 6

    def test_fallback_on_error(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = CodeMutator(backend=backend, fallback_on_error=True)
        g = CodeGenome(code="print(1)", description="t")
        child = mutator.mutate(g)
        assert child.code == "print(1)"  # original preserved

    def test_strips_markdown_fences(self) -> None:
        backend = self._mock_backend("```python\nprint(42)\n```")
        mutator = CodeMutator(backend=backend)
        g = CodeGenome(code="print(1)", description="t")
        child = mutator.mutate(g)
        assert "```" not in child.code
        assert "print(42)" in child.code

    def test_raises_when_fallback_false(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = CodeMutator(backend=backend, fallback_on_error=False)
        g = CodeGenome(code="print(1)", description="t")
        with pytest.raises(RuntimeError):
            mutator.mutate(g)

    def test_clean_code_no_fence(self) -> None:
        raw = "x = 1\nprint(x)"
        assert CodeMutator._clean_code(raw) == raw

    def test_evaluation_notes_used(self) -> None:
        backend = self._mock_backend("print(1)")
        mutator = CodeMutator(backend=backend)
        g = CodeGenome(code="print(0)", description="t")
        eval_result = CodeEvaluationResult(
            pass_rate=0.5, runtime_s=0.1, loc=1, passed=1, total=2,
            error="Wrong output", fitness=0.5
        )
        child = mutator.mutate(g, evaluation=eval_result)
        assert child is not None
        # ensure backend was called with something containing the notes
        call_args = backend.generate.call_args[0][0]
        assert "Wrong output" in call_args or "Error" in call_args


# ── CodeEvolutionEngine ────────────────────────────────────────────────────────


class TestCodeEvolutionEngine:
    def _make_engine(self, code: str = 'print("hello")') -> CodeEvolutionEngine:
        backend = MagicMock()
        backend.generate.return_value = code
        return CodeEvolutionEngine(backend=backend, population_size=4, elite_ratio=0.25)

    def test_returns_code_genome(self) -> None:
        engine = self._make_engine()
        seed = CodeGenome(description="print hello")
        tcs = [TestCase(input_data="", expected_output="hello")]
        best = engine.evolve(seed, "print hello", n_generations=2, test_cases=tcs)
        assert isinstance(best, CodeGenome)

    def test_best_has_description(self) -> None:
        engine = self._make_engine()
        seed = CodeGenome(description="my task")
        tcs = [TestCase(input_data="", expected_output="hello")]
        best = engine.evolve(seed, "my task", n_generations=1, test_cases=tcs)
        assert best.description == "my task"

    def test_raises_without_test_cases_or_evaluator(self) -> None:
        engine = self._make_engine()
        seed = CodeGenome(description="test")
        with pytest.raises(ValueError, match="test_cases"):
            engine.evolve(seed, "test", n_generations=1)

    def test_on_generation_callback(self) -> None:
        engine = self._make_engine()
        seed = CodeGenome(description="t")
        tcs = [TestCase(input_data="", expected_output="hello")]
        calls: list[int] = []
        engine.evolve(
            seed, "t", n_generations=3, test_cases=tcs,
            on_generation=lambda g, pop, best: calls.append(g),
        )
        assert len(calls) == 3

    def test_early_exit_on_perfect(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = 'print("hello")'
        ev = CodeEvaluator([TestCase(input_data="", expected_output="hello")])
        engine = CodeEvolutionEngine(backend=backend, evaluator=ev, population_size=3)
        seed = CodeGenome(code='print("hello")', description="t")
        # Should exit after 1 generation (perfect fitness)
        best = engine.evolve(seed, "t", n_generations=10)
        assert isinstance(best, CodeGenome)

    def test_evaluator_at_construction(self) -> None:
        ev = CodeEvaluator([TestCase(input_data="", expected_output="hello")])
        backend = MagicMock()
        backend.generate.return_value = 'print("hello")'
        engine = CodeEvolutionEngine(backend=backend, evaluator=ev, population_size=3)
        seed = CodeGenome(code='print("hello")', description="t")
        best = engine.evolve(seed, "t", n_generations=2)
        assert isinstance(best, CodeGenome)

    def test_population_size_respected(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = 'print("x")'
        engine = CodeEvolutionEngine(backend=backend, population_size=5)
        seed = CodeGenome(description="t")
        tcs = [TestCase(input_data="", expected_output="x")]
        # Just checking it runs without error
        engine.evolve(seed, "t", n_generations=2, test_cases=tcs)
        assert backend.generate.called
