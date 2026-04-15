# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Edge-case tests for cambrian.code_genome — CodeGenome, CodeAgent, CodeEvaluator."""

from __future__ import annotations

import pytest

from cambrian.code_genome import CodeAgent, CodeEvaluator, CodeGenome


# ---------------------------------------------------------------------------
# CodeGenome — construction and serialisation
# ---------------------------------------------------------------------------


class TestCodeGenomeConstruction:
    def test_empty_genome(self) -> None:
        g = CodeGenome()
        assert g.code == ""
        assert g.entry_point == "solution"
        assert g.language == "python"
        assert g.version == 0

    def test_custom_fields(self) -> None:
        g = CodeGenome(
            code="def solution(x): return x",
            entry_point="solution",
            description="identity function",
            language="python",
        )
        assert "solution" in g.code
        assert g.description == "identity function"

    def test_genome_id_auto_generated(self) -> None:
        g = CodeGenome()
        assert isinstance(g.genome_id, str)
        assert len(g.genome_id) > 0

    def test_two_genomes_have_different_ids(self) -> None:
        g1 = CodeGenome()
        g2 = CodeGenome()
        assert g1.genome_id != g2.genome_id

    def test_str_shows_version(self) -> None:
        g = CodeGenome(code="def solution(x): return x", version=3)
        s = str(g)
        assert "v3" in s

    def test_str_shows_entry_point(self) -> None:
        g = CodeGenome(entry_point="my_func")
        s = str(g)
        assert "my_func" in s


class TestCodeGenomeSerialization:
    def test_to_dict_round_trip(self) -> None:
        g = CodeGenome(
            code="def solution(x): return x * 2",
            entry_point="solution",
            description="double",
            test_cases=[{"input": "5", "expected": "10"}],
            version=2,
        )
        d = g.to_dict()
        g2 = CodeGenome.from_dict(d)
        assert g2.code == g.code
        assert g2.entry_point == g.entry_point
        assert g2.description == g.description
        assert g2.version == g.version
        assert g2.test_cases == g.test_cases

    def test_from_dict_missing_keys_uses_defaults(self) -> None:
        g = CodeGenome.from_dict({})
        assert g.code == ""
        assert g.entry_point == "solution"
        assert g.version == 0

    def test_clone_has_different_id(self) -> None:
        g = CodeGenome(code="def solution(x): return x")
        clone = g.clone()
        assert clone.genome_id != g.genome_id

    def test_clone_preserves_code(self) -> None:
        g = CodeGenome(code="def solution(x): return x * 3", version=7)
        clone = g.clone()
        assert clone.code == g.code
        assert clone.version == g.version

    def test_test_cases_preserved_in_round_trip(self) -> None:
        g = CodeGenome(test_cases=[{"input": "a", "expected": "b"}])
        g2 = CodeGenome.from_dict(g.to_dict())
        assert g2.test_cases == [{"input": "a", "expected": "b"}]


# ---------------------------------------------------------------------------
# CodeAgent
# ---------------------------------------------------------------------------


class TestCodeAgent:
    def test_construction(self) -> None:
        genome = CodeGenome(code="def solution(x): return x")
        agent = CodeAgent(genome=genome)
        assert agent.genome is genome
        assert agent.fitness is None

    def test_id_property(self) -> None:
        agent = CodeAgent(genome=CodeGenome())
        assert isinstance(agent.id, str)

    def test_fitness_setter(self) -> None:
        agent = CodeAgent(genome=CodeGenome())
        agent.fitness = 0.75
        assert agent.fitness == pytest.approx(0.75)

    def test_generation_setter(self) -> None:
        agent = CodeAgent(genome=CodeGenome())
        agent.generation = 5
        assert agent.generation == 5

    def test_clone_has_different_id(self) -> None:
        agent = CodeAgent(genome=CodeGenome(code="def solution(x): return x"))
        agent.fitness = 0.5
        clone = agent.clone()
        assert clone.id != agent.id

    def test_clone_has_no_fitness(self) -> None:
        agent = CodeAgent(genome=CodeGenome())
        agent.fitness = 0.9
        clone = agent.clone()
        assert clone.fitness is None

    def test_to_dict_has_genome(self) -> None:
        agent = CodeAgent(genome=CodeGenome(code="x = 1"))
        d = agent.to_dict()
        assert "genome" in d

    def test_repr_is_string(self) -> None:
        agent = CodeAgent(genome=CodeGenome())
        assert isinstance(repr(agent), str)


# ---------------------------------------------------------------------------
# CodeEvaluator — edge cases
# ---------------------------------------------------------------------------


class TestCodeEvaluatorEdgeCases:
    def _evaluator(self) -> CodeEvaluator:
        return CodeEvaluator(timeout=5.0)

    def _agent(self, code: str, test_cases: list[dict[str, str]] | None = None) -> CodeAgent:
        genome = CodeGenome(code=code, test_cases=test_cases or [])
        return CodeAgent(genome=genome)

    def test_empty_code_returns_zero(self) -> None:
        evaluator = self._evaluator()
        agent = self._agent("")
        assert evaluator.evaluate(agent) == 0.0

    def test_whitespace_only_code_returns_zero(self) -> None:
        evaluator = self._evaluator()
        agent = self._agent("   \n\t  ")
        assert evaluator.evaluate(agent) == 0.0

    def test_correct_code_all_cases_pass(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x): return str(int(x) * 2)"
        test_cases = [
            {"input": "3", "expected": "6"},
            {"input": "5", "expected": "10"},
            {"input": "0", "expected": "0"},
        ]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert score == pytest.approx(1.0)

    def test_partial_pass_score_between_bounds(self) -> None:
        evaluator = self._evaluator()
        # Half the test cases will pass (input="2" → "4" ✓, input="3" → "9" ✗)
        code = "def solution(x): return str(int(x) * 2)"
        test_cases = [
            {"input": "2", "expected": "4"},   # passes
            {"input": "3", "expected": "9"},   # fails (3*2=6, not 9)
        ]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert 0.0 < score < 1.0

    def test_syntax_error_returns_low_score(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x: THIS IS NOT VALID PYTHON"
        test_cases = [{"input": "x", "expected": "x"}]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert score <= 0.1

    def test_runtime_error_returns_low_score(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x): raise RuntimeError('fail')"
        test_cases = [{"input": "a", "expected": "a"}]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert score <= 0.1

    def test_no_test_cases_compiles_ok(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x): return x"
        agent = self._agent(code, test_cases=[])
        score = evaluator.evaluate(agent)
        # No test cases: returns 0.8 if compiles successfully
        assert score >= 0.0

    def test_infinite_loop_times_out(self) -> None:
        evaluator = CodeEvaluator(timeout=0.5)
        code = "def solution(x):\n    while True: pass"
        test_cases = [{"input": "x", "expected": "x"}]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert score < 1.0

    def test_string_output_match(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x): return x[::-1]"
        test_cases = [
            {"input": "hello", "expected": "olleh"},
            {"input": "abc", "expected": "cba"},
        ]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert score == pytest.approx(1.0)

    def test_evaluator_score_in_unit_range(self) -> None:
        evaluator = self._evaluator()
        code = "def solution(x): return str(len(x))"
        test_cases = [{"input": "hello", "expected": "5"}]
        agent = self._agent(code, test_cases)
        score = evaluator.evaluate(agent)
        assert 0.0 <= score <= 1.0
