"""tests/test_pipeline.py — Unit tests for Pipeline evolution (Forge mode).

Covers:
- PipelineStep: serialisation, clone, repr
- Pipeline: construction, serialisation, clone, fitness
- PipelineMutator._parse_pipeline: JSON parsing, fence stripping, fallback
- PipelineMutator: mutate/crossover with mock backend
- PipelineEvaluator: exact-match scoring, LLM-judge scoring
- PipelineEvolutionEngine: initialization, tournament, update_best, evolve
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from cambrian.pipeline import (
    Pipeline,
    PipelineEvaluator,
    PipelineEvolutionEngine,
    PipelineMutator,
    PipelineStep,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_backend(response: str = "0.8") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _three_step_pipeline() -> Pipeline:
    return Pipeline(
        name="test-pipe",
        steps=[
            PipelineStep(name="plan", system_prompt="Plan.", role="extractor"),
            PipelineStep(name="solve", system_prompt="Solve.", role="transformer"),
            PipelineStep(name="review", system_prompt="Review.", role="validator"),
        ],
    )


def _pipeline_json(name: str = "p", n_steps: int = 2) -> str:
    steps = [
        {"name": f"step{i}", "system_prompt": f"Step {i}.", "role": "transformer", "temperature": 0.7}
        for i in range(n_steps)
    ]
    return json.dumps({"name": name, "steps": steps})


# ─────────────────────────────────────────────────────────────────────────────
# PipelineStep
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineStep:
    def test_defaults(self) -> None:
        s = PipelineStep()
        assert s.name == "step"
        assert s.role == "transformer"
        assert 0.0 <= s.temperature <= 2.0

    def test_to_dict_round_trip(self) -> None:
        s = PipelineStep(name="foo", system_prompt="bar", role="extractor", temperature=0.3)
        restored = PipelineStep.from_dict(s.to_dict())
        assert restored.name == s.name
        assert restored.system_prompt == s.system_prompt
        assert restored.role == s.role
        assert restored.temperature == pytest.approx(s.temperature)

    def test_from_dict_defaults_on_missing(self) -> None:
        s = PipelineStep.from_dict({})
        assert s.name == "step"
        assert s.role == "transformer"
        assert s.temperature == pytest.approx(0.7)

    def test_clone_is_independent(self) -> None:
        s = PipelineStep(name="orig")
        clone = s.clone()
        clone.name = "mutated"
        assert s.name == "orig"

    def test_repr_contains_name_and_role(self) -> None:
        s = PipelineStep(name="analyser", role="extractor")
        r = repr(s)
        assert "analyser" in r
        assert "extractor" in r


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestPipeline:
    def test_defaults(self) -> None:
        p = Pipeline()
        assert p.name == "pipeline"
        assert p.steps == []
        assert p.version == 0
        assert len(p.pipeline_id) == 8
        assert p.fitness is None

    def test_fitness_setter(self) -> None:
        p = Pipeline()
        p.fitness = 0.75
        assert p.fitness == pytest.approx(0.75)

    def test_pipeline_id_unique(self) -> None:
        ids = {Pipeline().pipeline_id for _ in range(20)}
        assert len(ids) == 20

    def test_to_dict_round_trip(self) -> None:
        p = _three_step_pipeline()
        p.version = 5
        restored = Pipeline.from_dict(p.to_dict())
        assert restored.name == p.name
        assert restored.version == p.version
        assert len(restored.steps) == 3
        assert restored.steps[0].name == "plan"
        assert restored.pipeline_id == p.pipeline_id

    def test_from_dict_no_steps(self) -> None:
        p = Pipeline.from_dict({"name": "empty"})
        assert p.steps == []
        assert p.name == "empty"

    def test_clone_has_fresh_id(self) -> None:
        p = _three_step_pipeline()
        clone = p.clone()
        assert clone.pipeline_id != p.pipeline_id

    def test_clone_clears_fitness(self) -> None:
        p = Pipeline()
        p.fitness = 0.9
        clone = p.clone()
        assert clone.fitness is None

    def test_clone_deep_steps(self) -> None:
        p = _three_step_pipeline()
        clone = p.clone()
        clone.steps[0].name = "modified"
        assert p.steps[0].name == "plan"

    def test_repr_shows_fitness_none(self) -> None:
        p = Pipeline()
        assert "None" in repr(p)

    def test_repr_shows_fitness_value(self) -> None:
        p = Pipeline()
        p.fitness = 0.5
        assert "0.5000" in repr(p)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineMutator._parse_pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestParsePipeline:
    def test_valid_json(self) -> None:
        raw = _pipeline_json("my-pipe", 2)
        fallback = Pipeline(name="fallback")
        result = PipelineMutator._parse_pipeline(raw, fallback)
        assert result.name == "my-pipe"
        assert len(result.steps) == 2

    def test_json_with_markdown_fence(self) -> None:
        raw = f"```json\n{_pipeline_json('fenced', 3)}\n```"
        fallback = Pipeline(name="fallback")
        result = PipelineMutator._parse_pipeline(raw, fallback)
        assert result.name == "fenced"
        assert len(result.steps) == 3

    def test_invalid_json_returns_fallback(self) -> None:
        fallback = Pipeline(name="fallback", steps=[PipelineStep(name="s")])
        result = PipelineMutator._parse_pipeline("this is not json", fallback)
        assert result.name == "fallback"

    def test_empty_string_returns_fallback(self) -> None:
        fallback = Pipeline(name="fb")
        result = PipelineMutator._parse_pipeline("", fallback)
        assert result.name == "fb"

    def test_partial_json_extracts_object(self) -> None:
        raw = f"Here is the mutation:\n{_pipeline_json('partial', 1)}\nEnd."
        fallback = Pipeline(name="fb")
        result = PipelineMutator._parse_pipeline(raw, fallback)
        assert result.name == "partial"


# ─────────────────────────────────────────────────────────────────────────────
# PipelineMutator with mock backend
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineMutatorWithMock:
    def test_mutate_increments_version(self) -> None:
        backend = _mock_backend(_pipeline_json("mutated", 2))
        mutator = PipelineMutator(backend)
        p = _three_step_pipeline()
        p.version = 3
        p.fitness = 0.5
        child = mutator.mutate(p, task="some task")
        assert child.version == 4
        assert child.fitness is None

    def test_mutate_calls_backend(self) -> None:
        backend = _mock_backend(_pipeline_json("x", 1))
        mutator = PipelineMutator(backend)
        p = Pipeline(steps=[PipelineStep()])
        mutator.mutate(p, task="task")
        assert backend.generate.call_count == 1

    def test_mutate_fallback_on_error(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = PipelineMutator(backend, fallback_on_error=True)
        p = _three_step_pipeline()
        child = mutator.mutate(p, task="task")
        assert len(child.steps) == len(p.steps)

    def test_mutate_raises_when_no_fallback(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        mutator = PipelineMutator(backend, fallback_on_error=False)
        p = Pipeline(steps=[PipelineStep()])
        with pytest.raises(RuntimeError):
            mutator.mutate(p)

    def test_mutate_enforces_max_steps(self) -> None:
        # Return a pipeline with 10 steps — should be capped at max_steps=3
        big = {"name": "big", "steps": [
            {"name": f"s{i}", "system_prompt": "x", "role": "transformer", "temperature": 0.7}
            for i in range(10)
        ]}
        backend = _mock_backend(json.dumps(big))
        mutator = PipelineMutator(backend, max_steps=3)
        p = Pipeline(steps=[PipelineStep()])
        child = mutator.mutate(p, task="t")
        assert len(child.steps) <= 3

    def test_mutate_ensures_at_least_one_step(self) -> None:
        backend = _mock_backend(json.dumps({"name": "empty", "steps": []}))
        mutator = PipelineMutator(backend)
        p = Pipeline(steps=[PipelineStep(name="orig")])
        child = mutator.mutate(p, task="t")
        assert len(child.steps) >= 1

    def test_crossover_version_is_max_plus_one(self) -> None:
        backend = _mock_backend(_pipeline_json("crossed", 2))
        mutator = PipelineMutator(backend)
        a = _three_step_pipeline()
        a.version = 5
        a.fitness = 0.8
        b = _three_step_pipeline()
        b.version = 3
        b.fitness = 0.6
        child = mutator.crossover(a, b, "task")
        assert child.version == 6  # max(5,3)+1

    def test_crossover_clears_fitness(self) -> None:
        backend = _mock_backend(_pipeline_json("c", 1))
        mutator = PipelineMutator(backend)
        a = _three_step_pipeline()
        a.fitness = 0.9
        b = _three_step_pipeline()
        b.fitness = 0.7
        child = mutator.crossover(a, b, "t")
        assert child.fitness is None


# ─────────────────────────────────────────────────────────────────────────────
# PipelineEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineEvaluator:
    def test_empty_pipeline_scores_zero(self) -> None:
        ev = PipelineEvaluator(_mock_backend())
        p = Pipeline(steps=[])
        assert ev.evaluate(p, "task") == pytest.approx(0.0)

    def test_exact_match_correct(self) -> None:
        backend = _mock_backend("expected-output")
        ev = PipelineEvaluator(backend, expected_output="expected-output")
        p = Pipeline(steps=[PipelineStep()])
        assert ev.evaluate(p, "task") == pytest.approx(1.0)

    def test_exact_match_wrong(self) -> None:
        backend = _mock_backend("wrong-output")
        ev = PipelineEvaluator(backend, expected_output="expected-output")
        p = Pipeline(steps=[PipelineStep()])
        assert ev.evaluate(p, "task") == pytest.approx(0.0)

    def test_llm_judge_parses_float(self) -> None:
        # First call: step output. Second call: judge returns "0.85"
        backend = MagicMock()
        backend.generate.side_effect = ["step output", "0.85"]
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[PipelineStep()])
        score = ev.evaluate(p, "task")
        assert score == pytest.approx(0.85)

    def test_llm_judge_clamps_to_one(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["out", "1.5"]
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[PipelineStep()])
        score = ev.evaluate(p, "t")
        assert score == pytest.approx(1.0)

    def test_llm_judge_clamps_to_zero(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["out", "-0.5"]
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[PipelineStep()])
        score = ev.evaluate(p, "t")
        assert score == pytest.approx(0.0)

    def test_step_failure_returns_low_score(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("step failed")
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[PipelineStep()])
        score = ev.evaluate(p, "task")
        assert score < 0.2

    def test_judge_parse_failure_returns_neutral(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ["step output", "not a number"]
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[PipelineStep()])
        score = ev.evaluate(p, "t")
        assert score == pytest.approx(0.5)

    def test_sequential_step_chaining(self) -> None:
        """Each step must receive the output of the previous step."""
        outputs = ["after-step-1", "after-step-2", "0.9"]
        backend = MagicMock()
        backend.generate.side_effect = outputs
        ev = PipelineEvaluator(backend)
        p = Pipeline(steps=[
            PipelineStep(name="s1", system_prompt="sys1"),
            PipelineStep(name="s2", system_prompt="sys2"),
        ])
        ev.evaluate(p, "initial-task")
        # Step 1: user="initial-task", system=sys1
        assert backend.generate.call_args_list[0][0][0] == "initial-task"
        # Step 2: user="after-step-1" (output of step 1), system=sys2
        assert backend.generate.call_args_list[1][0][0] == "after-step-1"


# ─────────────────────────────────────────────────────────────────────────────
# PipelineEvolutionEngine internals
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineEvolutionEngine:
    def _engine(self, pop_size: int = 4) -> PipelineEvolutionEngine:
        # Judge always returns 0.9; step responses just pass text through
        backend = MagicMock()
        backend.generate.return_value = "0.9"
        return PipelineEvolutionEngine(
            backend=backend,
            population_size=pop_size,
            elite_ratio=0.25,
            tournament_k=2,
            seed=42,
        )

    def test_initialize_correct_size(self) -> None:
        engine = self._engine(pop_size=5)
        pop = engine._initialize(_three_step_pipeline())
        assert len(pop) == 5

    def test_initialize_unique_ids(self) -> None:
        engine = self._engine(pop_size=4)
        pop = engine._initialize(_three_step_pipeline())
        ids = [p.pipeline_id for p in pop]
        assert len(set(ids)) == 4

    def test_tournament_returns_highest_fitness(self) -> None:
        engine = self._engine()
        pop = [Pipeline() for _ in range(5)]
        for i, p in enumerate(pop):
            p.fitness = float(i) * 0.1
        best = engine._tournament(pop)
        assert best.fitness == pytest.approx(0.4)

    def test_update_best_monotonic(self) -> None:
        engine = self._engine()
        pop1 = [Pipeline()]
        pop1[0].fitness = 0.7
        engine._update_best(pop1)
        assert engine.best is not None
        assert engine.best.fitness == pytest.approx(0.7)

        pop2 = [Pipeline()]
        pop2[0].fitness = 0.3
        engine._update_best(pop2)
        assert engine.best.fitness == pytest.approx(0.7)  # not downgraded

    def test_update_best_skips_none(self) -> None:
        engine = self._engine()
        pop = [Pipeline()]  # fitness is None
        engine._update_best(pop)
        assert engine.best is None

    def test_elite_n_minimum_one(self) -> None:
        backend = _mock_backend("0.5")
        engine = PipelineEvolutionEngine(backend=backend, population_size=3, elite_ratio=0.0)
        assert engine._elite_n == 1

    def test_evolve_returns_pipeline(self) -> None:
        backend = MagicMock()
        # Step outputs + judge scores
        backend.generate.return_value = "0.75"
        engine = PipelineEvolutionEngine(
            backend=backend,
            population_size=3,
            seed=0,
        )
        seed = _three_step_pipeline()
        best = engine.evolve(seed=seed, task="some task", n_generations=2)
        assert isinstance(best, Pipeline)
        assert best.fitness is not None
        assert best.fitness >= 0.0

    def test_on_generation_called(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = "0.6"
        engine = PipelineEvolutionEngine(backend=backend, population_size=2, seed=0)
        seed = Pipeline(steps=[PipelineStep()])
        calls: list[int] = []

        def cb(gen: int, pop: list[Pipeline]) -> None:
            calls.append(gen)

        engine.evolve(seed=seed, task="t", n_generations=3, on_generation=cb)
        assert calls == [0, 1, 2, 3]

    def test_best_property_initially_none(self) -> None:
        engine = self._engine()
        assert engine.best is None
