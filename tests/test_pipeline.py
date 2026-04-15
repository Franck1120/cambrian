"""Tests for cambrian.pipeline — PipelineStep, Pipeline, PipelineMutator, etc."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.pipeline import (
    Pipeline,
    PipelineEvaluator,
    PipelineEvolutionEngine,
    PipelineMutator,
    PipelineRunner,
    PipelineStep,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _step(role: str = "processor", prompt: str = "Process input.") -> PipelineStep:
    return PipelineStep(role=role, system_prompt=prompt)


def _pipeline(n: int = 2) -> Pipeline:
    return Pipeline(
        steps=[_step(f"step{i}", f"Do step {i}.") for i in range(n)],
        description="test pipeline",
    )


def _mock_backend(return_text: str = "result") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = return_text
    return backend


# ── PipelineStep ───────────────────────────────────────────────────────────────


class TestPipelineStep:
    def test_auto_step_id(self) -> None:
        s1 = PipelineStep(role="a", system_prompt="x")
        s2 = PipelineStep(role="b", system_prompt="y")
        assert s1.step_id != s2.step_id

    def test_roundtrip(self) -> None:
        step = PipelineStep(role="extractor", system_prompt="Extract facts.", temperature=0.5)
        r = PipelineStep.from_dict(step.to_dict())
        assert r.role == "extractor"
        assert r.system_prompt == "Extract facts."
        assert r.temperature == pytest.approx(0.5)

    def test_repr(self) -> None:
        assert "extractor" in repr(PipelineStep(role="extractor", system_prompt="x"))


# ── Pipeline ───────────────────────────────────────────────────────────────────


class TestPipeline:
    def test_step_count(self) -> None:
        assert _pipeline(3).step_count() == 3

    def test_empty_pipeline(self) -> None:
        assert Pipeline().step_count() == 0

    def test_roundtrip(self) -> None:
        p = _pipeline(2)
        r = Pipeline.from_dict(p.to_dict())
        assert r.step_count() == 2
        assert r.steps[0].role == p.steps[0].role

    def test_pipeline_id_unique(self) -> None:
        assert Pipeline().pipeline_id != Pipeline().pipeline_id

    def test_str_includes_roles(self) -> None:
        p = _pipeline(2)
        s = str(p)
        assert "step0" in s and "step1" in s

    def test_from_dict_empty_steps(self) -> None:
        p = Pipeline.from_dict({"steps": [], "description": "test"})
        assert p.step_count() == 0


# ── PipelineRunner ─────────────────────────────────────────────────────────────


class TestPipelineRunner:
    def test_single_step(self) -> None:
        backend = _mock_backend("output_text")
        runner = PipelineRunner(backend)
        p = Pipeline(steps=[_step("s1", "system")])
        result = runner.run(p, "input")
        assert result == "output_text"
        backend.generate.assert_called_once_with("input", system="system", temperature=0.7)

    def test_chain_passes_output_forward(self) -> None:
        responses = ["step1_out", "step2_out"]
        backend = MagicMock()
        backend.generate.side_effect = responses
        runner = PipelineRunner(backend)
        p = Pipeline(steps=[_step("s1"), _step("s2")])
        result = runner.run(p, "initial")
        assert result == "step2_out"
        assert backend.generate.call_count == 2
        # second call receives first call's output
        second_call_input = backend.generate.call_args_list[1][0][0]
        assert second_call_input == "step1_out"

    def test_empty_pipeline_returns_empty(self) -> None:
        backend = _mock_backend()
        runner = PipelineRunner(backend)
        assert runner.run(Pipeline(), "task") == ""
        backend.generate.assert_not_called()

    def test_temperature_per_step(self) -> None:
        backend = _mock_backend("out")
        runner = PipelineRunner(backend)
        step = PipelineStep(role="r", system_prompt="s", temperature=0.3)
        p = Pipeline(steps=[step])
        runner.run(p, "task")
        backend.generate.assert_called_once_with("task", system="s", temperature=0.3)


# ── PipelineEvaluator ──────────────────────────────────────────────────────────


class TestPipelineEvaluator:
    def _evaluator(self, return_text: str = "good output") -> PipelineEvaluator:
        backend = _mock_backend(return_text)

        def score_fn(output: str, task: str) -> float:
            return 0.8 if output else 0.0

        return PipelineEvaluator(backend, score_fn)

    def test_returns_score(self) -> None:
        ev = self._evaluator("good output")
        p = _pipeline(1)
        score = ev.evaluate(p, "task")
        assert score == pytest.approx(0.8)

    def test_empty_pipeline_returns_penalty(self) -> None:
        backend = _mock_backend()
        ev = PipelineEvaluator(backend, lambda o, t: 0.9, empty_pipeline_penalty=0.1)
        assert ev.evaluate(Pipeline(), "task") == pytest.approx(0.1)

    def test_score_clipped_to_range(self) -> None:
        backend = _mock_backend("out")
        ev = PipelineEvaluator(backend, lambda o, t: 2.0)  # returns > 1
        p = _pipeline(1)
        assert ev.evaluate(p, "task") == pytest.approx(1.0)

    def test_score_negative_clipped(self) -> None:
        backend = _mock_backend("out")
        ev = PipelineEvaluator(backend, lambda o, t: -0.5)
        p = _pipeline(1)
        assert ev.evaluate(p, "task") == pytest.approx(0.0)

    def test_backend_error_returns_zero(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("API down")
        ev = PipelineEvaluator(backend, lambda o, t: 1.0)
        p = _pipeline(1)
        assert ev.evaluate(p, "task") == pytest.approx(0.0)


# ── PipelineMutator ────────────────────────────────────────────────────────────


class TestPipelineMutator:
    def _mutated_pipeline_json(self, n: int = 2) -> str:
        steps = [{"role": f"r{i}", "system_prompt": f"s{i}", "temperature": 0.7} for i in range(n)]
        return json.dumps({"steps": steps, "description": "test"})

    def test_mutate_returns_pipeline(self) -> None:
        backend = _mock_backend(self._mutated_pipeline_json(2))
        mutator = PipelineMutator(backend=backend)
        child = mutator.mutate(_pipeline(2))
        assert isinstance(child, Pipeline)

    def test_version_incremented(self) -> None:
        backend = _mock_backend(self._mutated_pipeline_json(2))
        mutator = PipelineMutator(backend=backend)
        parent = _pipeline(2)
        parent.version = 3
        child = mutator.mutate(parent)
        assert child.version == 4

    def test_parent_id_set(self) -> None:
        backend = _mock_backend(self._mutated_pipeline_json(2))
        mutator = PipelineMutator(backend=backend)
        parent = _pipeline(2)
        child = mutator.mutate(parent)
        assert child.parent_id == parent.pipeline_id

    def test_seed_empty_pipeline(self) -> None:
        backend = _mock_backend(self._mutated_pipeline_json(3))
        mutator = PipelineMutator(backend=backend)
        seed = Pipeline(description="do something")
        child = mutator.mutate(seed)
        assert isinstance(child, Pipeline)
        assert child.step_count() >= 1

    def test_fallback_on_error(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("API down")
        mutator = PipelineMutator(backend=backend, fallback_on_error=True)
        parent = _pipeline(2)
        child = mutator.mutate(parent)
        # Should return original pipeline data
        assert isinstance(child, Pipeline)

    def test_raises_on_error_when_no_fallback(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("API down")
        mutator = PipelineMutator(backend=backend, fallback_on_error=False)
        with pytest.raises(RuntimeError):
            mutator.mutate(_pipeline(2))

    def test_strips_markdown_json(self) -> None:
        raw = f"```json\n{self._mutated_pipeline_json(1)}\n```"
        backend = _mock_backend(raw)
        mutator = PipelineMutator(backend=backend)
        child = mutator.mutate(_pipeline(1))
        assert child.step_count() >= 1

    def test_crossover_combines_steps(self) -> None:
        backend = _mock_backend()
        mutator = PipelineMutator(backend=backend)
        pa = Pipeline(
            steps=[_step("a1"), _step("a2")],
            description="pa",
        )
        pb = Pipeline(
            steps=[_step("b1"), _step("b2")],
            description="pb",
        )
        child = mutator.crossover(pa, pb)
        assert child.step_count() >= 1
        roles = {s.role for s in child.steps}
        # Should contain a mix from both parents
        assert len(roles) >= 2

    def test_crossover_deduplicates_roles(self) -> None:
        backend = _mock_backend()
        mutator = PipelineMutator(backend=backend)
        shared_step = _step("shared", "shared system")
        pa = Pipeline(steps=[shared_step, _step("a2")])
        pb = Pipeline(steps=[shared_step, _step("b2")])
        child = mutator.crossover(pa, pb)
        roles = [s.role for s in child.steps]
        # 'shared' should appear only once
        assert roles.count("shared") == 1

    def test_crossover_max_six_steps(self) -> None:
        backend = _mock_backend()
        mutator = PipelineMutator(backend=backend)
        pa = Pipeline(steps=[_step(f"a{i}") for i in range(5)])
        pb = Pipeline(steps=[_step(f"b{i}") for i in range(5)])
        child = mutator.crossover(pa, pb)
        assert child.step_count() <= 6


# ── PipelineEvolutionEngine ────────────────────────────────────────────────────


class TestPipelineEvolutionEngine:
    def _make_engine(self, return_text: str = "output") -> PipelineEvolutionEngine:
        backend = _mock_backend(return_text)
        return PipelineEvolutionEngine(
            backend=backend,
            score_fn=lambda o, t: 0.7 if o else 0.0,
            population_size=4,
        )

    def test_returns_pipeline(self) -> None:
        engine = self._make_engine()
        seed = _pipeline(1)
        best = engine.evolve(seed, "task", n_generations=2)
        assert isinstance(best, Pipeline)

    def test_raises_without_score_fn_or_evaluator(self) -> None:
        backend = _mock_backend()
        engine = PipelineEvolutionEngine(backend=backend, population_size=3)
        with pytest.raises(ValueError, match="score_fn"):
            engine.evolve(_pipeline(1), "task", n_generations=1)

    def test_on_generation_callback(self) -> None:
        engine = self._make_engine()
        calls: list[int] = []
        engine.evolve(
            _pipeline(1), "task", n_generations=3,
            on_generation=lambda g, scored, best: calls.append(g),
        )
        assert len(calls) == 3

    def test_evaluator_at_construction(self) -> None:
        backend = _mock_backend("output")
        ev = PipelineEvaluator(backend, lambda o, t: 0.5)
        engine = PipelineEvolutionEngine(
            backend=backend, evaluator=ev, population_size=3
        )
        best = engine.evolve(_pipeline(1), "task", n_generations=2)
        assert isinstance(best, Pipeline)

    def test_description_set_from_task(self) -> None:
        engine = self._make_engine()
        seed = Pipeline()  # no description
        best = engine.evolve(seed, "my task", n_generations=1)
        assert isinstance(best, Pipeline)


# ── forge CLI command ──────────────────────────────────────────────────────────


class TestForgeCLI:
    def test_forge_requires_api_key(self) -> None:
        import os
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        runner = CliRunner()
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        result = runner.invoke(
            cli, ["forge", "test task", "--api-key", ""],
            env={**env, "OPENAI_API_KEY": ""},
        )
        assert result.exit_code != 0

    def test_forge_help(self) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        result = CliRunner().invoke(cli, ["forge", "--help"])
        assert result.exit_code == 0
        assert "Forge mode" in result.output or "forge" in result.output.lower()
