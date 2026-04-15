# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Edge-case tests for cambrian.pipeline — PipelineStep, Pipeline."""

from __future__ import annotations

import pytest

from cambrian.pipeline import Pipeline, PipelineStep


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_defaults(self) -> None:
        step = PipelineStep()
        assert step.name == "step"
        assert step.role == "transformer"
        assert 0.0 <= step.temperature <= 2.0

    def test_custom_fields(self) -> None:
        step = PipelineStep(
            name="extractor",
            system_prompt="Extract entities.",
            role="extractor",
            temperature=0.3,
        )
        assert step.name == "extractor"
        assert step.role == "extractor"
        assert step.temperature == pytest.approx(0.3)

    def test_to_dict_has_required_keys(self) -> None:
        step = PipelineStep(name="my_step")
        d = step.to_dict()
        assert "name" in d
        assert "system_prompt" in d
        assert "role" in d
        assert "temperature" in d

    def test_from_dict_round_trip(self) -> None:
        step = PipelineStep(name="validator", role="validator", temperature=0.2)
        d = step.to_dict()
        step2 = PipelineStep.from_dict(d)
        assert step2.name == step.name
        assert step2.role == step.role
        assert step2.temperature == pytest.approx(step.temperature)

    def test_from_dict_missing_keys_uses_defaults(self) -> None:
        step = PipelineStep.from_dict({})
        assert step.name == "step"
        assert step.role == "transformer"
        assert step.temperature == pytest.approx(0.7)

    def test_clone_has_same_content(self) -> None:
        step = PipelineStep(name="my_step", role="extractor")
        clone = step.clone()
        assert clone.name == step.name
        assert clone.role == step.role

    def test_clone_is_independent(self) -> None:
        step = PipelineStep(name="original")
        clone = step.clone()
        clone.name = "modified"
        assert step.name == "original"

    def test_repr_is_string(self) -> None:
        step = PipelineStep()
        assert isinstance(repr(step), str)

    def test_repr_contains_name(self) -> None:
        step = PipelineStep(name="unique_name_xyz")
        assert "unique_name_xyz" in repr(step)


# ---------------------------------------------------------------------------
# Pipeline — construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_empty_pipeline(self) -> None:
        p = Pipeline()
        assert p.steps == []
        assert p.name == "pipeline"
        assert p.version == 0

    def test_pipeline_with_steps(self) -> None:
        steps = [PipelineStep(name=f"step_{i}") for i in range(3)]
        p = Pipeline(name="my_pipeline", steps=steps)
        assert len(p.steps) == 3
        assert p.name == "my_pipeline"

    def test_pipeline_id_auto_generated(self) -> None:
        p = Pipeline()
        assert isinstance(p.pipeline_id, str)
        assert len(p.pipeline_id) > 0

    def test_two_pipelines_have_different_ids(self) -> None:
        p1 = Pipeline()
        p2 = Pipeline()
        assert p1.pipeline_id != p2.pipeline_id

    def test_custom_pipeline_id(self) -> None:
        p = Pipeline(pipeline_id="custom-id-123")
        assert p.pipeline_id == "custom-id-123"

    def test_fitness_starts_none(self) -> None:
        p = Pipeline()
        assert p.fitness is None

    def test_fitness_setter(self) -> None:
        p = Pipeline()
        p.fitness = 0.65
        assert p.fitness == pytest.approx(0.65)

    def test_fitness_setter_coerces_to_float(self) -> None:
        p = Pipeline()
        p.fitness = 1  # type: ignore[assignment]  # int
        assert isinstance(p.fitness, float)


# ---------------------------------------------------------------------------
# Pipeline — serialisation
# ---------------------------------------------------------------------------


class TestPipelineSerialization:
    def test_to_dict_has_steps(self) -> None:
        p = Pipeline(steps=[PipelineStep(name="s1"), PipelineStep(name="s2")])
        d = p.to_dict()
        assert "steps" in d
        assert len(d["steps"]) == 2

    def test_to_dict_has_name(self) -> None:
        p = Pipeline(name="test_pipeline")
        d = p.to_dict()
        assert d["name"] == "test_pipeline"

    def test_from_dict_round_trip(self) -> None:
        steps = [PipelineStep(name="a"), PipelineStep(name="b")]
        p = Pipeline(name="rt_test", steps=steps, version=3)
        d = p.to_dict()
        p2 = Pipeline.from_dict(d)
        assert p2.name == p.name
        assert p2.version == p.version
        assert len(p2.steps) == len(p.steps)
        assert p2.steps[0].name == "a"
        assert p2.steps[1].name == "b"

    def test_from_dict_missing_steps(self) -> None:
        p = Pipeline.from_dict({"name": "empty"})
        assert p.steps == []

    def test_clone_creates_fresh_id(self) -> None:
        p = Pipeline(steps=[PipelineStep(name="s1")])
        clone = p.clone()
        assert clone.pipeline_id != p.pipeline_id

    def test_clone_preserves_steps(self) -> None:
        steps = [PipelineStep(name="a"), PipelineStep(name="b")]
        p = Pipeline(name="original", steps=steps)
        clone = p.clone()
        assert len(clone.steps) == 2
        assert clone.steps[0].name == "a"

    def test_clone_steps_are_independent(self) -> None:
        p = Pipeline(steps=[PipelineStep(name="original_step")])
        clone = p.clone()
        clone.steps[0].name = "mutated_step"
        assert p.steps[0].name == "original_step"

    def test_clone_no_fitness_inherited(self) -> None:
        p = Pipeline()
        p.fitness = 0.9
        clone = p.clone()
        assert clone.fitness is None


# ---------------------------------------------------------------------------
# Pipeline — repr
# ---------------------------------------------------------------------------


class TestPipelineRepr:
    def test_repr_is_string(self) -> None:
        p = Pipeline()
        assert isinstance(repr(p), str)

    def test_repr_contains_name(self) -> None:
        p = Pipeline(name="unique_xyz_pipeline")
        assert "unique_xyz_pipeline" in repr(p)

    def test_repr_contains_step_count(self) -> None:
        steps = [PipelineStep() for _ in range(4)]
        p = Pipeline(steps=steps)
        r = repr(p)
        assert "4" in r

    def test_repr_shows_none_fitness(self) -> None:
        p = Pipeline()
        r = repr(p)
        assert "None" in r

    def test_repr_shows_fitness_when_set(self) -> None:
        p = Pipeline()
        p.fitness = 0.7777
        r = repr(p)
        assert "0.7777" in r
