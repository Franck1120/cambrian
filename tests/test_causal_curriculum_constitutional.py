# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.causal, cambrian.curriculum, cambrian.constitutional."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.causal import (
    CausalEdge,
    CausalGraph,
    CausalMutator,
    CausalStrategyExtractor,
    inject_causal_context,
)
from cambrian.constitutional import (
    DEFAULT_CONSTITUTION,
    ConstitutionalWrapper,
    build_constitutional_evaluator,
)
from cambrian.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    make_coding_curriculum,
    make_reasoning_curriculum,
)
from cambrian.mutator import LLMMutator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(prompt: str = "You are helpful.", fitness: float = 0.5) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


def _backend(response: str = "improved") -> MagicMock:
    genome_dict = {
        "system_prompt": response,
        "strategy": "step-by-step",
        "temperature": 0.7,
        "model": "gpt-4o-mini",
        "tools": [],
        "few_shot_examples": [],
    }
    b = MagicMock()
    b.generate = MagicMock(return_value=json.dumps(genome_dict))
    return b


def _base_eval(agent: Agent, task: str) -> float:
    return agent.fitness or 0.5


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------


class TestCausalGraph:
    def test_empty_construction(self) -> None:
        g = CausalGraph()
        assert g is not None

    def test_add_edge_and_get_effects(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.8, confidence=0.9)
        effects = g.get_effects("A")
        assert len(effects) == 1
        assert effects[0].effect == "B"
        assert effects[0].strength == pytest.approx(0.8)

    def test_add_edge_and_get_causes(self) -> None:
        g = CausalGraph()
        g.add_edge("X", "Y")
        causes = g.get_causes("Y")
        assert len(causes) == 1
        assert causes[0].cause == "X"

    def test_multiple_edges_same_cause(self) -> None:
        g = CausalGraph()
        g.add_edge("root", "child1")
        g.add_edge("root", "child2")
        effects = g.get_effects("root")
        assert len(effects) == 2

    def test_get_effects_unknown_node(self) -> None:
        g = CausalGraph()
        assert g.get_effects("ghost") == []

    def test_get_causes_unknown_node(self) -> None:
        g = CausalGraph()
        assert g.get_causes("ghost") == []

    def test_to_dict_and_from_dict_round_trip(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.7, confidence=0.8)
        g.add_edge("B", "C", strength=0.5, confidence=0.6)
        d = g.to_dict()
        g2 = CausalGraph.from_dict(d)
        assert len(g2.get_effects("A")) == 1
        assert len(g2.get_effects("B")) == 1

    def test_to_prompt_block_is_string(self) -> None:
        g = CausalGraph()
        g.add_edge("temperature", "creativity", strength=0.9)
        block = g.to_prompt_block()
        assert isinstance(block, str)
        assert "temperature" in block.lower() or "IF" in block

    def test_from_text_parses_edges(self) -> None:
        g = CausalGraph.from_text("High temperature causes creative output.")
        assert isinstance(g, CausalGraph)

    def test_prune_removes_weak_edges(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.1, confidence=0.1)
        g.add_edge("A", "C", strength=0.9, confidence=0.9)
        pruned = g.prune(min_strength=0.5, min_confidence=0.5)
        assert len(pruned.get_effects("A")) == 1
        assert pruned.get_effects("A")[0].effect == "C"

    def test_prune_keeps_all_when_threshold_zero(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.2, confidence=0.2)
        g.add_edge("A", "C", strength=0.8, confidence=0.8)
        pruned = g.prune(min_strength=0.0, min_confidence=0.0)
        assert len(pruned.get_effects("A")) == 2

    def test_merge_combines_edges(self) -> None:
        g1 = CausalGraph()
        g1.add_edge("A", "B")
        g2 = CausalGraph()
        g2.add_edge("C", "D")
        merged = g1.merge(g2)
        assert len(merged.get_effects("A")) >= 1
        assert len(merged.get_effects("C")) >= 1

    def test_causal_edge_dataclass_fields(self) -> None:
        g = CausalGraph()
        g.add_edge("X", "Y", strength=0.6, confidence=0.7)
        edge = g.get_effects("X")[0]
        assert isinstance(edge, CausalEdge)
        assert hasattr(edge, "cause")
        assert hasattr(edge, "effect")
        assert hasattr(edge, "strength")
        assert hasattr(edge, "confidence")


# ---------------------------------------------------------------------------
# inject_causal_context
# ---------------------------------------------------------------------------


class TestInjectCausalContext:
    def test_inject_returns_genome(self) -> None:
        g = CausalGraph()
        g.add_edge("temperature", "creativity")
        genome = Genome(system_prompt="Base prompt.")
        result = inject_causal_context(genome, g)
        assert isinstance(result, Genome)

    def test_inject_enriches_prompt(self) -> None:
        g = CausalGraph()
        g.add_edge("X", "Y", strength=0.9)
        genome = Genome(system_prompt="Original prompt.")
        result = inject_causal_context(genome, g)
        assert isinstance(result, Genome)

    def test_inject_empty_graph_passthrough(self) -> None:
        g = CausalGraph()
        genome = Genome(system_prompt="Unchanged.")
        result = inject_causal_context(genome, g)
        assert isinstance(result, Genome)


# ---------------------------------------------------------------------------
# CausalStrategyExtractor + CausalMutator
# ---------------------------------------------------------------------------


class TestCausalStrategyExtractor:
    def test_extract_returns_causal_graph(self) -> None:
        backend = MagicMock()
        backend.generate = MagicMock(return_value="A causes B. B leads to C.")
        extractor = CausalStrategyExtractor(backend=backend)
        g = extractor.extract("My strategy is systematic analysis.", task="coding")
        assert isinstance(g, CausalGraph)

    def test_extract_calls_backend(self) -> None:
        backend = MagicMock()
        backend.generate = MagicMock(return_value="X causes Y.")
        extractor = CausalStrategyExtractor(backend=backend)
        extractor.extract("some strategy")
        assert backend.generate.called


class TestCausalMutator:
    def test_mutate_with_causality_returns_tuple(self) -> None:
        b = _backend()
        mutator = LLMMutator(backend=b)
        extractor = CausalStrategyExtractor(backend=b)
        cm = CausalMutator(base_mutator=mutator, extractor=extractor)
        agent = _agent()
        result, graph = cm.mutate_with_causality(agent, task="solve math")
        assert isinstance(result, Agent)
        assert isinstance(graph, CausalGraph)


# ---------------------------------------------------------------------------
# CurriculumStage
# ---------------------------------------------------------------------------


class TestCurriculumStage:
    def test_defaults(self) -> None:
        stage = CurriculumStage(task="easy task")
        assert stage.task == "easy task"
        assert stage.difficulty == pytest.approx(0.0)
        assert stage.threshold == pytest.approx(0.7)
        assert stage.max_generations is None

    def test_custom_values(self) -> None:
        stage = CurriculumStage(
            task="hard task",
            difficulty=0.9,
            threshold=0.95,
            max_generations=20,
        )
        assert stage.difficulty == pytest.approx(0.9)
        assert stage.threshold == pytest.approx(0.95)
        assert stage.max_generations == 20

    def test_metadata_default_empty(self) -> None:
        stage = CurriculumStage(task="t")
        assert stage.metadata == {}

    def test_metadata_custom(self) -> None:
        stage = CurriculumStage(task="t", metadata={"subject": "math"})
        assert stage.metadata["subject"] == "math"


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class TestCurriculumScheduler:
    def _two_stage(self) -> CurriculumScheduler:
        stages = [
            CurriculumStage(task="easy", threshold=0.5),
            CurriculumStage(task="hard", threshold=0.8),
        ]
        return CurriculumScheduler(stages)

    def test_current_task_starts_at_first_stage(self) -> None:
        sched = self._two_stage()
        assert sched.current_task() == "easy"

    def test_advance_below_threshold_stays(self) -> None:
        sched = self._two_stage()
        advanced = sched.advance([0.3, 0.4])
        assert not advanced
        assert sched.current_task() == "easy"

    def test_advance_above_threshold_moves(self) -> None:
        sched = self._two_stage()
        advanced = sched.advance([0.6, 0.7, 0.8])
        assert advanced
        assert sched.current_task() == "hard"

    def test_advance_at_final_stage_returns_false(self) -> None:
        sched = self._two_stage()
        sched.advance([0.9])  # move to hard
        result = sched.advance([0.9, 0.95])
        assert not result

    def test_reset_returns_to_first_stage(self) -> None:
        sched = self._two_stage()
        sched.advance([0.9])
        sched.reset()
        assert sched.current_task() == "easy"

    def test_stage_summary_is_list(self) -> None:
        sched = self._two_stage()
        summary = sched.stage_summary()
        assert isinstance(summary, list)
        assert len(summary) == 2

    def test_stage_summary_has_task_key(self) -> None:
        sched = self._two_stage()
        summary = sched.stage_summary()
        for entry in summary:
            assert "task" in entry

    def test_loop_mode_cycles_back(self) -> None:
        stages = [CurriculumStage(task="A", threshold=0.5)]
        sched = CurriculumScheduler(stages, loop=True)
        sched.advance([0.9])
        assert sched.current_task() == "A"

    def test_make_coding_curriculum(self) -> None:
        sched = make_coding_curriculum()
        assert isinstance(sched, CurriculumScheduler)
        assert len(sched.stage_summary()) >= 2

    def test_make_reasoning_curriculum(self) -> None:
        sched = make_reasoning_curriculum()
        assert isinstance(sched, CurriculumScheduler)
        assert len(sched.stage_summary()) >= 2

    def test_metric_best(self) -> None:
        stages = [
            CurriculumStage(task="a", threshold=0.5),
            CurriculumStage(task="b", threshold=0.8),
        ]
        sched = CurriculumScheduler(stages, metric="best")
        sched.advance([0.6])  # best is 0.6 >= 0.5

    def test_empty_fitness_list_does_not_advance(self) -> None:
        sched = self._two_stage()
        advanced = sched.advance([])
        assert not advanced


# ---------------------------------------------------------------------------
# ConstitutionalWrapper
# ---------------------------------------------------------------------------


class TestConstitutionalWrapper:
    def test_construction(self) -> None:
        wrapper = ConstitutionalWrapper(base_evaluator=_base_eval)
        assert wrapper is not None

    def test_default_constitution_is_list(self) -> None:
        assert isinstance(DEFAULT_CONSTITUTION, list)
        assert len(DEFAULT_CONSTITUTION) > 0

    def test_call_no_backend_skips_returns_base(self) -> None:
        wrapper = ConstitutionalWrapper(
            base_evaluator=_base_eval, skip_if_no_backend=True
        )
        agent = _agent(fitness=0.75)
        score = wrapper(agent, "solve math")
        assert isinstance(score, float)
        assert score == pytest.approx(0.75)

    def test_call_no_backend_raises_when_skip_false(self) -> None:
        wrapper = ConstitutionalWrapper(
            base_evaluator=_base_eval, skip_if_no_backend=False
        )
        agent = _agent()  # no backend
        with pytest.raises(RuntimeError):
            wrapper(agent, "task")

    def test_custom_constitution(self) -> None:
        my_principles = ["Is the response helpful?", "Is it safe?"]
        wrapper = ConstitutionalWrapper(
            base_evaluator=_base_eval, constitution=my_principles
        )
        agent = _agent(fitness=0.6)
        score = wrapper(agent, "task")
        assert isinstance(score, float)

    def test_build_constitutional_evaluator_returns_wrapper(self) -> None:
        result = build_constitutional_evaluator(
            base_evaluator=_base_eval, n_principles=2, n_revisions=1
        )
        assert isinstance(result, ConstitutionalWrapper)

    def test_build_constitutional_evaluator_callable(self) -> None:
        wrapper = build_constitutional_evaluator(_base_eval)
        agent = _agent(fitness=0.8)
        score = wrapper(agent, "test task")
        assert isinstance(score, float)
