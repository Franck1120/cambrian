# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.immune_memory — BCellMemory, TCellMemory, ImmuneCortex."""
from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.immune_memory import (
    BCellMemory,
    ImmuneCortex,
    MemoryCell,
    RecallResult,
    TCellMemory,
    _task_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float = 0.9, prompt: str = "agent system prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


# ---------------------------------------------------------------------------
# _task_similarity
# ---------------------------------------------------------------------------


class TestTaskSimilarity:
    def test_identical_strings(self) -> None:
        assert _task_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_strings(self) -> None:
        assert _task_similarity("hello world", "foo bar baz") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        sim = _task_similarity("hello world foo", "hello world bar")
        # Intersection: {hello, world} = 2; Union: {hello, world, foo, bar} = 4 → 0.5
        assert sim == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert _task_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self) -> None:
        assert _task_similarity("hello", "") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        assert _task_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_punctuation_ignored(self) -> None:
        sim = _task_similarity("hello, world!", "hello world")
        assert sim == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BCellMemory — init
# ---------------------------------------------------------------------------


class TestBCellMemoryInit:
    def test_defaults(self) -> None:
        mem = BCellMemory()
        assert mem._threshold == 0.8
        assert mem._max == 100
        assert mem.size == 0

    def test_custom_params(self) -> None:
        mem = BCellMemory(similarity_threshold=0.9, max_cells=50)
        assert mem._threshold == 0.9
        assert mem._max == 50


# ---------------------------------------------------------------------------
# BCellMemory — store
# ---------------------------------------------------------------------------


class TestBCellMemoryStore:
    def test_store_increases_size(self) -> None:
        mem = BCellMemory()
        agent = _make_agent()
        mem.store(agent, "solve math problem")
        assert mem.size == 1

    def test_store_ignores_no_fitness(self) -> None:
        mem = BCellMemory()
        # Agent created without setting fitness → fitness is None
        g = Genome(system_prompt="x")
        agent = Agent(genome=g)
        mem.store(agent, "task")
        assert mem.size == 0

    def test_fifo_eviction(self) -> None:
        mem = BCellMemory(max_cells=2)
        mem.store(_make_agent(), "task 1")
        mem.store(_make_agent(), "task 2")
        mem.store(_make_agent(), "task 3")
        assert mem.size == 2

    def test_stored_cell_is_memory_cell(self) -> None:
        mem = BCellMemory()
        agent = _make_agent()
        mem.store(agent, "task")
        cell = mem._cells[0]
        assert isinstance(cell, MemoryCell)
        assert cell.task == "task"
        assert cell.fitness == agent.fitness


# ---------------------------------------------------------------------------
# BCellMemory — recall
# ---------------------------------------------------------------------------


class TestBCellMemoryRecall:
    def test_recall_returns_none_when_empty(self) -> None:
        mem = BCellMemory()
        assert mem.recall("any task") is None

    def test_recall_returns_cell_above_threshold(self) -> None:
        mem = BCellMemory(similarity_threshold=0.5)
        agent = _make_agent()
        mem.store(agent, "solve math problem")
        cell = mem.recall("solve math problem")
        assert cell is not None
        assert cell.task == "solve math problem"

    def test_recall_returns_none_below_threshold(self) -> None:
        mem = BCellMemory(similarity_threshold=0.9)
        agent = _make_agent()
        mem.store(agent, "solve calculus")
        result = mem.recall("cook pasta")
        assert result is None

    def test_recall_increments_recall_count(self) -> None:
        mem = BCellMemory(similarity_threshold=0.5)
        agent = _make_agent()
        mem.store(agent, "hello world task")
        cell = mem.recall("hello world task")
        assert cell is not None
        assert cell.recall_count == 1

    def test_recall_returns_best_match(self) -> None:
        mem = BCellMemory(similarity_threshold=0.3)
        mem.store(_make_agent(), "solve math problem step by step")
        mem.store(_make_agent(), "solve calculus derivative")
        # "solve math problem" shares more with first
        cell = mem.recall("solve math problem addition")
        assert cell is not None
        assert "math" in cell.task

    def test_genome_clone_is_independent(self) -> None:
        """Stored genome should be a deep copy, not a reference."""
        mem = BCellMemory(similarity_threshold=0.5)
        agent = _make_agent(prompt="original prompt")
        mem.store(agent, "task")
        agent.genome.system_prompt = "modified prompt"
        # Stored cell should still have original
        assert mem._cells[0].genome.system_prompt == "original prompt"


# ---------------------------------------------------------------------------
# TCellMemory — init
# ---------------------------------------------------------------------------


class TestTCellMemoryInit:
    def test_defaults(self) -> None:
        mem = TCellMemory()
        assert mem._min_sim == 0.2
        assert mem._max == 50
        assert mem.size == 0

    def test_custom_params(self) -> None:
        mem = TCellMemory(min_similarity=0.3, max_cells=20)
        assert mem._min_sim == 0.3
        assert mem._max == 20


# ---------------------------------------------------------------------------
# TCellMemory — store & best_match
# ---------------------------------------------------------------------------


class TestTCellMemoryStore:
    def test_store_increases_size(self) -> None:
        mem = TCellMemory()
        mem.store(_make_agent(), "task")
        assert mem.size == 1

    def test_store_ignores_no_fitness(self) -> None:
        mem = TCellMemory()
        g = Genome(system_prompt="x")
        agent = Agent(genome=g)
        mem.store(agent, "task")
        assert mem.size == 0

    def test_fifo_eviction(self) -> None:
        mem = TCellMemory(max_cells=2)
        for i in range(3):
            mem.store(_make_agent(), f"task {i}")
        assert mem.size == 2


class TestTCellMemoryBestMatch:
    def test_returns_none_when_empty(self) -> None:
        mem = TCellMemory()
        assert mem.best_match("task") is None

    def test_returns_none_below_min_sim(self) -> None:
        mem = TCellMemory(min_similarity=0.8)
        mem.store(_make_agent(), "quantum physics experiment")
        result = mem.best_match("cook pasta recipe")
        assert result is None

    def test_returns_best_above_min_sim(self) -> None:
        mem = TCellMemory(min_similarity=0.1)
        mem.store(_make_agent(), "solve math problem")
        cell = mem.best_match("solve algebra")
        assert cell is not None

    def test_increments_recall_count(self) -> None:
        mem = TCellMemory(min_similarity=0.1)
        mem.store(_make_agent(), "hello world")
        cell = mem.best_match("hello world")
        assert cell is not None
        assert cell.recall_count == 1

    def test_returns_highest_similarity(self) -> None:
        mem = TCellMemory(min_similarity=0.1)
        mem.store(_make_agent(), "solve math algebra")
        mem.store(_make_agent(), "cook dinner recipe")
        # "solve math problem" is more similar to first
        cell = mem.best_match("solve math problem")
        assert cell is not None
        assert "math" in cell.task


# ---------------------------------------------------------------------------
# ImmuneCortex — init
# ---------------------------------------------------------------------------


class TestImmuneCortexInit:
    def test_defaults(self) -> None:
        cortex = ImmuneCortex()
        assert cortex._b_threshold == 0.8
        assert cortex._t_threshold == 0.5
        assert cortex.b_cell_count == 0
        assert cortex.t_cell_count == 0

    def test_custom_params(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.9, t_threshold=0.6)
        assert cortex._b_threshold == 0.9
        assert cortex._t_threshold == 0.6


# ---------------------------------------------------------------------------
# ImmuneCortex — record
# ---------------------------------------------------------------------------


class TestImmuneCortexRecord:
    def test_high_fitness_stored_in_both(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.7, t_threshold=0.5)
        agent = _make_agent(fitness=0.9)
        cortex.record(agent, "task")
        assert cortex.b_cell_count == 1
        assert cortex.t_cell_count == 1

    def test_medium_fitness_stored_in_t_only(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.8, t_threshold=0.5)
        agent = _make_agent(fitness=0.6)
        cortex.record(agent, "task")
        assert cortex.b_cell_count == 0
        assert cortex.t_cell_count == 1

    def test_low_fitness_not_stored(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.8, t_threshold=0.5)
        agent = _make_agent(fitness=0.3)
        cortex.record(agent, "task")
        assert cortex.b_cell_count == 0
        assert cortex.t_cell_count == 0

    def test_none_fitness_not_stored(self) -> None:
        cortex = ImmuneCortex()
        g = Genome(system_prompt="x")
        agent = Agent(genome=g)
        cortex.record(agent, "task")
        assert cortex.b_cell_count == 0
        assert cortex.t_cell_count == 0


# ---------------------------------------------------------------------------
# ImmuneCortex — recall
# ---------------------------------------------------------------------------


class TestImmuneCortexRecall:
    def test_recall_returns_recall_result(self) -> None:
        cortex = ImmuneCortex()
        result = cortex.recall("any task")
        assert isinstance(result, RecallResult)

    def test_no_memory_returns_not_recalled(self) -> None:
        cortex = ImmuneCortex()
        result = cortex.recall("solve problem")
        assert result.recalled is False
        assert result.agent is None
        assert result.cell_type == "none"
        assert result.similarity == 0.0
        assert result.source_task == ""

    def test_b_cell_hit(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.5, b_similarity=0.5)
        agent = _make_agent(fitness=0.9, prompt="expert math agent")
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        assert result.recalled is True
        assert result.cell_type == "b_cell"
        assert result.agent is not None

    def test_t_cell_hit_when_b_misses(self) -> None:
        cortex = ImmuneCortex(
            b_threshold=0.5,
            t_threshold=0.5,
            b_similarity=0.99,   # very high B-cell threshold → no B match
            t_min_similarity=0.1,
        )
        agent = _make_agent(fitness=0.7, prompt="algebra expert")
        cortex.record(agent, "solve algebra equations")
        # Task is semantically related but won't meet 0.99 B-cell threshold
        result = cortex.recall("algebra problem solve")
        assert result.recalled is True
        assert result.cell_type == "t_cell"

    def test_recalled_agent_has_fitness(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.5, b_similarity=0.5)
        agent = _make_agent(fitness=0.88)
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        assert result.recalled is True
        assert result.agent is not None
        assert result.agent.fitness == pytest.approx(0.88)

    def test_recalled_agent_is_independent_copy(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.5, b_similarity=0.5)
        agent = _make_agent(prompt="original prompt")
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        assert result.recalled is True
        assert result.agent is not None
        # Modifying the recalled agent should not affect stored cell
        result.agent.genome.system_prompt = "mutated"
        result2 = cortex.recall("solve math problem")
        assert result2.agent is not None
        assert result2.agent.genome.system_prompt == "original prompt"

    def test_b_cell_takes_priority_over_t_cell(self) -> None:
        cortex = ImmuneCortex(
            b_threshold=0.5,
            t_threshold=0.3,
            b_similarity=0.5,
            t_min_similarity=0.1,
        )
        agent = _make_agent(fitness=0.9)
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        # Both would match, but B-cell should win
        assert result.cell_type == "b_cell"

    def test_similarity_field_populated(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.5, b_similarity=0.5)
        agent = _make_agent()
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        assert result.similarity == pytest.approx(1.0)

    def test_source_task_field_populated(self) -> None:
        cortex = ImmuneCortex(b_threshold=0.5, b_similarity=0.5)
        agent = _make_agent()
        cortex.record(agent, "solve math problem")
        result = cortex.recall("solve math problem")
        assert result.source_task == "solve math problem"
