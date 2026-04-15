"""Tests for cambrian.transgenerational — TransgenerationalRegistry."""
from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.transgenerational import (
    EpigeneMark,
    InheritanceRecord,
    TransgenerationalRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(prompt: str = "agent prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    return Agent(genome=g)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self) -> None:
        reg = TransgenerationalRegistry()
        assert reg._max_gen == 5
        assert reg._threshold == 0.1
        assert reg._top_n == 5

    def test_records_starts_empty(self) -> None:
        reg = TransgenerationalRegistry()
        assert reg.records == []

    def test_generation_starts_zero(self) -> None:
        reg = TransgenerationalRegistry()
        assert reg._generation == 0


# ---------------------------------------------------------------------------
# record_mark
# ---------------------------------------------------------------------------


class TestRecordMark:
    def test_creates_new_mark(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        mark = reg.record_mark(agent, "step-by-step", strength=0.8)
        assert isinstance(mark, EpigeneMark)
        assert mark.name == "step-by-step"
        assert pytest.approx(mark.strength, abs=1e-9) == 0.8

    def test_boosts_existing_mark(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        reg.record_mark(agent, "verify", strength=0.6)
        reg.record_mark(agent, "verify", strength=0.4)
        marks = reg.get_marks(agent)
        # Should be 1 mark with boosted strength
        verify = [m for m in marks if m.name == "verify"]
        assert len(verify) == 1
        assert verify[0].strength > 0.6

    def test_strength_clamped_to_one(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        mark = reg.record_mark(agent, "x", strength=2.0)
        assert mark.strength == 1.0

    def test_strength_clamped_to_zero(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        mark = reg.record_mark(agent, "x", strength=-1.0)
        assert mark.strength == 0.0


# ---------------------------------------------------------------------------
# get_marks
# ---------------------------------------------------------------------------


class TestGetMarks:
    def test_returns_empty_for_unknown_agent(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        assert reg.get_marks(agent) == []

    def test_sorted_by_strength_desc(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        reg.record_mark(agent, "low", strength=0.3)
        reg.record_mark(agent, "high", strength=0.9)
        marks = reg.get_marks(agent)
        assert marks[0].name == "high"
        assert marks[1].name == "low"


# ---------------------------------------------------------------------------
# advance_generation (decay)
# ---------------------------------------------------------------------------


class TestAdvanceGeneration:
    def test_increments_generation(self) -> None:
        reg = TransgenerationalRegistry()
        reg.advance_generation()
        assert reg._generation == 1

    def test_decays_marks(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5, strength_threshold=0.1)
        agent = _make_agent()
        reg.record_mark(agent, "mark", strength=0.8)
        reg.advance_generation()
        marks = reg.get_marks(agent)
        assert len(marks) == 1
        # Decayed by 1/5 = 0.2 → strength = 0.6
        assert pytest.approx(marks[0].strength, abs=1e-6) == 0.6

    def test_prunes_weak_marks(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5, strength_threshold=0.2)
        agent = _make_agent()
        reg.record_mark(agent, "weak", strength=0.21)
        reg.advance_generation()  # 0.21 - 0.2 = 0.01 < 0.2 → pruned
        assert reg.get_marks(agent) == []


# ---------------------------------------------------------------------------
# inherit
# ---------------------------------------------------------------------------


class TestInherit:
    def test_transfers_marks_to_offspring(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5)
        parent = _make_agent()
        offspring = _make_agent()
        reg.record_mark(parent, "good-strategy", strength=0.9)
        n = reg.inherit(parent, offspring)
        assert n == 1
        offspring_marks = reg.get_marks(offspring)
        assert len(offspring_marks) == 1
        assert offspring_marks[0].name == "good-strategy"

    def test_mark_decayed_in_offspring(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5, strength_threshold=0.1)
        parent = _make_agent()
        offspring = _make_agent()
        reg.record_mark(parent, "mark", strength=0.8)
        reg.inherit(parent, offspring)
        offspring_marks = reg.get_marks(offspring)
        # Inherited strength = 0.8 - 0.2 = 0.6
        assert pytest.approx(offspring_marks[0].strength, abs=1e-9) == 0.6

    def test_weak_marks_not_inherited(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5, strength_threshold=0.3)
        parent = _make_agent()
        offspring = _make_agent()
        reg.record_mark(parent, "weak", strength=0.35)
        n = reg.inherit(parent, offspring)
        # Inherited: 0.35 - 0.2 = 0.15 < 0.3 → not inherited
        assert n == 0

    def test_record_stored(self) -> None:
        reg = TransgenerationalRegistry()
        parent = _make_agent()
        offspring = _make_agent()
        reg.record_mark(parent, "m", strength=0.9)
        reg.inherit(parent, offspring)
        assert len(reg.records) == 1
        rec = reg.records[0]
        assert isinstance(rec, InheritanceRecord)
        assert rec.parent_id == parent.agent_id
        assert rec.offspring_id == offspring.agent_id

    def test_records_returns_copy(self) -> None:
        reg = TransgenerationalRegistry()
        reg.inherit(_make_agent(), _make_agent())
        r1 = reg.records
        r1.clear()
        assert len(reg.records) == 1

    def test_top_n_limit(self) -> None:
        reg = TransgenerationalRegistry(max_generations=5, inherit_top_n=2)
        parent = _make_agent()
        offspring = _make_agent()
        reg.record_mark(parent, "a", strength=0.9)
        reg.record_mark(parent, "b", strength=0.8)
        reg.record_mark(parent, "c", strength=0.7)
        reg.inherit(parent, offspring)
        # Only top 2 transferred
        marks = reg.get_marks(offspring)
        assert len(marks) <= 2


# ---------------------------------------------------------------------------
# inject_context / apply_to_genome
# ---------------------------------------------------------------------------


class TestInjectContext:
    def test_empty_when_no_marks(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        assert reg.inject_context(agent) == ""

    def test_contains_mark_name(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent()
        reg.record_mark(agent, "step-by-step", strength=0.9)
        ctx = reg.inject_context(agent)
        assert "step-by-step" in ctx

    def test_apply_to_genome_injects_context(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent("original prompt")
        reg.record_mark(agent, "verify-answers", strength=0.8)
        new_genome = reg.apply_to_genome(agent)
        assert "verify-answers" in new_genome.system_prompt
        assert "original prompt" in new_genome.system_prompt

    def test_apply_to_genome_no_marks_unchanged(self) -> None:
        reg = TransgenerationalRegistry()
        agent = _make_agent("original prompt")
        new_genome = reg.apply_to_genome(agent)
        assert new_genome.system_prompt == "original prompt"
