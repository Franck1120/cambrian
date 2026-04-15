# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""tests/test_dream.py — Unit tests for the Dream Phase module.

Covers:
- DreamPhase.should_dream: interval logic
- DreamPhase.dream: offspring generation with mock backend + mock memory
- DreamPhase._build_experience_text: formatting
- DreamPhase._parse_offspring: JSON array, wrapped object, fence stripping, fallback
- Integration: dream_count tracking
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock


from cambrian.dream import DreamPhase


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_memory(ancestors: list[dict[str, Any]] | None = None) -> MagicMock:
    mem = MagicMock()
    mem.get_top_ancestors.return_value = ancestors or []
    return mem


def _mock_backend(response: str = "") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _offspring_json(n: int = 2) -> str:
    items = [
        {
            "system_prompt": f"You are a helpful agent #{i}.",
            "strategy": "chain-of-thought",
            "temperature": 0.7 + i * 0.05,
        }
        for i in range(n)
    ]
    return json.dumps(items)


def _ancestor(rank: int = 1, fitness: float = 0.8) -> dict[str, Any]:
    return {
        "agent_id": f"agent-{rank:04d}",
        "generation": rank,
        "fitness": fitness,
        "genome": {
            "system_prompt": f"Think carefully. Strategy #{rank}.",
            "strategy": "direct",
            "temperature": 0.7,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# should_dream
# ─────────────────────────────────────────────────────────────────────────────


class TestShouldDream:
    def test_returns_false_at_generation_zero(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=5)
        assert dp.should_dream(0) is False

    def test_returns_true_at_exact_interval(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=5)
        assert dp.should_dream(5) is True

    def test_returns_true_at_multiples_of_interval(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=3)
        assert dp.should_dream(3) is True
        assert dp.should_dream(6) is True
        assert dp.should_dream(9) is True

    def test_returns_false_between_multiples(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=5)
        assert dp.should_dream(1) is False
        assert dp.should_dream(4) is False
        assert dp.should_dream(6) is False

    def test_interval_one_fires_every_generation(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=1)
        for gen in range(1, 6):
            assert dp.should_dream(gen) is True

    def test_interval_zero_never_fires(self) -> None:
        dp = DreamPhase(_mock_backend(), _mock_memory(), interval=0)
        for gen in range(0, 10):
            assert dp.should_dream(gen) is False


# ─────────────────────────────────────────────────────────────────────────────
# _build_experience_text
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildExperienceText:
    def test_includes_fitness(self) -> None:
        ancestors = [_ancestor(1, fitness=0.93)]
        text = DreamPhase._build_experience_text(ancestors)
        assert "0.9300" in text

    def test_includes_strategy(self) -> None:
        ancestors = [_ancestor(1)]
        text = DreamPhase._build_experience_text(ancestors)
        assert "direct" in text

    def test_includes_prompt_preview(self) -> None:
        ancestors = [_ancestor(1)]
        text = DreamPhase._build_experience_text(ancestors)
        assert "Think carefully" in text

    def test_multiple_ancestors_ranked(self) -> None:
        ancestors = [_ancestor(1, 0.9), _ancestor(2, 0.7)]
        text = DreamPhase._build_experience_text(ancestors)
        assert "[1]" in text
        assert "[2]" in text

    def test_empty_ancestors(self) -> None:
        text = DreamPhase._build_experience_text([])
        assert text == ""


# ─────────────────────────────────────────────────────────────────────────────
# _parse_offspring
# ─────────────────────────────────────────────────────────────────────────────


class TestParseOffspring:
    def test_valid_json_array(self) -> None:
        raw = _offspring_json(3)
        result = DreamPhase._parse_offspring(raw, 3)
        assert len(result) == 3
        assert "system_prompt" in result[0]

    def test_fenced_json_array(self) -> None:
        raw = f"```json\n{_offspring_json(2)}\n```"
        result = DreamPhase._parse_offspring(raw, 2)
        assert len(result) == 2

    def test_single_object_wraps_to_list(self) -> None:
        raw = json.dumps({
            "system_prompt": "Be helpful.",
            "strategy": "direct",
            "temperature": 0.7,
        })
        result = DreamPhase._parse_offspring(raw, 1)
        assert len(result) == 1
        assert result[0]["system_prompt"] == "Be helpful."

    def test_invalid_json_returns_empty(self) -> None:
        result = DreamPhase._parse_offspring("not json at all", 2)
        assert result == []

    def test_empty_string_returns_empty(self) -> None:
        result = DreamPhase._parse_offspring("", 2)
        assert result == []

    def test_non_dict_items_filtered(self) -> None:
        raw = json.dumps(["string", {"system_prompt": "ok"}, 42])
        result = DreamPhase._parse_offspring(raw, 3)
        assert len(result) == 1
        assert result[0]["system_prompt"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# DreamPhase.dream
# ─────────────────────────────────────────────────────────────────────────────


class TestDream:
    def test_returns_empty_when_no_ancestors(self) -> None:
        dp = DreamPhase(_mock_backend(_offspring_json(2)), _mock_memory([]))
        result = dp.dream("task", n_offspring=2)
        assert result == []

    def test_returns_genome_objects(self) -> None:
        from cambrian.agent import Genome

        mem = _mock_memory([_ancestor(1, 0.8), _ancestor(2, 0.7)])
        backend = _mock_backend(_offspring_json(2))
        dp = DreamPhase(backend, mem)
        result = dp.dream("Write code", n_offspring=2)
        assert len(result) == 2
        assert all(isinstance(g, Genome) for g in result)

    def test_temperature_clamped_to_valid_range(self) -> None:
        mem = _mock_memory([_ancestor(1)])
        raw = json.dumps([
            {"system_prompt": "p1", "strategy": "s1", "temperature": 5.0},  # too high
            {"system_prompt": "p2", "strategy": "s2", "temperature": -1.0},  # too low
        ])
        backend = _mock_backend(raw)
        dp = DreamPhase(backend, mem)
        result = dp.dream("task", n_offspring=2)
        for g in result:
            assert 0.0 <= g.temperature <= 2.0

    def test_llm_failure_returns_empty(self) -> None:
        mem = _mock_memory([_ancestor(1)])
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("network error")
        dp = DreamPhase(backend, mem)
        result = dp.dream("task")
        assert result == []

    def test_dream_count_increments(self) -> None:
        mem = _mock_memory([_ancestor(1)])
        backend = _mock_backend(_offspring_json(1))
        dp = DreamPhase(backend, mem)
        assert dp.dream_count == 0
        dp.dream("task")
        assert dp.dream_count == 1
        dp.dream("task")
        assert dp.dream_count == 2

    def test_dream_count_stays_zero_on_failure(self) -> None:
        mem = _mock_memory([_ancestor(1)])
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("error")
        dp = DreamPhase(backend, mem)
        dp.dream("task")
        assert dp.dream_count == 0

    def test_dream_count_stays_zero_on_empty_ancestors(self) -> None:
        dp = DreamPhase(_mock_backend(_offspring_json(1)), _mock_memory([]))
        dp.dream("task")
        assert dp.dream_count == 0

    def test_min_fitness_filter_applied(self) -> None:
        mem = _mock_memory()
        # min_fitness=0.5 should be forwarded to get_top_ancestors
        dp = DreamPhase(_mock_backend("[]"), mem, min_fitness=0.5)
        dp.dream("task")
        mem.get_top_ancestors.assert_called_once_with(n=5, min_fitness=0.5)

    def test_backend_called_with_dream_system(self) -> None:
        mem = _mock_memory([_ancestor(1)])
        backend = _mock_backend(_offspring_json(1))
        dp = DreamPhase(backend, mem)
        dp.dream("some task", n_offspring=1)
        call_kwargs = backend.generate.call_args[1]
        assert "system" in call_kwargs
        assert "dream" in call_kwargs["system"].lower() or "recombination" in call_kwargs["system"].lower()

    def test_system_prompt_used_from_genome_snapshot(self) -> None:
        from cambrian.agent import Genome

        ancestor = _ancestor(1, 0.95)
        ancestor["genome"]["system_prompt"] = "Think step by step."
        mem = _mock_memory([ancestor])
        raw = json.dumps([
            {"system_prompt": "Dream-inspired prompt.", "strategy": "cot", "temperature": 0.8}
        ])
        backend = _mock_backend(raw)
        dp = DreamPhase(backend, mem)
        result = dp.dream("task", n_offspring=1)
        assert len(result) == 1
        assert isinstance(result[0], Genome)
        assert result[0].system_prompt == "Dream-inspired prompt."
