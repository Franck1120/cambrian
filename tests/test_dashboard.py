# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.dashboard pure helper functions.

The Streamlit UI layer itself cannot be tested without a running Streamlit
server, but the data-parsing helpers are pure functions that can be tested
in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# _load_json
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadJson:
    """Tests for the _load_json helper."""

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        from cambrian.dashboard import _load_json
        result = _load_json(str(tmp_path / "does_not_exist.json"))
        assert result is None

    def test_returns_list_for_valid_json(self, tmp_path: Path) -> None:
        from cambrian.dashboard import _load_json
        data = [{"generation": 0, "agents": []}, {"generation": 1, "agents": []}]
        p = tmp_path / "log.json"
        p.write_text(json.dumps(data))
        result = _load_json(str(p))
        assert result == data

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        from cambrian.dashboard import _load_json
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }")
        result = _load_json(str(p))
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        from cambrian.dashboard import _load_json
        p = tmp_path / "empty.json"
        p.write_text("")
        result = _load_json(str(p))
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _flatten_evolve
# ─────────────────────────────────────────────────────────────────────────────


class TestFlattenEvolve:
    """Tests for the _flatten_evolve helper."""

    def _sample_log(self) -> list[dict[str, object]]:
        return [
            {
                "generation": 0,
                "agents": [
                    {"id": "a0", "fitness": 0.3},
                    {"id": "a1", "fitness": 0.7},
                ],
            },
            {
                "generation": 1,
                "agents": [
                    {"id": "b0", "fitness": 0.5},
                    {"id": "b1", "fitness": 0.9},
                ],
            },
        ]

    def test_gen_numbers_extracted(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        gens, _, _, _ = _flatten_evolve(self._sample_log())  # type: ignore[arg-type]
        assert gens == [0, 1]

    def test_best_fitness_per_gen(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        _, best, _, _ = _flatten_evolve(self._sample_log())  # type: ignore[arg-type]
        assert best == pytest.approx([0.7, 0.9])

    def test_mean_fitness_per_gen(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        _, _, mean, _ = _flatten_evolve(self._sample_log())  # type: ignore[arg-type]
        assert mean == pytest.approx([0.5, 0.7])

    def test_all_agents_flat(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        _, _, _, agents = _flatten_evolve(self._sample_log())  # type: ignore[arg-type]
        assert len(agents) == 4

    def test_generation_tag_added_to_agents(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        _, _, _, agents = _flatten_evolve(self._sample_log())  # type: ignore[arg-type]
        assert all("_generation" in a for a in agents)
        assert agents[0]["_generation"] == 0
        assert agents[2]["_generation"] == 1

    def test_skips_generations_with_no_agents(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        log: list[dict[str, object]] = [
            {"generation": 0, "agents": []},
            {"generation": 1, "agents": [{"id": "x", "fitness": 0.8}]},
        ]
        gens, best, _, _ = _flatten_evolve(log)  # type: ignore[arg-type]
        assert gens == [1]
        assert best == pytest.approx([0.8])

    def test_empty_log_returns_empty_lists(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        gens, best, mean, agents = _flatten_evolve([])
        assert gens == []
        assert best == []
        assert mean == []
        assert agents == []

    def test_none_fitness_treated_as_zero(self) -> None:
        from cambrian.dashboard import _flatten_evolve
        log: list[dict[str, object]] = [
            {"generation": 0, "agents": [{"id": "a", "fitness": None}]}
        ]
        _, best, _, _ = _flatten_evolve(log)  # type: ignore[arg-type]
        assert best == pytest.approx([0.0])
