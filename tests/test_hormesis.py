"""Tests for cambrian.hormesis — HormesisAdapter."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.hormesis import HormesisAdapter, HormesisEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: Optional[float], prompt: str = "default prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    if fitness is not None:
        a.fitness = fitness
    return a


def _backend(response: str = "reprogrammed") -> MagicMock:
    b = MagicMock()
    b.generate.return_value = response
    return b


from typing import Optional


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestHormesisAdapterInit:
    def test_defaults(self) -> None:
        adapter = HormesisAdapter(backend=_backend())
        assert adapter._stress_threshold == 0.5
        assert adapter._mild_cutoff == 0.33
        assert adapter._severe_cutoff == 0.66

    def test_history_starts_empty(self) -> None:
        adapter = HormesisAdapter(backend=_backend())
        assert adapter.history == []


# ---------------------------------------------------------------------------
# stress_level
# ---------------------------------------------------------------------------


class TestStressLevel:
    def test_above_threshold_no_stress(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        agent = _make_agent(0.8)
        assert a.stress_level(agent) == "none"

    def test_at_threshold_no_stress(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        agent = _make_agent(0.5)
        assert a.stress_level(agent) == "none"

    def test_mild_stress(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5, mild_cutoff=0.33)
        # fitness=0.45 → s = 1 - 0.45/0.5 = 0.1 < 0.33
        agent = _make_agent(0.45)
        assert a.stress_level(agent) == "mild"

    def test_moderate_stress(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5, severe_cutoff=0.66)
        # fitness=0.3 → s = 1 - 0.3/0.5 = 0.4 ∈ [0.33, 0.66)
        agent = _make_agent(0.3)
        assert a.stress_level(agent) == "moderate"

    def test_severe_stress(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5, severe_cutoff=0.66)
        # fitness=0.1 → s = 1 - 0.1/0.5 = 0.8 ≥ 0.66
        agent = _make_agent(0.1)
        assert a.stress_level(agent) == "severe"

    def test_none_fitness_treated_as_zero(self) -> None:
        a = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        agent = _make_agent(None)
        assert a.stress_level(agent) == "severe"


# ---------------------------------------------------------------------------
# stimulate — none
# ---------------------------------------------------------------------------


class TestStimulateNone:
    def test_healthy_agent_returned_unchanged(self) -> None:
        adapter = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        agent = _make_agent(0.9)
        result = adapter.stimulate(agent, "task")
        assert result is agent  # same object — no clone

    def test_no_history_on_healthy(self) -> None:
        adapter = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        agent = _make_agent(0.9)
        adapter.stimulate(agent, "task")
        assert adapter.history == []


# ---------------------------------------------------------------------------
# stimulate — mild
# ---------------------------------------------------------------------------


class TestStimulateMild:
    def _adapter(self) -> HormesisAdapter:
        return HormesisAdapter(
            backend=_backend(),
            stress_threshold=0.5,
            mild_cutoff=0.33,
            temp_mild_boost=0.15,
        )

    def test_clones_agent(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.45)
        result = adapter.stimulate(agent, "task")
        assert result is not agent

    def test_temperature_boosted(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.45)
        agent.genome.temperature = 0.7
        result = adapter.stimulate(agent, "task")
        assert pytest.approx(result.genome.temperature, abs=1e-9) == 0.85

    def test_prompt_unchanged_on_mild(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.45, "original prompt")
        result = adapter.stimulate(agent, "task")
        assert result.genome.system_prompt == "original prompt"

    def test_history_recorded(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.45)
        adapter.stimulate(agent, "task")
        assert len(adapter.history) == 1
        ev = adapter.history[0]
        assert isinstance(ev, HormesisEvent)
        assert ev.stress_level == "mild"

    def test_temp_clamped_at_max(self) -> None:
        adapter = HormesisAdapter(
            backend=_backend(),
            stress_threshold=0.5,
            mild_cutoff=0.33,
            temp_mild_boost=0.5,
            max_temperature=1.0,
        )
        agent = _make_agent(0.45)
        agent.genome.temperature = 0.9
        result = adapter.stimulate(agent, "task")
        assert result.genome.temperature == 1.0


# ---------------------------------------------------------------------------
# stimulate — moderate
# ---------------------------------------------------------------------------


class TestStimulateModerate:
    def _adapter(self) -> HormesisAdapter:
        return HormesisAdapter(
            backend=_backend(),
            stress_threshold=0.5,
            severe_cutoff=0.66,
            temp_moderate_boost=0.2,
        )

    def test_hint_appended_to_prompt(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.3, "base prompt")
        result = adapter.stimulate(agent, "task")
        assert "HORMESIS" in result.genome.system_prompt
        assert "base prompt" in result.genome.system_prompt

    def test_temperature_boosted(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.3)
        agent.genome.temperature = 0.5
        result = adapter.stimulate(agent, "task")
        assert pytest.approx(result.genome.temperature, abs=1e-9) == 0.7

    def test_history_level_moderate(self) -> None:
        adapter = self._adapter()
        adapter.stimulate(_make_agent(0.3), "task")
        assert adapter.history[0].stress_level == "moderate"


# ---------------------------------------------------------------------------
# stimulate — severe
# ---------------------------------------------------------------------------


class TestStimulateSevere:
    def _adapter(self, response: str = "reprogrammed") -> HormesisAdapter:
        return HormesisAdapter(
            backend=_backend(response),
            stress_threshold=0.5,
            severe_cutoff=0.66,
            temp_severe_boost=0.3,
        )

    def test_llm_called_for_reprogramming(self) -> None:
        adapter = self._adapter("new system prompt")
        agent = _make_agent(0.05)
        result = adapter.stimulate(agent, "task")
        assert result.genome.system_prompt == "new system prompt"

    def test_temperature_boosted_on_severe(self) -> None:
        adapter = self._adapter()
        agent = _make_agent(0.05)
        agent.genome.temperature = 0.5
        result = adapter.stimulate(agent, "task")
        assert pytest.approx(result.genome.temperature, abs=1e-9) == 0.8

    def test_llm_failure_fallback(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("API error")
        adapter = HormesisAdapter(backend=b, stress_threshold=0.5, severe_cutoff=0.66)
        agent = _make_agent(0.05, "original")
        result = adapter.stimulate(agent, "task")
        assert "original" in result.genome.system_prompt
        assert "HORMESIS" in result.genome.system_prompt

    def test_history_level_severe(self) -> None:
        adapter = self._adapter()
        adapter.stimulate(_make_agent(0.05), "task")
        assert adapter.history[0].stress_level == "severe"


# ---------------------------------------------------------------------------
# stimulate_population
# ---------------------------------------------------------------------------


class TestStimulatePopulation:
    def test_all_agents_processed(self) -> None:
        adapter = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        pop = [_make_agent(0.9), _make_agent(0.1), _make_agent(0.3)]
        result = adapter.stimulate_population(pop, "task")
        assert len(result) == 3

    def test_history_grows(self) -> None:
        adapter = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        pop = [_make_agent(0.1), _make_agent(0.2), _make_agent(0.3)]
        adapter.stimulate_population(pop, "task")
        # All three below threshold → 3 events
        assert len(adapter.history) == 3

    def test_history_returns_copy(self) -> None:
        adapter = HormesisAdapter(backend=_backend(), stress_threshold=0.5)
        adapter.stimulate(_make_agent(0.1), "t")
        h1 = adapter.history
        h1.clear()
        assert len(adapter.history) == 1
