"""Tests for cambrian.catalysis — CatalysisEngine and CatalystSelector."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.catalysis import CatalysisEngine, CatalystSelector, CatalysisEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float | None = None, prompt: str = "agent prompt here") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    if fitness is not None:
        a.fitness = fitness
    return a


def _mock_mutator(return_prompt: str = "mutated") -> MagicMock:
    m = MagicMock()
    mutated_agent = _make_agent(prompt=return_prompt)
    m.mutate.return_value = mutated_agent
    return m


# ---------------------------------------------------------------------------
# CatalystSelector
# ---------------------------------------------------------------------------


class TestCatalystSelectorInit:
    def test_defaults(self) -> None:
        sel = CatalystSelector()
        assert sel._fw == 0.6
        assert sel._vw == 0.2
        assert sel._sw == 0.2
        assert sel._min_fitness == 0.3


class TestCatalystSelectorSelect:
    def test_returns_none_on_empty_population(self) -> None:
        sel = CatalystSelector()
        assert sel.select([]) is None

    def test_returns_none_all_below_min_fitness(self) -> None:
        sel = CatalystSelector(min_fitness=0.5)
        pop = [_make_agent(0.1), _make_agent(0.2)]
        assert sel.select(pop) is None

    def test_returns_highest_fitness_candidate(self) -> None:
        sel = CatalystSelector(min_fitness=0.0)
        low = _make_agent(0.3, "short")
        high = _make_agent(0.9, "elaborate python math expert step by step")
        pop = [low, high]
        result = sel.select(pop)
        assert result is high

    def test_filters_below_min_fitness(self) -> None:
        sel = CatalystSelector(min_fitness=0.5)
        excluded = _make_agent(0.2, "elaborate python expert step by step math")
        included = _make_agent(0.6, "ok")
        result = sel.select([excluded, included])
        assert result is included

    def test_single_candidate_returned(self) -> None:
        sel = CatalystSelector(min_fitness=0.0)
        agent = _make_agent(0.7)
        assert sel.select([agent]) is agent


# ---------------------------------------------------------------------------
# CatalysisEngine — init
# ---------------------------------------------------------------------------


class TestCatalysisEngineInit:
    def test_defaults(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        assert engine._n == 3
        assert engine._header == "[CATALYST]"

    def test_events_starts_empty(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        assert engine.events == []


# ---------------------------------------------------------------------------
# _extract_snippet
# ---------------------------------------------------------------------------


class TestExtractSnippet:
    def test_extracts_first_n_sentences(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator(), inject_n_sentences=2)
        text = "First sentence. Second sentence. Third sentence."
        snippet = engine._extract_snippet(text)
        assert "First sentence." in snippet
        assert "Second sentence." in snippet
        assert "Third sentence." not in snippet

    def test_returns_all_when_fewer_sentences(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator(), inject_n_sentences=5)
        text = "Only one sentence."
        snippet = engine._extract_snippet(text)
        assert snippet == "Only one sentence."

    def test_empty_string(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        assert engine._extract_snippet("") == ""


# ---------------------------------------------------------------------------
# catalyse
# ---------------------------------------------------------------------------


class TestCatalyse:
    def test_calls_mutator(self) -> None:
        mutator = _mock_mutator()
        engine = CatalysisEngine(mutator=mutator, inject_n_sentences=2)
        target = _make_agent(0.4)
        catalyst = _make_agent(0.9, "Excellent strategy here. Use chain of thought.")
        engine.catalyse(target, catalyst, "task")
        mutator.mutate.assert_called_once()

    def test_target_prompt_restored_after_mutation(self) -> None:
        mutator = _mock_mutator()
        engine = CatalysisEngine(mutator=mutator)
        target = _make_agent(0.4, "original prompt")
        catalyst = _make_agent(0.9, "Best strategy. Always verify. Be concise.")
        engine.catalyse(target, catalyst, "task")
        assert target.genome.system_prompt == "original prompt"

    def test_prompt_restored_even_on_mutator_error(self) -> None:
        mutator = MagicMock()
        mutator.mutate.side_effect = RuntimeError("LLM error")
        engine = CatalysisEngine(mutator=mutator)
        target = _make_agent(0.4, "original prompt")
        catalyst = _make_agent(0.9, "Strategy. Go fast. Be correct.")
        with pytest.raises(RuntimeError):
            engine.catalyse(target, catalyst, "task")
        assert target.genome.system_prompt == "original prompt"

    def test_catalyst_not_modified(self) -> None:
        mutator = _mock_mutator()
        engine = CatalysisEngine(mutator=mutator)
        catalyst = _make_agent(0.9, "catalyst prompt here")
        target = _make_agent(0.4, "target prompt")
        engine.catalyse(target, catalyst, "task")
        assert catalyst.genome.system_prompt == "catalyst prompt here"

    def test_augmented_prompt_contains_catalyst_snippet(self) -> None:
        captured: list[str] = []

        def fake_mutate(agent: Agent, task: str) -> Agent:
            captured.append(agent.genome.system_prompt)
            return agent

        mutator = MagicMock()
        mutator.mutate.side_effect = fake_mutate

        engine = CatalysisEngine(mutator=mutator, inject_n_sentences=2)
        target = _make_agent(0.4, "target")
        catalyst = _make_agent(0.9, "Alpha strategy. Beta approach. Gamma method.")
        engine.catalyse(target, catalyst, "task")
        assert "[CATALYST]" in captured[0]
        assert "Alpha strategy." in captured[0]

    def test_event_recorded(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        target = _make_agent(0.4)
        catalyst = _make_agent(0.9, "Strategy. Be good.")
        engine.catalyse(target, catalyst, "task")
        assert len(engine.events) == 1
        ev = engine.events[0]
        assert isinstance(ev, CatalysisEvent)
        assert ev.target_id == target.agent_id
        assert ev.catalyst_id == catalyst.agent_id

    def test_events_returns_copy(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        engine.catalyse(_make_agent(0.4), _make_agent(0.9, "S. T."), "t")
        e1 = engine.events
        e1.clear()
        assert len(engine.events) == 1


# ---------------------------------------------------------------------------
# catalyse_population
# ---------------------------------------------------------------------------


class TestCatalysePopulation:
    def test_processes_all_agents(self) -> None:
        mutator = _mock_mutator()
        engine = CatalysisEngine(mutator=mutator)
        catalyst = _make_agent(0.9, "Strategy here. Be precise. Use examples.")
        pop = [_make_agent(0.3), _make_agent(0.4), _make_agent(0.2)]
        result = engine.catalyse_population(pop, catalyst, "task")
        assert len(result) == 3
        assert mutator.mutate.call_count == 3

    def test_events_grow_per_call(self) -> None:
        engine = CatalysisEngine(mutator=_mock_mutator())
        catalyst = _make_agent(0.9, "Good strategy. Use reasoning. Check work.")
        pop = [_make_agent(0.3), _make_agent(0.4)]
        engine.catalyse_population(pop, catalyst, "task")
        assert len(engine.events) == 2
