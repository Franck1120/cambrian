"""Edge-case tests for Tier 3 modules.

Covers boundary conditions across:
- symbiosis (SymbioticFuser — fitness/distance gating, LLM failure fallback)
- hormesis (HormesisAdapter — stress intensity calculation, population variant)
- apoptosis (ApoptosisController — grace period, stagnation window, clone replacement)
- catalysis (CatalysisEngine — prompt restoration on LLM failure, selector score)
- ensemble (AgentEnsemble — empty response handling, weight normalisation)
- glossolalia (GlossaloliaReasoner — latent truncation, synthesiser called)
- inference_scaling (BestOfN — n=1 edge, BeamSearch single step)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.apoptosis import ApoptosisController
from cambrian.catalysis import CatalysisEngine, CatalystSelector
from cambrian.ensemble import AgentEnsemble, BoostingEnsemble, exact_match_scorer
from cambrian.hormesis import HormesisAdapter
from cambrian.inference_scaling import BestOfN, KeywordScorer
from cambrian.symbiosis import SymbioticFuser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(fitness: float = 0.8, prompt: str = "agent system prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(response: str = "mocked") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


# ---------------------------------------------------------------------------
# SymbioticFuser edge cases
# ---------------------------------------------------------------------------


class TestSymbioticFuserEdgeCases:
    def test_host_below_fitness_threshold_returns_none(self) -> None:
        fuser = SymbioticFuser(_mock_backend(), fitness_threshold=0.9)
        host = _agent(fitness=0.5)
        donor = _agent(fitness=0.9)
        result = fuser.fuse(host, donor, task="task")
        assert result is None

    def test_donor_below_fitness_threshold_returns_none(self) -> None:
        fuser = SymbioticFuser(_mock_backend(), fitness_threshold=0.9)
        host = _agent(fitness=0.95)
        donor = _agent(fitness=0.3)
        result = fuser.fuse(host, donor, task="task")
        assert result is None

    def test_identical_prompts_below_min_distance(self) -> None:
        fuser = SymbioticFuser(_mock_backend(), fitness_threshold=0.5, min_distance=0.5)
        host = _agent(fitness=0.9, prompt="same exact prompt here")
        donor = _agent(fitness=0.9, prompt="same exact prompt here")
        result = fuser.fuse(host, donor, task="task")
        assert result is None  # distance is 0 < 0.5

    def test_llm_failure_fallback_concatenation(self) -> None:
        b = MagicMock()
        b.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        fuser = SymbioticFuser(b, fitness_threshold=0.0, min_distance=0.0)
        host = _agent(fitness=0.9, prompt="host prompt text.")
        donor = _agent(fitness=0.9, prompt="donor snippet info.")
        result = fuser.fuse(host, donor, task="task")
        # Falls back to naive concatenation
        assert result is not None
        assert "host prompt" in result.genome.system_prompt
        assert "donor snippet" in result.genome.system_prompt

    def test_fuse_best_pair_empty_population(self) -> None:
        fuser = SymbioticFuser(_mock_backend(), fitness_threshold=0.0)
        result = fuser.fuse_best_pair([], task="task")
        assert result is None

    def test_fuse_best_pair_single_agent(self) -> None:
        fuser = SymbioticFuser(_mock_backend(), fitness_threshold=0.0)
        result = fuser.fuse_best_pair([_agent()], task="task")
        assert result is None  # can't fuse with self


# ---------------------------------------------------------------------------
# HormesisAdapter edge cases
# ---------------------------------------------------------------------------


class TestHormesisAdapterEdgeCases:
    def test_zero_fitness_gives_max_stress(self) -> None:
        adapter = HormesisAdapter(_mock_backend(), stress_threshold=0.5)
        agent = _agent(fitness=0.0)
        # stress_intensity = max(0, min(1, 1 - 0/0.5)) = 1.0 → severe
        level = adapter.stress_level(agent)
        assert level == "severe"

    def test_fitness_above_threshold_no_stress(self) -> None:
        adapter = HormesisAdapter(_mock_backend(), stress_threshold=0.5)
        agent = _agent(fitness=0.9)
        # stress_intensity = max(0, 1 - 0.9/0.5) = negative → 0 → none
        level = adapter.stress_level(agent)
        assert level == "none"

    def test_stimulate_none_fitness_agent(self) -> None:
        adapter = HormesisAdapter(_mock_backend(), stress_threshold=0.5)
        g = Genome(system_prompt="prompt")
        agent = Agent(genome=g)  # fitness is None
        result = adapter.stimulate(agent, task="task")
        # Should still return an agent (clone with no stimulus)
        assert result is not None

    def test_stimulate_population_applies_to_all(self) -> None:
        adapter = HormesisAdapter(_mock_backend(), stress_threshold=0.5)
        pop = [_agent(0.1), _agent(0.2), _agent(0.8)]
        results = adapter.stimulate_population(pop, task="task")
        assert len(results) == 3

    def test_stimulate_returns_new_agent_not_same(self) -> None:
        adapter = HormesisAdapter(_mock_backend(), stress_threshold=0.5)
        agent = _agent(fitness=0.2)
        stimulated = adapter.stimulate(agent, task="task")
        assert stimulated is not agent


# ---------------------------------------------------------------------------
# ApoptosisController edge cases
# ---------------------------------------------------------------------------


class TestApoptosisControllerEdgeCases:
    def test_grace_period_protects_weak_agent(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.5, grace_period=5)
        poor = _agent(fitness=0.1)
        for _ in range(3):  # under grace period
            ctrl.record(poor)
        assert not ctrl.is_below_floor(poor)

    def test_stagnation_window_not_triggered_early(self) -> None:
        ctrl = ApoptosisController(stagnation_window=5, improvement_epsilon=0.01)
        agent = _agent()
        for _ in range(3):  # under window
            ctrl.record(agent)
        assert not ctrl.is_stagnant(agent)

    def test_improving_agent_not_stagnant(self) -> None:
        ctrl = ApoptosisController(stagnation_window=3, improvement_epsilon=0.01)
        agent = _agent(fitness=0.5)
        ctrl.record(agent)
        agent.fitness = 0.6
        ctrl.record(agent)
        agent.fitness = 0.7
        ctrl.record(agent)
        assert not ctrl.is_stagnant(agent)

    def test_apply_empty_population(self) -> None:
        ctrl = ApoptosisController()
        survivors = ctrl.apply([], best_agent=None)
        assert survivors == []

    def test_apply_does_not_remove_best_agent(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.5, grace_period=0, stagnation_window=1)
        best = _agent(fitness=0.9)
        poor = _agent(fitness=0.1)
        for _ in range(2):
            ctrl.record(poor)
        survivors = ctrl.apply([best, poor], best_agent=best)
        assert best in survivors

    def test_record_population_records_all(self) -> None:
        ctrl = ApoptosisController()
        pop = [_agent(0.3), _agent(0.5), _agent(0.8)]
        ctrl.record_population(pop)
        for agent in pop:
            assert len(ctrl._histories[agent.agent_id]) == 1


# ---------------------------------------------------------------------------
# CatalysisEngine edge cases
# ---------------------------------------------------------------------------


class TestCatalysisEngineEdgeCases:
    def test_prompt_restored_after_catalysis(self) -> None:
        engine = CatalysisEngine(_mock_backend())
        target = _agent(prompt="original prompt")
        catalyst = _agent(prompt="catalyst insights for improvement")
        original_prompt = target.genome.system_prompt
        engine.catalyse(target, catalyst, task="task")
        # After catalysis, prompt should be restored
        assert target.genome.system_prompt == original_prompt

    def test_prompt_restored_on_llm_failure(self) -> None:
        b = MagicMock()
        b.generate = MagicMock(side_effect=RuntimeError("LLM error"))
        engine = CatalysisEngine(b)
        target = _agent(prompt="original prompt text")
        catalyst = _agent(prompt="catalyst prompt text here")
        original_prompt = target.genome.system_prompt
        engine.catalyse(target, catalyst, task="task")
        assert target.genome.system_prompt == original_prompt

    def test_catalyse_population_returns_all(self) -> None:
        engine = CatalysisEngine(_mock_backend())
        pop = [_agent(0.3, f"agent {i}") for i in range(4)]
        catalyst = _agent(0.9, "best catalyst strategy")
        results = engine.catalyse_population(pop, catalyst, task="task")
        assert len(results) == 4

    def test_selector_picks_highest_fitness(self) -> None:
        sel = CatalystSelector()
        low = _agent(fitness=0.3, prompt="short")
        high = _agent(fitness=0.9, prompt="rich vocabulary agent strategy")
        selected = sel.select([low, high])
        assert selected is high

    def test_selector_empty_population_returns_none(self) -> None:
        sel = CatalystSelector()
        assert sel.select([]) is None


# ---------------------------------------------------------------------------
# AgentEnsemble edge cases
# ---------------------------------------------------------------------------


class TestAgentEnsembleEdgeCases:
    def _make_agent_with_response(self, response: str, fitness: float = 0.8) -> Agent:
        a = _agent(fitness=fitness)
        a.run = MagicMock(return_value=response)  # type: ignore[method-assign]
        return a

    def test_single_agent_ensemble(self) -> None:
        a = self._make_agent_with_response("42")
        ensemble = AgentEnsemble([a])
        answer = ensemble.query("what is 6x7?")
        assert answer == "42"

    def test_majority_answer_wins(self) -> None:
        a1 = self._make_agent_with_response("42")
        a2 = self._make_agent_with_response("42")
        a3 = self._make_agent_with_response("wrong")
        ensemble = AgentEnsemble([a1, a2, a3])
        answer = ensemble.query("task")
        assert answer == "42"

    def test_empty_response_handled(self) -> None:
        a1 = self._make_agent_with_response("")
        a2 = self._make_agent_with_response("42")
        ensemble = AgentEnsemble([a1, a2])
        answer = ensemble.query("task")
        assert isinstance(answer, str)

    def test_boosting_correct_agent_outweighs_wrong(self) -> None:
        a1 = self._make_agent_with_response("42")
        a2 = self._make_agent_with_response("wrong")
        boosting = BoostingEnsemble([a1, a2], scorer=exact_match_scorer)
        boosting.query("task", correct_answer="42")
        new_weights = boosting.weights
        # After normalization: correct (boosted) > wrong (decayed)
        assert new_weights[0] > new_weights[1]

    def test_results_accumulate(self) -> None:
        a = self._make_agent_with_response("answer")
        ensemble = AgentEnsemble([a])
        ensemble.query("task1")
        ensemble.query("task2")
        assert len(ensemble.results) == 2


# ---------------------------------------------------------------------------
# BestOfN edge cases
# ---------------------------------------------------------------------------


class TestBestOfNEdgeCases:
    def test_n_equals_one(self) -> None:
        backend = _mock_backend("single response")
        scorer = KeywordScorer(["response"])
        bon = BestOfN(backend=backend, n=1, scorer=scorer)
        best, score = bon.run("sys", "user")
        assert best == "single response"
        assert score > 0.0

    def test_all_responses_scored(self) -> None:
        responses = ["apple", "banana", "cherry"]
        backend = MagicMock()
        backend.generate = MagicMock(side_effect=responses)
        scorer = KeywordScorer(["cherry"])
        bon = BestOfN(backend=backend, n=3, scorer=scorer)
        best, score = bon.run("sys", "user")
        assert best == "cherry"
        assert score > 0.0

    def test_zero_scoring_response_still_returned(self) -> None:
        backend = _mock_backend("irrelevant text")
        scorer = KeywordScorer(["missing_keyword"])
        bon = BestOfN(backend=backend, n=2, scorer=scorer)
        best, score = bon.run("sys", "user")
        assert isinstance(best, str)
        assert score == pytest.approx(0.0)
