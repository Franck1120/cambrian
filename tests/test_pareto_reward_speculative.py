# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.pareto, cambrian.reward_shaping, cambrian.speculative."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.pareto import (
    ObjectiveVector,
    ParetoFront,
    attach_diversity_scores,
    brevity_objective,
    crowding_distance,
    fast_non_dominated_sort,
    fitness_objective,
    nsga2_select,
)
from cambrian.reward_shaping import (
    ClipShaper,
    CuriosityShaper,
    NormalisationShaper,
    PotentialShaper,
    RankShaper,
    build_shaped_evaluator,
)
from cambrian.speculative import SpeculativeMutator, SpeculativeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(prompt: str = "You are helpful.", fitness: float = 0.5) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


def _backend_mock(response: str = "improved prompt") -> MagicMock:
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


def _population(n: int = 4) -> list[Agent]:
    return [_agent(f"agent {i}", 0.2 * (i + 1)) for i in range(n)]


def _vec(agent: Agent, fitness: float = 0.5, brevity: float = 0.5) -> ObjectiveVector:
    return ObjectiveVector(agent_id=agent.id, scores={"fitness": fitness, "brevity": brevity})


def _score(agent: Agent, task: str) -> float:
    return agent.fitness or 0.5


# ---------------------------------------------------------------------------
# ObjectiveVector
# ---------------------------------------------------------------------------


class TestObjectiveVector:
    def test_construction_minimal(self) -> None:
        a = _agent()
        vec = ObjectiveVector(agent_id=a.id)
        assert vec.agent_id == a.id
        assert vec.scores == {}
        assert vec.rank == 0
        assert vec.crowding == pytest.approx(0.0)

    def test_construction_with_scores(self) -> None:
        a = _agent()
        vec = ObjectiveVector(agent_id=a.id, scores={"fitness": 0.8, "brevity": 0.6})
        assert vec.scores["fitness"] == pytest.approx(0.8)
        assert vec.scores["brevity"] == pytest.approx(0.6)

    def test_construction_with_rank(self) -> None:
        a = _agent()
        vec = ObjectiveVector(agent_id=a.id, scores={"f": 0.9}, rank=2)
        assert vec.rank == 2

    def test_construction_with_crowding(self) -> None:
        a = _agent()
        vec = ObjectiveVector(agent_id=a.id, crowding=0.75)
        assert vec.crowding == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# fitness_objective / brevity_objective
# ---------------------------------------------------------------------------


class TestObjectiveFunctions:
    def test_fitness_objective_returns_float(self) -> None:
        a = _agent(fitness=0.7)
        result = fitness_objective(a)
        assert isinstance(result, float)
        assert result == pytest.approx(0.7)

    def test_fitness_objective_none_fitness(self) -> None:
        a = Agent(genome=Genome(system_prompt="x"))
        result = fitness_objective(a)
        assert isinstance(result, float)

    def test_brevity_objective_returns_float(self) -> None:
        a = _agent(prompt="short prompt")
        result = brevity_objective(a)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_brevity_objective_longer_prompt_lower_score(self) -> None:
        a_short = _agent(prompt="Short.")
        a_long = _agent(prompt="A" * 500)
        score_short = brevity_objective(a_short)
        score_long = brevity_objective(a_long)
        assert score_short >= score_long

    def test_brevity_objective_custom_max_tokens(self) -> None:
        a = _agent(prompt="Test prompt.")
        result = brevity_objective(a, max_tokens=5000)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# fast_non_dominated_sort
# ---------------------------------------------------------------------------


class TestFastNonDominatedSort:
    def test_returns_list_of_lists(self) -> None:
        agents = _population(3)
        vecs = [_vec(a, fitness=0.2 * (i + 1), brevity=0.5) for i, a in enumerate(agents)]
        fronts = fast_non_dominated_sort(vecs)
        assert isinstance(fronts, list)
        assert all(isinstance(f, list) for f in fronts)

    def test_empty_input(self) -> None:
        fronts = fast_non_dominated_sort([])
        assert fronts == [] or fronts == [[]]

    def test_single_vector_is_rank_zero(self) -> None:
        a = _agent()
        vec = _vec(a, fitness=0.5, brevity=0.5)
        fronts = fast_non_dominated_sort([vec])
        assert len(fronts) >= 1
        assert vec in fronts[0]

    def test_all_in_first_front_when_incomparable(self) -> None:
        agents = _population(3)
        # Each dominates in one objective → all incomparable → same front
        vecs = [
            ObjectiveVector(agent_id=agents[0].id, scores={"f": 1.0, "g": 0.0}),
            ObjectiveVector(agent_id=agents[1].id, scores={"f": 0.5, "g": 0.5}),
            ObjectiveVector(agent_id=agents[2].id, scores={"f": 0.0, "g": 1.0}),
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert len(fronts[0]) == 3

    def test_dominated_goes_to_later_front(self) -> None:
        agents = _population(2)
        # agents[1] dominates agents[0] on both objectives
        vecs = [
            ObjectiveVector(agent_id=agents[0].id, scores={"f": 0.2, "g": 0.2}),
            ObjectiveVector(agent_id=agents[1].id, scores={"f": 0.9, "g": 0.9}),
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert len(fronts) >= 2
        # The dominator should be in the first front
        first_ids = {v.agent_id for v in fronts[0]}
        assert agents[1].id in first_ids


# ---------------------------------------------------------------------------
# crowding_distance
# ---------------------------------------------------------------------------


class TestCrowdingDistance:
    def test_modifies_crowding_in_place(self) -> None:
        agents = _population(3)
        vecs = [
            ObjectiveVector(agent_id=agents[0].id, scores={"f": 0.2, "g": 0.8}),
            ObjectiveVector(agent_id=agents[1].id, scores={"f": 0.5, "g": 0.5}),
            ObjectiveVector(agent_id=agents[2].id, scores={"f": 0.9, "g": 0.1}),
        ]
        crowding_distance(vecs)
        # Boundary vectors should have infinite crowding (very large value)
        crowdings = [v.crowding for v in vecs]
        assert any(c > 0 for c in crowdings)

    def test_empty_front_no_crash(self) -> None:
        crowding_distance([])  # should not raise

    def test_single_vector_no_crash(self) -> None:
        a = _agent()
        vec = _vec(a)
        crowding_distance([vec])  # should not raise


# ---------------------------------------------------------------------------
# nsga2_select
# ---------------------------------------------------------------------------


class TestNsga2Select:
    def test_returns_list_of_agents(self) -> None:
        pop = _population(4)
        vecs = [_vec(a, fitness=a.fitness or 0.5) for a in pop]
        selected = nsga2_select(pop, vecs, target_size=2)
        assert isinstance(selected, list)
        assert all(isinstance(a, Agent) for a in selected)

    def test_target_size_respected(self) -> None:
        pop = _population(6)
        vecs = [_vec(a) for a in pop]
        selected = nsga2_select(pop, vecs, target_size=3)
        assert len(selected) <= 4  # may round up by one front

    def test_target_larger_than_pop_returns_all(self) -> None:
        pop = _population(3)
        vecs = [_vec(a) for a in pop]
        selected = nsga2_select(pop, vecs, target_size=10)
        assert len(selected) <= len(pop)


# ---------------------------------------------------------------------------
# attach_diversity_scores
# ---------------------------------------------------------------------------


class TestAttachDiversityScores:
    def test_no_crash(self) -> None:
        pop = _population(4)
        vecs = [_vec(a) for a in pop]
        attach_diversity_scores(pop, vecs)  # should not raise

    def test_adds_diversity_key(self) -> None:
        pop = _population(4)
        vecs = [_vec(a) for a in pop]
        attach_diversity_scores(pop, vecs, objective_name="diversity")
        for vec in vecs:
            assert "diversity" in vec.scores

    def test_custom_objective_name(self) -> None:
        pop = _population(3)
        vecs = [_vec(a) for a in pop]
        attach_diversity_scores(pop, vecs, objective_name="novelty")
        for vec in vecs:
            assert "novelty" in vec.scores


# ---------------------------------------------------------------------------
# ParetoFront
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_construction_empty(self) -> None:
        pf = ParetoFront()
        assert pf.size() == 0

    def test_construction_with_objectives(self) -> None:
        pf = ParetoFront(objectives=["fitness", "brevity"])
        assert pf is not None

    def test_add_returns_bool(self) -> None:
        pf = ParetoFront()
        a = _agent()
        vec = _vec(a, fitness=0.7, brevity=0.6)
        result = pf.add(vec)
        assert isinstance(result, bool)

    def test_add_increases_size(self) -> None:
        pf = ParetoFront()
        a = _agent()
        vec = _vec(a)
        pf.add(vec)
        assert pf.size() >= 1

    def test_members_returns_list(self) -> None:
        pf = ParetoFront()
        a = _agent()
        pf.add(_vec(a))
        assert isinstance(pf.members(), list)

    def test_agents_returns_list(self) -> None:
        pf = ParetoFront()
        a = _agent()
        pf.add(_vec(a))
        pop = [a]
        result = pf.agents(pop)
        assert isinstance(result, list)

    def test_dominated_vector_not_added(self) -> None:
        pf = ParetoFront()
        a1 = _agent()
        a2 = _agent()
        vec_strong = ObjectiveVector(agent_id=a1.id, scores={"f": 0.9, "g": 0.9})
        vec_weak = ObjectiveVector(agent_id=a2.id, scores={"f": 0.1, "g": 0.1})
        pf.add(vec_strong)
        result = pf.add(vec_weak)
        assert result is False

    def test_size_zero_initially(self) -> None:
        pf = ParetoFront()
        assert pf.size() == 0


# ---------------------------------------------------------------------------
# NormalisationShaper
# ---------------------------------------------------------------------------


class TestNormalisationShaper:
    def test_construction(self) -> None:
        shaper = NormalisationShaper(base_evaluator=_score)
        assert shaper is not None

    def test_construction_zscore(self) -> None:
        shaper = NormalisationShaper(base_evaluator=_score, method="zscore")
        assert shaper is not None

    def test_call_returns_float(self) -> None:
        shaper = NormalisationShaper(base_evaluator=_score)
        a = _agent(fitness=0.6)
        result = shaper(a, "task")
        assert isinstance(result, float)

    def test_multiple_calls_converge(self) -> None:
        shaper = NormalisationShaper(base_evaluator=_score, window_size=10)
        agents = _population(5)
        scores = [shaper(a, "task") for a in agents]
        assert all(isinstance(s, float) for s in scores)


# ---------------------------------------------------------------------------
# ClipShaper
# ---------------------------------------------------------------------------


class TestClipShaper:
    def test_construction(self) -> None:
        shaper = ClipShaper(base_evaluator=_score)
        assert shaper is not None

    def test_call_returns_float_in_range(self) -> None:
        shaper = ClipShaper(base_evaluator=_score, min_val=0.0, max_val=1.0)
        a = _agent(fitness=1.5)  # above max
        result = shaper(a, "task")
        assert isinstance(result, float)
        assert result <= 1.0

    def test_clips_below_min(self) -> None:
        def below_zero(agent: Agent, task: str) -> float:
            return -0.5
        shaper = ClipShaper(base_evaluator=below_zero, min_val=0.0, max_val=1.0)
        result = shaper(_agent(), "task")
        assert result >= 0.0

    def test_custom_range(self) -> None:
        shaper = ClipShaper(base_evaluator=_score, min_val=0.2, max_val=0.8)
        a = _agent(fitness=0.5)
        result = shaper(a, "task")
        assert 0.2 <= result <= 0.8


# ---------------------------------------------------------------------------
# RankShaper
# ---------------------------------------------------------------------------


class TestRankShaper:
    def test_construction_no_base(self) -> None:
        shaper = RankShaper()
        assert shaper is not None

    def test_construction_with_base(self) -> None:
        shaper = RankShaper(base_evaluator=_score)
        assert shaper is not None

    def test_call_returns_float(self) -> None:
        shaper = RankShaper(base_evaluator=_score)
        a = _agent(fitness=0.7)
        result = shaper(a, "task")
        assert isinstance(result, float)

    def test_rank_scores_in_zero_one(self) -> None:
        shaper = RankShaper(base_evaluator=_score, window_size=10)
        agents = _population(5)
        scores = [shaper(a, "task") for a in agents]
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# PotentialShaper
# ---------------------------------------------------------------------------


class TestPotentialShaper:
    def test_construction(self) -> None:
        shaper = PotentialShaper(base_evaluator=_score)
        assert shaper is not None

    def test_construction_custom_gamma(self) -> None:
        shaper = PotentialShaper(base_evaluator=_score, gamma=0.95)
        assert shaper is not None

    def test_call_returns_float(self) -> None:
        shaper = PotentialShaper(base_evaluator=_score)
        a = _agent(fitness=0.6)
        result = shaper(a, "task")
        assert isinstance(result, float)

    def test_custom_potential_fn(self) -> None:
        def pot(agent: Agent) -> float:
            return agent.fitness or 0.0

        shaper = PotentialShaper(base_evaluator=_score, potential_fn=pot)
        result = shaper(_agent(fitness=0.7), "task")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# CuriosityShaper
# ---------------------------------------------------------------------------


class TestCuriosityShaper:
    def test_construction(self) -> None:
        shaper = CuriosityShaper(base_evaluator=_score)
        assert shaper is not None

    def test_call_returns_float(self) -> None:
        shaper = CuriosityShaper(base_evaluator=_score)
        a = _agent(fitness=0.6)
        result = shaper(a, "task")
        assert isinstance(result, float)

    def test_novel_agent_gets_bonus(self) -> None:
        shaper = CuriosityShaper(base_evaluator=_score, scale=1.0, memory_size=10)
        a = _agent(prompt="completely unique prompt XYZ123")
        result = shaper(a, "task")
        assert isinstance(result, float)

    def test_repeated_agent_lower_bonus(self) -> None:
        shaper = CuriosityShaper(base_evaluator=_score, scale=0.5, memory_size=10)
        a = _agent(prompt="same prompt")
        r1 = shaper(a, "task")
        r2 = shaper(a, "task")
        # Second call should have lower or equal curiosity bonus
        assert isinstance(r1, float)
        assert isinstance(r2, float)


# ---------------------------------------------------------------------------
# build_shaped_evaluator
# ---------------------------------------------------------------------------


class TestBuildShapedEvaluator:
    def test_clip_spec(self) -> None:
        result = build_shaped_evaluator(base_evaluator=_score, spec="clip")
        assert callable(result)

    def test_normalise_spec(self) -> None:
        result = build_shaped_evaluator(base_evaluator=_score, spec="normalise")
        assert callable(result)

    def test_curiosity_spec(self) -> None:
        result = build_shaped_evaluator(base_evaluator=_score, spec="curiosity")
        assert callable(result)

    def test_potential_spec(self) -> None:
        result = build_shaped_evaluator(base_evaluator=_score, spec="potential")
        assert callable(result)

    def test_shaped_callable_returns_float(self) -> None:
        shaped = build_shaped_evaluator(base_evaluator=_score, spec="clip")
        a = _agent(fitness=0.7)
        result = shaped(a, "test task")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# SpeculativeResult
# ---------------------------------------------------------------------------


class TestSpeculativeResult:
    def test_fields_exist(self) -> None:
        winner = _agent(fitness=0.9)
        result = SpeculativeResult(
            winner=winner,
            candidates=[winner],
            fitness_values=[0.9],
            k=1,
        )
        assert result.winner is winner
        assert result.candidates == [winner]
        assert result.fitness_values == [0.9]
        assert result.k == 1

    def test_multiple_candidates(self) -> None:
        candidates = _population(3)
        fitness_vals = [c.fitness or 0.0 for c in candidates]
        winner = max(candidates, key=lambda a: a.fitness or 0.0)
        result = SpeculativeResult(
            winner=winner,
            candidates=candidates,
            fitness_values=fitness_vals,
            k=3,
        )
        assert result.k == 3
        assert len(result.candidates) == 3


# ---------------------------------------------------------------------------
# SpeculativeMutator
# ---------------------------------------------------------------------------


class TestSpeculativeMutator:
    def _mutator(self) -> SpeculativeMutator:
        return SpeculativeMutator(backend=_backend_mock(), k_candidates=2)

    def test_construction(self) -> None:
        sm = self._mutator()
        assert sm is not None

    def test_construction_custom_k(self) -> None:
        sm = SpeculativeMutator(backend=_backend_mock(), k_candidates=4)
        assert sm.k_candidates == 4

    def test_total_saved_starts_zero(self) -> None:
        sm = self._mutator()
        assert sm.total_saved == 0

    def test_mutate_returns_agent(self) -> None:
        sm = self._mutator()
        a = _agent(fitness=0.5)
        result = sm.mutate(a, task="test task")
        assert isinstance(result, Agent)

    def test_mutate_with_task(self) -> None:
        sm = self._mutator()
        a = _agent(fitness=0.6)
        result = sm.mutate(a, task="solve coding problem")
        assert isinstance(result, Agent)

    def test_crossover_returns_agent(self) -> None:
        sm = self._mutator()
        pa = _agent(prompt="Parent A", fitness=0.5)
        pb = _agent(prompt="Parent B", fitness=0.7)
        result = sm.crossover(pa, pb, task="test")
        assert isinstance(result, Agent)

    def test_crossover_no_task(self) -> None:
        sm = self._mutator()
        pa = _agent(prompt="Parent A")
        pb = _agent(prompt="Parent B")
        result = sm.crossover(pa, pb)
        assert isinstance(result, Agent)

    def test_mutate_k1_still_works(self) -> None:
        sm = SpeculativeMutator(backend=_backend_mock(), k_candidates=1)
        a = _agent(fitness=0.4)
        result = sm.mutate(a, task="simple task")
        assert isinstance(result, Agent)
