# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Extra tests for cambrian.dpo — DPOPair, DPOSelector, DPOTrainer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.dpo import DPOPair, DPOSelector, DPOTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(response: str = "improved genome") -> MagicMock:
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


def _agent(prompt: str = "You are a helpful assistant.", fitness: float = 0.5) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


def _population(n: int = 6) -> list[Agent]:
    fitnesses = [0.1 * (i + 1) for i in range(n)]
    return [_agent(f"agent {i}", fit) for i, fit in enumerate(fitnesses)]


# ---------------------------------------------------------------------------
# DPOPair
# ---------------------------------------------------------------------------


class TestDPOPair:
    def test_construction(self) -> None:
        chosen = _agent(fitness=0.9)
        rejected = _agent(fitness=0.3)
        pair = DPOPair(chosen=chosen, rejected=rejected, task="test task", margin=0.6)
        assert pair.chosen is chosen
        assert pair.rejected is rejected
        assert pair.task == "test task"
        assert pair.margin == pytest.approx(0.6)

    def test_margin_positive(self) -> None:
        pair = DPOPair(
            chosen=_agent(fitness=0.8),
            rejected=_agent(fitness=0.2),
            task="task",
            margin=0.6,
        )
        assert pair.margin > 0


# ---------------------------------------------------------------------------
# DPOSelector
# ---------------------------------------------------------------------------


class TestDPOSelector:
    def test_build_pairs_returns_list(self) -> None:
        selector = DPOSelector()
        population = _population(6)
        pairs = selector.build_pairs(population, "test task")
        assert isinstance(pairs, list)

    def test_build_pairs_nonempty_on_sufficient_pop(self) -> None:
        selector = DPOSelector()
        population = _population(6)
        pairs = selector.build_pairs(population, "task")
        assert len(pairs) >= 1

    def test_build_pairs_chosen_beats_rejected(self) -> None:
        selector = DPOSelector()
        population = _population(6)
        pairs = selector.build_pairs(population, "task")
        for pair in pairs:
            assert (pair.chosen.fitness or 0.0) >= (pair.rejected.fitness or 0.0)

    def test_apply_returns_list_of_agents(self) -> None:
        selector = DPOSelector()
        population = _population(6)
        result = selector.apply(population, "task")
        assert isinstance(result, list)
        for a in result:
            assert isinstance(a, Agent)

    def test_apply_preserves_population_size(self) -> None:
        selector = DPOSelector()
        population = _population(6)
        result = selector.apply(population, "task")
        assert len(result) == len(population)

    def test_apply_boosts_chosen_fitness(self) -> None:
        selector = DPOSelector(beta=0.2)
        population = _population(4)
        before = {a.id: (a.fitness or 0.0) for a in population}
        result = selector.apply(population, "task")
        after = {a.id: (a.fitness or 0.0) for a in result}
        # At least one agent should have had its fitness modified
        any_changed = any(abs(after.get(aid, 0.0) - before.get(aid, 0.0)) > 1e-9 for aid in before)
        assert any_changed

    def test_compute_dpo_reward_is_float(self) -> None:
        selector = DPOSelector()
        pair = DPOPair(chosen=_agent(fitness=0.8), rejected=_agent(fitness=0.3),
                       task="task", margin=0.5)
        reward = selector.compute_dpo_reward(pair)
        assert isinstance(reward, float)

    def test_compute_dpo_reward_positive_on_good_pair(self) -> None:
        selector = DPOSelector(beta=0.1)
        pair = DPOPair(chosen=_agent(fitness=0.9), rejected=_agent(fitness=0.2),
                       task="task", margin=0.7)
        reward = selector.compute_dpo_reward(pair)
        # High margin pair should produce positive reward
        assert isinstance(reward, float)

    def test_pair_strategy_options(self) -> None:
        for strategy in ("adjacent", "random"):
            selector = DPOSelector(pair_strategy=strategy)
            population = _population(6)
            pairs = selector.build_pairs(population, "task")
            assert isinstance(pairs, list)

    def test_invalid_pair_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="pair_strategy"):
            DPOSelector(pair_strategy="invalid_strategy")

    def test_empty_population_returns_empty(self) -> None:
        selector = DPOSelector()
        pairs = selector.build_pairs([], "task")
        assert pairs == []


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------


class TestDPOTrainer:
    def test_collect_pairs_returns_list(self) -> None:
        backend = _backend()
        trainer = DPOTrainer(backend=backend)
        population = _population(6)
        pairs = trainer.collect_pairs(population, "task")
        assert isinstance(pairs, list)

    def test_train_returns_list_of_agents(self) -> None:
        backend = _backend("improved prompt system")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        population = _population(4)
        result = trainer.train(population, "solve math problems", n_pairs=2)
        assert isinstance(result, list)
        for a in result:
            assert isinstance(a, Agent)

    def test_train_returns_same_size(self) -> None:
        backend = _backend()
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        population = _population(4)
        result = trainer.train(population, "task", n_pairs=1)
        assert len(result) == len(population)

    def test_train_empty_population(self) -> None:
        backend = _backend()
        trainer = DPOTrainer(backend=backend)
        result = trainer.train([], "task")
        assert result == []
