# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian/dpo.py — DPOPair, DPOSelector, DPOTrainer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.dpo import DPOPair, DPOSelector, DPOTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(fitness: float = 0.5, prompt: str = "agent prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(response: str = "refined system prompt") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


# ---------------------------------------------------------------------------
# DPOPair
# ---------------------------------------------------------------------------


class TestDPOPair:
    def test_pair_stores_chosen_and_rejected(self) -> None:
        chosen = _agent(fitness=0.8)
        rejected = _agent(fitness=0.3)
        pair = DPOPair(chosen=chosen, rejected=rejected, task="t", margin=0.5)
        assert pair.chosen is chosen
        assert pair.rejected is rejected
        assert pair.margin == pytest.approx(0.5)

    def test_pair_task_stored(self) -> None:
        pair = DPOPair(chosen=_agent(), rejected=_agent(), task="my task", margin=0.2)
        assert pair.task == "my task"


# ---------------------------------------------------------------------------
# DPOSelector
# ---------------------------------------------------------------------------


class TestDPOSelector:
    def test_build_pairs_adjacent_returns_pairs(self) -> None:
        sel = DPOSelector(beta=0.1, pair_strategy="adjacent")
        population = [_agent(fitness=0.2 * i) for i in range(1, 7)]
        pairs = sel.build_pairs(population, task="task")
        assert len(pairs) > 0
        assert all(isinstance(p, DPOPair) for p in pairs)

    def test_build_pairs_random_returns_pairs(self) -> None:
        sel = DPOSelector(beta=0.1, pair_strategy="random")
        population = [_agent(fitness=0.1 * i) for i in range(1, 7)]
        pairs = sel.build_pairs(population, task="task")
        assert len(pairs) > 0

    def test_build_pairs_empty_population(self) -> None:
        sel = DPOSelector()
        assert sel.build_pairs([], task="task") == []

    def test_build_pairs_single_agent(self) -> None:
        sel = DPOSelector()
        assert sel.build_pairs([_agent()], task="task") == []

    def test_compute_dpo_reward_positive_margin(self) -> None:
        sel = DPOSelector(beta=0.1)
        pair = DPOPair(chosen=_agent(0.8), rejected=_agent(0.3), task="t", margin=0.5)
        reward = sel.compute_dpo_reward(pair)
        assert reward >= 0.0
        assert reward <= 1.0

    def test_compute_dpo_reward_zero_margin(self) -> None:
        sel = DPOSelector(beta=0.1)
        pair = DPOPair(chosen=_agent(0.5), rejected=_agent(0.5), task="t", margin=0.0)
        reward = sel.compute_dpo_reward(pair)
        assert reward == pytest.approx(0.0)

    def test_compute_dpo_reward_clamped_high(self) -> None:
        sel = DPOSelector(beta=100.0)  # very large beta
        pair = DPOPair(chosen=_agent(1.0), rejected=_agent(0.0), task="t", margin=1.0)
        reward = sel.compute_dpo_reward(pair)
        assert reward <= 1.0

    def test_apply_modifies_chosen_fitness(self) -> None:
        sel = DPOSelector(beta=0.5)
        high = _agent(fitness=0.8)
        low = _agent(fitness=0.2)
        population = [high, low]
        original_high = high.fitness or 0.0
        sel.apply(population, task="task")
        # chosen (high fitness) should get a bonus
        assert (high.fitness or 0.0) >= original_high

    def test_apply_fitness_never_exceeds_one(self) -> None:
        sel = DPOSelector(beta=10.0)
        population = [_agent(fitness=0.9 + 0.01 * i) for i in range(6)]
        sel.apply(population, task="task")
        for a in population:
            assert (a.fitness or 0.0) <= 1.0

    def test_apply_returns_population(self) -> None:
        sel = DPOSelector()
        population = [_agent(fitness=float(i) / 5) for i in range(5)]
        result = sel.apply(population, task="task")
        assert result is population or isinstance(result, list)
        assert len(result) == 5

    def test_apply_empty_population(self) -> None:
        sel = DPOSelector()
        result = sel.apply([], task="task")
        assert result == []

    def test_invalid_pair_strategy_raises_on_init(self) -> None:
        """DPOSelector raises ValueError on construction with invalid strategy."""
        with pytest.raises(ValueError):
            DPOSelector(pair_strategy="invalid")


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------


class TestDPOTrainer:
    def test_collect_pairs_returns_pairs(self) -> None:
        backend = _mock_backend()
        trainer = DPOTrainer(backend=backend, beta=0.1)
        population = [_agent(fitness=0.1 * i) for i in range(1, 9)]
        pairs = trainer.collect_pairs(population, task="task")
        assert isinstance(pairs, list)

    def test_collect_pairs_empty_population(self) -> None:
        backend = _mock_backend()
        trainer = DPOTrainer(backend=backend)
        assert trainer.collect_pairs([], task="task") == []

    def test_refine_returns_agent(self) -> None:
        backend = _mock_backend("refined: expert step-by-step reasoning")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        agent = _agent(fitness=0.4, prompt="original weak prompt")
        pairs = [
            DPOPair(
                chosen=_agent(fitness=0.9, prompt="expert analytical prompt"),
                rejected=_agent(fitness=0.1, prompt="bad prompt"),
                task="task",
                margin=0.8,
            )
        ]
        result = trainer.refine(agent, pairs, task="task")
        assert isinstance(result, Agent)
        assert isinstance(result.genome.system_prompt, str)

    def test_refine_backend_error_fallback(self) -> None:
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("LLM down")
        trainer = DPOTrainer(backend=backend)
        agent = _agent(fitness=0.3, prompt="original prompt")
        result = trainer.refine(agent, [], task="task")
        assert isinstance(result, Agent)

    def test_refine_with_empty_pairs(self) -> None:
        backend = _mock_backend("refined prompt")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        agent = _agent(fitness=0.4)
        result = trainer.refine(agent, [], task="task")
        assert isinstance(result, Agent)

    def test_train_returns_list(self) -> None:
        backend = _mock_backend("improved prompt")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        population = [_agent(fitness=0.1 * i) for i in range(1, 7)]
        result = trainer.train(population, task="task", n_pairs=2)
        assert isinstance(result, list)
        assert len(result) == len(population)

    def test_train_empty_population(self) -> None:
        backend = _mock_backend()
        trainer = DPOTrainer(backend=backend)
        result = trainer.train([], task="task")
        assert result == []

    def test_train_single_agent(self) -> None:
        backend = _mock_backend("single prompt")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        result = trainer.train([_agent(fitness=0.5)], task="task")
        assert len(result) == 1

    def test_train_preserves_top_agents(self) -> None:
        """Top-50% agents should not have their prompts rewritten."""
        backend = _mock_backend("rewritten")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        high = _agent(fitness=0.9, prompt="KEEP THIS PROMPT")
        low = _agent(fitness=0.1, prompt="bad prompt")
        result = trainer.train([high, low], task="task", n_pairs=1)
        # top agent should be unchanged
        top = max(result, key=lambda a: a.fitness or 0.0)
        assert "KEEP THIS PROMPT" in top.genome.system_prompt

    def test_train_all_agents_have_genomes(self) -> None:
        backend = _mock_backend("new prompt")
        trainer = DPOTrainer(backend=backend, n_refinements=1)
        population = [_agent(fitness=0.2 * i) for i in range(6)]
        result = trainer.train(population, task="task")
        for a in result:
            assert isinstance(a.genome.system_prompt, str)
            assert len(a.genome.system_prompt) > 0
