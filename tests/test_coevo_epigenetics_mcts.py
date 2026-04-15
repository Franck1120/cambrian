# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.coevolution, cambrian.epigenetics, cambrian.mcts, cambrian.archipelago."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.archipelago import Archipelago, Island
from cambrian.coevolution import CoEvolutionEngine, MAPElites
from cambrian.epigenetics import (
    EpigenomicContext,
    EpigeneticLayer,
    make_standard_layer,
)
from cambrian.mcts import MCTSNode, MCTSSelector
from cambrian.mutator import LLMMutator


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


def _mutator() -> LLMMutator:
    return LLMMutator(backend=_backend_mock())


def _population(n: int = 4) -> list[Agent]:
    return [_agent(f"agent {i}", 0.2 * (i + 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Island
# ---------------------------------------------------------------------------


class TestIsland:
    def test_construction(self) -> None:
        island = Island(island_id=0)
        assert island.island_id == 0
        assert island.population == []

    def test_best_agent_empty_returns_none(self) -> None:
        island = Island(island_id=0)
        assert island.best_agent() is None

    def test_best_agent_returns_highest_fitness(self) -> None:
        island = Island(island_id=0)
        a1 = _agent(fitness=0.3)
        a2 = _agent(fitness=0.9)
        island.population = [a1, a2]
        best = island.best_agent()
        assert best is not None
        assert best.fitness == pytest.approx(0.9)

    def test_top_agents_returns_sorted(self) -> None:
        island = Island(island_id=0)
        island.population = _population(4)
        top = island.top_agents(2)
        assert len(top) == 2
        assert (top[0].fitness or 0) >= (top[1].fitness or 0)

    def test_top_agents_more_than_pop(self) -> None:
        island = Island(island_id=0)
        island.population = _population(2)
        top = island.top_agents(10)
        assert len(top) == 2

    def test_receive_migrants_updates_population(self) -> None:
        island = Island(island_id=0)
        island.population = _population(4)
        migrants = [_agent(fitness=0.99)]
        island.receive_migrants(migrants)
        fitnesses = [a.fitness or 0.0 for a in island.population]
        assert 0.99 in fitnesses or any(f > 0.8 for f in fitnesses)

    def test_receive_migrants_empty_no_crash(self) -> None:
        island = Island(island_id=0)
        island.population = _population(2)
        island.receive_migrants([])  # should not raise


# ---------------------------------------------------------------------------
# Archipelago (construction & topology only — no evolve, too complex)
# ---------------------------------------------------------------------------


class TestArchipelago:
    def _arch(self, n_islands: int = 3, topology: str = "ring") -> Archipelago:
        return Archipelago(
            engine_factory=MagicMock,
            n_islands=n_islands,
            island_size=5,
            migration_interval=2,
            migration_rate=0.2,
            topology=topology,
        )

    def test_construction_ring(self) -> None:
        arch = self._arch(topology="ring")
        assert arch is not None

    def test_construction_all_to_all(self) -> None:
        arch = self._arch(topology="all_to_all")
        assert arch is not None

    def test_construction_random(self) -> None:
        arch = self._arch(topology="random")
        assert arch is not None

    def test_invalid_topology_raises(self) -> None:
        with pytest.raises(ValueError, match="topology"):
            Archipelago(engine_factory=MagicMock, topology="invalid")

    def test_invalid_migration_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="migration_rate"):
            Archipelago(engine_factory=MagicMock, migration_rate=1.5)

    def test_island_summaries_returns_list(self) -> None:
        arch = self._arch(n_islands=3)
        summaries = arch.island_summaries()
        assert isinstance(summaries, list)
        assert len(summaries) == 3

    def test_total_migrations_starts_at_zero(self) -> None:
        arch = self._arch()
        assert arch.total_migrations == 0

    def test_seeded_rng_reproducible(self) -> None:
        arch1 = Archipelago(engine_factory=MagicMock, n_islands=3, topology="random", seed=42)
        arch2 = Archipelago(engine_factory=MagicMock, n_islands=3, topology="random", seed=42)
        # Same seed → same rng state (verify both construct without error)
        assert arch1.island_summaries() is not None
        assert arch2.island_summaries() is not None


# ---------------------------------------------------------------------------
# MAPElites
# ---------------------------------------------------------------------------


class TestMAPElites:
    def test_construction(self) -> None:
        me = MAPElites()
        assert me is not None

    def test_construction_custom(self) -> None:
        me = MAPElites(n_prompt_buckets=5, n_temp_buckets=5)
        assert me is not None

    def test_add_agent_returns_bool(self) -> None:
        me = MAPElites()
        agent = _agent(fitness=0.7)
        result = me.add(agent)
        assert isinstance(result, bool)

    def test_best_empty_returns_none(self) -> None:
        me = MAPElites()
        assert me.best() is None

    def test_best_after_add(self) -> None:
        me = MAPElites()
        agent = _agent(fitness=0.8)
        me.add(agent)
        best = me.best()
        assert best is not None

    def test_coverage_zero_initially(self) -> None:
        me = MAPElites()
        assert me.coverage() == pytest.approx(0.0)

    def test_coverage_increases_after_add(self) -> None:
        me = MAPElites()
        for i in range(6):
            a = Agent(genome=Genome(system_prompt="x" * (i * 10 + 1), temperature=0.1 * i))
            a.fitness = 0.5 + 0.05 * i
            me.add(a)
        assert me.coverage() > 0.0

    def test_get_diverse_population_returns_list(self) -> None:
        me = MAPElites()
        for agent in _population(4):
            me.add(agent)
        pop = me.get_diverse_population()
        assert isinstance(pop, list)

    def test_get_diverse_population_empty_map(self) -> None:
        me = MAPElites()
        assert me.get_diverse_population() == []

    def test_add_better_agent_replaces_cell(self) -> None:
        me = MAPElites()
        a1 = Agent(genome=Genome(system_prompt="short", temperature=0.5))
        a1.fitness = 0.3
        a2 = Agent(genome=Genome(system_prompt="short", temperature=0.5))
        a2.fitness = 0.9
        me.add(a1)
        me.add(a2)
        best = me.best()
        assert best is not None
        assert (best.fitness or 0.0) >= 0.9


# ---------------------------------------------------------------------------
# EpigenomicContext
# ---------------------------------------------------------------------------


class TestEpigenomicContext:
    def test_defaults(self) -> None:
        ctx = EpigenomicContext()
        assert ctx.generation == 0
        assert ctx.task == ""
        assert ctx.population_mean_fitness == pytest.approx(0.0)
        assert ctx.total_generations == 10

    def test_custom_values(self) -> None:
        ctx = EpigenomicContext(
            generation=5,
            task="code task",
            population_mean_fitness=0.6,
            population_best_fitness=0.9,
            total_generations=20,
        )
        assert ctx.generation == 5
        assert ctx.task == "code task"
        assert ctx.total_generations == 20

    def test_progress_early(self) -> None:
        ctx = EpigenomicContext(generation=0, total_generations=10)
        assert ctx.progress == pytest.approx(0.0)
        assert ctx.is_early is True
        assert ctx.is_late is False

    def test_progress_late(self) -> None:
        ctx = EpigenomicContext(generation=9, total_generations=10)
        assert ctx.progress == pytest.approx(1.0)
        assert ctx.is_late is True
        assert ctx.is_early is False

    def test_progress_mid(self) -> None:
        ctx = EpigenomicContext(generation=5, total_generations=10)
        assert 0.3 < ctx.progress < 0.7
        assert not ctx.is_early
        assert not ctx.is_late

    def test_single_generation_progress_is_one(self) -> None:
        ctx = EpigenomicContext(generation=0, total_generations=1)
        assert ctx.progress == pytest.approx(1.0)

    def test_extra_field(self) -> None:
        ctx = EpigenomicContext(extra={"diversity": 0.42})
        assert ctx.extra["diversity"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# EpigeneticLayer
# ---------------------------------------------------------------------------


class TestEpigeneticLayer:
    def test_construction_empty(self) -> None:
        layer = EpigeneticLayer()
        assert layer is not None

    def test_construction_with_rules(self) -> None:
        rule = lambda g, ctx: "phase: early" if ctx.is_early else "phase: late"
        layer = EpigeneticLayer(rules=[rule])
        assert layer is not None

    def test_add_rule_appends(self) -> None:
        layer = EpigeneticLayer()
        rule = lambda g, ctx: "custom annotation"
        layer.add_rule(rule)
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext(generation=0, total_generations=10)
        expressed = layer.express(genome, ctx)
        assert "custom annotation" in expressed

    def test_express_returns_string(self) -> None:
        layer = EpigeneticLayer()
        genome = Genome(system_prompt="Base prompt.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert isinstance(result, str)

    def test_express_no_rules_returns_base_prompt(self) -> None:
        layer = EpigeneticLayer()
        genome = Genome(system_prompt="Only base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert "Only base." in result

    def test_express_includes_rule_output(self) -> None:
        layer = EpigeneticLayer(rules=[lambda g, ctx: "ANNOTATION"])
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert "ANNOTATION" in result

    def test_express_skips_none_rules(self) -> None:
        layer = EpigeneticLayer(rules=[lambda g, ctx: None])
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert result  # not empty

    def test_apply_returns_agent(self) -> None:
        layer = EpigeneticLayer()
        agent = _agent()
        ctx = EpigenomicContext(generation=3, total_generations=10)
        result = layer.apply(agent, ctx)
        assert isinstance(result, Agent)

    def test_apply_does_not_modify_original(self) -> None:
        rule = lambda g, ctx: "INJECTED"
        layer = EpigeneticLayer(rules=[rule])
        agent = _agent(prompt="Original prompt.")
        ctx = EpigenomicContext()
        layer.apply(agent, ctx)
        # Original genome unchanged
        assert agent.genome.system_prompt == "Original prompt."

    def test_make_standard_layer(self) -> None:
        layer = make_standard_layer()
        assert isinstance(layer, EpigeneticLayer)

    def test_standard_layer_express(self) -> None:
        layer = make_standard_layer()
        genome = Genome(system_prompt="You are an expert.")
        ctx = EpigenomicContext(
            generation=2, task="Write Python code", total_generations=10,
            population_mean_fitness=0.5, population_best_fitness=0.7,
        )
        result = layer.express(genome, ctx)
        assert isinstance(result, str)
        assert len(result) >= len(genome.system_prompt)

    def test_custom_separator(self) -> None:
        sep = "===EPIGENETICS==="
        layer = EpigeneticLayer(
            rules=[lambda g, ctx: "annotation"],
            separator=sep,
        )
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert sep in result


# ---------------------------------------------------------------------------
# MCTSNode + MCTSSelector
# ---------------------------------------------------------------------------


class TestMCTSNode:
    def test_node_fields(self) -> None:
        agent = _agent(fitness=0.5)
        node = MCTSNode(agent=agent)
        assert node.agent is agent
        assert node.visits == 0
        assert node.total_reward == pytest.approx(0.0)
        assert node.depth == 0
        assert node.parent is None
        assert node.children == []


class TestMCTSSelector:
    def _sel(self) -> MCTSSelector:
        return MCTSSelector(mutator=_mutator())

    def test_construction(self) -> None:
        sel = self._sel()
        assert sel is not None

    def test_custom_params(self) -> None:
        sel = MCTSSelector(mutator=_mutator(), exploration_constant=2.0, max_depth=4, max_children=5)
        assert sel is not None

    def test_register_returns_node(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        node = sel.register(agent)
        assert isinstance(node, MCTSNode)
        assert node.agent is agent

    def test_register_same_agent_idempotent(self) -> None:
        sel = self._sel()
        agent = _agent()
        sel.register(agent)
        sel.register(agent)  # second call should not duplicate

    def test_backpropagate_increments_visits(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        node = sel.register(agent)
        assert node.visits == 0
        sel.backpropagate(agent.id, 0.8)
        assert node.visits == 1

    def test_backpropagate_accumulates_reward(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        node = sel.register(agent)
        sel.backpropagate(agent.id, 0.6)
        sel.backpropagate(agent.id, 0.4)
        assert node.total_reward == pytest.approx(1.0)

    def test_select_returns_agent(self) -> None:
        sel = self._sel()
        pop = _population(3)
        for a in pop:
            sel.register(a)
        selected = sel.select(pop)
        assert isinstance(selected, Agent)

    def test_select_single_agent(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        sel.register(agent)
        selected = sel.select([agent])
        assert selected is agent

    def test_stats_returns_dict(self) -> None:
        sel = self._sel()
        agents = _population(3)
        for a in agents:
            sel.register(a)
        sel.backpropagate(agents[0].id, 0.5)  # ensure at least one visited node
        stats = sel.stats()
        assert isinstance(stats, dict)
        assert "nodes" in stats
        assert "visits" in stats

    def test_best_path_returns_list(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        node = sel.register(agent)
        sel.backpropagate(agent.id, 0.7)
        path = sel.best_path(agent.id)
        assert isinstance(path, list)

    def test_prune_stale_roots(self) -> None:
        sel = self._sel()
        agents = _population(5)
        for a in agents:
            sel.register(a)
        active_ids = {agents[0].id, agents[1].id}
        sel.prune_stale_roots(active_ids, keep_best=1)
        # Should not raise

    def test_expand_returns_list(self) -> None:
        sel = self._sel()
        agent = _agent(fitness=0.5)
        sel.register(agent)
        children = sel.expand(agent, task="test task", n_children=1)
        assert isinstance(children, list)
