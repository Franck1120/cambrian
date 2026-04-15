"""Tests for cambrian.ecosystem — EcologicalRole, EcosystemConfig,
EcosystemEvent, EcosystemInteraction, and EcosystemEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.ecosystem import (
    EcologicalRole,
    EcosystemConfig,
    EcosystemEvent,
    EcosystemEvaluator,
    EcosystemInteraction,
)
from cambrian.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    strategy: str = "step-by-step",
    fitness: float | None = None,
    agent_id: str | None = None,
) -> Agent:
    """Convenience factory for test agents."""
    g = Genome(strategy=strategy)
    a = Agent(genome=g, agent_id=agent_id)
    if fitness is not None:
        a.fitness = fitness
    return a


def _mock_evaluator(return_value: float = 0.5) -> Evaluator:
    """Return a mock Evaluator that always returns *return_value*."""
    ev: MagicMock = MagicMock(spec=Evaluator)
    ev.evaluate.return_value = return_value
    return ev  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# EcologicalRole
# ---------------------------------------------------------------------------


class TestEcologicalRole:
    def test_herbivore_value(self) -> None:
        assert EcologicalRole.HERBIVORE == "herbivore"

    def test_predator_value(self) -> None:
        assert EcologicalRole.PREDATOR == "predator"

    def test_decomposer_value(self) -> None:
        assert EcologicalRole.DECOMPOSER == "decomposer"

    def test_parasite_value(self) -> None:
        assert EcologicalRole.PARASITE == "parasite"

    def test_is_str_subclass(self) -> None:
        assert isinstance(EcologicalRole.HERBIVORE, str)

    def test_four_members(self) -> None:
        assert len(EcologicalRole) == 4

    def test_string_conversion(self) -> None:
        assert EcologicalRole("herbivore") is EcologicalRole.HERBIVORE
        assert EcologicalRole("predator") is EcologicalRole.PREDATOR


# ---------------------------------------------------------------------------
# EcosystemConfig
# ---------------------------------------------------------------------------


class TestEcosystemConfig:
    def test_default_herbivore_diversity_bonus(self) -> None:
        cfg = EcosystemConfig()
        assert cfg.herbivore_diversity_bonus == pytest.approx(0.05)

    def test_default_predator_hunt_threshold(self) -> None:
        assert EcosystemConfig().predator_hunt_threshold == pytest.approx(0.3)

    def test_default_predator_hunt_bonus(self) -> None:
        assert EcosystemConfig().predator_hunt_bonus == pytest.approx(0.1)

    def test_default_decomposer_recycle_threshold(self) -> None:
        assert EcosystemConfig().decomposer_recycle_threshold == pytest.approx(0.25)

    def test_default_decomposer_bonus(self) -> None:
        assert EcosystemConfig().decomposer_bonus == pytest.approx(0.08)

    def test_default_parasite_host_threshold(self) -> None:
        assert EcosystemConfig().parasite_host_threshold == pytest.approx(0.7)

    def test_default_parasite_drain(self) -> None:
        assert EcosystemConfig().parasite_drain == pytest.approx(0.03)

    def test_default_parasite_gain(self) -> None:
        assert EcosystemConfig().parasite_gain == pytest.approx(0.06)

    def test_custom_values(self) -> None:
        cfg = EcosystemConfig(herbivore_diversity_bonus=0.1, predator_hunt_bonus=0.2)
        assert cfg.herbivore_diversity_bonus == pytest.approx(0.1)
        assert cfg.predator_hunt_bonus == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# EcosystemEvent
# ---------------------------------------------------------------------------


class TestEcosystemEvent:
    def test_fields_stored(self) -> None:
        ev = EcosystemEvent(
            role=EcologicalRole.PREDATOR,
            agent_id="abc",
            target_id="xyz",
            delta=0.1,
            event_type="hunt",
        )
        assert ev.role is EcologicalRole.PREDATOR
        assert ev.agent_id == "abc"
        assert ev.target_id == "xyz"
        assert ev.delta == pytest.approx(0.1)
        assert ev.event_type == "hunt"

    def test_target_id_can_be_none(self) -> None:
        ev = EcosystemEvent(
            role=EcologicalRole.HERBIVORE,
            agent_id="h1",
            target_id=None,
            delta=0.05,
            event_type="forage",
        )
        assert ev.target_id is None

    def test_negative_delta(self) -> None:
        ev = EcosystemEvent(
            role=EcologicalRole.PARASITE,
            agent_id="host1",
            target_id="p1",
            delta=-0.03,
            event_type="parasite",
        )
        assert ev.delta < 0


# ---------------------------------------------------------------------------
# EcosystemInteraction — init
# ---------------------------------------------------------------------------


class TestEcosystemInteractionInit:
    def test_default_config(self) -> None:
        ei = EcosystemInteraction()
        assert isinstance(ei._config, EcosystemConfig)

    def test_custom_config(self) -> None:
        cfg = EcosystemConfig(predator_hunt_bonus=0.5)
        ei = EcosystemInteraction(config=cfg)
        assert ei._config.predator_hunt_bonus == pytest.approx(0.5)

    def test_roles_start_empty(self) -> None:
        ei = EcosystemInteraction()
        assert ei._roles == {}

    def test_events_start_empty(self) -> None:
        ei = EcosystemInteraction()
        assert ei.events == []


# ---------------------------------------------------------------------------
# assign_role / get_role
# ---------------------------------------------------------------------------


class TestAssignGetRole:
    def test_roundtrip(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent()
        ei.assign_role(agent, EcologicalRole.PREDATOR)
        assert ei.get_role(agent) is EcologicalRole.PREDATOR

    def test_get_role_unregistered_returns_none(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent()
        assert ei.get_role(agent) is None

    def test_overwrite_role(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent()
        ei.assign_role(agent, EcologicalRole.HERBIVORE)
        ei.assign_role(agent, EcologicalRole.PARASITE)
        assert ei.get_role(agent) is EcologicalRole.PARASITE


# ---------------------------------------------------------------------------
# auto_assign
# ---------------------------------------------------------------------------


class TestAutoAssign:
    def _make_population(self, fitnesses: list[float]) -> list[Agent]:
        return [_make_agent(fitness=f) for f in fitnesses]

    def test_all_agents_get_a_role(self) -> None:
        ei = EcosystemInteraction()
        pop = self._make_population([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        ei.auto_assign(pop)
        for agent in pop:
            assert ei.get_role(agent) is not None

    def test_top_20_percent_become_predator(self) -> None:
        ei = EcosystemInteraction()
        # 10 agents — top 2 (fitness 0.9, 1.0) → PREDATOR
        fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pop = self._make_population(fitnesses)
        ei.auto_assign(pop)
        predators = [a for a in pop if ei.get_role(a) is EcologicalRole.PREDATOR]
        predator_fitnesses = sorted([a.fitness for a in predators], reverse=True)  # type: ignore[type-var]
        # highest fitness agents are predators
        assert 0.9 in predator_fitnesses or 1.0 in predator_fitnesses

    def test_bottom_20_percent_become_decomposer(self) -> None:
        ei = EcosystemInteraction()
        fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pop = self._make_population(fitnesses)
        ei.auto_assign(pop)
        decomposers = [a for a in pop if ei.get_role(a) is EcologicalRole.DECOMPOSER]
        decomposer_fitnesses = [a.fitness for a in decomposers]
        # lowest fitness agents are decomposers
        assert any(f <= 0.2 for f in decomposer_fitnesses)

    def test_single_agent_gets_a_role(self) -> None:
        ei = EcosystemInteraction()
        pop = [_make_agent(fitness=0.5)]
        ei.auto_assign(pop)
        assert ei.get_role(pop[0]) is not None

    def test_population_of_5(self) -> None:
        ei = EcosystemInteraction()
        pop = self._make_population([0.1, 0.3, 0.5, 0.7, 0.9])
        ei.auto_assign(pop)
        for agent in pop:
            assert ei.get_role(agent) is not None

    def test_large_population_10(self) -> None:
        ei = EcosystemInteraction()
        pop = self._make_population([i / 10.0 for i in range(1, 11)])
        ei.auto_assign(pop)
        counts = ei.role_counts()
        # Predators assigned
        assert counts.get("predator", 0) >= 1
        # Decomposers assigned
        assert counts.get("decomposer", 0) >= 1

    def test_none_fitness_handled(self) -> None:
        ei = EcosystemInteraction()
        agents = [_make_agent(fitness=None) for _ in range(5)]
        ei.auto_assign(agents)  # should not raise
        for a in agents:
            assert ei.get_role(a) is not None


# ---------------------------------------------------------------------------
# interact — Herbivore
# ---------------------------------------------------------------------------


class TestInteractHerbivore:
    def test_herbivore_gets_diversity_bonus_for_unique_strategies(self) -> None:
        ei = EcosystemInteraction()
        h = _make_agent(strategy="explore", fitness=0.5)
        other1 = _make_agent(strategy="chain-of-thought", fitness=0.5)
        other2 = _make_agent(strategy="socratic", fitness=0.5)
        pop = [h, other1, other2]
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        events = ei.interact(pop, "task")
        herb_events = [e for e in events if e.agent_id == h.agent_id]
        assert len(herb_events) == 1
        # 2 unique strategies from others → bonus = 2 * 0.05 = 0.10
        assert herb_events[0].delta == pytest.approx(0.10)

    def test_herbivore_homogeneous_population(self) -> None:
        """When all agents share the same strategy, unique count = 1 → small bonus."""
        ei = EcosystemInteraction()
        h = _make_agent(strategy="same", fitness=0.5)
        other1 = _make_agent(strategy="same", fitness=0.5)
        other2 = _make_agent(strategy="same", fitness=0.5)
        pop = [h, other1, other2]
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        events = ei.interact(pop, "task")
        herb_events = [e for e in events if e.agent_id == h.agent_id]
        # Only 1 unique strategy among others
        assert herb_events[0].delta == pytest.approx(0.05)

    def test_herbivore_does_not_count_own_strategy(self) -> None:
        ei = EcosystemInteraction()
        h = _make_agent(strategy="unique-herb", fitness=0.5)
        pop = [h]
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        events = ei.interact(pop, "task")
        herb_events = [e for e in events if e.agent_id == h.agent_id]
        # No other agents → 0 unique strategies → delta = 0
        assert herb_events[0].delta == pytest.approx(0.0)

    def test_herbivore_event_type_is_forage(self) -> None:
        ei = EcosystemInteraction()
        h = _make_agent(fitness=0.5)
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        events = ei.interact([h], "task")
        assert events[0].event_type == "forage"

    def test_herbivore_target_id_is_none(self) -> None:
        ei = EcosystemInteraction()
        h = _make_agent(fitness=0.5)
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        events = ei.interact([h], "task")
        assert events[0].target_id is None


# ---------------------------------------------------------------------------
# interact — Predator
# ---------------------------------------------------------------------------


class TestInteractPredator:
    def test_predator_hunts_prey_below_threshold(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(strategy="aggressive", fitness=0.8)
        prey = _make_agent(strategy="weak", fitness=0.1)
        pop = [pred, prey]
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        events = ei.interact(pop, "task")
        pred_events = [e for e in events if e.agent_id == pred.agent_id]
        assert len(pred_events) == 1
        assert pred_events[0].delta == pytest.approx(0.1)

    def test_predator_no_bonus_when_no_prey(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(fitness=0.8)
        strong = _make_agent(fitness=0.9)
        pop = [pred, strong]
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        events = ei.interact(pop, "task")
        pred_events = [e for e in events if e.agent_id == pred.agent_id]
        assert pred_events[0].delta == pytest.approx(0.0)

    def test_predator_event_type_is_hunt(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(fitness=0.8)
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        events = ei.interact([pred], "task")
        assert events[0].event_type == "hunt"

    def test_multiple_predators_all_get_hunt_bonus(self) -> None:
        ei = EcosystemInteraction()
        pred1 = _make_agent(strategy="p1", fitness=0.9)
        pred2 = _make_agent(strategy="p2", fitness=0.85)
        prey = _make_agent(strategy="weak", fitness=0.05)
        pop = [pred1, pred2, prey]
        ei.assign_role(pred1, EcologicalRole.PREDATOR)
        ei.assign_role(pred2, EcologicalRole.PREDATOR)
        events = ei.interact(pop, "task")
        pred1_events = [e for e in events if e.agent_id == pred1.agent_id]
        pred2_events = [e for e in events if e.agent_id == pred2.agent_id]
        assert pred1_events[0].delta > 0
        assert pred2_events[0].delta > 0

    def test_predator_targets_weakest_prey(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(fitness=0.9)
        weak_prey = _make_agent(strategy="weak", fitness=0.05)
        medium_prey = _make_agent(strategy="medium", fitness=0.2)
        pop = [pred, weak_prey, medium_prey]
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        events = ei.interact(pop, "task")
        pred_events = [e for e in events if e.agent_id == pred.agent_id]
        assert pred_events[0].target_id == weak_prey.agent_id


# ---------------------------------------------------------------------------
# interact — Decomposer
# ---------------------------------------------------------------------------


class TestInteractDecomposer:
    def test_decomposer_gets_bonus_for_recyclable_agents(self) -> None:
        ei = EcosystemInteraction()
        dec = _make_agent(fitness=0.5)
        recyclable = _make_agent(fitness=0.1)
        pop = [dec, recyclable]
        ei.assign_role(dec, EcologicalRole.DECOMPOSER)
        events = ei.interact(pop, "task")
        dec_events = [e for e in events if e.agent_id == dec.agent_id]
        assert dec_events[0].delta == pytest.approx(0.08)

    def test_decomposer_no_bonus_when_no_recyclable(self) -> None:
        ei = EcosystemInteraction()
        dec = _make_agent(fitness=0.5)
        healthy = _make_agent(fitness=0.9)
        pop = [dec, healthy]
        ei.assign_role(dec, EcologicalRole.DECOMPOSER)
        events = ei.interact(pop, "task")
        dec_events = [e for e in events if e.agent_id == dec.agent_id]
        assert dec_events[0].delta == pytest.approx(0.0)

    def test_decomposer_event_type_is_recycle(self) -> None:
        ei = EcosystemInteraction()
        dec = _make_agent(fitness=0.5)
        ei.assign_role(dec, EcologicalRole.DECOMPOSER)
        events = ei.interact([dec], "task")
        assert events[0].event_type == "recycle"


# ---------------------------------------------------------------------------
# interact — Parasite
# ---------------------------------------------------------------------------


class TestInteractParasite:
    def test_parasite_drains_host_above_threshold(self) -> None:
        ei = EcosystemInteraction()
        parasite = _make_agent(strategy="latch", fitness=0.4)
        host = _make_agent(strategy="strong", fitness=0.9)
        pop = [parasite, host]
        ei.assign_role(parasite, EcologicalRole.PARASITE)
        events = ei.interact(pop, "task")
        parasite_gain_events = [e for e in events if e.agent_id == parasite.agent_id]
        host_drain_events = [e for e in events if e.agent_id == host.agent_id]
        assert len(parasite_gain_events) == 1
        assert parasite_gain_events[0].delta == pytest.approx(0.06)
        assert len(host_drain_events) == 1
        assert host_drain_events[0].delta == pytest.approx(-0.03)

    def test_parasite_no_effect_when_no_eligible_hosts(self) -> None:
        ei = EcosystemInteraction()
        parasite = _make_agent(fitness=0.4)
        weak = _make_agent(fitness=0.1)
        pop = [parasite, weak]
        ei.assign_role(parasite, EcologicalRole.PARASITE)
        events = ei.interact(pop, "task")
        parasite_events = [e for e in events if e.agent_id == parasite.agent_id]
        assert len(parasite_events) == 0

    def test_parasite_targets_strongest_host(self) -> None:
        ei = EcosystemInteraction()
        parasite = _make_agent(fitness=0.4)
        host1 = _make_agent(strategy="h1", fitness=0.75)
        host2 = _make_agent(strategy="h2", fitness=0.95)
        pop = [parasite, host1, host2]
        ei.assign_role(parasite, EcologicalRole.PARASITE)
        events = ei.interact(pop, "task")
        parasite_gain_events = [e for e in events if e.agent_id == parasite.agent_id]
        assert parasite_gain_events[0].target_id == host2.agent_id

    def test_parasite_event_type_is_parasite(self) -> None:
        ei = EcosystemInteraction()
        parasite = _make_agent(fitness=0.4)
        host = _make_agent(fitness=0.9)
        pop = [parasite, host]
        ei.assign_role(parasite, EcologicalRole.PARASITE)
        events = ei.interact(pop, "task")
        assert all(e.event_type == "parasite" for e in events)


# ---------------------------------------------------------------------------
# interact — general
# ---------------------------------------------------------------------------


class TestInteractGeneral:
    def test_returns_list_of_ecosystem_events(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent(fitness=0.5)
        ei.assign_role(agent, EcologicalRole.HERBIVORE)
        result = ei.interact([agent], "task")
        assert isinstance(result, list)
        assert all(isinstance(e, EcosystemEvent) for e in result)

    def test_empty_population_returns_empty_list(self) -> None:
        ei = EcosystemInteraction()
        result = ei.interact([], "task")
        assert result == []

    def test_unassigned_agents_produce_no_events(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent(fitness=0.5)
        # no role assigned
        events = ei.interact([agent], "task")
        assert events == []

    def test_agents_with_none_fitness_handled(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(fitness=0.9)
        no_fitness = _make_agent(fitness=None)
        pop = [pred, no_fitness]
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        # None fitness treated as 0.0 → below hunt threshold
        events = ei.interact(pop, "task")
        pred_events = [e for e in events if e.agent_id == pred.agent_id]
        assert pred_events[0].delta > 0  # hunted the None-fitness agent


# ---------------------------------------------------------------------------
# apply_events
# ---------------------------------------------------------------------------


class TestApplyEvents:
    def test_clamps_fitness_to_one(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent(fitness=0.95)
        ev = EcosystemEvent(
            role=EcologicalRole.HERBIVORE,
            agent_id=agent.agent_id,
            target_id=None,
            delta=0.2,
            event_type="forage",
        )
        ei.apply_events([ev], [agent])
        assert agent.fitness == pytest.approx(1.0)

    def test_clamps_fitness_to_zero(self) -> None:
        ei = EcosystemInteraction()
        agent = _make_agent(fitness=0.02)
        ev = EcosystemEvent(
            role=EcologicalRole.PARASITE,
            agent_id=agent.agent_id,
            target_id=None,
            delta=-0.5,
            event_type="parasite",
        )
        ei.apply_events([ev], [agent])
        assert agent.fitness == pytest.approx(0.0)

    def test_increases_predator_fitness(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(fitness=0.5)
        ev = EcosystemEvent(
            role=EcologicalRole.PREDATOR,
            agent_id=pred.agent_id,
            target_id=None,
            delta=0.1,
            event_type="hunt",
        )
        ei.apply_events([ev], [pred])
        assert pred.fitness == pytest.approx(0.6)

    def test_decreases_host_fitness(self) -> None:
        ei = EcosystemInteraction()
        host = _make_agent(fitness=0.9)
        ev = EcosystemEvent(
            role=EcologicalRole.PARASITE,
            agent_id=host.agent_id,
            target_id=None,
            delta=-0.03,
            event_type="parasite",
        )
        ei.apply_events([ev], [host])
        assert host.fitness == pytest.approx(0.87)

    def test_combined_interact_and_apply(self) -> None:
        ei = EcosystemInteraction()
        pred = _make_agent(strategy="strong", fitness=0.8)
        prey = _make_agent(strategy="weak", fitness=0.1)
        pop = [pred, prey]
        ei.assign_role(pred, EcologicalRole.PREDATOR)
        events = ei.interact(pop, "task")
        ei.apply_events(events, pop)
        assert pred.fitness is not None
        assert pred.fitness > 0.8


# ---------------------------------------------------------------------------
# role_counts
# ---------------------------------------------------------------------------


class TestRoleCounts:
    def test_correct_counts(self) -> None:
        ei = EcosystemInteraction()
        agents = [_make_agent() for _ in range(5)]
        ei.assign_role(agents[0], EcologicalRole.PREDATOR)
        ei.assign_role(agents[1], EcologicalRole.PREDATOR)
        ei.assign_role(agents[2], EcologicalRole.HERBIVORE)
        ei.assign_role(agents[3], EcologicalRole.DECOMPOSER)
        ei.assign_role(agents[4], EcologicalRole.PARASITE)
        counts = ei.role_counts()
        assert counts["predator"] == 2
        assert counts["herbivore"] == 1
        assert counts["decomposer"] == 1
        assert counts["parasite"] == 1

    def test_empty_dict_when_no_roles(self) -> None:
        ei = EcosystemInteraction()
        assert ei.role_counts() == {}


# ---------------------------------------------------------------------------
# events property
# ---------------------------------------------------------------------------


class TestEventsProperty:
    def test_events_accumulate_across_calls(self) -> None:
        ei = EcosystemInteraction()
        h = _make_agent(fitness=0.5)
        ei.assign_role(h, EcologicalRole.HERBIVORE)
        ei.interact([h], "task1")
        ei.interact([h], "task2")
        assert len(ei.events) == 2

    def test_events_returns_list(self) -> None:
        ei = EcosystemInteraction()
        assert isinstance(ei.events, list)


# ---------------------------------------------------------------------------
# EcosystemEvaluator
# ---------------------------------------------------------------------------


class TestEcosystemEvaluator:
    def test_returns_float_in_0_1(self) -> None:
        base_ev = _mock_evaluator(return_value=0.6)
        interaction = EcosystemInteraction()
        ev = EcosystemEvaluator(base_evaluator=base_ev, interaction=interaction)
        agent = _make_agent(fitness=0.5)
        score = ev.evaluate(agent, "task")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calls_base_evaluator(self) -> None:
        base_ev = _mock_evaluator(return_value=0.7)
        interaction = EcosystemInteraction()
        ev = EcosystemEvaluator(base_evaluator=base_ev, interaction=interaction)
        agent = _make_agent(fitness=0.5)
        ev.evaluate(agent, "task")
        base_ev.evaluate.assert_called_once_with(agent, "task")  # type: ignore[attr-defined]

    def test_blending_weights(self) -> None:
        """With weight=0.5, score = 0.5*base + 0.5*eco."""
        base_ev = _mock_evaluator(return_value=0.8)
        interaction = EcosystemInteraction()
        ev = EcosystemEvaluator(
            base_evaluator=base_ev,
            interaction=interaction,
            interaction_weight=0.5,
        )
        agent = _make_agent(fitness=0.4)
        score = ev.evaluate(agent, "task")
        expected = 0.5 * 0.8 + 0.5 * 0.4
        assert score == pytest.approx(expected)

    def test_none_fitness_defaults_to_0_5(self) -> None:
        base_ev = _mock_evaluator(return_value=0.0)
        interaction = EcosystemInteraction()
        ev = EcosystemEvaluator(
            base_evaluator=base_ev,
            interaction=interaction,
            interaction_weight=1.0,
        )
        agent = _make_agent(fitness=None)
        score = ev.evaluate(agent, "task")
        # weight=1.0 → all eco; eco_score for None fitness = 0.5
        assert score == pytest.approx(0.5)
