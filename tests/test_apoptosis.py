# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.apoptosis — ApoptosisController."""
from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.apoptosis import ApoptosisController, ApoptosisEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float | None = None) -> Agent:
    g = Genome(system_prompt="agent prompt")
    a = Agent(genome=g)
    if fitness is not None:
        a.fitness = fitness
    return a


def _record_n(ctrl: ApoptosisController, agent: Agent, scores: list[float]) -> None:
    for s in scores:
        agent.fitness = s
        ctrl.record(agent)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self) -> None:
        ctrl = ApoptosisController()
        assert ctrl._stagnation_window == 5
        assert ctrl._improvement_epsilon == 0.01
        assert ctrl._min_fitness == 0.05
        assert ctrl._grace_period == 3
        assert ctrl._replace_with_clone is True

    def test_events_starts_empty(self) -> None:
        ctrl = ApoptosisController()
        assert ctrl.events == []


# ---------------------------------------------------------------------------
# record / record_population
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_appends_fitness(self) -> None:
        ctrl = ApoptosisController()
        agent = _make_agent(0.5)
        ctrl.record(agent)
        assert ctrl._histories[agent.agent_id] == [0.5]

    def test_record_multiple(self) -> None:
        ctrl = ApoptosisController()
        agent = _make_agent()
        for s in [0.1, 0.2, 0.3]:
            agent.fitness = s
            ctrl.record(agent)
        assert ctrl._histories[agent.agent_id] == [0.1, 0.2, 0.3]

    def test_record_skips_none_fitness(self) -> None:
        ctrl = ApoptosisController()
        agent = _make_agent(None)
        ctrl.record(agent)
        assert agent.agent_id not in ctrl._histories

    def test_record_population(self) -> None:
        ctrl = ApoptosisController()
        pop = [_make_agent(0.1), _make_agent(0.2), _make_agent(0.3)]
        ctrl.record_population(pop)
        assert len(ctrl._histories) == 3


# ---------------------------------------------------------------------------
# is_stagnant
# ---------------------------------------------------------------------------


class TestIsStagnant:
    def test_not_enough_history(self) -> None:
        ctrl = ApoptosisController(stagnation_window=5)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        assert ctrl.is_stagnant(agent) is False

    def test_stagnant_flat(self) -> None:
        ctrl = ApoptosisController(stagnation_window=3, improvement_epsilon=0.01)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        assert ctrl.is_stagnant(agent) is True

    def test_not_stagnant_improving(self) -> None:
        ctrl = ApoptosisController(stagnation_window=3, improvement_epsilon=0.01)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.3, 0.5, 0.7])
        assert ctrl.is_stagnant(agent) is False

    def test_uses_only_recent_window(self) -> None:
        # Old improvements don't rescue stagnation in recent window
        ctrl = ApoptosisController(stagnation_window=3, improvement_epsilon=0.01)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.1, 0.9, 0.9, 0.9, 0.9])
        assert ctrl.is_stagnant(agent) is True


# ---------------------------------------------------------------------------
# is_below_floor
# ---------------------------------------------------------------------------


class TestIsBelowFloor:
    def test_below_floor_after_grace(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.1, grace_period=3)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.05, 0.05, 0.05])
        agent.fitness = 0.05
        assert ctrl.is_below_floor(agent) is True

    def test_not_below_floor_during_grace(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.1, grace_period=5)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.05, 0.05])
        agent.fitness = 0.05
        assert ctrl.is_below_floor(agent) is False

    def test_above_floor(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.1, grace_period=3)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        assert ctrl.is_below_floor(agent) is False


# ---------------------------------------------------------------------------
# should_die
# ---------------------------------------------------------------------------


class TestShouldDie:
    def test_stagnant_should_die(self) -> None:
        ctrl = ApoptosisController(stagnation_window=3, improvement_epsilon=0.01)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        dead, reason = ctrl.should_die(agent)
        assert dead is True
        assert reason == "stagnation"

    def test_below_floor_should_die(self) -> None:
        ctrl = ApoptosisController(min_fitness=0.1, grace_period=3)
        agent = _make_agent()
        _record_n(ctrl, agent, [0.01, 0.01, 0.01])
        agent.fitness = 0.01
        dead, reason = ctrl.should_die(agent)
        assert dead is True
        assert reason == "floor"

    def test_healthy_should_not_die(self) -> None:
        ctrl = ApoptosisController()
        agent = _make_agent(0.8)
        dead, _ = ctrl.should_die(agent)
        assert dead is False


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_removes_stagnant(self) -> None:
        ctrl = ApoptosisController(
            stagnation_window=3, improvement_epsilon=0.01, replace_with_clone=False
        )
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        result = ctrl.apply([agent])
        assert len(result) == 0

    def test_keeps_healthy(self) -> None:
        ctrl = ApoptosisController(replace_with_clone=False)
        agent = _make_agent(0.9)
        result = ctrl.apply([agent])
        assert len(result) == 1

    def test_replaces_dead_with_clone_of_best(self) -> None:
        ctrl = ApoptosisController(
            stagnation_window=3, improvement_epsilon=0.01, replace_with_clone=True
        )
        best = _make_agent(0.95)
        doomed = _make_agent()
        _record_n(ctrl, doomed, [0.5, 0.5, 0.5])
        doomed.fitness = 0.5
        result = ctrl.apply([doomed], best_agent=best)
        assert len(result) == 1
        assert result[0].agent_id != doomed.agent_id

    def test_event_recorded_on_death(self) -> None:
        ctrl = ApoptosisController(
            stagnation_window=3, improvement_epsilon=0.01, replace_with_clone=False
        )
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        ctrl.apply([agent])
        assert len(ctrl.events) == 1
        ev = ctrl.events[0]
        assert isinstance(ev, ApoptosisEvent)
        assert ev.reason == "stagnation"
        assert ev.fitness_at_death == pytest.approx(0.5)

    def test_events_returns_copy(self) -> None:
        ctrl = ApoptosisController(
            stagnation_window=3, improvement_epsilon=0.01, replace_with_clone=False
        )
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        ctrl.apply([agent])
        e1 = ctrl.events
        e1.clear()
        assert len(ctrl.events) == 1

    def test_no_replace_when_no_best(self) -> None:
        ctrl = ApoptosisController(
            stagnation_window=3, improvement_epsilon=0.01, replace_with_clone=True
        )
        agent = _make_agent()
        _record_n(ctrl, agent, [0.5, 0.5, 0.5])
        agent.fitness = 0.5
        result = ctrl.apply([agent], best_agent=None)
        assert len(result) == 0

    def test_generation_increments(self) -> None:
        ctrl = ApoptosisController()
        ctrl.apply([_make_agent(0.9)])
        assert ctrl._generation == 1
        ctrl.apply([_make_agent(0.9)])
        assert ctrl._generation == 2


# ---------------------------------------------------------------------------
# reset_history
# ---------------------------------------------------------------------------


class TestResetHistory:
    def test_clears_agent_history(self) -> None:
        ctrl = ApoptosisController()
        agent = _make_agent(0.5)
        ctrl.record(agent)
        assert agent.agent_id in ctrl._histories
        ctrl.reset_history(agent.agent_id)
        assert agent.agent_id not in ctrl._histories

    def test_reset_nonexistent_ok(self) -> None:
        ctrl = ApoptosisController()
        ctrl.reset_history("nonexistent-id")  # should not raise
