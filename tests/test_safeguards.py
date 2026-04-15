"""Tests for cambrian/safeguards.py — GoalDriftDetector, FitnessAnomalyDetector,
SafeguardController.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.safeguards import (
    DriftEvent,
    FitnessAnomalyDetector,
    GoalDriftDetector,
    SafeguardController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(fitness: float = 0.5, prompt: str = "expert analytical step-by-step") -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(response: str = "realigned prompt") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


# ---------------------------------------------------------------------------
# GoalDriftDetector
# ---------------------------------------------------------------------------


class TestGoalDriftDetector:
    def test_no_drift_identical_prompt(self) -> None:
        """Drift score near 0 when prompt matches intent exactly."""
        det = GoalDriftDetector(drift_threshold=0.4)
        agent = _agent(prompt="solve math problems step by step")
        det.register(agent, intent="solve math problems step by step")
        ev = det.measure(agent, generation=1)
        assert isinstance(ev, DriftEvent)
        assert ev.drift_score < 0.1

    def test_high_drift_completely_different_prompt(self) -> None:
        """Drift score near 1 when prompt has no word overlap with intent."""
        det = GoalDriftDetector(drift_threshold=0.4)
        agent = _agent(prompt="quantum entanglement photon polarisation")
        det.register(agent, intent="solve arithmetic math addition subtraction")
        ev = det.measure(agent, generation=1)
        assert ev.drift_score > 0.5

    def test_flagged_above_threshold(self) -> None:
        det = GoalDriftDetector(drift_threshold=0.3)
        agent = _agent(prompt="completely unrelated different domain words here")
        det.register(agent, intent="python code programming functions")
        ev = det.measure(agent, generation=1)
        assert ev.flagged is True

    def test_not_flagged_below_threshold(self) -> None:
        det = GoalDriftDetector(drift_threshold=0.9)
        agent = _agent(prompt="solve math step by step")
        det.register(agent, intent="solve math step by step carefully")
        ev = det.measure(agent, generation=1)
        assert ev.flagged is False

    def test_scan_population_returns_only_flagged(self) -> None:
        det = GoalDriftDetector(drift_threshold=0.4)
        stable = _agent(prompt="expert python code analysis")
        drifted = _agent(prompt="cooking recipes baking bread butter")
        det.register(stable, intent="expert python code analysis debugging")
        det.register(drifted, intent="expert python code analysis debugging")

        flagged = det.scan_population([stable, drifted], generation=2)
        flagged_ids = {e.agent_id for e in flagged}
        assert drifted.agent_id in flagged_ids

    def test_scan_skips_unregistered_agents(self) -> None:
        det = GoalDriftDetector()
        unregistered = _agent()
        # Should not raise
        result = det.scan_population([unregistered], generation=1)
        assert isinstance(result, list)

    def test_events_accumulate(self) -> None:
        det = GoalDriftDetector()
        agent = _agent(prompt="solving problems")
        det.register(agent, intent="solving problems carefully")
        det.measure(agent, generation=1)
        det.measure(agent, generation=2)
        assert len(det.events) == 2

    def test_drift_score_clamped_zero_to_one(self) -> None:
        det = GoalDriftDetector()
        agent = _agent(prompt="")
        det.register(agent, intent="some intent words")
        ev = det.measure(agent, generation=1)
        assert 0.0 <= ev.drift_score <= 1.0

    def test_drift_event_stores_agent_id(self) -> None:
        det = GoalDriftDetector()
        agent = _agent()
        det.register(agent, intent="intent text here")
        ev = det.measure(agent, generation=5)
        assert ev.agent_id == agent.agent_id
        assert ev.generation == 5

    def test_original_intent_stored_in_event(self) -> None:
        det = GoalDriftDetector()
        agent = _agent()
        det.register(agent, intent="my original intent")
        ev = det.measure(agent, generation=1)
        assert "my original intent" in ev.original_intent

    def test_events_property_returns_copy(self) -> None:
        det = GoalDriftDetector()
        agent = _agent()
        det.register(agent, intent="test intent")
        det.measure(agent, generation=1)
        evs = det.events
        evs.clear()
        assert len(det.events) == 1  # original unaffected


# ---------------------------------------------------------------------------
# FitnessAnomalyDetector
# ---------------------------------------------------------------------------


class TestFitnessAnomalyDetector:
    def test_not_anomalous_insufficient_history(self) -> None:
        det = FitnessAnomalyDetector(min_history=5)
        agent = _agent(fitness=1.0)
        for gen in range(3):
            det.record(agent, generation=gen)
        assert det.is_anomalous(agent) is False

    def test_not_anomalous_stable_fitness(self) -> None:
        det = FitnessAnomalyDetector(z_threshold=2.5, min_history=5)
        agent = _agent(fitness=0.5)
        for gen in range(6):
            agent.fitness = 0.5 + gen * 0.01  # gradual, stable increase
            det.record(agent, generation=gen)
        assert det.is_anomalous(agent) is False

    def test_anomalous_spike(self) -> None:
        det = FitnessAnomalyDetector(z_threshold=1.5, min_history=5)
        agent = _agent(fitness=0.3)
        # Establish stable baseline
        for gen in range(5):
            agent.fitness = 0.3
            det.record(agent, generation=gen)
        # Spike
        agent.fitness = 0.99
        assert det.is_anomalous(agent) is True

    def test_scan_returns_anomalous_agent_ids(self) -> None:
        det = FitnessAnomalyDetector(z_threshold=1.5, min_history=5)
        stable = _agent(fitness=0.3)
        spiky = _agent(fitness=0.3)

        for gen in range(5):
            stable.fitness = 0.3
            det.record(stable, generation=gen)
            spiky.fitness = 0.3
            det.record(spiky, generation=gen)

        spiky.fitness = 0.99
        anomalies = det.scan([stable, spiky], generation=5)
        assert spiky.agent_id in anomalies
        assert stable.agent_id not in anomalies

    def test_none_fitness_treated_as_zero(self) -> None:
        det = FitnessAnomalyDetector(min_history=3)
        # Agent with no fitness set (fitness=None by default)
        g = Genome(system_prompt="test")
        agent = Agent(genome=g)  # fitness is None by default
        # Should not raise when recording an agent with None fitness
        det.record(agent, generation=1)

    def test_scan_empty_population(self) -> None:
        det = FitnessAnomalyDetector()
        assert det.scan([], generation=0) == []

    def test_uniform_history_not_anomalous(self) -> None:
        """Zero std — uniform fitness history — should not flag as anomalous."""
        det = FitnessAnomalyDetector(z_threshold=2.5, min_history=5)
        agent = _agent(fitness=0.5)
        for gen in range(6):
            agent.fitness = 0.5
            det.record(agent, generation=gen)
        assert det.is_anomalous(agent) is False


# ---------------------------------------------------------------------------
# SafeguardController
# ---------------------------------------------------------------------------


class TestSafeguardController:
    def test_check_returns_structure(self) -> None:
        drift_det = GoalDriftDetector(drift_threshold=0.3)
        anomaly_det = FitnessAnomalyDetector()
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
        )
        agent = _agent()
        drift_det.register(agent, intent="expert coder")
        result = ctrl.check([agent], generation=1)
        assert "drift" in result
        assert "anomalies" in result
        assert isinstance(result["drift"], list)
        assert isinstance(result["anomalies"], list)

    def test_check_detects_drift(self) -> None:
        drift_det = GoalDriftDetector(drift_threshold=0.3)
        anomaly_det = FitnessAnomalyDetector()
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
        )
        agent = _agent(prompt="cooking baking butter bread oven kitchen")
        drift_det.register(agent, intent="python programming code algorithms")
        result = ctrl.check([agent], generation=1)
        assert len(result["drift"]) == 1

    def test_remediate_with_backend(self) -> None:
        drift_det = GoalDriftDetector()
        anomaly_det = FitnessAnomalyDetector()
        backend = _mock_backend("realigned: expert python coder")
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
            backend=backend,
        )
        agent = _agent(prompt="drifted prompt completely different")
        drift_det.register(agent, intent="expert python coding debugging")
        result = ctrl.remediate(agent, task="write code")
        assert isinstance(result, Agent)

    def test_remediate_without_backend_returns_clone(self) -> None:
        drift_det = GoalDriftDetector()
        anomaly_det = FitnessAnomalyDetector()
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
            backend=None,
        )
        agent = _agent(prompt="original prompt")
        result = ctrl.remediate(agent, task="task")
        assert isinstance(result, Agent)

    def test_remediate_backend_error_returns_clone(self) -> None:
        drift_det = GoalDriftDetector()
        anomaly_det = FitnessAnomalyDetector()
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("backend error")
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
            backend=backend,
        )
        drift_det.register(_agent(), intent="test intent")
        result = ctrl.remediate(_agent(prompt="drifted"), task="task")
        assert isinstance(result, Agent)

    def test_check_empty_population(self) -> None:
        drift_det = GoalDriftDetector()
        anomaly_det = FitnessAnomalyDetector()
        ctrl = SafeguardController(
            drift_detector=drift_det,
            anomaly_detector=anomaly_det,
        )
        result = ctrl.check([], generation=1)
        assert result["drift"] == []
        assert result["anomalies"] == []
