# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.a2a — AgentCard, A2AMessage, AgentNetwork."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.a2a import A2AMessage, AgentCard, AgentNetwork
from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(response: str = "mock response") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _agent(prompt: str, fitness: float = 0.5, response: str = "mock response") -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt), backend=_backend(response))
    agent.fitness = fitness
    return agent


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_matches_domain_keyword(self) -> None:
        card = AgentCard(domains=["math", "logic"], confidence=0.9)
        assert card.matches("Solve a math problem") is True

    def test_no_match_on_unknown_domain(self) -> None:
        card = AgentCard(domains=["math", "logic"], confidence=0.9)
        assert card.matches("Write a poem") is False

    def test_matches_case_insensitive(self) -> None:
        card = AgentCard(domains=["Python"], confidence=0.8)
        assert card.matches("write a python function") is True

    def test_matches_below_threshold_returns_false(self) -> None:
        card = AgentCard(domains=["code"], confidence=0.3)
        assert card.matches("write code", threshold=0.5) is False

    def test_relevance_score_proportional(self) -> None:
        card = AgentCard(domains=["math", "logic", "algebra"], confidence=1.0)
        score = card.relevance_score("solve a math logic problem")
        assert 0.0 < score <= 1.0

    def test_relevance_score_zero_on_no_match(self) -> None:
        card = AgentCard(domains=["math"], confidence=0.9)
        score = card.relevance_score("write a poem about nature")
        assert score == 0.0

    def test_relevance_score_zero_on_empty_domains(self) -> None:
        card = AgentCard(domains=[], confidence=0.9)
        score = card.relevance_score("anything")
        assert score == 0.0

    def test_default_confidence(self) -> None:
        card = AgentCard()
        assert 0.0 <= card.confidence <= 1.0


# ---------------------------------------------------------------------------
# A2AMessage
# ---------------------------------------------------------------------------


class TestA2AMessage:
    def test_fields_accessible(self) -> None:
        msg = A2AMessage(
            sender_id="agent_a",
            recipient_id="agent_b",
            task="solve this",
            result="done",
            latency_ms=12.5,
        )
        assert msg.sender_id == "agent_a"
        assert msg.recipient_id == "agent_b"
        assert msg.task == "solve this"
        assert msg.result == "done"
        assert msg.latency_ms == 12.5

    def test_metadata_default_empty(self) -> None:
        msg = A2AMessage(sender_id="x", recipient_id="y", task="t")
        assert isinstance(msg.metadata, dict)

    def test_result_default_empty_string(self) -> None:
        msg = A2AMessage(sender_id="x", recipient_id="y", task="t")
        assert msg.result == ""


# ---------------------------------------------------------------------------
# AgentNetwork — registration
# ---------------------------------------------------------------------------


class TestAgentNetworkRegistration:
    def test_register_single_agent(self) -> None:
        net = AgentNetwork()
        agent = _agent("code agent", fitness=0.7)
        net.register(agent, AgentCard(domains=["code"]))
        assert net.network_size == 1

    def test_register_multiple_agents(self) -> None:
        net = AgentNetwork()
        for i in range(5):
            net.register(_agent(f"agent {i}"), AgentCard(domains=[f"domain{i}"]))
        assert net.network_size == 5

    def test_unregister_removes_agent(self) -> None:
        net = AgentNetwork()
        agent = _agent("agent")
        net.register(agent)
        net.unregister(agent.id)
        assert net.network_size == 0

    def test_unregister_unknown_id_is_noop(self) -> None:
        net = AgentNetwork()
        net.unregister("nonexistent-id")  # must not raise

    def test_register_without_card(self) -> None:
        net = AgentNetwork()
        agent = _agent("general agent")
        net.register(agent)  # no card — auto-generated
        assert net.network_size == 1

    def test_register_population(self) -> None:
        net = AgentNetwork()
        pop = [_agent(f"pop agent {i}") for i in range(4)]
        net.register_population(pop)
        assert net.network_size == 4

    def test_register_population_with_card_factory(self) -> None:
        net = AgentNetwork()
        pop = [_agent(f"code agent {i}") for i in range(3)]
        net.register_population(pop, card_factory=lambda a: AgentCard(domains=["code"]))
        assert net.network_size == 3

    def test_agent_ids_returns_all(self) -> None:
        net = AgentNetwork()
        agents = [_agent(f"agent {i}") for i in range(3)]
        for a in agents:
            net.register(a)
        ids = net.agent_ids()
        assert len(ids) == 3
        for a in agents:
            assert a.id in ids


# ---------------------------------------------------------------------------
# AgentNetwork — routing
# ---------------------------------------------------------------------------


class TestAgentNetworkRouting:
    def test_route_returns_agent(self) -> None:
        net = AgentNetwork()
        agent = _agent("python coder", fitness=0.8)
        net.register(agent, AgentCard(domains=["python", "code"]))
        result = net.route("write a python function")
        assert result is not None

    def test_route_empty_network_returns_none(self) -> None:
        net = AgentNetwork()
        assert net.route("any task") is None

    def test_route_prefers_matching_domain(self) -> None:
        net = AgentNetwork()
        math_agent = _agent("math expert", fitness=0.6)
        prose_agent = _agent("prose writer", fitness=0.9)
        net.register(math_agent, AgentCard(domains=["math", "algebra"], confidence=0.9))
        net.register(prose_agent, AgentCard(domains=["writing", "prose"], confidence=0.5))
        result = net.route("solve a math algebra problem")
        assert result is math_agent

    def test_route_excludes_listed_ids(self) -> None:
        net = AgentNetwork()
        a = _agent("agent a", fitness=0.9)
        b = _agent("agent b", fitness=0.5)
        net.register(a)
        net.register(b)
        result = net.route("any task", exclude=[a.id])
        assert result is b

    def test_route_respects_require_fitness(self) -> None:
        net = AgentNetwork()
        low = _agent("low fitness agent", fitness=0.1)
        high = _agent("high fitness agent", fitness=0.9)
        net.register(low)
        net.register(high)
        result = net.route("task", require_fitness=0.5)
        assert result is high


# ---------------------------------------------------------------------------
# AgentNetwork — delegation
# ---------------------------------------------------------------------------


class TestAgentNetworkDelegation:
    def test_delegate_returns_a2a_message(self) -> None:
        net = AgentNetwork()
        agent = _agent("helper", fitness=0.8)
        net.register(agent)
        msg = net.delegate("do something")
        assert isinstance(msg, A2AMessage)

    def test_delegate_result_filled(self) -> None:
        net = AgentNetwork()
        net.register(_agent("helper", fitness=0.8, response="done it"))
        msg = net.delegate("task")
        assert msg.result == "done it"

    def test_delegate_empty_network(self) -> None:
        net = AgentNetwork()
        msg = net.delegate("task")
        assert msg.result == ""
        assert "error" in msg.metadata

    def test_delegate_logs_message(self) -> None:
        net = AgentNetwork()
        net.register(_agent("helper", fitness=0.7))
        net.delegate("task A")
        net.delegate("task B")
        assert len(net.message_log()) == 2

    def test_delegate_latency_positive(self) -> None:
        net = AgentNetwork()
        net.register(_agent("helper", fitness=0.8))
        msg = net.delegate("task")
        assert msg.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# AgentNetwork — broadcast
# ---------------------------------------------------------------------------


class TestAgentNetworkBroadcast:
    def test_broadcast_reaches_all_agents(self) -> None:
        net = AgentNetwork()
        for i in range(4):
            net.register(_agent(f"agent {i}", fitness=0.5 + i * 0.1))
        messages = net.broadcast("shared task")
        assert len(messages) == 4

    def test_broadcast_top_n(self) -> None:
        net = AgentNetwork()
        for i in range(5):
            net.register(_agent(f"agent {i}", fitness=0.1 * (i + 1)))
        messages = net.broadcast("task", top_n=2)
        assert len(messages) == 2

    def test_broadcast_empty_network(self) -> None:
        net = AgentNetwork()
        messages = net.broadcast("task")
        assert messages == []

    def test_broadcast_excludes_sender(self) -> None:
        net = AgentNetwork()
        sender = _agent("sender")
        receiver = _agent("receiver", fitness=0.8)
        net.register(sender)
        net.register(receiver)
        messages = net.broadcast("task", sender_id=sender.id)
        recipient_ids = {m.recipient_id for m in messages}
        assert sender.id not in recipient_ids


# ---------------------------------------------------------------------------
# AgentNetwork — chain
# ---------------------------------------------------------------------------


class TestAgentNetworkChain:
    def test_chain_single_agent(self) -> None:
        net = AgentNetwork()
        agent = _agent("step one", fitness=0.7, response="step one result")
        net.register(agent)
        msg = net.chain("input task", agent_ids=[agent.id])
        assert isinstance(msg, A2AMessage)

    def test_chain_passes_output_to_next(self) -> None:
        net = AgentNetwork()
        a1 = _agent("first agent", fitness=0.8, response="intermediate result")
        a2 = _agent("second agent", fitness=0.7, response="final result")
        net.register(a1)
        net.register(a2)
        msg = net.chain("initial task", agent_ids=[a1.id, a2.id])
        assert msg.result == "final result"

    def test_chain_empty_ids_raises(self) -> None:
        net = AgentNetwork()
        with pytest.raises(ValueError, match="at least one"):
            net.chain("task", agent_ids=[])

    def test_chain_exceeds_max_hops_raises(self) -> None:
        net = AgentNetwork(max_hops=2)
        agents = [_agent(f"agent {i}") for i in range(3)]
        for a in agents:
            net.register(a)
        with pytest.raises(ValueError, match="max_hops"):
            net.chain("task", agent_ids=[a.id for a in agents])


# ---------------------------------------------------------------------------
# AgentNetwork — majority_vote
# ---------------------------------------------------------------------------


class TestAgentNetworkMajorityVote:
    def test_majority_vote_returns_string(self) -> None:
        net = AgentNetwork()
        for i in range(3):
            net.register(_agent(f"agent {i}", fitness=0.7, response="Paris"))
        result = net.majority_vote("What is the capital of France?", top_n=3)
        assert isinstance(result, str)

    def test_majority_vote_picks_most_common(self) -> None:
        net = AgentNetwork()
        for _ in range(3):
            net.register(_agent("agree agent", fitness=0.7, response="yes"))
        net.register(_agent("disagree agent", fitness=0.5, response="no"))
        result = net.majority_vote("question?", top_n=4)
        assert result == "yes"

    def test_majority_vote_empty_network(self) -> None:
        net = AgentNetwork()
        result = net.majority_vote("question?")
        assert result == ""


# ---------------------------------------------------------------------------
# AgentNetwork — summary
# ---------------------------------------------------------------------------


class TestAgentNetworkSummary:
    def test_summary_has_network_size(self) -> None:
        net = AgentNetwork()
        net.register(_agent("a1"))
        net.register(_agent("a2"))
        s = net.summary()
        assert s["network_size"] == 2

    def test_summary_messages_zero_initially(self) -> None:
        net = AgentNetwork()
        s = net.summary()
        assert s["messages"] == 0

    def test_summary_after_delegation(self) -> None:
        net = AgentNetwork()
        net.register(_agent("helper", fitness=0.8))
        net.delegate("task one")
        net.delegate("task two")
        s = net.summary()
        assert s["messages"] == 2

    def test_repr_is_string(self) -> None:
        net = AgentNetwork()
        assert isinstance(repr(net), str)
