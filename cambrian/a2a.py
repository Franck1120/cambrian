# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Agent-to-Agent (A2A) protocol for Cambrian.

In a standard evolutionary run, agents operate independently: each is
evaluated in isolation on the same task.  The A2A module introduces
*inter-agent communication and delegation*, so that a capable agent can
decompose a complex task and delegate sub-tasks to specialist agents.

Architecture
------------

``AgentCard``
    A lightweight capability descriptor published by each agent.
    Declares what kinds of tasks the agent handles best (domains, skills,
    confidence threshold).

``A2AMessage``
    A structured request/response envelope passed between agents.
    Fields: sender, recipient, task, result, metadata.

``AgentNetwork``
    Manages a pool of agents and their capability cards.  The primary
    entry point:

    - :meth:`register` — add an agent to the network.
    - :meth:`route` — find the best agent for a given task description.
    - :meth:`delegate` — send a task to the best available agent and
      return its response.
    - :meth:`broadcast` — send a task to *all* agents and collect
      responses (useful for ensemble/voting).
    - :meth:`chain` — pipeline: output of agent A becomes input of agent B.

Delegation fitness boost
------------------------
When :meth:`delegate` is called, the *requesting* agent's fitness can be
augmented by a configurable ``delegation_bonus`` to reward agents that
effectively leverage the network.  This creates selection pressure for
agents that learn to decompose and delegate.

Usage::

    from cambrian.a2a import AgentNetwork, AgentCard

    network = AgentNetwork()

    # Register agents with capability descriptors
    network.register(math_agent,  AgentCard(domains=["math", "logic"], confidence=0.9))
    network.register(code_agent,  AgentCard(domains=["code", "python"], confidence=0.85))
    network.register(prose_agent, AgentCard(domains=["writing", "prose"], confidence=0.8))

    # Delegate: find best agent and run
    result = network.delegate("Implement a binary search in Python.", sender_id="root")

    # Chain: output of step 1 is input of step 2
    result = network.chain(
        task="Write a Python function for quicksort",
        agent_ids=[code_agent.id, review_agent.id],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cambrian.agent import Agent
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── AgentCard ─────────────────────────────────────────────────────────────────


@dataclass
class AgentCard:
    """Capability descriptor for an agent in the network.

    Attributes:
        domains: List of domain keywords this agent handles (e.g. ``["code",
            "python", "sorting"]``).  Used for keyword-based routing.
        confidence: Self-reported competence in [0.0, 1.0].  Higher = more
            likely to be selected when multiple agents match.
        max_tokens: Preferred response length limit (tokens).  ``None``
            means no preference.
        description: Free-text description of what this agent does.
    """

    domains: list[str] = field(default_factory=list)
    confidence: float = 0.5
    max_tokens: int | None = None
    description: str = ""

    def matches(self, task: str, threshold: float = 0.0) -> bool:
        """``True`` if any domain keyword appears in *task* (case-insensitive).

        Args:
            task: Task description.
            threshold: Minimum confidence required to match.  Default ``0.0``.
        """
        if self.confidence < threshold:
            return False
        task_lower = task.lower()
        return any(d.lower() in task_lower for d in self.domains)

    def relevance_score(self, task: str) -> float:
        """Heuristic relevance score for *task* in [0.0, 1.0].

        Counts matching domain keywords weighted by confidence.
        """
        task_lower = task.lower()
        matches = sum(1 for d in self.domains if d.lower() in task_lower)
        if not self.domains:
            return 0.0
        return self.confidence * (matches / len(self.domains))


# ── A2AMessage ────────────────────────────────────────────────────────────────


@dataclass
class A2AMessage:
    """Structured request/response envelope for inter-agent communication.

    Attributes:
        sender_id: ID of the requesting agent (or ``"root"`` for top-level).
        recipient_id: ID of the agent that will handle the task.
        task: The task description or question.
        result: The agent's response (filled after execution).
        latency_ms: Wall-clock time for the agent response in milliseconds.
        metadata: Arbitrary extra data (e.g. fitness scores, hop count).
    """

    sender_id: str
    recipient_id: str
    task: str
    result: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── AgentNetwork ──────────────────────────────────────────────────────────────


class AgentNetwork:
    """Network of agents that can communicate, delegate, and chain tasks.

    Args:
        delegation_bonus: Fitness multiplier applied to the best delegated
            result.  Set to ``0.0`` to disable.  Default ``0.05``.
        max_hops: Maximum delegation chain length to prevent infinite loops.
            Default ``5``.
        timeout_per_agent: Seconds allowed per agent call (soft limit via
            timing; agents are not forcibly interrupted).  Default ``30``.
    """

    def __init__(
        self,
        delegation_bonus: float = 0.05,
        max_hops: int = 5,
        timeout_per_agent: float = 30.0,
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self._cards: dict[str, AgentCard] = {}
        self._log: list[A2AMessage] = []
        self._bonus = delegation_bonus
        self._max_hops = max_hops
        self._timeout = timeout_per_agent

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent: Agent, card: AgentCard | None = None) -> None:
        """Add *agent* to the network with an optional capability card.

        Args:
            agent: The agent to register.
            card: Capability descriptor.  If ``None``, an empty card is used
                (the agent will match all tasks but with low confidence).
        """
        self._agents[agent.id] = agent
        self._cards[agent.id] = card or AgentCard()
        logger.debug("A2ANetwork: registered agent %s (domains=%s)",
                     agent.id[:8], (card.domains if card else []))

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the network.

        Args:
            agent_id: The agent ID to remove.
        """
        self._agents.pop(agent_id, None)
        self._cards.pop(agent_id, None)

    def register_population(
        self,
        agents: list[Agent],
        card_factory: Any = None,
    ) -> None:
        """Register all agents in *agents* at once.

        Args:
            agents: Population to register.
            card_factory: Optional callable ``(agent) → AgentCard``.  If
                ``None``, cards are auto-generated from the genome's strategy
                and system prompt keywords.
        """
        for agent in agents:
            card = card_factory(agent) if card_factory else self._auto_card(agent)
            self.register(agent, card)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(
        self,
        task: str,
        exclude: list[str] | None = None,
        require_fitness: float = 0.0,
    ) -> Agent | None:
        """Find the best agent for *task*.

        Selection criteria (in order):
        1. Agent's card relevance score (domain match × confidence)
        2. Agent's fitness (higher is better)
        3. Falls back to highest-fitness agent if no domain matches.

        Args:
            task: Task description.
            exclude: Agent IDs to exclude from routing (e.g. the sender).
            require_fitness: Minimum fitness threshold.

        Returns:
            Best matching :class:`~cambrian.agent.Agent`, or ``None`` if
            network is empty.
        """
        excluded = set(exclude or [])
        candidates = [
            (aid, agent)
            for aid, agent in self._agents.items()
            if aid not in excluded and (agent.fitness or 0.0) >= require_fitness
        ]
        if not candidates:
            return None

        def _score(aid_agent: tuple[str, Agent]) -> float:
            aid, agent = aid_agent
            card = self._cards.get(aid, AgentCard())
            rel = card.relevance_score(task)
            fit = agent.fitness or 0.0
            return rel * 0.7 + fit * 0.3

        best_id, best_agent = max(candidates, key=_score)
        return best_agent

    # ── Delegation ────────────────────────────────────────────────────────────

    def delegate(
        self,
        task: str,
        sender_id: str = "root",
        exclude: list[str] | None = None,
        require_fitness: float = 0.0,
    ) -> A2AMessage:
        """Delegate *task* to the best available agent.

        Args:
            task: Task description.
            sender_id: ID of the requesting agent or ``"root"``.
            exclude: Agent IDs to skip (prevents sender self-routing).
            require_fitness: Minimum fitness for the recipient.

        Returns:
            :class:`A2AMessage` with ``result`` filled.
        """
        agent = self.route(task, exclude=exclude or [sender_id], require_fitness=require_fitness)
        if agent is None:
            msg = A2AMessage(
                sender_id=sender_id,
                recipient_id="none",
                task=task,
                result="",
                metadata={"error": "no agent available"},
            )
            self._log.append(msg)
            return msg

        t0 = time.monotonic()
        try:
            result = agent.run(task)
        except Exception as exc:
            result = f"[error: {exc}]"
        latency = (time.monotonic() - t0) * 1000

        msg = A2AMessage(
            sender_id=sender_id,
            recipient_id=agent.id,
            task=task,
            result=result,
            latency_ms=round(latency, 2),
            metadata={"fitness": agent.fitness, "model": agent.genome.model},
        )
        self._log.append(msg)
        logger.debug(
            "A2A delegate: %s→%s latency=%.0fms",
            sender_id[:8], agent.id[:8], latency,
        )
        return msg

    def broadcast(
        self,
        task: str,
        sender_id: str = "root",
        top_n: int | None = None,
        require_fitness: float = 0.0,
    ) -> list[A2AMessage]:
        """Send *task* to all (or top-N) agents and collect responses.

        Useful for ensemble evaluation or majority voting.

        Args:
            task: Task description.
            sender_id: Requesting agent / ``"root"``.
            top_n: If set, only the top-N agents by fitness are queried.
            require_fitness: Minimum fitness threshold.

        Returns:
            List of :class:`A2AMessage` objects, one per responding agent.
        """
        agents = [
            a for aid, a in self._agents.items()
            if aid != sender_id and (a.fitness or 0.0) >= require_fitness
        ]
        agents.sort(key=lambda a: a.fitness or 0.0, reverse=True)
        if top_n is not None:
            agents = agents[:top_n]

        messages: list[A2AMessage] = []
        for agent in agents:
            t0 = time.monotonic()
            try:
                result = agent.run(task)
            except Exception as exc:
                result = f"[error: {exc}]"
            latency = (time.monotonic() - t0) * 1000
            msg = A2AMessage(
                sender_id=sender_id,
                recipient_id=agent.id,
                task=task,
                result=result,
                latency_ms=round(latency, 2),
                metadata={"fitness": agent.fitness},
            )
            messages.append(msg)
            self._log.append(msg)

        return messages

    def chain(
        self,
        task: str,
        agent_ids: list[str],
        sender_id: str = "root",
    ) -> A2AMessage:
        """Pipeline: output of step i becomes input of step i+1.

        Args:
            task: Initial task.
            agent_ids: Ordered list of agent IDs to call in sequence.
            sender_id: Root sender ID.

        Returns:
            Final :class:`A2AMessage` with the last agent's result.

        Raises:
            ValueError: If *agent_ids* is empty or exceeds *max_hops*.
        """
        if not agent_ids:
            raise ValueError("chain() requires at least one agent_id.")
        if len(agent_ids) > self._max_hops:
            raise ValueError(
                f"Chain length {len(agent_ids)} exceeds max_hops={self._max_hops}."
            )

        current_input = task
        last_msg = A2AMessage(sender_id=sender_id, recipient_id="", task=task)

        for i, aid in enumerate(agent_ids):
            agent = self._agents.get(aid)
            if agent is None:
                logger.warning("chain: agent %s not in network, skipping", aid[:8])
                continue

            t0 = time.monotonic()
            try:
                result = agent.run(current_input)
            except Exception as exc:
                result = f"[error: {exc}]"
            latency = (time.monotonic() - t0) * 1000

            last_msg = A2AMessage(
                sender_id=sender_id if i == 0 else agent_ids[i - 1],
                recipient_id=aid,
                task=current_input,
                result=result,
                latency_ms=round(latency, 2),
                metadata={"hop": i, "fitness": agent.fitness},
            )
            self._log.append(last_msg)
            current_input = result  # pipeline: output → next input

        return last_msg

    # ── Consensus / voting ────────────────────────────────────────────────────

    def majority_vote(
        self,
        task: str,
        sender_id: str = "root",
        top_n: int = 3,
    ) -> str:
        """Broadcast task and return the most common response.

        Args:
            task: Task description.
            sender_id: Root sender.
            top_n: Number of agents to query.

        Returns:
            Most common response string (longest match wins ties).
        """
        messages = self.broadcast(task, sender_id=sender_id, top_n=top_n)
        if not messages:
            return ""
        # Count responses
        counts: dict[str, int] = {}
        for m in messages:
            r = m.result.strip()
            counts[r] = counts.get(r, 0) + 1
        return max(counts, key=lambda r: (counts[r], len(r)))

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def network_size(self) -> int:
        """Number of agents in the network."""
        return len(self._agents)

    def message_log(self) -> list[A2AMessage]:
        """Return all messages sent through the network (read-only copy)."""
        return list(self._log)

    def agent_ids(self) -> list[str]:
        """Return all registered agent IDs."""
        return list(self._agents.keys())

    def summary(self) -> dict[str, Any]:
        """High-level summary of network activity."""
        total = len(self._log)
        if total == 0:
            return {"network_size": self.network_size, "messages": 0}
        latencies = [m.latency_ms for m in self._log if m.latency_ms > 0]
        return {
            "network_size": self.network_size,
            "messages": total,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "unique_recipients": len({m.recipient_id for m in self._log}),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_card(agent: Agent) -> AgentCard:
        """Generate a basic AgentCard from the agent's genome keywords."""
        text = (agent.genome.system_prompt + " " + agent.genome.strategy).lower()
        domain_keywords = [
            "code", "python", "math", "logic", "writing", "analysis",
            "reasoning", "creative", "search", "data", "sql", "test",
        ]
        domains = [kw for kw in domain_keywords if kw in text]
        confidence = min(0.9, 0.3 + (agent.fitness or 0.0) * 0.6)
        return AgentCard(domains=domains or ["general"], confidence=confidence)

    def __repr__(self) -> str:
        return (
            f"AgentNetwork(agents={self.network_size}, "
            f"messages={len(self._log)})"
        )
