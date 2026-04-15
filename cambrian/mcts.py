# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Monte Carlo Tree Search for strategic mutation selection.

Standard tournament selection is memoryless — it picks parents uniformly at
random from the top-k contestants and forgets immediately.  MCTS replaces
this with a *directed* search that remembers which agent lineages have
produced good mutations and which have stagnated.

Each :class:`MCTSNode` tracks how often a genome's descendants were selected
and what average reward those descendants earned.  The UCB1 formula then
balances *exploitation* (favoring high-reward lineages) against *exploration*
(revisiting under-sampled branches that may be hiding gains).

Typical usage in an evolution loop::

    selector = MCTSSelector(mutator=my_mutator)

    # Seed the tree with initial population
    for agent in population:
        selector.register(agent)

    # Every generation: select → expand → evaluate → backpropagate
    parent = selector.select(population)
    children = selector.expand(parent, task="...")
    for child in children:
        score = evaluator(child, task)
        child.fitness = score
        selector.backpropagate(child.agent_id, score)

    # Plug into EvolutionEngine via the `selection_fn` hook (see docs)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cambrian.agent import Agent
    from cambrian.mutator import LLMMutator


@dataclass
class MCTSNode:
    """A node in the MCTS mutation tree.

    Each node corresponds to one agent genome state.  Edges represent
    mutations (i.e. a child was produced by mutating this node's agent).

    Attributes:
        agent: The agent whose genome is represented by this node.
        parent: Parent node (``None`` for root nodes).
        children: Child nodes produced by expanding this node.
        visits: Number of times this node (or its descendants) were sampled.
        total_reward: Cumulative fitness of all visits via this node.
        depth: Distance from the nearest root.
    """

    agent: "Agent"
    parent: "MCTSNode | None" = field(default=None, repr=False)
    children: list["MCTSNode"] = field(default_factory=list, repr=False)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0

    def ucb1(self, exploration_constant: float = math.sqrt(2)) -> float:
        """UCB1 selection score.

        Unvisited nodes return ``+inf`` to force exploration.  Root nodes
        (no parent) fall back to pure exploitation.

        Args:
            exploration_constant: Trade-off coefficient ``C``.
                Higher → more exploration.  Default ``√2`` (theoretical optimum
                for reward in ``[0, 1]``).

        Returns:
            UCB1 score as a ``float``.
        """
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        if self.parent is None or self.parent.visits == 0:
            return exploitation
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    @property
    def mean_reward(self) -> float:
        """Average reward across all visits (0.0 when unvisited)."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self, max_children: int) -> bool:
        """``True`` if this node already has *max_children* children."""
        return len(self.children) >= max_children

    def __repr__(self) -> str:
        return (
            f"MCTSNode(agent={self.agent.id[:8]!r}, visits={self.visits}, "
            f"mean={self.mean_reward:.4f}, depth={self.depth})"
        )


class MCTSSelector:
    """UCB1-based agent selector that builds a persistent mutation tree.

    Replaces memoryless tournament selection with a directed exploration
    strategy: lineages that have produced above-average offspring are
    re-visited; stagnant lineages are still explored but less often.

    Args:
        mutator: :class:`~cambrian.mutator.LLMMutator` used to produce
            child agents during expansion.
        exploration_constant: UCB1 ``C`` parameter.  ``√2`` is the
            theoretical optimum for normalised rewards.  Default ``√2``.
        max_depth: Maximum tree depth before a node is marked terminal
            (no further expansion).  Default ``6``.
        max_children: Maximum children per node (branching factor).
            Default ``3``.
    """

    def __init__(
        self,
        mutator: "LLMMutator",
        exploration_constant: float = math.sqrt(2),
        max_depth: int = 6,
        max_children: int = 3,
    ) -> None:
        self._mutator = mutator
        self._c = exploration_constant
        self._max_depth = max_depth
        self._max_children = max_children
        # agent_id → node (all nodes ever registered)
        self._node_map: dict[str, MCTSNode] = {}
        # root nodes (agents without parents in the MCTS sense)
        self._roots: dict[str, MCTSNode] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, agent: "Agent") -> MCTSNode:
        """Register *agent* as a root node if not already in the tree.

        Safe to call multiple times — re-registration is a no-op.

        Args:
            agent: Agent to register.

        Returns:
            The :class:`MCTSNode` for this agent.
        """
        if agent.agent_id not in self._node_map:
            node = MCTSNode(agent=agent)
            self._roots[agent.agent_id] = node
            self._node_map[agent.agent_id] = node
        return self._node_map[agent.agent_id]

    def select(self, population: list["Agent"]) -> "Agent":
        """Select an agent from *population* using UCB1 scores.

        Ensures all agents in *population* are registered, then returns
        the one with the highest UCB1 score.  Ties are broken randomly.

        Args:
            population: Pool of candidate agents.

        Returns:
            The selected agent.

        Raises:
            ValueError: If *population* is empty.
        """
        if not population:
            raise ValueError("Cannot select from an empty population.")

        for agent in population:
            self.register(agent)

        # Score all candidates and pick the highest UCB1
        scored = [
            (self._node_map[a.agent_id].ucb1(self._c), a)
            for a in population
            if a.agent_id in self._node_map
        ]
        if not scored:
            return random.choice(population)

        best_score = max(s for s, _ in scored)
        # Collect ties to break randomly
        candidates = [a for s, a in scored if s == best_score]
        return random.choice(candidates)

    def expand(self, agent: "Agent", task: str, n_children: int = 1) -> list["Agent"]:
        """Produce *n_children* mutations of *agent* and register them.

        If the node is already fully expanded (has ``max_children``
        children), returns the existing children instead.  Terminal nodes
        (at ``max_depth``) return an empty list.

        Args:
            agent: Parent agent to mutate.
            task: Task context for the LLM mutator.
            n_children: Number of new children to create (default ``1``).

        Returns:
            List of new (unevaluated) child agents.
        """
        parent_node = self._node_map.get(agent.agent_id) or self.register(agent)

        if parent_node.depth >= self._max_depth:
            return []  # terminal node

        if parent_node.is_fully_expanded(self._max_children):
            return [c.agent for c in parent_node.children]

        children: list["Agent"] = []
        slots_available = self._max_children - len(parent_node.children)
        for _ in range(min(n_children, slots_available)):
            child_agent = self._mutator.mutate(agent, task)
            child_node = MCTSNode(
                agent=child_agent,
                parent=parent_node,
                depth=parent_node.depth + 1,
            )
            parent_node.children.append(child_node)
            self._node_map[child_agent.agent_id] = child_node
            children.append(child_agent)

        return children

    def backpropagate(self, agent_id: str, reward: float) -> None:
        """Update visit counts and cumulative rewards from *agent_id* to root.

        Args:
            agent_id: ID of the evaluated agent (leaf from the perspective
                of this update pass).
            reward: Fitness score in ``[0.0, 1.0]`` to propagate.
        """
        node = self._node_map.get(agent_id)
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_path(self, root_id: str) -> list[MCTSNode]:
        """Return the greedy best sequence from *root_id* to its best leaf.

        At each step, follows the child with the highest mean reward.

        Args:
            root_id: Agent ID of the starting root node.

        Returns:
            Ordered list of :class:`MCTSNode` objects from root to leaf.
        """
        node = self._roots.get(root_id)
        if node is None:
            return []

        path: list[MCTSNode] = [node]
        while node.children:
            node = max(node.children, key=lambda n: n.mean_reward)
            path.append(node)
        return path

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics about the MCTS tree.

        Returns:
            Dict with keys ``nodes``, ``visits``, ``roots``, ``max_depth``,
            ``mean_visits``, ``best_reward``.
        """
        if not self._node_map:
            return {"nodes": 0, "visits": 0, "roots": 0, "max_depth": 0,
                    "mean_visits": 0.0, "best_reward": 0.0}

        nodes_list = list(self._node_map.values())
        total_visits = sum(n.visits for n in nodes_list)
        max_depth = max(n.depth for n in nodes_list)
        best_reward = max(n.mean_reward for n in nodes_list if n.visits > 0)

        return {
            "nodes": len(self._node_map),
            "visits": total_visits,
            "roots": len(self._roots),
            "max_depth": max_depth,
            "mean_visits": total_visits / len(nodes_list),
            "best_reward": best_reward,
        }

    def prune_stale_roots(
        self, active_ids: set[str], keep_best: int = 5
    ) -> None:
        """Remove root nodes whose agents are no longer in the population.

        Keeps the *keep_best* highest-reward roots regardless of activity
        to preserve promising lineages.

        Args:
            active_ids: Set of agent IDs currently in the population.
            keep_best: Minimum number of roots to keep regardless of activity.
        """
        stale = [
            rid for rid in list(self._roots)
            if rid not in active_ids
        ]
        # Sort stale roots by mean reward; keep the best ones
        stale.sort(
            key=lambda rid: self._roots[rid].mean_reward,
            reverse=True,
        )
        to_remove = stale[keep_best:]
        for rid in to_remove:
            node = self._roots.pop(rid, None)
            if node is not None:
                self._remove_subtree(node)

    def _remove_subtree(self, node: MCTSNode) -> None:
        """Recursively remove *node* and all its descendants from the maps."""
        for child in node.children:
            self._remove_subtree(child)
        self._node_map.pop(node.agent.agent_id, None)
        self._roots.pop(node.agent.agent_id, None)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MCTSSelector(nodes={s['nodes']}, visits={s['visits']}, "
            f"max_depth={s['max_depth']}, c={self._c:.3f})"
        )
