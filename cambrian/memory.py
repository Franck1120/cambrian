# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""EvolutionaryMemory — lineage graph backed by NetworkX.

Tracks the genealogy of every agent across generations, enabling:
- Querying the top-performing ancestors as seeds for future runs.
- Tracing the evolutionary path that led to the best agent.
- Detecting stagnation (when all agents share the same lineage).

Stigmergy support
-----------------
In social insect colonies, individuals leave *pheromone traces* on paths they
have travelled.  Future individuals perceive these traces and bias their
navigation toward proven routes.

:class:`EvolutionaryMemory` implements an analogous mechanism: when an agent
performs well, it calls :meth:`add_trace` to deposit a text excerpt (e.g. a
key sentence from its system prompt) alongside the fitness score that produced
it.  The :class:`~cambrian.mutator.LLMMutator` queries :meth:`get_traces` and
injects the top traces into the mutation prompt so that offspring are nudged
toward high-performing prompt patterns rather than exploring blindly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NX_AVAILABLE = False
    nx = None


# ── Stigmergy ─────────────────────────────────────────────────────────────────


@dataclass
class StigmergyTrace:
    """A pheromone-like trace left by a high-performing agent.

    Attributes:
        agent_id: The agent that deposited this trace.
        content: Text excerpt from the agent's genome (e.g. a key sentence
            from its system prompt that contributed to high fitness).
        score: The fitness score achieved when this trace was deposited.
        task: Optional task string for task-specific trace retrieval.
    """

    agent_id: str
    content: str
    score: float
    task: str = ""


class EvolutionaryMemory:
    """Directed graph of agent lineage.

    Nodes hold agent metadata (fitness, generation, genome snapshot).
    Edges point from parent to child.

    Args:
        name: Human-readable label for the run, used in serialisation.
    """

    def __init__(self, name: str = "default") -> None:
        if not _NX_AVAILABLE:
            raise ImportError(
                "networkx is required for EvolutionaryMemory. "
                "Install it with: pip install networkx"
            )
        self.name = name
        self._graph: "nx.DiGraph" = nx.DiGraph()
        self._agent_count = 0
        self._traces: list[StigmergyTrace] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def add_agent(
        self,
        agent_id: str,
        generation: int,
        fitness: float | None,
        genome_snapshot: dict[str, Any],
        parents: list[str] | None = None,
    ) -> None:
        """Register an agent node and its parent edges.

        Args:
            agent_id: Unique agent identifier.
            generation: The generation this agent was born in.
            fitness: Fitness score at time of registration (may be updated later).
            genome_snapshot: Serialised genome dict for provenance.
            parents: List of parent ``agent_id`` strings. Omit for genesis agents.
        """
        self._graph.add_node(
            agent_id,
            generation=generation,
            fitness=fitness,
            genome=genome_snapshot,
        )
        for parent_id in (parents or []):
            if self._graph.has_node(parent_id):
                self._graph.add_edge(parent_id, agent_id)
        self._agent_count += 1

    def update_fitness(self, agent_id: str, fitness: float) -> None:
        """Update the stored fitness of *agent_id* after evaluation."""
        if self._graph.has_node(agent_id):
            self._graph.nodes[agent_id]["fitness"] = fitness

    def get_top_ancestors(
        self, n: int = 5, min_fitness: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return the *n* agents with the highest fitness meeting *min_fitness*.

        Args:
            n: Maximum number of agents to return.
            min_fitness: Minimum fitness threshold. Agents below this are excluded.

        Returns:
            List of node attribute dicts sorted by descending fitness.
        """
        candidates = [
            {"agent_id": node, **data}
            for node, data in self._graph.nodes(data=True)
            if (data.get("fitness") or 0.0) >= min_fitness
        ]
        candidates.sort(key=lambda x: x.get("fitness") or 0.0, reverse=True)
        return candidates[:n]

    def get_lineage(self, agent_id: str) -> list[str]:
        """Return the ancestor chain from a genesis agent to *agent_id*.

        Performs a topological search from the root(s) to *agent_id*.

        Returns:
            Ordered list of agent IDs from earliest ancestor → *agent_id*.
            Returns ``[agent_id]`` if no ancestors are found.
        """
        if not self._graph.has_node(agent_id):
            return [agent_id]

        # Collect all ancestors via BFS on reversed graph
        reversed_graph = self._graph.reverse()
        ancestors: list[str] = list(nx.bfs_tree(reversed_graph, agent_id).nodes())
        ancestors.reverse()
        if agent_id not in ancestors:
            ancestors.append(agent_id)
        return ancestors

    # ── Stigmergy ─────────────────────────────────────────────────────────────

    def add_trace(
        self,
        agent_id: str,
        content: str,
        score: float,
        task: str = "",
    ) -> None:
        """Deposit a stigmergy trace for *agent_id*.

        Call this after evaluating a high-performing agent to record the
        prompt excerpt (or any text artifact) that contributed to its success.
        Traces are stored in descending score order and can be retrieved by
        :meth:`get_traces` to guide future mutations.

        Args:
            agent_id: The agent that produced this trace.
            content: A short text excerpt (typically 1–3 sentences from the
                agent's system prompt) that encapsulates the effective behaviour.
            score: The fitness score that justifies the deposit.
            task: Optional task string for task-specific retrieval.
        """
        self._traces.append(
            StigmergyTrace(agent_id=agent_id, content=content, score=score, task=task)
        )
        # Keep sorted by score descending so get_traces is O(1)
        self._traces.sort(key=lambda t: t.score, reverse=True)

    def get_traces(self, task: str = "", limit: int = 10) -> list[StigmergyTrace]:
        """Return the top-scoring traces, optionally filtered by *task*.

        If *task* is non-empty, task-specific traces (exact or partial match)
        are returned first; remaining slots are filled with global top traces.

        Args:
            task: If non-empty, prefer traces deposited for this task.
            limit: Maximum number of traces to return. Default ``10``.

        Returns:
            List of :class:`StigmergyTrace` objects, best-score-first.
        """
        if not task:
            return self._traces[:limit]

        task_lower = task.lower()
        task_specific = [t for t in self._traces if task_lower in t.task.lower()]
        other = [t for t in self._traces if task_lower not in t.task.lower()]
        merged = task_specific + other
        return merged[:limit]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def total_agents(self) -> int:
        """Total number of agents ever registered."""
        return int(self._graph.number_of_nodes())

    def generation_stats(self) -> dict[int, dict[str, float]]:
        """Per-generation fitness statistics.

        Returns:
            ``{generation: {"best": …, "mean": …, "count": …}}``.
        """
        gens: dict[int, list[float]] = {}
        for _, data in self._graph.nodes(data=True):
            gen = data.get("generation", 0)
            fit = data.get("fitness") or 0.0
            gens.setdefault(gen, []).append(fit)

        return {
            gen: {
                "best": max(fits),
                "mean": sum(fits) / len(fits),
                "count": len(fits),
            }
            for gen, fits in sorted(gens.items())
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialise the full lineage graph to a JSON string."""
        data = {
            "name": self.name,
            "nodes": [
                {"id": n, **attrs}
                for n, attrs in self._graph.nodes(data=True)
            ],
            "edges": [
                {"from": u, "to": v}
                for u, v in self._graph.edges()
            ],
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "EvolutionaryMemory":
        """Deserialise a lineage graph from a JSON string."""
        data = json.loads(json_str)
        memory = cls(name=data.get("name", "default"))
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            memory._graph.add_node(node_id, **node)
        for edge in data.get("edges", []):
            memory._graph.add_edge(edge["from"], edge["to"])
        return memory

    def __repr__(self) -> str:
        return (
            f"EvolutionaryMemory(name={self.name!r}, "
            f"agents={self.total_agents}, edges={self._graph.number_of_edges()})"
        )
