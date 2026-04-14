"""Multi-objective Pareto optimisation for Cambrian — NSGA-II selection.

Cambrian agents are normally ranked by a single scalar fitness value.  This
module extends the framework with **multi-objective optimisation**: agents are
simultaneously evaluated on multiple objectives (e.g. task performance *and*
prompt brevity) and ranked using the NSGA-II algorithm [Deb2002].

Architecture
------------

:class:`ObjectiveVector`
    Named collection of per-objective scores for a single agent.

:class:`ParetoFront`
    Maintains the non-dominated set of agents across all objectives.
    Supports incremental addition and querying.

:func:`fast_non_dominated_sort`
    O(M·N²) NSGA-II dominance ranking.  Returns ranked layers (fronts).

:func:`crowding_distance`
    Diversity-preserving distance in objective space for agents within a front.

:func:`nsga2_select`
    Full NSGA-II selection: sort → crowding → truncate to population size.

Built-in objectives
-------------------
- ``fitness_objective`` — standard scalar fitness from ``agent.fitness``
- ``brevity_objective`` — inversely proportional to system prompt token count
- ``diversity_objective`` — per-population novelty score (call
  ``attach_diversity_scores`` before selection)

Usage::

    from cambrian.pareto import nsga2_select, ParetoFront, ObjectiveVector

    # After evaluating a population:
    vectors = [
        ObjectiveVector(agent_id=a.id, scores={"perf": a.fitness, "brevity": brevity(a)})
        for a in population
    ]
    selected = nsga2_select(population, vectors, target_size=50)

References
----------
[Deb2002] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cambrian.agent import Agent
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

# ── ObjectiveVector ────────────────────────────────────────────────────────────


@dataclass
class ObjectiveVector:
    """Per-agent scores across multiple objectives.

    Attributes:
        agent_id: The agent this vector belongs to.
        scores: Mapping of objective name → scalar score.
            All scores are assumed to be *maximised* (higher = better).
        rank: NSGA-II non-domination rank (filled by :func:`fast_non_dominated_sort`).
        crowding: Crowding distance within its front (filled by :func:`crowding_distance`).
    """

    agent_id: str
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding: float = 0.0

    def dominates(self, other: "ObjectiveVector") -> bool:
        """``True`` if *self* weakly dominates *other* on all objectives and
        strictly dominates on at least one.

        Both vectors must share the same objective keys; missing keys default
        to 0.0.
        """
        keys = set(self.scores) | set(other.scores)
        better_on_any = False
        for k in keys:
            s = self.scores.get(k, 0.0)
            o = other.scores.get(k, 0.0)
            if s < o:
                return False
            if s > o:
                better_on_any = True
        return better_on_any

    def objective_names(self) -> list[str]:
        """Sorted list of objective keys."""
        return sorted(self.scores.keys())


# ── ParetoFront ────────────────────────────────────────────────────────────────


class ParetoFront:
    """Maintains the non-dominated set of agents.

    Agents are added incrementally.  Any existing member that is dominated by
    the new agent is evicted; the new agent is only admitted if it is not
    dominated by any existing member.

    Args:
        objectives: Objective names this front tracks.  Used for display only;
            the actual scores are read from :class:`ObjectiveVector`.
    """

    def __init__(self, objectives: list[str] | None = None) -> None:
        self._objectives = objectives or []
        self._members: dict[str, ObjectiveVector] = {}

    def add(self, vec: ObjectiveVector) -> bool:
        """Attempt to add *vec* to the front.

        Returns:
            ``True`` if *vec* was admitted (i.e. it is non-dominated).
        """
        # Check if vec is dominated by any current member
        for member in self._members.values():
            if member.dominates(vec):
                return False

        # Remove current members dominated by vec
        dominated = [aid for aid, m in self._members.items() if vec.dominates(m)]
        for aid in dominated:
            del self._members[aid]

        self._members[vec.agent_id] = vec
        return True

    def members(self) -> list[ObjectiveVector]:
        """All non-dominated vectors, sorted by agent_id."""
        return sorted(self._members.values(), key=lambda v: v.agent_id)

    def agents(self, population: list[Agent]) -> list[Agent]:
        """Return agents from *population* that are in the front."""
        ids = set(self._members)
        return [a for a in population if a.id in ids]

    def size(self) -> int:
        """Number of non-dominated members."""
        return len(self._members)

    def __repr__(self) -> str:
        return f"ParetoFront(size={self.size()}, objectives={self._objectives})"


# ── NSGA-II core functions ────────────────────────────────────────────────────


def fast_non_dominated_sort(
    vectors: list[ObjectiveVector],
) -> list[list[ObjectiveVector]]:
    """NSGA-II non-dominated sorting.

    Assigns each vector a ``rank`` (front index, 0 = best) and returns the
    fronts as a list of lists.

    Args:
        vectors: Objective vectors for all agents.

    Returns:
        Ordered list of fronts.  ``fronts[0]`` is the Pareto-optimal set.

    Complexity:
        O(M·N²) where M = number of objectives, N = population size.
    """
    n = len(vectors)
    if n == 0:
        return []

    # dominated_by[i] = count of vectors that dominate i
    dominated_by: list[int] = [0] * n
    # dominates_set[i] = indices dominated by i
    dominates_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if vectors[i].dominates(vectors[j]):
                dominates_set[i].append(j)
            elif vectors[j].dominates(vectors[i]):
                dominated_by[i] += 1

    fronts: list[list[ObjectiveVector]] = []
    current_front_indices = [i for i in range(n) if dominated_by[i] == 0]

    rank = 0
    while current_front_indices:
        front = [vectors[i] for i in current_front_indices]
        for vec in front:
            vec.rank = rank
        fronts.append(front)

        next_front_indices: list[int] = []
        for i in current_front_indices:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front_indices.append(j)

        current_front_indices = next_front_indices
        rank += 1

    return fronts


def crowding_distance(front: list[ObjectiveVector]) -> None:
    """Assign crowding distances to all vectors in *front* (in-place).

    Boundary solutions receive infinite distance.  Interior solutions are
    scored by the average normalised gap to their nearest neighbours in each
    objective dimension.

    Args:
        front: A single non-dominated front from :func:`fast_non_dominated_sort`.
    """
    n = len(front)
    if n == 0:
        return
    for vec in front:
        vec.crowding = 0.0
    if n <= 2:
        for vec in front:
            vec.crowding = math.inf
        return

    keys = front[0].objective_names()
    for k in keys:
        sorted_front = sorted(front, key=lambda v: v.scores.get(k, 0.0))
        sorted_front[0].crowding = math.inf
        sorted_front[-1].crowding = math.inf
        f_min = sorted_front[0].scores.get(k, 0.0)
        f_max = sorted_front[-1].scores.get(k, 0.0)
        f_range = f_max - f_min
        if f_range == 0.0:
            continue
        for idx in range(1, n - 1):
            prev_val = sorted_front[idx - 1].scores.get(k, 0.0)
            next_val = sorted_front[idx + 1].scores.get(k, 0.0)
            sorted_front[idx].crowding += (next_val - prev_val) / f_range


def nsga2_select(
    population: list[Agent],
    vectors: list[ObjectiveVector],
    target_size: int,
) -> list[Agent]:
    """NSGA-II selection: non-dominated sort + crowding-distance truncation.

    Args:
        population: All candidate agents.
        vectors: Objective vectors aligned with *population* by index (or
            matching by ``agent_id``).
        target_size: Number of agents to return.

    Returns:
        Selected agents in NSGA-II rank/crowding order.
    """
    if not population:
        return []
    target_size = min(target_size, len(population))

    # Index vectors by agent_id for lookup
    vec_by_id: dict[str, ObjectiveVector] = {v.agent_id: v for v in vectors}

    fronts = fast_non_dominated_sort(vectors)
    for front in fronts:
        crowding_distance(front)

    selected: list[Agent] = []
    agent_by_id = {a.id: a for a in population}

    for front in fronts:
        if len(selected) >= target_size:
            break
        remaining = target_size - len(selected)
        if len(front) <= remaining:
            for vec in front:
                agent = agent_by_id.get(vec.agent_id)
                if agent is not None:
                    selected.append(agent)
        else:
            # Sort by crowding (descending) and take what fits
            sorted_front = sorted(front, key=lambda v: v.crowding, reverse=True)
            for vec in sorted_front[:remaining]:
                agent = agent_by_id.get(vec.agent_id)
                if agent is not None:
                    selected.append(agent)

    logger.debug(
        "NSGA-II selected %d/%d agents across %d fronts",
        len(selected), len(population), len(fronts),
    )
    return selected


# ── Built-in objective functions ──────────────────────────────────────────────


def fitness_objective(agent: Agent) -> float:
    """Standard single-objective fitness score (higher = better)."""
    return agent.fitness or 0.0


def brevity_objective(agent: Agent, max_tokens: int = 2000) -> float:
    """Reward for shorter system prompts.

    Returns a score in [0, 1] where 1 = empty prompt, 0 = exceeds max_tokens.

    Args:
        agent: The agent to score.
        max_tokens: Token budget (approximate: 1 token ≈ 4 chars).
    """
    char_budget = max_tokens * 4
    length = len(agent.genome.system_prompt)
    return max(0.0, 1.0 - length / char_budget)


def attach_diversity_scores(
    population: list[Agent],
    vectors: list[ObjectiveVector],
    objective_name: str = "diversity",
    k: int = 3,
) -> None:
    """Attach a novelty/diversity score to each vector (in-place).

    Diversity is the mean distance to the k nearest neighbours in the
    *existing objective space* (using all other objectives).  Agents in
    sparse regions get higher scores.

    Args:
        population: Population (used to extract prompt text for distance).
        vectors: Objective vectors to augment.
        objective_name: Key under which to store the diversity score.
        k: Number of nearest neighbours.
    """
    prompts = [a.genome.system_prompt for a in population]

    def _char_dist(a: str, b: str) -> float:
        """Approximate string distance (normalised by max length)."""
        max_len = max(len(a), len(b), 1)
        # Count differing characters at each position
        overlap = sum(ca == cb for ca, cb in zip(a, b))
        return 1.0 - overlap / max_len

    n = len(prompts)
    for i, vec in enumerate(vectors):
        if n <= 1:
            vec.scores[objective_name] = 1.0
            continue
        dists = sorted(
            _char_dist(prompts[i], prompts[j])
            for j in range(n) if j != i
        )
        neighbours = dists[:k]
        vec.scores[objective_name] = sum(neighbours) / len(neighbours)
