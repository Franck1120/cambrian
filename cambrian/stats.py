# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Advanced population statistics for Cambrian evolution runs.

Provides three analysis tools:

``ParetoAnalyzer``
    Multi-objective Pareto front: identifies agents that are not dominated
    by any other on *all* objectives simultaneously.  Useful for understanding
    trade-offs (e.g., fitness vs. prompt brevity).

    Note: for NSGA-II incremental selection use :class:`~cambrian.pareto.ParetoFront`.

``DiversityTracker``
    Tracks population diversity over time using several complementary metrics:
    - Unique strategies count
    - Temperature standard deviation
    - Prompt-length standard deviation (token estimate)
    - Mean pairwise edit distance (Levenshtein, on a sample)

``FitnessLandscape``
    Bins agents into a 2D grid (temperature × prompt-length) and reports the
    mean fitness in each cell — a coarse-grained view of the fitness landscape.

All three can operate on :class:`~cambrian.agent.Agent` objects directly or on
the plain dicts produced by :meth:`~cambrian.memory.EvolutionaryMemory.to_json`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cambrian.agent import Agent


# ── Pareto Front ──────────────────────────────────────────────────────────────


@dataclass
class ParetoPoint:
    """A single point in multi-objective space.

    Attributes:
        agent_id: Identifier of the agent this point represents.
        objectives: Dict mapping objective name → value (higher = better).
        is_pareto: ``True`` if this point is on the Pareto front.
        dominated_by: IDs of agents that dominate this point.
    """

    agent_id: str
    objectives: dict[str, float]
    is_pareto: bool = False
    dominated_by: list[str] = field(default_factory=list)


class ParetoAnalyzer:
    """Compute and query the Pareto front of a population.

    Two objectives are supported by default:

    - ``fitness``: agent fitness score (higher is better).
    - ``brevity``: inverse of prompt token count (shorter prompt = higher
      brevity = better for deployment cost).

    Custom objectives can be added via :meth:`add_objective`.

    Args:
        objectives: List of objective names to compute.  Default
            ``["fitness", "brevity"]``.

    .. note::
        For NSGA-II incremental non-dominated selection use
        :class:`~cambrian.pareto.ParetoFront` instead.

    Example::

        front = ParetoAnalyzer()
        front.compute(population)
        pareto_agents = front.pareto_agents()
        print(f"Pareto front: {len(pareto_agents)} agents")
    """

    DEFAULT_OBJECTIVES = ["fitness", "brevity"]

    def __init__(self, objectives: list[str] | None = None) -> None:
        self._objective_names = objectives or list(self.DEFAULT_OBJECTIVES)
        self._points: list[ParetoPoint] = []
        self._custom: dict[str, Any] = {}

    def add_objective(self, name: str, fn: Any) -> None:
        """Register a custom objective function ``(Agent) -> float``.

        Args:
            name: Objective name.
            fn: Callable that maps an :class:`~cambrian.agent.Agent` to a float
                (higher = better).
        """
        if name not in self._objective_names:
            self._objective_names.append(name)
        self._custom[name] = fn

    def compute(self, agents: "list[Agent]") -> list[ParetoPoint]:
        """Compute the Pareto front for *agents*.

        Args:
            agents: Population to analyse.

        Returns:
            List of :class:`ParetoPoint` objects (one per agent), with
            ``is_pareto`` set to ``True`` for non-dominated agents.
        """
        points = [self._extract_point(a) for a in agents]
        self._mark_pareto(points)
        self._points = points
        return points

    def pareto_agents(self) -> list[ParetoPoint]:
        """Return only the Pareto-optimal points from the last :meth:`compute` call."""
        return [p for p in self._points if p.is_pareto]

    def dominated_agents(self) -> list[ParetoPoint]:
        """Return dominated (non-Pareto) points from the last :meth:`compute` call."""
        return [p for p in self._points if not p.is_pareto]

    def summary(self) -> dict[str, Any]:
        """High-level summary of the Pareto analysis.

        Returns:
            Dict with ``total``, ``pareto_count``, ``dominated_count``,
            ``pareto_fraction``, ``objectives``.
        """
        if not self._points:
            return {"total": 0, "pareto_count": 0, "dominated_count": 0,
                    "pareto_fraction": 0.0, "objectives": self._objective_names}
        pareto_n = sum(1 for p in self._points if p.is_pareto)
        return {
            "total": len(self._points),
            "pareto_count": pareto_n,
            "dominated_count": len(self._points) - pareto_n,
            "pareto_fraction": pareto_n / len(self._points),
            "objectives": self._objective_names,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _extract_point(self, agent: "Agent") -> ParetoPoint:
        obj: dict[str, float] = {}
        if "fitness" in self._objective_names:
            obj["fitness"] = float(agent.fitness or 0.0)
        if "brevity" in self._objective_names:
            tokens = max(1, agent.genome.token_count())
            obj["brevity"] = 1.0 / tokens
        for name, fn in self._custom.items():
            if name in self._objective_names:
                try:
                    obj[name] = float(fn(agent))
                except Exception:
                    obj[name] = 0.0
        return ParetoPoint(agent_id=agent.agent_id, objectives=obj)

    @staticmethod
    def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        """``True`` if *a* dominates *b* (better or equal on all, strict on at least one)."""
        a_vals = list(a.objectives.values())
        b_vals = list(b.objectives.values())
        return (
            all(av >= bv for av, bv in zip(a_vals, b_vals))
            and any(av > bv for av, bv in zip(a_vals, b_vals))
        )

    @staticmethod
    def _mark_pareto(points: list[ParetoPoint]) -> None:
        for i, p in enumerate(points):
            p.dominated_by = []
            for j, q in enumerate(points):
                if i != j and ParetoAnalyzer._dominates(q, p):
                    p.dominated_by.append(q.agent_id)
            p.is_pareto = len(p.dominated_by) == 0

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ParetoAnalyzer(pareto={s.get('pareto_count')}/{s.get('total')}, "
            f"objectives={self._objective_names})"
        )


# Backwards-compatible alias — deprecated, use ParetoAnalyzer
ParetoFront = ParetoAnalyzer


# ── Diversity Tracker ─────────────────────────────────────────────────────────


@dataclass
class DiversitySnapshot:
    """Diversity metrics for one generation.

    Attributes:
        generation: Generation number.
        n_agents: Population size.
        unique_strategies: Count of distinct strategy strings.
        temperature_std: Standard deviation of temperatures (0.0 if population size < 2).
        prompt_token_std: Standard deviation of prompt token counts.
        strategy_entropy: Shannon entropy of strategy distribution (bits).
        mean_fitness: Mean fitness at this generation.
    """

    generation: int
    n_agents: int
    unique_strategies: int
    temperature_std: float
    prompt_token_std: float
    strategy_entropy: float
    mean_fitness: float


class DiversityTracker:
    """Track population diversity across generations.

    Call :meth:`record` at the end of each generation to snapshot the
    population's diversity metrics.

    Example::

        tracker = DiversityTracker()

        def on_gen(gen, pop):
            tracker.record(gen, pop)

        engine.evolve(..., on_generation=on_gen)

        timeline = tracker.timeline()
        print(f"Final diversity: {timeline[-1]}")
    """

    def __init__(self) -> None:
        self._snapshots: list[DiversitySnapshot] = []

    def record(self, generation: int, agents: "list[Agent]") -> DiversitySnapshot:
        """Capture diversity metrics for the current generation.

        Args:
            generation: Current generation number.
            agents: Current population.

        Returns:
            The :class:`DiversitySnapshot` created.
        """
        if not agents:
            snap = DiversitySnapshot(generation, 0, 0, 0.0, 0.0, 0.0, 0.0)
            self._snapshots.append(snap)
            return snap

        strategies = [a.genome.strategy or "" for a in agents]
        temperatures = [a.genome.temperature for a in agents]
        token_counts = [a.genome.token_count() for a in agents]
        fitnesses = [a.fitness or 0.0 for a in agents]

        unique_strategies = len(set(strategies))
        temp_std = statistics.stdev(temperatures) if len(temperatures) > 1 else 0.0
        token_std = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0
        entropy = self._strategy_entropy(strategies)
        mean_fit = sum(fitnesses) / len(fitnesses)

        snap = DiversitySnapshot(
            generation=generation,
            n_agents=len(agents),
            unique_strategies=unique_strategies,
            temperature_std=round(temp_std, 4),
            prompt_token_std=round(token_std, 2),
            strategy_entropy=round(entropy, 4),
            mean_fitness=round(mean_fit, 4),
        )
        self._snapshots.append(snap)
        return snap

    def timeline(self) -> list[DiversitySnapshot]:
        """Return all recorded snapshots in chronological order."""
        return list(self._snapshots)

    def diversity_collapsed(self, threshold_entropy: float = 0.3) -> bool:
        """``True`` if the last generation shows diversity collapse.

        Diversity collapse is a signal that the population has converged
        prematurely to a single phenotype.

        Args:
            threshold_entropy: Minimum strategy entropy to consider diverse.
                Default ``0.3``.
        """
        if not self._snapshots:
            return False
        return self._snapshots[-1].strategy_entropy < threshold_entropy

    def to_dicts(self) -> list[dict[str, Any]]:
        """Export snapshots as a list of plain dicts (for JSON serialisation)."""
        return [
            {
                "generation": s.generation,
                "n_agents": s.n_agents,
                "unique_strategies": s.unique_strategies,
                "temperature_std": s.temperature_std,
                "prompt_token_std": s.prompt_token_std,
                "strategy_entropy": s.strategy_entropy,
                "mean_fitness": s.mean_fitness,
            }
            for s in self._snapshots
        ]

    @staticmethod
    def _strategy_entropy(strategies: list[str]) -> float:
        """Shannon entropy of strategy distribution in bits."""
        import math

        if len(strategies) <= 1:
            return 0.0
        counts: dict[str, int] = {}
        for s in strategies:
            counts[s] = counts.get(s, 0) + 1
        n = len(strategies)
        return -sum(
            (c / n) * math.log2(c / n) for c in counts.values() if c > 0
        )

    def __repr__(self) -> str:
        return f"DiversityTracker(generations={len(self._snapshots)})"


# ── Fitness Landscape ─────────────────────────────────────────────────────────


class FitnessLandscape:
    """Coarse-grained 2D fitness landscape map (temperature × prompt-length).

    Bins agents into a grid and computes mean fitness per cell.  Helps
    identify which regions of the genome space are fertile (high mean
    fitness) versus barren.

    Args:
        n_temp_bins: Number of temperature bins. Default ``5``.
        n_token_bins: Number of prompt-token-count bins. Default ``5``.
        temp_range: (min, max) temperature. Default ``(0.0, 2.0)``.
        token_range: (min, max) token count. Default ``(0, 500)``.
    """

    def __init__(
        self,
        n_temp_bins: int = 5,
        n_token_bins: int = 5,
        temp_range: tuple[float, float] = (0.0, 2.0),
        token_range: tuple[int, int] = (0, 500),
    ) -> None:
        self._n_t = n_temp_bins
        self._n_k = n_token_bins
        self._temp_range = temp_range
        self._token_range = token_range
        # grid[temp_bin][token_bin] → list of fitness values
        self._grid: list[list[list[float]]] = [
            [[] for _ in range(n_token_bins)] for _ in range(n_temp_bins)
        ]

    def add(self, agent: "Agent") -> None:
        """Add one agent's (temperature, token_count, fitness) to the grid.

        Agents with ``None`` fitness are silently ignored.

        Args:
            agent: The agent to record.
        """
        if agent.fitness is None:
            return
        t_bin = self._temp_bin(agent.genome.temperature)
        k_bin = self._token_bin(agent.genome.token_count())
        self._grid[t_bin][k_bin].append(float(agent.fitness))

    def add_population(self, agents: "list[Agent]") -> None:
        """Add an entire population at once.

        Args:
            agents: Population to record.
        """
        for a in agents:
            self.add(a)

    def mean_fitness_grid(self) -> list[list[float]]:
        """Return a 2D grid of mean fitness values.

        Returns:
            ``grid[temp_bin][token_bin]`` → mean fitness (or ``0.0`` if empty).
        """
        return [
            [statistics.mean(cell) if cell else 0.0 for cell in row]
            for row in self._grid
        ]

    def peak(self) -> tuple[int, int, float]:
        """Return ``(temp_bin, token_bin, mean_fitness)`` of the highest cell.

        Returns:
            Tuple of (temperature_bin_index, token_bin_index, mean_fitness).
        """
        best = (0, 0, 0.0)
        for ti, row in enumerate(self._grid):
            for ki, cell in enumerate(row):
                if cell:
                    m = statistics.mean(cell)
                    if m > best[2]:
                        best = (ti, ki, m)
        return best

    def bin_labels(self) -> dict[str, list[str]]:
        """Human-readable labels for each bin.

        Returns:
            Dict with keys ``"temperature"`` and ``"tokens"``, each a list of
            label strings for the bins.
        """
        t_min, t_max = self._temp_range
        t_width = (t_max - t_min) / self._n_t
        temp_labels = [
            f"{t_min + i * t_width:.1f}-{t_min + (i + 1) * t_width:.1f}"
            for i in range(self._n_t)
        ]

        k_min, k_max = self._token_range
        k_width = (k_max - k_min) / self._n_k
        token_labels = [
            f"{int(k_min + i * k_width)}-{int(k_min + (i + 1) * k_width)}"
            for i in range(self._n_k)
        ]
        return {"temperature": temp_labels, "tokens": token_labels}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the landscape to a plain dict."""
        labels = self.bin_labels()
        return {
            "n_temp_bins": self._n_t,
            "n_token_bins": self._n_k,
            "temperature_labels": labels["temperature"],
            "token_labels": labels["tokens"],
            "mean_fitness_grid": self.mean_fitness_grid(),
            "peak": self.peak(),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _temp_bin(self, temperature: float) -> int:
        t_min, t_max = self._temp_range
        frac = (temperature - t_min) / max(t_max - t_min, 1e-9)
        return min(int(frac * self._n_t), self._n_t - 1)

    def _token_bin(self, token_count: int) -> int:
        k_min, k_max = self._token_range
        frac = (token_count - k_min) / max(k_max - k_min, 1)
        return min(max(int(frac * self._n_k), 0), self._n_k - 1)

    def __repr__(self) -> str:
        peak = self.peak()
        return (
            f"FitnessLandscape({self._n_t}x{self._n_k}, "
            f"peak_fitness={peak[2]:.4f})"
        )
