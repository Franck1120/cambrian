# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Population archipelago model for Cambrian — island-based evolution.

The archipelago model runs N *isolated island populations* in parallel.
Each island evolves independently for a configured number of generations
before a *migration event* exchanges top individuals between islands.
This prevents premature convergence to local optima and maintains
population diversity across the broader meta-population.

Architecture
------------

:class:`Island`
    Wraps an :class:`~cambrian.evolution.EvolutionEngine` with a unique island
    ID and a list of agents.  Exposes a ``run_generation`` step used by the
    coordinator.

:class:`Archipelago`
    Coordinator that manages multiple islands.  Drives the full
    ``evolve_archipelago`` loop:

    1. Each island runs ``migration_interval`` generations independently.
    2. A migration event selects ``migration_rate × island_size`` elites from
       each island and distributes them to randomly chosen neighbours.
    3. Repeat until ``n_generations`` total have elapsed.

Migration topologies
--------------------
- ``"ring"`` — each island sends migrants to the next island in a ring (default)
- ``"all_to_all"`` — every island can receive migrants from every other island
- ``"random"`` — each island picks a random neighbour per migration event

Usage::

    from cambrian.archipelago import Archipelago
    from cambrian.evolution import EvolutionEngine

    arch = Archipelago(
        engine_factory=lambda: EvolutionEngine(evaluator=my_eval, mutator=my_mutator),
        n_islands=4,
        island_size=20,
        migration_interval=5,
        migration_rate=0.1,
        topology="ring",
    )
    best = arch.evolve(seed_genomes=seeds, task="Write a haiku.", n_generations=40)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from cambrian.agent import Agent, Genome
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

_TOPOLOGIES = {"ring", "all_to_all", "random"}


# ── Island ─────────────────────────────────────────────────────────────────────


@dataclass
class Island:
    """A single isolated population within the archipelago.

    Attributes:
        island_id: Unique integer identifier.
        population: Current list of agents on this island.
        generation: Number of generations this island has completed.
        best_fitness_history: Best fitness seen at each generation.
    """

    island_id: int
    population: list[Agent] = field(default_factory=list)
    generation: int = 0
    best_fitness_history: list[float] = field(default_factory=list)

    def best_agent(self) -> Agent | None:
        """Return the highest-fitness agent on the island."""
        alive = [a for a in self.population if a.fitness is not None]
        if not alive:
            return None
        return max(alive, key=lambda a: a.fitness or 0.0)

    def top_agents(self, n: int) -> list[Agent]:
        """Return the top-*n* agents by fitness."""
        alive = [a for a in self.population if a.fitness is not None]
        alive.sort(key=lambda a: a.fitness or 0.0, reverse=True)
        return alive[:n]

    def receive_migrants(self, migrants: list[Agent]) -> None:
        """Integrate incoming *migrants*, replacing lowest-fitness locals.

        Args:
            migrants: Agents migrating from another island.
        """
        if not migrants or not self.population:
            return
        # Sort population by fitness ascending (weakest first)
        self.population.sort(key=lambda a: a.fitness or 0.0)
        n_replace = min(len(migrants), len(self.population))
        for i, migrant in enumerate(migrants[:n_replace]):
            clone = migrant.clone()
            if migrant.fitness is not None:
                clone.fitness = migrant.fitness  # preserve fitness across migration
            self.population[i] = clone
        logger.debug(
            "Island %d received %d migrants", self.island_id, n_replace
        )


# ── Archipelago ────────────────────────────────────────────────────────────────


class Archipelago:
    """Multi-island evolutionary coordinator.

    Args:
        engine_factory: Callable that returns a fresh
            :class:`~cambrian.evolution.EvolutionEngine` for each island.
            Called *n_islands* times during initialisation.
        n_islands: Number of independent island populations.
        island_size: Target population size per island.
        migration_interval: Generations between migration events.
        migration_rate: Fraction of island population that migrates (0.0–1.0).
        topology: Migration topology — ``"ring"``, ``"all_to_all"``, or
            ``"random"``.
        seed: Random seed for reproducibility (``None`` = non-deterministic).
    """

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        n_islands: int = 4,
        island_size: int = 20,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
        topology: str = "ring",
        seed: int | None = None,
    ) -> None:
        if topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {_TOPOLOGIES}, got {topology!r}")
        if not (0.0 <= migration_rate <= 1.0):
            raise ValueError("migration_rate must be in [0, 1]")

        self._engine_factory = engine_factory
        self._n_islands = n_islands
        self._island_size = island_size
        self._migration_interval = migration_interval
        self._migration_rate = migration_rate
        self._topology = topology
        self._rng = random.Random(seed)

        # Build engines and islands
        self._engines: list[Any] = [engine_factory() for _ in range(n_islands)]
        self._islands: list[Island] = [Island(island_id=i) for i in range(n_islands)]

        self._total_migrations: int = 0

    # ── Topology helpers ───────────────────────────────────────────────────────

    def _neighbours(self, island_id: int) -> list[int]:
        """Return island IDs that *island_id* sends migrants to."""
        n = self._n_islands
        if n <= 1:
            return []
        if self._topology == "ring":
            return [(island_id + 1) % n]
        if self._topology == "all_to_all":
            return [i for i in range(n) if i != island_id]
        # "random"
        candidates = [i for i in range(n) if i != island_id]
        return [self._rng.choice(candidates)]

    # ── Migration event ────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Perform one migration event across all islands."""
        n_migrants = max(1, int(self._island_size * self._migration_rate))
        # Collect emigrants from each island before modifying any
        emigrants: dict[int, list[Agent]] = {}
        for island in self._islands:
            emigrants[island.island_id] = island.top_agents(n_migrants)

        # Distribute to neighbours
        for island in self._islands:
            for neighbour_id in self._neighbours(island.island_id):
                migrants = emigrants[island.island_id]
                self._islands[neighbour_id].receive_migrants(migrants)

        self._total_migrations += 1
        logger.debug("Migration event #%d: %d migrants/island", self._total_migrations, n_migrants)

    # ── Main evolution loop ────────────────────────────────────────────────────

    def evolve(
        self,
        seed_genomes: list[Genome],
        task: str,
        n_generations: int,
        on_migration: Callable[["Archipelago", int], None] | None = None,
    ) -> Agent:
        """Run the full archipelago evolution.

        Args:
            seed_genomes: Initial genomes.  They are distributed round-robin
                across islands if fewer seeds than total slots.
            task: Task description passed to each island's evaluator.
            n_generations: *Total* generations to run across all islands.
            on_migration: Optional callback called after each migration event
                with ``(archipelago, migration_number)``.

        Returns:
            The best agent found across all islands.
        """
        # Seed each island
        self._seed_islands(seed_genomes, task)

        generations_done = 0
        while generations_done < n_generations:
            # Each island runs migration_interval steps
            steps = min(self._migration_interval, n_generations - generations_done)
            for i, (engine, island) in enumerate(zip(self._engines, self._islands)):
                island.population = self._run_island_steps(
                    engine=engine,
                    island=island,
                    task=task,
                    steps=steps,
                )
                if island.population:
                    best = island.best_agent()
                    if best is not None:
                        island.best_fitness_history.append(best.fitness or 0.0)

            generations_done += steps

            # Migration event (skip after the final batch)
            if generations_done < n_generations:
                self._migrate()
                if on_migration is not None:
                    on_migration(self, self._total_migrations)

        return self.best_agent()

    # ── Island step runner ─────────────────────────────────────────────────────

    def _seed_islands(self, seed_genomes: list[Genome], task: str) -> None:
        """Distribute seed genomes across islands and evaluate them."""
        for i, island in enumerate(self._islands):
            engine = self._engines[i]
            # Assign genomes round-robin
            island_seeds = [
                seed_genomes[j % len(seed_genomes)]
                for j in range(i, i + self._island_size)
            ] if seed_genomes else []
            # Use engine internals to build initial population
            population = engine._seed_population(island_seeds, self._island_size)
            engine._evaluate_population(population, task)
            island.population = population

    def _run_island_steps(
        self,
        engine: Any,
        island: Island,
        task: str,
        steps: int,
    ) -> list[Agent]:
        """Run *steps* generations on an island using its engine.

        Args:
            engine: EvolutionEngine for this island.
            island: Island metadata (read-only here).
            task: Task string.
            steps: Number of generations to run.

        Returns:
            Updated population after *steps* generations.
        """
        population = island.population
        for _ in range(steps):
            population = engine._run_one_generation(population, task)
            island.generation += 1
        return population

    # ── Queries ────────────────────────────────────────────────────────────────

    def best_agent(self) -> Agent:
        """Return the best agent across all islands."""
        candidates: list[Agent] = []
        for island in self._islands:
            best = island.best_agent()
            if best is not None:
                candidates.append(best)
        if not candidates:
            raise RuntimeError("No evaluated agents found in archipelago.")
        return max(candidates, key=lambda a: a.fitness or 0.0)

    def island_summaries(self) -> list[dict[str, Any]]:
        """Summary statistics for each island."""
        summaries: list[dict[str, Any]] = []
        for island in self._islands:
            alive = [a.fitness for a in island.population if a.fitness is not None]
            summaries.append({
                "island_id": island.island_id,
                "size": len(island.population),
                "generations": island.generation,
                "best_fitness": max(alive) if alive else None,
                "mean_fitness": sum(alive) / len(alive) if alive else None,
            })
        return summaries

    @property
    def total_migrations(self) -> int:
        """Total number of migration events that have occurred."""
        return self._total_migrations

    def __repr__(self) -> str:
        return (
            f"Archipelago(islands={self._n_islands}, "
            f"island_size={self._island_size}, "
            f"topology={self._topology!r}, "
            f"migrations={self._total_migrations})"
        )
