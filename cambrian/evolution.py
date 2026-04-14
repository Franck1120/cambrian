"""EvolutionEngine — the main evolutionary loop.

Orchestrates selection, crossover, mutation, evaluation, and archiving across
generations. Supports both synchronous and async evaluation patterns and
integrates MAP-Elites diversity archiving with EvolutionaryMemory lineage
tracking.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.diversity import MAPElites
from cambrian.memory import EvolutionaryMemory
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


class EvolutionEngine:
    """Runs the full evolutionary loop over a population of agents.

    Args:
        evaluator: Callable ``(agent, task) -> float`` that scores an agent.
        mutator: :class:`~cambrian.mutator.LLMMutator` for genome modification.
        backend: LLM backend attached to every agent created by the engine.
            When ``None`` (default), agents are created without a backend —
            valid only if the evaluator does not call ``agent.run()``.
        population_size: Number of agents maintained per generation. Default 10.
        mutation_rate: Probability that an agent is mutated each generation.
            Range ``[0.0, 1.0]``. Default ``0.8``.
        crossover_rate: Probability that two parents produce a crossover child
            instead of a direct clone. Default ``0.3``.
        elite_ratio: Fraction of top agents carried unchanged to the next
            generation (elitism). Default ``0.2`` (top 20 %).
        tournament_k: Tournament size for parent selection. Default ``3``.
        use_map_elites: Whether to maintain a MAP-Elites diversity archive in
            addition to the standard population. Default ``True``.
        memory_name: Label for the :class:`~cambrian.memory.EvolutionaryMemory`
            run. Default ``"default"``.
        seed: Optional random seed for reproducibility.
        compress_interval: Every this many generations, apply
            :func:`~cambrian.compress.procut_prune` to all agents to prevent
            prompt bloat. ``0`` disables auto-compression. Default ``0``.
        compress_max_tokens: Token budget passed to ``procut_prune``.
            Default ``256``.
    """

    def __init__(
        self,
        evaluator: Callable[[Agent, str], float],
        mutator: LLMMutator,
        backend: "LLMBackend | None" = None,
        population_size: int = 10,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        elite_ratio: float = 0.2,
        tournament_k: int = 3,
        use_map_elites: bool = True,
        memory_name: str = "default",
        seed: int | None = None,
        compress_interval: int = 0,
        compress_max_tokens: int = 256,
    ) -> None:
        self._evaluator = evaluator
        self._mutator = mutator
        self._backend = backend
        self._pop_size = population_size
        self._mut_rate = mutation_rate
        self._xo_rate = crossover_rate
        self._elite_n = max(1, int(population_size * elite_ratio))
        self._k = tournament_k
        self._use_map_elites = use_map_elites
        self._compress_interval = compress_interval
        self._compress_max_tokens = compress_max_tokens

        if seed is not None:
            random.seed(seed)

        self._archive = MAPElites() if use_map_elites else None
        self._memory = EvolutionaryMemory(name=memory_name)
        self._generation = 0
        self._best_agent: Agent | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def generation(self) -> int:
        """Current generation number (0-indexed before first evolve call)."""
        return self._generation

    @property
    def best(self) -> Agent | None:
        """Best agent seen across all generations."""
        return self._best_agent

    @property
    def archive(self) -> MAPElites | None:
        """MAP-Elites diversity archive, if enabled."""
        return self._archive

    @property
    def memory(self) -> EvolutionaryMemory:
        """Lineage graph for the current run."""
        return self._memory

    def initialize_population(self, seed_genomes: list[Genome]) -> list[Agent]:
        """Create the initial population from a list of seed genomes.

        If ``len(seed_genomes) < population_size``, genomes are randomly
        sampled (with replacement) and tweaked to fill the population.

        Args:
            seed_genomes: Starting genomes. At least one required.

        Returns:
            The initial population (length == ``population_size``).
        """
        if not seed_genomes:
            raise ValueError("At least one seed genome is required.")

        population: list[Agent] = []
        for i in range(self._pop_size):
            genome = seed_genomes[i % len(seed_genomes)]
            # Slightly diversify temperature for duplicates
            if i >= len(seed_genomes):
                data = genome.to_dict()
                delta = random.uniform(-0.15, 0.15)
                data["temperature"] = max(0.1, min(1.5, data["temperature"] + delta))
                genome = Genome.from_dict(data)
            agent = Agent(genome=genome, backend=self._backend)
            population.append(agent)

        return population

    def evaluate_population(
        self, population: list[Agent], task: str
    ) -> list[Agent]:
        """Evaluate every agent in *population* that has no fitness score yet.

        Args:
            population: Agents to evaluate.
            task: Task string passed to the evaluator.

        Returns:
            The same list, with ``fitness`` set on each agent.
        """
        for agent in population:
            if agent.fitness is not None:
                continue
            t0 = time.monotonic()
            try:
                score = self._evaluator(agent, task)
            except Exception as exc:
                logger.warning("Evaluator raised %s: %s", type(exc).__name__, exc)
                score = 0.0
            agent._fitness = float(score)
            elapsed = time.monotonic() - t0
            logger.debug(
                "Evaluated agent %s: fitness=%.4f (%.2fs)", agent.id[:8], score, elapsed
            )

            self._memory.update_fitness(agent.id, float(score))
            if self._archive is not None:
                self._archive.add(agent)

        return population

    def evolve(
        self,
        seed_genomes: list[Genome],
        task: str,
        n_generations: int = 10,
        on_generation: Callable[[int, list[Agent]], None] | None = None,
    ) -> Agent:
        """Run the full evolutionary loop.

        Args:
            seed_genomes: Starting genomes for the initial population.
            task: Task description used by evaluator and mutator.
            n_generations: Number of generations to run. Default 10.
            on_generation: Optional callback invoked at the end of each
                generation with ``(generation_number, population)``.

        Returns:
            The best :class:`~cambrian.agent.Agent` found across all
            generations.
        """
        logger.info(
            "Starting evolution: pop=%d, gens=%d, task=%r",
            self._pop_size,
            n_generations,
            task[:60],
        )

        population = self.initialize_population(seed_genomes)

        # Register genesis agents in memory
        for agent in population:
            self._memory.add_agent(
                agent_id=agent.id,
                generation=0,
                fitness=None,
                genome_snapshot=agent.genome.to_dict(),
            )

        population = self.evaluate_population(population, task)
        self._update_best(population)

        if on_generation:
            on_generation(0, population)

        for gen in range(1, n_generations + 1):
            self._generation = gen
            population = self.evolve_generation(population, task)
            self._update_best(population)

            # Auto-compress prompts every compress_interval generations
            if self._compress_interval > 0 and gen % self._compress_interval == 0:
                population = self._compress_population(population)
                logger.info("Gen %d: prompt auto-compression applied", gen)

            stats = self._generation_stats(population)
            logger.info(
                "Gen %d/%d — best=%.4f mean=%.4f",
                gen,
                n_generations,
                stats["best"],
                stats["mean"],
            )

            if on_generation:
                on_generation(gen, population)

        if self._best_agent is None and population:
            self._best_agent = max(population, key=lambda a: a.fitness or 0.0)

        logger.info(
            "Evolution complete. Best fitness=%.4f",
            self._best_agent.fitness if self._best_agent else 0.0,
        )
        return self._best_agent  # type: ignore[return-value]

    def evolve_generation(
        self, population: list[Agent], task: str
    ) -> list[Agent]:
        """Produce the next generation from the current *population*.

        Steps:
        1. Sort by fitness (descending).
        2. Carry the top ``elite_n`` agents unchanged (elitism).
        3. Fill the remaining slots via tournament selection + crossover/mutation.
        4. Evaluate new agents.

        Args:
            population: Current generation's agents (all must have fitness).
            task: Task string.

        Returns:
            Next generation population of the same size.
        """
        population.sort(key=lambda a: a.fitness or 0.0, reverse=True)

        # Elites survive unchanged
        next_gen: list[Agent] = population[: self._elite_n]

        # Fill remaining slots
        while len(next_gen) < self._pop_size:
            if random.random() < self._xo_rate and len(population) >= 2:
                parent_a = self.tournament_selection(population)
                parent_b = self.tournament_selection(population)
                # Avoid identical parents
                for _ in range(3):
                    if parent_b.id != parent_a.id:
                        break
                    parent_b = self.tournament_selection(population)
                child = self._mutator.crossover(parent_a, parent_b, task)
            else:
                parent = self.tournament_selection(population)
                if random.random() < self._mut_rate:
                    child = self._mutator.mutate(parent, task)
                else:
                    child = parent.clone()
                    child._fitness = None

            # Register in memory
            parents_used = (
                [parent_a.id, parent_b.id]
                if "parent_a" in dir()
                else [parent.id]  # type: ignore[name-defined]
            )
            self._memory.add_agent(
                agent_id=child.id,
                generation=self._generation,
                fitness=None,
                genome_snapshot=child.genome.to_dict(),
                parents=parents_used,
            )
            next_gen.append(child)

        next_gen = self.evaluate_population(next_gen, task)
        return next_gen

    def tournament_selection(self, population: list[Agent]) -> Agent:
        """Select one agent via tournament selection.

        Randomly samples ``tournament_k`` agents from *population* and returns
        the one with the highest fitness.

        Args:
            population: Pool of candidates. Must be non-empty.

        Returns:
            The tournament winner.
        """
        k = min(self._k, len(population))
        contestants = random.sample(population, k)
        return max(contestants, key=lambda a: a.fitness or 0.0)

    # ── Population persistence ────────────────────────────────────────────────

    def save_population(self, path: str | Path, population: list[Agent]) -> None:
        """Serialise *population* to a JSON file at *path*.

        The saved file is a JSON array of agent dicts compatible with
        :meth:`load_population`.  Each entry includes the genome,
        fitness, generation, and agent id.

        Args:
            path: Destination file path (created or overwritten).
            population: List of agents to serialise.
        """
        data = [agent.to_dict() for agent in population]
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved %d agents to %s", len(population), path)

    def load_population(self, path: str | Path) -> list[Agent]:
        """Deserialise a population from a JSON file written by :meth:`save_population`.

        Loaded agents have their genome, fitness, and generation restored.
        The engine's current backend is attached to each loaded agent.

        Args:
            path: Path to a JSON file produced by :meth:`save_population`.

        Returns:
            List of :class:`~cambrian.agent.Agent` objects ready for use.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be parsed as a valid population.
        """
        raw = Path(path).read_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse population file {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Population file must be a JSON array, got {type(data).__name__}")

        population: list[Agent] = []
        for entry in data:
            genome = Genome.from_dict(entry.get("genome", {}))
            agent = Agent(
                genome=genome,
                backend=self._backend,
                agent_id=entry.get("id"),
            )
            if entry.get("fitness") is not None:
                agent._fitness = float(entry["fitness"])
            agent._generation = int(entry.get("generation", 0))
            population.append(agent)

        logger.info("Loaded %d agents from %s", len(population), path)
        return population

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compress_population(self, population: list[Agent]) -> list[Agent]:
        """Apply procut_prune to all agents to prevent prompt bloat."""
        from cambrian.compress import procut_prune
        compressed = []
        for agent in population:
            new_genome = procut_prune(agent.genome, max_tokens=self._compress_max_tokens)
            if new_genome.system_prompt != agent.genome.system_prompt:
                agent.genome = new_genome
            compressed.append(agent)
        return compressed

    def _update_best(self, population: list[Agent]) -> None:
        for agent in population:
            if agent.fitness is None:
                continue
            if self._best_agent is None or agent.fitness > (self._best_agent.fitness or 0.0):
                self._best_agent = agent

    @staticmethod
    def _generation_stats(population: list[Agent]) -> dict[str, float]:
        scores = [a.fitness for a in population if a.fitness is not None]
        if not scores:
            return {"best": 0.0, "mean": 0.0, "count": 0.0}
        return {
            "best": max(scores),
            "mean": sum(scores) / len(scores),
            "count": float(len(scores)),
        }
