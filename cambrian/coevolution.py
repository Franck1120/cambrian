"""Adversarial co-evolution — two populations evolving against each other.

In standard evolution, agents compete only against a fixed fitness function.
Co-evolution makes the challenge *adaptive*: a red team of adversaries
continuously probes generators for weaknesses, and generators must evolve
to resist those probes.

Architecture
------------

``BlueTeam`` (generators)
    Evolve to produce high-quality solutions to the base task.
    Their fitness is penalised when adversaries successfully break them.

``RedTeam`` (adversaries)
    Evolve to discover edge cases, adversarial inputs, or failure modes.
    Their fitness is the fraction of generators they successfully challenge.

This dynamic prevents over-fitting to a single evaluator and produces more
robust agents.

Example usage::

    from cambrian.coevolution import CoEvolutionEngine
    from cambrian.evaluators.code import CodeEvaluator
    from cambrian.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(model="gpt-4o-mini", api_key="...")

    def adversary_probe(adversary, generator, task):
        # Adversary writes a tricky variant of the task;
        # generator must handle it. Returns 1.0 if generator fails.
        adversarial_task = adversary.run(f"Write a tricky test for: {task}")
        try:
            response = generator.run(adversarial_task)
            return 0.0  # generator handled it
        except Exception:
            return 1.0  # generator broke

    engine = CoEvolutionEngine(
        task_evaluator=CodeEvaluator(expected_output="..."),
        adversarial_probe=adversary_probe,
        backend=backend,
        ...
    )
    best_gen, best_adv = engine.evolve(gen_seeds, adv_seeds, task="...", n_generations=8)
"""

from __future__ import annotations

import random
import time
from typing import Callable

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.diversity import MAPElites
from cambrian.memory import EvolutionaryMemory
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

# Callable signature for the adversarial probe
# (adversary_agent, generator_agent, base_task) -> float in [0, 1]
# Return 1.0 if adversary "wins" (breaks the generator), 0.0 if generator survives.
AdversarialProbeFn = Callable[[Agent, Agent, str], float]


class CoEvolutionEngine:
    """Adversarial co-evolution of two competing agent populations.

    Two populations co-evolve over the same number of generations:

    - **Blue team** (generators): solve the task; fitness penalised by
      adversarial break rate.
    - **Red team** (adversaries): probe generators for failures; fitness
      equals mean break rate across all generators.

    Args:
        task_evaluator: ``(generator, task) -> float`` — base fitness for
            generators (correctness, quality, etc.).
        adversarial_probe: ``(adversary, generator, task) -> float`` —
            how hard the adversary's challenge is for the generator.
            Return ``1.0`` if adversary wins, ``0.0`` if generator survives.
        generator_mutator: :class:`~cambrian.mutator.LLMMutator` for
            evolving generators.
        adversary_mutator: :class:`~cambrian.mutator.LLMMutator` for
            evolving adversaries.  Defaults to *generator_mutator*.
        backend: LLM backend attached to all created agents.
        population_size: Number of agents per team per generation.
            Default ``8``.
        adversary_penalty: Weight of adversarial failure on generator
            fitness.  ``generator_fitness = base * (1 - penalty * break_rate)``.
            Default ``0.3``.
        n_challenges: How many adversaries challenge each generator per
            generation.  Default ``3`` (capped at population_size).
        elite_ratio: Fraction of top agents preserved unchanged each gen.
            Default ``0.25``.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        task_evaluator: Callable[[Agent, str], float],
        adversarial_probe: AdversarialProbeFn,
        generator_mutator: LLMMutator,
        adversary_mutator: LLMMutator | None = None,
        backend: "LLMBackend | None" = None,
        population_size: int = 8,
        adversary_penalty: float = 0.3,
        n_challenges: int = 3,
        elite_ratio: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self._task_ev = task_evaluator
        self._probe = adversarial_probe
        self._gen_mutator = generator_mutator
        self._adv_mutator = adversary_mutator or generator_mutator
        self._backend = backend
        self._pop_size = population_size
        self._penalty = adversary_penalty
        self._n_challenges = min(n_challenges, population_size)
        self._elite_n = max(1, int(population_size * elite_ratio))

        if seed is not None:
            random.seed(seed)

        self._gen_archive = MAPElites()
        self._adv_archive = MAPElites()
        self._gen_memory = EvolutionaryMemory(name="blue-team")
        self._adv_memory = EvolutionaryMemory(name="red-team")
        self._generation = 0
        self._best_generator: Agent | None = None
        self._best_adversary: Agent | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def best_generator(self) -> "Agent | None":
        """Best generator seen across all generations."""
        return self._best_generator

    @property
    def best_adversary(self) -> "Agent | None":
        """Best adversary seen across all generations."""
        return self._best_adversary

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    # ── Public API ────────────────────────────────────────────────────────────

    def evolve(
        self,
        generator_seeds: list[Genome],
        adversary_seeds: list[Genome],
        task: str,
        n_generations: int = 10,
        on_generation: Callable[[int, list[Agent], list[Agent]], None] | None = None,
    ) -> tuple[Agent, Agent]:
        """Run the full adversarial co-evolution loop.

        Args:
            generator_seeds: Starting genomes for the blue team.
            adversary_seeds: Starting genomes for the red team.
            task: Base task description used for evaluation and mutation.
            n_generations: Number of generations to run.
            on_generation: Optional callback ``(gen, generators, adversaries)``
                invoked at the end of each generation.

        Returns:
            Tuple of ``(best_generator, best_adversary)`` found across all
            generations.
        """
        logger.info(
            "CoEvolution start: pop=%d, gens=%d, task=%r",
            self._pop_size, n_generations, task[:60],
        )

        generators = self._init_population(generator_seeds, "gen")
        adversaries = self._init_population(adversary_seeds, "adv")

        # Initial evaluation
        generators = self._evaluate_generators(generators, adversaries, task)
        adversaries = self._evaluate_adversaries(adversaries, generators, task)
        self._update_best(generators, adversaries)

        if on_generation:
            on_generation(0, generators, adversaries)

        for gen in range(1, n_generations + 1):
            self._generation = gen
            generators = self._evolve_team(generators, self._gen_mutator, task)
            adversaries = self._evolve_team(adversaries, self._adv_mutator, task)

            generators = self._evaluate_generators(generators, adversaries, task)
            adversaries = self._evaluate_adversaries(adversaries, generators, task)
            self._update_best(generators, adversaries)

            gen_best = max((a.fitness or 0.0) for a in generators)
            adv_best = max((a.fitness or 0.0) for a in adversaries)
            logger.info(
                "CoEvo gen %d/%d — gen_best=%.4f adv_best=%.4f",
                gen, n_generations, gen_best, adv_best,
            )

            if on_generation:
                on_generation(gen, generators, adversaries)

        best_gen = self._best_generator or max(generators, key=lambda a: a.fitness or 0.0)
        best_adv = self._best_adversary or max(adversaries, key=lambda a: a.fitness or 0.0)
        return best_gen, best_adv

    # ── Internals ─────────────────────────────────────────────────────────────

    def _init_population(self, seeds: list[Genome], prefix: str) -> list[Agent]:
        """Initialise a population from seed genomes."""
        pop: list[Agent] = []
        for i in range(self._pop_size):
            genome = seeds[i % len(seeds)]
            if i >= len(seeds):
                data = genome.to_dict()
                data["temperature"] = max(0.1, min(1.5, data["temperature"] + random.uniform(-0.15, 0.15)))
                genome = Genome.from_dict(data)
            pop.append(Agent(genome=genome, backend=self._backend))
        return pop

    def _evaluate_generators(
        self, generators: list[Agent], adversaries: list[Agent], task: str
    ) -> list[Agent]:
        """Score generators: base fitness penalised by adversarial break rate."""
        for gen_agent in generators:
            if gen_agent.fitness is not None:
                continue  # already scored this generation

            # Base task score
            t0 = time.monotonic()
            try:
                base_score = float(self._task_ev(gen_agent, task))
            except Exception as exc:
                logger.warning("Generator evaluator error: %s", exc)
                base_score = 0.0

            # Adversarial challenge score
            challengers = random.sample(adversaries, min(self._n_challenges, len(adversaries)))
            break_scores: list[float] = []
            for adv_agent in challengers:
                try:
                    break_score = float(self._probe(adv_agent, gen_agent, task))
                except Exception as exc:
                    logger.warning("Adversarial probe error: %s", exc)
                    break_score = 0.0
                break_scores.append(break_score)

            break_rate = sum(break_scores) / len(break_scores) if break_scores else 0.0
            final = max(0.0, base_score * (1.0 - self._penalty * break_rate))
            gen_agent.fitness = final

            logger.debug(
                "Generator %s: base=%.4f break_rate=%.3f final=%.4f (%.2fs)",
                gen_agent.id[:8], base_score, break_rate, final,
                time.monotonic() - t0,
            )
            self._gen_archive.add(gen_agent)

        return generators

    def _evaluate_adversaries(
        self, adversaries: list[Agent], generators: list[Agent], task: str
    ) -> list[Agent]:
        """Score adversaries: mean break rate across generators."""
        for adv_agent in adversaries:
            if adv_agent.fitness is not None:
                continue

            targets = random.sample(generators, min(self._n_challenges, len(generators)))
            break_scores: list[float] = []
            for gen_agent in targets:
                try:
                    score = float(self._probe(adv_agent, gen_agent, task))
                except Exception as exc:
                    logger.warning("Adversarial probe error: %s", exc)
                    score = 0.0
                break_scores.append(score)

            adv_agent.fitness = sum(break_scores) / len(break_scores) if break_scores else 0.0
            self._adv_archive.add(adv_agent)

        return adversaries

    def _evolve_team(self, population: list[Agent], mutator: LLMMutator, task: str) -> list[Agent]:
        """Produce next generation for one team via elitism + mutation."""
        population.sort(key=lambda a: a.fitness or 0.0, reverse=True)
        next_gen: list[Agent] = list(population[: self._elite_n])

        while len(next_gen) < self._pop_size:
            parent = self._tournament(population)
            child = mutator.mutate(parent, task)
            child._fitness = None  # force re-evaluation
            next_gen.append(child)

        return next_gen

    def _tournament(self, population: list[Agent], k: int = 3) -> Agent:
        """Simple tournament selection."""
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=lambda a: a.fitness or 0.0)

    def _update_best(self, generators: list[Agent], adversaries: list[Agent]) -> None:
        for a in generators:
            if a.fitness is None:
                continue
            if self._best_generator is None or a.fitness > (self._best_generator.fitness or 0.0):
                self._best_generator = a
        for a in adversaries:
            if a.fitness is None:
                continue
            if self._best_adversary is None or a.fitness > (self._best_adversary.fitness or 0.0):
                self._best_adversary = a

    def __repr__(self) -> str:
        return (
            f"CoEvolutionEngine(pop={self._pop_size}, "
            f"gen={self._generation}, penalty={self._penalty})"
        )
