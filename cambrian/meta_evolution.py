# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Meta-evolution: MAML-inspired hyperparameter self-tuning.

Instead of fixing ``mutation_rate``, ``crossover_rate``, and ``temperature``
for the entire run, the meta-evolution layer treats these as *evolvable
parameters*.  Each generation, the engine tries small perturbations of the
hyperparameters and keeps configurations that improve population fitness.

The approach is inspired by MAML (Model-Agnostic Meta-Learning, Finn et al.
2017) — a few gradient-like steps on the meta-objective (population
fitness improvement) guide the hyperparameter search.

Architecture
------------

:class:`HyperParams`
    Mutable bundle of evolution hyperparameters.

:class:`MetaEvolutionEngine`
    Wraps :class:`~cambrian.evolution.EvolutionEngine`.  After each
    generation, it perturbs the hyperparameters, evaluates whether they
    improve fitness, and keeps the best configuration.

Usage::

    from cambrian.meta_evolution import MetaEvolutionEngine, HyperParams

    hp = HyperParams(mutation_rate=0.8, crossover_rate=0.3, temperature=0.6)
    meta = MetaEvolutionEngine(
        evaluator=my_evaluator,
        mutator=my_mutator,
        backend=backend,
        initial_hp=hp,
        meta_lr=0.05,
        n_candidates=3,
        population_size=8,
    )
    best = meta.evolve(seed_genomes=[seed], task="...", n_generations=15)
    print(meta.hp)  # final hyperparameters
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from cambrian.agent import Agent, Genome
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HyperParams
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HyperParams:
    """Mutable bundle of evolvable hyperparameters.

    All fields are clipped to their valid ranges before use.

    Attributes:
        mutation_rate: Probability of mutating a child ``[0.0, 1.0]``.
        crossover_rate: Probability of crossover vs direct clone ``[0.0, 1.0]``.
        temperature: Mutation LLM sampling temperature ``(0.0, 2.0]``.
        tournament_k: Tournament selection size (int ≥ 1).
        elite_ratio: Fraction of elite agents ``[0.0, 0.5]``.
    """

    mutation_rate: float = 0.8
    crossover_rate: float = 0.3
    temperature: float = 0.6
    tournament_k: int = 3
    elite_ratio: float = 0.2

    # History of fitness improvements per generation
    fitness_history: list[float] = field(default_factory=list)

    def clamp(self) -> "HyperParams":
        """Return a new HyperParams with all values clamped to valid ranges."""
        return HyperParams(
            mutation_rate=max(0.0, min(1.0, self.mutation_rate)),
            crossover_rate=max(0.0, min(1.0, self.crossover_rate)),
            temperature=max(0.05, min(2.0, self.temperature)),
            tournament_k=max(1, self.tournament_k),
            elite_ratio=max(0.0, min(0.5, self.elite_ratio)),
            fitness_history=list(self.fitness_history),
        )

    def perturb(self, scale: float = 0.05, rng: random.Random | None = None) -> "HyperParams":
        """Return a new HyperParams with small Gaussian perturbations.

        Args:
            scale: Standard deviation of each perturbation as a fraction of
                the parameter's current value.
            rng: Optional :class:`random.Random` instance for reproducibility.

        Returns:
            Perturbed and clamped :class:`HyperParams`.
        """
        _rng = rng or random
        return HyperParams(
            mutation_rate=self.mutation_rate + _rng.gauss(0, scale),
            crossover_rate=self.crossover_rate + _rng.gauss(0, scale),
            temperature=self.temperature + _rng.gauss(0, scale * 0.5),
            tournament_k=max(1, self.tournament_k + _rng.choice([-1, 0, 0, 1])),
            elite_ratio=self.elite_ratio + _rng.gauss(0, scale * 0.5),
        ).clamp()

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a plain dictionary (excludes history)."""
        return {
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "temperature": self.temperature,
            "tournament_k": self.tournament_k,
            "elite_ratio": self.elite_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> "HyperParams":
        """Deserialise from a plain dictionary."""
        return cls(
            mutation_rate=float(data.get("mutation_rate", 0.8)),
            crossover_rate=float(data.get("crossover_rate", 0.3)),
            temperature=float(data.get("temperature", 0.6)),
            tournament_k=int(data.get("tournament_k", 3)),
            elite_ratio=float(data.get("elite_ratio", 0.2)),
        )

    def __repr__(self) -> str:
        return (
            f"HyperParams(mut={self.mutation_rate:.3f}, "
            f"xover={self.crossover_rate:.3f}, "
            f"temp={self.temperature:.3f}, "
            f"k={self.tournament_k}, "
            f"elite={self.elite_ratio:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MetaEvolutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class MetaEvolutionEngine:
    """MAML-inspired meta-evolution: evolves hyperparameters alongside genomes.

    At the end of each generation, the engine:
    1. Generates ``n_candidates`` perturbed hyperparameter configurations.
    2. For each candidate, runs one quick evaluation step (single generation
       on the current population with the candidate's settings).
    3. Keeps the configuration with the highest mean population fitness.
    4. Applies that configuration to the next generation.

    This creates a second-order optimisation loop: the outer loop tunes
    *how* evolution works; the inner loop (standard EvolutionEngine) tunes
    *what* the agents do.

    Args:
        evaluator: Agent evaluator (same as EvolutionEngine).
        mutator: LLMMutator (same as EvolutionEngine).
        backend: LLM backend (optional; passed to EvolutionEngine).
        initial_hp: Starting hyperparameters.
        meta_lr: Perturbation scale for hyperparameter search. Default ``0.05``.
        n_candidates: Number of hyperparameter candidates to try each
            meta-step. Default ``3``.
        population_size: Population size (fixed across meta and inner loops).
        seed: Random seed.
        **engine_kwargs: Additional kwargs forwarded to EvolutionEngine
            (e.g. ``memory``, ``compress_interval``).
    """

    def __init__(
        self,
        evaluator: Callable[[Agent, str], float] | Evaluator,
        mutator: LLMMutator,
        backend: "LLMBackend | None" = None,
        initial_hp: HyperParams | None = None,
        meta_lr: float = 0.05,
        n_candidates: int = 3,
        population_size: int = 8,
        seed: int | None = None,
        **engine_kwargs: object,
    ) -> None:
        self._evaluator = evaluator
        self._mutator = mutator
        self._backend = backend
        self.hp = (initial_hp or HyperParams()).clamp()
        self._meta_lr = meta_lr
        self._n_candidates = n_candidates
        self._pop_size = population_size
        self._seed = seed
        self._engine_kwargs = engine_kwargs
        self._rng = random.Random(seed)
        self._hp_history: list[HyperParams] = [copy.copy(self.hp)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_engine(self, hp: HyperParams) -> EvolutionEngine:
        """Construct an EvolutionEngine with the given hyperparameters."""
        self._mutator._mut_temp = hp.temperature
        return EvolutionEngine(
            evaluator=self._evaluator,
            mutator=self._mutator,
            backend=self._backend,
            population_size=self._pop_size,
            mutation_rate=hp.mutation_rate,
            crossover_rate=hp.crossover_rate,
            tournament_k=hp.tournament_k,
            seed=self._seed,
            **self._engine_kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _mean_fitness(population: list[Agent]) -> float:
        scores = [a.fitness for a in population if a.fitness is not None]
        return sum(scores) / max(len(scores), 1)

    # ------------------------------------------------------------------
    # Meta-step: tune hyperparameters
    # ------------------------------------------------------------------

    def _meta_step(self, population: list[Agent], task: str) -> HyperParams:
        """Try n_candidates perturbations; return the best-performing hp."""
        candidates = [self.hp.perturb(self._meta_lr, self._rng) for _ in range(self._n_candidates)]
        best_hp = self.hp
        best_fitness = self._mean_fitness(population)

        for cand in candidates:
            engine = self._make_engine(cand)
            engine._population = population  # type: ignore[attr-defined]
            try:
                trial_pop = engine.evolve_generation(population, task)
                fitness = self._mean_fitness(trial_pop)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_hp = cand
            except Exception as exc:  # noqa: BLE001
                logger.debug("Meta-step candidate failed: %s", exc)

        logger.debug(
            "Meta-step: old=%s  best=%s  Δfitness=%.4f",
            self.hp,
            best_hp,
            best_fitness - self._mean_fitness(population),
        )
        return best_hp

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def evolve(
        self,
        seed_genomes: list[Genome],
        task: str,
        n_generations: int = 10,
        meta_interval: int = 2,
        on_generation: Callable[[int, list[Agent], HyperParams], None] | None = None,
    ) -> Agent:
        """Run meta-evolution: evolve agents and hyperparameters simultaneously.

        Args:
            seed_genomes: Initial genomes.
            task: Task description.
            n_generations: Total number of generations.
            meta_interval: Run a meta-step every this many generations.
                Default ``2`` (tune hps every other generation).
            on_generation: Optional callback ``(gen, population, hp)`` called
                after each generation.

        Returns:
            Best :class:`~cambrian.agent.Agent` found.
        """
        engine = self._make_engine(self.hp)
        population = engine.initialize_population(seed_genomes)
        population = engine.evaluate_population(population, task)

        best_agent: Agent | None = None

        def _update_best(pop: list[Agent]) -> None:
            nonlocal best_agent
            candidates = [a for a in pop if a.fitness is not None]
            if not candidates:
                return
            local_best = max(candidates, key=lambda a: a.fitness or 0.0)
            if best_agent is None or (local_best.fitness or 0.0) > (best_agent.fitness or 0.0):
                best_agent = local_best

        _update_best(population)
        if on_generation:
            on_generation(0, population, self.hp)

        for gen in range(1, n_generations + 1):
            population = engine.evolve_generation(population, task)
            _update_best(population)

            mean = self._mean_fitness(population)
            self.hp.fitness_history.append(mean)

            # Meta-step: tune hyperparameters every meta_interval gens
            if gen % meta_interval == 0:
                new_hp = self._meta_step(population, task)
                if new_hp is not self.hp:
                    self.hp = new_hp
                    self._hp_history.append(copy.copy(new_hp))
                    engine = self._make_engine(self.hp)

            logger.info(
                "Meta-gen %d/%d  mean=%.4f  hp=%s",
                gen,
                n_generations,
                mean,
                self.hp,
            )

            if on_generation:
                on_generation(gen, population, self.hp)

        if best_agent is None and population:
            best_agent = max(population, key=lambda a: a.fitness or 0.0)

        assert best_agent is not None, (
            "evolve() found no agents — ensure seeds is non-empty"
        )
        return best_agent

    @property
    def hp_history(self) -> list[HyperParams]:
        """All hyperparameter snapshots taken during evolution."""
        return list(self._hp_history)
