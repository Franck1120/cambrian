# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.evolution — EvolutionEngine."""

from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _EchoBackend:
    """Minimal backend that returns the input genome JSON unchanged."""

    model_name = "echo"

    def generate(self, prompt: str, **kwargs: object) -> str:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", prompt)
        if m:
            return m.group(1)
        # Try to find a JSON object in the prompt
        m2 = re.search(r"\{[\s\S]+\}", prompt)
        return m2.group(0) if m2 else "{}"


class _ConstantEvaluator:
    """Always returns the same fitness value."""

    def __init__(self, fitness: float = 0.5) -> None:
        self._fitness = fitness

    def __call__(self, agent: Agent, task: str) -> float:
        return self._fitness


class _CountingEvaluator:
    """Returns call_count / 100 so fitness increases monotonically."""

    def __init__(self) -> None:
        self._count = 0

    def __call__(self, agent: Agent, task: str) -> float:
        self._count += 1
        return min(1.0, self._count / 100)


def _make_engine(
    evaluator: object | None = None,
    population_size: int = 4,
    seed: int = 42,
) -> EvolutionEngine:
    backend = _EchoBackend()
    mutator = LLMMutator(backend=backend, fallback_on_error=True)  # type: ignore[arg-type]
    return EvolutionEngine(
        evaluator=evaluator or _ConstantEvaluator(),  # type: ignore[arg-type]
        mutator=mutator,
        population_size=population_size,
        mutation_rate=0.8,
        crossover_rate=0.3,
        elite_ratio=0.25,
        tournament_k=2,
        seed=seed,
    )


def _seed_genome(prompt: str = "solve the task") -> Genome:
    return Genome(system_prompt=prompt, temperature=0.7, model="gpt-4o-mini")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEvolutionEngine:
    def test_initialize_population_size(self) -> None:
        engine = _make_engine(population_size=6)
        pop = engine.initialize_population([_seed_genome()])
        assert len(pop) == 6

    def test_initialize_population_requires_seed(self) -> None:
        engine = _make_engine()
        with pytest.raises(ValueError):
            engine.initialize_population([])

    def test_evaluate_population_sets_fitness(self) -> None:
        engine = _make_engine()
        pop = engine.initialize_population([_seed_genome()])
        pop = engine.evaluate_population(pop, "test task")
        for a in pop:
            assert a.fitness is not None

    def test_evaluate_population_skips_already_evaluated(self) -> None:
        calls: list[int] = []

        def eval_fn(agent: Agent, task: str) -> float:
            calls.append(1)
            return 0.5

        engine = _make_engine(evaluator=eval_fn, population_size=3)
        pop = engine.initialize_population([_seed_genome()])
        # Pre-set fitness on two agents
        pop[0]._fitness = 0.9
        pop[1]._fitness = 0.1
        engine.evaluate_population(pop, "task")
        assert len(calls) == 1  # only the third agent evaluated

    def test_tournament_selection_returns_agent(self) -> None:
        engine = _make_engine(population_size=6)
        pop = engine.initialize_population([_seed_genome()])
        for a in pop:
            a._fitness = 0.5
        winner = engine.tournament_selection(pop)
        assert isinstance(winner, Agent)
        assert winner in pop

    def test_tournament_selection_prefers_high_fitness(self) -> None:
        engine = _make_engine(population_size=10, seed=0)
        pop = engine.initialize_population([_seed_genome()])
        for i, a in enumerate(pop):
            a._fitness = float(i) / len(pop)  # 0.0 → 0.9

        # With k=2 tournament from pool of 10, P(best wins) = k/n = 0.2.
        # Expected wins in 200 trials = 40. Threshold at 20 is a safe lower bound.
        wins = sum(
            1 for _ in range(200)
            if engine.tournament_selection(pop).fitness == pop[-1].fitness
        )
        assert wins > 20  # selection pressure sanity check (expected ~40)

    def test_evolve_generation_returns_same_size(self) -> None:
        engine = _make_engine(population_size=5)
        pop = engine.initialize_population([_seed_genome()])
        pop = engine.evaluate_population(pop, "task")
        next_gen = engine.evolve_generation(pop, "task")
        assert len(next_gen) == 5

    def test_full_evolve_returns_best_agent(self) -> None:
        counting_eval = _CountingEvaluator()
        engine = _make_engine(evaluator=counting_eval, population_size=4)
        best = engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test task",
            n_generations=2,
        )
        assert isinstance(best, Agent)
        assert best.fitness is not None
        assert best.fitness >= 0.0

    def test_best_property_tracks_global_best(self) -> None:
        engine = _make_engine(population_size=4)
        assert engine.best is None
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=1,
        )
        assert engine.best is not None

    def test_generation_counter(self) -> None:
        engine = _make_engine(population_size=4)
        assert engine.generation == 0
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=3,
        )
        assert engine.generation == 3

    def test_memory_populated(self) -> None:
        engine = _make_engine(population_size=4)
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=2,
        )
        assert engine.memory.total_agents > 0

    def test_on_generation_callback(self) -> None:
        calls: list[int] = []
        engine = _make_engine(population_size=4)

        def _cb(gen: int, pop: list[Agent]) -> None:
            calls.append(gen)

        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=3,
            on_generation=_cb,
        )
        # Called for gen 0 (initial) + gens 1, 2, 3
        assert calls == [0, 1, 2, 3]

    def test_archive_populated_when_enabled(self) -> None:
        engine = _make_engine(population_size=6)
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=2,
        )
        assert engine.archive is not None
        assert engine.archive.occupancy > 0
