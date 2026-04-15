# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Demo: Island-model evolution with Archipelago — no API key required.

Archipelago runs multiple independent evolutionary populations (islands) in
parallel, then periodically migrates the best agents between islands.  This
prevents premature convergence and explores more of the prompt space than a
single-island run.

Topology options
----------------
- ``ring``        — each island sends migrants to its right neighbour
- ``all_to_all``  — every island migrates to every other island
- ``random``      — migration targets chosen randomly each interval

This demo uses a keyword-based evaluator so it runs fully offline.

Usage
-----
    python examples/demo_archipelago.py
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.archipelago import Archipelago
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator

# ---------------------------------------------------------------------------
# Keyword evaluator — offline, no API key needed
# ---------------------------------------------------------------------------

_KEYWORDS = [
    "step-by-step", "expert", "verify", "systematic", "analytical",
    "precise", "structured", "rigorous", "methodical", "validate",
]


class KeywordEvaluator(Evaluator):
    """Score: 0.1 base + 0.09 per keyword found in the system prompt."""

    def evaluate(self, agent: Agent, task: str) -> float:  # noqa: ARG002
        prompt = agent.genome.system_prompt.lower()
        hits = sum(1 for kw in _KEYWORDS if kw in prompt)
        return min(0.1 + 0.09 * hits, 1.0)


# ---------------------------------------------------------------------------
# Mock backend — deterministically adds keywords to every genome it mutates
# ---------------------------------------------------------------------------

def _make_backend() -> MagicMock:
    """Return a mock backend that enriches prompts with performance keywords."""
    def generate(prompt: str) -> str:  # noqa: ARG001
        genome_dict = {
            "system_prompt": (
                "You are an expert. step-by-step systematic analytical rigorous "
                "methodical validate verify structured precise."
            ),
            "strategy": "step-by-step",
            "temperature": 0.7,
            "model": "gpt-4o-mini",
            "tools": [],
            "few_shot_examples": [],
        }
        return json.dumps(genome_dict)

    b = MagicMock()
    b.generate = MagicMock(side_effect=generate)
    return b


# ---------------------------------------------------------------------------
# EvolutionEngine adapter
#
# Archipelago calls engine._seed_population, engine._evaluate_population, and
# engine._run_one_generation — thin wrappers around the public EvolutionEngine
# API (initialize_population, evaluate_population, evolve_generation).
# ---------------------------------------------------------------------------

class ArchipelagoEngine(EvolutionEngine):
    """EvolutionEngine subclass that exposes the private methods Archipelago needs."""

    def _seed_population(self, seed_genomes: list[Genome], size: int) -> list[Agent]:
        # Use a subset of seeds and let the engine fill the rest
        seeds = seed_genomes[:size] if seed_genomes else []
        return self.initialize_population(seeds)

    def _evaluate_population(self, population: list[Agent], task: str) -> None:
        self.evaluate_population(population, task)

    def _run_one_generation(self, population: list[Agent], task: str) -> list[Agent]:
        return self.evolve_generation(population, task)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine() -> ArchipelagoEngine:
    """Factory called once per island by Archipelago."""
    backend = _make_backend()
    return ArchipelagoEngine(
        evaluator=KeywordEvaluator(),
        mutator=LLMMutator(backend=backend),
        backend=backend,
        population_size=4,
        elite_ratio=0.25,
        mutation_rate=0.8,
    )


# ---------------------------------------------------------------------------
# Archipelago demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Cambrian — Archipelago (island-model) demo")
    print("=" * 60)

    arch = Archipelago(
        engine_factory=make_engine,
        n_islands=3,
        island_size=4,
        migration_interval=2,
        migration_rate=0.3,
        topology="ring",
        seed=42,
    )

    seed_genomes = [
        Genome(system_prompt="You are a helpful assistant."),
        Genome(system_prompt="Answer questions clearly and concisely."),
    ]

    print("\nIslands: 3  |  Topology: ring  |  Migration every 2 generations\n")

    def on_migration(a: Archipelago, generation: int) -> None:
        print(f"  [gen {generation:2d}] Migration #{a.total_migrations}  —  island snapshots:")
        for s in a.island_summaries():
            best_f = s.get("best_fitness") or 0.0
            size = s.get("size", 0)
            print(f"           island {s['island_id']}: pop={size}, best={best_f:.3f}")

    best = arch.evolve(
        seed_genomes=seed_genomes,
        task="Explain quantum computing to a high-school student.",
        n_generations=6,
        on_migration=on_migration,
    )

    print("\n" + "=" * 60)
    print(f"Total migrations: {arch.total_migrations}")
    if best.fitness is not None:
        print(f"Best fitness    : {best.fitness:.4f}")
    print(f"Best prompt     : {best.genome.system_prompt[:120]!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
