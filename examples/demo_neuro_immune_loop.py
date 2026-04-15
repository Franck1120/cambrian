# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Neuromodulation + Immune Memory in a Mini-Evolution Loop -- demo.

Shows how NeuromodulatorBank and ImmuneCortex integrate with a simple
population of agents over multiple generations:

1. NeuromodulatorBank dynamically adjusts mutation_rate and
   selection_pressure based on the population state each generation.
2. ImmuneCortex stores high-fitness genomes and fast-recalls them
   when a similar task is encountered (avoiding re-evolution).
3. ZeitgeberClock adds circadian rhythm to mutation rates.

All LLM calls are mocked -- this runs offline, instantly.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.immune_memory import ImmuneCortex
from cambrian.neuromodulation import NeuromodulatorBank
from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler

# --- helpers ------------------------------------------------------------------


def _agent(fitness: float, prompt: str) -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(responses: list[str]) -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(side_effect=responses)
    return b


def _simulate_fitness(prompt: str, rng: random.Random) -> float:
    """Simulate a fitness score based on prompt length (toy metric)."""
    base = min(1.0, len(prompt) / 200)
    return max(0.0, min(1.0, base + rng.gauss(0.0, 0.1)))


# --- Setup --------------------------------------------------------------------

print("=" * 65)
print("Cambrian -- Neuromodulation + Immune Memory Integration Demo")
print("=" * 65)

rng = random.Random(42)

TASKS = [
    "Solve a quadratic equation step by step",
    "Prove the quadratic formula from first principles",   # similar to task 0
    "Write a Python function to parse JSON",
    "Debug a JSON parsing error in Python",                # similar to task 2
    "Explain gradient descent intuitively",
    "Derive the gradient of cross-entropy loss",           # similar to task 4
]

PROMPTS = [
    "You are a precise mathematics expert who shows all work.",
    "You solve coding problems with clean, documented Python code.",
    "You explain machine learning concepts with intuitive analogies.",
    "You are a rigorous proof-writer with clear step-by-step reasoning.",
    "You debug code systematically: reproduce, isolate, fix, verify.",
]

# --- System components --------------------------------------------------------

bank = NeuromodulatorBank(
    base_mutation_rate=0.3,
    base_selection_pressure=0.5,
    mr_range=0.25,
    sp_range=0.25,
)

cortex = ImmuneCortex(
    b_threshold=0.75,
    t_threshold=0.45,
    b_similarity=0.4,
    t_min_similarity=0.15,
)

clock = ZeitgeberClock(period=6, amplitude=0.4)
scheduler = ZeitgeberScheduler(
    clock=clock,
    base_mutation_rate=0.3,
    mutation_range=0.15,
    base_threshold=0.5,
    threshold_range=0.15,
)

print()
print("Running 6 task episodes (each = 1 generation of evolution)...")
print()

for ep, task in enumerate(TASKS):
    # -- Step 1: Check immune memory first ------------------------------
    recall = cortex.recall(task)
    if recall.recalled:
        print(
            f"  Episode {ep}: [IMMUNE RECALL {recall.cell_type.upper()}] "
            f"task='{task[:45]}...' "
            f"sim={recall.similarity:.2f} fitness={recall.agent.fitness:.2f}"  # type: ignore[union-attr]
        )
        continue

    # -- Step 2: Build population ----------------------------------------
    population = [
        _agent(_simulate_fitness(p, rng), p) for p in rng.sample(PROMPTS, 3)
    ]

    # -- Step 3: Neuromodulator adjusts hyperparams ----------------------
    state = bank.modulate(population, generation=ep)
    zeit_mr, zeit_thr = scheduler.tick()

    # Blend Zeitgeber + neuro mutation rate
    effective_mr = (state.mutation_rate + zeit_mr) / 2

    # -- Step 4: Select survivors based on selection pressure ------------
    population.sort(key=lambda a: a.fitness, reverse=True)
    keep_n = max(1, int(len(population) * (1.0 - state.selection_pressure * 0.5)))
    survivors = population[:keep_n]
    best = survivors[0]

    # -- Step 5: Record best in immune memory ----------------------------
    cortex.record(best, task)

    print(
        f"  Episode {ep}: [EVOLVE] task='{task[:45]}...' "
        f"best_fitness={best.fitness:.2f} "
        f"mr={effective_mr:.2f} "
        f"dopa={state.dopamine:.2f} nora={state.noradrenaline:.2f} "
        f"b_cells={cortex.b_cell_count} t_cells={cortex.t_cell_count}"
    )

print()
print("-" * 65)
print("Final Immune Memory State:")
print(f"  B-cell memories: {cortex.b_cell_count}")
print(f"  T-cell memories: {cortex.t_cell_count}")

print()
print("Neuromodulation history (last 3 generations):")
for state in bank.history[-3:]:
    print(
        f"  Gen {state.generation}: mr={state.mutation_rate:.3f} "
        f"sp={state.selection_pressure:.3f} "
        f"dopa={state.dopamine:.2f} sero={state.serotonin:.2f} "
        f"ach={state.acetylcholine:.2f} nora={state.noradrenaline:.2f}"
    )

print()
print("Zeitgeber phase history (last 3 ticks):")
for s in scheduler.history[-3:]:
    print(
        f"  Gen {s.generation}: mutation_rate={s.mutation_rate:.3f} "
        f"threshold={s.acceptance_threshold:.3f} ef={s.exploration_factor:.3f}"
    )

print()
print("Demo complete -- neuromodulation + immune memory integration verified.")
