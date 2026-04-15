# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Demo: Multi-objective evolution with Pareto/NSGA-II — no API key required.

Standard single-objective evolution maximises one fitness score.  In real
settings you often care about multiple goals simultaneously, e.g.:

  - Maximise response quality  AND  minimise prompt verbosity
  - Maximise accuracy           AND  minimise latency (token count)

Cambrian's NSGA-II implementation maintains a Pareto front: agents that are
not dominated on any objective.  It rewards diversity across the trade-off
surface, avoiding the collapse to a single solution.

Key classes
-----------
- ``ObjectiveVector``      — holds per-agent multi-objective scores
- ``ParetoFront``          — tracks the non-dominated set
- ``fast_non_dominated_sort`` — rank agents into Pareto fronts
- ``nsga2_select``         — select survivors using rank + crowding distance

No API key required — evaluators run offline.

Usage
-----
    python examples/demo_multi_objective.py
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.pareto import (
    ObjectiveVector,
    ParetoFront,
    attach_diversity_scores,
    brevity_objective,
    crowding_distance,
    fast_non_dominated_sort,
    fitness_objective,
    nsga2_select,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUALITY_WORDS = [
    "step-by-step", "expert", "verify", "systematic", "analytical",
    "precise", "structured", "rigorous", "methodical", "validate",
]


def quality_score(agent: Agent) -> float:
    """Quality: keyword coverage of the system prompt."""
    prompt = agent.genome.system_prompt.lower()
    hits = sum(1 for kw in _QUALITY_WORDS if kw in prompt)
    return min(0.1 + 0.09 * hits, 1.0)


def brevity_score(agent: Agent) -> float:
    """Brevity: shorter prompts score higher."""
    return brevity_objective(agent, max_tokens=200)


def make_agent(prompt: str) -> Agent:
    a = Agent(genome=Genome(system_prompt=prompt))
    a.fitness = quality_score(a)
    return a


# ---------------------------------------------------------------------------
# Build a diverse population: short-vs-verbose trade-off
# ---------------------------------------------------------------------------

SEED_PROMPTS = [
    # Short + low quality
    "Help me.",
    "Answer.",
    "You help.",
    # Medium — some keywords
    "You are an expert. Be precise and verify your answers.",
    "Answer systematically step-by-step.",
    "Structured, methodical responses please.",
    # Long + high quality
    (
        "You are an expert analytical assistant. Answer every question "
        "step-by-step, verify each step, use structured reasoning, and "
        "validate your conclusions rigorously and methodically."
    ),
    (
        "You are a precise, rigorous, systematic expert. Think analytically, "
        "validate assumptions, structure all responses clearly, and verify "
        "the final answer with methodical review."
    ),
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Cambrian — Multi-objective evolution (Pareto / NSGA-II) demo")
    print("=" * 65)

    population = [make_agent(p) for p in SEED_PROMPTS]

    print(f"\nPopulation: {len(population)} agents")
    print("Objectives: quality (keyword coverage) vs brevity (shorter = better)\n")

    # Build objective vectors
    vectors = [
        ObjectiveVector(
            agent_id=a.id,
            scores={
                "quality": quality_score(a),
                "brevity": brevity_score(a),
            },
        )
        for a in population
    ]

    # Pareto-front analysis
    fronts = fast_non_dominated_sort(vectors)
    print(f"Non-dominated fronts: {len(fronts)}")
    for i, front in enumerate(fronts):
        print(f"  Front {i}: {len(front)} agent(s)")

    # Compute crowding distances (in-place)
    for front in fronts:
        crowding_distance(front)

    # Pareto front tracker
    pf = ParetoFront(objectives=["quality", "brevity"])
    for vec in vectors:
        pf.add(vec)

    print(f"\nPareto front size: {pf.size()}")
    front_agents = pf.agents(population)
    print("Pareto-optimal agents:")
    for a in front_agents:
        q = quality_score(a)
        b = brevity_score(a)
        tok = len(a.genome.system_prompt.split())
        print(f"  quality={q:.2f}  brevity={b:.2f}  words={tok:3d}  |  {a.genome.system_prompt[:60]!r}")

    # Attach diversity bonuses
    attach_diversity_scores(population, vectors, objective_name="diversity", k=3)
    print("\nDiversity scores attached to vectors.")

    # NSGA-II selection — keep best 4
    selected = nsga2_select(population, vectors, target_size=4)
    print(f"\nNSGA-II selected {len(selected)} survivors:")
    for a in selected:
        q = quality_score(a)
        b = brevity_score(a)
        print(f"  quality={q:.2f}  brevity={b:.2f}  |  {a.genome.system_prompt[:60]!r}")

    print("\n" + "=" * 65)
    print("Multi-objective demo complete.")
    print("In production: wrap this logic inside an EvolutionEngine loop")
    print("to evolve agents across generations on the Pareto front.")
    print("=" * 65)


if __name__ == "__main__":
    main()
