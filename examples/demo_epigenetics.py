# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Demo: EpigeneticLayer — runtime context injection without genome mutation.

Traditional genetic algorithms mutate the genome directly.  Epigenetics
injects context-sensitive annotations into the *expressed* prompt at runtime,
leaving the underlying genome unchanged.  This means:

- The annotation adapts to the current generation, task, and population state.
- Rolling back to the base genome is free — just remove the layer.
- Multiple layers can stack, each adding a different kind of context.

Typical epigenetic annotations
-------------------------------
- Phase markers: "You are in the early exploration phase."
- Population pressure: "Mean fitness is 0.60 — aim to exceed 0.85."
- Task focus: "Task: Write Python code." injected at generation 0.
- Custom rules: add your own lambda in seconds.

No API key required.

Usage
-----
    python examples/demo_epigenetics.py
"""

from __future__ import annotations

from cambrian.agent import Agent, Genome
from cambrian.epigenetics import (
    EpigenomicContext,
    EpigeneticLayer,
    make_standard_layer,
)


def show_expressed(
    label: str,
    genome: Genome,
    layer: EpigeneticLayer,
    ctx: EpigenomicContext,
) -> None:
    expressed = layer.express(genome, ctx)
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"{'-' * 60}")
    print(expressed)


def main() -> None:
    print("=" * 60)
    print("Cambrian — EpigeneticLayer demo")
    print("=" * 60)

    genome = Genome(system_prompt="You are an expert problem solver.")

    # ── Standard layer (built-in rules) ──────────────────────────────────────

    std = make_standard_layer()

    show_expressed(
        "Standard layer — generation 0 (early phase)",
        genome, std,
        EpigenomicContext(
            generation=0,
            task="Debug a Python script.",
            population_mean_fitness=0.3,
            population_best_fitness=0.5,
            total_generations=20,
        ),
    )

    show_expressed(
        "Standard layer — generation 10 (mid phase)",
        genome, std,
        EpigenomicContext(
            generation=10,
            task="Debug a Python script.",
            population_mean_fitness=0.65,
            population_best_fitness=0.82,
            total_generations=20,
        ),
    )

    show_expressed(
        "Standard layer — generation 19 (late phase)",
        genome, std,
        EpigenomicContext(
            generation=19,
            task="Debug a Python script.",
            population_mean_fitness=0.78,
            population_best_fitness=0.91,
            total_generations=20,
        ),
    )

    # ── Custom rules ──────────────────────────────────────────────────────────

    print("\n\n" + "=" * 60)
    print("Custom EpigeneticLayer with user-defined rules")
    print("=" * 60)

    custom = EpigeneticLayer(
        rules=[
            # Rule 1: always surface the task
            lambda g, ctx: f"[Task] {ctx.task}" if ctx.task else None,
            # Rule 2: high-competition warning
            lambda g, ctx: (
                "[Warning] Population has stagnated — try unconventional approaches."
                if ctx.population_best_fitness is not None
                and ctx.population_best_fitness < 0.5
                else None
            ),
            # Rule 3: late-stage refinement nudge
            lambda g, ctx: (
                "[Refinement] Focus on polishing correctness, not exploration."
                if ctx.is_late else None
            ),
        ],
        separator="\n\n<!-- epigenetics -->\n",
    )

    show_expressed(
        "Custom layer — early, low fitness (stagnation warning fires)",
        genome, custom,
        EpigenomicContext(
            generation=1,
            task="Solve the travelling salesman problem.",
            population_mean_fitness=0.2,
            population_best_fitness=0.4,
            total_generations=10,
        ),
    )

    show_expressed(
        "Custom layer — late, high fitness (refinement nudge fires)",
        genome, custom,
        EpigenomicContext(
            generation=9,
            task="Solve the travelling salesman problem.",
            population_mean_fitness=0.75,
            population_best_fitness=0.88,
            total_generations=10,
        ),
    )

    # ── apply() — creates a copy, never mutates the original ─────────────────

    print("\n\n" + "=" * 60)
    print("apply() — ephemeral agent copy, original genome unchanged")
    print("=" * 60)

    original_agent = Agent(genome=genome)
    ctx = EpigenomicContext(generation=5, task="Write a unit test.", total_generations=10)
    expressed_agent = std.apply(original_agent, ctx)

    print(f"\nOriginal prompt : {original_agent.genome.system_prompt!r}")
    print(f"Expressed prompt: {expressed_agent.genome.system_prompt[:80]!r}...")
    print(f"Same object?    : {original_agent is expressed_agent}")

    print("\n" + "=" * 60)
    print("EpigeneticLayer demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
