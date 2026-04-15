# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Cambrian Performance Benchmark.

Measures wall-clock time, simulated token throughput, and peak memory usage
for the core evolution loop at various scales. Uses a mock backend — no API
key required.

Run with:
    python examples/benchmark.py

Output example:
    [benchmark] 100 gen × 20 agents
    Elapsed        : 1.234 s
    Throughput     : 1620 evals/s
    Peak memory    : 12.3 MB
    Simulated toks : 200000
    Best fitness   : 0.820
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

_KEYWORDS = ["expert", "step-by-step", "systematic", "analytical", "verify",
             "precise", "structured", "critical", "thorough", "methodical"]

_BEST_PROMPT = " ".join(_KEYWORDS)

_GENOME_TEMPLATE = {
    "strategy": "step-by-step",
    "temperature": 0.7,
    "model": "gpt-4o-mini",
    "tools": [],
    "few_shot_examples": [],
}

# Track simulated token usage across mock calls
_simulated_tokens: list[int] = []


def _mock_backend(tokens_per_call: int = 200) -> MagicMock:
    """Return a mock backend that records token usage."""
    b = MagicMock()

    def _generate(prompt: str, **kwargs: Any) -> str:
        _simulated_tokens.append(tokens_per_call)
        return json.dumps({**_GENOME_TEMPLATE, "system_prompt": _BEST_PROMPT})

    b.generate = MagicMock(side_effect=_generate)
    return b


class _KeywordEvaluator(Evaluator):
    """Score by keyword presence — O(n) scan, no I/O."""

    def evaluate(self, agent: Agent, task: str) -> float:  # noqa: ARG002
        text = agent.genome.system_prompt.lower()
        hits = sum(1 for kw in _KEYWORDS if kw in text)
        return min(1.0, hits * 0.1)


def _engine(pop_size: int) -> EvolutionEngine:
    backend = _mock_backend()
    return EvolutionEngine(
        evaluator=_KeywordEvaluator(),
        mutator=LLMMutator(backend=backend),
        backend=backend,
        population_size=pop_size,
        mutation_rate=1.0,
        crossover_rate=0.0,
        elite_ratio=0.2,
        tournament_k=3,
    )


def _seeds(n: int) -> list[Genome]:
    return [Genome(system_prompt=f"agent {i} base prompt") for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    label: str
    pop_size: int
    n_generations: int
    elapsed_s: float
    throughput_evals_per_s: float
    peak_memory_mb: float
    simulated_tokens: int
    best_fitness: float


def run_benchmark(pop_size: int, n_generations: int, label: str) -> BenchmarkResult:
    """Run a single benchmark scenario and return results."""
    _simulated_tokens.clear()

    engine = _engine(pop_size)
    seeds = _seeds(pop_size)

    tracemalloc.start()
    t0 = time.perf_counter()

    best = engine.evolve(seed_genomes=seeds, task="benchmark task", n_generations=n_generations)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_evals = pop_size * n_generations
    throughput = total_evals / elapsed if elapsed > 0 else float("inf")
    peak_mb = peak_bytes / (1024 * 1024)
    total_tokens = sum(_simulated_tokens)

    return BenchmarkResult(
        label=label,
        pop_size=pop_size,
        n_generations=n_generations,
        elapsed_s=elapsed,
        throughput_evals_per_s=throughput,
        peak_memory_mb=peak_mb,
        simulated_tokens=total_tokens,
        best_fitness=best.fitness or 0.0,
    )


def print_result(r: BenchmarkResult) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {r.label}")
    print(sep)
    print(f"  Config         : {r.pop_size} agents × {r.n_generations} generations")
    print(f"  Elapsed        : {r.elapsed_s:.3f} s")
    print(f"  Throughput     : {r.throughput_evals_per_s:,.0f} evals/s")
    print(f"  Peak memory    : {r.peak_memory_mb:.1f} MB")
    print(f"  Simulated toks : {r.simulated_tokens:,}")
    print(f"  Best fitness   : {r.best_fitness:.4f}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    (10,  10,  "Small  — 10 agents × 10 gen"),
    (20,  20,  "Medium — 20 agents × 20 gen"),
    (20, 100,  "Large  — 20 agents × 100 gen"),
    (50, 100,  "XLarge — 50 agents × 100 gen"),
]


def main() -> None:
    print("=" * 52)
    print("  Cambrian Performance Benchmark")
    print("  (mock backend — no API key required)")
    print("=" * 52)

    results: list[BenchmarkResult] = []
    for pop_size, n_gen, label in SCENARIOS:
        print(f"\nRunning: {label} ...", end=" ", flush=True)
        r = run_benchmark(pop_size, n_gen, label)
        results.append(r)
        print("done.")

    print("\n\n" + "=" * 52)
    print("  RESULTS SUMMARY")
    print("=" * 52)
    print(f"  {'Scenario':<32}  {'Time':>7}  {'Evals/s':>9}  {'Mem':>7}  {'Fitness':>7}")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r.label:<32}  {r.elapsed_s:>6.2f}s"
            f"  {r.throughput_evals_per_s:>8,.0f}"
            f"  {r.peak_memory_mb:>6.1f}MB"
            f"  {r.best_fitness:>7.4f}"
        )

    # Assertions — fail loudly if performance regresses severely
    for r in results:
        # 20×100 should finish in under 30s even on slow CI
        if r.n_generations >= 100 and r.pop_size <= 20:
            assert r.elapsed_s < 30.0, (
                f"{r.label}: took {r.elapsed_s:.2f}s (limit 30s)"
            )
        assert r.throughput_evals_per_s > 50, (
            f"{r.label}: throughput {r.throughput_evals_per_s:.0f} evals/s (min 50)"
        )

    print("\n  All performance assertions passed.")
    print("=" * 52)


if __name__ == "__main__":
    main()
