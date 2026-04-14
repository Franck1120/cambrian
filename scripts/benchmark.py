#!/usr/bin/env python3
"""Benchmark Cambrian's per-generation throughput without real API calls.

Measures the wall-clock time of the evolutionary loop's non-LLM components:
selection, population management, MAP-Elites archiving, and memory writes.
All LLM calls (mutator, scorer) are replaced with no-op stubs.

Output: a table of per-generation timings and a summary report.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --population 20 --generations 15 --runs 3
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator


# -- Stubs ----------------------------------------------------------------------

class _NullBackend(LLMBackend):
    """Returns the input genome JSON unchanged (zero latency)."""

    model_name = "null"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import re
        m = re.search(r"\{[\s\S]+\}", prompt)
        return m.group(0) if m else "{}"


def _jitter_scorer(base_score: float = 0.5):
    """Return a fitness-scoring callable that adds tiny random jitter.

    Used to exercise MAP-Elites and selection pressure without real LLM calls.
    """

    def _score(agent: Agent, task: str) -> float:
        return min(1.0, max(0.0, base_score + random.uniform(-0.05, 0.05)))

    return _score


# -- Benchmark run --------------------------------------------------------------

def _run_single(population_size: int, n_generations: int) -> dict[str, list[float]]:
    """Execute one full evolution run and collect per-generation timing.

    Args:
        population_size: Number of agents per generation.
        n_generations: Total generations to run.

    Returns:
        Dict with ``"gen_times"`` (seconds per generation) and
        ``"total"`` (total wall-clock seconds).
    """
    backend = _NullBackend()
    mutator = LLMMutator(backend=backend, fallback_on_error=True)
    scorer = _jitter_scorer(base_score=0.6)

    engine = EvolutionEngine(
        evaluator=scorer,
        mutator=mutator,
        backend=backend,
        population_size=population_size,
        mutation_rate=0.8,
        crossover_rate=0.3,
        elite_ratio=0.2,
        tournament_k=3,
        seed=42,
    )

    seed = Genome(
        system_prompt="You are a Python expert. Output only code.",
        temperature=0.7,
    )

    gen_times: list[float] = []
    _state: dict[str, float] = {"t_start": time.monotonic()}

    def _cb(gen: int, pop: list[Agent]) -> None:
        if gen > 0:
            gen_times.append(time.monotonic() - _state["t_start"])
        _state["t_start"] = time.monotonic()

    t0 = time.monotonic()
    engine.evolve(
        seed_genomes=[seed],
        task="benchmark",
        n_generations=n_generations,
        on_generation=_cb,
    )
    total = time.monotonic() - t0

    return {"gen_times": gen_times, "total": [total]}


def _fmt_ms(seconds: float) -> str:
    """Format *seconds* as a milliseconds string."""
    return f"{seconds * 1000:.1f} ms"


# -- Main -----------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run the benchmark and print a summary report.

    Args:
        args: Parsed CLI arguments (population, generations, runs, histogram).
    """
    print(f"\nCambrian Benchmark (no LLM calls)")
    print(f"  Population : {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Runs       : {args.runs}")
    print()

    all_gen_times: list[float] = []
    all_totals: list[float] = []

    for run in range(1, args.runs + 1):
        result = _run_single(args.population, args.generations)
        all_gen_times.extend(result["gen_times"])
        all_totals.extend(result["total"])
        gen_mean = statistics.mean(result["gen_times"]) if result["gen_times"] else 0.0
        print(f"  Run {run}: total={_fmt_ms(result['total'][0])}  gen_mean={_fmt_ms(gen_mean)}")

    if not all_gen_times:
        print("No generation timing data collected.")
        return

    print()
    print("-" * 48)
    print(f"  Per-generation (ms):")
    print(f"    min    = {_fmt_ms(min(all_gen_times))}")
    print(f"    mean   = {_fmt_ms(statistics.mean(all_gen_times))}")
    print(f"    median = {_fmt_ms(statistics.median(all_gen_times))}")
    print(f"    max    = {_fmt_ms(max(all_gen_times))}")
    if len(all_gen_times) > 1:
        print(f"    stdev  = {_fmt_ms(statistics.stdev(all_gen_times))}")
    print()
    print(f"  Total run time:")
    print(f"    mean   = {_fmt_ms(statistics.mean(all_totals))}")
    print()

    # Throughput
    mean_gen_s = statistics.mean(all_gen_times)
    if mean_gen_s > 0:
        throughput = args.population / mean_gen_s
        print(f"  Throughput: {throughput:.0f} agents/second (excluding LLM latency)")
    print("-" * 48)

    # Optional histogram
    if args.histogram and all_gen_times:
        print()
        buckets = 10
        lo, hi = min(all_gen_times), max(all_gen_times)
        width = (hi - lo) / buckets if hi > lo else 1e-9
        hist = [0] * buckets
        for t in all_gen_times:
            idx = min(int((t - lo) / width), buckets - 1)
            hist[idx] += 1
        bar_max = max(hist)
        print("  Latency distribution:")
        for i, count in enumerate(hist):
            bucket_lo = lo + i * width
            bar = "#" * int(count / bar_max * 20) if bar_max else ""
            print(f"  {_fmt_ms(bucket_lo):>10}  {bar} {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cambrian benchmark (no API calls)")
    parser.add_argument("-p", "--population", type=int, default=10,
                        help="Population size per generation")
    parser.add_argument("-g", "--generations", type=int, default=8,
                        help="Number of generations per run")
    parser.add_argument("-r", "--runs", type=int, default=3,
                        help="Number of independent benchmark runs")
    parser.add_argument("--histogram", action="store_true",
                        help="Print latency distribution histogram")
    main(parser.parse_args())
