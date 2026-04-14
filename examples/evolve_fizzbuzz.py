#!/usr/bin/env python3
"""Evolve an agent that can write correct FizzBuzz code.

This example demonstrates the full Cambrian pipeline:

1. A seed genome with a generic problem-solving system prompt.
2. A CodeEvaluator that runs the agent's code output in a sandbox and checks
   for correct FizzBuzz output (1–30 range).
3. An LLMMutator that refines system prompts toward Python expertise.
4. EvolutionEngine running 5 generations with population 6.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python examples/evolve_fizzbuzz.py

    # Use a local Ollama endpoint instead:
    CAMBRIAN_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \\
        python examples/evolve_fizzbuzz.py --model llama3.2
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the repo root is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger, log_generation_summary

logger = get_logger(__name__)

# ── FizzBuzz reference output (1–30) ─────────────────────────────────────────

_EXPECTED_LINES = []
for n in range(1, 31):
    if n % 15 == 0:
        _EXPECTED_LINES.append("FizzBuzz")
    elif n % 3 == 0:
        _EXPECTED_LINES.append("Fizz")
    elif n % 5 == 0:
        _EXPECTED_LINES.append("Buzz")
    else:
        _EXPECTED_LINES.append(str(n))
_EXPECTED_OUTPUT = "\n".join(_EXPECTED_LINES)

_TASK = (
    "Write a Python program that prints FizzBuzz for the numbers 1 through 30. "
    "Rules: print 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
    "'FizzBuzz' for multiples of both, and the number itself otherwise. "
    "Print exactly one value per line, nothing else."
)

_SEED_PROMPT = (
    "You are an expert Python programmer. When given a coding task, you respond "
    "with a complete, runnable Python script — no explanation, no markdown, just code. "
    "Your code is always correct, clean, and produces exactly the requested output."
)


# ── Custom evaluator: checks FizzBuzz correctness ────────────────────────────

class FizzBuzzEvaluator(CodeEvaluator):
    """Extends CodeEvaluator with FizzBuzz-specific output checking."""

    def _score_output(self, stdout: str, stderr: str, returncode: int) -> float:
        if returncode != 0 or not stdout.strip():
            return 0.1 if stderr else 0.0

        actual_lines = [l.strip() for l in stdout.strip().splitlines()]
        expected_lines = _EXPECTED_OUTPUT.splitlines()

        if actual_lines == expected_lines:
            return 1.0

        # Partial credit: count matching lines
        matches = sum(1 for a, e in zip(actual_lines, expected_lines) if a == e)
        return matches / len(expected_lines) * 0.9


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    backend = OpenAICompatBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=60,
        max_retries=2,
    )

    mutator = LLMMutator(backend=backend, mutation_temperature=0.5)
    fb_eval = FizzBuzzEvaluator(timeout_seconds=5, expected_output=_EXPECTED_OUTPUT)

    engine = EvolutionEngine(
        evaluator=fb_eval,
        mutator=mutator,
        backend=backend,
        population_size=args.population,
        mutation_rate=0.8,
        crossover_rate=0.3,
        elite_ratio=0.2,
        seed=args.seed,
    )

    seed_genome = Genome(
        system_prompt=_SEED_PROMPT,
        tools=[],
        strategy="chain-of-thought",
        temperature=0.3,
        model=args.model,
    )

    generation_log: list[tuple[int, float, float]] = []

    def _on_gen(gen: int, population: list) -> None:
        scores = [a.fitness for a in population if a.fitness is not None]
        best = max(scores, default=0.0)
        mean = sum(scores) / len(scores) if scores else 0.0
        generation_log.append((gen, best, mean))
        print(f"  Gen {gen:2d}  best={best:.4f}  mean={mean:.4f}")

    print(f"\nCambrian FizzBuzz Example")
    print(f"Model: {args.model}  |  Population: {args.population}  |  Generations: {args.generations}")
    print(f"Task: {_TASK[:80]}...")
    print()

    best = engine.evolve(
        seed_genomes=[seed_genome],
        task=_TASK,
        n_generations=args.generations,
        on_generation=_on_gen,
    )

    print("\n" + "=" * 60)
    print(f"Evolution complete!")
    print(f"Best fitness : {best.fitness:.4f}")
    print(f"Model        : {best.genome.model}")
    print(f"Temperature  : {best.genome.temperature:.2f}")
    print(f"Strategy     : {best.genome.strategy}")
    print(f"\nWinning system prompt:\n{best.genome.system_prompt}")
    print("=" * 60)

    if args.output:
        import json
        from pathlib import Path
        out = Path(args.output)
        out.write_text(json.dumps(best.genome.to_dict(), indent=2))
        print(f"\nBest genome saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve a FizzBuzz-solving agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAMBRIAN_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--api-key", default=None, help="API key (falls back to OPENAI_API_KEY)")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population", type=int, default=6, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Path to save best genome JSON")
    main(parser.parse_args())
