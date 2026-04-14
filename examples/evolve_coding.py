#!/usr/bin/env python3
"""Evolve an agent that solves multiple Python coding challenges.

This example demonstrates Cambrian on a harder problem than FizzBuzz:
the agent must simultaneously solve five diverse coding challenges, each
tested independently with expected stdout. A CompositeEvaluator aggregates
the five CodeEvaluators into one fitness signal.

Challenges
----------
1. FizzBuzz (1–20)
2. Fibonacci sequence (first 10 numbers, space-separated)
3. Palindrome checker (prints True/False for "racecar")
4. Bubble sort (sorts [5, 3, 8, 1, 9] and prints result)
5. Prime sieve (all primes up to 30)

Usage
-----
    export OPENAI_API_KEY=sk-...
    python examples/evolve_coding.py

    # Fewer generations for a quick test:
    python examples/evolve_coding.py --generations 3 --population 4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluator import Evaluator
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evaluators.composite import CompositeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── Challenge definitions ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Challenge:
    """One coding challenge with its expected output and task description."""

    name: str
    task: str
    expected_output: str
    weight: float = 1.0


def _fizzbuzz_output() -> str:
    out = []
    for n in range(1, 21):
        if n % 15 == 0:
            out.append("FizzBuzz")
        elif n % 3 == 0:
            out.append("Fizz")
        elif n % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(n))
    return "\n".join(out)


_CHALLENGES: list[Challenge] = [
    Challenge(
        name="fizzbuzz",
        task=(
            "Write a Python program that prints FizzBuzz for numbers 1 to 20. "
            "Print 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
            "'FizzBuzz' for both. One value per line."
        ),
        expected_output=_fizzbuzz_output(),
        weight=1.5,
    ),
    Challenge(
        name="fibonacci",
        task=(
            "Write a Python program that prints the first 10 Fibonacci numbers "
            "separated by spaces on a single line. "
            "The sequence starts: 0 1 1 2 3 5 ..."
        ),
        expected_output="0 1 1 2 3 5 8 13 21 34",
        weight=1.0,
    ),
    Challenge(
        name="palindrome",
        task=(
            "Write a Python program that checks whether the string 'racecar' is a "
            "palindrome and prints exactly True or False."
        ),
        expected_output="True",
        weight=0.8,
    ),
    Challenge(
        name="bubble_sort",
        task=(
            "Write a Python program that sorts the list [5, 3, 8, 1, 9] using "
            "bubble sort (implement the algorithm yourself, do not use sorted()) "
            "and prints the sorted list."
        ),
        expected_output="[1, 3, 5, 8, 9]",
        weight=1.2,
    ),
    Challenge(
        name="primes",
        task=(
            "Write a Python program that prints all prime numbers up to 30, "
            "space-separated on one line."
        ),
        expected_output="2 3 5 7 11 13 17 19 23 29",
        weight=1.0,
    ),
]


# ── Multi-challenge evaluator ─────────────────────────────────────────────────

class MultiChallengeEvaluator(Evaluator):
    """Evaluates an agent across all :data:`_CHALLENGES`.

    Each challenge uses its own :class:`~cambrian.evaluators.code.CodeEvaluator`
    sub-evaluator. Results are combined via :class:`~cambrian.evaluators.composite.CompositeEvaluator`
    with per-challenge weights.

    Args:
        timeout: Seconds per sandbox execution. Default ``8.0``.
    """

    def __init__(self, timeout: float = 8.0) -> None:
        sub_evals: list[Evaluator] = []
        weights: list[float] = []
        self._challenges = _CHALLENGES

        for ch in _CHALLENGES:
            sub_evals.append(
                CodeEvaluator(expected_output=ch.expected_output, timeout=timeout)
            )
            weights.append(ch.weight)

        self._composite = CompositeEvaluator(
            evaluators=sub_evals,
            weights=weights,
            aggregate="mean",
        )

    def evaluate(self, agent: object, task: str) -> float:
        """Score *agent* across all five coding challenges.

        Args:
            agent: The agent to score.
            task: The overall task description (used for logging only).

        Returns:
            Weighted mean fitness across all challenges, in ``[0.0, 1.0]``.
        """
        from cambrian.agent import Agent as _Agent
        assert isinstance(agent, _Agent)

        per_challenge: list[float] = []
        for ch, sub in zip(self._challenges, self._composite._evaluators):
            score = sub.evaluate(agent, ch.task)
            per_challenge.append(score)
            logger.debug("  challenge=%s score=%.3f", ch.name, score)

        composite_score = sum(
            s * w for s, w in zip(per_challenge, [c.weight for c in self._challenges])
        ) / sum(c.weight for c in self._challenges)
        return composite_score


_MASTER_TASK = (
    "You are an expert Python programmer. When given a coding task, output ONLY "
    "a complete, runnable Python script wrapped in ```python ... ``` fences. "
    "No explanations. No imports other than what is needed. "
    "The script must produce exactly the expected output when run."
)

_SEED_PROMPT = (
    "You are an expert Python programmer. "
    "Respond with ONLY a complete, executable Python script in a ```python``` block. "
    "Do not explain. Do not add comments. Produce exactly the requested output."
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Run multi-challenge coding evolution."""
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
    evaluator = MultiChallengeEvaluator(timeout=8.0)

    engine = EvolutionEngine(
        evaluator=evaluator,
        mutator=mutator,
        backend=backend,
        population_size=args.population,
        mutation_rate=0.8,
        crossover_rate=0.25,
        elite_ratio=0.25,
        tournament_k=3,
        seed=args.seed,
    )

    seed_genome = Genome(
        system_prompt=_SEED_PROMPT,
        strategy="chain-of-thought",
        temperature=0.2,   # low temperature → more deterministic code
        model=args.model,
    )

    print(f"\nCambrian — Multi-Challenge Coding Benchmark")
    print(f"Model: {args.model}  |  Challenges: {len(_CHALLENGES)}")
    print(f"Generations: {args.generations}  |  Population: {args.population}")
    print()
    for ch in _CHALLENGES:
        print(f"  [{ch.name}] weight={ch.weight}")
    print()

    challenge_best: dict[str, float] = {c.name: 0.0 for c in _CHALLENGES}

    def _on_gen(gen: int, population: list) -> None:
        scores = [a.fitness or 0.0 for a in population]
        best = max(scores)
        mean = sum(scores) / len(scores)
        print(f"  Gen {gen:2d}  best={best:.4f}  mean={mean:.4f}")

    best = engine.evolve(
        seed_genomes=[seed_genome],
        task=_MASTER_TASK,
        n_generations=args.generations,
        on_generation=_on_gen,
    )

    print("\n" + "=" * 64)
    print(f"Evolution complete!")
    print(f"Best composite fitness : {best.fitness:.4f}")
    print(f"Model                  : {best.genome.model}")
    print(f"Temperature            : {best.genome.temperature:.2f}")
    print(f"\nWinning system prompt:\n{best.genome.system_prompt}")
    print("=" * 64)

    if args.output:
        import json
        from pathlib import Path
        Path(args.output).write_text(json.dumps(best.genome.to_dict(), indent=2))
        print(f"\nBest genome saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-challenge coding evolution")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAMBRIAN_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
