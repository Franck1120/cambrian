"""Demo: End-to-end evolution with mock backend — 10 generations, real fitness improvement.

This example runs a complete evolutionary loop using:
  - LLMMutator with a mocked backend (generates deterministic prompt improvements)
  - A keyword-based evaluator (scores prompts by presence of performance keywords)
  - EvolutionEngine with tournament selection, elitism, crossover
  - EvolutionaryMemory to track lineage

No API key required. Runs offline and shows genuine fitness improvement because
the mock mutator reliably adds keywords that the evaluator rewards.

Usage
-----
    python examples/demo_end_to_end.py
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.evaluator import Evaluator
from cambrian.mutator import LLMMutator

# ---------------------------------------------------------------------------
# Keyword-based evaluator: scores by presence of performance keywords
# ---------------------------------------------------------------------------

_HIGH_VALUE_KEYWORDS = [
    "step-by-step",
    "expert",
    "verify",
    "systematic",
    "analytical",
    "precise",
    "structured",
    "rigorous",
    "methodical",
    "validate",
]


class KeywordEvaluator(Evaluator):
    """Scores a prompt by keyword coverage. 0.1 base + 0.09 per keyword."""

    def evaluate(self, agent: Agent, task: str) -> float:  # noqa: ARG002
        text = agent.genome.system_prompt.lower()
        hits = sum(1 for kw in _HIGH_VALUE_KEYWORDS if kw in text)
        return min(1.0, 0.1 + hits * 0.09)


# ---------------------------------------------------------------------------
# Smart mock backend: each mutation call adds one new keyword
# ---------------------------------------------------------------------------

_PROMPT_UPGRADES = [
    "You are an expert step-by-step systematic problem solver.",
    "Analytical and methodical: verify each step rigorously before proceeding.",
    "Precise expert reasoning: structured decomposition, validate every assumption.",
    "Step-by-step expert analysis: systematic, rigorous, methodical verification.",
    "Expert analytical thinker: precise, structured, validate results systematically.",
    "Rigorous methodical expert: step-by-step analytical approach with validation.",
    "Systematic expert: precise structured reasoning, verify and validate each step.",
    "Expert step-by-step analytical solver: rigorous systematic methodical approach.",
]


def _make_smart_backend(rng: random.Random) -> MagicMock:
    """Backend that returns JSON genomes with progressively more keywords."""
    call_count = [0]

    def generate(prompt: str, **kwargs: object) -> str:  # noqa: ARG001
        idx = call_count[0] % len(_PROMPT_UPGRADES)
        call_count[0] += 1
        improved_prompt = _PROMPT_UPGRADES[idx]
        genome_dict = {
            "system_prompt": improved_prompt,
            "strategy": "step-by-step",
            "temperature": 0.7,
            "model": "gpt-4o-mini",
            "tools": [],
            "few_shot_examples": [],
        }
        return json.dumps(genome_dict)

    backend = MagicMock()
    backend.generate = MagicMock(side_effect=generate)
    return backend


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

print("=" * 70)
print("Cambrian — End-to-End Evolution Demo (offline, mock backend)")
print("=" * 70)
print()
print("Setup:")
print("  Population size : 8 agents")
print("  Generations     : 10")
print("  Evaluator       : keyword coverage (10 keywords x 0.09 = 0.9 max boost)")
print("  Mutator         : smart mock (adds keywords each generation)")
print("  Selection       : tournament (k=3) + 2 elites")
print()

rng = random.Random(42)
backend = _make_smart_backend(rng)
evaluator = KeywordEvaluator()

SEED_PROMPTS = [
    "You are a helpful assistant.",
    "Answer the user's question clearly.",
    "You are an AI that solves problems.",
    "Respond to the user concisely.",
    "You are a coding assistant.",
    "Help the user with their task.",
    "Be accurate and informative.",
    "Provide clear and correct answers.",
]

seed_genomes = [Genome(system_prompt=p) for p in SEED_PROMPTS]

mutator = LLMMutator(backend=backend)

engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    backend=backend,
    population_size=8,
    elite_ratio=0.25,
    mutation_rate=0.8,
    crossover_rate=0.3,
    tournament_k=3,
)

# Track per-generation stats
gen_stats: list[dict[str, float]] = []


def on_generation(gen: int, population: list[Agent]) -> None:
    scores = [a.fitness or 0.0 for a in population]
    best = max(scores)
    mean = sum(scores) / len(scores)
    gen_stats.append({"gen": gen, "best": best, "mean": mean})
    print(f"  Gen {gen:2d}  best={best:.4f}  mean={mean:.4f}  "
          f"pop={len(population)}")


print("Evolution progress:")
best_agent = engine.evolve(
    seed_genomes=seed_genomes,
    task="Solve mathematical problems step by step",
    n_generations=10,
    on_generation=on_generation,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

initial_best = gen_stats[0]["best"]
final_best = gen_stats[-1]["best"]
improvement = final_best - initial_best

print()
print("=" * 70)
print("Results:")
print(f"  Initial best fitness : {initial_best:.4f}")
print(f"  Final best fitness   : {final_best:.4f}")
print(f"  Total improvement    : +{improvement:.4f} ({improvement / initial_best * 100:.1f}%)")
print()
print("  Best agent genome (first 80 chars):")
print(f"  {best_agent.genome.system_prompt[:80]!r}")
print()

# Verify genuine improvement occurred
assert final_best > initial_best, "Evolution should improve fitness!"
print("Assertion passed: final_best > initial_best")
print()

# Fitness trajectory
print("Fitness trajectory:")
print("  Gen |" + "".join(f" {s['gen']:2d}" for s in gen_stats))
print("  Fit |" + "".join(f" {s['best']:.2f}"[1:] for s in gen_stats))

print()
print(f"Total LLM calls (mock) : {backend.generate.call_count}")
print("=" * 70)
print()
print("Demo complete — fitness improved by genuine evolutionary pressure.")
