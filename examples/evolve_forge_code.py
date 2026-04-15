"""Forge mode example — evolve Python code to solve a numeric task.

This example shows how to use :class:`~cambrian.code_genome.CodeEvolutionEngine`
to evolve a Python script that sums two integers from stdin.

Usage::

    OPENAI_API_KEY=... python examples/evolve_forge_code.py
"""

from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.code_genome import CodeEvaluator, CodeEvolutionEngine, CodeGenome, TestCase

backend = OpenAICompatBackend(model="gpt-4o-mini")

# Define test cases: input → expected stdout
test_cases = [
    TestCase(input_data="3 4", expected_output="7", label="3+4=7"),
    TestCase(input_data="10 20", expected_output="30", label="10+20=30"),
    TestCase(input_data="-1 1", expected_output="0", label="-1+1=0"),
    TestCase(input_data="0 0", expected_output="0", label="0+0=0"),
]

evaluator = CodeEvaluator(test_cases, timeout=5.0)

engine = CodeEvolutionEngine(
    backend=backend,
    evaluator=evaluator,
    population_size=6,
    elite_ratio=0.33,
)

seed = CodeGenome(
    description="Read two integers from stdin (space-separated) and print their sum."
)

print("Starting Forge evolution…")

best = engine.evolve(
    seed,
    task="Read two integers from stdin and print their sum",
    n_generations=8,
    on_generation=lambda gen, results, b: print(
        f"  gen={gen}  best_fitness={results[0][1].fitness:.4f}  "
        f"pass={results[0][1].passed}/{results[0][1].total}  "
        f"loc={results[0][0].loc()}"
    ),
)

print("\n=== BEST CODE ===")
print(best.code)
print(f"\nVersion: {best.version}  LOC: {best.loc()}")
