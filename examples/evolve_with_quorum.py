"""Quorum Sensing example — auto-adjust mutation rate based on diversity.

This example shows how to integrate :class:`~cambrian.quorum.QuorumSensor`
into an evolution loop to auto-regulate mutation rate and elitism.

Usage::

    OPENAI_API_KEY=... python examples/evolve_with_quorum.py
"""

from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.quorum import QuorumSensor

# Setup
backend = OpenAICompatBackend(model="gpt-4o-mini")
mutator = LLMMutator(backend=backend)
evaluator = LLMJudgeEvaluator(judge_backend=backend)

# Quorum sensor monitors diversity, target_entropy=0.6
sensor = QuorumSensor(n_bins=8, target_entropy=0.6, lr=0.05)

task = "Write a haiku about the autumn wind"
N_GENERATIONS = 10
POPULATION_SIZE = 8

# Build engine with standard settings
engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    backend=backend,
    population_size=POPULATION_SIZE,
    mutation_rate=0.8,
    elite_ratio=0.2,
    seed=42,
)

seed_genomes = [
    Genome(system_prompt="You are a creative haiku poet.", strategy="creative"),
]

print("Starting evolution with Quorum Sensing…")
print(f"Task: {task}\n")


def _on_generation(gen: int, population: list) -> None:
    # Update quorum sensor
    state = sensor.update(population)
    print(
        f"  gen={gen:3d}  "
        f"best={max(a.fitness or 0.0 for a in population):.4f}  "
        f"entropy={state.entropy:.4f}  "
        f"mut_rate={state.mutation_rate:.3f}  "
        f"elite_n={state.elite_n}"
    )
    # Apply quorum-adjusted parameters back to engine
    engine._mut_rate = state.mutation_rate
    engine._elite_n = state.elite_n


best = engine.evolve(
    seed_genomes=seed_genomes,
    task=task,
    n_generations=N_GENERATIONS,
    on_generation=_on_generation,
)

print("\n=== BEST AGENT ===")
print(f"Fitness: {best.fitness:.4f}")
print(f"System prompt: {best.genome.system_prompt}")
print("\nQuorum Summary:")
summary = sensor.summary()
for k, v in summary.items():
    print(f"  {k}: {v}")
