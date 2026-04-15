"""Dream Phase example — memory consolidation to maintain diversity.

This example shows how to integrate :class:`~cambrian.dream.DreamPhase` into
an evolution loop.  After each generation, past experiences are recombined via
LLM into synthetic hybrid scenarios.  Agents evaluated on dreams get a blended
fitness signal that rewards generalisation over specialisation.

Usage::

    OPENAI_API_KEY=... python examples/evolve_with_dream.py
"""

from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.dream import DreamPhase, Experience
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator

# Setup
backend = OpenAICompatBackend(model="gpt-4o-mini")
mutator = LLMMutator(backend=backend)
evaluator = LLMJudgeEvaluator(judge_backend=backend)

# Dream phase: blends 4 experiences into n_dreams=3 synthetic scenarios
# blend_weight=0.3 means 30% of fitness comes from dream evaluation
dream = DreamPhase(backend=backend, n_experiences=4, n_dreams=3, blend_weight=0.3)

task = "Explain a complex concept using a simple analogy"
N_GENERATIONS = 10
POPULATION_SIZE = 6

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
    Genome(system_prompt="You are a great teacher who uses clear, memorable analogies."),
    Genome(system_prompt="You explain things step by step, building on simple foundations."),
]

# Accumulate experiences across generations
experiences: list[Experience] = []

print("Starting evolution with Dream Phase…")
print(f"Task: {task}\n")


def _on_generation(gen: int, population: list) -> None:
    global experiences

    best_fitness = max(a.fitness or 0.0 for a in population)
    print(f"  gen={gen:3d}  best={best_fitness:.4f}  dream_pool={len(experiences)}")

    # Collect new experiences from this generation's top agents
    for agent in sorted(population, key=lambda a: a.fitness or 0.0, reverse=True)[:3]:
        if agent.fitness and agent.fitness > 0.5:
            response = agent.run(task)
            experiences.append(
                Experience(
                    task=task,
                    response=response,
                    score=agent.fitness,
                    genome_id=agent.agent_id,
                )
            )

    # Keep the experience pool manageable
    experiences = experiences[-20:]

    # Apply dream phase if we have enough experiences
    if len(experiences) >= 4:
        dream.run(population, task, evaluator, experiences=experiences)
        new_best = max(a.fitness or 0.0 for a in population)
        if new_best != best_fitness:
            print(f"           dream adjusted best: {best_fitness:.4f} → {new_best:.4f}")


best = engine.evolve(
    seed_genomes=seed_genomes,
    task=task,
    n_generations=N_GENERATIONS,
    on_generation=_on_generation,
)

print("\n=== BEST AGENT ===")
print(f"Fitness: {best.fitness:.4f}")
print(f"System prompt: {best.genome.system_prompt}")
print(f"Strategy: {best.genome.strategy}")
print(f"\nTotal experiences collected: {len(experiences)}")
