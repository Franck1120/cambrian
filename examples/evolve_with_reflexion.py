"""Reflexion example — generate → critique → revise loop for improved responses.

This example shows two ways to use Reflexion:
1. :class:`~cambrian.reflexion.ReflexionAgent` as a standalone improver.
2. :class:`~cambrian.reflexion.ReflexionEvaluator` as a drop-in evaluator
   that automatically improves responses before scoring.

Usage::

    OPENAI_API_KEY=... python examples/evolve_with_reflexion.py
"""

from cambrian.agent import Agent, Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.reflexion import ReflexionAgent, ReflexionEvaluator

backend = OpenAICompatBackend(model="gpt-4o-mini")

# ── Part 1: Standalone ReflexionAgent ──────────────────────────────────────────

print("=== Part 1: Standalone ReflexionAgent ===\n")

agent = Agent(
    genome=Genome(
        system_prompt="You are a helpful assistant.",
        strategy="step-by-step",
    ),
    backend=backend,
)

reflexion = ReflexionAgent(
    agent=agent,
    n_rounds=2,
    critique_temperature=0.3,
    revision_temperature=0.5,
    stop_if_excellent=True,
)

task = "Explain why the sky is blue in 2-3 sentences."
result = reflexion.run(task)

print(f"Task: {task}\n")
print(f"Initial response (round 0):")
print(f"  {result.initial_response[:200]}")
print(f"\nFinal response (after {result.n_rounds_used} round(s)):")
print(f"  {result.final_response[:200]}")
print(f"\nImproved: {result.improved}")
for r in result.rounds:
    if r.critique:
        print(f"  Round {r.round_number} critique: {r.critique[:100]}…")

# ── Part 2: ReflexionEvaluator in an evolution loop ──────────────────────────

print("\n\n=== Part 2: Reflexion-wrapped Evolution ===\n")

base_evaluator = LLMJudgeEvaluator(judge_backend=backend)

# ReflexionEvaluator improves responses before scoring
reflexion_evaluator = ReflexionEvaluator(
    base_evaluator=base_evaluator,
    n_rounds=2,
    reflection_backend=backend,
)

mutator = LLMMutator(backend=backend)

engine = EvolutionEngine(
    evaluator=reflexion_evaluator,
    mutator=mutator,
    backend=backend,
    population_size=4,
    mutation_rate=0.8,
    elite_ratio=0.25,
    seed=42,
)

seed_genomes = [
    Genome(system_prompt="You are a science educator who explains clearly."),
]

evolution_task = "Explain how photosynthesis works in simple terms."
print(f"Task: {evolution_task}\n")

best = engine.evolve(
    seed_genomes=seed_genomes,
    task=evolution_task,
    n_generations=5,
    on_generation=lambda gen, pop: print(
        f"  gen={gen}  best={max(a.fitness or 0.0 for a in pop):.4f}"
    ),
)

print(f"\n=== BEST AGENT ===")
print(f"Fitness: {best.fitness:.4f}")
print(f"System prompt: {best.genome.system_prompt}")
