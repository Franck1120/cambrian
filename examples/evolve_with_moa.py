"""Mixture of Agents example — ensemble synthesis for higher-quality answers.

This example shows how to use :class:`~cambrian.moa.MixtureOfAgents` to run
multiple evolved agents in parallel and synthesise their answers.  Optionally,
:class:`~cambrian.moa.QuantumTunneler` is applied between generations to escape
local optima by replacing low-fitness agents with random genomes.

Usage::

    OPENAI_API_KEY=... python examples/evolve_with_moa.py
"""

from cambrian.agent import Agent, Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.moa import MixtureOfAgents, QuantumTunneler
from cambrian.mutator import LLMMutator

# Setup
backend = OpenAICompatBackend(model="gpt-4o-mini")
mutator = LLMMutator(backend=backend)
evaluator = LLMJudgeEvaluator(judge_backend=backend)

# Quantum Tunneler: 20% chance per non-elite agent to be replaced each generation
tunneler = QuantumTunneler(tunnel_prob=0.2, protect_elites=True, n_elites=2, seed=42)

task = "What is the best strategy for learning a new programming language?"
N_GENERATIONS = 8
POPULATION_SIZE = 6

engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    backend=backend,
    population_size=POPULATION_SIZE,
    mutation_rate=0.7,
    elite_ratio=0.33,
    seed=42,
)

seed_genomes = [
    Genome(system_prompt="You are an expert software engineer and educator."),
    Genome(system_prompt="You are a pragmatic developer who learns by doing."),
    Genome(system_prompt="You are a systematic learner who builds mental models."),
]

print("Starting evolution with Mixture of Agents + Quantum Tunneling…")
print(f"Task: {task}\n")

final_population: list[Agent] = []


def _on_generation(gen: int, population: list) -> None:
    best = max(a.fitness or 0.0 for a in population)
    print(
        f"  gen={gen:3d}  best={best:.4f}  "
        f"tunnel_events={tunneler.tunnel_count}"
    )
    # Apply quantum tunneling to inject diversity
    tunneler.apply(population)
    # Track final population for MoA
    final_population.clear()
    final_population.extend(population)


best_agent = engine.evolve(
    seed_genomes=seed_genomes,
    task=task,
    n_generations=N_GENERATIONS,
    on_generation=_on_generation,
)

print("\n=== EVOLUTION COMPLETE ===")
print(f"Best agent fitness: {best_agent.fitness:.4f}")
print(f"Total tunnel events: {tunneler.tunnel_count}")
tunnel_summary = tunneler.summary()
print(f"Tunnel summary: {tunnel_summary}")

# Now run Mixture of Agents on the final evolved population
print("\n=== MIXTURE OF AGENTS ===")
print(f"Running {len(final_population)} agents on the task…\n")

moa = MixtureOfAgents.from_population(final_population, aggregator_backend=backend)
result = moa.run(task)

print(f"Individual answers ({result.n_agents} agents):")
for i, ans in enumerate(result.individual_answers):
    print(f"  Agent {i+1}: {ans[:120]}…" if len(ans) > 120 else f"  Agent {i+1}: {ans}")

print("\nSynthesised answer:")
print(result.final_answer)
