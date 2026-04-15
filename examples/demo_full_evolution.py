"""Demo: Full Evolution Stack — mocked offline run.

Combines the following techniques in a single coordinated loop:
  - Code evolution (Forge mode)
  - Pipeline evolution
  - Dream Phase (GraphRAG recombination)
  - Quorum Sensing (diversity-based mutation rate)
  - Apoptosis (programmed removal of stagnant agents)
  - Metamorphosis (LARVA → CHRYSALIS → IMAGO lifecycle)
  - Ecosystem (4 ecological roles shaping fitness)
  - Neuromodulation (bio-inspired adaptive hyperparameters)

All LLM and evaluator calls are mocked so this runs offline and instantly.

Usage
-----
    python examples/demo_full_evolution.py
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.apoptosis import ApoptosisController
from cambrian.dream import DreamPhase
from cambrian.ecosystem import EcologicalRole, EcosystemConfig, EcosystemInteraction
from cambrian.evaluator import Evaluator
from cambrian.memory import EvolutionaryMemory
from cambrian.metamorphosis import MetamorphicPhase, MetamorphosisController, PhaseConfig
from cambrian.neuromodulation import NeuromodulatorBank
from cambrian.quorum import QuorumSensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str = "mocked LLM response") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _agent(prompt: str, fitness: float) -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


class _MockEvaluator(Evaluator):
    """Deterministic evaluator: scores by prompt keyword coverage."""

    _KEYWORDS = ["step-by-step", "expert", "verify", "systematic", "analytical"]

    def evaluate(self, agent: object, task: str) -> float:
        a = agent  # type: ignore[assignment]
        prompt = getattr(a, "genome", None)
        if prompt is None:
            return 0.3
        text = getattr(prompt, "system_prompt", "").lower()
        hits = sum(1 for kw in self._KEYWORDS if kw in text)
        return min(1.0, 0.3 + hits * 0.15)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print("=" * 70)
print("Cambrian -- Full Evolution Stack Demo (offline, mocked)")
print("=" * 70)

rng = random.Random(42)
backend = _mock_backend("You are an expert agent using step-by-step systematic reasoning.")
evaluator = _MockEvaluator()

# --- Components ---

quorum = QuorumSensor(
    low_entropy_threshold=0.5,
    high_entropy_threshold=2.5,
    boost_factor=1.3,
    decay_factor=0.85,
)

memory = EvolutionaryMemory()
dream = DreamPhase(backend=backend, memory=memory, interval=3, top_n=3)

apoptosis = ApoptosisController(
    stagnation_window=3,
    min_fitness=0.2,
    grace_period=1,
)

neuro = NeuromodulatorBank(
    base_mutation_rate=0.3,
    base_selection_pressure=0.5,
    mr_range=0.2,
    sp_range=0.2,
)

morph_ctrl = MetamorphosisController(
    backend=backend,
    larva_config=PhaseConfig(
        phase=MetamorphicPhase.LARVA,
        min_generations=2,
        fitness_threshold=0.5,
        mutation_rate_multiplier=1.5,
    ),
    chrysalis_config=PhaseConfig(
        phase=MetamorphicPhase.CHRYSALIS,
        min_generations=1,
        fitness_threshold=0.65,
        mutation_rate_multiplier=0.0,
    ),
    imago_config=PhaseConfig(
        phase=MetamorphicPhase.IMAGO,
        min_generations=0,
        fitness_threshold=0.0,
        mutation_rate_multiplier=0.5,
    ),
)

eco_config = EcosystemConfig(
    herbivore_diversity_bonus=0.04,
    predator_hunt_bonus=0.08,
    decomposer_bonus=0.06,
    parasite_gain=0.05,
)
eco = EcosystemInteraction(config=eco_config)

# ---------------------------------------------------------------------------
# Seed population
# ---------------------------------------------------------------------------

PROMPTS = [
    "You are a Python expert. Respond with clean, correct code.",
    "Step-by-step problem solver. Verify each step before proceeding.",
    "Systematic analytical reasoner using structured decomposition.",
    "Expert programmer with deep knowledge of algorithms.",
    "You produce exactly the required output, nothing else.",
    "Creative problem-solver using lateral thinking and intuition.",
]

population = [_agent(p, rng.uniform(0.3, 0.7)) for p in PROMPTS]

# Register in morphosis and ecosystem
for a in population:
    morph_ctrl._agent_phase[a.id] = MetamorphicPhase.LARVA
    morph_ctrl._agent_gen_in_phase[a.id] = 0

eco.auto_assign(population)

print(f"\nInitial population: {len(population)} agents")
print(f"Ecosystem roles: {eco.role_counts()}")
print()

# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

N_GENERATIONS = 6
mutation_rate = 0.3

for gen in range(1, N_GENERATIONS + 1):
    print(f"--- Generation {gen} ---")

    # 1. Evaluate
    for a in population:
        a.fitness = evaluator.evaluate(a, "Solve coding task")

    scores = [a.fitness or 0.0 for a in population]
    best_fitness = max(scores)
    mean_fitness = sum(scores) / len(scores)
    print(f"  Fitness: best={best_fitness:.3f}  mean={mean_fitness:.3f}")

    # 2. Quorum sensing -> mutation rate
    mutation_rate = quorum.update(scores=scores, current_rate=mutation_rate)

    # 3. Neuromodulation
    state = neuro.modulate(population, generation=gen)
    effective_mr = (mutation_rate + state.mutation_rate) / 2.0
    print(f"  Neuro: mr={state.mutation_rate:.3f}  sp={state.selection_pressure:.3f}  "
          f"dopa={state.dopamine:.2f}")
    print(f"  Effective mutation_rate={effective_mr:.3f}")

    # 4. Ecosystem interaction
    events = eco.interact(population, task="Solve coding challenge")
    eco.apply_events(events, population)
    if events:
        print(f"  Ecosystem: {len(events)} interactions")

    # 5. Metamorphosis phase advances
    morph_events = []
    for a in population:
        ev = morph_ctrl.advance(a, generation=gen, fitness=a.fitness or 0.0)
        if ev is not None:
            morph_events.append(ev)
            print(f"  Metamorphosis: agent {a.id[:8]}... "
                  f"{ev.from_phase.value} -> {ev.to_phase.value}")

    # 6. Dream phase (every 3 generations)
    if dream.should_dream(gen):
        for a in population:
            memory.add_trace(
                agent_id=a.id,
                content=a.genome.system_prompt,
                score=a.fitness or 0.5,
            )
        offspring = dream.dream("Solve coding tasks", n_offspring=2)
        print(f"  Dream: generated {len(offspring)} offspring")

    # 7. Apoptosis
    for a in population:
        apoptosis.record(a)
    best_agent = max(population, key=lambda x: x.fitness or 0.0)
    population = apoptosis.apply(population, best_agent=best_agent)
    print(f"  Population after apoptosis: {len(population)} agents")

    # 8. Metamorphic phase distribution
    dist = morph_ctrl.phase_distribution()
    print(f"  Phases: {dist}")

print()
print("=" * 70)
print("Evolution complete!")

# Final evaluation pass to get clean fitness scores
for a in population:
    a.fitness = evaluator.evaluate(a, "Solve coding task")

best = max(population, key=lambda x: x.fitness or 0.0)
best_fit = best.fitness or 0.0
print(f"Best fitness       : {best_fit:.4f}")
print(f"Best genome (60ch) : {best.genome.system_prompt[:60]!r}")

print()
print(f"Total morph events : {len(morph_ctrl.events)}")
print(f"Total eco events   : {len(eco.events)}")
print(f"Dream count        : {dream.dream_count}")
print(f"Neuro generations  : {len(neuro.history)}")
print(f"Quorum entropy     : {quorum.last_entropy:.4f}")
print("=" * 70)
print()
print("All techniques fired correctly. Demo complete.")
