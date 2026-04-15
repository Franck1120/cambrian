# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tier 3 & Tier 4 Feature Demo -- cambrian.

Demonstrates how to wire together the advanced bio-inspired modules introduced
in Tier 3 (Techniques 51-65) and Tier 4 (Techniques 57-66) without a live LLM.

All backend calls are mocked so this script runs offline, instantly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.annealing import AnnealingSchedule, AnnealingSelector
from cambrian.apoptosis import ApoptosisController
from cambrian.ensemble import AgentEnsemble
from cambrian.hgt import HGTPool, HGTransfer
from cambrian.hormesis import HormesisAdapter
from cambrian.immune_memory import ImmuneCortex
from cambrian.llm_cascade import CascadeLevel, LLMCascade
from cambrian.neuromodulation import NeuromodulatorBank
from cambrian.symbiosis import SymbioticFuser
from cambrian.transgenerational import TransgenerationalRegistry
from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float, prompt: str) -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(response: str = "mocked response") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


# ---------------------------------------------------------------------------
# TIER 3: Symbiotic Fusion
# ---------------------------------------------------------------------------

print("=" * 60)
print("TIER 3 -- Symbiotic Fusion")
print("=" * 60)

backend = _mock_backend("You are a hybrid expert agent combining both strategies.")
fuser = SymbioticFuser(backend=backend, fitness_threshold=0.6, min_distance=0.1)

host = _make_agent(0.9, "Step-by-step analytical reasoning agent.")
donor = _make_agent(0.85, "Creative lateral thinking approach agent.")

fused = fuser.fuse(host, donor, task="Solve complex optimisation problems")
if fused:
    print(f"Fused genome (first 80 chars): {fused.genome.system_prompt[:80]!r}")
else:
    print("Fusion skipped (below threshold or incompatible)")

# ---------------------------------------------------------------------------
# TIER 3: Hormesis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 3 -- Hormesis (graduated stress response)")
print("=" * 60)

backend = _mock_backend("Re-prompted under severe stress: focus on efficiency.")
adapter = HormesisAdapter(backend=backend, stress_threshold=0.5)

weak_agent = _make_agent(0.2, "Basic agent.")
stimulated = adapter.stimulate(weak_agent, task="Solve hard maths problem")
print(f"Original fitness: {weak_agent.fitness:.2f}")
print(f"Stimulated (stress applied): {stimulated is not None}")

# ---------------------------------------------------------------------------
# TIER 3: Apoptosis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 3 -- Apoptosis (programmed removal)")
print("=" * 60)

ctrl = ApoptosisController(stagnation_window=3, min_fitness=0.3, grace_period=2)
best_ctrl = _make_agent(0.9, "Best agent.")
poor_ctrl = _make_agent(0.1, "Poor agent.")

for _ in range(4):
    ctrl.record(poor_ctrl)

population = [best_ctrl, poor_ctrl]
survivors = ctrl.apply(population, best_agent=best_ctrl)
print(f"Population before: {len(population)}")
print(f"Population after apoptosis: {len(survivors)}")

# ---------------------------------------------------------------------------
# TIER 3: LLM Cascade
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 3 -- LLM Cascade (tiered routing)")
print("=" * 60)

fast_backend = _mock_backend("I'm not entirely sure, but probably 42.")
smart_backend = _mock_backend("The answer is definitively 42.")

cascade = LLMCascade(
    levels=[
        CascadeLevel(fast_backend, confidence_threshold=0.8),
        CascadeLevel(smart_backend, confidence_threshold=0.0),
    ]
)
response, level_idx = cascade.query("You are a math expert.", "What is 6x7?")
print(f"Answer from cascade: {response!r}")
print(f"Answered at level: {level_idx}")

# ---------------------------------------------------------------------------
# TIER 3: Ensemble + Boosting
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 3 -- Ensemble + Boosting")
print("=" * 60)

a1 = _make_agent(0.8, "Answer 42 always.")
a2 = _make_agent(0.7, "Answer 42 when asked.")
a3 = _make_agent(0.5, "Compute the answer.")

for a in [a1, a2, a3]:
    a.run = MagicMock(return_value="42")  # type: ignore[method-assign]

ensemble = AgentEnsemble(agents=[a1, a2, a3])
answer = ensemble.query(task="What is 6x7?", correct_answer="42")
print(f"Ensemble answer: {answer!r}")
print(f"Agent weights: {[f'{w:.2f}' for w in ensemble.weights]}")

# ---------------------------------------------------------------------------
# TIER 4: Immune Memory
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- B/T-cell Immune Memory")
print("=" * 60)

cortex = ImmuneCortex(b_threshold=0.7, t_threshold=0.4, b_similarity=0.5)

good_agent = _make_agent(0.95, "Expert Python debugger.")
cortex.record(good_agent, task="Debug a Python IndexError in a list comprehension")

recall = cortex.recall("Fix Python IndexError in my list comprehension code")
print(f"Recalled: {recall.recalled}  (type: {recall.cell_type})")
if recall.agent:
    print(f"Recalled genome: {recall.agent.genome.system_prompt!r}")
print(f"Similarity: {recall.similarity:.3f}")
print(f"B-cell count: {cortex.b_cell_count}, T-cell count: {cortex.t_cell_count}")

# ---------------------------------------------------------------------------
# TIER 4: Neuromodulation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- Neuromodulation")
print("=" * 60)

bank = NeuromodulatorBank(
    base_mutation_rate=0.25,
    base_selection_pressure=0.5,
    mr_range=0.2,
    sp_range=0.2,
)

for gen in range(6):
    pop = [_make_agent(0.3 + gen * 0.1 + i * 0.05, f"agent {i}") for i in range(4)]
    state = bank.modulate(pop, generation=gen)
    print(
        f"  Gen {gen}: mr={state.mutation_rate:.3f}  sp={state.selection_pressure:.3f}"
        f"  dopa={state.dopamine:.2f}  sero={state.serotonin:.2f}"
        f"  ach={state.acetylcholine:.2f}  nora={state.noradrenaline:.2f}"
    )

# ---------------------------------------------------------------------------
# TIER 4: Zeitgeber
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- Zeitgeber (circadian oscillator)")
print("=" * 60)

clock = ZeitgeberClock(period=8, amplitude=0.5)
scheduler = ZeitgeberScheduler(
    clock=clock,
    base_mutation_rate=0.3,
    base_threshold=0.5,
)
for tick in range(9):
    mr, thr = scheduler.tick()
    print(f"  Tick {tick}: mutation_rate={mr:.3f}  threshold={thr:.3f}")

# ---------------------------------------------------------------------------
# TIER 4: Transgenerational Epigenetics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- Transgenerational Epigenetics")
print("=" * 60)

registry = TransgenerationalRegistry(max_generations=5)

parent = _make_agent(0.9, "Expert agent.")
child = _make_agent(0.7, "Child agent.")
grandchild = _make_agent(0.6, "Grandchild agent.")

registry.record_mark(parent, "step-by-step", strength=0.9)
registry.record_mark(parent, "verify-output", strength=0.7)

n1 = registry.inherit(parent, child)
n2 = registry.inherit(child, grandchild)

print(f"Marks transferred parent->child: {n1}")
print(f"Marks transferred child->grandchild: {n2}")
print(f"Grandchild marks: {[(m.name, f'{m.strength:.2f}') for m in registry.get_marks(grandchild)]}")

genome_with_epi = registry.apply_to_genome(grandchild)
print(f"Genome with epigenetic context (first 150 chars):\n  {genome_with_epi.system_prompt[:150]!r}")

# ---------------------------------------------------------------------------
# TIER 4: HGT
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- Horizontal Gene Transfer (HGT)")
print("=" * 60)

transfer = HGTransfer(n_sentences=1, mode="suffix", fitness_threshold=0.6)
pool = HGTPool(max_plasmids=10)

donor = _make_agent(0.9, "First strategy. Second improvement. Third key insight.")
recipient = _make_agent(0.5, "Baseline approach. Simple method.")

pool.contribute(donor, domain="reasoning")
plasmid = pool.draw(domain="reasoning")
if plasmid:
    print(f"Drawn plasmid content: {plasmid.content!r}")
    print(f"Donor fitness: {plasmid.donor_fitness:.2f}")

offspring = transfer.transfer(donor, recipient)
if offspring:
    print(f"HGT offspring prompt: {offspring.genome.system_prompt!r}")

# ---------------------------------------------------------------------------
# TIER 4: Simulated Annealing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIER 4 -- Simulated Annealing")
print("=" * 60)

schedule = AnnealingSchedule(T_max=1.0, T_min=0.05, n_steps=20, schedule_type="cosine")
selector = AnnealingSelector(schedule)

accepted = rejected = 0
for t in range(20):
    current = 0.5
    candidate = 0.4 if t % 3 == 0 else 0.6
    if selector.step(current, candidate):
        accepted += 1
    else:
        rejected += 1

print(f"Accepted: {accepted}, Rejected: {rejected}")
print(f"Final temperature: {schedule.temperature(20):.4f}")
print(f"Acceptance rate: {selector.acceptance_rate():.2f}")

print("\nAll Tier 3 & Tier 4 demos completed successfully.")
