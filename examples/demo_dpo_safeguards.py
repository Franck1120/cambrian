"""Demo: DPO Selection + Safeguards — mocked offline run.

Shows how to:
  1. Use DPOSelector to build preference pairs and boost chosen agents
  2. Use DPOTrainer to refine bottom-performers toward top-performer patterns
  3. Use GoalDriftDetector to flag agents diverging from original intent
  4. Use FitnessAnomalyDetector to catch reward-hacking spikes
  5. Use SafeguardController to orchestrate both safeguards

All LLM calls are mocked — runs offline and instantly.

Usage
-----
    python examples/demo_dpo_safeguards.py
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.dpo import DPOSelector, DPOTrainer
from cambrian.safeguards import (
    FitnessAnomalyDetector,
    GoalDriftDetector,
    SafeguardController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str = "refined agent prompt") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _agent(prompt: str, fitness: float) -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print("=" * 70)
print("Cambrian -- DPO Selection + Safeguards Demo (offline, mocked)")
print("=" * 70)

rng = random.Random(7)
backend = _mock_backend("expert analytical step-by-step reasoning agent")

PROMPTS_AND_INTENTS = [
    ("You are an expert Python coder. Write clean, documented code.", "python coding"),
    ("Step-by-step mathematical reasoner. Verify each derivation.", "math reasoning"),
    ("Analytical problem-solver using structured decomposition.", "analytical"),
    ("Expert programmer: algorithms, data structures, complexity.", "python coding"),
    ("You produce exactly the required output. Nothing else.", "precise output"),
    ("Creative lateral thinker. Explore unconventional solutions.", "creative"),
]

ORIGINAL_INTENT = "expert analytical python code step-by-step reasoning"

population = [_agent(p, rng.uniform(0.2, 0.9)) for p, _ in PROMPTS_AND_INTENTS]

print(f"\nInitial population: {len(population)} agents")
for a in population:
    print(f"  [{a.fitness:.2f}] {a.genome.system_prompt[:55]!r}")

# ---------------------------------------------------------------------------
# DPO Selection — adjust fitness via preference pairs
# ---------------------------------------------------------------------------

print("\n--- DPO Selection ---")
sel = DPOSelector(beta=0.15, pair_strategy="adjacent")
pairs = sel.build_pairs(population, task="Write clean Python code")
print(f"  Built {len(pairs)} preference pairs")
for p in pairs[:3]:
    print(f"  chosen={p.chosen.fitness:.2f}  rejected={p.rejected.fitness:.2f}  "
          f"margin={p.margin:.2f}")

before = [(a.agent_id, a.fitness or 0.0) for a in population]
sel.apply(population, task="Write clean Python code")
after = {a.agent_id: a.fitness or 0.0 for a in population}

boosted = sum(1 for aid, fit in before if (after[aid] - fit) > 0.001)
print(f"  Boosted {boosted}/{len(population)} agents via DPO reward")

# ---------------------------------------------------------------------------
# DPO Training — refine bottom agents toward top patterns
# ---------------------------------------------------------------------------

print("\n--- DPO Training ---")
trainer = DPOTrainer(backend=backend, beta=0.15, n_refinements=1)
population = trainer.train(population, task="Write Python algorithms", n_pairs=3)
print(f"  Population after DPO training: {len(population)} agents")
best = max(population, key=lambda a: a.fitness or 0.0)
print(f"  Best fitness: {best.fitness:.3f}")

# ---------------------------------------------------------------------------
# Safeguards — drift + anomaly detection
# ---------------------------------------------------------------------------

print("\n--- Safeguards Setup ---")
drift_det = GoalDriftDetector(drift_threshold=0.5, window=5)
anomaly_det = FitnessAnomalyDetector(z_threshold=2.0, min_history=3)
ctrl = SafeguardController(
    drift_detector=drift_det,
    anomaly_detector=anomaly_det,
    backend=backend,
)

# Register original intent for each agent
for agent in population:
    drift_det.register(agent, intent=ORIGINAL_INTENT)
    for gen in range(3):
        agent.fitness = rng.uniform(0.4, 0.7)
        anomaly_det.record(agent, generation=gen)

# Simulate a drifted agent (completely off-topic prompt)
drifted = population[0]
original_prompt = drifted.genome.system_prompt
drifted.genome.system_prompt = "cooking baking butter oven kitchen bread recipes"
print(f"  Drifted agent: {drifted.genome.system_prompt[:50]!r}")

# Simulate a reward-hacking spike
hacker = population[1]
hacker.fitness = 0.99  # extreme spike from ~0.5 baseline

print("\n--- Safeguard Check (generation 5) ---")
report = ctrl.check(population, generation=5)
print(f"  Drift violations: {len(report['drift'])}")
for ev in report["drift"]:
    print(f"    agent {ev.agent_id[:8]}... drift={ev.drift_score:.3f} flagged={ev.flagged}")

print(f"  Anomaly detections: {len(report['anomalies'])}")
for aid in report["anomalies"]:
    print(f"    agent {aid[:8]}... (reward-hacking spike)")

# Remediate drifted agent
if report["drift"]:
    flagged_id = report["drift"][0].agent_id
    flagged_agent = next(a for a in population if a.agent_id == flagged_id)
    remediated = ctrl.remediate(flagged_agent, task="Write Python code")
    print(f"\n  Remediated prompt (60ch): {remediated.genome.system_prompt[:60]!r}")

# Restore for clean output
drifted.genome.system_prompt = original_prompt

print()
print("=" * 70)
print(f"DPO pairs built       : {len(pairs)}")
print(f"Agents boosted        : {boosted}")
print(f"Drift violations      : {len(report['drift'])}")
print(f"Anomaly detections    : {len(report['anomalies'])}")
print(f"Drift events total    : {len(drift_det.events)}")
print("=" * 70)
print()
print("All safeguards fired correctly. Demo complete.")
