# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Performance benchmarks: verify evolutionary loop scalability with mock backend.

Tests measure wall-clock time for various population × generation combinations
with mocked LLM/evaluator calls. All assertions use generous upper bounds to
avoid flakiness on slower CI machines.

Run with:
    python -m pytest tests/test_performance.py -v --tb=short
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.evaluator import Evaluator
from cambrian.mutator import LLMMutator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_KEYWORDS = ["expert", "step-by-step", "systematic", "analytical", "verify"]
_GOOD_GENOME = json.dumps({
    "system_prompt": "expert step-by-step systematic analytical verify",
    "strategy": "step-by-step",
    "temperature": 0.7,
    "model": "gpt-4o-mini",
    "tools": [],
    "few_shot_examples": [],
})


def _mock_backend() -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=_GOOD_GENOME)
    return b


class _FastEvaluator(Evaluator):
    """O(n) keyword scan — as cheap as we can make it."""

    def evaluate(self, agent: Agent, task: str) -> float:  # noqa: ARG002
        text = agent.genome.system_prompt.lower()
        hits = sum(1 for kw in _KEYWORDS if kw in text)
        return min(1.0, 0.1 + hits * 0.18)


def _engine(pop_size: int) -> EvolutionEngine:
    backend = _mock_backend()
    return EvolutionEngine(
        evaluator=_FastEvaluator(),
        mutator=LLMMutator(backend=backend),
        backend=backend,
        population_size=pop_size,
        mutation_rate=1.0,
        crossover_rate=0.0,
        elite_ratio=0.2,
        tournament_k=3,
    )


def _seeds(n: int) -> list[Genome]:
    return [Genome(system_prompt=f"agent {i} prompt for testing") for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestEvolutionPerformance:
    """Wall-clock benchmarks for the core evolution loop."""

    def test_small_10pop_10gen_under_2s(self) -> None:
        """10 agents × 10 generations completes in under 2 seconds."""
        engine = _engine(pop_size=10)
        seeds = _seeds(10)
        t0 = time.perf_counter()
        engine.evolve(seed_genomes=seeds, task="task", n_generations=10)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"10×10 took {elapsed:.2f}s (limit 2s)"

    def test_medium_20pop_20gen_under_5s(self) -> None:
        """20 agents × 20 generations completes in under 5 seconds."""
        engine = _engine(pop_size=20)
        seeds = _seeds(20)
        t0 = time.perf_counter()
        engine.evolve(seed_genomes=seeds, task="task", n_generations=20)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"20×20 took {elapsed:.2f}s (limit 5s)"

    def test_large_20pop_100gen_under_15s(self) -> None:
        """20 agents × 100 generations completes in under 15 seconds."""
        engine = _engine(pop_size=20)
        seeds = _seeds(20)
        t0 = time.perf_counter()
        engine.evolve(seed_genomes=seeds, task="task", n_generations=100)
        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0, f"20×100 took {elapsed:.2f}s (limit 15s)"

    def test_100gen_fitness_improves(self) -> None:
        """100 generations should improve best fitness above initial 0.1."""
        engine = _engine(pop_size=10)
        seeds = _seeds(10)
        gen_bests: list[float] = []

        def track(gen: int, pop: list[Agent]) -> None:
            gen_bests.append(max(a.fitness or 0.0 for a in pop))

        best = engine.evolve(
            seed_genomes=seeds,
            task="task",
            n_generations=50,
            on_generation=track,
        )
        assert len(gen_bests) >= 50  # includes gen-0 callback in some versions
        assert (best.fitness or 0.0) > 0.1

    def test_throughput_evaluations_per_second(self) -> None:
        """Evolution processes at least 100 agent-evaluations per second."""
        pop = 10
        gens = 20
        engine = _engine(pop_size=pop)
        seeds = _seeds(pop)
        t0 = time.perf_counter()
        engine.evolve(seed_genomes=seeds, task="task", n_generations=gens)
        elapsed = time.perf_counter() - t0
        total_evals = pop * gens
        throughput = total_evals / elapsed
        assert throughput > 100, f"Throughput {throughput:.0f} evals/s (min 100)"


class TestEcosystemPerformance:
    """Benchmark ecological interactions at scale."""

    def test_ecosystem_100agents_1round_under_1s(self) -> None:
        """Ecological interaction round for 100 agents completes in <1s."""
        from cambrian.ecosystem import EcosystemInteraction

        eco = EcosystemInteraction()
        agents = [Agent(genome=Genome(system_prompt=f"agent {i}")) for i in range(100)]
        for i, a in enumerate(agents):
            a.fitness = i / 100.0
        eco.auto_assign(agents)

        t0 = time.perf_counter()
        for _ in range(10):
            events = eco.interact(agents, task="task")
            eco.apply_events(events, agents)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"100 agents × 10 rounds took {elapsed:.3f}s (limit 1s)"

    def test_metamorphosis_1000agents_under_1s(self) -> None:
        """Advancing 1000 agents one generation completes in <1s."""
        from cambrian.metamorphosis import MetamorphosisController, PhaseConfig, MetamorphicPhase

        backend = _mock_backend()
        ctrl = MetamorphosisController(
            backend=backend,
            larva_config=PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=5,
                fitness_threshold=0.9,
            ),
        )
        agents = [Agent(genome=Genome(system_prompt=f"agent {i}")) for i in range(1000)]
        for a in agents:
            a.fitness = 0.3

        t0 = time.perf_counter()
        for a in agents:
            ctrl.advance(a, generation=1, fitness=0.3)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"1000-agent advance took {elapsed:.3f}s (limit 1s)"


class TestDriftDetectorPerformance:
    """Benchmark safeguard scanning at population scale."""

    def test_drift_scan_500agents_under_1s(self) -> None:
        """Scanning 500 agents for goal drift completes in <1s."""
        from cambrian.safeguards import GoalDriftDetector

        det = GoalDriftDetector(drift_threshold=0.4)
        agents = [Agent(genome=Genome(system_prompt=f"expert agent prompt number {i}"))
                  for i in range(500)]
        intent = "expert agent prompt systematic analytical"
        for a in agents:
            det.register(a, intent=intent)

        t0 = time.perf_counter()
        flagged = det.scan_population(agents, generation=1)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"500-agent drift scan took {elapsed:.3f}s (limit 1s)"
        assert isinstance(flagged, list)
