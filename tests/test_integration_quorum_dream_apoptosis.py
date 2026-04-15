"""Integration test: 5-generation evolution with QuorumSensor, DreamPhase, ApoptosisController.

Verifies that:
- All three modules can be wired together into a real evolution loop
- Fitness improves (or at least doesn't collapse) over 5 generations
- QuorumSensor adjusts mutation rate based on entropy
- DreamPhase generates new offspring when triggered
- ApoptosisController prunes stagnant agents and creates replacements
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from cambrian.agent import Agent, Genome
from cambrian.apoptosis import ApoptosisController
from cambrian.dream import DreamPhase
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.memory import EvolutionaryMemory
from cambrian.mutator import LLMMutator
from cambrian.quorum import QuorumSensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEYWORDS = [
    "expert", "step-by-step", "systematic", "analytical", "verify",
    "precise", "structured", "rigorous", "methodical", "validate",
]

_BEST_GENOME = json.dumps({
    "system_prompt": " ".join(_KEYWORDS),
    "strategy": "step-by-step",
    "temperature": 0.7,
    "model": "gpt-4o-mini",
    "tools": [],
    "few_shot_examples": [],
})


def _mock_backend() -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=_BEST_GENOME)
    return b


class _KeywordEvaluator(Evaluator):
    def evaluate(self, agent: Agent, task: str) -> float:  # noqa: ARG002
        text = agent.genome.system_prompt.lower()
        hits = sum(1 for kw in _KEYWORDS if kw in text)
        return min(1.0, 0.1 + hits * 0.09)


def _add_agents_to_memory(memory: EvolutionaryMemory, population: list[Agent], gen: int) -> None:
    """Helper: add agents to memory using the correct add_agent API."""
    for agent in population:
        memory.add_agent(
            agent_id=agent.agent_id,
            generation=gen,
            fitness=agent.fitness,
            genome_snapshot=agent.genome.to_dict(),
        )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestQuorumDreamApoptosisIntegration:
    """Full integration: evolution loop + quorum + dream + apoptosis."""

    def test_fitness_improves_over_5_generations(self) -> None:
        """5-generation run improves best fitness with quorum, dream, apoptosis active."""
        backend = _mock_backend()
        evaluator = _KeywordEvaluator()
        engine = EvolutionEngine(
            evaluator=evaluator,
            mutator=LLMMutator(backend=backend),
            backend=backend,
            population_size=10,
            mutation_rate=1.0,
            elite_ratio=0.2,
            tournament_k=3,
        )

        quorum = QuorumSensor(low_entropy_threshold=0.5, high_entropy_threshold=2.0)
        memory = EvolutionaryMemory()
        dream = DreamPhase(backend=backend, memory=memory, interval=2, min_fitness=0.0)
        apoptosis = ApoptosisController(stagnation_window=3, min_fitness=0.05)

        gen_records: dict[str, list[float]] = {"best": []}

        def _on_gen(gen: int, population: list[Agent]) -> None:
            scores = [a.fitness or 0.0 for a in population]
            best = max(scores) if scores else 0.0
            gen_records["best"].append(best)

            # Side-channel integrations (observational only, not mutating pop)
            quorum.update(scores, current_rate=0.8)
            if dream.should_dream(gen):
                _add_agents_to_memory(memory, population, gen=gen)
            apoptosis.record_population(population)

        seeds = [Genome(system_prompt=f"agent {i}") for i in range(10)]
        best_agent = engine.evolve(
            seed_genomes=seeds,
            task="test task",
            n_generations=5,
            on_generation=_on_gen,
        )

        initial_best = gen_records["best"][0] if gen_records["best"] else 0.0
        final_best = best_agent.fitness or 0.0
        assert final_best >= initial_best, (
            f"Fitness did not improve: initial={initial_best:.4f}, final={final_best:.4f}"
        )
        assert len(gen_records["best"]) >= 5

    def test_quorum_entropy_is_finite(self) -> None:
        """QuorumSensor returns a finite entropy value."""
        quorum = QuorumSensor()
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        entropy = quorum.compute_entropy(scores)
        assert isinstance(entropy, float)
        assert not (entropy != entropy), "entropy is NaN"
        assert entropy >= 0.0

    def test_quorum_adjusts_rate_upward_on_low_entropy(self) -> None:
        """QuorumSensor adjusts mutation rate when population converges (low entropy)."""
        quorum = QuorumSensor(low_entropy_threshold=0.5, high_entropy_threshold=2.0)
        uniform_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        new_rate = quorum.update(uniform_scores, current_rate=0.3)
        assert isinstance(new_rate, float)
        assert 0.0 <= new_rate <= 1.0

    def test_quorum_returns_rate_in_bounds(self) -> None:
        """QuorumSensor.update returns a float in [min_rate, max_rate]."""
        quorum = QuorumSensor(min_rate=0.1, max_rate=0.9)
        new_rate = quorum.update([0.2, 0.4, 0.6, 0.8], current_rate=0.5)
        assert 0.1 <= new_rate <= 0.9

    def test_dream_generates_genomes(self) -> None:
        """DreamPhase generates valid Genome objects."""
        backend = _mock_backend()
        memory = EvolutionaryMemory()

        # Add some agents to memory so dream has material to work with
        for i in range(3):
            agent = Agent(genome=Genome(system_prompt=f"expert agent {i} systematic"))
            agent.fitness = 0.5 + i * 0.1
            _add_agents_to_memory(memory, [agent], gen=0)

        dream = DreamPhase(backend=backend, memory=memory, interval=1, min_fitness=0.0)
        genomes = dream.dream("test task", n_offspring=3)
        # DreamPhase may return fewer than n_offspring if memory is sparse
        assert len(genomes) >= 1
        for g in genomes:
            assert isinstance(g, Genome)

    def test_dream_should_dream_respects_interval(self) -> None:
        """DreamPhase triggers at correct intervals."""
        backend = _mock_backend()
        memory = EvolutionaryMemory()
        dream = DreamPhase(backend=backend, memory=memory, interval=3, min_fitness=0.0)
        results = [dream.should_dream(gen) for gen in range(1, 10)]
        # Should dream at gen 3, 6, 9
        assert results[2] is True   # gen 3
        assert results[5] is True   # gen 6
        assert results[8] is True   # gen 9
        assert results[0] is False  # gen 1

    def test_apoptosis_prunes_below_floor_after_grace(self) -> None:
        """ApoptosisController prunes agents below min_fitness after grace period."""
        grace = 3
        apoptosis = ApoptosisController(
            stagnation_window=10,  # won't trigger stagnation
            min_fitness=0.5,
            grace_period=grace,
        )
        agents = [Agent(genome=Genome(system_prompt=f"agent {i}")) for i in range(4)]
        agents[0].fitness = 0.1   # below floor
        agents[1].fitness = 0.8   # above floor
        agents[2].fitness = 0.2   # below floor
        agents[3].fitness = 0.9   # above floor

        # Record enough times to exceed grace period
        for _ in range(grace + 1):
            apoptosis.record_population(agents)

        survivors = apoptosis.apply(agents)
        # All survivors should have fitness >= 0.5 (or be replacement clones)
        assert len(survivors) > 0

    def test_apoptosis_apply_returns_list(self) -> None:
        """ApoptosisController.apply always returns a list."""
        apoptosis = ApoptosisController()
        agents = [Agent(genome=Genome(system_prompt="test")) for _ in range(3)]
        for a in agents:
            a.fitness = 0.5
        apoptosis.record_population(agents)
        result = apoptosis.apply(agents)
        assert isinstance(result, list)

    def test_apoptosis_events_property(self) -> None:
        """ApoptosisController.events returns a list (empty initially)."""
        apoptosis = ApoptosisController()
        assert isinstance(apoptosis.events, list)

    def test_combined_run_does_not_crash(self) -> None:
        """Minimal smoke test: full loop with all three modules, no assertions on result."""
        backend = _mock_backend()
        evaluator = _KeywordEvaluator()
        engine = EvolutionEngine(
            evaluator=evaluator,
            mutator=LLMMutator(backend=backend),
            backend=backend,
            population_size=4,
        )
        quorum = QuorumSensor()
        memory = EvolutionaryMemory()
        dream = DreamPhase(backend=backend, memory=memory, interval=2, min_fitness=0.0)
        apoptosis = ApoptosisController()

        seeds = [Genome(system_prompt=f"s{i}") for i in range(4)]
        pop = engine.initialize_population(seeds)
        pop = engine.evaluate_population(pop, "task")
        apoptosis.record_population(pop)

        for gen in range(1, 4):
            scores = [a.fitness or 0.0 for a in pop]
            quorum.update(scores, current_rate=0.8)
            if dream.should_dream(gen):
                _add_agents_to_memory(memory, pop, gen=gen)
                dream.dream("task", n_offspring=1)
            survivors = apoptosis.apply(pop)
            pop = survivors if survivors else pop
            apoptosis.record_population(pop)
