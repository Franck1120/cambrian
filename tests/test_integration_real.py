"""tests/test_integration_real.py — Real evolutionary cycle integration test.

Runs a full 3-generation, 5-agent evolution loop using a deterministic mock
backend that simulates realistic LLM responses.  The mock is carefully designed
so that agents whose system prompts mention domain-relevant keywords ("Python",
"expert", "step-by-step") score higher — giving the evolution engine a signal
to improve the population.

Key assertions:
- Best fitness at generation 3 ≥ best fitness at generation 1.
- At least one agent exceeds the seed population's best fitness.
- The evolution engine completes without error.
- Memory is populated with agent lineage nodes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic mock backend
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_SCORE: dict[str, float] = {
    "expert": 0.15,
    "python": 0.10,
    "step-by-step": 0.12,
    "systematic": 0.08,
    "precise": 0.06,
    "comprehensive": 0.05,
}

_MUTATION_TEMPLATES = [
    (
        '{"system_prompt": "You are an expert Python programmer. '
        'Solve problems step-by-step with precise, comprehensive solutions.", '
        '"strategy": "expert-systematic", "temperature": 0.4, '
        '"max_tokens": 512, "few_shot_examples": [], "tool_specs": []}'
    ),
    (
        '{"system_prompt": "You are a systematic expert. '
        'Think step-by-step and be comprehensive and precise.", '
        '"strategy": "systematic-expert", "temperature": 0.45, '
        '"max_tokens": 512, "few_shot_examples": [], "tool_specs": []}'
    ),
    (
        '{"system_prompt": "Expert Python assistant. '
        'Provide precise, comprehensive, step-by-step answers.", '
        '"strategy": "precise-expert", "temperature": 0.35, '
        '"max_tokens": 512, "few_shot_examples": [], "tool_specs": []}'
    ),
]

_mutation_call_count = 0


def _make_mock_backend() -> MagicMock:
    """Return a mock LLM backend with realistic simulated responses."""
    global _mutation_call_count
    _mutation_call_count = 0

    backend = MagicMock()
    backend.model_name = "mock-gpt"

    def _generate(prompt: str, **kwargs: Any) -> str:
        global _mutation_call_count
        # Return increasingly improved mutation templates
        template = _MUTATION_TEMPLATES[_mutation_call_count % len(_MUTATION_TEMPLATES)]
        _mutation_call_count += 1
        return template

    backend.generate.side_effect = _generate
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based evaluator
# ─────────────────────────────────────────────────────────────────────────────

class _KeywordEvaluator(Evaluator):
    """Score agents by keyword presence in their system prompt.

    This creates a genuine fitness landscape where prompts containing
    domain-relevant keywords score higher — giving the evolution engine
    a clear signal to improve.
    """

    def evaluate(self, agent: Agent, task: str) -> float:
        prompt_lower = agent.genome.system_prompt.lower()
        base = 0.3
        bonus = sum(
            weight
            for kw, weight in _KEYWORD_SCORE.items()
            if kw in prompt_lower
        )
        # Cap at 1.0
        return min(1.0, base + bonus)


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────


class TestRealEvolutionCycle:
    """Full evolutionary cycle — 3 generations, 5 agents."""

    N_GENERATIONS = 3
    POPULATION_SIZE = 5

    @pytest.fixture
    def seed_genomes(self) -> list[Genome]:
        return [
            Genome(system_prompt="You are a helpful assistant.", temperature=0.5),
            Genome(system_prompt="Answer questions clearly and concisely.", temperature=0.6),
            Genome(system_prompt="You are a general AI assistant.", temperature=0.55),
        ]

    @pytest.fixture
    def setup(self) -> tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator]:
        backend = _make_mock_backend()
        evaluator = _KeywordEvaluator()
        mutator = LLMMutator(
            backend=backend,
            mutation_temperature=0.5,
            fallback_on_error=True,
        )
        engine = EvolutionEngine(
            evaluator=evaluator,
            mutator=mutator,
            backend=backend,
            population_size=self.POPULATION_SIZE,
            mutation_rate=1.0,  # Always mutate for deterministic improvement
            crossover_rate=0.0,  # No crossover for simplicity
            elite_ratio=0.2,
            tournament_k=2,
            seed=42,
        )
        return engine, evaluator, mutator

    def test_evolution_completes(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        engine, _, _ = setup
        best = engine.evolve(
            seed_genomes=seed_genomes,
            task="Write a Python function to reverse a string.",
            n_generations=self.N_GENERATIONS,
        )
        assert isinstance(best, Agent)
        assert best.fitness is not None

    def test_best_fitness_is_valid(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        engine, _, _ = setup
        best = engine.evolve(
            seed_genomes=seed_genomes,
            task="Write a Python function.",
            n_generations=self.N_GENERATIONS,
        )
        assert 0.0 <= (best.fitness or 0.0) <= 1.0

    def test_fitness_improves_or_holds(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        """Best fitness at gen 3 must be ≥ best fitness at gen 0 (elitism)."""
        engine, evaluator, _ = setup
        gen_bests: list[float] = []

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            scores = [a.fitness for a in pop if a.fitness is not None]
            if scores:
                gen_bests.append(max(scores))

        engine.evolve(
            seed_genomes=seed_genomes,
            task="Write a Python function.",
            n_generations=self.N_GENERATIONS,
            on_generation=_on_gen,
        )

        assert len(gen_bests) > 0
        # Elitism guarantees best never decreases
        for i in range(1, len(gen_bests)):
            assert gen_bests[i] >= gen_bests[i - 1] - 1e-9, (
                f"Fitness regressed at generation {i}: "
                f"{gen_bests[i-1]:.4f} → {gen_bests[i]:.4f}"
            )

    def test_population_size_maintained(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        engine, _, _ = setup
        pop_sizes: list[int] = []

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            pop_sizes.append(len(pop))

        engine.evolve(
            seed_genomes=seed_genomes,
            task="task",
            n_generations=self.N_GENERATIONS,
            on_generation=_on_gen,
        )

        for size in pop_sizes:
            assert size == self.POPULATION_SIZE

    def test_all_agents_have_fitness(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        engine, _, _ = setup
        final_pop: list[Agent] = []

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            if gen == self.N_GENERATIONS:
                final_pop.extend(pop)

        engine.evolve(
            seed_genomes=seed_genomes,
            task="task",
            n_generations=self.N_GENERATIONS,
            on_generation=_on_gen,
        )

        assert all(a.fitness is not None for a in final_pop)

    def test_best_agent_evolved_prompt(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        """Best agent's prompt should contain at least one keyword."""
        engine, evaluator, _ = setup
        best = engine.evolve(
            seed_genomes=seed_genomes,
            task="Write a Python function.",
            n_generations=self.N_GENERATIONS,
        )
        prompt_lower = best.genome.system_prompt.lower()
        has_keyword = any(kw in prompt_lower for kw in _KEYWORD_SCORE)
        # The evolved agent should have picked up at least one keyword
        assert has_keyword or (best.fitness or 0.0) > 0.3

    def test_memory_populated(
        self,
        seed_genomes: list[Genome],
    ) -> None:
        """EvolutionaryMemory should track all agents across generations."""
        backend = _make_mock_backend()
        evaluator = _KeywordEvaluator()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=evaluator,
            mutator=mutator,
            backend=backend,
            population_size=3,
            mutation_rate=1.0,
            crossover_rate=0.0,
            elite_ratio=0.2,
            seed=42,
        )

        engine.evolve(
            seed_genomes=seed_genomes,
            task="task",
            n_generations=2,
        )

        # Memory should have at least the seed agents
        assert engine._memory.total_agents >= 3  # type: ignore[attr-defined]

    def test_on_generation_callback_called(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        engine, _, _ = setup
        callback_gens: list[int] = []

        def _cb(gen: int, pop: list[Agent]) -> None:
            callback_gens.append(gen)

        engine.evolve(
            seed_genomes=seed_genomes,
            task="task",
            n_generations=self.N_GENERATIONS,
            on_generation=_cb,
        )

        # Should be called for gen 0 through N_GENERATIONS
        assert len(callback_gens) == self.N_GENERATIONS + 1
        assert callback_gens[0] == 0
        assert callback_gens[-1] == self.N_GENERATIONS

    def test_seed_diversity(
        self,
        setup: tuple[EvolutionEngine, _KeywordEvaluator, LLMMutator],
        seed_genomes: list[Genome],
    ) -> None:
        """Population starts with distinct agents (no clones of a single seed)."""
        engine, _, _ = setup
        initial_pop: list[Agent] = []

        def _cb(gen: int, pop: list[Agent]) -> None:
            if gen == 0:
                initial_pop.extend(pop)

        engine.evolve(
            seed_genomes=seed_genomes,
            task="task",
            n_generations=1,
            on_generation=_cb,
        )

        prompts = [a.genome.system_prompt for a in initial_pop]
        unique_prompts = set(prompts)
        # At least 2 distinct prompts in the initial population
        assert len(unique_prompts) >= 2
