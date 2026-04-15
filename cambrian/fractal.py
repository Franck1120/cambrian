# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""cambrian/fractal.py — Recursive multi-scale fractal evolution.

Implements self-similar evolutionary search across multiple genome granularities:

    Scale 0 (macro)  — strategy + high-level structure
    Scale 1 (meso)   — paragraph / section level
    Scale 2 (micro)  — word / phrase level

Each scale runs an independent sub-population. Results propagate up (seed
coarser scales) and down (constrain finer scales), creating self-similar
evolutionary pressure across resolutions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.evaluator import Evaluator

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Enums & configs
# ---------------------------------------------------------------------------


class FractalScale(int, Enum):
    """Granularity level for fractal evolution.

    Each scale targets a different scope of the genome:

    * ``MACRO`` — overall strategy and high-level prompt structure.
    * ``MESO``  — individual paragraphs / sentence blocks.
    * ``MICRO`` — specific words and short phrases.
    """

    MACRO = 0
    MESO = 1
    MICRO = 2


@dataclass
class ScaleConfig:
    """Configuration for a single fractal scale.

    Attributes:
        scale: Which :class:`FractalScale` this config applies to.
        population_size: Number of agents maintained at this scale.
        n_generations: Evolutionary generations run per call to
            :meth:`FractalPopulation.evolve_step`.
        mutation_temperature: LLM sampling temperature used during mutation.
            Lower values = conservative rewrites; higher = creative.
        fragment_size: Character budget for the scope targeted by mutation
            (500 for macro, 100 for meso, 20 for micro).
    """

    scale: FractalScale
    population_size: int = 4
    n_generations: int = 3
    mutation_temperature: float = field(default=0.5)
    fragment_size: int = field(default=100)

    def __post_init__(self) -> None:
        """Apply scale-specific defaults when none were provided explicitly."""
        _temp_defaults: dict[FractalScale, float] = {
            FractalScale.MACRO: 0.7,
            FractalScale.MESO: 0.5,
            FractalScale.MICRO: 0.3,
        }
        _fragment_defaults: dict[FractalScale, int] = {
            FractalScale.MACRO: 500,
            FractalScale.MESO: 100,
            FractalScale.MICRO: 20,
        }
        # Only override if the caller left the dataclass at its default.
        # We detect "at default" by checking against the dataclass default;
        # the simplest approach is to always set from the per-scale map here.
        self.mutation_temperature = _temp_defaults[self.scale]
        self.fragment_size = _fragment_defaults[self.scale]


@dataclass
class FractalResult:
    """Result produced by a single fractal-scale evolution run.

    Attributes:
        scale: The :class:`FractalScale` that produced this result.
        best_genome: Genome with the highest fitness found.
        best_fitness: Fitness of :attr:`best_genome`; always ≥ 0.0.
        n_evaluations: Total evaluator calls made during this run.
    """

    scale: FractalScale
    best_genome: Genome
    best_fitness: float
    n_evaluations: int

    def __post_init__(self) -> None:
        if self.best_fitness < 0.0:
            self.best_fitness = 0.0


# ---------------------------------------------------------------------------
# FractalMutator
# ---------------------------------------------------------------------------


class FractalMutator:
    """LLM-guided mutations at each fractal scale.

    Args:
        backend: LLM backend used to generate rewrites.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Scale-specific mutation methods
    # ------------------------------------------------------------------

    def mutate_macro(self, genome: Genome, task: str) -> Genome:
        """Rewrite the overall strategy and high-level prompt structure.

        The LLM is asked to improve the full system prompt with a focus on
        strategy and approach.  Falls back to the original genome on any
        backend error.

        Args:
            genome: Source genome to mutate.
            task: Natural-language description of the task being evolved for.

        Returns:
            A new :class:`Genome` with an improved ``system_prompt``, or the
            original genome unchanged if the backend raises.
        """
        prompt = (
            "Improve the overall strategy and approach of this system prompt "
            "for the task. Return only the improved system prompt.\n\n"
            f"Task: {task}\n\n"
            f"Current system prompt:\n{genome.system_prompt}"
        )
        try:
            new_prompt = self._backend.generate(prompt)
            return Genome(
                system_prompt=new_prompt.strip() or genome.system_prompt,
                temperature=genome.temperature,
                strategy=genome.strategy,
                model=genome.model,
            )
        except Exception:
            return genome

    def mutate_meso(self, genome: Genome, task: str) -> Genome:
        """Rewrite one randomly chosen paragraph/sentence block.

        Splits the system prompt on ``\\n\\n``, picks a random chunk, rewrites
        it, and reassembles the prompt.  Falls back to the original genome on
        any backend error.

        Args:
            genome: Source genome to mutate.
            task: Natural-language description of the task being evolved for.

        Returns:
            A new :class:`Genome` with one paragraph rewritten, or the original
            genome unchanged if the backend raises.
        """
        chunks = genome.system_prompt.split("\n\n")
        if not chunks:
            return genome

        idx = random.randrange(len(chunks))
        target_chunk = chunks[idx]

        prompt = (
            "Rewrite the following paragraph to be clearer and more effective "
            f"for the task: {task}\n\n"
            f"Paragraph:\n{target_chunk}\n\n"
            "Return only the rewritten paragraph."
        )
        try:
            new_chunk = self._backend.generate(prompt)
            chunks[idx] = new_chunk.strip() or target_chunk
            new_prompt = "\n\n".join(chunks)
            return Genome(
                system_prompt=new_prompt,
                temperature=genome.temperature,
                strategy=genome.strategy,
                model=genome.model,
            )
        except Exception:
            return genome

    def mutate_micro(self, genome: Genome, task: str) -> Genome:
        """Rewrite a short phrase extracted from the system prompt.

        Takes the first ``fragment_size=20`` characters of a randomly chosen
        word, asks the LLM to rephrase it in context, and replaces the original
        fragment.  Falls back to the original genome on any backend error.

        Args:
            genome: Source genome to mutate.
            task: Natural-language description of the task being evolved for.

        Returns:
            A new :class:`Genome` with a short phrase rewritten, or the original
            genome unchanged if the backend raises.
        """
        fragment_size = 20
        words = genome.system_prompt.split()
        if not words:
            return genome

        word = random.choice(words)
        fragment = word[:fragment_size]

        prompt = (
            f"Rephrase the following word or phrase in the context of this task: {task}\n\n"
            f"Phrase: {fragment}\n\n"
            "Return only the rephrased word or phrase."
        )
        try:
            new_fragment = self._backend.generate(prompt)
            new_fragment = new_fragment.strip() or fragment
            new_prompt = genome.system_prompt.replace(fragment, new_fragment, 1)
            return Genome(
                system_prompt=new_prompt,
                temperature=genome.temperature,
                strategy=genome.strategy,
                model=genome.model,
            )
        except Exception:
            return genome

    def mutate(self, genome: Genome, task: str, scale: FractalScale) -> Genome:
        """Dispatch mutation to the appropriate scale-specific method.

        Args:
            genome: Source genome to mutate.
            task: Natural-language description of the task being evolved for.
            scale: Which :class:`FractalScale` to apply.

        Returns:
            Mutated :class:`Genome`.
        """
        if scale is FractalScale.MACRO:
            return self.mutate_macro(genome, task)
        if scale is FractalScale.MESO:
            return self.mutate_meso(genome, task)
        return self.mutate_micro(genome, task)


# ---------------------------------------------------------------------------
# FractalPopulation
# ---------------------------------------------------------------------------

_STRATEGY_BY_SCALE: dict[FractalScale, str] = {
    FractalScale.MACRO: "chain-of-thought",
    FractalScale.MESO: "step-by-step",
    FractalScale.MICRO: "direct",
}

_TEMP_BY_SCALE: dict[FractalScale, float] = {
    FractalScale.MACRO: 0.7,
    FractalScale.MESO: 0.5,
    FractalScale.MICRO: 0.3,
}


class FractalPopulation:
    """An independent sub-population operating at a single fractal scale.

    Args:
        scale: The :class:`FractalScale` this population is responsible for.
        config: :class:`ScaleConfig` parameterising this scale.
        evaluator: :class:`~cambrian.evaluator.Evaluator` used to score agents.
        mutator: :class:`FractalMutator` used to produce offspring.
    """

    def __init__(
        self,
        scale: FractalScale,
        config: ScaleConfig,
        evaluator: Evaluator,
        mutator: FractalMutator,
    ) -> None:
        self._scale = scale
        self._config = config
        self._evaluator = evaluator
        self._mutator = mutator
        self._agents: list[Agent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, genome: Genome) -> None:
        """Initialise the population from *genome*.

        The seed genome is used directly as the first agent; the remaining
        ``population_size - 1`` agents are temperature/strategy variants.

        Args:
            genome: Starting genome for this scale.
        """
        self._agents = []
        # First agent is the seed itself.
        self._agents.append(Agent(genome=genome))

        # Fill the rest with variants.
        temperature = _TEMP_BY_SCALE[self._scale]
        strategy = _STRATEGY_BY_SCALE[self._scale]

        for _ in range(self._config.population_size - 1):
            variant_genome = Genome(
                system_prompt=genome.system_prompt,
                temperature=temperature,
                strategy=strategy,
                model=genome.model,
            )
            self._agents.append(Agent(genome=variant_genome))

    def best_agent(self) -> Agent | None:
        """Return the agent with the highest fitness, or ``None`` before seeding.

        Returns:
            Best :class:`~cambrian.agent.Agent` or ``None``.
        """
        if not self._agents:
            return None
        scored = [a for a in self._agents if a.fitness is not None]
        if not scored:
            return self._agents[0]
        return max(scored, key=lambda a: a.fitness or 0.0)

    def evolve_step(self, task: str) -> Agent:
        """Run one generation of evolution.

        Steps:
        1. Evaluate all agents.
        2. Sort by fitness (descending).
        3. Keep the top half.
        4. Mutate survivors to fill the population back up.
        5. Return the current best agent.

        Args:
            task: Natural-language task description.

        Returns:
            The best :class:`~cambrian.agent.Agent` after this generation.
        """
        # 1. Evaluate all agents.
        for agent in self._agents:
            score = self._evaluator.evaluate(agent, task)
            agent.fitness = score

        # 2. Sort descending by fitness.
        self._agents.sort(key=lambda a: a.fitness or 0.0, reverse=True)

        # 3. Keep the top half (at least 1).
        keep = max(1, len(self._agents) // 2)
        survivors = self._agents[:keep]

        # 4. Mutate to fill back to population_size.
        offspring: list[Agent] = list(survivors)
        while len(offspring) < self._config.population_size:
            parent = random.choice(survivors)
            mutated_genome = self._mutator.mutate(parent.genome, task, self._scale)
            offspring.append(Agent(genome=mutated_genome))

        self._agents = offspring[: self._config.population_size]

        best = self.best_agent()
        # best_agent() returns None only when _agents is empty, which cannot
        # happen here (we just filled the list to population_size ≥ 1).
        assert best is not None
        return best


# ---------------------------------------------------------------------------
# FractalEvolution
# ---------------------------------------------------------------------------


class FractalEvolution:
    """Orchestrates multi-scale fractal evolution across MACRO, MESO, MICRO.

    Each call to :meth:`evolve` runs ``n_cycles`` of coarse-to-fine evolution:
    MACRO → MESO → MICRO.  The best genome from MICRO becomes the seed for the
    next cycle.  After all cycles the best :class:`FractalResult` across all
    scales and cycles is returned.

    Args:
        backend: LLM backend forwarded to :class:`FractalMutator`.
        evaluator: Fitness evaluator.
        macro_config: Optional custom config for the MACRO scale.
        meso_config: Optional custom config for the MESO scale.
        micro_config: Optional custom config for the MICRO scale.
    """

    def __init__(
        self,
        backend: LLMBackend,
        evaluator: Evaluator,
        macro_config: ScaleConfig | None = None,
        meso_config: ScaleConfig | None = None,
        micro_config: ScaleConfig | None = None,
    ) -> None:
        self._backend = backend
        self._evaluator = evaluator
        self._mutator = FractalMutator(backend)

        self._configs: dict[FractalScale, ScaleConfig] = {
            FractalScale.MACRO: macro_config or ScaleConfig(scale=FractalScale.MACRO),
            FractalScale.MESO: meso_config or ScaleConfig(scale=FractalScale.MESO),
            FractalScale.MICRO: micro_config or ScaleConfig(scale=FractalScale.MICRO),
        }

        self._results: list[FractalResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def results(self) -> list[FractalResult]:
        """All :class:`FractalResult` objects collected so far."""
        return list(self._results)

    def evolve(
        self,
        seed_genome: Genome,
        task: str,
        n_cycles: int = 2,
    ) -> FractalResult:
        """Run fractal evolution and return the best result.

        For each cycle:
        1. Evolve the MACRO population; use best genome as seed for next scale.
        2. Evolve the MESO population with the MACRO best as seed.
        3. Evolve the MICRO population with the MESO best as seed.
        4. The MICRO best becomes the new seed genome for the next cycle.

        The best :class:`FractalResult` across all cycles and scales is
        returned and appended to :attr:`results`.

        Args:
            seed_genome: Starting genome for the first cycle.
            task: Natural-language task description.
            n_cycles: Number of coarse-to-fine cycles to run.

        Returns:
            Best :class:`FractalResult` found across all cycles and scales.
        """
        current_seed = seed_genome
        cycle_results: list[FractalResult] = []

        for _cycle in range(n_cycles):
            cycle_seed = current_seed

            scale_order: list[FractalScale] = [
                FractalScale.MACRO,
                FractalScale.MESO,
                FractalScale.MICRO,
            ]

            scale_best_genome = cycle_seed
            scale_best_fitness = 0.0

            for scale in scale_order:
                config = self._configs[scale]
                population = FractalPopulation(
                    scale=scale,
                    config=config,
                    evaluator=self._evaluator,
                    mutator=self._mutator,
                )
                population.seed(cycle_seed if scale is FractalScale.MACRO else scale_best_genome)

                n_evals = 0
                best_agent: Agent | None = None

                for _gen in range(config.n_generations):
                    best_agent = population.evolve_step(task)
                    n_evals += config.population_size

                if best_agent is None:
                    best_agent = population.best_agent()

                fitness = (best_agent.fitness or 0.0) if best_agent else 0.0
                genome = best_agent.genome if best_agent else cycle_seed

                result = FractalResult(
                    scale=scale,
                    best_genome=genome,
                    best_fitness=max(0.0, fitness),
                    n_evaluations=n_evals,
                )
                cycle_results.append(result)

                # Propagate best genome downward to finer scales.
                scale_best_genome = genome
                scale_best_fitness = result.best_fitness

            # MICRO best becomes seed for next cycle.
            current_seed = scale_best_genome
            _ = scale_best_fitness  # referenced to silence unused-var linters

        # Pick the best result across all cycles and scales.
        best_result = max(cycle_results, key=lambda r: r.best_fitness)
        self._results.append(best_result)
        return best_result
