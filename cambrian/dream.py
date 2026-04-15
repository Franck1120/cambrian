"""Dream Phase — LLM-guided recombination of evolution experiences.

The Dream Phase is an optional phase that runs every N generations.
It takes the N best experiences from the :class:`~cambrian.memory.EvolutionaryMemory`
(or a plain list of ``(task, response, score)`` tuples), recombines them
via LLM into hybrid "dream scenarios", and evaluates agents on those.

Agents that perform well on dream scenarios are boosted slightly — this
mimics the neuroscience idea that sleep/dreaming consolidates memory by
replaying and recombining past experiences.

Usage::

    from cambrian.dream import DreamPhase

    dream = DreamPhase(backend=backend, n_experiences=10, blend_weight=0.1)
    # Call once every N generations:
    dream.run(population=agents, task=task, evaluator=evaluator)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent
    from cambrian.evaluator import Evaluator

logger = get_logger(__name__)


# ── DreamScenario ─────────────────────────────────────────────────────────────


@dataclass
class DreamScenario:
    """A synthetic hybrid task created by recombining past experiences.

    Attributes:
        task: The dream task text (a blend of real experiences).
        source_scores: The fitness scores of the experiences that seeded it.
        blend_description: Human-readable description of how it was created.
    """

    task: str
    source_scores: list[float] = field(default_factory=list)
    blend_description: str = ""

    @property
    def expected_difficulty(self) -> float:
        """Mean source score (proxy for expected difficulty)."""
        if not self.source_scores:
            return 0.5
        return sum(self.source_scores) / len(self.source_scores)


# ── Experience ────────────────────────────────────────────────────────────────


@dataclass
class Experience:
    """A single evolution experience (task, response, score) triple.

    Args:
        task: The task given to the agent.
        response: The agent's response (may be empty for non-LLM evaluators).
        score: The fitness score achieved.
        genome_id: ID of the genome that produced this experience.
    """

    task: str
    response: str
    score: float
    genome_id: str = ""


# ── DreamPhase ────────────────────────────────────────────────────────────────


_DREAM_SYSTEM = """You are a creative AI trainer designing challenging hybrid test scenarios.
Your task: take multiple real agent experiences and blend them into a novel synthetic scenario
that tests similar skills but with new variation.
Return ONLY the scenario description as plain text — no headers, no explanation."""

_DREAM_BLEND_TEMPLATE = """Here are {n} real agent experiences, each with a task and fitness score:

{experience_block}

Create a NEW synthetic task that:
1. Combines the core challenges from these experiences
2. Introduces a novel twist or variation
3. Is clearly stated and unambiguous
4. Is similar in difficulty (expected fitness ≈ {expected_fitness:.2f})

Return ONLY the new task description as plain text."""


class DreamPhase:
    """Generates dream scenarios and boosts agents that handle them well.

    Args:
        backend: LLM backend used to generate dream scenarios.
        n_experiences: Number of experiences to sample per dream. Default 5.
        n_dreams: Number of dream scenarios to generate per run. Default 3.
        blend_weight: How much to blend dream fitness into agent fitness.
            Final fitness = (1 - blend_weight) * real_fitness + blend_weight * dream_fitness.
            Default 0.1.
        min_score_threshold: Only use experiences above this score as seeds.
            Default 0.4.
        rng_seed: Optional random seed.
    """

    def __init__(
        self,
        backend: LLMBackend,
        n_experiences: int = 5,
        n_dreams: int = 3,
        blend_weight: float = 0.1,
        min_score_threshold: float = 0.4,
        rng_seed: "int | None" = None,
    ) -> None:
        self._backend = backend
        self._n_exp = n_experiences
        self._n_dreams = n_dreams
        self._blend_weight = max(0.0, min(1.0, blend_weight))
        self._min_score = min_score_threshold
        self._rng = random.Random(rng_seed)
        self._last_dreams: list[DreamScenario] = []

    @property
    def last_dreams(self) -> list[DreamScenario]:
        """Dream scenarios generated in the last :meth:`run` call."""
        return list(self._last_dreams)

    def generate_scenario(self, experiences: list[Experience]) -> DreamScenario:
        """Generate a single dream scenario from a list of experiences.

        Args:
            experiences: Source experiences to blend (2+ recommended).

        Returns:
            A :class:`DreamScenario` with the blended task.
        """
        if not experiences:
            return DreamScenario(task="Solve a challenging problem.", source_scores=[])

        # Build experience block for the prompt
        lines = []
        for i, exp in enumerate(experiences, 1):
            lines.append(
                f"  [{i}] Task: {exp.task[:200]}\n"
                f"       Score: {exp.score:.3f}"
            )
        experience_block = "\n\n".join(lines)
        expected_fitness = sum(e.score for e in experiences) / len(experiences)

        prompt = _DREAM_BLEND_TEMPLATE.format(
            n=len(experiences),
            experience_block=experience_block,
            expected_fitness=expected_fitness,
        )
        try:
            dream_task = self._backend.generate(
                prompt,
                system=_DREAM_SYSTEM,
                temperature=0.9,  # High temperature for creative blending
            )
            dream_task = dream_task.strip()
        except Exception as exc:
            logger.warning("DreamPhase.generate_scenario failed: %s", exc)
            # Fallback: use the highest-scoring experience's task
            best = max(experiences, key=lambda e: e.score)
            dream_task = f"[Dream fallback] {best.task}"

        return DreamScenario(
            task=dream_task,
            source_scores=[e.score for e in experiences],
            blend_description=f"Blended from {len(experiences)} experiences",
        )

    def run(
        self,
        population: "list[Agent]",
        task: str,
        evaluator: "Evaluator",
        experiences: "list[Experience] | None" = None,
    ) -> "list[Agent]":
        """Run the dream phase on a population.

        Generates :attr:`n_dreams` dream scenarios, evaluates each agent on them,
        and blends the dream fitness into each agent's current fitness.

        Args:
            population: Current population of agents.
            task: The main task (used as fallback if no experiences).
            evaluator: Evaluator used to score agents on dream scenarios.
            experiences: List of past experiences to sample from.
                If ``None``, a single generic dream scenario is created from *task*.

        Returns:
            The same population list with updated fitness values.
        """
        if not population:
            return population

        # Build dream scenarios
        self._last_dreams = []
        if experiences and len(experiences) >= 2:
            # Filter to high-quality experiences
            good_exps = [e for e in experiences if e.score >= self._min_score]
            if not good_exps:
                good_exps = sorted(experiences, key=lambda e: e.score, reverse=True)[
                    : max(2, len(experiences) // 2)
                ]

            for _ in range(self._n_dreams):
                sample_size = min(self._n_exp, len(good_exps))
                sample = self._rng.sample(good_exps, sample_size)
                scenario = self.generate_scenario(sample)
                self._last_dreams.append(scenario)
        else:
            # No experiences: single generic dream
            scenario = DreamScenario(
                task=task,
                source_scores=[0.5],
                blend_description="No experiences — using main task",
            )
            self._last_dreams = [scenario]

        if not self._last_dreams:
            return population

        # Evaluate population on dream scenarios
        for agent in population:
            if agent.fitness is None:
                continue  # skip unevaluated agents

            dream_scores: list[float] = []
            for dream in self._last_dreams:
                try:
                    score = evaluator.evaluate(agent, dream.task)
                    dream_scores.append(float(score))
                except Exception as exc:
                    logger.warning(
                        "DreamPhase eval error for agent %s: %s", agent.id, exc
                    )

            if not dream_scores:
                continue

            dream_fitness = sum(dream_scores) / len(dream_scores)
            # Blend: preserve mostly real fitness, add small dream signal
            blended = (
                (1.0 - self._blend_weight) * agent.fitness
                + self._blend_weight * dream_fitness
            )
            agent.fitness = max(0.0, min(1.0, blended))
            logger.debug(
                "DreamPhase agent=%s dream_fitness=%.4f blended=%.4f",
                agent.id,
                dream_fitness,
                agent.fitness,
            )

        return population

    def extract_experiences_from_memory(
        self,
        memory: Any,
        task: str = "",
        limit: int = 50,
    ) -> list[Experience]:
        """Extract :class:`Experience` objects from an :class:`~cambrian.memory.EvolutionaryMemory`.

        Args:
            memory: An :class:`~cambrian.memory.EvolutionaryMemory` instance.
            task: Optional task filter for trace retrieval.
            limit: Maximum number of experiences to extract. Default 50.

        Returns:
            List of :class:`Experience` objects.
        """
        try:
            traces = memory.get_traces(task=task, limit=limit)
            return [
                Experience(
                    task=tr.content[:300],
                    response="",
                    score=tr.score,
                )
                for tr in traces
            ]
        except Exception as exc:
            logger.warning("extract_experiences_from_memory failed: %s", exc)
            return []
