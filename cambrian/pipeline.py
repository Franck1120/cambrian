"""Pipeline — multi-step agent pipeline for Forge mode.

A :class:`Pipeline` is an ordered list of :class:`PipelineStep` objects.
Each step has a system prompt and role.  Input flows through all steps
sequentially; the final output is scored by the evaluator.

The :class:`PipelineMutator` asks the LLM to add, remove, or reorder steps.
The :class:`PipelineEvaluator` runs the pipeline and scores the final output.
The :class:`PipelineEvolutionEngine` runs the evolutionary loop.

Example::

    from cambrian.pipeline import Pipeline, PipelineStep, PipelineEvolutionEngine
    from cambrian.backends.openai_compat import OpenAICompatBackend
    from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    engine = PipelineEvolutionEngine(backend=backend, population_size=4)
    seed = Pipeline(
        description="Summarise a news article",
        steps=[
            PipelineStep(role="extractor", system_prompt="Extract the key facts."),
            PipelineStep(role="summariser", system_prompt="Write a 3-sentence summary."),
        ],
    )
    best = engine.evolve(seed, task="Summarise this article: ...", n_generations=6)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ── PipelineStep ───────────────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    """A single step in a :class:`Pipeline`.

    Attributes:
        role: Human-readable name for this step (e.g. ``"extractor"``).
        system_prompt: System-level instructions for the LLM at this step.
        temperature: Sampling temperature for this step's LLM call.
        step_id: Unique identifier, auto-generated if not supplied.
    """

    role: str
    system_prompt: str
    temperature: float = 0.7
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:6])

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "step_id": self.step_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        return cls(
            role=str(data.get("role", "step")),
            system_prompt=str(data.get("system_prompt", "")),
            temperature=float(data.get("temperature", 0.7)),
            step_id=str(data.get("step_id", str(uuid.uuid4())[:6])),
        )

    def __repr__(self) -> str:
        return f"PipelineStep(role={self.role!r}, temp={self.temperature})"


# ── Pipeline ───────────────────────────────────────────────────────────────────


@dataclass
class Pipeline:
    """An evolvable ordered list of :class:`PipelineStep` objects.

    Attributes:
        steps: Ordered list of steps. Input → step[0] → step[1] → … → output.
        description: Natural-language description of the pipeline's purpose.
        version: Mutation counter (incremented on each mutation).
        pipeline_id: Unique identifier.
        parent_id: ID of parent pipeline (empty for seed).
        metadata: Arbitrary extra data.
    """

    steps: list[PipelineStep] = field(default_factory=list)
    description: str = ""
    version: int = 0
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def step_count(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
            "version": self.version,
            "pipeline_id": self.pipeline_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        steps = [
            PipelineStep.from_dict(s)
            for s in data.get("steps", [])
            if isinstance(s, dict)
        ]
        return cls(
            steps=steps,
            description=str(data.get("description", "")),
            version=int(data.get("version", 0)),
            pipeline_id=str(data.get("pipeline_id", str(uuid.uuid4())[:8])),
            parent_id=str(data.get("parent_id", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def __str__(self) -> str:
        roles = " → ".join(s.role for s in self.steps)
        return (
            f"Pipeline(id={self.pipeline_id}, v={self.version}, "
            f"steps={self.step_count()}, [{roles}])"
        )


# ── PipelineRunner ─────────────────────────────────────────────────────────────


class PipelineRunner:
    """Executes a :class:`Pipeline` end-to-end using a backend.

    Args:
        backend: LLM backend used for each step.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def run(self, pipeline: Pipeline, task: str) -> str:
        """Run *task* through all steps sequentially.

        The output of each step becomes the input for the next step.
        The first step receives the original task.

        Args:
            pipeline: The pipeline to execute.
            task: Initial input task/text.

        Returns:
            The output of the final step, or empty string if no steps.
        """
        if not pipeline.steps:
            return ""

        current_input = task
        for step in pipeline.steps:
            current_input = self._backend.generate(
                current_input,
                system=step.system_prompt,
                temperature=step.temperature,
            )
        return current_input


# ── PipelineEvaluator ──────────────────────────────────────────────────────────


class PipelineEvaluator:
    """Evaluates a :class:`Pipeline` by running it and scoring the output.

    The scoring function is a callable ``(output: str, task: str) -> float``.
    You can pass an :class:`~cambrian.evaluator.Evaluator`-like object or a
    plain function.

    Args:
        backend: LLM backend used to execute the pipeline steps.
        score_fn: Callable ``(output, task) -> float`` in range [0, 1].
        empty_pipeline_penalty: Fitness returned when pipeline has no steps.
    """

    def __init__(
        self,
        backend: LLMBackend,
        score_fn: "Any",
        empty_pipeline_penalty: float = 0.0,
    ) -> None:
        self._runner = PipelineRunner(backend)
        self._score_fn = score_fn
        self._empty_penalty = empty_pipeline_penalty

    def evaluate(self, pipeline: Pipeline, task: str) -> float:
        """Run *pipeline* on *task* and return a fitness in [0, 1].

        Args:
            pipeline: The pipeline to evaluate.
            task: The task input.

        Returns:
            Fitness score in [0, 1].  Returns ``empty_pipeline_penalty`` if
            the pipeline has no steps.
        """
        if not pipeline.steps:
            return self._empty_penalty

        try:
            output = self._runner.run(pipeline, task)
            score = self._score_fn(output, task)
            return float(max(0.0, min(1.0, score)))
        except Exception as exc:
            logger.warning("PipelineEvaluator error: %s", exc)
            return 0.0


# ── PipelineMutator ────────────────────────────────────────────────────────────


_PIPELINE_MUTATE_SYSTEM = """You are an expert AI pipeline architect.
Your task: improve a multi-step AI pipeline by modifying its steps.
You may ADD new steps, REMOVE ineffective steps, or REORDER steps.
Return ONLY a valid JSON object representing the improved pipeline."""

_PIPELINE_MUTATE_TEMPLATE = """Current pipeline:
{pipeline_json}

Task description: {description}
Current fitness: {fitness:.4f}

Improve this pipeline. You may:
1. ADD a new step (insert at any position) with a clear role and system_prompt
2. REMOVE a step that adds no value
3. REORDER steps to improve the flow
4. MODIFY a step's system_prompt to be more effective
5. ADJUST temperature on any step (0.1 = deterministic, 1.2 = creative)

Rules:
- Keep between 1 and 6 steps (no empty pipelines, no bloated chains)
- Each step must have a unique role name
- Each system_prompt must be actionable and specific
- Return ONLY the JSON object with the same structure, no explanation

Return ONLY the JSON."""

_PIPELINE_SEED_TEMPLATE = """Design a multi-step AI pipeline to solve this task:

{description}

Create 2–4 pipeline steps. Each step should:
- Have a clear, distinct role (e.g. "extractor", "analyser", "formatter")
- Have a focused system_prompt that describes exactly what this step does

Return ONLY a JSON object with this structure:
{{
  "steps": [
    {{"role": "step_name", "system_prompt": "...", "temperature": 0.7}},
    ...
  ],
  "description": "{description}"
}}"""


class PipelineMutator:
    """LLM-guided mutator for :class:`Pipeline` objects.

    Args:
        backend: LLM backend for generating mutations.
        mutation_temperature: Sampling temperature. Default 0.6.
        fallback_on_error: Return original pipeline on errors. Default True.
    """

    def __init__(
        self,
        backend: LLMBackend,
        mutation_temperature: float = 0.6,
        fallback_on_error: bool = True,
    ) -> None:
        self._backend = backend
        self._temp = mutation_temperature
        self._fallback = fallback_on_error

    def mutate(
        self,
        pipeline: Pipeline,
        fitness: float = 0.0,
    ) -> Pipeline:
        """Return a new :class:`Pipeline` with LLM-mutated steps.

        If the pipeline has no steps, seeds a fresh one from the description.

        Args:
            pipeline: The pipeline to mutate.
            fitness: Most recent fitness score, used to guide the LLM.

        Returns:
            A new Pipeline with incremented version.
        """
        if not pipeline.steps:
            return self._seed(pipeline)

        prompt = _PIPELINE_MUTATE_TEMPLATE.format(
            pipeline_json=json.dumps(pipeline.to_dict(), indent=2),
            description=pipeline.description or "general task",
            fitness=fitness,
        )
        try:
            raw = self._backend.generate(
                prompt,
                system=_PIPELINE_MUTATE_SYSTEM,
                temperature=self._temp,
            )
            child_data = self._parse_pipeline_json(raw, pipeline)
        except Exception as exc:
            logger.warning("PipelineMutator.mutate failed: %s", exc)
            if not self._fallback:
                raise
            child_data = pipeline.to_dict()

        child = Pipeline.from_dict(child_data)
        child.version = pipeline.version + 1
        child.pipeline_id = str(uuid.uuid4())[:8]
        child.parent_id = pipeline.pipeline_id
        child.description = pipeline.description
        return child

    def crossover(self, parent_a: Pipeline, parent_b: Pipeline) -> Pipeline:
        """Combine step lists from two parent pipelines.

        Interleaves steps from both parents (alternating) and deduplicates by
        role name, keeping the first occurrence.

        Args:
            parent_a: First parent pipeline.
            parent_b: Second parent pipeline.

        Returns:
            A new Pipeline with combined steps.
        """
        combined: list[PipelineStep] = []
        seen_roles: set[str] = set()
        # Interleave steps from both parents
        for a_step, b_step in zip(parent_a.steps, parent_b.steps):
            if a_step.role not in seen_roles:
                combined.append(a_step)
                seen_roles.add(a_step.role)
            if b_step.role not in seen_roles:
                combined.append(b_step)
                seen_roles.add(b_step.role)

        # Add remaining steps from the longer parent
        for step in parent_a.steps[len(parent_b.steps):]:
            if step.role not in seen_roles:
                combined.append(step)
                seen_roles.add(step.role)
        for step in parent_b.steps[len(parent_a.steps):]:
            if step.role not in seen_roles:
                combined.append(step)
                seen_roles.add(step.role)

        # Keep at most 6 steps
        combined = combined[:6]

        child = Pipeline(
            steps=combined,
            description=parent_a.description,
            version=max(parent_a.version, parent_b.version) + 1,
            pipeline_id=str(uuid.uuid4())[:8],
            parent_id=parent_a.pipeline_id,
        )
        return child

    def _seed(self, pipeline: Pipeline) -> Pipeline:
        """Generate initial steps from pipeline description."""
        prompt = _PIPELINE_SEED_TEMPLATE.format(
            description=pipeline.description or "general AI task"
        )
        try:
            raw = self._backend.generate(
                prompt,
                system=_PIPELINE_MUTATE_SYSTEM,
                temperature=self._temp,
            )
            data = self._parse_pipeline_json(raw, pipeline)
            child = Pipeline.from_dict(data)
        except Exception as exc:
            logger.warning("PipelineMutator._seed failed: %s", exc)
            if not self._fallback:
                raise
            child = Pipeline(
                steps=[
                    PipelineStep(
                        role="processor",
                        system_prompt="Process the input carefully and produce the best output.",
                    )
                ],
                description=pipeline.description,
            )
        child.version = 1
        child.pipeline_id = str(uuid.uuid4())[:8]
        child.parent_id = pipeline.pipeline_id
        child.description = pipeline.description
        return child

    @staticmethod
    def _parse_pipeline_json(raw: str, fallback: Pipeline) -> dict[str, Any]:
        """Extract JSON from LLM response, falling back to original pipeline."""
        # Try to strip markdown fences
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            raw = match.group(1)
        try:
            data = json.loads(raw.strip())
            # Validate minimal structure
            if "steps" in data and isinstance(data["steps"], list):
                return data  # type: ignore[return-value]
        except (json.JSONDecodeError, TypeError):
            pass
        return fallback.to_dict()


# ── PipelineEvolutionEngine ────────────────────────────────────────────────────


class PipelineEvolutionEngine:
    """Evolutionary loop specialised for :class:`Pipeline` objects.

    Each generation:
    1. Evaluate all pipelines using :class:`PipelineEvaluator`.
    2. Keep top ``elite_n`` unchanged.
    3. Mutate the rest with :class:`PipelineMutator`.
    4. Occasionally crossover two parent pipelines.

    Args:
        backend: LLM backend for running and mutating pipelines.
        score_fn: Callable ``(output, task) -> float`` for scoring pipeline output.
            Required unless ``evaluator`` is passed.
        evaluator: Optional pre-built :class:`PipelineEvaluator`.
        population_size: Number of pipelines per generation. Default 6.
        elite_ratio: Fraction of top pipelines carried unchanged. Default 0.33.
        crossover_rate: Fraction of offspring produced via crossover. Default 0.2.
        mutation_temperature: Temperature for :class:`PipelineMutator`. Default 0.6.
    """

    def __init__(
        self,
        backend: LLMBackend,
        score_fn: "Any | None" = None,
        evaluator: "PipelineEvaluator | None" = None,
        population_size: int = 6,
        elite_ratio: float = 0.33,
        crossover_rate: float = 0.2,
        mutation_temperature: float = 0.6,
    ) -> None:
        self._backend = backend
        self._score_fn = score_fn
        self._evaluator = evaluator
        self._pop_size = population_size
        self._elite_n = max(1, int(population_size * elite_ratio))
        self._crossover_rate = crossover_rate
        self._mutator = PipelineMutator(
            backend=backend, mutation_temperature=mutation_temperature
        )

    def evolve(
        self,
        seed: Pipeline,
        task: str,
        n_generations: int = 6,
        on_generation: "Any | None" = None,
    ) -> Pipeline:
        """Run the evolutionary loop and return the best :class:`Pipeline`.

        Args:
            seed: Initial pipeline (steps may be empty; description should be set).
            task: The task to evaluate pipelines on.
            n_generations: Number of generations. Default 6.
            on_generation: Optional callback ``(gen, pop_with_fitness, best) -> None``.

        Returns:
            The Pipeline with the highest fitness found.
        """
        evaluator = self._evaluator
        if evaluator is None:
            if self._score_fn is None:
                raise ValueError(
                    "Provide score_fn= or pass a PipelineEvaluator at construction time."
                )
            evaluator = PipelineEvaluator(self._backend, self._score_fn)

        if not seed.description:
            seed = Pipeline(
                steps=seed.steps,
                description=task,
                pipeline_id=seed.pipeline_id,
            )

        # Seed population
        population = [seed]
        while len(population) < self._pop_size:
            population.append(self._mutator.mutate(seed))

        best_pipeline = seed
        best_fitness = -1.0

        import random as _rnd

        for gen in range(n_generations):
            # Evaluate all
            scored: list[tuple[Pipeline, float]] = []
            for pipeline in population:
                score = evaluator.evaluate(pipeline, task)
                scored.append((pipeline, score))
                logger.debug(
                    "gen=%d id=%s fitness=%.4f steps=%d",
                    gen,
                    pipeline.pipeline_id,
                    score,
                    pipeline.step_count(),
                )
                if score > best_fitness:
                    best_fitness = score
                    best_pipeline = pipeline

            scored.sort(key=lambda x: x[1], reverse=True)

            if on_generation is not None:
                on_generation(gen, scored, best_pipeline)

            if best_fitness >= 1.0:
                break

            elites = [p for p, _ in scored[: self._elite_n]]

            new_pop: list[Pipeline] = list(elites)
            remaining = scored[self._elite_n :]

            for pipeline, fitness in remaining:
                if (
                    len(new_pop) < self._pop_size
                    and _rnd.random() < self._crossover_rate
                    and len(elites) >= 2
                ):
                    parent_b = _rnd.choice(elites[1:])
                    child = self._mutator.crossover(elites[0], parent_b)
                else:
                    child = self._mutator.mutate(pipeline, fitness=fitness)
                new_pop.append(child)

            population = new_pop[: self._pop_size]

        return best_pipeline
