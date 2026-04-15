"""Pipeline evolution — multi-step agent pipelines for Forge mode.

A :class:`Pipeline` is an ordered list of :class:`PipelineStep` objects.
Each step has its own system prompt, role, and temperature.  During evaluation,
the task input flows through every step in order; the final step's output is
scored.

A :class:`PipelineMutator` uses an LLM to add, remove, reorder, or rewrite
steps.  A :class:`PipelineEvaluator` orchestrates the sequential step execution
and returns a fitness score via an LLM judge.

A :class:`PipelineEvolutionEngine` wraps these into a full evolutionary loop
parallel to :class:`~cambrian.code_genome.CodeEvolutionEngine`.

Usage::

    from cambrian.pipeline import Pipeline, PipelineStep, PipelineEvolutionEngine
    from cambrian.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    seed = Pipeline(
        name="summariser",
        steps=[
            PipelineStep(name="planner", system_prompt="Plan how to summarise.", role="transformer"),
            PipelineStep(name="writer", system_prompt="Write a concise summary.", role="transformer"),
        ],
    )
    engine = PipelineEvolutionEngine(backend=backend, population_size=4)
    best = engine.evolve(seed=seed, task="Summarise the following text: ...", n_generations=5)
    print(best.name, best.version)
"""

from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    """A single step in an agent pipeline.

    Attributes:
        name: Human-readable step name.
        system_prompt: System instruction for this step's LLM call.
        role: Step role hint: ``"transformer"``, ``"extractor"``, or ``"validator"``.
        temperature: Sampling temperature for this step. Range ``[0.0, 2.0]``.
    """

    name: str = "step"
    system_prompt: str = "Process the input and produce output."
    role: str = "transformer"
    temperature: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "role": self.role,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        return cls(
            name=str(data.get("name", "step")),
            system_prompt=str(data.get("system_prompt", "Process the input.")),
            role=str(data.get("role", "transformer")),
            temperature=float(data.get("temperature", 0.7)),
        )

    def clone(self) -> "PipelineStep":
        return PipelineStep.from_dict(self.to_dict())

    def __repr__(self) -> str:
        preview = self.system_prompt[:40].replace("\n", " ")
        return f"PipelineStep(name={self.name!r}, role={self.role!r}, prompt={preview!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class Pipeline:
    """An ordered sequence of :class:`PipelineStep` objects.

    Args:
        name: Pipeline name (used in logs and output filenames).
        steps: Ordered list of pipeline steps.
        version: Monotonically increasing version counter.
        pipeline_id: Unique identifier, auto-generated if not supplied.
    """

    def __init__(
        self,
        name: str = "pipeline",
        steps: list[PipelineStep] | None = None,
        version: int = 0,
        pipeline_id: str | None = None,
    ) -> None:
        self.name = name
        self.steps: list[PipelineStep] = steps if steps is not None else []
        self.version = version
        self.pipeline_id: str = pipeline_id or str(uuid.uuid4())[:8]
        self._fitness: float | None = None

    @property
    def fitness(self) -> float | None:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "version": self.version,
            "pipeline_id": self.pipeline_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        steps = [PipelineStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=str(data.get("name", "pipeline")),
            steps=steps,
            version=int(data.get("version", 0)),
            pipeline_id=str(data.get("pipeline_id", str(uuid.uuid4())[:8])),
        )

    def clone(self) -> "Pipeline":
        """Return a deep copy with a fresh pipeline_id and no fitness."""
        p = Pipeline(
            name=self.name,
            steps=[s.clone() for s in self.steps],
            version=self.version,
            pipeline_id=None,
        )
        return p

    def __repr__(self) -> str:
        fit = f"{self._fitness:.4f}" if self._fitness is not None else "None"
        return (
            f"Pipeline(id={self.pipeline_id}, name={self.name!r}, "
            f"v{self.version}, steps={len(self.steps)}, fitness={fit})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PipelineMutator
# ─────────────────────────────────────────────────────────────────────────────

_MUTATE_SYSTEM = """\
You are an expert AI pipeline architect. You will be shown a multi-step
agent pipeline and a task description. Your job is to mutate the pipeline
to improve its performance on the task.

You may:
1. Add a new step (insert it in the most useful position)
2. Remove a step that adds no value
3. Reorder steps for better logical flow
4. Rewrite a step's system_prompt to be clearer or more focused
5. Change a step's role to "transformer", "extractor", or "validator"

Return the mutated pipeline as a JSON object with this schema:
{
  "name": "<pipeline name>",
  "steps": [
    {"name": "<step name>", "system_prompt": "<prompt>",
     "role": "<transformer|extractor|validator>", "temperature": <float>},
    ...
  ]
}

IMPORTANT: Return ONLY the JSON object. No explanation, no markdown fences.
"""

_MUTATE_TEMPLATE = """\
Task: {task}
Current pipeline (v{version}, fitness={fitness}):
{pipeline_json}

Mutate this pipeline to better solve the task.
"""

_CROSSOVER_SYSTEM = """\
You are an expert AI pipeline architect. Combine the best steps from two
pipelines to create a superior pipeline for the given task.

Return the merged pipeline as a JSON object with this schema:
{
  "name": "<pipeline name>",
  "steps": [
    {"name": "<step name>", "system_prompt": "<prompt>",
     "role": "<transformer|extractor|validator>", "temperature": <float>},
    ...
  ]
}

Return ONLY the JSON object.
"""

_CROSSOVER_TEMPLATE = """\
Task: {task}

Pipeline A (fitness={fitness_a}):
{pipeline_a}

Pipeline B (fitness={fitness_b}):
{pipeline_b}

Combine the strongest steps from both pipelines.
"""


class PipelineMutator:
    """LLM-guided pipeline mutation and crossover.

    Args:
        backend: LLM backend for generating pipeline modifications.
        temperature: Sampling temperature. Default ``0.7``.
        fallback_on_error: Return original pipeline on parse failure. Default ``True``.
        max_steps: Maximum steps allowed in a pipeline. Default ``8``.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        temperature: float = 0.7,
        fallback_on_error: bool = True,
        max_steps: int = 8,
    ) -> None:
        self._backend = backend
        self._temp = temperature
        self._fallback = fallback_on_error
        self._max_steps = max_steps

    def mutate(self, pipeline: Pipeline, task: str = "") -> Pipeline:
        """Return a mutated copy of *pipeline*.

        Args:
            pipeline: Source pipeline.
            task: Task description for the mutation prompt.

        Returns:
            New :class:`Pipeline` with modified steps.
        """
        import json as _json

        prompt = _MUTATE_TEMPLATE.format(
            task=task or "improve the pipeline",
            version=pipeline.version,
            fitness=(
                f"{pipeline.fitness:.4f}" if pipeline.fitness is not None else "not evaluated"
            ),
            pipeline_json=_json.dumps(pipeline.to_dict(), indent=2),
        )

        try:
            raw = self._backend.generate(prompt, system=_MUTATE_SYSTEM, temperature=self._temp)
            child = self._parse_pipeline(raw, pipeline)
        except Exception:
            if self._fallback:
                child = pipeline.clone()
            else:
                raise

        child.version = pipeline.version + 1
        child._fitness = None
        # Enforce step count limit
        if len(child.steps) > self._max_steps:
            child.steps = child.steps[: self._max_steps]
        # Ensure at least one step
        if not child.steps:
            child.steps = [s.clone() for s in pipeline.steps] or [PipelineStep()]
        return child

    def crossover(self, parent_a: Pipeline, parent_b: Pipeline, task: str = "") -> Pipeline:
        """Combine steps from two pipelines.

        Args:
            parent_a: First parent pipeline.
            parent_b: Second parent pipeline.
            task: Task description.

        Returns:
            New :class:`Pipeline` with combined steps.
        """
        import json as _json

        base = parent_a if (parent_a.fitness or 0.0) >= (parent_b.fitness or 0.0) else parent_b

        prompt = _CROSSOVER_TEMPLATE.format(
            task=task or "solve the problem",
            fitness_a=f"{parent_a.fitness:.4f}" if parent_a.fitness is not None else "0",
            pipeline_a=_json.dumps(parent_a.to_dict(), indent=2),
            fitness_b=f"{parent_b.fitness:.4f}" if parent_b.fitness is not None else "0",
            pipeline_b=_json.dumps(parent_b.to_dict(), indent=2),
        )

        try:
            raw = self._backend.generate(prompt, system=_CROSSOVER_SYSTEM, temperature=self._temp)
            child = self._parse_pipeline(raw, base)
        except Exception:
            if self._fallback:
                child = base.clone()
            else:
                raise

        child.version = max(parent_a.version, parent_b.version) + 1
        child._fitness = None
        return child

    @staticmethod
    def _parse_pipeline(raw: str, fallback: Pipeline) -> Pipeline:
        """Parse a JSON pipeline from an LLM response.

        Strips markdown fences if present, then attempts JSON decode.
        Falls back to *fallback*.clone() on any parse error.
        """
        import json as _json

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return fallback.clone()

        try:
            data = _json.loads(text[start:end])
            return Pipeline.from_dict(data)
        except Exception:
            return fallback.clone()


# ─────────────────────────────────────────────────────────────────────────────
# PipelineEvaluator
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an impartial judge evaluating how well an AI pipeline solved a task.
Score the output on a scale from 0.0 (completely wrong) to 1.0 (perfect).
Return ONLY a single float between 0.0 and 1.0. No explanation.
"""

_JUDGE_TEMPLATE = """\
Task: {task}

Pipeline output:
{output}

Score (0.0–1.0):"""


class PipelineEvaluator:
    """Runs a :class:`Pipeline` sequentially and scores the final output.

    Each step receives the previous step's output as the user message and
    uses its own system prompt.  The final output is judged by the LLM.

    Args:
        backend: LLM backend for running steps and the judge call.
        judge_temperature: Temperature for the judge call. Default ``0.0``.
        expected_output: Optional exact string; if set, use exact-match scoring
            instead of LLM-judge scoring.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        judge_temperature: float = 0.0,
        expected_output: str | None = None,
    ) -> None:
        self._backend = backend
        self._judge_temp = judge_temperature
        self._expected = expected_output

    def evaluate(self, pipeline: Pipeline, task: str) -> float:
        """Run *pipeline* on *task* and return a fitness in ``[0.0, 1.0]``.

        Args:
            pipeline: Pipeline to evaluate.
            task: User task / input for the first step.

        Returns:
            Fitness score.
        """
        if not pipeline.steps:
            return 0.0

        current_input = task
        for step in pipeline.steps:
            try:
                current_input = self._backend.generate(
                    current_input,
                    system=step.system_prompt,
                    temperature=step.temperature,
                )
            except Exception as exc:
                logger.warning("PipelineEvaluator step %r failed: %s", step.name, exc)
                return 0.1

        final_output = current_input

        if self._expected is not None:
            return 1.0 if final_output.strip() == self._expected.strip() else 0.0

        return self._judge(task, final_output)

    def _judge(self, task: str, output: str) -> float:
        prompt = _JUDGE_TEMPLATE.format(task=task, output=output)
        try:
            raw = self._backend.generate(
                prompt, system=_JUDGE_SYSTEM, temperature=self._judge_temp
            )
            score = float(re.search(r"-?[\d.]+", raw).group())  # type: ignore[union-attr]
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Neutral on parse failure


# ─────────────────────────────────────────────────────────────────────────────
# PipelineEvolutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class PipelineEvolutionEngine:
    """Evolutionary loop over :class:`Pipeline` populations.

    Args:
        backend: LLM backend for mutation, step execution, and judge calls.
        population_size: Number of pipelines per generation. Default ``6``.
        mutation_rate: Probability of mutating each non-elite pipeline. Default ``0.8``.
        crossover_rate: Probability of crossover vs direct mutation. Default ``0.3``.
        elite_ratio: Fraction of top pipelines preserved unchanged. Default ``0.2``.
        tournament_k: Tournament pool size. Default ``3``.
        temperature: Mutation temperature. Default ``0.7``.
        expected_output: Passed to :class:`PipelineEvaluator` for exact-match scoring.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        population_size: int = 6,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        elite_ratio: float = 0.2,
        tournament_k: int = 3,
        temperature: float = 0.7,
        expected_output: str | None = None,
        seed: int | None = None,
    ) -> None:
        self._backend = backend
        self._pop_size = population_size
        self._mut_rate = mutation_rate
        self._xo_rate = crossover_rate
        self._elite_n = max(1, int(population_size * elite_ratio))
        self._k = tournament_k
        self._mutator = PipelineMutator(backend, temperature)
        self._evaluator = PipelineEvaluator(backend, expected_output=expected_output)
        self._generation = 0
        self._best: Pipeline | None = None
        if seed is not None:
            random.seed(seed)

    @property
    def best(self) -> Pipeline | None:
        """Best pipeline seen across all generations."""
        return self._best

    def evolve(
        self,
        seed: Pipeline,
        task: str,
        n_generations: int = 10,
        on_generation: Any | None = None,
    ) -> Pipeline:
        """Run the evolutionary loop.

        Args:
            seed: Starting pipeline template.
            task: Task description / first-step input.
            n_generations: Number of generations.
            on_generation: Optional callback ``(gen, population)``.

        Returns:
            The best :class:`Pipeline` found.
        """
        population = self._initialize(seed)
        population = self._evaluate_all(population, task)
        self._update_best(population)
        if on_generation:
            on_generation(0, population)

        for gen in range(1, n_generations + 1):
            self._generation = gen
            population = self._next_generation(population, task)
            self._update_best(population)
            scores = [p.fitness or 0.0 for p in population]
            logger.info(
                "PipelineEvo gen %d/%d — best=%.4f mean=%.4f",
                gen, n_generations,
                max(scores) if scores else 0.0,
                sum(scores) / max(len(scores), 1),
            )
            if on_generation:
                on_generation(gen, population)

        if self._best is None and population:
            self._best = max(population, key=lambda p: p.fitness or 0.0)

        return self._best  # type: ignore[return-value]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _initialize(self, seed: Pipeline) -> list[Pipeline]:
        return [seed.clone() for _ in range(self._pop_size)]

    def _evaluate_all(self, population: list[Pipeline], task: str) -> list[Pipeline]:
        for pipeline in population:
            if pipeline.fitness is not None:
                continue
            try:
                score = self._evaluator.evaluate(pipeline, task)
            except Exception as exc:
                logger.warning("PipelineEvaluator raised: %s", exc)
                score = 0.0
            pipeline.fitness = score
            logger.debug(
                "Pipeline %s: fitness=%.4f steps=%d",
                pipeline.pipeline_id, score, len(pipeline.steps),
            )
        return population

    def _next_generation(self, population: list[Pipeline], task: str) -> list[Pipeline]:
        population.sort(key=lambda p: p.fitness or 0.0, reverse=True)
        next_gen: list[Pipeline] = [p.clone() for p in population[: self._elite_n]]
        # Preserve elite fitness
        for elite, orig in zip(next_gen, population[: self._elite_n]):
            elite.fitness = orig.fitness or 0.0

        while len(next_gen) < self._pop_size:
            if random.random() < self._xo_rate and len(population) >= 2:
                pa = self._tournament(population)
                pb = self._tournament(population)
                for _ in range(3):
                    if pb.pipeline_id != pa.pipeline_id:
                        break
                    pb = self._tournament(population)
                child = self._mutator.crossover(pa, pb, task)
            else:
                parent = self._tournament(population)
                if random.random() < self._mut_rate:
                    child = self._mutator.mutate(parent, task)
                else:
                    child = parent.clone()
                    child._fitness = None
            next_gen.append(child)

        return self._evaluate_all(next_gen, task)

    def _tournament(self, population: list[Pipeline]) -> Pipeline:
        k = min(self._k, len(population))
        contestants = random.sample(population, k)
        return max(contestants, key=lambda p: p.fitness or 0.0)

    def _update_best(self, population: list[Pipeline]) -> None:
        for p in population:
            if p.fitness is None:
                continue
            if self._best is None or p.fitness > (self._best.fitness or 0.0):
                self._best = p
