"""CodeGenome — executable Python code as an evolvable genome (Forge mode).

Forge mode treats a Python function/script as the genome.  The LLM mutator
rewrites the code to improve correctness, efficiency, and robustness.  The
evaluator executes the code in a sandbox and scores it by test-case pass rate.

Classes
-------
CodeGenome
    A genome whose DNA is Python source code rather than a text prompt.
CodeMutator
    LLM-guided rewriter for CodeGenomes.
TestCase
    A single (input, expected_output) pair used to evaluate a CodeGenome.
CodeEvaluationResult
    Full evaluation report: pass rate, runtime, LOC, partial scores.
CodeEvolutionEngine
    High-level evolutionary loop specialised for CodeGenomes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger
from cambrian.utils.sandbox import SandboxResult, run_in_sandbox

logger = get_logger(__name__)


# ── TestCase ──────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """A single (input, expected_output) test pair for code evaluation.

    Args:
        input_data: The stdin/argument string passed to the code.
        expected_output: The expected stdout produced by the code (stripped).
        weight: Relative weight for partial-credit scoring. Default 1.0.
        label: Human-readable label for reporting. Auto-generated if omitted.
    """

    input_data: str
    expected_output: str
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"test_{uuid.uuid4().hex[:6]}"


# ── CodeGenome ─────────────────────────────────────────────────────────────────


@dataclass
class CodeGenome:
    """Executable Python code as an evolvable genome.

    The ``code`` attribute is the full Python source.  The mutator asks the LLM
    to rewrite it, and the evaluator runs it in a sandbox against test cases.

    Attributes:
        code: Python source code (may be empty — mutator will generate it).
        description: Natural-language description of what the code should do.
            Used to seed the first mutation when ``code`` is empty.
        version: Monotonically increasing counter — incremented on each mutation.
        genome_id: Unique identifier, auto-generated if not supplied.
        parent_id: genome_id of the parent (for lineage tracking).
        metadata: Arbitrary extra data (strategy hints, runtime stats, etc.).
    """

    code: str = ""
    description: str = ""
    version: int = 0
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def loc(self) -> int:
        """Return the number of non-empty, non-comment lines of code."""
        return sum(
            1
            for line in self.code.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "version": self.version,
            "genome_id": self.genome_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGenome":
        return cls(
            code=str(data.get("code", "")),
            description=str(data.get("description", "")),
            version=int(data.get("version", 0)),
            genome_id=str(data.get("genome_id", str(uuid.uuid4())[:8])),
            parent_id=str(data.get("parent_id", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def __str__(self) -> str:
        return (
            f"CodeGenome(id={self.genome_id}, v={self.version}, "
            f"loc={self.loc()}, desc={self.description[:40]!r})"
        )


# ── CodeEvaluationResult ───────────────────────────────────────────────────────


@dataclass
class CodeEvaluationResult:
    """Full evaluation report for a CodeGenome against a test suite.

    Attributes:
        pass_rate: Fraction of test cases passed, weighted. Range [0, 1].
        runtime_s: Wall-clock seconds spent in the sandbox.
        loc: Lines of code (non-empty, non-comment).
        passed: Number of test cases fully passed.
        total: Total number of test cases.
        error: Sandbox error message if code crashed, else empty string.
        timed_out: Whether the sandbox timed out.
        fitness: Combined fitness score (see :meth:`compute_fitness`).
    """

    pass_rate: float
    runtime_s: float
    loc: int
    passed: int
    total: int
    error: str = ""
    timed_out: bool = False
    fitness: float = 0.0

    @property
    def success(self) -> bool:
        """True when all test cases pass."""
        return self.passed == self.total and self.total > 0

    def __repr__(self) -> str:
        return (
            f"CodeEvaluationResult(pass={self.passed}/{self.total}, "
            f"fitness={self.fitness:.4f}, runtime={self.runtime_s:.2f}s)"
        )


# ── CodeEvaluator ──────────────────────────────────────────────────────────────


class CodeEvaluator:
    """Evaluates a :class:`CodeGenome` against a suite of :class:`TestCase` objects.

    The evaluator runs each test case in a subprocess sandbox.  The final
    fitness blends pass rate, a small LOC efficiency bonus, and a runtime bonus.

    Args:
        test_cases: List of test cases to run.
        timeout: Maximum sandbox execution time per test case. Default 10s.
        loc_bonus: Weight of LOC efficiency term. Shorter code gets a bonus
            capped at 0.05. Default 0.02.
        runtime_bonus_cap: Max fitness bonus for fast execution. Default 0.03.
    """

    def __init__(
        self,
        test_cases: list[TestCase],
        timeout: float = 10.0,
        loc_bonus: float = 0.02,
        runtime_bonus_cap: float = 0.03,
    ) -> None:
        if not test_cases:
            raise ValueError("test_cases must not be empty")
        self._test_cases = test_cases
        self._timeout = timeout
        self._loc_bonus = loc_bonus
        self._runtime_bonus_cap = runtime_bonus_cap

    def evaluate(self, genome: CodeGenome) -> CodeEvaluationResult:
        """Run all test cases and return a :class:`CodeEvaluationResult`."""
        if not genome.code.strip():
            return CodeEvaluationResult(
                pass_rate=0.0,
                runtime_s=0.0,
                loc=0,
                passed=0,
                total=len(self._test_cases),
                error="Empty code",
                fitness=0.0,
            )

        passed = 0
        weighted_passed = 0.0
        total_weight = sum(tc.weight for tc in self._test_cases)
        total_runtime = 0.0
        last_error = ""
        timed_out = False

        for tc in self._test_cases:
            t0 = time.monotonic()
            result: SandboxResult = run_in_sandbox(
                genome.code,
                timeout=self._timeout,
                stdin=tc.input_data,
            )
            elapsed = time.monotonic() - t0
            total_runtime += elapsed

            if result.timed_out:
                timed_out = True
                last_error = result.stderr
                continue
            if result.returncode != 0:
                last_error = result.stderr.strip()
                continue

            actual = result.stdout.strip()
            expected = tc.expected_output.strip()
            if actual == expected:
                passed += 1
                weighted_passed += tc.weight

        pass_rate = weighted_passed / total_weight if total_weight > 0 else 0.0
        loc = genome.loc()
        avg_runtime = total_runtime / len(self._test_cases)

        fitness = self._compute_fitness(pass_rate, loc, avg_runtime)
        return CodeEvaluationResult(
            pass_rate=pass_rate,
            runtime_s=avg_runtime,
            loc=loc,
            passed=passed,
            total=len(self._test_cases),
            error=last_error,
            timed_out=timed_out,
            fitness=fitness,
        )

    def _compute_fitness(
        self,
        pass_rate: float,
        loc: int,
        runtime_s: float,
    ) -> float:
        """Blend pass rate with small LOC and runtime bonuses."""
        # LOC bonus: shorter code (up to loc_bonus weight) — cap at 100 LOC ref
        loc_score = max(0.0, 1.0 - loc / 200.0)
        loc_term = self._loc_bonus * loc_score

        # Runtime bonus: faster is better — cap at runtime_bonus_cap
        runtime_score = max(0.0, 1.0 - runtime_s / 5.0)
        runtime_term = self._runtime_bonus_cap * runtime_score

        base_weight = 1.0 - self._loc_bonus - self._runtime_bonus_cap
        return base_weight * pass_rate + loc_term + runtime_term


# ── CodeMutator ────────────────────────────────────────────────────────────────


_CODE_MUTATE_SYSTEM = """You are an expert Python developer and evolutionary algorithm researcher.
Your task: rewrite the given Python code to make it more correct, efficient, and robust.
The code must solve the task described.
Return ONLY valid Python source code — no markdown fences, no explanation."""

_CODE_MUTATE_TEMPLATE = """Current Python code:
```python
{code}
```

Task description: {description}

Current fitness score: {fitness:.4f} (range 0.0–1.0; higher is better)
Lines of code: {loc}
Evaluation notes: {notes}

Rewrite the code to improve:
1. Correctness — handle all edge cases, parse input correctly
2. Efficiency — fewer lines, faster runtime (avoid unnecessary imports)
3. Robustness — don't crash on unusual inputs

Return ONLY valid Python source code. No markdown, no explanation."""

_CODE_SEED_TEMPLATE = """Write a Python script that solves this task:

{description}

Requirements:
- The script reads from stdin if input is needed
- The script writes the result to stdout
- No external libraries that are not in the Python standard library
- Clean, correct, minimal code

Return ONLY valid Python source code. No markdown, no explanation."""


class CodeMutator:
    """LLM-guided rewriter for :class:`CodeGenome` objects.

    Args:
        backend: LLM backend used to generate mutations.
        mutation_temperature: Sampling temperature for mutations. Default 0.5.
        fallback_on_error: If True, return original genome on LLM/parse errors.
    """

    def __init__(
        self,
        backend: LLMBackend,
        mutation_temperature: float = 0.5,
        fallback_on_error: bool = True,
    ) -> None:
        self._backend = backend
        self._temp = mutation_temperature
        self._fallback = fallback_on_error

    def mutate(
        self,
        genome: CodeGenome,
        evaluation: "CodeEvaluationResult | None" = None,
    ) -> CodeGenome:
        """Return a new :class:`CodeGenome` with LLM-rewritten code.

        If ``genome.code`` is empty, seeds the first implementation from the
        description.  Otherwise, rewrites the existing code.

        Args:
            genome: The genome to mutate.
            evaluation: Optional last evaluation result (used to guide the LLM).

        Returns:
            A new CodeGenome with incremented version.
        """
        if not genome.code.strip():
            new_code = self._seed(genome.description)
        else:
            notes = ""
            fitness = 0.0
            if evaluation is not None:
                fitness = evaluation.fitness
                if evaluation.error:
                    notes = f"Error: {evaluation.error[:200]}"
                elif evaluation.passed < evaluation.total:
                    notes = (
                        f"{evaluation.passed}/{evaluation.total} test cases passed"
                    )
            new_code = self._rewrite(genome, fitness, notes)

        child = CodeGenome(
            code=new_code,
            description=genome.description,
            version=genome.version + 1,
            genome_id=str(uuid.uuid4())[:8],
            parent_id=genome.genome_id,
            metadata=dict(genome.metadata),
        )
        return child

    def _seed(self, description: str) -> str:
        """Generate initial code from description."""
        prompt = _CODE_SEED_TEMPLATE.format(description=description)
        try:
            return self._backend.generate(
                prompt, system=_CODE_MUTATE_SYSTEM, temperature=self._temp
            )
        except Exception as exc:
            logger.warning("CodeMutator seed failed: %s", exc)
            if not self._fallback:
                raise
            return f"# TODO: implement\n# {description}\nprint('')\n"

    def _rewrite(
        self,
        genome: CodeGenome,
        fitness: float,
        notes: str,
    ) -> str:
        """Ask the LLM to improve existing code."""
        prompt = _CODE_MUTATE_TEMPLATE.format(
            code=genome.code,
            description=genome.description,
            fitness=fitness,
            loc=genome.loc(),
            notes=notes or "No specific issues",
        )
        try:
            raw = self._backend.generate(
                prompt, system=_CODE_MUTATE_SYSTEM, temperature=self._temp
            )
            return self._clean_code(raw)
        except Exception as exc:
            logger.warning("CodeMutator rewrite failed: %s", exc)
            if not self._fallback:
                raise
            return genome.code

    @staticmethod
    def _clean_code(raw: str) -> str:
        """Strip markdown fences if the LLM accidentally included them."""
        lines = raw.splitlines()
        collected: list[str] = []
        in_fence = False
        for line in lines:
            stripped = line.strip()
            if not in_fence and (
                stripped.startswith("```python")
                or stripped.startswith("```py")
                or stripped == "```"
            ):
                in_fence = True
                continue
            if in_fence and stripped == "```":
                break
            if in_fence:
                collected.append(line)
        if collected:
            return "\n".join(collected)
        return raw.strip()


# ── CodeEvolutionEngine ────────────────────────────────────────────────────────


class CodeEvolutionEngine:
    """High-level evolutionary loop for :class:`CodeGenome` objects (Forge mode).

    Each generation:
    1. Evaluate all genomes in the population.
    2. Keep the top ``elite_n`` unchanged (elitism).
    3. Mutate the rest using :class:`CodeMutator`.
    4. Return the best genome found across all generations.

    Args:
        backend: LLM backend used by :class:`CodeMutator`.
        evaluator: Optional :class:`CodeEvaluator`.  If ``None`` you must pass
            ``test_cases`` at ``evolve()`` time.
        population_size: Number of genomes per generation. Default 6.
        mutation_temperature: Temperature forwarded to :class:`CodeMutator`.
        elite_ratio: Fraction of top genomes carried unchanged. Default 0.33.
        seed: Random seed (unused currently; reserved for future use).
    """

    def __init__(
        self,
        backend: LLMBackend,
        evaluator: "CodeEvaluator | None" = None,
        population_size: int = 6,
        mutation_temperature: float = 0.5,
        elite_ratio: float = 0.33,
        seed: "int | None" = None,
    ) -> None:
        self._backend = backend
        self._evaluator = evaluator
        self._pop_size = population_size
        self._elite_n = max(1, int(population_size * elite_ratio))
        self._mutator = CodeMutator(
            backend=backend,
            mutation_temperature=mutation_temperature,
        )
        _ = seed  # reserved

    def evolve(
        self,
        seed: CodeGenome,
        task: str,
        n_generations: int = 8,
        test_cases: "list[TestCase] | None" = None,
        on_generation: "Any | None" = None,
    ) -> CodeGenome:
        """Run the evolutionary loop and return the best :class:`CodeGenome`.

        Args:
            seed: Initial genome (code may be empty; description must be set).
            task: Task description used to guide the mutator and evaluator.
            n_generations: Number of generations to run. Default 8.
            test_cases: Test cases for evaluation (required if no evaluator was
                set at construction time).
            on_generation: Optional callback ``(gen, pop, best) -> None``.

        Returns:
            The :class:`CodeGenome` with the highest fitness found.
        """
        evaluator = self._evaluator
        if evaluator is None:
            if not test_cases:
                raise ValueError(
                    "Provide test_cases= or pass a CodeEvaluator at construction time."
                )
            evaluator = CodeEvaluator(test_cases)

        if not seed.description:
            seed = CodeGenome(
                code=seed.code,
                description=task,
                genome_id=seed.genome_id,
            )

        # Seed population: mutate the seed to fill population
        population = [seed]
        while len(population) < self._pop_size:
            population.append(self._mutator.mutate(seed))

        best_genome: CodeGenome = seed
        best_fitness: float = -1.0

        for gen in range(n_generations):
            # Evaluate
            results: list[tuple[CodeGenome, CodeEvaluationResult]] = []
            for genome in population:
                result = evaluator.evaluate(genome)
                results.append((genome, result))
                logger.debug(
                    "gen=%d id=%s fitness=%.4f pass=%d/%d",
                    gen,
                    genome.genome_id,
                    result.fitness,
                    result.passed,
                    result.total,
                )
                if result.fitness > best_fitness:
                    best_fitness = result.fitness
                    best_genome = genome

            # Sort by fitness descending
            results.sort(key=lambda x: x[1].fitness, reverse=True)

            if on_generation is not None:
                on_generation(gen, results, best_genome)

            # Early exit if perfect solution found
            if best_fitness >= 1.0:
                logger.info("Perfect solution found at generation %d", gen)
                break

            # Elitism: keep top genomes unchanged
            elites = [r[0] for r in results[: self._elite_n]]

            # Mutate the rest
            new_pop: list[CodeGenome] = list(elites)
            for genome, eval_result in results[self._elite_n :]:
                child = self._mutator.mutate(genome, evaluation=eval_result)
                new_pop.append(child)

            population = new_pop[: self._pop_size]

        return best_genome
