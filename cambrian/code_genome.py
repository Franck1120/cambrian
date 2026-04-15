# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Code genome — evolvable Python code for Forge mode.

Forge mode treats *executable Python code* as the evolvable artifact instead
of a natural-language system prompt.  A :class:`CodeGenome` carries the source
code, an entry-point name, a description used in mutation prompts, and a list
of test cases for scoring.

A :class:`CodeAgent` wraps a :class:`CodeGenome` with an LLM backend and
exposes fitness tracking compatible with the rest of the Cambrian ecosystem.

:class:`CodeMutator` uses an LLM to rewrite code to be more correct, efficient,
and robust.

:class:`CodeEvaluator` runs the code in the API-key-safe subprocess sandbox and
scores it using test-case pass rate.

:class:`CodeEvolutionEngine` orchestrates the full evolutionary loop over a
population of :class:`CodeAgent` objects.

Usage::

    from cambrian.code_genome import CodeGenome, CodeEvolutionEngine
    from cambrian.backends.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    engine = CodeEvolutionEngine(backend=backend, population_size=6)
    best = engine.evolve(
        seed=CodeGenome(description="reverse a string"),
        task="Write a Python function reverse(s: str) -> str that reverses the string.",
        test_cases=[{"input": "hello", "expected": "olleh"}],
        n_generations=8,
    )
    print(best.genome.code)
"""

from __future__ import annotations

import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.utils.logging import get_logger
from cambrian.utils.sandbox import run_in_sandbox

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CodeGenome
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CodeGenome:
    """Evolvable Python code artifact.

    Attributes:
        code: Python source code (the evolvable body).
        entry_point: Name of the callable to invoke for evaluation.
        description: Intent description used in the mutation prompt.
        language: Programming language. Currently only ``"python"`` is supported.
        test_cases: List of ``{"input": ..., "expected": ...}`` dicts.
        version: Monotonically increasing version counter (incremented by mutator).
        genome_id: Unique identifier, auto-generated if not supplied.
    """

    code: str = ""
    entry_point: str = "solution"
    description: str = ""
    language: str = "python"
    test_cases: list[dict[str, str]] = field(default_factory=list)
    version: int = 0
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "code": self.code,
            "entry_point": self.entry_point,
            "description": self.description,
            "language": self.language,
            "test_cases": list(self.test_cases),
            "version": self.version,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGenome":
        """Deserialise from a plain dictionary."""
        return cls(
            code=str(data.get("code", "")),
            entry_point=str(data.get("entry_point", "solution")),
            description=str(data.get("description", "")),
            language=str(data.get("language", "python")),
            test_cases=list(data.get("test_cases", [])),
            version=int(data.get("version", 0)),
            genome_id=str(data.get("genome_id", str(uuid.uuid4())[:8])),
        )

    def clone(self) -> "CodeGenome":
        """Return a deep copy with a fresh genome_id."""
        d = self.to_dict()
        d["genome_id"] = str(uuid.uuid4())[:8]
        return CodeGenome.from_dict(d)

    def __str__(self) -> str:
        lines = self.code.count("\n") + 1 if self.code else 0
        return (
            f"CodeGenome(id={self.genome_id}, v{self.version}, "
            f"entry={self.entry_point!r}, lines={lines})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CodeAgent
# ─────────────────────────────────────────────────────────────────────────────


class CodeAgent:
    """Wraps a :class:`CodeGenome` with fitness tracking.

    Args:
        genome: The evolvable code specification.
        backend: LLM backend (used by :class:`CodeMutator` — not stored here).
        agent_id: Unique identifier, auto-generated if not supplied.
    """

    def __init__(
        self,
        genome: CodeGenome,
        agent_id: str | None = None,
    ) -> None:
        self.genome = genome
        self.agent_id: str = agent_id or str(uuid.uuid4())[:8]
        self._fitness: float | None = None
        self._generation: int = 0

    @property
    def id(self) -> str:
        """Alias for :attr:`agent_id`."""
        return self.agent_id

    @property
    def fitness(self) -> float | None:
        """Last recorded fitness, or ``None`` if not evaluated."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = float(value)

    @property
    def generation(self) -> int:
        """Generation in which this agent was born."""
        return self._generation

    @generation.setter
    def generation(self, value: int) -> None:
        self._generation = int(value)

    def clone(self) -> "CodeAgent":
        """Return a deep copy with fresh id and no fitness."""
        clone = CodeAgent(genome=self.genome.clone(), agent_id=None)
        clone._generation = self._generation
        return clone

    def to_dict(self) -> dict[str, Any]:
        """Serialise agent state to a plain dict."""
        return {
            "id": self.agent_id,
            "generation": self._generation,
            "fitness": self._fitness,
            "genome": self.genome.to_dict(),
        }

    def __repr__(self) -> str:
        fit = f"{self._fitness:.4f}" if self._fitness is not None else "None"
        return (
            f"CodeAgent(id={self.agent_id}, gen={self._generation}, "
            f"fitness={fit}, genome={self.genome})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CodeMutator
# ─────────────────────────────────────────────────────────────────────────────

_CODE_MUTATE_SYSTEM = """\
You are an expert software engineer and evolutionary algorithm researcher.
Your task is to improve a Python function so that it is more correct, efficient,
and robust.  Return ONLY a valid Python code block — no explanations.
The code must define the entry-point function and nothing else that would break it.
"""

_CODE_MUTATE_TEMPLATE = """\
Current Python code (entry point: {entry_point}):
```python
{code}
```

Task description: {task}
Current fitness score: {fitness}
Test cases:
{test_cases}

Previous version: v{version}

Rewrite the code to improve correctness and efficiency.
- Keep the function signature identical.
- Handle edge cases.
- Return ONLY the improved Python code, no markdown, no explanation.
"""

_CODE_CROSSOVER_SYSTEM = """\
You are an expert software engineer.  Combine the best parts of two Python
functions to produce a superior implementation.  Return ONLY valid Python code.
"""

_CODE_CROSSOVER_TEMPLATE = """\
Parent A (fitness {fitness_a}):
```python
{code_a}
```

Parent B (fitness {fitness_b}):
```python
{code_b}
```

Task: {task}

Combine the strongest patterns from both implementations.
Return ONLY the combined Python code.
"""


class CodeMutator:
    """LLM-guided code mutation and crossover.

    Args:
        backend: LLM backend for generating code.
        mutation_temperature: Sampling temperature (default ``0.6``).
        fallback_on_error: Return original on parse failure if ``True``.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        mutation_temperature: float = 0.6,
        fallback_on_error: bool = True,
    ) -> None:
        self._backend = backend
        self._temp = mutation_temperature
        self._fallback = fallback_on_error

    def mutate(self, agent: CodeAgent, task: str = "") -> CodeAgent:
        """Return a new agent with LLM-improved code.

        Args:
            agent: The agent to mutate.
            task: Task description.

        Returns:
            New :class:`CodeAgent` with improved code genome.
        """
        tc_text = "\n".join(
            f"  input={tc.get('input', '')!r} → expected={tc.get('expected', '')!r}"
            for tc in agent.genome.test_cases[:5]
        ) or "  (no test cases provided)"

        prompt = _CODE_MUTATE_TEMPLATE.format(
            entry_point=agent.genome.entry_point,
            code=agent.genome.code or "(empty — write a fresh implementation)",
            task=task or agent.genome.description or "solve the problem",
            fitness=(
                f"{agent.fitness:.4f}" if agent.fitness is not None else "not evaluated"
            ),
            test_cases=tc_text,
            version=agent.genome.version,
        )

        try:
            raw = self._backend.generate(
                prompt,
                system=_CODE_MUTATE_SYSTEM,
                temperature=self._temp,
            )
            new_code = self._extract_code(raw)
        except Exception:
            if self._fallback:
                new_code = agent.genome.code
            else:
                raise

        child = agent.clone()
        child.genome.code = new_code
        child.genome.version = agent.genome.version + 1
        child._fitness = None
        return child

    def crossover(self, parent_a: CodeAgent, parent_b: CodeAgent, task: str = "") -> CodeAgent:
        """Combine code from two parents to produce an offspring.

        Args:
            parent_a: First parent.
            parent_b: Second parent.
            task: Task description.

        Returns:
            New :class:`CodeAgent` with combined code.
        """
        prompt = _CODE_CROSSOVER_TEMPLATE.format(
            code_a=parent_a.genome.code or "(empty)",
            fitness_a=f"{parent_a.fitness:.4f}" if parent_a.fitness is not None else "0",
            code_b=parent_b.genome.code or "(empty)",
            fitness_b=f"{parent_b.fitness:.4f}" if parent_b.fitness is not None else "0",
            task=task or "solve the problem",
        )

        base = parent_a if (parent_a.fitness or 0.0) >= (parent_b.fitness or 0.0) else parent_b

        try:
            raw = self._backend.generate(
                prompt,
                system=_CODE_CROSSOVER_SYSTEM,
                temperature=self._temp,
            )
            new_code = self._extract_code(raw)
        except Exception:
            if self._fallback:
                new_code = base.genome.code
            else:
                raise

        child = base.clone()
        child.genome.code = new_code
        child.genome.version = max(parent_a.genome.version, parent_b.genome.version) + 1
        child._fitness = None
        return child

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Extract Python code from LLM response, stripping fences."""
        # Try fenced block first
        m = re.search(r"```(?:python|py)?\s*([\s\S]+?)```", raw)
        if m:
            return m.group(1).strip()
        return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CodeEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class CodeEvaluator:
    """Evaluates a :class:`CodeAgent` by running test cases in the sandbox.

    Scoring:
    - Empty code → 0.0
    - Syntax/runtime error → 0.1
    - k/N test cases pass → 0.1 + 0.9 × (k/N)
    - All test cases pass → 1.0

    Args:
        timeout: Max seconds per test case execution. Default ``10.0``.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    def evaluate(self, agent: CodeAgent, task: str = "") -> float:
        """Score *agent* by running its code against test cases.

        Args:
            agent: Agent to evaluate.
            task: Unused (kept for interface compatibility).

        Returns:
            Fitness in ``[0.0, 1.0]``.
        """
        code = agent.genome.code
        if not code.strip():
            return 0.0

        test_cases = agent.genome.test_cases
        if not test_cases:
            # No test cases — just check the code compiles and runs
            result = run_in_sandbox(code, timeout=self._timeout)
            if result.timed_out:
                return 0.0
            return 0.8 if result.returncode == 0 else 0.1

        passed = 0
        for tc in test_cases:
            inp = tc.get("input", "")
            expected = str(tc.get("expected", "")).strip()
            runner = self._make_runner(code, agent.genome.entry_point, inp)
            result = run_in_sandbox(runner, timeout=self._timeout)
            if result.timed_out or result.returncode != 0:
                continue
            actual = result.stdout.strip()
            if actual == expected:
                passed += 1

        if not test_cases:
            return 0.0
        ratio = passed / len(test_cases)
        return 0.1 + 0.9 * ratio if ratio < 1.0 else 1.0

    @staticmethod
    def _make_runner(code: str, entry_point: str, inp: str) -> str:
        """Build a sandbox script that imports the code and calls entry_point."""
        # Escape backslashes, single quotes, and newlines so the string literal
        # is valid Python regardless of what the test-case input contains.
        safe_inp = (
            inp.replace("\\", "\\\\")
               .replace("'", "\\'")
               .replace("\n", "\\n")
               .replace("\r", "\\r")
        )
        return (
            f"{code}\n\n"
            f"_result = {entry_point}('{safe_inp}')\n"
            f"print(_result)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CodeEvolutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class CodeEvolutionEngine:
    """Evolutionary loop over :class:`CodeAgent` populations (Forge mode).

    Args:
        backend: LLM backend used by :class:`CodeMutator`.
        population_size: Number of agents per generation. Default ``8``.
        mutation_rate: Probability of mutating each non-elite agent. Default ``0.8``.
        crossover_rate: Probability of crossover vs direct mutation. Default ``0.3``.
        elite_ratio: Fraction of top agents preserved unchanged. Default ``0.2``.
        tournament_k: Tournament selection pool size. Default ``3``.
        mutation_temperature: Temperature for the mutation LLM call. Default ``0.6``.
        timeout: Sandbox timeout per evaluation. Default ``10.0``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        population_size: int = 8,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        elite_ratio: float = 0.2,
        tournament_k: int = 3,
        mutation_temperature: float = 0.6,
        timeout: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self._backend = backend
        self._pop_size = population_size
        self._mut_rate = mutation_rate
        self._xo_rate = crossover_rate
        self._elite_n = max(1, int(population_size * elite_ratio))
        self._k = tournament_k
        self._timeout = timeout
        self._mutator = CodeMutator(backend, mutation_temperature)
        self._evaluator = CodeEvaluator(timeout)
        self._generation = 0
        self._best: CodeAgent | None = None
        if seed is not None:
            random.seed(seed)

    @property
    def best(self) -> CodeAgent | None:
        """Best agent seen across all generations."""
        return self._best

    def evolve(
        self,
        seed: CodeGenome,
        task: str,
        test_cases: list[dict[str, str]] | None = None,
        n_generations: int = 10,
        on_generation: Any | None = None,
    ) -> CodeAgent:
        """Run the full evolutionary loop.

        Args:
            seed: Starting genome (code may be empty — mutator will write it).
            task: Task description for the mutator.
            test_cases: Optional test cases overriding those in *seed*.
            n_generations: Number of generations.
            on_generation: Optional callback ``(gen, population)``.

        Returns:
            The best :class:`CodeAgent` found.
        """
        if test_cases is not None:
            seed.test_cases = test_cases

        population = self._initialize(seed)
        population = self._evaluate_all(population, task)
        self._update_best(population)
        if on_generation:
            on_generation(0, population)

        for gen in range(1, n_generations + 1):
            self._generation = gen
            population = self._next_generation(population, task)
            self._update_best(population)
            scores = [a.fitness or 0.0 for a in population]
            logger.info(
                "CodeEvo gen %d/%d — best=%.4f mean=%.4f",
                gen, n_generations,
                max(scores) if scores else 0.0,
                sum(scores) / max(len(scores), 1),
            )
            if on_generation:
                on_generation(gen, population)

        if self._best is None and population:
            self._best = max(population, key=lambda a: a.fitness or 0.0)

        return self._best  # type: ignore[return-value]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _initialize(self, seed: CodeGenome) -> list[CodeAgent]:
        population: list[CodeAgent] = []
        for _ in range(self._pop_size):
            g = seed.clone()
            population.append(CodeAgent(genome=g))
        return population

    def _evaluate_all(self, population: list[CodeAgent], task: str) -> list[CodeAgent]:
        for agent in population:
            if agent.fitness is not None:
                continue
            t0 = time.monotonic()
            try:
                score = self._evaluator.evaluate(agent, task)
            except Exception as exc:
                logger.warning("CodeEvaluator raised %s", exc)
                score = 0.0
            agent.fitness = score
            logger.debug(
                "CodeAgent %s: fitness=%.4f (%.2fs)",
                agent.id[:8], score, time.monotonic() - t0,
            )
        return population

    def _next_generation(self, population: list[CodeAgent], task: str) -> list[CodeAgent]:
        population.sort(key=lambda a: a.fitness or 0.0, reverse=True)
        next_gen: list[CodeAgent] = list(population[: self._elite_n])

        while len(next_gen) < self._pop_size:
            if random.random() < self._xo_rate and len(population) >= 2:
                pa = self._tournament(population)
                pb = self._tournament(population)
                for _ in range(3):
                    if pb.id != pa.id:
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
            child.generation = self._generation
            next_gen.append(child)

        return self._evaluate_all(next_gen, task)

    def _tournament(self, population: list[CodeAgent]) -> CodeAgent:
        k = min(self._k, len(population))
        contestants = random.sample(population, k)
        return max(contestants, key=lambda a: a.fitness or 0.0)

    def _update_best(self, population: list[CodeAgent]) -> None:
        for agent in population:
            if agent.fitness is None:
                continue
            if self._best is None or agent.fitness > (self._best.fitness or 0.0):
                self._best = agent
