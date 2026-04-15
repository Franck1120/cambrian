# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Epigenetic layer — context-dependent genome expression.

In biological epigenetics, the same DNA sequence can produce different
phenotypes depending on environmental signals: genes are *silenced* or
*upregulated* without changing the underlying sequence.

Cambrian's :class:`EpigeneticLayer` applies the same idea to AI agent
genomes.  The **genome** (system prompt + hyperparameters) is the fixed
DNA; the **epigenetic expression** is the runtime modification of the
system prompt based on the current evolutionary *context* — generation
number, task type, population fitness, etc.

This allows a single genome to behave differently under different
conditions without those differences being inherited.  The genome itself
remains unchanged; only the prompt actually seen by the LLM is altered.

Architecture
------------

:class:`EpigenomicContext`
    A snapshot of the current environment: generation, task, fitness
    statistics, and arbitrary extra signals.

:class:`EpigeneticLayer`
    A set of **rules** (each a callable ``(Genome, EpigenomicContext) → str | None``).
    When :meth:`express` is called, all rules are evaluated and the
    resulting annotations are appended to the base system prompt.

:func:`make_standard_layer`
    A ready-to-use layer with four practical rules:

    - **generation_pressure** — tells the agent it's in early vs. late
      evolution so it can calibrate risk-taking.
    - **fitness_signal** — reminds the agent of its current standing.
    - **task_mode** — detects coding vs. reasoning vs. creative tasks and
      adds a mode hint.
    - **population_pressure** — warns about diversity collapse.

Usage::

    from cambrian.epigenetics import make_standard_layer, EpigenomicContext

    layer = make_standard_layer()

    ctx = EpigenomicContext(
        generation=5,
        task="Sort a list in Python",
        population_mean_fitness=0.42,
        population_best_fitness=0.68,
        total_generations=20,
    )

    expressed_prompt = layer.express(agent.genome, ctx)
    # expressed_prompt is the genome's system_prompt plus contextual annotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cambrian.agent import Agent, Genome


# ── Context ───────────────────────────────────────────────────────────────────


@dataclass
class EpigenomicContext:
    """Environmental snapshot that shapes epigenetic expression.

    Attributes:
        generation: Current generation index (0-based).
        task: Current task description.
        population_mean_fitness: Mean fitness of the current population.
        population_best_fitness: Best fitness in the current population.
        total_generations: Total generations planned for this run.
        extra: Arbitrary extra signals (e.g. diversity metric, task tags).
    """

    generation: int = 0
    task: str = ""
    population_mean_fitness: float = 0.0
    population_best_fitness: float = 0.0
    total_generations: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Fraction of evolution completed, in ``[0.0, 1.0]``."""
        if self.total_generations <= 1:
            return 1.0
        return self.generation / (self.total_generations - 1)

    @property
    def is_early(self) -> bool:
        """``True`` for the first 30% of the run."""
        return self.progress < 0.3

    @property
    def is_late(self) -> bool:
        """``True`` for the last 30% of the run."""
        return self.progress > 0.7


# ── Layer ─────────────────────────────────────────────────────────────────────

# A rule: takes (genome, context) and returns an annotation string or None
EpigeneticRule = Callable[[Genome, EpigenomicContext], "str | None"]


class EpigeneticLayer:
    """Runtime modifier that annotates agent system prompts based on context.

    Rules are callables ``(Genome, EpigenomicContext) → str | None``.
    A rule returning ``None`` or an empty string contributes no annotation.
    Non-empty returns are collected and appended to the genome's system prompt
    as a ``--- Epigenetic context ---`` section.

    Args:
        rules: Initial list of rule callables.
        separator: String inserted between the base prompt and the annotation
            block. Default ``"\\n\\n--- Epigenetic context ---\\n"``.

    Example::

        layer = EpigeneticLayer(rules=[
            lambda g, ctx: f"Phase: {'explore' if ctx.is_early else 'exploit'}",
        ])
        expressed = layer.express(genome, ctx)
    """

    def __init__(
        self,
        rules: list[EpigeneticRule] | None = None,
        separator: str = "\n\n--- Epigenetic context ---\n",
    ) -> None:
        self._rules: list[EpigeneticRule] = list(rules or [])
        self._separator = separator

    def add_rule(self, rule: EpigeneticRule) -> None:
        """Append a rule to the layer.

        Args:
            rule: Callable ``(Genome, EpigenomicContext) → str | None``.
        """
        self._rules.append(rule)

    def express(self, genome: Genome, ctx: EpigenomicContext) -> str:
        """Return the epigenetically expressed system prompt.

        Evaluates all rules against *genome* and *ctx*, collects non-empty
        annotations, and appends them to the genome's ``system_prompt``.

        Args:
            genome: The agent's genome (never modified in place).
            ctx: Current environmental context.

        Returns:
            Modified system prompt string.  If no rules fire, returns the
            original ``system_prompt`` unchanged.
        """
        annotations: list[str] = []
        for rule in self._rules:
            try:
                result = rule(genome, ctx)
                if result and result.strip():
                    annotations.append(result.strip())
            except Exception:
                pass  # rules must not break evaluation

        if not annotations:
            return genome.system_prompt

        block = "\n".join(f"• {a}" for a in annotations)
        return genome.system_prompt + self._separator + block

    def apply(self, agent: Agent, ctx: EpigenomicContext) -> Agent:
        """Return a *copy* of *agent* with the expressed system prompt.

        The original agent and its genome are never modified.

        Args:
            agent: Agent to express epigenetically.
            ctx: Current environmental context.

        Returns:
            A cloned :class:`~cambrian.agent.Agent` whose genome has the
            expressed system prompt.  All other attributes are identical.
        """
        expressed_prompt = self.express(agent.genome, ctx)
        if expressed_prompt == agent.genome.system_prompt:
            return agent  # no-op — avoid unnecessary clone
        clone = agent.clone()
        clone.genome.system_prompt = expressed_prompt
        return clone

    def __repr__(self) -> str:
        return f"EpigeneticLayer(rules={len(self._rules)})"


# ── Standard rules ────────────────────────────────────────────────────────────


def _rule_generation_pressure(genome: Genome, ctx: EpigenomicContext) -> str | None:
    """Communicate evolutionary phase to the agent."""
    if ctx.is_early:
        return (
            "You are in the early exploration phase. "
            "Be creative and try diverse approaches — don't converge yet."
        )
    if ctx.is_late:
        return (
            "You are in the final exploitation phase. "
            "Refine and perfect rather than explore new territory."
        )
    return (
        "You are in the middle of the evolutionary run. "
        "Balance exploration with exploitation."
    )


def _rule_fitness_signal(genome: Genome, ctx: EpigenomicContext) -> str | None:
    """Communicate the agent's relative standing in the population."""
    if ctx.population_mean_fitness <= 0:
        return None
    return (
        f"Current population mean fitness: {ctx.population_mean_fitness:.3f}, "
        f"best: {ctx.population_best_fitness:.3f}. "
        "Aim to exceed the best."
    )


def _rule_task_mode(genome: Genome, ctx: EpigenomicContext) -> str | None:
    """Detect task type and inject a mode hint."""
    task_lower = ctx.task.lower()
    coding_keywords = {"python", "code", "function", "implement", "algorithm", "sort", "search"}
    reasoning_keywords = {"reason", "explain", "why", "logic", "proof", "infer"}
    creative_keywords = {"write", "story", "poem", "creative", "imagine", "describe"}

    if any(k in task_lower for k in coding_keywords):
        return "Task mode: CODING — prioritise correctness, edge-case handling, and clean code."
    if any(k in task_lower for k in reasoning_keywords):
        return "Task mode: REASONING — show your chain of thought step by step."
    if any(k in task_lower for k in creative_keywords):
        return "Task mode: CREATIVE — be expressive, vivid, and original."
    return None


def _rule_population_pressure(genome: Genome, ctx: EpigenomicContext) -> str | None:
    """Warn about diversity collapse risk."""
    diversity = ctx.extra.get("strategy_entropy", None)
    if diversity is not None and float(diversity) < 0.3:
        return (
            "Population diversity is LOW. "
            "Introduce a novel strategy or perspective to avoid premature convergence."
        )
    return None


def make_standard_layer() -> EpigeneticLayer:
    """Build a ready-to-use :class:`EpigeneticLayer` with four practical rules.

    Rules included:

    - ``generation_pressure`` — phase signal (explore / mid / exploit)
    - ``fitness_signal`` — population standing relative to the agent
    - ``task_mode`` — coding / reasoning / creative hint
    - ``population_pressure`` — diversity collapse warning

    Returns:
        Configured :class:`EpigeneticLayer`.
    """
    return EpigeneticLayer(
        rules=[
            _rule_generation_pressure,
            _rule_fitness_signal,
            _rule_task_mode,
            _rule_population_pressure,
        ]
    )
