# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Constitutional AI as a selective pressure in evolutionary search.

Standard evaluation scores the agent's *first* response to a task.
Constitutional AI adds a self-critique loop: the agent generates an initial
response, critiques it against a list of principles, and produces a revised
response.  Only the *revised* response is shown to the evaluator.

This has two evolutionary effects:

1. **Selection pressure** — agents whose prompts naturally produce
   self-correcting behaviour are rewarded; agents whose prompts produce
   rigid, non-revisable outputs are penalised.

2. **Improved baseline** — even a mediocre genome can produce better
   outputs when it has the opportunity to self-correct before evaluation.

Architecture
------------

:class:`ConstitutionalWrapper` wraps any evaluator.  When called, it:

1. Runs the agent to get an *initial draft*.
2. Prompts the agent to critique the draft against each constitutional
   principle.
3. Prompts the agent to produce a *revised* response incorporating the
   critiques.
4. Temporarily substitutes the revised response into the agent so the
   wrapped evaluator can score it.
5. Restores the agent to its original state.

Because the critique and revision calls use the agent's own backend, the
cost scales with the number of constitutional principles (typically 3–5).

Usage::

    from cambrian.constitutional import ConstitutionalWrapper, DEFAULT_CONSTITUTION

    inner_evaluator = CodeEvaluator(expected_output="...")
    constitutional_ev = ConstitutionalWrapper(
        base_evaluator=inner_evaluator,
        n_revisions=1,
    )

    engine = EvolutionEngine(evaluator=constitutional_ev, ...)
"""

from __future__ import annotations

from typing import Callable

from cambrian.agent import Agent
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)

# Default set of constitutional principles
DEFAULT_CONSTITUTION: list[str] = [
    "Does the response fully and directly address the task?",
    "Is the response accurate and free from factual errors or hallucinations?",
    "Is the response concise — no padding, no unnecessary disclaimers?",
    "Does the response follow any explicit formatting requirements in the task?",
    "Could any part of the response be harmful or misleading to the user?",
]

_CRITIQUE_SYSTEM = (
    "You are a rigorous self-critic. Given a task and a draft response, "
    "identify specific flaws with respect to the principle provided. "
    "Be concrete and brief (1-3 sentences per flaw). "
    "If the draft fully satisfies the principle, say 'OK'."
)

_REVISION_SYSTEM = (
    "You are an AI assistant that improves its own responses. "
    "Given the original task, a draft response, and a list of critiques, "
    "produce a final revised response that addresses all critiques. "
    "Output ONLY the revised response, no preamble."
)


class ConstitutionalWrapper:
    """Evaluator wrapper that forces agents to self-critique before scoring.

    Args:
        base_evaluator: The underlying ``(agent, task) -> float`` callable.
        constitution: List of constitutional principles (questions) to
            critique against.  Defaults to :data:`DEFAULT_CONSTITUTION`.
        n_revisions: Number of critique-revise cycles to run.  Default ``1``.
        critique_temperature: Temperature used for the critique LLM call.
            Low values (0.1) produce consistent critiques.  Default ``0.2``.
        revision_temperature: Temperature for the revision call.  Slightly
            higher to allow creative improvement.  Default ``0.4``.
        skip_if_no_backend: If ``True`` (default), skip the constitutional
            cycle and fall back to the base evaluator when the agent has no
            backend.  If ``False``, raises ``RuntimeError``.
    """

    def __init__(
        self,
        base_evaluator: Callable[[Agent, str], float],
        constitution: list[str] | None = None,
        n_revisions: int = 1,
        critique_temperature: float = 0.2,
        revision_temperature: float = 0.4,
        skip_if_no_backend: bool = True,
    ) -> None:
        self._base = base_evaluator
        self._constitution = constitution or list(DEFAULT_CONSTITUTION)
        self._n_revisions = n_revisions
        self._t_critique = critique_temperature
        self._t_revision = revision_temperature
        self._skip_no_backend = skip_if_no_backend

    def __call__(self, agent: Agent, task: str) -> float:
        """Run constitutional self-critique, then score the revised response.

        Args:
            agent: The agent to evaluate.
            task: The task description.

        Returns:
            The base evaluator's score for the (potentially improved) response.
        """
        if agent.backend is None:
            if self._skip_no_backend:
                logger.debug(
                    "ConstitutionalWrapper: agent %s has no backend, skipping critique",
                    agent.id[:8],
                )
                return self._base(agent, task)
            raise RuntimeError(
                f"Agent {agent.id} has no backend — cannot run constitutional critique."
            )

        original_prompt = agent.genome.system_prompt

        try:
            revised_prompt = self._run_constitutional_cycle(agent, task)
            if revised_prompt and revised_prompt != original_prompt:
                agent.genome.system_prompt = revised_prompt
                logger.debug(
                    "ConstitutionalWrapper: agent %s prompt revised (%d → %d chars)",
                    agent.id[:8], len(original_prompt), len(revised_prompt),
                )
            score = self._base(agent, task)
        finally:
            # Always restore original prompt so we don't permanently mutate the genome
            agent.genome.system_prompt = original_prompt

        return score

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_constitutional_cycle(self, agent: Agent, task: str) -> str:
        """Run one or more critique-revise cycles.

        Returns:
            The final revised system prompt for the agent.
        """
        current_prompt = agent.genome.system_prompt

        for _ in range(self._n_revisions):
            critiques = self._gather_critiques(agent, task, current_prompt)
            if not any(c.strip().upper() not in ("OK", "") for c in critiques):
                break  # all principles satisfied — no revision needed

            current_prompt = self._revise(agent, task, current_prompt, critiques)

        return current_prompt

    def _gather_critiques(
        self, agent: Agent, task: str, current_prompt: str
    ) -> list[str]:
        """Ask the agent to critique the current prompt against each principle."""
        critiques: list[str] = []
        for principle in self._constitution:
            prompt = (
                f"Task: {task}\n\n"
                f"Current system prompt being evaluated:\n{current_prompt}\n\n"
                f"Constitutional principle: {principle}\n\n"
                "Critique the system prompt against this principle. "
                "If it fully satisfies it, reply with only 'OK'."
            )
            try:
                assert agent.backend is not None
                critique = agent.backend.generate(
                    prompt,
                    system=_CRITIQUE_SYSTEM,
                    temperature=self._t_critique,
                )
                critiques.append(critique.strip())
            except Exception as exc:
                logger.warning("Critique call failed for principle %r: %s", principle[:40], exc)
                critiques.append("")

        return critiques

    def _revise(
        self,
        agent: Agent,
        task: str,
        current_prompt: str,
        critiques: list[str],
    ) -> str:
        """Generate a revised system prompt incorporating all critiques."""
        non_ok = [
            f"- Principle: {p}\n  Critique: {c}"
            for p, c in zip(self._constitution, critiques)
            if c.strip().upper() not in ("OK", "")
        ]
        if not non_ok:
            return current_prompt

        revision_prompt = (
            f"Task the agent must solve: {task}\n\n"
            f"Current system prompt:\n{current_prompt}\n\n"
            "Critiques to address:\n" + "\n".join(non_ok) + "\n\n"
            "Rewrite the system prompt to address all critiques while remaining "
            "effective for the task. Output ONLY the improved system prompt."
        )

        try:
            assert agent.backend is not None
            revised = agent.backend.generate(
                revision_prompt,
                system=_REVISION_SYSTEM,
                temperature=self._t_revision,
            )
            return revised.strip() or current_prompt
        except Exception as exc:
            logger.warning("Revision call failed: %s", exc)
            return current_prompt

    def __repr__(self) -> str:
        return (
            f"ConstitutionalWrapper(n_principles={len(self._constitution)}, "
            f"n_revisions={self._n_revisions})"
        )


def build_constitutional_evaluator(
    base_evaluator: Callable[[Agent, str], float],
    n_principles: int = 3,
    n_revisions: int = 1,
) -> ConstitutionalWrapper:
    """Build a constitutional evaluator with the first *n_principles* defaults.

    Args:
        base_evaluator: The underlying evaluator to wrap.
        n_principles: How many principles from :data:`DEFAULT_CONSTITUTION`
            to use.  Default ``3`` (first three are the most impactful).
        n_revisions: Number of critique-revise cycles.  Default ``1``.

    Returns:
        A configured :class:`ConstitutionalWrapper`.
    """
    return ConstitutionalWrapper(
        base_evaluator=base_evaluator,
        constitution=DEFAULT_CONSTITUTION[:n_principles],
        n_revisions=n_revisions,
    )
