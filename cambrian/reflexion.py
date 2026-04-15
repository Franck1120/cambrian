# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Reflexion — self-critique loop for iterative response improvement.

Reflexion (Technique 35) implements the *generate → reflect → revise* cycle
from Shinn et al. (2023).  An agent produces an initial response, then
evaluates its own output against a rubric, identifies weaknesses, and revises
the response.  This loop runs for a configurable number of iterations.

Usage::

    from cambrian.reflexion import ReflexionEvaluator
    from cambrian.backends.openai_compat import OpenAICompatBackend
    from cambrian.agent import Agent, Genome

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    agent = Agent(genome=Genome(system_prompt="You are a helpful assistant."))
    evaluator = ReflexionEvaluator(backend=backend, n_reflections=2)

    final_response, score = evaluator.evaluate(agent, task="Write a haiku about AI.")
    print(score, final_response)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent
    from cambrian.backends.base import LLMBackend

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_REFLECT_SYSTEM = """\
You are a critical evaluator.  You will receive a task and a response produced
by an AI agent.  Your job is to:

1. Identify specific weaknesses, errors, or missed requirements in the response.
2. Provide a concise critique (3–5 bullet points).
3. Rate the response on a scale from 0.0 (terrible) to 1.0 (perfect).

Respond in this exact format:
CRITIQUE:
- <weakness 1>
- <weakness 2>
...

SCORE: <float>
"""

_REVISE_SYSTEM = """\
You are an expert reviser.  You will receive a task, an original response,
and a critique pointing out weaknesses.  Your job is to produce an improved
version of the response that addresses every point in the critique.

Return ONLY the improved response — no preamble, no explanation.
"""

_REVISE_TEMPLATE = """\
Task: {task}

Original response:
{response}

Critique:
{critique}

Write an improved response that addresses every weakness in the critique.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ReflexionEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class ReflexionEvaluator:
    """Iterative generate → reflect → revise evaluator.

    Args:
        backend: LLM backend for generation, reflection, and revision.
        n_reflections: Number of critique-revise cycles. Default ``2``.
        reflect_temperature: Temperature for the critique call. Default ``0.2``.
        revise_temperature: Temperature for the revision call. Default ``0.5``.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        n_reflections: int = 2,
        reflect_temperature: float = 0.2,
        revise_temperature: float = 0.5,
    ) -> None:
        self._backend = backend
        self._n = n_reflections
        self._reflect_temp = reflect_temperature
        self._revise_temp = revise_temperature

    def evaluate(
        self,
        agent: "Agent",
        task: str,
    ) -> tuple[str, float]:
        """Run the reflexion loop and return the final response and score.

        The agent generates an initial response.  Then for each reflection
        cycle:
        - The critique model assesses the response and assigns a score.
        - If the score is < 1.0 and more cycles remain, the revision model
          improves the response.

        Args:
            agent: The agent whose genome drives the initial generation.
            task: The task / question for the agent.

        Returns:
            ``(final_response, final_score)`` where score is in ``[0.0, 1.0]``.
        """
        # Step 1: Initial generation
        try:
            response = self._backend.generate(
                task,
                system=agent.genome.system_prompt,
                temperature=agent.genome.temperature,
            )
        except Exception as exc:
            logger.warning("ReflexionEvaluator initial gen failed: %s", exc)
            return "", 0.0

        final_score = 0.5  # Neutral default if critique never runs
        critique = ""

        # Step 2: Reflect + revise loop
        for cycle in range(self._n):
            critique, score = self._reflect(task, response)
            final_score = score
            logger.debug(
                "Reflexion cycle %d/%d: score=%.3f", cycle + 1, self._n, score
            )

            if score >= 1.0:
                break  # Perfect — no need to revise

            # Step 3: Revise (even on the last cycle — caller sees revised output)
            try:
                response = self._revise(task, response, critique)
            except Exception as exc:
                logger.warning("ReflexionEvaluator revision failed: %s", exc)
                break  # Keep current response

        return response, final_score

    # ── Internals ─────────────────────────────────────────────────────────────

    def _reflect(self, task: str, response: str) -> tuple[str, float]:
        """Ask the LLM to critique *response* and return (critique, score)."""
        prompt = (
            f"Task: {task}\n\nResponse to evaluate:\n{response}"
        )
        try:
            raw = self._backend.generate(
                prompt,
                system=_REFLECT_SYSTEM,
                temperature=self._reflect_temp,
            )
            critique, score = self._parse_reflection(raw)
        except Exception as exc:
            logger.warning("ReflexionEvaluator reflection call failed: %s", exc)
            critique = ""
            score = 0.5

        return critique, score

    def _revise(self, task: str, response: str, critique: str) -> str:
        """Ask the LLM to revise *response* given the *critique*."""
        prompt = _REVISE_TEMPLATE.format(
            task=task, response=response, critique=critique
        )
        return self._backend.generate(
            prompt,
            system=_REVISE_SYSTEM,
            temperature=self._revise_temp,
        )

    @staticmethod
    def _parse_reflection(raw: str) -> tuple[str, float]:
        """Extract critique text and score from the reflection response.

        Expects:
            CRITIQUE:
            - ...
            SCORE: <float>

        Returns:
            ``(critique_text, score_float)``.  Falls back to ``(raw, 0.5)``
            on parse failure.
        """
        import re

        # Extract score
        score_match = re.search(r"SCORE\s*:\s*(-?[\d.]+)", raw, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.5
        else:
            score = 0.5

        # Extract critique section
        critique_match = re.search(
            r"CRITIQUE\s*:\s*([\s\S]*?)(?=SCORE\s*:|$)", raw, re.IGNORECASE
        )
        critique = critique_match.group(1).strip() if critique_match else raw.strip()

        return critique, score
