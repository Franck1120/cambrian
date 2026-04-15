"""Reflexion — self-critique and revision loop for agent responses.

A :class:`ReflexionAgent` wraps a base agent and runs a generate → reflect → revise
cycle.  After each initial response, the LLM critiques its own answer and
produces an improved version.  Multiple rounds of reflection further improve quality.

Based on:
  Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"

Usage::

    from cambrian.reflexion import ReflexionAgent

    reflexion = ReflexionAgent(agent=my_agent, n_rounds=2)
    result = reflexion.run("Write a haiku about the ocean")
    print(result.final_response)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cambrian.backends.base import LLMBackend
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent

logger = get_logger(__name__)


# ── ReflexionRound ─────────────────────────────────────────────────────────────


@dataclass
class ReflexionRound:
    """A single round of the Reflexion cycle.

    Attributes:
        round_number: 0-indexed round number.
        response: The agent's response in this round.
        critique: The self-critique produced in this round (empty for round 0).
        improved: Whether the critique led to revision.
    """

    round_number: int
    response: str
    critique: str = ""
    improved: bool = False


# ── ReflexionResult ────────────────────────────────────────────────────────────


@dataclass
class ReflexionResult:
    """Full result of a Reflexion run.

    Attributes:
        final_response: The response after all rounds of reflection.
        rounds: List of :class:`ReflexionRound` objects.
        task: The original task.
        n_rounds_used: Actual number of rounds run (may be less than max).
    """

    final_response: str
    rounds: list[ReflexionRound] = field(default_factory=list)
    task: str = ""
    n_rounds_used: int = 0

    @property
    def initial_response(self) -> str:
        """The response before any reflection."""
        return self.rounds[0].response if self.rounds else ""

    @property
    def improved(self) -> bool:
        """True if at least one round produced a non-trivial improvement."""
        return any(r.improved for r in self.rounds[1:])


# ── ReflexionAgent ─────────────────────────────────────────────────────────────


_CRITIQUE_SYSTEM = """You are a rigorous self-critic.
Analyse the given response to a task and identify:
1. Factual errors or inaccuracies
2. Missing important information
3. Unclear or ambiguous phrasing
4. Logical inconsistencies
5. Ways to make the response more complete and helpful

Be specific and actionable. If the response is already excellent, say so briefly."""

_CRITIQUE_TEMPLATE = """Task: {task}

Response to critique:
{response}

Provide a specific, actionable critique. Focus on what is WRONG or MISSING.
If the response is excellent with no significant issues, respond with: "EXCELLENT: No significant issues found."
Otherwise, list the specific problems and how to fix them."""

_REVISE_SYSTEM = """You are an expert who improves responses based on critique.
Given a task, an initial response, and a critique, produce an improved response.
Address all issues raised in the critique. Make the response better.
Return ONLY the improved response — no meta-commentary, no "Improved response:" prefix."""

_REVISE_TEMPLATE = """Task: {task}

Initial response:
{response}

Critique of the response:
{critique}

Produce an improved response that addresses all critique points.
Return ONLY the improved response."""

_EXCELLENCE_MARKER = "EXCELLENT:"


class ReflexionAgent:
    """Wraps an agent with a generate → critique → revise cycle.

    Args:
        agent: The base agent to run and improve. Must have a backend.
        n_rounds: Maximum number of reflection rounds. Default 2.
            Round 0 = initial generation (no reflection).
            Round 1 = first critique + revision.
            Round N = N-th critique + revision.
        critique_temperature: Temperature for critique generation. Default 0.3.
        revision_temperature: Temperature for revision. Default 0.5.
        stop_if_excellent: If True, stop early when the critique says the
            response is excellent. Default True.
        reflection_backend: Optional separate backend for critique/revision.
            If ``None``, uses the agent's own backend.
    """

    def __init__(
        self,
        agent: "Agent",
        n_rounds: int = 2,
        critique_temperature: float = 0.3,
        revision_temperature: float = 0.5,
        stop_if_excellent: bool = True,
        reflection_backend: "LLMBackend | None" = None,
    ) -> None:
        self._agent = agent
        self._n_rounds = max(0, n_rounds)
        self._critique_temp = critique_temperature
        self._revision_temp = revision_temperature
        self._stop_excellent = stop_if_excellent
        self._reflection_backend = reflection_backend or agent.backend

    def run(self, task: str) -> ReflexionResult:
        """Run the generate → critique → revise cycle on *task*.

        Args:
            task: The task to solve.

        Returns:
            A :class:`ReflexionResult` with all rounds and the final response.
        """
        # Round 0: initial generation
        initial = self._agent.run(task)
        rounds = [ReflexionRound(round_number=0, response=initial)]
        current_response = initial

        for i in range(1, self._n_rounds + 1):
            critique = self._critique(task, current_response)
            is_excellent = critique.startswith(_EXCELLENCE_MARKER)

            if is_excellent and self._stop_excellent:
                rounds[-1].critique = critique
                break

            if critique.strip():
                revised = self._revise(task, current_response, critique)
                improved = revised.strip() != current_response.strip()
                rounds.append(
                    ReflexionRound(
                        round_number=i,
                        response=revised,
                        critique=critique,
                        improved=improved,
                    )
                )
                current_response = revised
            else:
                break

        return ReflexionResult(
            final_response=current_response,
            rounds=rounds,
            task=task,
            n_rounds_used=len(rounds),
        )

    def _critique(self, task: str, response: str) -> str:
        """Ask the LLM to critique the response."""
        prompt = _CRITIQUE_TEMPLATE.format(task=task, response=response[:1500])
        try:
            return self._reflection_backend.generate(
                prompt,
                system=_CRITIQUE_SYSTEM,
                temperature=self._critique_temp,
            )
        except Exception as exc:
            logger.warning("ReflexionAgent._critique failed: %s", exc)
            return ""

    def _revise(self, task: str, response: str, critique: str) -> str:
        """Ask the LLM to revise the response based on the critique."""
        prompt = _REVISE_TEMPLATE.format(
            task=task,
            response=response[:1500],
            critique=critique[:800],
        )
        try:
            return self._reflection_backend.generate(
                prompt,
                system=_REVISE_SYSTEM,
                temperature=self._revision_temp,
            )
        except Exception as exc:
            logger.warning("ReflexionAgent._revise failed: %s", exc)
            return response  # fallback: return original


# ── ReflexionEvaluator ─────────────────────────────────────────────────────────


class ReflexionEvaluator:
    """Wraps any evaluator and applies Reflexion before scoring.

    The agent's response is first improved via Reflexion, then the improved
    response is used for scoring.  This can significantly improve evaluation
    scores for tasks where quality of reasoning matters more than first-shot
    accuracy.

    Args:
        base_evaluator: The evaluator that scores the final response.
        n_rounds: Number of Reflexion rounds. Default 2.
        reflection_backend: Optional backend for critique/revision.
    """

    def __init__(
        self,
        base_evaluator: Any,
        n_rounds: int = 2,
        reflection_backend: "LLMBackend | None" = None,
    ) -> None:
        self._evaluator = base_evaluator
        self._n_rounds = n_rounds
        self._reflection_backend = reflection_backend

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Run Reflexion on *agent*, then score the improved response.

        Args:
            agent: The agent to evaluate.
            task: The task to solve.

        Returns:
            Fitness score from the base evaluator, after Reflexion improvement.
        """
        backend = self._reflection_backend or agent.backend
        if backend is None:
            # No backend available — fall through to base evaluator
            return float(self._evaluator.evaluate(agent, task))

        reflexion = ReflexionAgent(
            agent=agent,
            n_rounds=self._n_rounds,
            reflection_backend=backend,
        )
        result = reflexion.run(task)
        # Temporarily inject the improved response back into the evaluation
        # by creating a thin wrapper agent that returns the improved response
        original_run = agent.run

        def _patched_run(_task: str) -> str:
            return result.final_response

        try:
            agent.run = _patched_run  # type: ignore[method-assign]
            score = float(self._evaluator.evaluate(agent, task))
        finally:
            agent.run = original_run  # type: ignore[method-assign]

        return score
