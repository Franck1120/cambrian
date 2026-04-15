"""Glossolalia Emergente — Technique 56.

Named after the phenomenon of "speaking in tongues", this technique invites
the LLM to first produce an unconstrained internal monologue — metaphors,
symbols, half-formed thoughts, associative leaps — before translating that
latent stream into a coherent answer.  The unconstrained phase can surface
non-obvious connections the model would normally suppress to appear fluent.

Design
------
GlossaloliaReasoner
    Two-step prompting:
    1. **Latent phase** — LLM produces an unconstrained stream with no
       grammar/format requirements (brevity kept via ``max_latent_tokens``).
    2. **Synthesis phase** — LLM translates the latent stream into a
       final, structured answer.

    ``latent_temperature`` is deliberately higher than ``synth_temperature``
    to encourage associative exploration.

GlossaloliaEvaluator
    Wraps any base evaluator and runs its ``evaluate()`` through
    ``GlossaloliaReasoner`` before scoring, so the agent's response
    benefits from the two-phase reasoning.

Usage::

    from cambrian.glossolalia import GlossaloliaReasoner

    reasoner = GlossaloliaReasoner(backend=backend, latent_temperature=1.2)
    answer, latent = reasoner.reason(task, system_prompt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cambrian.agent import Agent
from cambrian.evaluator import Evaluator

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LATENT_SYSTEM = (
    "You are thinking aloud in an unconstrained internal space.  "
    "Do NOT write a polished answer yet.  Instead, produce raw, associative, "
    "stream-of-consciousness thoughts: metaphors, fragments, connections, "
    "wild hypotheses.  Grammar, coherence, and structure do NOT matter here.  "
    "Just think freely."
)

_LATENT_TEMPLATE = "TASK: {task}\n\nBegin your unconstrained internal monologue:"

_SYNTH_SYSTEM = (
    "You are a translator from raw thought to clear language.  "
    "Given an unstructured internal monologue, extract the most useful "
    "insights and synthesise a clear, accurate, well-structured answer to the task."
)

_SYNTH_TEMPLATE = """\
ORIGINAL TASK: {task}

INTERNAL MONOLOGUE (raw, unstructured):
{latent}

Now write the final, clear answer to the task using insights from the monologue:
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GlossaloliaResult:
    """Stores both the latent stream and the synthesised answer."""

    latent_stream: str
    final_answer: str
    task: str


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------


class GlossaloliaReasoner:
    """Two-phase unconstrained-then-structured reasoner.

    Parameters
    ----------
    backend:
        LLM backend for both phases.
    latent_temperature:
        High temperature for the unconstrained phase (default 1.2).
    synth_temperature:
        Lower temperature for the synthesis phase (default 0.6).
    max_latent_tokens:
        Rough character budget for the latent stream (used as context hint
        in the prompt; actual truncation is left to the backend).
    """

    def __init__(
        self,
        backend: "LLMBackend",
        latent_temperature: float = 1.2,
        synth_temperature: float = 0.6,
        max_latent_tokens: int = 300,
    ) -> None:
        self._backend = backend
        self._latent_temp = latent_temperature
        self._synth_temp = synth_temperature
        self._max_latent = max_latent_tokens
        self._history: list[GlossaloliaResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[GlossaloliaResult]:
        """Return a copy of all reasoning results."""
        return list(self._history)

    def reason(
        self,
        task: str,
        system_prompt: str = "",
        temperature_override: Optional[float] = None,
    ) -> tuple[str, str]:
        """Run the two-phase reasoning pipeline.

        Returns
        -------
        (final_answer, latent_stream)
        """
        latent = self._latent_phase(task, system_prompt, temperature_override)
        answer = self._synth_phase(task, latent, temperature_override)

        self._history.append(
            GlossaloliaResult(
                latent_stream=latent,
                final_answer=answer,
                task=task,
            )
        )
        return answer, latent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _latent_phase(
        self, task: str, system_prompt: str, temp_override: Optional[float]
    ) -> str:
        temp = temp_override if temp_override is not None else self._latent_temp
        user_msg = _LATENT_TEMPLATE.format(task=task)
        combined_system = (
            f"{system_prompt}\n\n{_LATENT_SYSTEM}"
            if system_prompt
            else _LATENT_SYSTEM
        )
        try:
            latent = str(
                self._backend.generate(
                    f"{combined_system}\n\n{user_msg}",
                    temperature=temp,
                )
            )
        except Exception:  # noqa: BLE001
            latent = f"[latent unavailable for: {task}]"
        # Truncate latent to max_latent_tokens characters (rough budget)
        return latent[: self._max_latent * 4]

    def _synth_phase(
        self, task: str, latent: str, temp_override: Optional[float]
    ) -> str:
        temp = temp_override if temp_override is not None else self._synth_temp
        user_msg = _SYNTH_TEMPLATE.format(task=task, latent=latent)
        try:
            return str(
                self._backend.generate(
                    f"{_SYNTH_SYSTEM}\n\n{user_msg}",
                    temperature=temp,
                )
            )
        except Exception:  # noqa: BLE001
            return latent  # degrade gracefully to the latent stream


class GlossaloliaEvaluator(Evaluator):
    """Evaluator that pipes agent responses through GlossaloliaReasoner.

    Parameters
    ----------
    base_evaluator:
        Any ``Evaluator`` used for final scoring.
    reasoner:
        A ``GlossaloliaReasoner`` instance.
    """

    def __init__(
        self,
        base_evaluator: Evaluator,
        reasoner: GlossaloliaReasoner,
    ) -> None:
        self._base = base_evaluator
        self._reasoner = reasoner

    def evaluate(self, agent: Agent, task: str) -> float:
        """Run two-phase reasoning then delegate scoring to base evaluator."""
        # First generate a response normally
        try:
            raw_response = agent.run(task)
        except Exception:  # noqa: BLE001
            raw_response = ""

        # Enhance through glossolalia
        enhanced_response, _ = self._reasoner.reason(
            task=f"{task}\n\nInitial response to refine:\n{raw_response}",
            system_prompt=agent.genome.system_prompt,
        )

        # Temporarily substitute the agent's run() result for scoring
        # by creating a mock agent-like wrapper
        class _PatchedAgent:
            def __init__(self, inner: Agent, resp: str) -> None:
                self._inner = inner
                self._resp = resp
                self.genome = inner.genome
                self.agent_id = inner.agent_id
                self.fitness = inner.fitness

            def run(self, _task: str) -> str:
                return self._resp

        patched = _PatchedAgent(agent, enhanced_response)
        return self._base.evaluate(patched, task)  # type: ignore[arg-type]
