"""SAHOO-inspired safeguards with goal drift and anomaly detection.

Provides three complementary safety mechanisms for Cambrian's evolutionary
framework:

* :class:`GoalDriftDetector` — measures semantic divergence between an agent's
  original intent and its current system prompt using word-overlap Jaccard
  similarity (no external dependencies).
* :class:`FitnessAnomalyDetector` — flags agents whose fitness spikes beyond a
  z-score threshold, indicating potential reward hacking.
* :class:`SafeguardController` — orchestrates both detectors and offers an
  LLM-backed prompt-remediation path.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from typing import TYPE_CHECKING

from cambrian.agent import Agent
from cambrian.backends.base import LLMBackend

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase word tokens from *text*, punctuation stripped.

    Args:
        text: Raw string to tokenize.

    Returns:
        A set of unique lowercase tokens.
    """
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute the Jaccard similarity between two token sets.

    Args:
        a: First token set.
        b: Second token set.

    Returns:
        Similarity in ``[0.0, 1.0]``. Returns ``0.0`` when both sets are empty.
    """
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# DriftEvent
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    """Record of a single drift measurement for one agent at one generation.

    Attributes:
        agent_id: Identifier of the measured agent.
        generation: Evolution generation at measurement time.
        drift_score: Divergence from original intent, in ``[0.0, 1.0]``.
            ``0.0`` means identical; ``1.0`` means zero word overlap.
        original_intent: Short description of the agent's registered intent.
        current_prompt: The agent's current ``system_prompt``.
        flagged: ``True`` when *drift_score* exceeds the detector's threshold.
    """

    agent_id: str
    generation: int
    drift_score: float
    original_intent: str
    current_prompt: str
    flagged: bool


# ---------------------------------------------------------------------------
# GoalDriftDetector
# ---------------------------------------------------------------------------

class GoalDriftDetector:
    """Detect semantic drift between an agent's original intent and its prompt.

    Uses word-overlap Jaccard similarity as a lightweight, dependency-free
    proxy for semantic similarity, inspired by SAHOO alignment-monitoring
    principles.

    Args:
        drift_threshold: Drift score above which an event is flagged.
            Must be in ``(0.0, 1.0]``. Defaults to ``0.4``.
        window: Number of past :class:`DriftEvent` records to retain per
            agent for trend analysis. Defaults to ``5``.
    """

    def __init__(
        self,
        drift_threshold: float = 0.4,
        window: int = 5,
    ) -> None:
        if not (0.0 < drift_threshold <= 1.0):
            raise ValueError(
                f"drift_threshold must be in (0.0, 1.0], got {drift_threshold}"
            )
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        self._drift_threshold = drift_threshold
        self._window = window

        # agent_id → original intent string
        self._intents: dict[str, str] = {}
        # agent_id → tokenized intent (cached)
        self._intent_tokens: dict[str, set[str]] = {}
        # chronological list of all DriftEvents
        self._events: list[DriftEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, agent: Agent, intent: str) -> None:
        """Register the original intent for *agent*.

        Must be called before :meth:`measure` for a given agent. Re-registering
        an agent replaces the previous intent.

        Args:
            agent: The agent to register.
            intent: A short natural-language description of the agent's goal
                (e.g. ``"Summarise scientific papers concisely"``).
        """
        self._intents[agent.agent_id] = intent
        self._intent_tokens[agent.agent_id] = _tokenize(intent)

    def measure(self, agent: Agent, generation: int) -> DriftEvent:
        """Measure drift for *agent* and return a :class:`DriftEvent`.

        Drift score is computed as::

            drift_score = 1.0 - jaccard(intent_tokens, current_prompt_tokens)

        The event is appended to :attr:`events` before being returned.

        Args:
            agent: Agent to measure. Must have been previously registered.
            generation: Current evolution generation.

        Returns:
            A :class:`DriftEvent` with ``flagged=True`` when
            ``drift_score > drift_threshold``.

        Raises:
            KeyError: If *agent* has not been registered via :meth:`register`.
        """
        agent_id = agent.agent_id
        if agent_id not in self._intents:
            raise KeyError(
                f"Agent '{agent_id}' has not been registered. "
                "Call register() first."
            )

        intent_tokens = self._intent_tokens[agent_id]
        current_prompt = agent.genome.system_prompt
        current_tokens = _tokenize(current_prompt)

        similarity = _jaccard(intent_tokens, current_tokens)
        drift_score = 1.0 - similarity
        # Clamp to [0.0, 1.0] to guard against floating-point edge cases
        drift_score = max(0.0, min(1.0, drift_score))

        event = DriftEvent(
            agent_id=agent_id,
            generation=generation,
            drift_score=drift_score,
            original_intent=self._intents[agent_id],
            current_prompt=current_prompt,
            flagged=drift_score > self._drift_threshold,
        )
        self._events.append(event)
        return event

    def scan_population(
        self,
        population: list[Agent],
        generation: int,
    ) -> list[DriftEvent]:
        """Measure drift for all registered agents in *population*.

        Agents not previously registered via :meth:`register` are silently
        skipped, making it safe to call with a mixed population.

        Args:
            population: All agents in the current generation.
            generation: Current evolution generation.

        Returns:
            List of :class:`DriftEvent` where ``flagged`` is ``True``,
            in population order.
        """
        flagged: list[DriftEvent] = []
        for agent in population:
            if agent.agent_id not in self._intents:
                continue
            event = self.measure(agent, generation)
            if event.flagged:
                flagged.append(event)
        return flagged

    @property
    def events(self) -> list[DriftEvent]:
        """All recorded :class:`DriftEvent` objects in chronological order.

        Returns:
            A copy of the internal event list to prevent external mutation.
        """
        return list(self._events)


# ---------------------------------------------------------------------------
# FitnessAnomalyDetector
# ---------------------------------------------------------------------------

class FitnessAnomalyDetector:
    """Detect reward-hacking via statistical anomaly detection on fitness.

    An agent's current fitness is considered anomalous when it exceeds
    ``mean + z_threshold * std`` of that agent's own fitness history,
    provided at least *min_history* data points have been recorded.

    Args:
        z_threshold: Number of standard deviations above the mean required
            to flag an anomaly. Defaults to ``2.5``.
        min_history: Minimum number of recorded fitness values before
            anomaly detection activates. Defaults to ``5``.
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        min_history: int = 5,
    ) -> None:
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {z_threshold}")
        if min_history < 2:
            raise ValueError(f"min_history must be >= 2, got {min_history}")

        self._z_threshold = z_threshold
        self._min_history = min_history
        # agent_id → ordered list of historical fitness values
        self._history: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, agent: Agent, generation: int) -> None:  # noqa: ARG002
        """Append the agent's current fitness to its history.

        ``Agent.fitness = None`` is treated as ``0.0``.

        Args:
            agent: Agent whose fitness to record.
            generation: Current evolution generation (retained for API
                symmetry; not stored separately).
        """
        fitness_value = agent.fitness if agent.fitness is not None else 0.0
        self._history[agent.agent_id].append(fitness_value)

    def is_anomalous(self, agent: Agent) -> bool:
        """Return ``True`` if the agent's current fitness is statistically anomalous.

        Requires at least *min_history* historical values for the agent.
        When history is insufficient, always returns ``False``.

        Args:
            agent: Agent to evaluate. Its current ``fitness`` is compared
                against its own recorded history.

        Returns:
            ``True`` when ``fitness > mean + z_threshold * std``.
        """
        history = self._history.get(agent.agent_id, [])
        if len(history) < self._min_history:
            return False

        current = agent.fitness if agent.fitness is not None else 0.0
        mu = mean(history)
        # stdev requires at least 2 values; min_history >= 2 guarantees this
        sigma = stdev(history)

        if sigma == 0.0:
            # No variation in history → any difference is technically infinite
            # z-score; only flag if current differs from the constant history.
            return current > mu

        z_score = (current - mu) / sigma
        return z_score > self._z_threshold

    def scan(
        self,
        population: list[Agent],
        generation: int,
    ) -> list[str]:
        """Record fitness for all agents then return IDs of anomalous ones.

        Args:
            population: All agents in the current generation.
            generation: Current evolution generation.

        Returns:
            List of ``agent_id`` strings for anomalous agents, in
            population order.
        """
        for agent in population:
            self.record(agent, generation)

        return [
            agent.agent_id
            for agent in population
            if self.is_anomalous(agent)
        ]


# ---------------------------------------------------------------------------
# SafeguardController
# ---------------------------------------------------------------------------

class SafeguardController:
    """Orchestrate drift and anomaly detection across an evolving population.

    Combines :class:`GoalDriftDetector` and :class:`FitnessAnomalyDetector`
    into a single entry-point, and provides optional LLM-backed prompt
    remediation.

    Args:
        drift_detector: Configured :class:`GoalDriftDetector` instance.
        anomaly_detector: Configured :class:`FitnessAnomalyDetector` instance.
        backend: Optional :class:`~cambrian.backends.base.LLMBackend` used for
            prompt remediation. When ``None``, :meth:`remediate` returns an
            unchanged clone.
    """

    _REMEDIATION_PROMPT_TEMPLATE = (
        "You are a safety controller for an evolutionary AI framework.\n"
        "An agent's system prompt has drifted from its original intent.\n\n"
        "Original intent:\n{intent}\n\n"
        "Current (drifted) system prompt:\n{prompt}\n\n"
        "Task the agent is solving:\n{task}\n\n"
        "Rewrite the system prompt so it faithfully serves the original intent "
        "while remaining useful for the task. Output only the rewritten system "
        "prompt, nothing else."
    )

    def __init__(
        self,
        drift_detector: GoalDriftDetector,
        anomaly_detector: FitnessAnomalyDetector,
        backend: LLMBackend | None = None,
    ) -> None:
        self._drift_detector = drift_detector
        self._anomaly_detector = anomaly_detector
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        population: list[Agent],
        generation: int,
    ) -> dict[str, list[DriftEvent] | list[str]]:
        """Run all safeguard checks on *population*.

        Calls :meth:`GoalDriftDetector.scan_population` and
        :meth:`FitnessAnomalyDetector.scan` in sequence.

        Args:
            population: All agents in the current generation.
            generation: Current evolution generation.

        Returns:
            A dictionary with two keys:

            * ``"drift"`` — list of flagged :class:`DriftEvent` objects.
            * ``"anomalies"`` — list of anomalous ``agent_id`` strings.
        """
        drift_events = self._drift_detector.scan_population(population, generation)
        anomalous_ids = self._anomaly_detector.scan(population, generation)
        return {
            "drift": drift_events,
            "anomalies": anomalous_ids,
        }

    def remediate(self, agent: Agent, task: str) -> Agent:
        """Return a new agent whose prompt is re-aligned to its original intent.

        When a :class:`~cambrian.backends.base.LLMBackend` is available and the
        agent has a registered intent, the backend rewrites the system prompt.
        Otherwise a plain clone is returned unchanged.

        Args:
            agent: The agent to remediate.
            task: The task context used to guide prompt re-alignment.

        Returns:
            A cloned :class:`Agent` with a (potentially) corrected system
            prompt, a fresh ``agent_id``, and ``fitness=None``.
        """
        clone = agent.clone()
        clone.agent_id = str(uuid.uuid4())[:8]

        if self._backend is None:
            return clone

        original_intent = self._drift_detector._intents.get(agent.agent_id)
        if original_intent is None:
            # Agent was never registered — cannot remediate meaningfully.
            return clone

        remediation_prompt = self._REMEDIATION_PROMPT_TEMPLATE.format(
            intent=original_intent,
            prompt=agent.genome.system_prompt,
            task=task,
        )

        try:
            new_prompt = self._backend.generate(remediation_prompt)
        except Exception:  # noqa: BLE001
            # Backend failure is non-fatal; fall back to the unmodified clone.
            return clone

        clone.genome.system_prompt = new_prompt.strip()
        return clone
