"""Causal reasoning layer for Cambrian agents.

Allows agents to explicitly represent cause-effect relationships in their
strategies. Relationships take the form "IF <cause> THEN <effect>" and can
be extracted from free-form strategy text, evolved alongside genomes, and
injected back into system prompts.

Public API:
    - :class:`CausalEdge` — a single directed causal relationship.
    - :class:`CausalGraph` — a collection of causal edges with query, merge,
      prune, serialisation, and prompt-formatting helpers.
    - :class:`CausalStrategyExtractor` — LLM-powered extractor that turns
      strategy text into a :class:`CausalGraph`.
    - :class:`CausalMutator` — wraps :class:`~cambrian.mutator.LLMMutator`
      and also evolves the causal graph alongside the genome.
    - :func:`inject_causal_context` — helper that appends a graph's block to a
      genome's system prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.mutator import LLMMutator


# ---------------------------------------------------------------------------
# CausalEdge
# ---------------------------------------------------------------------------


@dataclass
class CausalEdge:
    """A single directed causal relationship.

    Attributes:
        cause: The antecedent condition ("IF …").
        effect: The consequent outcome ("THEN …").
        strength: Edge weight in ``[0, 1]``. 1.0 = certain causation.
        confidence: How certain we are that this relationship exists,
            also in ``[0, 1]``.
    """

    cause: str
    effect: str
    strength: float = 1.0
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalEdge":
        """Deserialise from a plain dictionary."""
        return cls(
            cause=str(data["cause"]),
            effect=str(data["effect"]),
            strength=float(data.get("strength", 1.0)),
            confidence=float(data.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# Keyword → strength mapping used by from_text
# ---------------------------------------------------------------------------

_STRENGTH_KEYWORDS: list[tuple[str, float]] = [
    ("always", 1.0),
    ("usually", 0.8),
    ("sometimes", 0.5),
    ("rarely", 0.2),
]

# Patterns to detect causal relationships (case-insensitive)
# Each compiled pattern must yield two named groups: `cause` and `effect`.
_CAUSAL_PATTERNS: list[re.Pattern[str]] = [
    # IF <cause> THEN <effect>
    re.compile(
        r"\bif\s+(?P<cause>.+?)\s+then\s+(?P<effect>[^.\n]+)",
        re.IGNORECASE,
    ),
    # <cause> → <effect>  (arrow, with or without spaces)
    re.compile(
        r"(?P<cause>[^.\n]+?)\s*→\s*(?P<effect>[^.\n]+)",
    ),
    # <cause> leads to <effect>
    re.compile(
        r"(?P<cause>[^.\n]+?)\s+leads\s+to\s+(?P<effect>[^.\n]+)",
        re.IGNORECASE,
    ),
    # <cause> causes <effect>
    re.compile(
        r"(?P<cause>[^.\n]+?)\s+causes\s+(?P<effect>[^.\n]+)",
        re.IGNORECASE,
    ),
]

_DEFAULT_STRENGTH = 0.7
_DEFAULT_CONFIDENCE = 0.8


def _infer_strength(text: str) -> float:
    """Return a strength value based on keyword hints found in *text*."""
    lower = text.lower()
    for keyword, value in _STRENGTH_KEYWORDS:
        if keyword in lower:
            return value
    return _DEFAULT_STRENGTH


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------


class CausalGraph:
    """Directed graph of causal relationships.

    Edges are stored in insertion order. Duplicate edges (same cause+effect
    pair) are silently deduplicated on add — the first one wins.
    """

    def __init__(self) -> None:
        # Primary store: list preserves insertion order.
        self._edges: list[CausalEdge] = []
        # Index for O(1) duplicate detection.
        self._edge_keys: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        """Add a causal edge, ignoring duplicates (same cause+effect key)."""
        key = (cause.strip(), effect.strip())
        if key in self._edge_keys:
            return
        self._edges.append(
            CausalEdge(
                cause=cause.strip(),
                effect=effect.strip(),
                strength=max(0.0, min(1.0, strength)),
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
        self._edge_keys.add(key)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_effects(self, cause: str) -> list[CausalEdge]:
        """Return all edges whose cause matches *cause* (exact, stripped)."""
        target = cause.strip()
        return [e for e in self._edges if e.cause == target]

    def get_causes(self, effect: str) -> list[CausalEdge]:
        """Return all edges whose effect matches *effect* (exact, stripped)."""
        target = effect.strip()
        return [e for e in self._edges if e.effect == target]

    # ------------------------------------------------------------------
    # Prompt integration
    # ------------------------------------------------------------------

    def to_prompt_block(self) -> str:
        """Format the graph as a block of ``IF X THEN Y`` lines.

        Each line ends with ``(strength=…, confidence=…)`` metadata.
        Returns an empty string when the graph has no edges.
        """
        if not self._edges:
            return ""
        lines: list[str] = []
        for edge in self._edges:
            lines.append(
                f"IF {edge.cause} THEN {edge.effect} "
                f"(strength={edge.strength:.2f}, confidence={edge.confidence:.2f})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "CausalGraph":
        """Parse LLM output and return a :class:`CausalGraph`.

        Scans for:
        - ``IF <cause> THEN <effect>``
        - ``<cause> → <effect>``
        - ``<cause> leads to <effect>``
        - ``<cause> causes <effect>``

        Strength is inferred from adverbs (always / usually / sometimes /
        rarely) found in the surrounding sentence; confidence defaults to
        :data:`_DEFAULT_CONFIDENCE`.
        """
        graph = cls()
        for pattern in _CAUSAL_PATTERNS:
            for match in pattern.finditer(text):
                cause = match.group("cause").strip()
                effect = match.group("effect").strip()
                # Strip trailing punctuation from the effect field
                effect = effect.rstrip(".,;:!?")
                if not cause or not effect:
                    continue
                # Derive strength from keywords present in the full match
                full_match = match.group(0)
                strength = _infer_strength(full_match)
                graph.add_edge(
                    cause=cause,
                    effect=effect,
                    strength=strength,
                    confidence=_DEFAULT_CONFIDENCE,
                )
        return graph

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def merge(self, other: "CausalGraph") -> "CausalGraph":
        """Return a new graph containing the union of edges from both graphs.

        When the same cause+effect pair appears in both, the first insertion
        (from *self*) is kept.
        """
        merged = CausalGraph()
        for edge in self._edges:
            merged.add_edge(edge.cause, edge.effect, edge.strength, edge.confidence)
        for edge in other._edges:
            merged.add_edge(edge.cause, edge.effect, edge.strength, edge.confidence)
        return merged

    def prune(
        self,
        min_strength: float = 0.3,
        min_confidence: float = 0.3,
    ) -> "CausalGraph":
        """Return a new graph with weak or uncertain edges removed.

        Args:
            min_strength: Edges with ``strength < min_strength`` are dropped.
            min_confidence: Edges with ``confidence < min_confidence`` are dropped.
        """
        pruned = CausalGraph()
        for edge in self._edges:
            if edge.strength >= min_strength and edge.confidence >= min_confidence:
                pruned.add_edge(
                    edge.cause, edge.effect, edge.strength, edge.confidence
                )
        return pruned

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a JSON-compatible dictionary."""
        return {"edges": [e.to_dict() for e in self._edges]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalGraph":
        """Deserialise a :class:`CausalGraph` from a plain dictionary."""
        graph = cls()
        for edge_data in data.get("edges", []):
            edge = CausalEdge.from_dict(edge_data)
            graph.add_edge(edge.cause, edge.effect, edge.strength, edge.confidence)
        return graph

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return f"CausalGraph(edges={len(self._edges)})"


# ---------------------------------------------------------------------------
# CausalStrategyExtractor
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are an expert in causal reasoning. "
    "Given a description of an agent's strategy, identify the causal "
    "relationships embedded in it. "
    "Output ONLY a list of IF-THEN statements, one per line, using the exact "
    "format: IF <cause> THEN <effect>. "
    "Use adverbs like 'always', 'usually', 'sometimes', or 'rarely' in the "
    "<cause> or <effect> when appropriate to express probability. "
    "Do not include any other text."
)

_EXTRACT_TEMPLATE = """Strategy text:
{strategy_text}

Task context:
{task}

List all IF-THEN causal relationships present in the strategy (one per line):"""


class CausalStrategyExtractor:
    """Uses an LLM to extract causal relationships from an agent's strategy text.

    Args:
        backend: The LLM backend used for extraction.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def extract(self, strategy_text: str, task: str = "") -> CausalGraph:
        """Extract causal relationships from *strategy_text*.

        Sends *strategy_text* (plus optional *task* context) to the LLM and
        asks it to produce IF-THEN lines, then parses the result via
        :meth:`CausalGraph.from_text`.

        Args:
            strategy_text: The agent's strategy description.
            task: Optional task context to help the LLM focus the extraction.

        Returns:
            A :class:`CausalGraph` with the extracted relationships.
        """
        prompt = _EXTRACT_TEMPLATE.format(
            strategy_text=strategy_text,
            task=task or "general problem solving",
        )
        raw = self._backend.generate(
            prompt,
            system=_EXTRACT_SYSTEM,
            temperature=0.2,  # deterministic extraction
        )
        return CausalGraph.from_text(raw)


# ---------------------------------------------------------------------------
# CausalMutator
# ---------------------------------------------------------------------------


class CausalMutator:
    """Mutation operator that evolves causal graphs alongside genomes.

    Wraps :class:`~cambrian.mutator.LLMMutator` for genome mutation and
    :class:`CausalStrategyExtractor` for causal graph extraction.

    Args:
        base_mutator: The underlying genome mutator.
        extractor: Extractor used to derive causal graphs from mutated strategies.
    """

    def __init__(
        self,
        base_mutator: LLMMutator,
        extractor: CausalStrategyExtractor,
    ) -> None:
        self._mutator = base_mutator
        self._extractor = extractor

    def mutate_with_causality(
        self,
        agent: Agent,
        task: str = "",
    ) -> tuple[Agent, CausalGraph]:
        """Mutate *agent*'s genome and extract its causal graph.

        Steps:
        1. Mutate the genome normally via :class:`~cambrian.mutator.LLMMutator`.
        2. Extract a :class:`CausalGraph` from the new strategy text.
        3. Return both.

        Args:
            agent: The agent to mutate.
            task: Task description forwarded to both mutator and extractor.

        Returns:
            A ``(mutated_agent, causal_graph)`` tuple. The original agent is
            not modified.
        """
        mutated_agent = self._mutator.mutate(agent, task=task)
        causal_graph = self._extractor.extract(
            strategy_text=mutated_agent.genome.strategy,
            task=task,
        )
        return mutated_agent, causal_graph


# ---------------------------------------------------------------------------
# inject_causal_context
# ---------------------------------------------------------------------------

_CAUSAL_HEADER = "\n\n--- Causal context ---\n"


def inject_causal_context(genome: Genome, graph: CausalGraph) -> Genome:
    """Return a new :class:`~cambrian.agent.Genome` with causal context injected.

    Appends the graph's :meth:`~CausalGraph.to_prompt_block` output to the
    genome's ``system_prompt`` under a ``--- Causal context ---`` header.
    If the graph is empty, the genome is returned unchanged (as a clone).

    Args:
        genome: The source genome to clone and augment.
        graph: The causal graph to embed.

    Returns:
        A new :class:`~cambrian.agent.Genome` instance (the original is not
        modified).
    """
    new_genome = Genome.from_dict(genome.to_dict())
    block = graph.to_prompt_block()
    if block:
        new_genome.system_prompt = genome.system_prompt + _CAUSAL_HEADER + block
    return new_genome
