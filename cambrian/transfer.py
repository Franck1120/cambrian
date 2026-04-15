"""Transfer Learning — Technique 58.

A source agent that has been trained (evolved) on one task domain donates
its genome as a "pretrained backbone".  A lightweight LLM-based adapter
then fine-tunes the backbone system prompt for a new target task, preserving
broadly useful behaviours while adding task-specific knowledge.

Components
----------
TransferAdapter
    Extracts the transferable portion of a source genome (strategy +
    system-prompt skeleton) and adapts it to the target task via an LLM
    call.  The adapter intensity is configurable: ``light`` keeps ~80% of
    the original prompt, ``medium`` rewrites ~50%, ``heavy`` produces a
    near-fresh prompt inspired by the source.

TransferBank
    Stores pre-trained source genomes indexed by domain tag so they can be
    retrieved and re-used across multiple target tasks.

Usage::

    from cambrian.transfer import TransferAdapter, TransferBank

    bank = TransferBank()
    bank.register(source_agent, domain="math")
    source = bank.best_for("math")
    adapter = TransferAdapter(backend=backend, intensity="medium")
    adapted_agent = adapter.adapt(source, target_task="Explain quantum entanglement")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from cambrian.agent import Agent, Genome

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Intensity = Literal["light", "medium", "heavy"]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ADAPT_SYSTEM = (
    "You are an expert at adapting AI agent prompts from one domain to another. "
    "Preserve broadly useful strategies and reasoning patterns from the source prompt "
    "while specialising for the target task.  Return ONLY the adapted system prompt."
)

_ADAPT_TEMPLATE_LIGHT = """\
INTENSITY: light — preserve most of the source prompt, only tweak domain-specific parts.

SOURCE DOMAIN: {source_domain}
SOURCE SYSTEM PROMPT:
{source_prompt}

TARGET TASK: {target_task}

Return the lightly adapted system prompt:
"""

_ADAPT_TEMPLATE_MEDIUM = """\
INTENSITY: medium — rewrite ~50% of the source prompt for the target task.

SOURCE STRATEGY: {source_strategy}
SOURCE SYSTEM PROMPT SKELETON:
{source_prompt}

TARGET TASK: {target_task}

Return the adapted system prompt:
"""

_ADAPT_TEMPLATE_HEAVY = """\
INTENSITY: heavy — use the source as inspiration only; write a fresh prompt for the target.

SOURCE INSIGHTS (extracted from source agent):
Strategy: {source_strategy}
Key phrases: {key_phrases}

TARGET TASK: {target_task}

Write a completely new system prompt inspired by the source's best practices:
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TransferRecord:
    """Record of a transfer adaptation event."""

    source_agent_id: str
    target_task_preview: str
    intensity: Intensity
    source_domain: str


# ---------------------------------------------------------------------------
# TransferAdapter
# ---------------------------------------------------------------------------


class TransferAdapter:
    """Adapt a source agent's genome to a new target task.

    Parameters
    ----------
    backend:
        LLM backend used for adaptation.
    intensity:
        How much to modify the source prompt: ``"light"`` | ``"medium"``
        | ``"heavy"`` (default ``"medium"``).
    temperature:
        Sampling temperature for the adaptation call (default 0.6).
    """

    def __init__(
        self,
        backend: "LLMBackend",
        intensity: Intensity = "medium",
        temperature: float = 0.6,
    ) -> None:
        self._backend = backend
        self._intensity = intensity
        self._temperature = temperature
        self._records: list[TransferRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[TransferRecord]:
        """Return a copy of all adaptation records."""
        return list(self._records)

    def adapt(
        self,
        source: Agent,
        target_task: str,
        source_domain: str = "general",
        temperature: Optional[float] = None,
    ) -> Agent:
        """Adapt *source* genome to *target_task*, returning a new Agent."""
        temp = temperature if temperature is not None else self._temperature
        adapted_prompt = self._llm_adapt(source, target_task, source_domain, temp)

        new_genome = Genome(
            system_prompt=adapted_prompt,
            temperature=source.genome.temperature,
            strategy=f"transfer({self._intensity},{source_domain}→{target_task[:20]})",
        )
        new_agent = Agent(genome=new_genome)

        self._records.append(
            TransferRecord(
                source_agent_id=source.agent_id,
                target_task_preview=target_task[:80],
                intensity=self._intensity,
                source_domain=source_domain,
            )
        )
        return new_agent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_adapt(
        self,
        source: Agent,
        target_task: str,
        source_domain: str,
        temperature: float,
    ) -> str:
        """Build the appropriate template and call the LLM."""
        if self._intensity == "light":
            user_msg = _ADAPT_TEMPLATE_LIGHT.format(
                source_domain=source_domain,
                source_prompt=source.genome.system_prompt,
                target_task=target_task,
            )
        elif self._intensity == "medium":
            user_msg = _ADAPT_TEMPLATE_MEDIUM.format(
                source_strategy=source.genome.strategy,
                source_prompt=source.genome.system_prompt,
                target_task=target_task,
            )
        else:  # heavy
            # Extract "key phrases": first 5 unique words with >4 chars
            words = source.genome.system_prompt.split()
            seen: set[str] = set()
            key: list[str] = []
            for w in words:
                clean = w.strip(".,;:").lower()
                if len(clean) > 4 and clean not in seen:
                    seen.add(clean)
                    key.append(clean)
                    if len(key) >= 5:
                        break
            user_msg = _ADAPT_TEMPLATE_HEAVY.format(
                source_strategy=source.genome.strategy,
                key_phrases=", ".join(key) if key else "n/a",
                target_task=target_task,
            )

        try:
            return str(
                self._backend.generate(
                    f"{_ADAPT_SYSTEM}\n\n{user_msg}",
                    temperature=temperature,
                )
            )
        except Exception:  # noqa: BLE001
            # Fallback: prepend target hint to source prompt
            return f"[TARGET: {target_task[:60]}]\n\n{source.genome.system_prompt}"


# ---------------------------------------------------------------------------
# TransferBank
# ---------------------------------------------------------------------------


class TransferBank:
    """Repository of pre-trained source agents indexed by domain.

    Parameters
    ----------
    max_per_domain:
        Maximum agents stored per domain tag (oldest are evicted, default 5).
    """

    def __init__(self, max_per_domain: int = 5) -> None:
        self._max = max_per_domain
        self._store: dict[str, list[Agent]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, agent: Agent, domain: str = "general") -> None:
        """Add *agent* to the bank under *domain*."""
        pool = self._store.setdefault(domain, [])
        pool.append(agent)
        # Evict oldest if over limit
        if len(pool) > self._max:
            self._store[domain] = pool[-self._max :]

    def best_for(self, domain: str) -> Optional[Agent]:
        """Return the highest-fitness agent for *domain*, or None."""
        pool = self._store.get(domain, [])
        if not pool:
            return None
        return max(pool, key=lambda a: a.fitness or 0.0)

    def all_domains(self) -> list[str]:
        """Return list of registered domain tags."""
        return list(self._store.keys())

    def count(self, domain: str) -> int:
        """Return number of agents stored for *domain*."""
        return len(self._store.get(domain, []))
