# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Horizontal Gene Transfer (HGT) — Technique 63.

In bacteria, HGT allows a cell to acquire genetic material directly from
another organism without parent-offspring inheritance — a lateral transfer
that can rapidly spread adaptive traits.

Applied to agent evolution: a high-fitness "donor" agent can transfer a
contiguous fragment of its system prompt (a "plasmid") to a "recipient"
agent via insertion, replacement, or prefix injection.

Components
----------
HGTPlasmid
    A named fragment extracted from a donor's system prompt.

HGTransfer
    Extracts plasmids from a donor agent and injects them into a recipient
    agent.  Three insertion modes:
    * ``"prefix"``   — prepend the plasmid to the recipient prompt.
    * ``"suffix"``   — append the plasmid.
    * ``"replace"``  — find a semantically similar sentence in the recipient
      and replace it, falling back to suffix.

HGTPool
    Population-level plasmid registry: agents contribute their best
    plasmids; recipients draw from the pool without direct pairwise contact
    (mimics environmental plasmid acquisition).

Usage::

    from cambrian.hgt import HGTransfer, HGTPool

    transfer = HGTransfer(n_sentences=2, mode="prefix")
    offspring = transfer.transfer(donor=best_agent, recipient=target_agent)

    pool = HGTPool(max_plasmids=30)
    pool.contribute(agent, domain="reasoning")
    plasmid = pool.draw(domain="reasoning")
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Literal, Optional

from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

TransferMode = Literal["prefix", "suffix", "replace"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HGTPlasmid:
    """A transferable genetic fragment (system-prompt excerpt)."""

    content: str
    donor_id: str
    donor_fitness: float
    domain: str


@dataclass
class HGTEvent:
    """Record of a horizontal gene transfer event."""

    donor_id: str
    recipient_id: str
    mode: TransferMode
    plasmid_preview: str
    offspring_id: str


# ---------------------------------------------------------------------------
# HGTransfer
# ---------------------------------------------------------------------------


class HGTransfer:
    """Extract a plasmid from a donor and inject it into a recipient.

    Parameters
    ----------
    n_sentences:
        Number of contiguous sentences to extract from the donor (default 2).
    mode:
        Injection mode: ``"prefix"`` | ``"suffix"`` | ``"replace"``
        (default ``"suffix"``).
    fitness_threshold:
        Donor must have fitness ≥ this value (default 0.3).
    """

    def __init__(
        self,
        n_sentences: int = 2,
        mode: TransferMode = "suffix",
        fitness_threshold: float = 0.3,
    ) -> None:
        self._n = n_sentences
        self._mode = mode
        self._threshold = fitness_threshold
        self._events: list[HGTEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[HGTEvent]:
        """Return a copy of all HGT events."""
        return list(self._events)

    def extract_plasmid(
        self,
        donor: Agent,
        domain: str = "general",
        rng: Optional[random.Random] = None,
    ) -> Optional[HGTPlasmid]:
        """Extract a plasmid from *donor*.  Returns None if donor below threshold."""
        if (donor.fitness or 0.0) < self._threshold:
            return None
        rng = rng or random.Random()
        sentences = self._split_sentences(donor.genome.system_prompt)
        if not sentences:
            return None
        # Pick a random contiguous slice of n sentences
        start = rng.randint(0, max(0, len(sentences) - self._n))
        fragment = " ".join(sentences[start : start + self._n])
        return HGTPlasmid(
            content=fragment,
            donor_id=donor.agent_id,
            donor_fitness=donor.fitness or 0.0,
            domain=domain,
        )

    def transfer(
        self,
        donor: Agent,
        recipient: Agent,
        domain: str = "general",
        rng: Optional[random.Random] = None,
    ) -> Optional[Agent]:
        """Inject a donor plasmid into recipient; return new Agent or None."""
        plasmid = self.extract_plasmid(donor, domain, rng)
        if plasmid is None:
            return None
        return self._inject(recipient, plasmid)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject(self, recipient: Agent, plasmid: HGTPlasmid) -> Agent:
        original = recipient.genome.system_prompt

        if self._mode == "prefix":
            new_prompt = f"[HGT]\n{plasmid.content}\n\n{original}"
        elif self._mode == "suffix":
            new_prompt = f"{original}\n\n[HGT]\n{plasmid.content}"
        else:  # replace
            new_prompt = self._replace_mode(original, plasmid.content)

        new_genome = Genome(
            system_prompt=new_prompt,
            temperature=recipient.genome.temperature,
            strategy=f"hgt({self._mode},{plasmid.domain})",
        )
        offspring = Agent(genome=new_genome)

        self._events.append(
            HGTEvent(
                donor_id=plasmid.donor_id,
                recipient_id=recipient.agent_id,
                mode=self._mode,
                plasmid_preview=plasmid.content[:80],
                offspring_id=offspring.agent_id,
            )
        )
        return offspring

    @staticmethod
    def _replace_mode(original: str, plasmid: str) -> str:
        """Find the shortest sentence in original and replace it with the plasmid."""
        sentences = re.split(r"(?<=[.!?])\s+", original.strip())
        if len(sentences) <= 1:
            return f"{original}\n\n[HGT]\n{plasmid}"
        # Replace the shortest sentence (most likely to be replaceable)
        idx = min(range(len(sentences)), key=lambda i: len(sentences[i]))
        sentences[idx] = plasmid
        return " ".join(sentences)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# HGTPool
# ---------------------------------------------------------------------------


class HGTPool:
    """Environmental plasmid registry for population-level HGT.

    Parameters
    ----------
    max_plasmids:
        Maximum plasmids stored (FIFO eviction, default 50).
    """

    def __init__(self, max_plasmids: int = 50) -> None:
        self._max = max_plasmids
        self._pool: list[HGTPlasmid] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of plasmids in the pool."""
        return len(self._pool)

    def contribute(
        self,
        agent: Agent,
        domain: str = "general",
        n_sentences: int = 2,
        rng: Optional[random.Random] = None,
    ) -> Optional[HGTPlasmid]:
        """Extract a plasmid from *agent* and add it to the pool.

        Returns the plasmid on success, None if agent below fitness threshold.
        """
        transfer = HGTransfer(n_sentences=n_sentences, fitness_threshold=0.0)
        plasmid = transfer.extract_plasmid(agent, domain, rng)
        if plasmid is None:
            return None
        self._pool.append(plasmid)
        if len(self._pool) > self._max:
            self._pool.pop(0)
        return plasmid

    def draw(
        self,
        domain: str = "general",
        rng: Optional[random.Random] = None,
    ) -> Optional[HGTPlasmid]:
        """Return a random plasmid from *domain*, or any domain if none available."""
        rng = rng or random.Random()
        domain_pool = [p for p in self._pool if p.domain == domain]
        if domain_pool:
            return rng.choice(domain_pool)
        if self._pool:
            return rng.choice(self._pool)
        return None

    def best_for(self, domain: str = "general") -> Optional[HGTPlasmid]:
        """Return the highest-fitness plasmid for *domain*."""
        domain_pool = [p for p in self._pool if p.domain == domain]
        if not domain_pool:
            return None
        return max(domain_pool, key=lambda p: p.donor_fitness)
