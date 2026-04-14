"""SemanticCache — prompt-level response cache.

Caches LLM responses keyed by a hash of the prompt string so identical
prompts (during evaluation re-runs or across generations) don't incur
redundant API calls.

For a semantically-aware cache (embedding similarity), swap the hash-based
key for a vector lookup — the interface is identical.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any


class SemanticCache:
    """LRU cache for LLM prompt→response pairs.

    Keys are SHA-256 hashes of the (prompt, model, temperature) tuple so
    different hyperparameters produce different cache entries.

    Args:
        max_size: Maximum number of entries before LRU eviction. Default 1024.
        ttl_seconds: Time-to-live per entry in seconds.  ``None`` (default)
            means entries never expire.
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: float | None = None,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        # OrderedDict used as an LRU store: most-recently-used → end
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, prompt: str, model: str = "", temperature: float = 0.0) -> str | None:
        """Return cached response or ``None`` on cache miss.

        Args:
            prompt: The exact prompt string.
            model: Model identifier (part of the cache key).
            temperature: Sampling temperature (part of the cache key).
        """
        key = self._key(prompt, model, temperature)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        response, inserted_at = entry
        if self._ttl is not None and (time.monotonic() - inserted_at) > self._ttl:
            del self._store[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return response

    def set(
        self,
        prompt: str,
        response: str,
        model: str = "",
        temperature: float = 0.0,
    ) -> None:
        """Store *response* for *prompt*.

        Evicts the least-recently-used entry when the cache is full.
        """
        key = self._key(prompt, model, temperature)
        self._store[key] = (response, time.monotonic())
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict LRU

    def hit_rate(self) -> float:
        """Return the cache hit rate as a float in ``[0.0, 1.0]``.

        Returns ``0.0`` if no lookups have been performed yet.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._store)

    def clear(self) -> None:
        """Evict all entries and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(prompt: str, model: str, temperature: float) -> str:
        raw = f"{prompt}||{model}||{temperature:.4f}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"SemanticCache(size={self.size}/{self._max_size}, "
            f"hit_rate={self.hit_rate():.1%})"
        )
