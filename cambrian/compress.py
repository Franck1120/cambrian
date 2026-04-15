# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Prompt compression utilities.

Reducing prompt length cuts API costs and improves cache hit rates. Two
strategies are provided:

1. ``caveman_compress`` — aggressive stopword removal for maximum compression.
2. ``procut_prune`` — trims a genome's system prompt to fit within a token budget
   while preserving meaning.
"""

from __future__ import annotations

import re

from cambrian.agent import Genome

# English stopwords and filler phrases to strip
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "or", "but", "nor", "so", "for", "yet",
    "in", "on", "at", "to", "of", "with", "by", "from", "up", "down",
    "out", "into", "through", "during", "before", "after", "above",
    "below", "between", "this", "that", "these", "those", "it", "its",
    "which", "who", "whom", "what", "how", "when", "where", "why",
    "very", "quite", "rather", "really", "just", "also", "well",
    "please", "kindly", "hereby",
})

_FILLER_PHRASES: list[str] = [
    r"\bAs an AI assistant\b,?\s*",
    r"\bI will\b\s+",
    r"\bPlease note that\b\s*",
    r"\bIt is important to\b\s*",
    r"\bMake sure to\b\s*",
    r"\bIn order to\b\s*",
    r"\bAs a result\b,?\s*",
    r"\bDue to the fact that\b\s*",
    r"\bAt this point in time\b\s*",
]

_FILLER_RE = re.compile("|".join(_FILLER_PHRASES), re.IGNORECASE)


def caveman_compress(prompt: str) -> str:
    """Aggressively compress *prompt* by removing stopwords and filler phrases.

    Designed for short, dense prompts where token count matters more than
    natural readability. Not suitable for end-user-facing text.

    Args:
        prompt: The text to compress.

    Returns:
        Compressed string with stopwords and filler phrases removed.

    Example::

        >>> caveman_compress("Please make sure to implement a very simple function")
        'implement simple function'
    """
    # Remove filler phrases first (multi-word)
    text = _FILLER_RE.sub("", prompt)

    # Tokenise preserving punctuation-adjacent words
    tokens = text.split()
    compressed = [t for t in tokens if t.lower().rstrip(".,;:") not in _STOPWORDS]

    # Collapse whitespace
    result = " ".join(compressed)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def procut_prune(genome: Genome, max_tokens: int = 256) -> Genome:
    """Prune *genome*'s system prompt to fit within *max_tokens*.

    Uses a paragraph-dropping strategy: iteratively removes the least
    informative paragraph (shortest) until the estimated token count is
    within budget.

    Args:
        genome: The genome whose ``system_prompt`` should be pruned.
        max_tokens: Target token budget (rough estimate: 4 chars ≈ 1 token).

    Returns:
        A new :class:`~cambrian.agent.Genome` with the pruned prompt.
        The original is not modified.
    """
    from cambrian.agent import Genome as G  # local to avoid circular import

    current = genome.system_prompt
    estimated_tokens = len(current) // 4

    if estimated_tokens <= max_tokens:
        # Already within budget — return a copy
        return G.from_dict({**genome.to_dict()})

    # Split into paragraphs, preserve at least 1
    paragraphs = [p.strip() for p in current.split("\n\n") if p.strip()]

    while len(paragraphs) > 1 and (sum(len(p) for p in paragraphs) // 4) > max_tokens:
        # Remove the shortest paragraph (least content)
        shortest_idx = min(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))
        paragraphs.pop(shortest_idx)

    pruned_prompt = "\n\n".join(paragraphs)

    new_data = genome.to_dict()
    new_data["system_prompt"] = pruned_prompt
    return G.from_dict(new_data)
