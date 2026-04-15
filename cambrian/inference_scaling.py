"""Inference-Time Scaling — Technique 57.

Generate N candidate responses at inference time and select the best via
a scoring function.  This trades compute for quality without retraining —
directly inspired by "test-time compute scaling" work (DeepSeek-R1, o1).

Components
----------
BestOfN
    Generates ``n`` responses in sequence and returns the one with the
    highest score.  Falls back to the first response on all-empty outputs.

BeamSearch
    Maintains a ``beam_width`` beam of partial responses, expanding each
    step by sampling ``branching_factor`` continuations and keeping the
    top-scoring ones.  Terminates after ``n_steps`` steps.

ScoringFunction
    Protocol and built-in implementations:
    * ``LengthScorer`` — prefers longer, more detailed responses.
    * ``KeywordScorer`` — scores by keyword coverage.
    * ``SelfConsistencyScorer`` — majority-vote on short answers
      extracted from responses.

Usage::

    from cambrian.inference_scaling import BestOfN, KeywordScorer

    scorer = KeywordScorer(keywords=["step", "therefore", "because"])
    best_of_n = BestOfN(backend=backend, n=5, scorer=scorer)
    answer, score = best_of_n.run(system_prompt, task)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from cambrian.backends.base import LLMBackend


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

ScoringFn = Callable[[str], float]


def length_scorer(response: str) -> float:
    """Score proportional to response length (normalised to [0,1])."""
    n = len(response.strip())
    return min(1.0, n / 500)


class KeywordScorer:
    """Score by fraction of target keywords present in the response."""

    def __init__(self, keywords: list[str]) -> None:
        self._kw = [k.lower() for k in keywords]

    def __call__(self, response: str) -> float:
        if not self._kw:
            return 0.0
        lower = response.lower()
        hits = sum(1 for k in self._kw if k in lower)
        return hits / len(self._kw)


class SelfConsistencyScorer:
    """Score by agreement with the majority answer across candidates.

    Extracts a short "answer token" from each response (last number or
    last capitalised word) and assigns 1.0 to responses matching the
    majority, 0.0 to others.
    """

    _NUM = re.compile(r"[-\d.]+")
    _WORD = re.compile(r"[A-Z][a-z]+")

    def __init__(self, candidates: list[str]) -> None:
        tokens = [self._extract(r) for r in candidates]
        if not tokens:
            self._majority = ""
        else:
            count = Counter(t for t in tokens if t)
            self._majority = count.most_common(1)[0][0] if count else ""

    def __call__(self, response: str) -> float:
        if not self._majority:
            return 0.0
        return 1.0 if self._extract(response) == self._majority else 0.0

    def _extract(self, text: str) -> str:
        nums: list[str] = self._NUM.findall(text)
        if nums:
            return str(nums[-1])
        words: list[str] = self._WORD.findall(text)
        return str(words[-1]) if words else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScalingResult:
    """Result from a BestOfN or BeamSearch run."""

    best_response: str
    best_score: float
    all_responses: list[str]
    all_scores: list[float]
    n_generated: int


# ---------------------------------------------------------------------------
# BestOfN
# ---------------------------------------------------------------------------


class BestOfN:
    """Generate N responses and return the highest-scoring one.

    Parameters
    ----------
    backend:
        LLM backend to call.
    n:
        Number of candidate responses to generate (default 5).
    scorer:
        Scoring function ``str → float``.  Defaults to ``length_scorer``.
    temperature:
        Sampling temperature (default 0.9 for diversity).
    """

    def __init__(
        self,
        backend: "LLMBackend",
        n: int = 5,
        scorer: ScoringFn = length_scorer,
        temperature: float = 0.9,
    ) -> None:
        if n < 1:
            raise ValueError("n must be ≥ 1")
        self._backend = backend
        self._n = n
        self._scorer = scorer
        self._temperature = temperature
        self._results: list[ScalingResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def results(self) -> list[ScalingResult]:
        """Return a copy of all run results."""
        return list(self._results)

    def run(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> tuple[str, float]:
        """Generate N responses, return ``(best_response, best_score)``."""
        temp = temperature if temperature is not None else self._temperature
        responses: list[str] = []
        for _ in range(self._n):
            try:
                r = str(self._backend.generate(f"{system}\n\n{user}", temperature=temp))
            except Exception:  # noqa: BLE001
                r = ""
            responses.append(r)

        scores = [self._scorer(r) for r in responses]

        # Use self-consistency if all scores are equal (tie-breaking)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = responses[best_idx]
        best_score = scores[best_idx]

        result = ScalingResult(
            best_response=best,
            best_score=best_score,
            all_responses=list(responses),
            all_scores=list(scores),
            n_generated=self._n,
        )
        self._results.append(result)
        return best, best_score


# ---------------------------------------------------------------------------
# BeamSearch (step-wise)
# ---------------------------------------------------------------------------


class BeamSearch:
    """Step-wise beam search over LLM continuations.

    At each step, every beam candidate is expanded by sampling
    ``branching_factor`` continuations.  The top ``beam_width`` candidates
    (by cumulative score) are retained.

    Parameters
    ----------
    backend:
        LLM backend.
    beam_width:
        Number of beams to maintain (default 3).
    branching_factor:
        Continuations sampled per beam per step (default 3).
    n_steps:
        Number of expansion steps (default 2).
    scorer:
        Scoring function for each beam candidate (default length_scorer).
    temperature:
        Sampling temperature (default 0.8).
    """

    def __init__(
        self,
        backend: "LLMBackend",
        beam_width: int = 3,
        branching_factor: int = 3,
        n_steps: int = 2,
        scorer: ScoringFn = length_scorer,
        temperature: float = 0.8,
    ) -> None:
        self._backend = backend
        self._bw = beam_width
        self._bf = branching_factor
        self._steps = n_steps
        self._scorer = scorer
        self._temperature = temperature

    def run(self, system: str, user: str) -> tuple[str, float]:
        """Run beam search; return ``(best_response, best_score)``."""
        # Initialise beam with branching_factor seed responses
        beams: list[str] = []
        for _ in range(self._bf):
            try:
                r = str(
                    self._backend.generate(
                        f"{system}\n\n{user}", temperature=self._temperature
                    )
                )
            except Exception:  # noqa: BLE001
                r = ""
            beams.append(r)

        # Keep top beam_width
        beams = self._top_k(beams, self._bw)

        # Expand for n_steps - 1 additional steps
        for _ in range(self._steps - 1):
            candidates: list[str] = []
            for beam in beams:
                for _ in range(self._bf):
                    prompt = f"{system}\n\n{user}\n\nContinuation of:\n{beam}\n\nContinue:"
                    try:
                        cont = str(
                            self._backend.generate(prompt, temperature=self._temperature)
                        )
                    except Exception:  # noqa: BLE001
                        cont = beam
                    candidates.append(beam + " " + cont)
            beams = self._top_k(candidates, self._bw)

        best = max(beams, key=self._scorer)
        return best, self._scorer(best)

    def _top_k(self, candidates: list[str], k: int) -> list[str]:
        """Return top-k candidates by scorer."""
        return sorted(candidates, key=self._scorer, reverse=True)[:k]
