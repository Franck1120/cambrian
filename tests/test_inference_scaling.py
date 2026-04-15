"""Tests for cambrian.inference_scaling — BestOfN, BeamSearch, scorers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.inference_scaling import (
    BeamSearch,
    BestOfN,
    KeywordScorer,
    ScalingResult,
    SelfConsistencyScorer,
    length_scorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(*responses: str) -> MagicMock:
    b = MagicMock()
    b.generate.side_effect = list(responses)
    return b


def _cycling_backend(*responses: str) -> MagicMock:
    """Backend that cycles through responses indefinitely."""
    b = MagicMock()
    cycle = list(responses)
    calls = [0]

    def gen(prompt: str, **kw: object) -> str:
        r = cycle[calls[0] % len(cycle)]
        calls[0] += 1
        return r

    b.generate.side_effect = gen
    return b


# ---------------------------------------------------------------------------
# length_scorer
# ---------------------------------------------------------------------------


class TestLengthScorer:
    def test_empty_is_zero(self) -> None:
        assert length_scorer("") == 0.0

    def test_long_response_is_one(self) -> None:
        assert length_scorer("x" * 600) == 1.0

    def test_partial_length(self) -> None:
        score = length_scorer("x" * 250)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# KeywordScorer
# ---------------------------------------------------------------------------


class TestKeywordScorer:
    def test_all_keywords_present(self) -> None:
        scorer = KeywordScorer(["step", "because", "therefore"])
        assert scorer("step by step because therefore") == pytest.approx(1.0)

    def test_no_keywords(self) -> None:
        scorer = KeywordScorer(["step"])
        assert scorer("nothing relevant") == 0.0

    def test_partial_keywords(self) -> None:
        scorer = KeywordScorer(["a", "b", "c"])
        assert scorer("a b") == pytest.approx(2 / 3, abs=1e-6)

    def test_empty_keywords_list(self) -> None:
        scorer = KeywordScorer([])
        assert scorer("anything") == 0.0

    def test_case_insensitive(self) -> None:
        scorer = KeywordScorer(["STEP"])
        assert scorer("step by step") == 1.0


# ---------------------------------------------------------------------------
# SelfConsistencyScorer
# ---------------------------------------------------------------------------


class TestSelfConsistencyScorer:
    def test_majority_number_wins(self) -> None:
        candidates = ["answer is 42", "result: 42", "I think 99"]
        scorer = SelfConsistencyScorer(candidates)
        assert scorer("the answer is 42") == 1.0
        assert scorer("clearly 99 is correct") == 0.0

    def test_empty_candidates(self) -> None:
        scorer = SelfConsistencyScorer([])
        assert scorer("anything") == 0.0


# ---------------------------------------------------------------------------
# BestOfN — init
# ---------------------------------------------------------------------------


class TestBestOfNInit:
    def test_n_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            BestOfN(backend=MagicMock(), n=0)

    def test_n_one_ok(self) -> None:
        b = BestOfN(backend=MagicMock(), n=1)
        assert b._n == 1

    def test_results_starts_empty(self) -> None:
        b = BestOfN(backend=MagicMock())
        assert b.results == []


# ---------------------------------------------------------------------------
# BestOfN — run
# ---------------------------------------------------------------------------


class TestBestOfNRun:
    def test_returns_best_response(self) -> None:
        # Scorer prefers longest — "long answer" is longest
        b = BestOfN(
            backend=_cycling_backend("short", "long answer here much longer"),
            n=2,
            scorer=length_scorer,
        )
        best, score = b.run("sys", "user")
        assert best == "long answer here much longer"

    def test_n_calls_to_backend(self) -> None:
        backend = _cycling_backend("x")
        b = BestOfN(backend=backend, n=4, scorer=length_scorer)
        b.run("sys", "user")
        assert backend.generate.call_count == 4

    def test_result_stored(self) -> None:
        b = BestOfN(backend=_cycling_backend("answer"), n=2)
        b.run("sys", "user")
        assert len(b.results) == 1
        r = b.results[0]
        assert isinstance(r, ScalingResult)
        assert r.n_generated == 2

    def test_results_returns_copy(self) -> None:
        b = BestOfN(backend=_cycling_backend("x"), n=1)
        b.run("s", "u")
        r1 = b.results
        r1.clear()
        assert len(b.results) == 1

    def test_backend_error_returns_empty(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("API error")
        b = BestOfN(backend=backend, n=2)
        best, score = b.run("sys", "user")
        assert best == ""

    def test_temperature_passed_to_backend(self) -> None:
        backend = _cycling_backend("ok")
        b = BestOfN(backend=backend, n=1)
        b.run("sys", "user", temperature=0.1)
        _, kwargs = backend.generate.call_args
        assert kwargs.get("temperature") == 0.1

    def test_keyword_scorer_selects_best(self) -> None:
        scorer = KeywordScorer(["step", "because"])
        b = BestOfN(
            backend=_cycling_backend(
                "no relevant content",
                "step by step because of the reason",
            ),
            n=2,
            scorer=scorer,
        )
        best, score = b.run("sys", "user")
        assert "step" in best
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BeamSearch — run
# ---------------------------------------------------------------------------


class TestBeamSearch:
    def test_returns_string_and_score(self) -> None:
        b = BeamSearch(
            backend=_cycling_backend("short", "longer answer here"),
            beam_width=2,
            branching_factor=2,
            n_steps=1,
            scorer=length_scorer,
        )
        best, score = b.run("sys", "user")
        assert isinstance(best, str)
        assert 0.0 <= score <= 1.0

    def test_best_beam_selected(self) -> None:
        # Always produce two responses — "a" (short) and "b" * 200 (long)
        b = BeamSearch(
            backend=_cycling_backend("a", "b" * 200),
            beam_width=2,
            branching_factor=2,
            n_steps=1,
            scorer=length_scorer,
        )
        best, score = b.run("sys", "user")
        assert score == pytest.approx(1.0) or "b" in best

    def test_multiple_steps_run(self) -> None:
        backend = _cycling_backend("response")
        b = BeamSearch(
            backend=backend,
            beam_width=2,
            branching_factor=2,
            n_steps=2,
            scorer=length_scorer,
        )
        b.run("sys", "user")
        # Step 0: 2 seeds; step 1: 2 beams × 2 branches = 4; total ≥ 6
        assert backend.generate.call_count >= 6

    def test_beam_error_falls_back(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("err")
        b = BeamSearch(backend=backend, beam_width=1, branching_factor=1, n_steps=1)
        best, score = b.run("sys", "user")
        assert isinstance(best, str)
