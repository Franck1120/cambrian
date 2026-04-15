"""Tests for cambrian.llm_cascade — LLMCascade."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.llm_cascade import (
    CascadeLevel,
    CascadeResult,
    LLMCascade,
    hedging_confidence,
    length_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(response: str = "answer") -> MagicMock:
    b = MagicMock()
    b.generate.return_value = response
    return b


def _level(
    response: str = "answer",
    threshold: float = 0.7,
    extractor=None,
) -> CascadeLevel:
    if extractor is None:
        extractor = lambda r: 1.0  # noqa: E731 — always confident
    return CascadeLevel(
        backend=_backend(response),
        confidence_threshold=threshold,
        confidence_extractor=extractor,
    )


# ---------------------------------------------------------------------------
# Confidence extractors
# ---------------------------------------------------------------------------


class TestHedgingConfidence:
    def test_no_hedging(self) -> None:
        assert hedging_confidence("The answer is 42.") == 1.0

    def test_one_hedge(self) -> None:
        result = hedging_confidence("It might be 42.")
        assert pytest.approx(result, abs=1e-9) == 0.75

    def test_multiple_hedges(self) -> None:
        result = hedging_confidence("I'm not sure, it may perhaps be 42.")
        assert result <= 0.25  # 3 hedges → max(0, 1 - 0.75) = 0.25

    def test_clamped_at_zero(self) -> None:
        text = "I'm not sure it may might possibly perhaps be correct."
        assert hedging_confidence(text) == 0.0

    def test_case_insensitive(self) -> None:
        assert hedging_confidence("MIGHT be correct") < 1.0


class TestLengthConfidence:
    def test_empty_is_zero(self) -> None:
        assert length_confidence("") == 0.0

    def test_short_is_low(self) -> None:
        assert length_confidence("hi", min_len=50) < 1.0

    def test_long_is_one(self) -> None:
        assert length_confidence("x" * 100, min_len=50) == 1.0

    def test_exactly_min_len_is_one(self) -> None:
        assert length_confidence("x" * 50, min_len=50) == 1.0


# ---------------------------------------------------------------------------
# LLMCascade init
# ---------------------------------------------------------------------------


class TestLLMCascadeInit:
    def test_empty_levels_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMCascade(levels=[])

    def test_single_level_ok(self) -> None:
        cascade = LLMCascade(levels=[_level()])
        assert len(cascade._levels) == 1

    def test_stats_starts_empty(self) -> None:
        cascade = LLMCascade(levels=[_level()])
        assert cascade.stats == []


# ---------------------------------------------------------------------------
# query — routing logic
# ---------------------------------------------------------------------------


class TestQuery:
    def test_level_0_answers_when_confident(self) -> None:
        # extractor always returns 1.0 → level 0 answers
        cascade = LLMCascade(
            levels=[
                _level("fast answer", threshold=0.7),
                _level("slow answer", threshold=0.0),
            ]
        )
        response, idx = cascade.query("sys", "user")
        assert response == "fast answer"
        assert idx == 0

    def test_escalates_to_level_1_on_low_confidence(self) -> None:
        # level 0 extractor always returns 0.0
        low_conf = lambda r: 0.0  # noqa: E731
        high_conf = lambda r: 1.0  # noqa: E731
        l0 = CascadeLevel(
            backend=_backend("weak answer"),
            confidence_threshold=0.7,
            confidence_extractor=low_conf,
        )
        l1 = CascadeLevel(
            backend=_backend("strong answer"),
            confidence_threshold=0.7,
            confidence_extractor=high_conf,
        )
        cascade = LLMCascade(levels=[l0, l1])
        response, idx = cascade.query("sys", "user")
        assert response == "strong answer"
        assert idx == 1

    def test_last_level_always_returns(self) -> None:
        # All levels have low confidence but last one still answers
        low_conf = lambda r: 0.0  # noqa: E731
        levels = [
            CascadeLevel(
                backend=_backend(f"level{i}"),
                confidence_threshold=0.9,
                confidence_extractor=low_conf,
            )
            for i in range(3)
        ]
        cascade = LLMCascade(levels=levels)
        response, idx = cascade.query("sys", "user")
        assert idx == 2
        assert response == "level2"

    def test_result_recorded_in_stats(self) -> None:
        cascade = LLMCascade(levels=[_level("answer")])
        cascade.query("sys", "user")
        assert len(cascade.stats) == 1
        r = cascade.stats[0]
        assert isinstance(r, CascadeResult)
        assert r.response == "answer"
        assert r.level_index == 0

    def test_stats_returns_copy(self) -> None:
        cascade = LLMCascade(levels=[_level("x")])
        cascade.query("sys", "user")
        s1 = cascade.stats
        s1.clear()
        assert len(cascade.stats) == 1

    def test_temperature_override(self) -> None:
        b = MagicMock()
        b.generate.return_value = "ok"
        level = CascadeLevel(backend=b, confidence_threshold=0.0, temperature=0.5)
        cascade = LLMCascade(levels=[level])
        cascade.query("sys", "user", temperature=0.9)
        _, kwargs = b.generate.call_args
        assert kwargs.get("temperature") == 0.9

    def test_backend_error_uses_empty_response(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("API down")
        level = CascadeLevel(backend=b, confidence_threshold=0.0)
        cascade = LLMCascade(levels=[level])
        response, idx = cascade.query("sys", "user")
        assert response == ""
        assert idx == 0


# ---------------------------------------------------------------------------
# level_usage_counts
# ---------------------------------------------------------------------------


class TestLevelUsageCounts:
    def test_counts_levels(self) -> None:
        low_conf = lambda r: 0.0  # noqa: E731
        high_conf = lambda r: 1.0  # noqa: E731
        l0 = CascadeLevel(
            backend=_backend("fast"),
            confidence_threshold=0.7,
            confidence_extractor=high_conf,
        )
        l1 = CascadeLevel(
            backend=_backend("slow"),
            confidence_threshold=0.7,
            confidence_extractor=low_conf,
        )
        cascade = LLMCascade(levels=[l0, l1])
        # 2 queries handled by l0, 1 query escalated to l1
        cascade.query("s", "u1")
        cascade.query("s", "u2")
        # Force escalation for third query
        l0.confidence_extractor = low_conf
        cascade.query("s", "u3")
        counts = cascade.level_usage_counts()
        assert counts.get(0, 0) == 2
        assert counts.get(1, 0) == 1
