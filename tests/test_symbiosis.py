"""Tests for cambrian.symbiosis — SymbioticFuser."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.symbiosis import SymbioticFuser, SymbioticPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(prompt: str, fitness: float, strategy: str = "default") -> Agent:
    g = Genome(system_prompt=prompt, strategy=strategy)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _backend(response: str = "merged prompt") -> MagicMock:
    b = MagicMock()
    b.generate.return_value = response
    return b


# ---------------------------------------------------------------------------
# SymbioticFuser — initialisation
# ---------------------------------------------------------------------------


class TestSymbioticFuserInit:
    def test_defaults(self) -> None:
        fuser = SymbioticFuser(backend=_backend())
        assert fuser._fitness_threshold == 0.4
        assert fuser._min_distance == 0.15
        assert fuser._temperature == 0.7

    def test_custom_params(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend(),
            fitness_threshold=0.6,
            min_distance=0.3,
            temperature=0.5,
        )
        assert fuser._fitness_threshold == 0.6
        assert fuser._min_distance == 0.3
        assert fuser._temperature == 0.5

    def test_history_starts_empty(self) -> None:
        fuser = SymbioticFuser(backend=_backend())
        assert fuser.history == []


# ---------------------------------------------------------------------------
# _prompt_distance
# ---------------------------------------------------------------------------


class TestPromptDistance:
    def test_identical_prompts_distance_zero(self) -> None:
        d = SymbioticFuser._prompt_distance("hello world", "hello world")
        assert d == 0.0

    def test_completely_different(self) -> None:
        d = SymbioticFuser._prompt_distance("alpha beta", "gamma delta")
        assert d == 1.0

    def test_partial_overlap(self) -> None:
        d = SymbioticFuser._prompt_distance("a b c", "b c d")
        # intersection={b,c}=2, union={a,b,c,d}=4 → jaccard=0.5 → dist=0.5
        assert pytest.approx(d, abs=1e-9) == 0.5

    def test_empty_strings(self) -> None:
        d = SymbioticFuser._prompt_distance("", "")
        assert d == 0.0

    def test_case_insensitive(self) -> None:
        d = SymbioticFuser._prompt_distance("Hello World", "hello world")
        assert d == 0.0


# ---------------------------------------------------------------------------
# _compatible
# ---------------------------------------------------------------------------


class TestCompatible:
    def test_both_above_threshold_and_distant(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5, min_distance=0.4)
        host = _make_agent("python expert math solver", 0.8)
        donor = _make_agent("creative writing novelist", 0.7)
        assert fuser._compatible(host, donor) is True

    def test_host_below_threshold(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5)
        host = _make_agent("x y z", 0.3)
        donor = _make_agent("a b c", 0.8)
        assert fuser._compatible(host, donor) is False

    def test_donor_below_threshold(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5)
        host = _make_agent("x y z", 0.8)
        donor = _make_agent("a b c", 0.2)
        assert fuser._compatible(host, donor) is False

    def test_prompts_too_similar(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5, min_distance=0.8)
        host = _make_agent("python expert", 0.9)
        donor = _make_agent("python expert", 0.9)
        assert fuser._compatible(host, donor) is False


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


class TestFuse:
    def test_returns_none_on_incompatible(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5)
        host = _make_agent("a b c", 0.2)
        donor = _make_agent("d e f", 0.9)
        result = fuser.fuse(host, donor, "task")
        assert result is None

    def test_returns_agent_on_compatible(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend("merged system prompt"),
            fitness_threshold=0.5,
            min_distance=0.3,
        )
        host = _make_agent("python math solver expert", 0.9, "math")
        donor = _make_agent("creative writing novelist poet", 0.8, "writing")
        result = fuser.fuse(host, donor, "write a poem about maths")
        assert result is not None
        assert result.genome.system_prompt == "merged system prompt"

    def test_temperature_averaged(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend("merged"),
            fitness_threshold=0.0,
            min_distance=0.0,
        )
        host = _make_agent("aaa bbb ccc", 0.9)
        host.genome.temperature = 0.4
        donor = _make_agent("xxx yyy zzz", 0.8)
        donor.genome.temperature = 0.8
        result = fuser.fuse(host, donor, "task")
        assert result is not None
        assert pytest.approx(result.genome.temperature, abs=1e-9) == 0.6

    def test_history_updated(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend("merged"),
            fitness_threshold=0.0,
            min_distance=0.0,
        )
        host = _make_agent("aaa bbb ccc", 0.9)
        donor = _make_agent("xxx yyy zzz", 0.8)
        fuser.fuse(host, donor, "task")
        assert len(fuser.history) == 1
        pair = fuser.history[0]
        assert isinstance(pair, SymbioticPair)
        assert pair.host_fitness == 0.9
        assert pair.donor_fitness == 0.8

    def test_history_returns_copy(self) -> None:
        fuser = SymbioticFuser(backend=_backend("m"), fitness_threshold=0.0, min_distance=0.0)
        host = _make_agent("aaa bbb", 0.9)
        donor = _make_agent("xxx yyy", 0.8)
        fuser.fuse(host, donor, "t")
        h1 = fuser.history
        h1.clear()
        assert len(fuser.history) == 1  # original unaffected

    def test_llm_failure_falls_back_to_concatenation(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("API error")
        fuser = SymbioticFuser(backend=b, fitness_threshold=0.0, min_distance=0.0)
        host = _make_agent("alpha beta gamma", 0.9)
        donor = _make_agent("delta epsilon zeta", 0.8)
        result = fuser.fuse(host, donor, "task")
        assert result is not None
        assert "alpha beta gamma" in result.genome.system_prompt
        assert "delta epsilon zeta" in result.genome.system_prompt

    def test_strategy_reflects_both(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend("merged"), fitness_threshold=0.0, min_distance=0.0
        )
        host = _make_agent("aaa bbb ccc", 0.9, strategy="cot")
        donor = _make_agent("xxx yyy zzz", 0.8, strategy="few-shot")
        result = fuser.fuse(host, donor, "t")
        assert result is not None
        assert "cot" in result.genome.strategy
        assert "few-shot" in result.genome.strategy


# ---------------------------------------------------------------------------
# fuse_best_pair
# ---------------------------------------------------------------------------


class TestFuseBestPair:
    def test_returns_none_when_not_enough_candidates(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.5)
        pop = [_make_agent("a b c", 0.3)]
        assert fuser.fuse_best_pair(pop, "task") is None

    def test_returns_none_when_all_below_threshold(self) -> None:
        fuser = SymbioticFuser(backend=_backend(), fitness_threshold=0.9)
        pop = [_make_agent("a b c", 0.4), _make_agent("x y z", 0.5)]
        assert fuser.fuse_best_pair(pop, "task") is None

    def test_fuses_best_pair(self) -> None:
        fuser = SymbioticFuser(
            backend=_backend("merged"),
            fitness_threshold=0.5,
            min_distance=0.4,
        )
        pop = [
            _make_agent("python math solver expert", 0.9),
            _make_agent("creative writing novelist poet", 0.8),
            _make_agent("data science statistics", 0.7),
        ]
        result = fuser.fuse_best_pair(pop, "task")
        assert result is not None

    def test_returns_none_if_no_distant_donor(self) -> None:
        # All agents have identical prompts → distance = 0 < min_distance
        fuser = SymbioticFuser(
            backend=_backend("merged"),
            fitness_threshold=0.5,
            min_distance=0.5,
        )
        pop = [
            _make_agent("python expert", 0.9),
            _make_agent("python expert", 0.8),
        ]
        assert fuser.fuse_best_pair(pop, "task") is None
