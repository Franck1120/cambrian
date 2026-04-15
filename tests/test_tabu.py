"""Tests for cambrian.tabu — TabuList and TabuMutator."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.tabu import TabuEntry, TabuList, TabuMutator, _fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(prompt: str = "agent prompt") -> Agent:
    g = Genome(system_prompt=prompt)
    return Agent(genome=g)


def _mock_mutator(return_agents: list[Agent]) -> MagicMock:
    """Mutator that returns agents from the list in sequence."""
    m = MagicMock()
    m.mutate.side_effect = list(return_agents)
    return m


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_identical_prompts_same_fp(self) -> None:
        a = _fingerprint("hello world how are you today")
        b = _fingerprint("hello world how are you today")
        assert a == b

    def test_different_prompts_different_fp(self) -> None:
        a = _fingerprint("you are a math expert")
        b = _fingerprint("you are a creative writer")
        assert a != b

    def test_near_identical_prompts_same_fp(self) -> None:
        # Only whitespace/punctuation differs — same bi-gram set
        a = _fingerprint("hello, world!")
        b = _fingerprint("hello world")
        assert a == b

    def test_length_16(self) -> None:
        assert len(_fingerprint("some prompt")) == 16


# ---------------------------------------------------------------------------
# TabuList — init
# ---------------------------------------------------------------------------


class TestTabuListInit:
    def test_defaults(self) -> None:
        tl = TabuList()
        assert tl._max == 20

    def test_starts_empty(self) -> None:
        tl = TabuList()
        assert tl.size == 0


# ---------------------------------------------------------------------------
# TabuList — is_tabu / add
# ---------------------------------------------------------------------------


class TestTabuListOperations:
    def test_new_agent_not_tabu(self) -> None:
        tl = TabuList()
        agent = _make_agent("unique prompt")
        assert tl.is_tabu(agent) is False

    def test_added_agent_is_tabu(self) -> None:
        tl = TabuList()
        agent = _make_agent("some prompt")
        tl.add(agent)
        assert tl.is_tabu(agent) is True

    def test_different_agent_not_tabu(self) -> None:
        tl = TabuList()
        a1 = _make_agent("python math expert step by step")
        a2 = _make_agent("creative writing novelist fiction")
        tl.add(a1)
        assert tl.is_tabu(a2) is False

    def test_fifo_eviction(self) -> None:
        tl = TabuList(max_size=2)
        a1 = _make_agent("prompt alpha beta gamma delta")
        a2 = _make_agent("prompt zeta eta theta iota")
        a3 = _make_agent("prompt kappa lambda mu nu")
        tl.add(a1)
        tl.add(a2)
        tl.add(a3)  # evicts a1
        assert tl.size == 2
        assert tl.is_tabu(a1) is False
        assert tl.is_tabu(a2) is True

    def test_duplicate_add_not_double_counted(self) -> None:
        tl = TabuList(max_size=5)
        agent = _make_agent("same prompt")
        tl.add(agent)
        tl.add(agent)
        assert tl.size == 1

    def test_entries_returns_copy(self) -> None:
        tl = TabuList()
        agent = _make_agent("prompt")
        tl.add(agent)
        e1 = tl.entries
        e1.clear()
        assert tl.size == 1

    def test_clear_empties_list(self) -> None:
        tl = TabuList()
        tl.add(_make_agent("prompt"))
        tl.clear()
        assert tl.size == 0

    def test_advance_generation(self) -> None:
        tl = TabuList()
        assert tl._generation == 0
        tl.advance_generation()
        assert tl._generation == 1


# ---------------------------------------------------------------------------
# TabuMutator
# ---------------------------------------------------------------------------


class TestTabuMutatorInit:
    def test_defaults(self) -> None:
        mutator = TabuMutator(base_mutator=MagicMock(), tabu_list=TabuList())
        assert mutator._max_retries == 3
        assert mutator.tabu_hit_rate == 0.0


class TestTabuMutatorMutate:
    def test_returns_non_tabu_mutation(self) -> None:
        fresh = _make_agent("fresh completely new prompt here today")
        mutator = TabuMutator(
            base_mutator=_mock_mutator([fresh]),
            tabu_list=TabuList(),
        )
        result = mutator.mutate(_make_agent("original"), "task")
        assert result.genome.system_prompt == "fresh completely new prompt here today"

    def test_skips_tabu_mutation_and_retries(self) -> None:
        tabu_list = TabuList()
        tabu_agent = _make_agent("already tabu system prompt here alpha beta")
        fresh_agent = _make_agent("brand new completely different prompt gamma")
        tabu_list.add(tabu_agent)

        mutator = TabuMutator(
            base_mutator=_mock_mutator([tabu_agent, fresh_agent]),
            tabu_list=tabu_list,
            max_retries=3,
        )
        result = mutator.mutate(_make_agent(), "task")
        assert result.genome.system_prompt == "brand new completely different prompt gamma"

    def test_returns_first_candidate_after_exhausting_retries(self) -> None:
        tabu_list = TabuList()
        always_tabu = _make_agent("always tabu system prompt here repeated")
        tabu_list.add(always_tabu)

        mutator = TabuMutator(
            base_mutator=_mock_mutator([always_tabu] * 5),
            tabu_list=tabu_list,
            max_retries=2,
        )
        result = mutator.mutate(_make_agent(), "task")
        # Returns first even if tabu
        assert result.genome.system_prompt == "always tabu system prompt here repeated"

    def test_accepted_mutation_added_to_tabu(self) -> None:
        tabu_list = TabuList()
        fresh = _make_agent("brand new fresh prompt completely unique")
        mutator = TabuMutator(
            base_mutator=_mock_mutator([fresh]),
            tabu_list=tabu_list,
        )
        mutator.mutate(_make_agent(), "task")
        assert tabu_list.is_tabu(fresh) is True

    def test_tabu_hit_rate(self) -> None:
        tabu_list = TabuList()
        tabu_agent = _make_agent("tabu system prompt here alpha beta repeated")
        tabu_list.add(tabu_agent)
        fresh = _make_agent("completely different prompt here fresh novel text")

        mutator = TabuMutator(
            base_mutator=_mock_mutator([tabu_agent, fresh]),
            tabu_list=tabu_list,
            max_retries=3,
        )
        mutator.mutate(_make_agent(), "task")
        # 1 hit (tabu_agent blocked), 1 total mutation call
        assert mutator.tabu_hit_rate > 0.0
