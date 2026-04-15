"""Tests for cambrian.hgt — HGTransfer, HGTPool, HGTPlasmid."""
from __future__ import annotations

import random

import pytest

from cambrian.agent import Agent, Genome
from cambrian.hgt import HGTEvent, HGTPlasmid, HGTPool, HGTransfer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    fitness: float = 0.8,
    prompt: str = "First sentence here. Second sentence too. Third and final.",
) -> Agent:
    g = Genome(system_prompt=prompt)
    a = Agent(genome=g)
    a.fitness = fitness
    return a


# ---------------------------------------------------------------------------
# HGTransfer — init
# ---------------------------------------------------------------------------


class TestHGTransferInit:
    def test_defaults(self) -> None:
        t = HGTransfer()
        assert t._n == 2
        assert t._mode == "suffix"
        assert t._threshold == 0.3

    def test_events_starts_empty(self) -> None:
        t = HGTransfer()
        assert t.events == []


# ---------------------------------------------------------------------------
# extract_plasmid
# ---------------------------------------------------------------------------


class TestExtractPlasmid:
    def test_returns_none_below_threshold(self) -> None:
        t = HGTransfer(fitness_threshold=0.5)
        agent = _make_agent(fitness=0.2)
        assert t.extract_plasmid(agent) is None

    def test_returns_plasmid_above_threshold(self) -> None:
        t = HGTransfer(fitness_threshold=0.5)
        agent = _make_agent(fitness=0.9)
        plasmid = t.extract_plasmid(agent)
        assert isinstance(plasmid, HGTPlasmid)

    def test_plasmid_content_is_substring_of_donor(self) -> None:
        t = HGTransfer(n_sentences=1, fitness_threshold=0.0)
        agent = _make_agent(prompt="Alpha. Beta. Gamma.")
        rng = random.Random(0)
        plasmid = t.extract_plasmid(agent, rng=rng)
        assert plasmid is not None
        # Content should be one of the sentences
        assert plasmid.content in ("Alpha.", "Beta.", "Gamma.")

    def test_plasmid_records_donor_id(self) -> None:
        t = HGTransfer(fitness_threshold=0.0)
        agent = _make_agent()
        plasmid = t.extract_plasmid(agent)
        assert plasmid is not None
        assert plasmid.donor_id == agent.agent_id

    def test_plasmid_records_fitness(self) -> None:
        t = HGTransfer(fitness_threshold=0.0)
        agent = _make_agent(fitness=0.77)
        plasmid = t.extract_plasmid(agent)
        assert plasmid is not None
        assert plasmid.donor_fitness == pytest.approx(0.77)

    def test_empty_prompt_returns_none(self) -> None:
        t = HGTransfer(fitness_threshold=0.0)
        agent = _make_agent(prompt="")
        assert t.extract_plasmid(agent) is None


# ---------------------------------------------------------------------------
# transfer — mode tests
# ---------------------------------------------------------------------------


class TestTransferSuffix:
    def test_suffix_mode_appends_plasmid(self) -> None:
        t = HGTransfer(n_sentences=1, mode="suffix", fitness_threshold=0.0)
        donor = _make_agent(prompt="Donor strategy. Is excellent.")
        recipient = _make_agent(prompt="Recipient default. Approach.")
        rng = random.Random(0)
        offspring = t.transfer(donor, recipient, rng=rng)
        assert offspring is not None
        assert "Recipient default." in offspring.genome.system_prompt
        assert "[HGT]" in offspring.genome.system_prompt

    def test_suffix_mode_event_recorded(self) -> None:
        t = HGTransfer(n_sentences=1, mode="suffix", fitness_threshold=0.0)
        donor = _make_agent()
        recipient = _make_agent(prompt="Recipient. Approach.")
        t.transfer(donor, recipient)
        assert len(t.events) == 1
        ev = t.events[0]
        assert isinstance(ev, HGTEvent)
        assert ev.mode == "suffix"


class TestTransferPrefix:
    def test_prefix_mode_prepends(self) -> None:
        t = HGTransfer(n_sentences=1, mode="prefix", fitness_threshold=0.0)
        donor = _make_agent(prompt="Donor strategy. Best approach.")
        recipient = _make_agent(prompt="Recipient. Content.")
        offspring = t.transfer(donor, recipient, rng=random.Random(0))
        assert offspring is not None
        assert offspring.genome.system_prompt.startswith("[HGT]")


class TestTransferReplace:
    def test_replace_mode_modifies_prompt(self) -> None:
        t = HGTransfer(n_sentences=1, mode="replace", fitness_threshold=0.0)
        donor = _make_agent(prompt="Superior strategy. Works well.")
        recipient = _make_agent(prompt="Short. Longer sentence here. Medium one.")
        offspring = t.transfer(donor, recipient, rng=random.Random(0))
        assert offspring is not None
        # Original prompt should be changed
        assert offspring.genome.system_prompt != recipient.genome.system_prompt


class TestTransferNone:
    def test_returns_none_when_donor_below_threshold(self) -> None:
        t = HGTransfer(fitness_threshold=0.9)
        donor = _make_agent(fitness=0.2)
        recipient = _make_agent()
        assert t.transfer(donor, recipient) is None

    def test_events_returns_copy(self) -> None:
        t = HGTransfer(fitness_threshold=0.0)
        donor = _make_agent()
        recipient = _make_agent(prompt="Alpha. Beta.")
        t.transfer(donor, recipient)
        e1 = t.events
        e1.clear()
        assert len(t.events) == 1


# ---------------------------------------------------------------------------
# HGTPool
# ---------------------------------------------------------------------------


class TestHGTPool:
    def test_starts_empty(self) -> None:
        pool = HGTPool()
        assert pool.size == 0

    def test_contribute_adds_plasmid(self) -> None:
        pool = HGTPool()
        agent = _make_agent(prompt="Alpha sentence. Beta one. Gamma too.")
        pool.contribute(agent, domain="math")
        assert pool.size == 1

    def test_fifo_eviction(self) -> None:
        pool = HGTPool(max_plasmids=2)
        for _ in range(3):
            pool.contribute(_make_agent(), domain="x")
        assert pool.size == 2

    def test_draw_returns_plasmid_for_domain(self) -> None:
        pool = HGTPool()
        agent = _make_agent(prompt="Alpha. Beta. Gamma.")
        pool.contribute(agent, domain="science")
        plasmid = pool.draw(domain="science", rng=random.Random(0))
        assert plasmid is not None
        assert plasmid.domain == "science"

    def test_draw_falls_back_to_any_domain(self) -> None:
        pool = HGTPool()
        agent = _make_agent(prompt="Alpha. Beta. Gamma.")
        pool.contribute(agent, domain="math")
        plasmid = pool.draw(domain="science", rng=random.Random(0))
        assert plasmid is not None

    def test_draw_returns_none_when_empty(self) -> None:
        pool = HGTPool()
        assert pool.draw() is None

    def test_best_for_returns_highest_fitness(self) -> None:
        pool = HGTPool()
        low = _make_agent(fitness=0.3, prompt="Low fitness agent prompt here.")
        high = _make_agent(fitness=0.9, prompt="High fitness agent prompt there.")
        pool.contribute(low, domain="x")
        pool.contribute(high, domain="x")
        best = pool.best_for("x")
        assert best is not None
        assert best.donor_fitness == pytest.approx(0.9)

    def test_best_for_returns_none_for_unknown_domain(self) -> None:
        pool = HGTPool()
        pool.contribute(_make_agent(prompt="Alpha. Beta. Gamma."), domain="math")
        assert pool.best_for("unknown") is None
