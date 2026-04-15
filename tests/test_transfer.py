# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.transfer — TransferAdapter and TransferBank."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.transfer import TransferAdapter, TransferBank, TransferRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(fitness: float = 0.8, prompt: str = "source system prompt") -> Agent:
    g = Genome(system_prompt=prompt, strategy="chain-of-thought")
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _backend(response: str = "adapted prompt") -> MagicMock:
    b = MagicMock()
    b.generate.return_value = response
    return b


# ---------------------------------------------------------------------------
# TransferAdapter — init
# ---------------------------------------------------------------------------


class TestTransferAdapterInit:
    def test_defaults(self) -> None:
        adapter = TransferAdapter(backend=_backend())
        assert adapter._intensity == "medium"
        assert adapter._temperature == 0.6

    def test_records_starts_empty(self) -> None:
        adapter = TransferAdapter(backend=_backend())
        assert adapter.records == []


# ---------------------------------------------------------------------------
# adapt — returns Agent
# ---------------------------------------------------------------------------


class TestAdapt:
    def test_returns_agent(self) -> None:
        adapter = TransferAdapter(backend=_backend("new prompt"))
        source = _make_agent()
        result = adapter.adapt(source, "Explain quantum physics")
        assert isinstance(result, Agent)

    def test_adapted_prompt_from_llm(self) -> None:
        adapter = TransferAdapter(backend=_backend("adapted system prompt"))
        source = _make_agent()
        result = adapter.adapt(source, "new task")
        assert result.genome.system_prompt == "adapted system prompt"

    def test_source_temperature_preserved(self) -> None:
        adapter = TransferAdapter(backend=_backend("p"))
        source = _make_agent()
        source.genome.temperature = 0.42
        result = adapter.adapt(source, "task")
        assert pytest.approx(result.genome.temperature, abs=1e-9) == 0.42

    def test_strategy_reflects_transfer(self) -> None:
        adapter = TransferAdapter(backend=_backend("p"), intensity="light")
        source = _make_agent()
        result = adapter.adapt(source, "target task here", source_domain="math")
        assert "transfer" in result.genome.strategy
        assert "math" in result.genome.strategy

    def test_record_stored(self) -> None:
        adapter = TransferAdapter(backend=_backend("p"))
        source = _make_agent()
        adapter.adapt(source, "some task", source_domain="science")
        assert len(adapter.records) == 1
        rec = adapter.records[0]
        assert isinstance(rec, TransferRecord)
        assert rec.intensity == "medium"
        assert rec.source_domain == "science"

    def test_records_returns_copy(self) -> None:
        adapter = TransferAdapter(backend=_backend("p"))
        adapter.adapt(_make_agent(), "t")
        r1 = adapter.records
        r1.clear()
        assert len(adapter.records) == 1

    def test_llm_failure_fallback(self) -> None:
        b = MagicMock()
        b.generate.side_effect = RuntimeError("API error")
        adapter = TransferAdapter(backend=b)
        source = _make_agent(prompt="original source prompt")
        result = adapter.adapt(source, "new task")
        assert "original source prompt" in result.genome.system_prompt

    def test_temperature_override(self) -> None:
        b = MagicMock()
        b.generate.return_value = "ok"
        adapter = TransferAdapter(backend=b)
        adapter.adapt(_make_agent(), "task", temperature=0.1)
        _, kwargs = b.generate.call_args
        assert kwargs.get("temperature") == 0.1


class TestAdaptIntensities:
    def test_light_intensity(self) -> None:
        b = MagicMock()
        b.generate.return_value = "light adapted"
        adapter = TransferAdapter(backend=b, intensity="light")
        adapter.adapt(_make_agent(), "task")
        prompt_sent = b.generate.call_args[0][0]
        assert "light" in prompt_sent.lower()

    def test_medium_intensity(self) -> None:
        b = MagicMock()
        b.generate.return_value = "medium adapted"
        adapter = TransferAdapter(backend=b, intensity="medium")
        adapter.adapt(_make_agent(), "task")
        prompt_sent = b.generate.call_args[0][0]
        assert "medium" in prompt_sent.lower()

    def test_heavy_intensity(self) -> None:
        b = MagicMock()
        b.generate.return_value = "heavy adapted"
        adapter = TransferAdapter(backend=b, intensity="heavy")
        adapter.adapt(_make_agent(), "task")
        prompt_sent = b.generate.call_args[0][0]
        assert "heavy" in prompt_sent.lower()


# ---------------------------------------------------------------------------
# TransferBank
# ---------------------------------------------------------------------------


class TestTransferBankInit:
    def test_defaults(self) -> None:
        bank = TransferBank()
        assert bank._max == 5

    def test_empty_domains(self) -> None:
        bank = TransferBank()
        assert bank.all_domains() == []


class TestTransferBankRegister:
    def test_register_and_count(self) -> None:
        bank = TransferBank()
        agent = _make_agent()
        bank.register(agent, domain="math")
        assert bank.count("math") == 1

    def test_evicts_oldest_over_max(self) -> None:
        bank = TransferBank(max_per_domain=2)
        for _ in range(3):
            bank.register(_make_agent(), domain="math")
        assert bank.count("math") == 2

    def test_multiple_domains(self) -> None:
        bank = TransferBank()
        bank.register(_make_agent(), domain="math")
        bank.register(_make_agent(), domain="science")
        assert set(bank.all_domains()) == {"math", "science"}


class TestTransferBankBestFor:
    def test_returns_none_for_unknown_domain(self) -> None:
        bank = TransferBank()
        assert bank.best_for("unknown") is None

    def test_returns_highest_fitness(self) -> None:
        bank = TransferBank()
        low = _make_agent(fitness=0.3)
        high = _make_agent(fitness=0.9)
        bank.register(low, domain="math")
        bank.register(high, domain="math")
        best = bank.best_for("math")
        assert best is high

    def test_none_fitness_treated_as_zero(self) -> None:
        bank = TransferBank()
        a = _make_agent(fitness=0.5)
        b_agent = Agent(genome=Genome(system_prompt="no fitness"))
        # b_agent.fitness is None
        bank.register(a, domain="x")
        bank.register(b_agent, domain="x")
        best = bank.best_for("x")
        assert best is a
