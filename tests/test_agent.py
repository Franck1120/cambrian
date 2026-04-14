"""Tests for cambrian.agent — Genome and Agent."""

from __future__ import annotations

import pytest

from cambrian.agent import Agent, Genome


# ── Genome ────────────────────────────────────────────────────────────────────

class TestGenome:
    def test_defaults(self) -> None:
        g = Genome(system_prompt="hello")
        assert g.model == "gpt-4o-mini"
        assert g.temperature == 0.7
        assert g.strategy == "step-by-step"
        assert g.tools == []

    def test_to_dict_round_trip(self) -> None:
        g = Genome(
            system_prompt="You are a physicist.",
            tools=["search", "calculator"],
            strategy="react",
            temperature=0.5,
            model="gpt-4o",
        )
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.system_prompt == g.system_prompt
        assert g2.tools == g.tools
        assert g2.strategy == g.strategy
        assert g2.temperature == g.temperature
        assert g2.model == g.model

    def test_from_dict_missing_optional_fields(self) -> None:
        # Minimal dict — only required field
        g = Genome.from_dict({"system_prompt": "minimal"})
        assert g.system_prompt == "minimal"
        assert g.tools == []

    def test_from_dict_extra_fields_ignored(self) -> None:
        g = Genome.from_dict({"system_prompt": "x", "unknown_key": 42})
        assert g.system_prompt == "x"


# ── Agent ─────────────────────────────────────────────────────────────────────

class TestAgent:
    def _make_agent(self, prompt: str = "test") -> Agent:
        return Agent(genome=Genome(system_prompt=prompt))

    def test_unique_ids(self) -> None:
        a1 = self._make_agent()
        a2 = self._make_agent()
        assert a1.id != a2.id

    def test_fitness_initially_none(self) -> None:
        a = self._make_agent()
        assert a.fitness is None

    def test_set_fitness(self) -> None:
        a = self._make_agent()
        a._fitness = 0.75
        assert a.fitness == 0.75

    def test_clone_independence(self) -> None:
        a = self._make_agent("original")
        a._fitness = 0.5
        b = a.clone()
        assert b.id != a.id
        assert b.fitness is None  # clone resets fitness
        assert b.genome.system_prompt == a.genome.system_prompt
        # Mutate clone — original unaffected
        b.genome = Genome(system_prompt="modified")
        assert a.genome.system_prompt == "original"

    def test_run_returns_string(self) -> None:
        """run() must return the backend's response as a string."""

        class _FakeBackend:
            model_name = "stub"

            def generate(self, prompt: str, **kwargs: object) -> str:
                return "stub response"

        a = Agent(genome=Genome(system_prompt="test"), backend=_FakeBackend())  # type: ignore[arg-type]
        result = a.run("some task")
        assert result == "stub response"
        assert isinstance(result, str)

    def test_run_raises_without_backend(self) -> None:
        """run() must raise RuntimeError when no backend is configured."""
        a = self._make_agent()
        with pytest.raises(RuntimeError, match="no backend"):
            a.run("some task")

    def test_to_dict_includes_id_and_fitness(self) -> None:
        a = self._make_agent("hello")
        a._fitness = 0.9
        d = a.to_dict()
        assert "id" in d
        assert d["fitness"] == 0.9
        assert "genome" in d
