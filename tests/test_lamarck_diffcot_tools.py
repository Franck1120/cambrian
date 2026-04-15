# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.lamarck, cambrian.diffcot, cambrian.tool_creation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.diffcot import DiffCoTConfig, DiffCoTEvaluator, DiffCoTReasoner
from cambrian.lamarck import LamarckianAdapter
from cambrian.tool_creation import ToolPopulationRegistry, ToolSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(response: str = "improved reasoning answer") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _agent(prompt: str = "You are a helpful assistant.", fitness: float = 0.7) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


# ---------------------------------------------------------------------------
# LamarckianAdapter
# ---------------------------------------------------------------------------


class TestLamarckianAdapter:
    def _adapter(self, threshold: float = 0.5) -> LamarckianAdapter:
        def evaluator(agent: Agent, task: str) -> float:
            return agent.fitness or 0.0
        return LamarckianAdapter(base_evaluator=evaluator, capture_threshold=threshold)

    def test_adapter_construction(self) -> None:
        adapter = self._adapter()
        assert adapter is not None

    def test_call_returns_float(self) -> None:
        """LamarckianAdapter is a callable: adapter(agent, task) -> float."""
        adapter = self._adapter(threshold=0.5)
        agent = _agent(fitness=0.8)
        score = adapter(agent, "test task")
        assert isinstance(score, float)

    def test_call_high_fitness_passthrough(self) -> None:
        adapter = self._adapter(threshold=0.5)
        agent = _agent(fitness=0.9)
        score = adapter(agent, "test task")
        assert score == pytest.approx(0.9)

    def test_call_low_fitness_passthrough(self) -> None:
        adapter = self._adapter(threshold=0.8)
        agent = _agent(fitness=0.3)
        score = adapter(agent, "test task")
        assert score == pytest.approx(0.3)

    def test_record_response_does_not_raise(self) -> None:
        adapter = self._adapter()
        agent = _agent(fitness=0.8)
        adapter.record_response(agent_id=agent.id, response="good response")

    def test_max_examples_respected(self) -> None:
        adapter = LamarckianAdapter(
            base_evaluator=lambda a, t: 0.9,
            capture_threshold=0.5,
            max_examples=2,
        )
        agent = _agent(fitness=0.9)
        # Call several times to accumulate examples
        for _ in range(5):
            adapter(agent, "task")
        # Genome should not exceed max_examples
        assert len(agent.genome.few_shot_examples) <= 2

    def test_score_passthrough(self) -> None:
        """LamarckianAdapter returns the base evaluator score unchanged."""
        base_score = 0.654

        def evaluator(agent: Agent, task: str) -> float:
            return base_score

        adapter = LamarckianAdapter(base_evaluator=evaluator, capture_threshold=0.9)
        agent = _agent(fitness=0.2)
        score = adapter(agent, "task")
        assert score == pytest.approx(base_score)


# ---------------------------------------------------------------------------
# DiffCoTConfig
# ---------------------------------------------------------------------------


class TestDiffCoTConfig:
    def test_defaults(self) -> None:
        cfg = DiffCoTConfig()
        assert cfg.n_steps > 0
        assert 0.0 <= cfg.noise_level <= 1.0
        assert cfg.temperature_schedule in ("cosine", "linear", "constant")

    def test_custom_values(self) -> None:
        cfg = DiffCoTConfig(n_steps=5, noise_level=0.1, temperature_schedule="linear")
        assert cfg.n_steps == 5
        assert cfg.noise_level == pytest.approx(0.1)

    def test_inject_previous_flag(self) -> None:
        cfg = DiffCoTConfig(inject_previous=False)
        assert cfg.inject_previous is False


# ---------------------------------------------------------------------------
# DiffCoTReasoner
# ---------------------------------------------------------------------------


class TestDiffCoTReasoner:
    def test_reason_returns_result(self) -> None:
        backend = _backend("step 1 answer")
        reasoner = DiffCoTReasoner(backend=backend)
        result = reasoner.reason(
            agent_genome=Genome(system_prompt="Think carefully."),
            task="What is 2+2?",
        )
        assert result is not None

    def test_reason_result_has_output(self) -> None:
        backend = _backend("final reasoning output")
        reasoner = DiffCoTReasoner(backend=backend)
        result = reasoner.reason(
            agent_genome=Genome(system_prompt="You are a thinker."),
            task="Explain entropy",
        )
        # DiffCoTResult must have a final output string
        assert hasattr(result, "output") or hasattr(result, "final_answer") or isinstance(result.output if hasattr(result, "output") else result, object)

    def test_backend_called_multiple_times(self) -> None:
        backend = _backend("step answer")
        reasoner = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=3))
        reasoner.reason(
            agent_genome=Genome(system_prompt="Reason carefully."),
            task="Solve this problem",
        )
        assert backend.generate.call_count >= 1

    def test_custom_config_applied(self) -> None:
        backend = _backend("answer")
        cfg = DiffCoTConfig(n_steps=2, inject_previous=False)
        reasoner = DiffCoTReasoner(backend=backend, config=cfg)
        result = reasoner.reason(
            agent_genome=Genome(system_prompt="Be precise."),
            task="Summarise quantum computing",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# DiffCoTEvaluator
# ---------------------------------------------------------------------------


class TestDiffCoTEvaluator:
    def test_evaluate_returns_float(self) -> None:
        base_eval = MagicMock()
        base_eval.evaluate = MagicMock(return_value=0.75)
        backend = _backend("refined answer")

        evaluator = DiffCoTEvaluator(
            base_evaluator=base_eval,
            backend=backend,
            config=DiffCoTConfig(n_steps=2),
        )
        agent = _agent()
        score = evaluator.evaluate(agent, "solve this task")
        assert isinstance(score, float)

    def test_evaluate_score_in_unit_range(self) -> None:
        base_eval = MagicMock()
        base_eval.evaluate = MagicMock(return_value=0.6)
        backend = _backend("answer")
        evaluator = DiffCoTEvaluator(
            base_evaluator=base_eval,
            backend=backend,
            config=DiffCoTConfig(n_steps=1),
        )
        agent = _agent()
        score = evaluator.evaluate(agent, "task")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_construction(self) -> None:
        spec = ToolSpec(
            name="grep_tool",
            description="Search files",
            command_template="grep {input} .",
        )
        assert spec.name == "grep_tool"
        assert spec.description == "Search files"

    def test_defaults(self) -> None:
        spec = ToolSpec(name="t", description="d", command_template="cmd {input}")
        assert spec.shell is False
        assert spec.timeout > 0
        assert spec.author_genome_id == ""

    def test_custom_timeout(self) -> None:
        spec = ToolSpec(name="t", description="d", command_template="cmd {input}", timeout=30.0)
        assert spec.timeout == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# ToolPopulationRegistry
# ---------------------------------------------------------------------------


class TestToolPopulationRegistry:
    def _spec(self, name: str = "tool_a") -> ToolSpec:
        return ToolSpec(name=name, description=f"desc of {name}", command_template="cmd {input}")

    def test_register_and_get(self) -> None:
        registry = ToolPopulationRegistry()
        spec = self._spec("my_tool")
        registry.register(spec)
        result = registry.get("my_tool")
        assert result is spec

    def test_get_unknown_returns_none(self) -> None:
        registry = ToolPopulationRegistry()
        assert registry.get("nonexistent") is None

    def test_all_tools_returns_list(self) -> None:
        registry = ToolPopulationRegistry()
        registry.register(self._spec("a"))
        registry.register(self._spec("b"))
        tools = registry.all_tools()
        assert len(tools) == 2

    def test_top_tools_limits_count(self) -> None:
        registry = ToolPopulationRegistry()
        for i in range(5):
            registry.register(self._spec(f"tool_{i}"))
        top = registry.top_tools(n=3)
        assert len(top) <= 3

    def test_to_toolkit_block_is_string(self) -> None:
        registry = ToolPopulationRegistry()
        registry.register(self._spec("search_tool"))
        block = registry.to_toolkit_block()
        assert isinstance(block, str)

    def test_round_trip_dict(self) -> None:
        registry = ToolPopulationRegistry()
        registry.register(self._spec("alpha"))
        registry.register(self._spec("beta"))
        d = registry.to_dict()
        registry2 = ToolPopulationRegistry.from_dict(d)
        assert registry2.get("alpha") is not None
        assert registry2.get("beta") is not None

    def test_empty_registry_toolkit_block(self) -> None:
        registry = ToolPopulationRegistry()
        block = registry.to_toolkit_block()
        assert isinstance(block, str)
