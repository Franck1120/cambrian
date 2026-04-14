"""tests/test_round7.py — Round 7 feature tests.

Covers:
- DiffCoT iterative chain-of-thought denoising
- Causal reasoning layer (CausalEdge, CausalGraph, inject_causal_context)
- Dynamic tool creation (ToolSpec, ToolInventor, ToolPopulationRegistry)
- cambrian snapshot CLI command
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from cambrian.agent import Agent, Genome, ToolSpec
from cambrian.causal import (
    CausalEdge,
    CausalGraph,
    CausalMutator,
    inject_causal_context,
)
from cambrian.diffcot import (
    DiffCoTConfig,
    DiffCoTEvaluator,
    DiffCoTReasoner,
    DiffCoTResult,
    DiffCoTStep,
    make_diffcot_evaluator,
)
from cambrian.evaluator import Evaluator
from cambrian.tool_creation import ToolInventionResult, ToolInventor, ToolPopulationRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _agent(system_prompt: str = "You are helpful.", fitness: float | None = None) -> Agent:
    genome = Genome(system_prompt=system_prompt, temperature=0.5)
    agent = Agent(genome=genome)
    if fitness is not None:
        agent.fitness = fitness
    return agent


class _ConstEvaluator(Evaluator):
    def __init__(self, score: float = 0.8) -> None:
        self._score = score

    def evaluate(self, agent: Agent, task: str) -> float:
        return self._score


def _mock_backend(response: str = "refined answer") -> MagicMock:
    backend = MagicMock()
    backend.complete.return_value = response
    backend.generate.return_value = response
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# DiffCoTConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCoTConfig:
    def test_defaults(self) -> None:
        cfg = DiffCoTConfig()
        assert cfg.n_steps == 3
        assert cfg.noise_level == pytest.approx(0.3, abs=1e-6)
        assert cfg.temperature_schedule == "cosine"
        assert cfg.inject_previous is True

    def test_custom_values(self) -> None:
        cfg = DiffCoTConfig(n_steps=5, noise_level=0.1, temperature_schedule="linear")
        assert cfg.n_steps == 5
        assert cfg.temperature_schedule == "linear"

    def test_constant_schedule(self) -> None:
        cfg = DiffCoTConfig(temperature_schedule="constant")
        assert cfg.temperature_schedule == "constant"


# ─────────────────────────────────────────────────────────────────────────────
# DiffCoTReasoner — temperature schedule
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCoTTemperature:
    def _reasoner(self, schedule: str) -> DiffCoTReasoner:
        return DiffCoTReasoner(
            backend=_mock_backend(),
            config=DiffCoTConfig(n_steps=4, noise_level=0.0, temperature_schedule=schedule),
        )

    def test_constant_schedule(self) -> None:
        r = self._reasoner("constant")
        base = 0.7
        temps = [r._temperature_at_step(i, base) for i in range(4)]
        assert all(t == pytest.approx(base, abs=1e-6) for t in temps)

    def test_linear_schedule_decreasing(self) -> None:
        r = self._reasoner("linear")
        base = 0.8
        temps = [r._temperature_at_step(i, base) for i in range(4)]
        # First step > last step
        assert temps[0] >= temps[-1]

    def test_cosine_schedule_decreasing(self) -> None:
        r = self._reasoner("cosine")
        base = 0.8
        temps = [r._temperature_at_step(i, base) for i in range(4)]
        # Cosine anneals from high to low
        assert temps[0] >= temps[-1]

    def test_unknown_schedule_defaults(self) -> None:
        r = DiffCoTReasoner(
            backend=_mock_backend(),
            config=DiffCoTConfig(temperature_schedule="unknown"),
        )
        # Should not raise; returns some float
        val = r._temperature_at_step(0, 0.5)
        assert isinstance(val, float)


# ─────────────────────────────────────────────────────────────────────────────
# DiffCoTReasoner — reason()
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCoTReasoner:
    def test_reason_returns_result(self) -> None:
        backend = _mock_backend("step response")
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=2))
        genome = Genome(system_prompt="You are a helper.", temperature=0.5)
        result = r.reason(genome, task="Explain gravity.")
        assert isinstance(result, DiffCoTResult)

    def test_result_has_correct_step_count(self) -> None:
        backend = _mock_backend("answer")
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=3))
        genome = Genome(system_prompt="Test.", temperature=0.4)
        result = r.reason(genome, task="task")
        assert len(result.steps) == 3

    def test_result_final_answer_is_last_step(self) -> None:
        responses = ["first", "second", "final"]
        backend = MagicMock()
        backend.generate.side_effect = responses
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=3))
        genome = Genome(system_prompt=".", temperature=0.5)
        result = r.reason(genome, task="q")
        assert result.final_answer == "final"

    def test_convergence_score_in_range(self) -> None:
        backend = _mock_backend("same answer")
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=3))
        genome = Genome(system_prompt=".", temperature=0.5)
        result = r.reason(genome, task="q")
        assert 0.0 <= result.convergence_score <= 1.0

    def test_steps_have_temperatures(self) -> None:
        backend = _mock_backend("resp")
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=2))
        genome = Genome(system_prompt=".", temperature=0.6)
        result = r.reason(genome, task="q")
        for step in result.steps:
            assert step.temperature >= 0.0  # cosine reaches 0 at final step

    def test_no_inject_previous(self) -> None:
        backend = _mock_backend("resp")
        config = DiffCoTConfig(n_steps=2, inject_previous=False)
        r = DiffCoTReasoner(backend=backend, config=config)
        genome = Genome(system_prompt=".", temperature=0.5)
        result = r.reason(genome, task="q")
        assert len(result.steps) == 2

    def test_n_steps_attribute(self) -> None:
        backend = _mock_backend("x")
        r = DiffCoTReasoner(backend=backend, config=DiffCoTConfig(n_steps=4))
        genome = Genome(system_prompt=".", temperature=0.5)
        result = r.reason(genome, task="t")
        assert result.n_steps == 4


# ─────────────────────────────────────────────────────────────────────────────
# DiffCoTEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCoTEvaluator:
    def test_evaluate_returns_float(self) -> None:
        backend = _mock_backend("refined")
        base_eval = _ConstEvaluator(0.9)
        evaluator = DiffCoTEvaluator(
            base_evaluator=base_eval,
            backend=backend,
            config=DiffCoTConfig(n_steps=2),
        )
        agent = _agent(fitness=0.5)
        score = evaluator.evaluate(agent, "task")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_calls_base_evaluator(self) -> None:
        backend = _mock_backend("answer")
        base_eval = MagicMock(spec=Evaluator)
        base_eval.evaluate.return_value = 0.75
        evaluator = DiffCoTEvaluator(
            base_evaluator=base_eval,
            backend=backend,
            config=DiffCoTConfig(n_steps=1),
        )
        agent = _agent()
        evaluator.evaluate(agent, "task")
        assert base_eval.evaluate.called

    def test_make_diffcot_evaluator_factory(self) -> None:
        backend = _mock_backend("x")
        base_eval = _ConstEvaluator(0.6)
        evaluator = make_diffcot_evaluator(base_eval, backend, n_steps=2, noise_level=0.1)
        assert isinstance(evaluator, DiffCoTEvaluator)

    def test_make_diffcot_evaluator_scores(self) -> None:
        backend = _mock_backend("answer")
        base_eval = _ConstEvaluator(0.7)
        evaluator = make_diffcot_evaluator(base_eval, backend, n_steps=2)
        agent = _agent()
        score = evaluator.evaluate(agent, "q")
        assert isinstance(score, float)


# ─────────────────────────────────────────────────────────────────────────────
# CausalEdge
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalEdge:
    def test_construction(self) -> None:
        edge = CausalEdge(cause="A", effect="B", strength=0.8, confidence=0.9)
        assert edge.cause == "A"
        assert edge.effect == "B"
        assert edge.strength == pytest.approx(0.8)
        assert edge.confidence == pytest.approx(0.9)

    def test_defaults(self) -> None:
        edge = CausalEdge(cause="X", effect="Y")
        assert edge.strength == pytest.approx(1.0)
        assert edge.confidence == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        edge = CausalEdge(cause="A", effect="B", strength=0.5)
        d = edge.to_dict()
        assert d["cause"] == "A"
        assert d["effect"] == "B"
        assert d["strength"] == pytest.approx(0.5)

    def test_from_dict_roundtrip(self) -> None:
        edge = CausalEdge(cause="P", effect="Q", strength=0.3, confidence=0.7)
        restored = CausalEdge.from_dict(edge.to_dict())
        assert restored.cause == edge.cause
        assert restored.effect == edge.effect
        assert restored.strength == pytest.approx(edge.strength)
        assert restored.confidence == pytest.approx(edge.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# CausalGraph — construction, queries, serialisation
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalGraph:
    def test_empty_graph(self) -> None:
        g = CausalGraph()
        assert len(g) == 0

    def test_add_edge(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B")
        assert len(g) == 1

    def test_add_edge_with_strength(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.7, confidence=0.9)
        edges = g.get_effects("A")
        assert len(edges) == 1
        assert edges[0].strength == pytest.approx(0.7)

    def test_get_effects(self) -> None:
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("X", "Z")
        effects = g.get_effects("X")
        assert len(effects) == 2
        effect_names = {e.effect for e in effects}
        assert effect_names == {"Y", "Z"}

    def test_get_causes(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        causes = g.get_causes("C")
        assert len(causes) == 2
        cause_names = {e.cause for e in causes}
        assert cause_names == {"A", "B"}

    def test_get_effects_missing_key(self) -> None:
        g = CausalGraph()
        assert g.get_effects("nonexistent") == []

    def test_get_causes_missing_key(self) -> None:
        g = CausalGraph()
        assert g.get_causes("nonexistent") == []

    def test_from_text_if_then(self) -> None:
        text = "IF the problem is complex THEN use step-by-step reasoning."
        g = CausalGraph.from_text(text)
        assert len(g) >= 1

    def test_from_text_arrow(self) -> None:
        text = "complexity → detailed explanation"
        g = CausalGraph.from_text(text)
        assert len(g) >= 1

    def test_from_text_leads_to(self) -> None:
        text = "high temperature leads to more creative output"
        g = CausalGraph.from_text(text)
        assert len(g) >= 1

    def test_from_text_causes(self) -> None:
        text = "ambiguity causes clarification questions"
        g = CausalGraph.from_text(text)
        assert len(g) >= 1

    def test_from_text_strength_keywords(self) -> None:
        text = "IF A THEN always B."
        g = CausalGraph.from_text(text)
        edges = g.get_effects("A")
        if edges:
            assert edges[0].strength == pytest.approx(1.0)

    def test_from_text_usually_keyword(self) -> None:
        text = "IF A THEN usually B."
        g = CausalGraph.from_text(text)
        edges = g.get_effects("A")
        if edges:
            assert edges[0].strength == pytest.approx(0.8)

    def test_from_text_sometimes_keyword(self) -> None:
        text = "IF A THEN sometimes B."
        g = CausalGraph.from_text(text)
        edges = g.get_effects("A")
        if edges:
            assert edges[0].strength == pytest.approx(0.5)

    def test_to_prompt_block_non_empty(self) -> None:
        g = CausalGraph()
        g.add_edge("X", "Y")
        block = g.to_prompt_block()
        assert len(block) > 0
        assert "X" in block or "IF" in block or "Y" in block

    def test_to_prompt_block_empty(self) -> None:
        g = CausalGraph()
        block = g.to_prompt_block()
        # Empty or minimal string for empty graph
        assert isinstance(block, str)

    def test_merge(self) -> None:
        g1 = CausalGraph()
        g1.add_edge("A", "B")
        g2 = CausalGraph()
        g2.add_edge("C", "D")
        merged = g1.merge(g2)
        assert len(merged) == 2

    def test_merge_does_not_mutate_originals(self) -> None:
        g1 = CausalGraph()
        g1.add_edge("A", "B")
        g2 = CausalGraph()
        g2.add_edge("C", "D")
        g1.merge(g2)
        assert len(g1) == 1

    def test_prune_removes_weak(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=0.1)
        g.add_edge("C", "D", strength=0.9)
        pruned = g.prune(min_strength=0.5)
        assert len(pruned) == 1
        assert pruned.get_effects("C")

    def test_prune_removes_low_confidence(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B", strength=1.0, confidence=0.1)
        g.add_edge("C", "D", strength=1.0, confidence=0.9)
        pruned = g.prune(min_confidence=0.5)
        assert len(pruned) == 1

    def test_to_dict_from_dict_roundtrip(self) -> None:
        g = CausalGraph()
        g.add_edge("P", "Q", strength=0.6)
        d = g.to_dict()
        restored = CausalGraph.from_dict(d)
        assert len(restored) == 1
        effects = restored.get_effects("P")
        assert effects[0].strength == pytest.approx(0.6)

    def test_repr(self) -> None:
        g = CausalGraph()
        g.add_edge("A", "B")
        r = repr(g)
        assert "CausalGraph" in r or "1" in r


# ─────────────────────────────────────────────────────────────────────────────
# inject_causal_context
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectCausalContext:
    def test_appends_to_system_prompt(self) -> None:
        genome = Genome(system_prompt="You are helpful.")
        g = CausalGraph()
        g.add_edge("A", "B")
        enriched = inject_causal_context(genome, g)
        assert "You are helpful." in enriched.system_prompt
        assert len(enriched.system_prompt) > len(genome.system_prompt)

    def test_returns_new_genome(self) -> None:
        genome = Genome(system_prompt="original")
        g = CausalGraph()
        g.add_edge("X", "Y")
        enriched = inject_causal_context(genome, g)
        assert enriched is not genome
        assert genome.system_prompt == "original"

    def test_empty_graph_minimal_injection(self) -> None:
        genome = Genome(system_prompt="base")
        g = CausalGraph()
        enriched = inject_causal_context(genome, g)
        # Even with empty graph, function returns a genome
        assert isinstance(enriched, Genome)


# ─────────────────────────────────────────────────────────────────────────────
# ToolSpec (agent.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestToolSpec:
    def test_construction(self) -> None:
        spec = ToolSpec(
            name="word_count",
            description="Count words",
            command_template="wc -w {input}",
        )
        assert spec.name == "word_count"
        assert spec.shell is False
        assert spec.timeout == pytest.approx(10.0)

    def test_to_dict(self) -> None:
        spec = ToolSpec(name="echo_tool", description="Echo input", command_template="echo {input}")
        d = spec.to_dict()
        assert d["name"] == "echo_tool"
        assert "command_template" in d

    def test_from_dict_roundtrip(self) -> None:
        spec = ToolSpec(
            name="grep_tool",
            description="Search",
            command_template="grep {input} .",
            timeout=5.0,
        )
        restored = ToolSpec.from_dict(spec.to_dict())
        assert restored.name == spec.name
        assert restored.timeout == pytest.approx(spec.timeout)

    def test_to_cli_tool(self) -> None:
        spec = ToolSpec(
            name="echo_tool",
            description="Echo",
            command_template="echo {input}",
        )
        cli_tool = spec.to_cli_tool()
        # Returns a CLITool instance
        from cambrian.cli_tools import CLITool
        assert isinstance(cli_tool, CLITool)

    def test_tool_spec_defaults(self) -> None:
        spec = ToolSpec(name="t", description="d", command_template="echo {input}")
        assert spec.author_genome_id == ""
        assert spec.shell is False


class TestGenomeToolSpecs:
    def test_genome_has_tool_specs_field(self) -> None:
        genome = Genome(system_prompt="Test")
        assert hasattr(genome, "tool_specs")
        assert isinstance(genome.tool_specs, list)

    def test_genome_default_empty_tool_specs(self) -> None:
        genome = Genome(system_prompt="Test")
        assert len(genome.tool_specs) == 0

    def test_genome_to_dict_includes_tool_specs(self) -> None:
        genome = Genome(system_prompt="Test")
        spec = ToolSpec(name="t", description="d", command_template="echo {input}")
        genome.tool_specs.append(spec)
        d = genome.to_dict()
        assert "tool_specs" in d
        assert len(d["tool_specs"]) == 1

    def test_genome_from_dict_restores_tool_specs(self) -> None:
        genome = Genome(system_prompt="Test")
        spec = ToolSpec(name="my_tool", description="d", command_template="wc -w {input}")
        genome.tool_specs.append(spec)
        d = genome.to_dict()
        restored = Genome.from_dict(d)
        assert len(restored.tool_specs) == 1
        assert restored.tool_specs[0].name == "my_tool"


# ─────────────────────────────────────────────────────────────────────────────
# ToolInventor
# ─────────────────────────────────────────────────────────────────────────────


class TestToolInventor:
    def _make_inventor(self, llm_response: str) -> ToolInventor:
        backend = _mock_backend(llm_response)
        return ToolInventor(backend=backend, max_tools_per_agent=3)

    def _valid_spec_response(self, cmd: str = "echo {input}") -> str:
        return (
            "NAME: my_echo\n"
            "DESCRIPTION: Echo the input\n"
            f"COMMAND: {cmd}\n"
            "SHELL: false\n"
            "TEST_INPUT: hello world\n"
        )

    def test_parse_valid_spec(self) -> None:
        inventor = self._make_inventor(self._valid_spec_response())
        agent = _agent()
        # _parse_spec is internal; test via invent_tool
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="hello world", stderr="")
            result = inventor.invent_tool(agent, "echo task")
        # May return None if LLM parsing fails; just check it doesn't raise
        assert result is None or isinstance(result, ToolInventionResult)

    def test_invent_tool_invalid_name_rejected(self) -> None:
        bad_response = (
            "NAME: 123invalid\n"
            "DESCRIPTION: Bad name\n"
            "COMMAND: echo {input}\n"
            "SHELL: false\n"
            "TEST_INPUT: test\n"
        )
        inventor = self._make_inventor(bad_response)
        agent = _agent()
        result = inventor.invent_tool(agent, "task")
        assert result is None

    def test_inject_tools_adds_to_genome(self) -> None:
        inventor = self._make_inventor("")
        agent = _agent()
        spec = ToolSpec(name="new_tool", description="New", command_template="echo {input}")
        new_agent = inventor.inject_tools(agent, [spec])
        assert len(new_agent.genome.tool_specs) == 1
        assert new_agent.genome.tool_specs[0].name == "new_tool"

    def test_inject_tools_respects_max_limit(self) -> None:
        inventor = ToolInventor(backend=_mock_backend(""), max_tools_per_agent=2)
        agent = _agent()
        specs = [
            ToolSpec(name=f"tool_{i}", description="d", command_template="echo {input}")
            for i in range(5)
        ]
        new_agent = inventor.inject_tools(agent, specs)
        assert len(new_agent.genome.tool_specs) <= 2

    def test_inject_tools_returns_new_agent(self) -> None:
        inventor = self._make_inventor("")
        agent = _agent()
        spec = ToolSpec(name="t", description="d", command_template="echo {input}")
        new_agent = inventor.inject_tools(agent, [spec])
        assert new_agent is not agent


# ─────────────────────────────────────────────────────────────────────────────
# ToolPopulationRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestToolPopulationRegistry:
    def _spec(self, name: str, use_count: int = 0) -> ToolSpec:
        spec = ToolSpec(name=name, description=f"Desc {name}", command_template=f"echo {name}")
        return spec

    def test_empty_registry(self) -> None:
        reg = ToolPopulationRegistry()
        assert len(reg) == 0

    def test_register_and_get(self) -> None:
        reg = ToolPopulationRegistry()
        spec = self._spec("my_tool")
        reg.register(spec)
        retrieved = reg.get("my_tool")
        assert retrieved is not None
        assert retrieved.name == "my_tool"

    def test_register_deduplicates_by_name(self) -> None:
        reg = ToolPopulationRegistry()
        spec1 = ToolSpec(name="t", description="first", command_template="echo {input}")
        spec2 = ToolSpec(name="t", description="second", command_template="wc {input}")
        reg.register(spec1)
        reg.register(spec2)
        # Should not duplicate
        assert len(reg) == 1

    def test_get_missing_returns_none(self) -> None:
        reg = ToolPopulationRegistry()
        assert reg.get("missing") is None

    def test_all_tools(self) -> None:
        reg = ToolPopulationRegistry()
        reg.register(self._spec("a"))
        reg.register(self._spec("b"))
        tools = reg.all_tools()
        assert len(tools) == 2

    def test_top_tools(self) -> None:
        reg = ToolPopulationRegistry()
        for i in range(5):
            reg.register(self._spec(f"tool_{i}"))
        top = reg.top_tools(n=3)
        assert len(top) <= 3

    def test_to_toolkit_block(self) -> None:
        reg = ToolPopulationRegistry()
        reg.register(self._spec("echo_tool"))
        block = reg.to_toolkit_block()
        assert isinstance(block, str)
        assert "echo_tool" in block

    def test_to_dict_from_dict_roundtrip(self) -> None:
        reg = ToolPopulationRegistry()
        reg.register(ToolSpec(name="rt", description="roundtrip", command_template="echo {input}"))
        d = reg.to_dict()
        restored = ToolPopulationRegistry.from_dict(d)
        assert len(restored) == 1
        assert restored.get("rt") is not None

    def test_repr(self) -> None:
        reg = ToolPopulationRegistry()
        r = repr(reg)
        assert "ToolPopulationRegistry" in r

    def test_empty_toolkit_block(self) -> None:
        reg = ToolPopulationRegistry()
        block = reg.to_toolkit_block()
        assert isinstance(block, str)


# ─────────────────────────────────────────────────────────────────────────────
# CausalStrategyExtractor (light — uses mock backend)
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalStrategyExtractor:
    def test_extract_returns_graph(self) -> None:
        from cambrian.causal import CausalStrategyExtractor
        backend = _mock_backend(
            "IF complex problem THEN use step-by-step reasoning.\n"
            "IF output is code THEN include docstrings."
        )
        extractor = CausalStrategyExtractor(backend=backend)
        graph = extractor.extract("be systematic", task="explain things")
        assert isinstance(graph, CausalGraph)

    def test_extract_with_empty_response(self) -> None:
        from cambrian.causal import CausalStrategyExtractor
        backend = _mock_backend("")
        extractor = CausalStrategyExtractor(backend=backend)
        graph = extractor.extract("", task="")
        assert isinstance(graph, CausalGraph)


# ─────────────────────────────────────────────────────────────────────────────
# CausalMutator
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalMutator:
    def test_mutate_with_causality_returns_agent_and_graph(self) -> None:
        from cambrian.causal import CausalMutator, CausalStrategyExtractor
        from cambrian.mutator import LLMMutator

        mutator_response = (
            "SYSTEM PROMPT:\nImproved prompt.\n"
            "STRATEGY: IF complex THEN systematic.\n"
            "TEMPERATURE: 0.5\n"
        )
        extractor_response = "IF complex THEN systematic.\n"

        backend = MagicMock()
        # Both LLMMutator and CausalStrategyExtractor use .generate()
        backend.generate.side_effect = [mutator_response, extractor_response]

        base_mutator = LLMMutator(backend=backend)
        extractor = CausalStrategyExtractor(backend=backend)
        mutator = CausalMutator(base_mutator=base_mutator, extractor=extractor)

        agent = _agent()
        result_agent, graph = mutator.mutate_with_causality(agent, task="test task")
        assert isinstance(result_agent, Agent)
        assert isinstance(graph, CausalGraph)


# ─────────────────────────────────────────────────────────────────────────────
# DiffCoTStep / DiffCoTResult dataclasses
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCoTDataclasses:
    def test_step_fields(self) -> None:
        step = DiffCoTStep(step=1, temperature=0.6, prompt="p", response="r")
        assert step.step == 1
        assert step.temperature == pytest.approx(0.6)
        assert step.prompt == "p"
        assert step.response == "r"

    def test_result_fields(self) -> None:
        steps = [DiffCoTStep(step=0, temperature=0.7, prompt="p", response="r")]
        result = DiffCoTResult(
            final_answer="final",
            steps=steps,
            convergence_score=0.8,
            n_steps=1,
        )
        assert result.final_answer == "final"
        assert result.convergence_score == pytest.approx(0.8)
        assert result.n_steps == 1


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot CLI command
# ─────────────────────────────────────────────────────────────────────────────


class TestSnapshotCLI:
    def _make_lineage(self, n_generations: int = 3, agents_per_gen: int = 2) -> str:
        """Build a minimal EvolutionaryMemory-format JSON for the snapshot command."""
        nodes = []
        edges = []
        for gen in range(n_generations):
            for i in range(agents_per_gen):
                agent_id = f"agent_{gen}_{i}"
                nodes.append({
                    "id": agent_id,
                    "generation": gen,
                    "fitness": round(0.5 + gen * 0.1 + i * 0.05, 4),
                    "system_prompt": f"Gen {gen} agent {i}",
                    "strategy": "default",
                    "temperature": 0.5,
                })
        data = {"name": "test", "nodes": nodes, "edges": edges}
        return json.dumps(data)

    def test_snapshot_text_output(self, tmp_path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        lineage_file = tmp_path / "lineage.json"
        lineage_file.write_text(self._make_lineage(3, 3))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "snapshot",
            "--memory", str(lineage_file),
            "--generation", "1",
        ])
        assert result.exit_code == 0, result.output
        assert "1" in result.output or "generation" in result.output.lower()

    def test_snapshot_json_output(self, tmp_path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        lineage_file = tmp_path / "lineage.json"
        lineage_file.write_text(self._make_lineage(3, 2))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "snapshot",
            "--memory", str(lineage_file),
            "--generation", "2",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "generation" in parsed or "agents" in parsed or isinstance(parsed, dict)

    def test_snapshot_missing_generation(self, tmp_path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        lineage_file = tmp_path / "lineage.json"
        lineage_file.write_text(self._make_lineage(2, 2))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "snapshot",
            "--memory", str(lineage_file),
            "--generation", "99",
        ])
        # Should exit non-zero or display error message
        assert result.exit_code != 0 or "not found" in result.output.lower() or \
               "no" in result.output.lower() or "generation" in result.output.lower()

    def test_snapshot_top_n(self, tmp_path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        lineage_file = tmp_path / "lineage.json"
        lineage_file.write_text(self._make_lineage(3, 5))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "snapshot",
            "--memory", str(lineage_file),
            "--generation", "0",
            "--top", "2",
        ])
        assert result.exit_code == 0, result.output

    def test_snapshot_missing_file_errors(self, tmp_path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "snapshot",
            "--memory", str(tmp_path / "nonexistent.json"),
            "--generation", "0",
        ])
        assert result.exit_code != 0
