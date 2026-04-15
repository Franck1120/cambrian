"""Round 6 test suite — A2A, CLI tools, Pareto/NSGA-II, Archipelago,
Speculative execution, Reward shaping, Export, CLI run command.

All network calls are mocked.  Tests run offline with no API keys.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_genome(**kwargs: Any):  # type: ignore[no-untyped-def]
    from cambrian.agent import Genome
    defaults = dict(
        system_prompt="You solve problems step by step.",
        strategy="chain-of-thought",
        temperature=0.7,
        model="gpt-4o-mini",
    )
    defaults.update(kwargs)
    return Genome(**defaults)


def _make_agent(fitness: float = 0.5, **kwargs: Any):  # type: ignore[no-untyped-def]
    from cambrian.agent import Agent
    backend = MagicMock()
    backend.generate.return_value = "answer"
    agent = Agent(genome=_make_genome(**kwargs), backend=backend)
    agent.fitness = fitness
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# A2A Protocol
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentCard:
    def test_matches_domain(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=["python", "code"], confidence=0.8)
        assert card.matches("Write a python function")
        assert not card.matches("Compose a haiku")

    def test_matches_confidence_threshold(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=["code"], confidence=0.3)
        assert not card.matches("code review", threshold=0.5)

    def test_relevance_score_zero_for_no_match(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=["math"], confidence=0.9)
        assert card.relevance_score("write a poem") == 0.0

    def test_relevance_score_partial_match(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=["python", "math", "logic"], confidence=1.0)
        score = card.relevance_score("write a python script")
        assert 0.0 < score < 1.0

    def test_relevance_score_full_match(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=["python"], confidence=1.0)
        score = card.relevance_score("write a python function")
        assert score == 1.0

    def test_empty_domains_returns_zero(self):
        from cambrian.a2a import AgentCard
        card = AgentCard(domains=[], confidence=0.8)
        assert card.relevance_score("anything") == 0.0


class TestAgentNetwork:
    def _network_with_agents(self):
        from cambrian.a2a import AgentCard, AgentNetwork
        net = AgentNetwork()
        for fitness, domains in [(0.9, ["python"]), (0.7, ["math"]), (0.5, ["writing"])]:
            agent = _make_agent(fitness=fitness)
            card = AgentCard(domains=domains, confidence=fitness)
            net.register(agent, card)
        return net

    def test_register_and_size(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        net.register(_make_agent())
        assert net.network_size == 1

    def test_unregister(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        agent = _make_agent()
        net.register(agent)
        net.unregister(agent.id)
        assert net.network_size == 0

    def test_route_returns_best_match(self):
        net = self._network_with_agents()
        best = net.route("implement a python algorithm")
        assert best is not None
        assert best.fitness == 0.9

    def test_route_respects_exclude(self):
        net = self._network_with_agents()
        agent_ids = net.agent_ids()
        best = net.route("python task", exclude=[agent_ids[0]])
        assert best is not None
        assert best.id != agent_ids[0]

    def test_route_empty_network_returns_none(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        assert net.route("task") is None

    def test_delegate_returns_a2a_message(self):
        net = self._network_with_agents()
        msg = net.delegate("write a python function", sender_id="root")
        assert msg.result == "answer"
        assert msg.latency_ms >= 0.0

    def test_delegate_no_agents_gives_error_metadata(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        msg = net.delegate("task")
        assert "error" in msg.metadata

    def test_broadcast_queries_all_agents(self):
        net = self._network_with_agents()
        messages = net.broadcast("explain recursion", sender_id="root")
        assert len(messages) == 3

    def test_broadcast_top_n(self):
        net = self._network_with_agents()
        messages = net.broadcast("task", top_n=2)
        assert len(messages) == 2

    def test_chain_pipelines_output(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        a1 = _make_agent()
        a2 = _make_agent()
        a1.backend.generate.return_value = "step1_output"
        a2.backend.generate.return_value = "step2_output"
        net.register(a1)
        net.register(a2)
        msg = net.chain("initial task", agent_ids=[a1.id, a2.id])
        assert msg.result == "step2_output"
        assert msg.metadata["hop"] == 1

    def test_chain_raises_on_empty(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        with pytest.raises(ValueError, match="at least one"):
            net.chain("task", agent_ids=[])

    def test_chain_raises_on_too_many_hops(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork(max_hops=2)
        with pytest.raises(ValueError, match="max_hops"):
            net.chain("task", agent_ids=["a", "b", "c"])

    def test_majority_vote_returns_most_common(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        for _ in range(3):
            a = _make_agent()
            a.backend.generate.return_value = "42"
            net.register(a)
        result = net.majority_vote("What is 6*7?", top_n=3)
        assert result == "42"

    def test_summary_structure(self):
        net = self._network_with_agents()
        net.delegate("test")
        s = net.summary()
        assert "network_size" in s
        assert s["network_size"] == 3
        assert "messages" in s

    def test_message_log_grows(self):
        net = self._network_with_agents()
        net.delegate("task1")
        net.delegate("task2")
        assert len(net.message_log()) == 2

    def test_register_population(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        agents = [_make_agent(fitness=0.5 + i * 0.1) for i in range(4)]
        net.register_population(agents)
        assert net.network_size == 4

    def test_auto_card_generation(self):
        from cambrian.a2a import AgentNetwork
        net = AgentNetwork()
        agent = _make_agent(
            system_prompt="You are a Python coding assistant."
        )
        net.register_population([agent])
        assert net.network_size == 1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Tools
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLITool:
    def test_name_validation(self):
        from cambrian.cli_tools import CLITool
        with pytest.raises(ValueError, match="alphanumeric"):
            CLITool(name="bad-name", command_template="echo {input}")

    def test_run_success(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(
            name="echo_tool",
            command_template="echo hello",
            shell=True,
        )
        result = tool.run("ignored")
        assert result.ok is True
        assert "hello" in result.output

    def test_run_timeout(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(name="slow_tool", command_template="echo {input}", timeout=0.001)
        result = tool.run("x")
        assert isinstance(result.output, str)

    def test_template_error(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(name="bad_tpl", command_template="{bad_key}")
        result = tool.run("anything")
        assert not result.ok
        assert "template error" in result.output

    def test_call_shorthand(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(name="echotool", command_template="echo {input}", shell=True)
        out = tool("world")
        assert isinstance(out, str)

    def test_call_count_increments(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(name="counter", command_template="echo {input}", shell=True)
        tool.run("a")
        tool.run("b")
        assert tool._call_count == 2

    def test_max_output_chars_truncation(self):
        from cambrian.cli_tools import CLITool
        tool = CLITool(
            name="big_echo",
            command_template="echo {input}",
            shell=True,
            max_output_chars=5,
        )
        result = tool.run("hello world this is a long string")
        assert len(result.output) <= 5


class TestCLIToolkit:
    def _make_toolkit(self):
        from cambrian.cli_tools import CLITool, CLIToolkit
        toolkit = CLIToolkit()
        toolkit.add(CLITool(name="echo_t", command_template="echo {input}", shell=True))
        return toolkit

    def test_add_and_get(self):
        from cambrian.cli_tools import CLITool, CLIToolkit
        tk = CLIToolkit()
        tool = CLITool(name="mytool", command_template="echo {input}", shell=True)
        tk.add(tool)
        assert tk.get("mytool") is tool

    def test_get_missing_returns_none(self):
        from cambrian.cli_tools import CLIToolkit
        tk = CLIToolkit()
        assert tk.get("nonexistent") is None

    def test_system_prompt_block_lists_tools(self):
        tk = self._make_toolkit()
        block = tk.system_prompt_block()
        assert "echo_t" in block
        assert "TOOL" in block

    def test_parse_and_execute_valid(self):
        tk = self._make_toolkit()
        response = "[TOOL: echo_t | hello]"
        results = tk.parse_and_execute(response)
        assert len(results) == 1
        assert results[0].ok

    def test_parse_and_execute_unknown_tool(self):
        tk = self._make_toolkit()
        results = tk.parse_and_execute("[TOOL: ghost | data]")
        assert len(results) == 1
        assert not results[0].ok
        assert "unknown tool" in results[0].output

    def test_augment_response_injects_result(self):
        tk = self._make_toolkit()
        response = "Before [TOOL: echo_t | hi] After"
        augmented = tk.augment_response(response)
        assert "[RESULT:" in augmented

    def test_tool_names_property(self):
        tk = self._make_toolkit()
        assert "echo_t" in tk.tool_names

    def test_make_python_tool_factory(self):
        from cambrian.cli_tools import make_python_tool
        tool = make_python_tool()
        assert tool.name == "python_exec"

    def test_make_shell_tool_factory(self):
        from cambrian.cli_tools import make_shell_tool
        tool = make_shell_tool()
        assert tool.name == "shell"


# ═══════════════════════════════════════════════════════════════════════════════
# Pareto / NSGA-II
# ═══════════════════════════════════════════════════════════════════════════════


class TestObjectiveVector:
    def test_dominates_simple(self):
        from cambrian.pareto import ObjectiveVector
        v1 = ObjectiveVector("a", scores={"x": 0.9, "y": 0.8})
        v2 = ObjectiveVector("b", scores={"x": 0.5, "y": 0.5})
        assert v1.dominates(v2)
        assert not v2.dominates(v1)

    def test_no_dominance_when_tradeoff(self):
        from cambrian.pareto import ObjectiveVector
        v1 = ObjectiveVector("a", scores={"x": 0.9, "y": 0.1})
        v2 = ObjectiveVector("b", scores={"x": 0.1, "y": 0.9})
        assert not v1.dominates(v2)
        assert not v2.dominates(v1)

    def test_equal_scores_no_dominance(self):
        from cambrian.pareto import ObjectiveVector
        v1 = ObjectiveVector("a", scores={"x": 0.5})
        v2 = ObjectiveVector("b", scores={"x": 0.5})
        assert not v1.dominates(v2)


class TestParetoFront:
    def test_add_dominated_rejected(self):
        from cambrian.pareto import ObjectiveVector, ParetoFront
        front = ParetoFront()
        v1 = ObjectiveVector("a", scores={"f": 0.9})
        v2 = ObjectiveVector("b", scores={"f": 0.5})
        front.add(v1)
        assert not front.add(v2)
        assert front.size() == 1

    def test_new_dominant_evicts_old(self):
        from cambrian.pareto import ObjectiveVector, ParetoFront
        front = ParetoFront()
        v1 = ObjectiveVector("a", scores={"f": 0.5})
        v2 = ObjectiveVector("b", scores={"f": 0.9})
        front.add(v1)
        front.add(v2)
        assert front.size() == 1
        assert front.members()[0].agent_id == "b"

    def test_tradeoff_both_admitted(self):
        from cambrian.pareto import ObjectiveVector, ParetoFront
        front = ParetoFront()
        v1 = ObjectiveVector("a", scores={"x": 0.9, "y": 0.1})
        v2 = ObjectiveVector("b", scores={"x": 0.1, "y": 0.9})
        front.add(v1)
        front.add(v2)
        assert front.size() == 2

    def test_agents_method(self):
        from cambrian.pareto import ObjectiveVector, ParetoFront
        front = ParetoFront()
        a1 = _make_agent(fitness=0.9)
        v1 = ObjectiveVector(a1.id, scores={"f": 0.9})
        front.add(v1)
        agents = front.agents([a1])
        assert a1 in agents


class TestNSGAII:
    def _make_vectors(self):
        from cambrian.pareto import ObjectiveVector
        return [
            ObjectiveVector("a", scores={"perf": 0.9, "brevity": 0.3}),
            ObjectiveVector("b", scores={"perf": 0.7, "brevity": 0.7}),
            ObjectiveVector("c", scores={"perf": 0.3, "brevity": 0.9}),
            ObjectiveVector("d", scores={"perf": 0.1, "brevity": 0.1}),
        ]

    def test_non_dominated_sort_ranks(self):
        from cambrian.pareto import fast_non_dominated_sort
        vecs = self._make_vectors()
        fronts = fast_non_dominated_sort(vecs)
        # d is dominated by all others
        assert len(fronts) >= 2
        front0_ids = {v.agent_id for v in fronts[0]}
        assert "d" not in front0_ids

    def test_crowding_distance_boundaries_are_inf(self):
        from cambrian.pareto import ObjectiveVector, crowding_distance
        front = [
            ObjectiveVector("a", scores={"f": 0.1}),
            ObjectiveVector("b", scores={"f": 0.5}),
            ObjectiveVector("c", scores={"f": 0.9}),
        ]
        crowding_distance(front)
        sorted_front = sorted(front, key=lambda v: v.scores["f"])
        assert math.isinf(sorted_front[0].crowding)
        assert math.isinf(sorted_front[-1].crowding)

    def test_nsga2_select_returns_target_size(self):
        from cambrian.pareto import nsga2_select
        vecs = self._make_vectors()
        population = [_make_agent() for _ in vecs]
        for agent, vec in zip(population, vecs):
            vec.agent_id = agent.id
        selected = nsga2_select(population, vecs, target_size=2)
        assert len(selected) == 2

    def test_fitness_objective(self):
        from cambrian.pareto import fitness_objective
        agent = _make_agent(fitness=0.75)
        assert fitness_objective(agent) == 0.75

    def test_brevity_objective(self):
        from cambrian.pareto import brevity_objective
        short_agent = _make_agent(system_prompt="Hi")
        long_agent = _make_agent(system_prompt="x" * 10000)
        assert brevity_objective(short_agent) > brevity_objective(long_agent)
        assert brevity_objective(long_agent) == 0.0

    def test_attach_diversity_scores(self):
        from cambrian.pareto import ObjectiveVector, attach_diversity_scores
        agents = [
            _make_agent(system_prompt="Python expert agent."),
            _make_agent(system_prompt="Math genius assistant."),
        ]
        vecs = [ObjectiveVector(a.id, scores={"perf": 0.5}) for a in agents]
        attach_diversity_scores(agents, vecs)
        assert all("diversity" in v.scores for v in vecs)

    def test_empty_population_returns_empty(self):
        from cambrian.pareto import nsga2_select
        assert nsga2_select([], [], target_size=5) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Archipelago
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsland:
    def test_best_agent(self):
        from cambrian.archipelago import Island
        island = Island(island_id=0)
        a1 = _make_agent(fitness=0.3)
        a2 = _make_agent(fitness=0.8)
        island.population = [a1, a2]
        assert island.best_agent() is a2

    def test_top_agents(self):
        from cambrian.archipelago import Island
        island = Island(island_id=0)
        agents = [_make_agent(fitness=i * 0.1) for i in range(5)]
        island.population = agents
        top = island.top_agents(3)
        assert len(top) == 3
        assert all(top[i].fitness >= top[i + 1].fitness for i in range(2))  # type: ignore[operator]

    def test_receive_migrants_replaces_weakest(self):
        from cambrian.archipelago import Island
        island = Island(island_id=0)
        weaklings = [_make_agent(fitness=0.1) for _ in range(3)]
        island.population = weaklings
        strong = [_make_agent(fitness=0.9)]
        island.receive_migrants(strong)
        fitnesses = [a.fitness for a in island.population]
        assert max(fitnesses) == pytest.approx(0.9, abs=0.01)  # type: ignore[arg-type]

    def test_best_agent_none_when_empty(self):
        from cambrian.archipelago import Island
        island = Island(island_id=0)
        assert island.best_agent() is None


class TestArchipelago:
    def _make_arch(self):
        from cambrian.archipelago import Archipelago

        def _engine_factory() -> Any:
            engine = MagicMock()
            agents = [_make_agent(fitness=0.5 + i * 0.05) for i in range(4)]
            engine._seed_population.return_value = agents
            engine._evaluate_population.return_value = None
            engine._run_one_generation.side_effect = lambda pop, task: pop
            return engine

        return Archipelago(
            engine_factory=_engine_factory,
            n_islands=3,
            island_size=4,
            migration_interval=2,
            migration_rate=0.25,
            topology="ring",
        )

    def test_topology_validation(self):
        from cambrian.archipelago import Archipelago
        with pytest.raises(ValueError, match="topology"):
            Archipelago(engine_factory=MagicMock, topology="invalid")

    def test_migration_rate_validation(self):
        from cambrian.archipelago import Archipelago
        with pytest.raises(ValueError, match="migration_rate"):
            Archipelago(engine_factory=MagicMock, migration_rate=1.5)

    def test_neighbours_ring(self):
        from cambrian.archipelago import Archipelago
        arch = Archipelago(engine_factory=MagicMock, n_islands=4, topology="ring")
        assert arch._neighbours(0) == [1]
        assert arch._neighbours(3) == [0]

    def test_neighbours_all_to_all(self):
        from cambrian.archipelago import Archipelago
        arch = Archipelago(engine_factory=MagicMock, n_islands=4, topology="all_to_all")
        neighbours = arch._neighbours(0)
        assert sorted(neighbours) == [1, 2, 3]

    def test_island_summaries_structure(self):
        arch = self._make_arch()
        arch._seed_islands([_make_genome()], "task")
        summaries = arch.island_summaries()
        assert len(summaries) == 3
        for s in summaries:
            assert "island_id" in s
            assert "size" in s

    def test_repr(self):
        arch = self._make_arch()
        r = repr(arch)
        assert "Archipelago" in r
        assert "ring" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Speculative Execution
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpeculativeResult:
    def test_best_fitness(self):
        from cambrian.speculative import SpeculativeResult
        winner = _make_agent(fitness=0.9)
        r = SpeculativeResult(winner=winner, fitness_values=[0.5, 0.9, 0.7], k=3)
        assert r.best_fitness == 0.9

    def test_mean_fitness(self):
        from cambrian.speculative import SpeculativeResult
        winner = _make_agent(fitness=0.9)
        r = SpeculativeResult(winner=winner, fitness_values=[0.5, 0.9, None], k=3)
        assert r.mean_fitness == pytest.approx(0.7, abs=0.01)

    def test_improvement_over_mean(self):
        from cambrian.speculative import SpeculativeResult
        winner = _make_agent(fitness=0.9)
        r = SpeculativeResult(winner=winner, fitness_values=[0.7, 0.9], k=2)
        assert r.improvement_over_mean == pytest.approx(0.1, abs=0.01)


class TestSpeculate:
    def test_speculate_returns_best_candidate(self):
        from cambrian.speculative import speculate

        base_agent = _make_agent(fitness=0.3)
        candidates_returned = [_make_agent(fitness=0.5 + i * 0.1) for i in range(3)]
        call_count = [0]

        def mock_mutate(agent: Any, task: str) -> Any:
            c = candidates_returned[call_count[0] % len(candidates_returned)]
            call_count[0] += 1
            return c

        def score_fn(agent: Any, task: str) -> float:
            return agent.fitness or 0.0

        mock_mutator = MagicMock()
        mock_mutator.mutate.side_effect = mock_mutate

        result = asyncio.run(
            speculate(
                agent=base_agent,
                task="task",
                mutator=mock_mutator,
                evaluator=score_fn,
                k=3,
            )
        )
        assert result.k == 3
        assert result.best_fitness == pytest.approx(0.7, abs=0.01)

    def test_speculate_falls_back_to_parent_on_all_failures(self):
        from cambrian.speculative import speculate

        base_agent = _make_agent(fitness=0.5)

        def bad_mutate(agent: Any, task: str) -> Any:
            raise RuntimeError("mutation failed")

        mock_mutator = MagicMock()
        mock_mutator.mutate.side_effect = bad_mutate

        result = asyncio.run(
            speculate(
                agent=base_agent,
                task="task",
                mutator=mock_mutator,
                evaluator=lambda a, t: 0.5,
                k=2,
            )
        )
        assert result.winner is not None


class TestSpeculativeMutator:
    def test_mutate_returns_agent(self):
        from cambrian.speculative import SpeculativeMutator

        backend = MagicMock()
        candidate = _make_agent(fitness=0.8)

        with patch("cambrian.mutator.LLMMutator.mutate", return_value=candidate):
            mutator = SpeculativeMutator(backend=backend, k_candidates=2)
            result = mutator.mutate(_make_agent(), "task")
            assert result is not None

    def test_k_candidates_property(self):
        from cambrian.speculative import SpeculativeMutator
        m = SpeculativeMutator(backend=MagicMock(), k_candidates=5)
        assert m.k_candidates == 5

    def test_repr(self):
        from cambrian.speculative import SpeculativeMutator
        m = SpeculativeMutator(backend=MagicMock(), k_candidates=3)
        assert "SpeculativeMutator" in repr(m)
        assert "k=3" in repr(m)


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Shaping
# ═══════════════════════════════════════════════════════════════════════════════


class TestClipShaper:
    def test_clips_high(self):
        from cambrian.reward_shaping import ClipShaper
        shaper = ClipShaper(lambda a, t: 1.5)
        assert shaper(_make_agent(), "t") == 1.0

    def test_clips_low(self):
        from cambrian.reward_shaping import ClipShaper
        shaper = ClipShaper(lambda a, t: -0.5)
        assert shaper(_make_agent(), "t") == 0.0

    def test_passes_through_in_range(self):
        from cambrian.reward_shaping import ClipShaper
        shaper = ClipShaper(lambda a, t: 0.6)
        assert shaper(_make_agent(), "t") == pytest.approx(0.6)


class TestNormalisationShaper:
    def test_minmax_in_zero_one(self):
        from cambrian.reward_shaping import NormalisationShaper
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        shaper = NormalisationShaper(lambda a, t: 0.5, method="minmax")
        agent = _make_agent()
        for s in scores:
            shaper._window.append(s)
        result = shaper.shape(0.5, agent, "t")
        assert 0.0 <= result <= 1.0

    def test_zscore_centers_data(self):
        from cambrian.reward_shaping import NormalisationShaper
        shaper = NormalisationShaper(lambda a, t: 0.5, method="zscore")
        for _ in range(10):
            shaper._window.append(0.5)
        result = shaper.shape(0.5, _make_agent(), "t")
        assert abs(result) < 1.0

    def test_invalid_method_raises(self):
        from cambrian.reward_shaping import NormalisationShaper
        with pytest.raises(ValueError, match="method"):
            NormalisationShaper(lambda a, t: 0.0, method="bad")

    def test_reset_clears_window(self):
        from cambrian.reward_shaping import NormalisationShaper
        shaper = NormalisationShaper(lambda a, t: 0.5)
        shaper._window.extend([0.1, 0.9])
        shaper.reset()
        assert len(shaper._window) == 0


class TestPotentialShaper:
    def test_brevity_bonus_for_shorter_prompt(self):
        from cambrian.reward_shaping import PotentialShaper
        shaper = PotentialShaper(lambda a, t: 0.5)
        short_agent = _make_agent(system_prompt="Hi")
        long_agent = _make_agent(system_prompt="x" * 5000)
        shaper.shape(0.5, long_agent, "t")
        r_short = shaper.shape(0.5, short_agent, "t")
        r_long = shaper.shape(0.5, long_agent, "t")
        assert isinstance(r_short, float)
        assert isinstance(r_long, float)


class TestRankShaper:
    def test_rank_population_returns_fractions(self):
        from cambrian.reward_shaping import RankShaper
        shaper = RankShaper()
        agents = [_make_agent(fitness=f) for f in [0.1, 0.5, 0.9]]
        scores = [0.1, 0.5, 0.9]
        ranks = shaper.rank_population(agents, scores)
        assert len(ranks) == 3
        assert all(0.0 <= r <= 1.0 for r in ranks)
        assert ranks[2] > ranks[0]

    def test_single_agent_call(self):
        from cambrian.reward_shaping import RankShaper
        shaper = RankShaper(base_evaluator=lambda a, t: 0.7)
        result = shaper(_make_agent(), "task")
        assert 0.0 <= result <= 1.0

    def test_no_base_evaluator_raises(self):
        from cambrian.reward_shaping import RankShaper
        shaper = RankShaper()
        with pytest.raises(RuntimeError):
            shaper(_make_agent(), "task")


class TestCuriosityShaper:
    def test_novel_agent_gets_bonus(self):
        from cambrian.reward_shaping import CuriosityShaper
        shaper = CuriosityShaper(lambda a, t: 0.5, scale=0.2)
        agent = _make_agent(system_prompt="Completely unique prompt that has not been seen.")
        result = shaper(agent, "task")
        assert result >= 0.5

    def test_repeated_agent_lower_bonus(self):
        from cambrian.reward_shaping import CuriosityShaper
        shaper = CuriosityShaper(lambda a, t: 0.5, scale=0.2)
        prompt = "This prompt will be repeated multiple times."
        for _ in range(5):
            shaper(_make_agent(system_prompt=prompt), "task")
        result = shaper(_make_agent(system_prompt=prompt), "task")
        assert result <= 0.5 + 0.2 + 1e-9


class TestBuildShapedEvaluator:
    def test_clip_spec(self):
        from cambrian.reward_shaping import build_shaped_evaluator
        composed = build_shaped_evaluator(lambda a, t: 2.0, "clip")
        assert composed(_make_agent(), "task") == 1.0

    def test_chained_spec(self):
        from cambrian.reward_shaping import build_shaped_evaluator
        composed = build_shaped_evaluator(lambda a, t: 0.5, "clip+curiosity")
        result = composed(_make_agent(), "task")
        assert isinstance(result, float)

    def test_invalid_spec_raises(self):
        from cambrian.reward_shaping import build_shaped_evaluator
        with pytest.raises(ValueError, match="Unknown shaper"):
            build_shaped_evaluator(lambda a, t: 0.0, "magic")


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


class TestExport:
    def test_export_genome_json_and_load(self):
        from cambrian.export import export_genome_json, load_genome_json
        agent = _make_agent(fitness=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "genome.json"
            export_genome_json(agent, path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "genome" in data
            assert data["fitness"] == pytest.approx(0.8)

            genome = load_genome_json(path)
            assert genome.system_prompt == agent.genome.system_prompt

    def test_load_genome_json_missing_file(self):
        from cambrian.export import load_genome_json
        with pytest.raises(FileNotFoundError):
            load_genome_json("/nonexistent/path/genome.json")

    def test_export_standalone(self):
        from cambrian.export import export_standalone
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.py"
            result = export_standalone(agent, path)
            assert result.exists()
            content = result.read_text()
            assert "SYSTEM_PROMPT" in content
            assert "def run(" in content

    def test_export_mcp(self):
        from cambrian.export import export_mcp
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            out = export_mcp(agent, tmp)
            assert (Path(out) / "manifest.json").exists()
            assert (Path(out) / "mcp_server.py").exists()
            manifest = json.loads((Path(out) / "manifest.json").read_text())
            assert "tools" in manifest

    def test_export_api(self):
        from cambrian.export import export_api
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api.py"
            result = export_api(agent, path)
            assert result.exists()
            content = result.read_text()
            assert "FastAPI" in content
            assert "/run" in content

    def test_standalone_contains_model(self):
        from cambrian.export import export_standalone
        agent = _make_agent(model="gpt-4o")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.py"
            export_standalone(agent, path)
            content = path.read_text()
            assert "gpt-4o" in content

    def test_mcp_manifest_has_run_agent_tool(self):
        from cambrian.export import export_mcp
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            export_mcp(agent, tmp)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            tool_names = [t["name"] for t in manifest["tools"]]
            assert "run_agent" in tool_names


# ═══════════════════════════════════════════════════════════════════════════════
# CLI run command
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIRunCommand:
    def test_run_text_output(self):
        from click.testing import CliRunner
        from cambrian.cli import main
        from cambrian.export import export_genome_json

        agent = _make_agent()
        agent.backend.generate.return_value = "The answer is 42."

        with tempfile.TemporaryDirectory() as tmp:
            genome_path = Path(tmp) / "genome.json"
            export_genome_json(agent, genome_path)

            runner = CliRunner()
            with patch("cambrian.backends.openai_compat.OpenAICompatBackend.generate",
                       return_value="The answer is 42."):
                result = runner.invoke(
                    main,
                    [
                        "run",
                        "--agent", str(genome_path),
                        "--api-key", "test-key",
                        "What is 6 times 7?",
                    ],
                )
            assert result.exit_code == 0
            assert "42" in result.output

    def test_run_json_output(self):
        from click.testing import CliRunner
        from cambrian.cli import main
        from cambrian.export import export_genome_json

        agent = _make_agent()

        with tempfile.TemporaryDirectory() as tmp:
            genome_path = Path(tmp) / "genome.json"
            export_genome_json(agent, genome_path)

            runner = CliRunner()
            with patch("cambrian.backends.openai_compat.OpenAICompatBackend.generate",
                       return_value="json answer"):
                result = runner.invoke(
                    main,
                    [
                        "run",
                        "--agent", str(genome_path),
                        "--api-key", "test-key",
                        "--format", "json",
                        "What is 6 times 7?",
                    ],
                )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "result" in data
            assert "latency_ms" in data

    def test_run_no_api_key_exits_nonzero(self):
        from click.testing import CliRunner
        from cambrian.cli import main
        from cambrian.export import export_genome_json

        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            genome_path = Path(tmp) / "genome.json"
            export_genome_json(agent, genome_path)

            runner = CliRunner()
            with patch.dict("os.environ", {}, clear=True):
                result = runner.invoke(
                    main,
                    ["run", "--agent", str(genome_path), "task"],
                    catch_exceptions=False,
                )
            assert result.exit_code != 0
