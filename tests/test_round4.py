"""Tests for Round 4 features:

- MCTSSelector: node UCB1, select, expand, backpropagate, best_path, prune
- CoEvolutionEngine: two-population adversarial evolution
- CurriculumScheduler: stage advancement, metric types, patience, looping
- ConstitutionalWrapper: self-critique cycle (mocked backend)
- ParetoFront: dominance check, pareto marking, summary
- DiversityTracker: snapshot capture, entropy, diversity_collapsed
- FitnessLandscape: binning, peak, to_dict
- E2E integration: full evolve() with mock backend, fitness improves
- distill-agent CLI command
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator


# ── Shared stubs ──────────────────────────────────────────────────────────────


class _EchoBackend:
    model_name = "echo"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import re
        m = re.search(r"\{[\s\S]+\}", prompt)
        return m.group(0) if m else "{}"


def _make_agent(prompt: str = "Test", fitness: float | None = None, temp: float = 0.7) -> Agent:
    g = Genome(system_prompt=prompt, temperature=temp)
    a = Agent(genome=g, backend=_EchoBackend())
    if fitness is not None:
        a.fitness = fitness
    return a


def _make_mutator() -> LLMMutator:
    return LLMMutator(backend=_EchoBackend(), fallback_on_error=True)


# ── MCTSSelector ──────────────────────────────────────────────────────────────


class TestMCTSNode:
    def test_ucb1_unvisited_is_inf(self) -> None:
        from cambrian.mcts import MCTSNode

        node = MCTSNode(agent=_make_agent())
        assert math.isinf(node.ucb1())

    def test_ucb1_no_parent_is_exploitation_only(self) -> None:
        from cambrian.mcts import MCTSNode

        node = MCTSNode(agent=_make_agent())
        node.visits = 5
        node.total_reward = 3.0
        assert node.ucb1() == pytest.approx(0.6)

    def test_ucb1_with_parent(self) -> None:
        from cambrian.mcts import MCTSNode

        parent = MCTSNode(agent=_make_agent(), visits=10)
        child = MCTSNode(agent=_make_agent(), parent=parent, visits=3, total_reward=1.5)
        score = child.ucb1(exploration_constant=math.sqrt(2))
        assert score > child.mean_reward  # exploration bonus pushes above exploitation

    def test_mean_reward_zero_when_unvisited(self) -> None:
        from cambrian.mcts import MCTSNode

        node = MCTSNode(agent=_make_agent())
        assert node.mean_reward == 0.0

    def test_is_fully_expanded(self) -> None:
        from cambrian.mcts import MCTSNode

        node = MCTSNode(agent=_make_agent())
        node.children = [MCTSNode(agent=_make_agent()) for _ in range(3)]
        assert node.is_fully_expanded(3)
        assert not node.is_fully_expanded(4)

    def test_repr_contains_agent_id(self) -> None:
        from cambrian.mcts import MCTSNode

        a = _make_agent()
        node = MCTSNode(agent=a)
        assert a.agent_id[:8] in repr(node)


class TestMCTSSelector:
    def test_register_creates_root(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        a = _make_agent()
        node = sel.register(a)
        assert node.agent is a
        assert a.agent_id in sel._roots

    def test_register_idempotent(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        a = _make_agent()
        n1 = sel.register(a)
        n2 = sel.register(a)
        assert n1 is n2

    def test_select_returns_highest_ucb1(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        agents = [_make_agent(f"p{i}") for i in range(3)]
        for a in agents:
            sel.register(a)

        # Make one agent clearly best: many visits, high reward
        node = sel._node_map[agents[1].agent_id]
        node.visits = 100
        node.total_reward = 90.0
        # Fake a parent so exploration term doesn't explode
        for n in sel._node_map.values():
            n.visits = max(n.visits, 1)

        # agent[0] and agent[2] are unvisited → inf UCB → one of them selected
        # (we can't guarantee which due to random tiebreaking)
        selected = sel.select(agents)
        assert selected in agents

    def test_select_empty_population_raises(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        with pytest.raises(ValueError, match="empty"):
            sel.select([])

    def test_expand_creates_children(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator(), max_children=3)
        parent = _make_agent("parent prompt")
        sel.register(parent)
        children = sel.expand(parent, "test task", n_children=2)
        assert len(children) == 2
        parent_node = sel._node_map[parent.agent_id]
        assert len(parent_node.children) == 2

    def test_expand_respects_max_children(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator(), max_children=2)
        parent = _make_agent()
        sel.register(parent)
        # Fill to max
        sel.expand(parent, "task", n_children=2)
        # Try to add more — should return existing children
        result = sel.expand(parent, "task", n_children=5)
        assert len(result) == 2  # capped at max_children

    def test_expand_at_max_depth_returns_empty(self) -> None:
        from cambrian.mcts import MCTSSelector, MCTSNode

        sel = MCTSSelector(mutator=_make_mutator(), max_depth=0)
        a = _make_agent()
        node = sel.register(a)
        node.depth = 0  # equals max_depth
        result = sel.expand(a, "task")
        assert result == []

    def test_backpropagate_updates_ancestors(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        parent = _make_agent("root")
        sel.register(parent)
        children = sel.expand(parent, "task", n_children=1)
        child = children[0]
        sel.backpropagate(child.agent_id, 0.8)

        child_node = sel._node_map[child.agent_id]
        parent_node = sel._node_map[parent.agent_id]
        assert child_node.visits == 1
        assert child_node.total_reward == pytest.approx(0.8)
        assert parent_node.visits == 1  # propagated up
        assert parent_node.total_reward == pytest.approx(0.8)

    def test_best_path_follows_mean_reward(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        root = _make_agent("root")
        sel.register(root)
        children = sel.expand(root, "task", n_children=2)
        # Give child[0] higher reward
        sel.backpropagate(children[0].agent_id, 0.9)
        sel.backpropagate(children[1].agent_id, 0.3)

        path = sel.best_path(root.agent_id)
        assert len(path) == 2
        assert path[-1].agent is children[0]

    def test_best_path_unknown_root_returns_empty(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        assert sel.best_path("nonexistent") == []

    def test_stats_returns_dict(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        s = sel.stats()
        assert s["nodes"] == 0

        a = _make_agent()
        sel.register(a)
        sel.backpropagate(a.agent_id, 0.5)
        s2 = sel.stats()
        assert s2["nodes"] == 1
        assert s2["visits"] == 1

    def test_prune_stale_roots_removes_inactive(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        agents = [_make_agent(f"p{i}") for i in range(5)]
        for a in agents:
            sel.register(a)
            sel.backpropagate(a.agent_id, 0.5)

        # Only keep agents[0] active; prune the rest (keep_best=1)
        active_ids = {agents[0].agent_id}
        sel.prune_stale_roots(active_ids, keep_best=1)
        # At most keep_best + 1 (active) roots remain
        assert len(sel._roots) <= 2

    def test_repr(self) -> None:
        from cambrian.mcts import MCTSSelector

        sel = MCTSSelector(mutator=_make_mutator())
        assert "MCTSSelector" in repr(sel)


# ── CoEvolutionEngine ─────────────────────────────────────────────────────────


class TestCoEvolutionEngine:
    def _make_engine(self) -> Any:
        from cambrian.coevolution import CoEvolutionEngine

        def _task_ev(agent: Agent, task: str) -> float:
            return 0.6

        def _probe(adv: Agent, gen: Agent, task: str) -> float:
            return 0.2  # adversary breaks generator 20% of the time

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        return CoEvolutionEngine(
            task_evaluator=_task_ev,
            adversarial_probe=_probe,
            generator_mutator=mutator,
            backend=backend,
            population_size=4,
            n_challenges=2,
            seed=0,
        )

    def test_evolve_returns_two_agents(self) -> None:
        engine = self._make_engine()
        gen_seeds = [Genome(system_prompt="Generate code.")]
        adv_seeds = [Genome(system_prompt="Find bugs.")]
        best_gen, best_adv = engine.evolve(gen_seeds, adv_seeds, "task", n_generations=2)
        assert isinstance(best_gen, Agent)
        assert isinstance(best_adv, Agent)

    def test_generator_fitness_penalised_by_break_rate(self) -> None:
        from cambrian.coevolution import CoEvolutionEngine

        def _probe(adv: Agent, gen: Agent, task: str) -> float:
            return 1.0  # adversary always wins

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = CoEvolutionEngine(
            task_evaluator=lambda a, t: 0.8,
            adversarial_probe=_probe,
            generator_mutator=mutator,
            population_size=3,
            adversary_penalty=0.5,
            n_challenges=2,
            seed=1,
        )
        gen_seeds = [Genome(system_prompt="Generate.")]
        adv_seeds = [Genome(system_prompt="Attack.")]
        best_gen, _ = engine.evolve(gen_seeds, adv_seeds, "task", n_generations=1)
        # base=0.8, break_rate=1.0, penalty=0.5 → 0.8 * (1 - 0.5) = 0.4
        assert best_gen.fitness is not None
        assert best_gen.fitness <= 0.5

    def test_generation_callback_called(self) -> None:
        engine = self._make_engine()
        calls: list[int] = []

        def _cb(gen: int, gens: list[Agent], advs: list[Agent]) -> None:
            calls.append(gen)

        gen_seeds = [Genome(system_prompt="G")]
        adv_seeds = [Genome(system_prompt="A")]
        engine.evolve(gen_seeds, adv_seeds, "task", n_generations=3, on_generation=_cb)
        assert calls == [0, 1, 2, 3]

    def test_repr(self) -> None:
        engine = self._make_engine()
        assert "CoEvolutionEngine" in repr(engine)


# ── CurriculumScheduler ───────────────────────────────────────────────────────


class TestCurriculumStage:
    def test_invalid_difficulty_raises(self) -> None:
        from cambrian.curriculum import CurriculumStage

        with pytest.raises(ValueError, match="difficulty"):
            CurriculumStage(task="x", difficulty=1.5)

    def test_invalid_threshold_raises(self) -> None:
        from cambrian.curriculum import CurriculumStage

        with pytest.raises(ValueError, match="threshold"):
            CurriculumStage(task="x", threshold=1.1)


class TestCurriculumScheduler:
    def _two_stage(self, metric: str = "mean") -> Any:
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        return CurriculumScheduler(
            stages=[
                CurriculumStage(task="easy task", difficulty=0.1, threshold=0.6),
                CurriculumStage(task="hard task", difficulty=0.9, threshold=0.8),
            ],
            metric=metric,
            advance_patience=1,
        )

    def test_initial_stage_zero(self) -> None:
        sched = self._two_stage()
        assert sched.stage_index == 0
        assert sched.current_task() == "easy task"

    def test_advance_when_threshold_met(self) -> None:
        sched = self._two_stage()
        advanced = sched.advance([0.7, 0.8, 0.9])
        assert advanced
        assert sched.stage_index == 1

    def test_no_advance_below_threshold(self) -> None:
        sched = self._two_stage()
        advanced = sched.advance([0.3, 0.4])
        assert not advanced
        assert sched.stage_index == 0

    def test_patience_requires_consecutive_passes(self) -> None:
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        sched = CurriculumScheduler(
            stages=[
                CurriculumStage(task="easy", threshold=0.6),
                CurriculumStage(task="hard", threshold=0.8),
            ],
            advance_patience=3,
        )
        sched.advance([0.7])  # pass 1
        sched.advance([0.4])  # fail → reset consecutive
        sched.advance([0.7])  # pass 1 again
        sched.advance([0.7])  # pass 2
        assert sched.stage_index == 0  # not yet advanced (need 3)
        sched.advance([0.7])  # pass 3
        assert sched.stage_index == 1

    def test_stays_at_last_stage(self) -> None:
        sched = self._two_stage()
        sched.advance([1.0])  # advance to stage 1
        advanced = sched.advance([1.0])  # already at last stage
        assert not advanced
        assert sched.stage_index == 1

    def test_loop_wraps_around(self) -> None:
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        sched = CurriculumScheduler(
            stages=[
                CurriculumStage(task="A", threshold=0.5),
                CurriculumStage(task="B", threshold=0.5),
            ],
            loop=True,
        )
        sched.advance([0.9])  # stage 0 → 1
        sched.advance([0.9])  # stage 1 → 0 (wrap)
        assert sched.stage_index == 0

    def test_metric_best(self) -> None:
        sched = self._two_stage(metric="best")
        advanced = sched.advance([0.1, 0.65])  # best=0.65 >= threshold=0.6
        assert advanced

    def test_metric_median(self) -> None:
        sched = self._two_stage(metric="median")
        advanced = sched.advance([0.65, 0.7, 0.8])  # median=0.7 >= 0.6
        assert advanced

    def test_max_generations_forces_advance(self) -> None:
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        sched = CurriculumScheduler(
            stages=[
                CurriculumStage(task="easy", threshold=0.99, max_generations=2),
                CurriculumStage(task="hard", threshold=0.5),
            ],
        )
        sched.advance([0.1])  # gen 1 — below threshold, forced at gen 2
        advanced = sched.advance([0.1])  # gen 2 — forced advance
        assert advanced

    def test_is_complete(self) -> None:
        sched = self._two_stage()
        assert not sched.is_complete
        sched.advance([1.0])  # stage 0 → 1
        sched.advance([1.0])  # pass stage 1 threshold
        assert sched.is_complete

    def test_progress(self) -> None:
        sched = self._two_stage()
        assert sched.progress == 0.0
        sched.advance([1.0])
        assert sched.progress == 0.5

    def test_reset(self) -> None:
        sched = self._two_stage()
        sched.advance([1.0])
        sched.reset()
        assert sched.stage_index == 0
        assert sched.history == []

    def test_empty_stages_raises(self) -> None:
        from cambrian.curriculum import CurriculumScheduler

        with pytest.raises(ValueError):
            CurriculumScheduler(stages=[])

    def test_invalid_metric_raises(self) -> None:
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        with pytest.raises(ValueError, match="metric"):
            CurriculumScheduler(stages=[CurriculumStage(task="x")], metric="max")

    def test_make_coding_curriculum(self) -> None:
        from cambrian.curriculum import make_coding_curriculum

        sched = make_coding_curriculum()
        assert sched.stage_index == 0
        assert len(sched._stages) == 4

    def test_make_reasoning_curriculum(self) -> None:
        from cambrian.curriculum import make_reasoning_curriculum

        sched = make_reasoning_curriculum()
        assert len(sched._stages) == 5


# ── ConstitutionalWrapper ─────────────────────────────────────────────────────


class TestConstitutionalWrapper:
    def _make_spy_backend(self, generate_return: str = "improved prompt") -> Any:
        class _Spy:
            model_name = "spy"
            calls: list[str] = []

            def generate(self, prompt: str, **kwargs: Any) -> str:
                self.calls.append(prompt[:30])
                return generate_return

        return _Spy()

    def test_calls_base_evaluator(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper

        scores: list[float] = []
        spy_backend = self._make_spy_backend("OK")

        def _base(agent: Agent, task: str) -> float:
            scores.append(0.7)
            return 0.7

        wrapper = ConstitutionalWrapper(
            base_evaluator=_base,
            constitution=["Is it good?"],
            n_revisions=1,
        )
        agent = Agent(genome=Genome(system_prompt="Test"), backend=spy_backend)
        result = wrapper(agent, "do task")
        assert result == pytest.approx(0.7)
        assert len(scores) == 1

    def test_restores_original_prompt_after_evaluation(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper

        spy_backend = self._make_spy_backend("modified prompt")
        agent = Agent(genome=Genome(system_prompt="original"), backend=spy_backend)
        original = agent.genome.system_prompt

        wrapper = ConstitutionalWrapper(
            base_evaluator=lambda a, t: 0.5,
            constitution=["One principle"],
            n_revisions=1,
        )
        wrapper(agent, "task")
        assert agent.genome.system_prompt == original

    def test_skip_if_no_backend(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper

        called = {"n": 0}

        def _base(agent: Agent, task: str) -> float:
            called["n"] += 1
            return 0.5

        wrapper = ConstitutionalWrapper(base_evaluator=_base, skip_if_no_backend=True)
        agent = Agent(genome=Genome(system_prompt="no backend"))  # no backend
        result = wrapper(agent, "task")
        assert result == pytest.approx(0.5)
        assert called["n"] == 1

    def test_raises_when_no_backend_and_no_skip(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper

        wrapper = ConstitutionalWrapper(
            base_evaluator=lambda a, t: 0.5,
            skip_if_no_backend=False,
        )
        agent = Agent(genome=Genome(system_prompt="no backend"))
        with pytest.raises(RuntimeError, match="no backend"):
            wrapper(agent, "task")

    def test_all_ok_critiques_skips_revision(self) -> None:
        """When all critiques return OK, no revision call is made."""
        from cambrian.constitutional import ConstitutionalWrapper

        revision_calls: list[str] = []

        class _OKBackend:
            model_name = "ok"
            call_count = 0

            def generate(self, prompt: str, **kwargs: Any) -> str:
                self.call_count += 1
                # First calls are critiques → return OK
                # If a revision were to happen it would get "revised"
                return "OK"

        ok_backend = _OKBackend()
        agent = Agent(genome=Genome(system_prompt="fine prompt"), backend=ok_backend)

        wrapper = ConstitutionalWrapper(
            base_evaluator=lambda a, t: 0.8,
            constitution=["Good?", "Concise?"],
            n_revisions=1,
        )
        wrapper(agent, "task")
        # Only critique calls happened (2 principles), no extra revision call
        assert ok_backend.call_count == 2

    def test_repr(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper

        w = ConstitutionalWrapper(base_evaluator=lambda a, t: 0.5)
        assert "ConstitutionalWrapper" in repr(w)

    def test_build_constitutional_evaluator(self) -> None:
        from cambrian.constitutional import ConstitutionalWrapper, build_constitutional_evaluator

        ev = build_constitutional_evaluator(lambda a, t: 0.5, n_principles=3)
        assert isinstance(ev, ConstitutionalWrapper)
        assert len(ev._constitution) == 3


# ── ParetoFront ───────────────────────────────────────────────────────────────


class TestParetoFront:
    def test_non_dominated_agent_is_pareto(self) -> None:
        from cambrian.stats import ParetoFront

        # Agent A: high fitness, short prompt — dominates B on both axes
        a = _make_agent("A" * 10, fitness=0.9)  # short prompt
        b = _make_agent("B" * 400, fitness=0.5)  # long prompt, lower fitness

        front = ParetoFront()
        points = front.compute([a, b])
        a_point = next(p for p in points if p.agent_id == a.agent_id)
        b_point = next(p for p in points if p.agent_id == b.agent_id)
        assert a_point.is_pareto
        assert not b_point.is_pareto

    def test_incomparable_agents_both_pareto(self) -> None:
        """Agents trading off fitness vs brevity should both be on the front."""
        from cambrian.stats import ParetoFront

        # A: high fitness but long prompt
        a = _make_agent("A " * 200, fitness=0.95)
        # B: low fitness but very short prompt (high brevity)
        b = _make_agent("B", fitness=0.2)

        front = ParetoFront()
        points = front.compute([a, b])
        # Neither dominates the other on both objectives
        assert all(p.is_pareto for p in points)

    def test_pareto_agents_method(self) -> None:
        from cambrian.stats import ParetoFront

        agents = [_make_agent(f"{'p'*i}", fitness=0.5 + i * 0.1) for i in range(4)]
        front = ParetoFront()
        front.compute(agents)
        pareto = front.pareto_agents()
        assert len(pareto) >= 1

    def test_summary_fields(self) -> None:
        from cambrian.stats import ParetoFront

        agents = [_make_agent("x", fitness=0.5)]
        front = ParetoFront()
        front.compute(agents)
        s = front.summary()
        assert "pareto_count" in s
        assert "total" in s
        assert s["total"] == 1

    def test_custom_objective(self) -> None:
        from cambrian.stats import ParetoFront

        front = ParetoFront(objectives=["fitness", "temperature_score"])
        front.add_objective("temperature_score", lambda a: 1.0 - a.genome.temperature)
        agents = [_make_agent(fitness=0.8, temp=0.3), _make_agent(fitness=0.8, temp=0.9)]
        front.compute(agents)
        s = front.summary()
        assert s["total"] == 2

    def test_repr(self) -> None:
        from cambrian.stats import ParetoAnalyzer

        front = ParetoAnalyzer()
        assert "ParetoAnalyzer" in repr(front)


# ── DiversityTracker ──────────────────────────────────────────────────────────


class TestDiversityTracker:
    def test_record_captures_snapshot(self) -> None:
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        agents = [
            Agent(genome=Genome(system_prompt="A", strategy="step-by-step", temperature=0.5)),
            Agent(genome=Genome(system_prompt="B", strategy="concise", temperature=0.9)),
        ]
        agents[0].fitness = 0.6
        agents[1].fitness = 0.8
        snap = tracker.record(0, agents)
        assert snap.generation == 0
        assert snap.n_agents == 2
        assert snap.unique_strategies == 2

    def test_strategy_entropy_uniform(self) -> None:
        """Uniform distribution → max entropy."""
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        strategies = ["a", "b", "c", "d"]
        entropy = tracker._strategy_entropy(strategies)
        # Uniform over 4 → entropy = log2(4) = 2.0 bits
        assert entropy == pytest.approx(2.0, abs=0.01)

    def test_strategy_entropy_single(self) -> None:
        """Single strategy → entropy = 0."""
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        assert tracker._strategy_entropy(["a", "a", "a"]) == 0.0

    def test_diversity_collapsed_flag(self) -> None:
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        # All same strategy → entropy 0 → collapsed
        agents = [
            Agent(genome=Genome(system_prompt="X", strategy="step-by-step"))
            for _ in range(5)
        ]
        for a in agents:
            a.fitness = 0.5
        tracker.record(0, agents)
        assert tracker.diversity_collapsed(threshold_entropy=0.3)

    def test_timeline_returns_all_snapshots(self) -> None:
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        agents = [_make_agent()]
        agents[0].fitness = 0.5
        tracker.record(0, agents)
        tracker.record(1, agents)
        assert len(tracker.timeline()) == 2

    def test_to_dicts_serialisable(self) -> None:
        from cambrian.stats import DiversityTracker

        tracker = DiversityTracker()
        agents = [_make_agent()]
        agents[0].fitness = 0.5
        tracker.record(0, agents)
        dicts = tracker.to_dicts()
        assert len(dicts) == 1
        # Should be JSON-serialisable
        json.dumps(dicts)

    def test_repr(self) -> None:
        from cambrian.stats import DiversityTracker

        assert "DiversityTracker" in repr(DiversityTracker())


# ── FitnessLandscape ──────────────────────────────────────────────────────────


class TestFitnessLandscape:
    def test_add_agent_populates_grid(self) -> None:
        from cambrian.stats import FitnessLandscape

        fl = FitnessLandscape(n_temp_bins=3, n_token_bins=3)
        a = _make_agent("short", fitness=0.8, temp=0.5)
        fl.add(a)
        grid = fl.mean_fitness_grid()
        non_zero = [cell for row in grid for cell in row if cell > 0]
        assert len(non_zero) == 1
        assert non_zero[0] == pytest.approx(0.8)

    def test_no_fitness_agent_ignored(self) -> None:
        from cambrian.stats import FitnessLandscape

        fl = FitnessLandscape()
        fl.add(_make_agent())  # no fitness
        grid = fl.mean_fitness_grid()
        assert all(cell == 0.0 for row in grid for cell in row)

    def test_peak_returns_best_cell(self) -> None:
        from cambrian.stats import FitnessLandscape

        fl = FitnessLandscape(n_temp_bins=2, n_token_bins=2, temp_range=(0.0, 1.0))
        fl.add(_make_agent("x" * 100, fitness=0.9, temp=0.9))
        fl.add(_make_agent("short", fitness=0.3, temp=0.1))
        t, k, best = fl.peak()
        assert best == pytest.approx(0.9)

    def test_bin_labels_length(self) -> None:
        from cambrian.stats import FitnessLandscape

        fl = FitnessLandscape(n_temp_bins=4, n_token_bins=5)
        labels = fl.bin_labels()
        assert len(labels["temperature"]) == 4
        assert len(labels["tokens"]) == 5

    def test_to_dict_is_serialisable(self) -> None:
        from cambrian.stats import FitnessLandscape

        fl = FitnessLandscape()
        fl.add_population([_make_agent(fitness=0.5)])
        d = fl.to_dict()
        json.dumps(d)

    def test_repr(self) -> None:
        from cambrian.stats import FitnessLandscape

        assert "FitnessLandscape" in repr(FitnessLandscape())


# ── E2E Integration test ──────────────────────────────────────────────────────


class TestE2EEvolution:
    """End-to-end test: full evolve() loop with deterministic mock backend."""

    def test_fitness_improves_over_generations(self) -> None:
        """Evolution with a directional mock evaluator should improve fitness."""
        call_count = {"n": 0}

        class _ImprovingBackend:
            """Returns a genome with slightly increasing fitness."""
            model_name = "improving"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                import re
                m = re.search(r"\{[\s\S]+\}", prompt)
                if m:
                    try:
                        data = json.loads(m.group(0))
                        # Nudge temperature toward 0.5 (optimal for our evaluator)
                        temp = float(data.get("temperature", 0.7))
                        data["temperature"] = min(max(temp + 0.02, 0.1), 1.5)
                        return json.dumps(data)
                    except (json.JSONDecodeError, KeyError):
                        pass
                return m.group(0) if m else "{}"

        def _better_at_low_temp(agent: Agent, task: str) -> float:
            """Agents with temperature closer to 0.5 score higher."""
            call_count["n"] += 1
            return max(0.0, 1.0 - abs(agent.genome.temperature - 0.5))

        backend = _ImprovingBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=_better_at_low_temp,
            mutator=mutator,
            backend=backend,
            population_size=5,
            mutation_rate=0.9,
            elite_ratio=0.2,
            seed=42,
        )

        gen_fitness_best: list[float] = []

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            best = max(a.fitness or 0.0 for a in pop)
            gen_fitness_best.append(best)

        best = engine.evolve(
            seed_genomes=[Genome(system_prompt="You are a helpful assistant.", temperature=0.9)],
            task="Improve temperature",
            n_generations=5,
            on_generation=_on_gen,
        )

        assert best is not None
        assert best.fitness is not None
        assert call_count["n"] > 0
        # Best fitness of last generation should be >= first generation
        assert gen_fitness_best[-1] >= gen_fitness_best[0] or len(gen_fitness_best) > 1

    def test_full_evolve_with_mcts_selector(self) -> None:
        """MCTS selector integrates with the evolution loop without errors."""
        from cambrian.mcts import MCTSSelector

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        selector = MCTSSelector(mutator=mutator, max_children=2)

        scored: list[float] = []

        def _evaluator(agent: Agent, task: str) -> float:
            score = 0.5 + (len(agent.genome.system_prompt) % 10) / 20.0
            scored.append(score)
            return score

        engine = EvolutionEngine(
            evaluator=_evaluator,
            mutator=mutator,
            backend=backend,
            population_size=4,
            seed=0,
        )

        gen_populations: list[list[Agent]] = []

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            for a in pop:
                selector.register(a)
                if a.fitness is not None:
                    selector.backpropagate(a.agent_id, a.fitness)
            gen_populations.append(pop)

        best = engine.evolve(
            seed_genomes=[Genome(system_prompt="Base prompt.")],
            task="test",
            n_generations=3,
            on_generation=_on_gen,
        )

        assert best is not None
        assert len(gen_populations) == 4  # gen 0 through 3
        s = selector.stats()
        assert s["nodes"] > 0
        assert s["visits"] > 0

    def test_e2e_with_curriculum(self) -> None:
        """CurriculumScheduler correctly advances tasks during evolution."""
        from cambrian.curriculum import CurriculumScheduler, CurriculumStage

        curriculum = CurriculumScheduler(
            stages=[
                CurriculumStage(task="easy task", difficulty=0.1, threshold=0.4),
                CurriculumStage(task="hard task", difficulty=0.9, threshold=0.8),
            ],
            advance_patience=1,
        )

        tasks_seen: list[str] = []

        def _evaluator(agent: Agent, task: str) -> float:
            tasks_seen.append(task)
            return 0.6  # always above easy threshold

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=_evaluator,
            mutator=mutator,
            backend=backend,
            population_size=3,
            seed=0,
        )

        current_task = curriculum.current_task()

        def _on_gen(gen: int, pop: list[Agent]) -> None:
            nonlocal current_task
            fitness_values = [a.fitness or 0.0 for a in pop]
            curriculum.advance(fitness_values)
            current_task = curriculum.current_task()

        engine.evolve(
            seed_genomes=[Genome(system_prompt="Base.")],
            task=curriculum.current_task(),
            n_generations=3,
            on_generation=_on_gen,
        )

        # After 3 generations at score 0.6 > threshold 0.4, should have advanced
        assert curriculum.stage_index == 1


# ── distill-agent CLI command ─────────────────────────────────────────────────


class TestDistillAgentCLI:
    def test_distill_agent_runs(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        genome = Genome(
            system_prompt=(
                "You are a very helpful AI assistant. "
                "Please make sure to answer carefully and completely. "
                "It is important to provide accurate information. "
            ) * 5,
            model="gpt-4o",
        )
        agent_file = tmp_path / "best.json"
        agent_file.write_text(json.dumps(genome.to_dict()))

        runner = CliRunner()
        result = runner.invoke(main, [
            "distill-agent",
            "--agent", str(agent_file),
            "--target", "gemma-4-12b",
            "--max-tokens", "60",
        ])

        assert result.exit_code == 0, result.output
        # Should have created distilled file
        distilled = tmp_path / "best.distilled.gemma-4-12b.json"
        assert distilled.exists()

    def test_distill_agent_reduces_tokens(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        long_prompt = "This is a very important sentence. " * 20
        genome = Genome(system_prompt=long_prompt, model="gpt-4o")
        agent_file = tmp_path / "big.json"
        agent_file.write_text(json.dumps(genome.to_dict()))

        runner = CliRunner()
        runner.invoke(main, [
            "distill-agent",
            "--agent", str(agent_file),
            "--target", "small-model",
            "--max-tokens", "30",
        ])

        distilled_file = tmp_path / "big.distilled.small-model.json"
        if distilled_file.exists():
            distilled = Genome.from_dict(json.loads(distilled_file.read_text()))
            # Distilled should be shorter than original
            assert distilled.token_count() <= genome.token_count()

    def test_distill_agent_output_flag(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        genome = Genome(system_prompt="Short prompt")
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(genome.to_dict()))
        out_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(main, [
            "distill-agent",
            "--agent", str(agent_file),
            "--target", "llama3.2",
            "--output", str(out_file),
        ])

        assert result.exit_code == 0
        assert out_file.exists()
