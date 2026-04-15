"""Round 5 test suite.

Covers:
- Stigmergy (StigmergyTrace, add_trace, get_traces, mutator integration)
- EpigeneticLayer (express, apply, rules, make_standard_layer)
- ImmuneMemory (register, is_suppressed, recall_score, eviction, suppression_rate)
- BaldwinEvaluator (multi-trial, bonus, improvement_stats, edge cases)
- GeminiBackend (structure, repr, SDK mock)
- Dashboard (module importable, _build_app callable structure)
- CLI dashboard command
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend


# ── Shared fixtures ───────────────────────────────────────────────────────────


class _EchoBackend(LLMBackend):
    """Returns the prompt verbatim — useful for deterministic tests."""

    @property
    def model_name(self) -> str:
        return "echo"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return prompt[:50]


def _make_agent(
    system_prompt: str = "You are helpful.",
    strategy: str = "step-by-step",
    temperature: float = 0.7,
    fitness: float | None = None,
) -> Agent:
    genome = Genome(system_prompt=system_prompt, strategy=strategy, temperature=temperature)
    agent = Agent(genome=genome, backend=_EchoBackend())
    if fitness is not None:
        agent.fitness = fitness
    return agent


# ── Stigmergy ─────────────────────────────────────────────────────────────────


class TestStigmergyTrace:
    def test_dataclass_fields(self) -> None:
        from cambrian.memory import StigmergyTrace

        t = StigmergyTrace(agent_id="abc", content="Use concise prompts.", score=0.85)
        assert t.agent_id == "abc"
        assert t.score == 0.85
        assert t.task == ""  # default

    def test_with_task(self) -> None:
        from cambrian.memory import StigmergyTrace

        t = StigmergyTrace(agent_id="x", content="Be direct.", score=0.9, task="coding")
        assert t.task == "coding"


class TestEvolutionaryMemoryTraces:
    def _make_memory(self) -> Any:
        from cambrian.memory import EvolutionaryMemory
        return EvolutionaryMemory(name="test")

    def test_add_trace_increases_count(self) -> None:
        mem = self._make_memory()
        assert len(mem._traces) == 0
        mem.add_trace("a1", "Concise responses work best.", 0.75)
        assert len(mem._traces) == 1

    def test_traces_sorted_by_score_descending(self) -> None:
        mem = self._make_memory()
        mem.add_trace("a1", "low score content", 0.3)
        mem.add_trace("a2", "high score content", 0.9)
        mem.add_trace("a3", "mid score content", 0.6)
        scores = [t.score for t in mem._traces]
        assert scores == sorted(scores, reverse=True)

    def test_get_traces_limit(self) -> None:
        mem = self._make_memory()
        for i in range(20):
            mem.add_trace(f"a{i}", f"content {i}", float(i) / 20)
        traces = mem.get_traces(limit=5)
        assert len(traces) == 5

    def test_get_traces_task_specific_first(self) -> None:
        mem = self._make_memory()
        mem.add_trace("a1", "general hint", 0.5, task="")
        mem.add_trace("a2", "coding hint", 0.4, task="python coding")
        mem.add_trace("a3", "another coding hint", 0.3, task="coding task")
        # Task-specific traces should appear before general ones
        traces = mem.get_traces(task="coding", limit=5)
        task_specific = [t for t in traces if "coding" in t.task]
        assert task_specific[0] == traces[0]  # task-specific first

    def test_get_traces_no_task_returns_top(self) -> None:
        mem = self._make_memory()
        mem.add_trace("a1", "c1", 0.9)
        mem.add_trace("a2", "c2", 0.1)
        traces = mem.get_traces(limit=1)
        assert traces[0].score == 0.9

    def test_add_trace_empty_memory_returns_empty(self) -> None:
        mem = self._make_memory()
        traces = mem.get_traces()
        assert traces == []


class TestMutatorStigmergy:
    def test_mutator_accepts_memory_param(self) -> None:
        from cambrian.mutator import LLMMutator
        from cambrian.memory import EvolutionaryMemory

        backend = _EchoBackend()
        mem = EvolutionaryMemory()
        mutator = LLMMutator(backend=backend, memory=mem, stigmergy_traces=3)
        assert mutator._memory is mem
        assert mutator._stigmergy_traces == 3

    def test_mutator_without_memory_uses_no_traces(self) -> None:
        from cambrian.mutator import LLMMutator

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        agent = _make_agent(fitness=0.5)
        result = mutator.mutate(agent, task="test task")
        assert isinstance(result, Agent)

    def test_mutator_with_traces_injects_into_prompt(self) -> None:
        """Verify traces appear in the mutation prompt via prompt capture."""
        from cambrian.mutator import LLMMutator
        from cambrian.memory import EvolutionaryMemory

        captured_prompts: list[str] = []

        class _CapturingBackend(LLMBackend):
            @property
            def model_name(self) -> str:
                return "capture"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                captured_prompts.append(prompt)
                return json.dumps({
                    "system_prompt": "improved",
                    "strategy": "step-by-step",
                    "temperature": 0.7,
                    "model": "gpt-4o-mini",
                })

        mem = EvolutionaryMemory()
        mem.add_trace("a1", "Always be concise and precise.", 0.92)

        mutator = LLMMutator(
            backend=_CapturingBackend(),
            memory=mem,
            stigmergy_traces=3,
            fallback_on_error=False,
        )
        agent = _make_agent(fitness=0.5)
        mutator.mutate(agent, task="solve task")

        assert captured_prompts, "No prompt was captured"
        assert "Stigmergy" in captured_prompts[0] or "stigmergy" in captured_prompts[0].lower()
        assert "Always be concise" in captured_prompts[0]


# ── Epigenetics ───────────────────────────────────────────────────────────────


class TestEpigenomicContext:
    def test_progress_zero_to_one(self) -> None:
        from cambrian.epigenetics import EpigenomicContext

        ctx = EpigenomicContext(generation=0, total_generations=10)
        assert ctx.progress == 0.0

        ctx2 = EpigenomicContext(generation=9, total_generations=10)
        assert ctx2.progress == 1.0

    def test_is_early_late(self) -> None:
        from cambrian.epigenetics import EpigenomicContext

        early = EpigenomicContext(generation=1, total_generations=10)
        assert early.is_early
        assert not early.is_late

        late = EpigenomicContext(generation=9, total_generations=10)
        assert not late.is_early
        assert late.is_late

    def test_single_generation(self) -> None:
        from cambrian.epigenetics import EpigenomicContext

        ctx = EpigenomicContext(generation=0, total_generations=1)
        assert ctx.progress == 1.0


class TestEpigeneticLayer:
    def test_no_rules_returns_original_prompt(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        layer = EpigeneticLayer()
        genome = Genome(system_prompt="Base prompt.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert result == "Base prompt."

    def test_rule_appends_annotation(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        layer = EpigeneticLayer(rules=[lambda g, ctx: "Phase: explore"])
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert "Base." in result
        assert "Phase: explore" in result

    def test_none_returning_rule_ignored(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        layer = EpigeneticLayer(rules=[lambda g, ctx: None])
        genome = Genome(system_prompt="Base.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert result == "Base."

    def test_rule_exception_does_not_crash(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        def _bad_rule(g: Genome, ctx: EpigenomicContext) -> str:
            raise RuntimeError("bad rule")

        layer = EpigeneticLayer(rules=[_bad_rule, lambda g, ctx: "OK"])
        genome = Genome(system_prompt="Safe.")
        ctx = EpigenomicContext()
        result = layer.express(genome, ctx)
        assert "Safe." in result
        assert "OK" in result

    def test_apply_returns_clone_not_original(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        layer = EpigeneticLayer(rules=[lambda g, ctx: "annotation"])
        agent = _make_agent()
        ctx = EpigenomicContext()
        expressed = layer.apply(agent, ctx)
        # Original unchanged
        assert agent.genome.system_prompt == "You are helpful."
        # Clone has annotation
        assert "annotation" in expressed.genome.system_prompt

    def test_apply_no_op_returns_same_agent(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer, EpigenomicContext

        layer = EpigeneticLayer(rules=[lambda g, ctx: None])
        agent = _make_agent()
        ctx = EpigenomicContext()
        result = layer.apply(agent, ctx)
        assert result is agent  # no clone needed

    def test_add_rule(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer

        layer = EpigeneticLayer()
        layer.add_rule(lambda g, ctx: "added rule")
        assert len(layer._rules) == 1

    def test_repr(self) -> None:
        from cambrian.epigenetics import EpigeneticLayer

        layer = EpigeneticLayer(rules=[lambda g, ctx: "x", lambda g, ctx: "y"])
        assert "2" in repr(layer)


class TestMakeStandardLayer:
    def test_has_four_rules(self) -> None:
        from cambrian.epigenetics import make_standard_layer

        layer = make_standard_layer()
        assert len(layer._rules) == 4

    def test_early_generation_signal(self) -> None:
        from cambrian.epigenetics import make_standard_layer, EpigenomicContext

        layer = make_standard_layer()
        genome = Genome()
        ctx = EpigenomicContext(generation=0, total_generations=10)
        result = layer.express(genome, ctx)
        assert "early" in result.lower() or "explor" in result.lower()

    def test_coding_task_mode(self) -> None:
        from cambrian.epigenetics import make_standard_layer, EpigenomicContext

        layer = make_standard_layer()
        genome = Genome()
        ctx = EpigenomicContext(generation=5, task="implement binary search in python")
        result = layer.express(genome, ctx)
        assert "CODING" in result or "coding" in result.lower()

    def test_diversity_collapse_warning(self) -> None:
        from cambrian.epigenetics import make_standard_layer, EpigenomicContext

        layer = make_standard_layer()
        genome = Genome()
        ctx = EpigenomicContext(
            generation=5, extra={"strategy_entropy": 0.1}
        )
        result = layer.express(genome, ctx)
        assert "diversity" in result.lower() or "novel" in result.lower()


# ── Immune System ─────────────────────────────────────────────────────────────


class TestFingerprint:
    def test_same_agent_same_fingerprint(self) -> None:
        from cambrian.immune import fingerprint

        a1 = _make_agent("Hello world.", "step-by-step", 0.7)
        a2 = _make_agent("Hello world.", "step-by-step", 0.7)
        assert fingerprint(a1) == fingerprint(a2)

    def test_different_prompt_different_fingerprint(self) -> None:
        from cambrian.immune import fingerprint

        a1 = _make_agent("Be concise.", "step-by-step", 0.7)
        a2 = _make_agent("Be verbose.", "step-by-step", 0.7)
        assert fingerprint(a1) != fingerprint(a2)

    def test_fingerprint_is_16_hex_chars(self) -> None:
        from cambrian.immune import fingerprint

        fp = fingerprint(_make_agent())
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_whitespace_normalised(self) -> None:
        from cambrian.immune import fingerprint

        a1 = _make_agent("Be  concise.")
        a2 = _make_agent("Be concise.")
        assert fingerprint(a1) == fingerprint(a2)


class TestImmuneMemory:
    def test_register_no_fitness_ignored(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory()
        agent = _make_agent()  # no fitness
        mem.register(agent)
        assert mem.memory_size == 0

    def test_register_with_fitness(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory()
        agent = _make_agent(fitness=0.5)
        mem.register(agent)
        assert mem.memory_size == 1

    def test_is_suppressed_below_threshold(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(suppression_threshold=0.3, min_evals_before_suppress=2)
        agent = _make_agent("same prompt", fitness=0.1)
        mem.register(agent)
        mem.register(agent)  # second eval to meet min_evals
        assert mem.is_suppressed(agent)

    def test_is_not_suppressed_above_threshold(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(suppression_threshold=0.3, min_evals_before_suppress=1)
        agent = _make_agent(fitness=0.8)
        mem.register(agent)
        assert not mem.is_suppressed(agent)

    def test_is_not_suppressed_before_min_evals(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(suppression_threshold=0.5, min_evals_before_suppress=3)
        agent = _make_agent(fitness=0.1)
        mem.register(agent)  # only 1 eval
        assert not mem.is_suppressed(agent)

    def test_recall_score_returns_best(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory()
        agent = _make_agent(fitness=0.4)
        mem.register(agent)
        agent.fitness = 0.7
        mem.register(agent)
        score = mem.recall_score(agent)
        assert score == pytest.approx(0.7)

    def test_recall_score_unknown_returns_none(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory()
        agent = _make_agent()
        assert mem.recall_score(agent) is None

    def test_eviction_keeps_max_memory(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(max_memory=3, min_evals_before_suppress=1)
        for i in range(5):
            a = _make_agent(f"prompt {i}", fitness=float(i) / 5)
            mem.register(a)
        assert mem.memory_size == 3

    def test_suppression_rate(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(suppression_threshold=0.5, min_evals_before_suppress=1)
        agents = [_make_agent(f"p{i}", fitness=0.1) for i in range(4)]
        for a in agents:
            mem.register(a)
        rate = mem.suppression_rate(agents)
        assert rate == pytest.approx(1.0)

    def test_to_dict_is_serialisable(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory()
        mem.register(_make_agent(fitness=0.5))
        d = mem.to_dict()
        assert json.dumps(d)  # must not raise

    def test_repr(self) -> None:
        from cambrian.immune import ImmuneMemory

        mem = ImmuneMemory(suppression_threshold=0.4)
        assert "0.4" in repr(mem)

    def test_invalid_threshold_raises(self) -> None:
        from cambrian.immune import ImmuneMemory

        with pytest.raises(ValueError):
            ImmuneMemory(suppression_threshold=1.5)


# ── Baldwin Effect ────────────────────────────────────────────────────────────


class TestBaldwinEvaluator:
    def _make_counter_evaluator(self, scores: list[float]) -> Any:
        """Returns scores in sequence."""
        idx = [0]

        def _ev(agent: Agent, task: str) -> float:
            score = scores[idx[0] % len(scores)]
            idx[0] += 1
            return score

        return _ev

    def test_requires_n_trials_ge_2(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        with pytest.raises(ValueError):
            BaldwinEvaluator(base_evaluator=lambda a, t: 0.5, n_trials=1)

    def test_invalid_bonus_raises(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        with pytest.raises(ValueError):
            BaldwinEvaluator(base_evaluator=lambda a, t: 0.5, baldwin_bonus=1.5)

    def test_invalid_aggregate_raises(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        with pytest.raises(ValueError):
            BaldwinEvaluator(base_evaluator=lambda a, t: 0.5, aggregate_base="max")

    def test_no_improvement_returns_base_score(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.5, 0.5, 0.5]),
            n_trials=3,
            baldwin_bonus=0.2,
        )
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.5)

    def test_improvement_adds_bonus(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        # Trial 1: 0.5, trial 3: 0.7 → improvement = 0.2 → bonus = 0.2 * 0.2 = 0.04
        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.5, 0.6, 0.7]),
            n_trials=3,
            baldwin_bonus=0.2,
        )
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        expected = min(1.0, 0.5 + 0.2 * (0.7 - 0.5))
        assert score == pytest.approx(expected, abs=1e-6)

    def test_clamped_to_one(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.9, 1.0]),
            n_trials=2,
            baldwin_bonus=1.0,
        )
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        assert score <= 1.0

    def test_improvement_stats_structure(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.4, 0.6, 0.8]),
            n_trials=3,
        )
        agent = _make_agent()
        stats = ev.improvement_stats(agent, "task")
        assert "scores" in stats
        assert "base" in stats
        assert "improvement" in stats
        assert "final_fitness" in stats
        assert "learnable" in stats
        assert len(stats["scores"]) == 3  # type: ignore[arg-type]

    def test_aggregate_last(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.3, 0.8, 0.9]),
            n_trials=3,
            aggregate_base="last",
            baldwin_bonus=0.0,
        )
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.9)

    def test_aggregate_best(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(
            base_evaluator=self._make_counter_evaluator([0.3, 0.9, 0.7]),
            n_trials=3,
            aggregate_base="best",
            baldwin_bonus=0.0,
        )
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.9)

    def test_evaluator_exception_scores_zero(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        call_count = [0]

        def _raising(agent: Agent, task: str) -> float:
            call_count[0] += 1
            raise RuntimeError("test error")

        ev = BaldwinEvaluator(base_evaluator=_raising, n_trials=2)
        agent = _make_agent()
        score = ev.evaluate(agent, "task")
        assert score == pytest.approx(0.0)

    def test_repr(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        ev = BaldwinEvaluator(base_evaluator=lambda a, t: 0.5, n_trials=3, baldwin_bonus=0.1)
        assert "n_trials=3" in repr(ev)
        assert "0.1" in repr(ev)

    def test_feedback_template_used(self) -> None:
        """The feedback task should contain the template-expanded content."""
        from cambrian.evaluators.baldwin import BaldwinEvaluator

        tasks_seen: list[str] = []

        def _recording_ev(agent: Agent, task: str) -> float:
            tasks_seen.append(task)
            return 0.5

        ev = BaldwinEvaluator(
            base_evaluator=_recording_ev,
            n_trials=3,
            feedback_template="SCORE:{score:.2f} TASK:{task}",
        )
        agent = _make_agent()
        ev.evaluate(agent, "original task")
        assert tasks_seen[0] == "original task"
        assert "SCORE:0.50" in tasks_seen[1]
        assert "original task" in tasks_seen[1]


# ── Gemini Backend ────────────────────────────────────────────────────────────


class TestGeminiBackend:
    def test_default_model(self) -> None:
        from cambrian.backends.gemini import GeminiBackend

        # Patching google.genai import to avoid ImportError
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            backend = GeminiBackend()
            assert backend.model_name == "gemini-2.0-flash"

    def test_custom_model(self) -> None:
        from cambrian.backends.gemini import GeminiBackend

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            backend = GeminiBackend(model="gemini-1.5-pro")
            assert backend.model_name == "gemini-1.5-pro"

    def test_repr(self) -> None:
        from cambrian.backends.gemini import GeminiBackend

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            backend = GeminiBackend(model="gemini-2.0-flash")
            assert "gemini-2.0-flash" in repr(backend)

    def test_import_error_if_sdk_missing(self) -> None:
        from cambrian.backends.gemini import GeminiBackend

        # Setting a module to None in sys.modules raises ImportError on import
        with patch.dict("sys.modules", {"google.genai": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="google-genai"):
                GeminiBackend()

    def test_generate_calls_sdk(self) -> None:
        from cambrian.backends.gemini import GeminiBackend

        # Build a mock response with .text set to a plain string
        mock_response = MagicMock()
        mock_response.text = "Gemini says hello."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        # Create mock_genai and make sure mock_google.genai == mock_genai
        # so both attribute-chain and sys.modules paths resolve identically
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()
        mock_genai.types = mock_types

        mock_google = MagicMock()
        mock_google.genai = mock_genai  # attribute chain path

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            backend = GeminiBackend(api_key="test-key")
            result = backend.generate("Write hello.")

        assert result == "Gemini says hello."


# ── Dashboard module ──────────────────────────────────────────────────────────


class TestDashboardModule:
    def test_module_importable(self) -> None:
        import cambrian.dashboard as dash
        assert callable(dash.run_dashboard)
        assert callable(dash._build_app)

    def test_run_dashboard_raises_import_error_without_streamlit(self) -> None:
        import sys
        import cambrian.dashboard as dash

        original = sys.modules.pop("streamlit", None)
        try:
            with pytest.raises(ImportError, match="streamlit"):
                dash.run_dashboard(log_file="nonexistent.json")
        finally:
            if original is not None:
                sys.modules["streamlit"] = original

    def test_build_app_with_nonexistent_file(self) -> None:
        """_build_app should call st.info (not error) when Evolve log not found."""
        mock_st = MagicMock()
        mock_st.sidebar.slider.return_value = 5
        # st.tabs() must return exactly 2 context-manager mock objects
        tab1, tab2 = MagicMock(), MagicMock()
        tab1.__enter__ = MagicMock(return_value=tab1)
        tab1.__exit__ = MagicMock(return_value=False)
        tab2.__enter__ = MagicMock(return_value=tab2)
        tab2.__exit__ = MagicMock(return_value=False)
        mock_st.tabs.return_value = [tab1, tab2]

        with patch.dict("sys.modules", {"streamlit": mock_st}):
            import importlib
            import cambrian.dashboard as dash
            importlib.reload(dash)
            dash._build_app("/nonexistent/path/to/log.json")

        # Dashboard renders page title successfully
        mock_st.title.assert_called_once()
        # Evolve tab displayed an info/warning (file not found)
        # In Streamlit, calls inside `with tab1:` go to the module-level st object
        assert mock_st.info.called or mock_st.warning.called

    def test_build_app_with_valid_log(self) -> None:
        """_build_app should render without error for valid log data."""
        log_data = [
            {
                "generation": 0,
                "agents": [
                    {"id": "a1", "fitness": 0.6, "genome": {
                        "system_prompt": "Hello.",
                        "model": "gpt-4o-mini",
                        "temperature": 0.7,
                        "strategy": "step-by-step",
                    }},
                ],
            }
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(log_data, f)
            tmp_path = f.name

        mock_st = MagicMock()
        mock_st.sidebar.slider.return_value = 5
        col_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.columns.return_value = col_mocks
        # st.tabs() must return exactly 2 context-manager mock objects
        tab1, tab2 = MagicMock(), MagicMock()
        tab1.__enter__ = MagicMock(return_value=tab1)
        tab1.__exit__ = MagicMock(return_value=False)
        tab2.__enter__ = MagicMock(return_value=tab2)
        tab2.__exit__ = MagicMock(return_value=False)
        mock_st.tabs.return_value = [tab1, tab2]

        try:
            with patch.dict("sys.modules", {"streamlit": mock_st}):
                import importlib
                import cambrian.dashboard as dash
                importlib.reload(dash)
                dash._build_app(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Page title rendered
        mock_st.title.assert_called_once()
        # st.tabs was called with 2 tab labels
        mock_st.tabs.assert_called_once()
        assert len(mock_st.tabs.call_args[0][0]) == 2


# ── CLI dashboard command ─────────────────────────────────────────────────────


class TestCLIDashboardCommand:
    def test_dashboard_command_exists(self) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_stats_command_exists(self) -> None:
        """Old 'dashboard' renamed to 'stats' should still work."""
        from click.testing import CliRunner
        from cambrian.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0

    def test_dashboard_import_error_graceful(self) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main
        import sys

        original = sys.modules.pop("streamlit", None)
        try:
            runner = CliRunner()
            # dashboard with missing log file but streamlit missing — gets ClickException
            result = runner.invoke(main, ["dashboard", "--no-browser", "--log-file", "no.json"])
            # Either ImportError (no streamlit) or other — should not crash with traceback
            assert result.exit_code != 0 or "no.json" in result.output
        finally:
            if original is not None:
                sys.modules["streamlit"] = original


# ── mypy: all new modules pass strict ─────────────────────────────────────────


class TestMypyStrictRound5:
    """Smoke tests that all new modules import cleanly (mypy is run in CI)."""

    def test_stigmergy_imports(self) -> None:
        from cambrian.memory import StigmergyTrace, EvolutionaryMemory
        assert StigmergyTrace
        assert EvolutionaryMemory

    def test_epigenetics_imports(self) -> None:
        from cambrian.epigenetics import (
            EpigeneticLayer,
            EpigenomicContext,
            make_standard_layer,
        )
        assert EpigeneticLayer
        assert EpigenomicContext
        assert make_standard_layer

    def test_immune_imports(self) -> None:
        from cambrian.immune import ImmuneMemory, fingerprint, ImmuneCellRecord
        assert ImmuneMemory
        assert fingerprint
        assert ImmuneCellRecord

    def test_baldwin_imports(self) -> None:
        from cambrian.evaluators.baldwin import BaldwinEvaluator
        assert BaldwinEvaluator

    def test_gemini_imports_without_sdk(self) -> None:
        # Should be importable at module level (lazy SDK import in generate())
        import cambrian.backends.gemini as gm
        assert gm.GeminiBackend
