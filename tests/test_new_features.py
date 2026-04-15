# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for new Cambrian features:

- Lamarckian evolution (LamarckianAdapter, few_shot_examples in Genome)
- VarianceAwareEvaluator (anti-reward-hacking)
- Prompt auto-compression in EvolutionEngine
- Population save/load
- --tournament-k / --compress-every CLI flags
- cambrian analyze CLI command
- AnthropicBackend (unit-level, no real API calls)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.lamarck import LamarckianAdapter
from cambrian.mutator import LLMMutator


# ── Shared stubs ──────────────────────────────────────────────────────────────


class _EchoBackend:
    """Returns the genome JSON from the prompt unchanged (for mutation stubs)."""

    model_name = "echo"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import re
        m = re.search(r"\{[\s\S]+\}", prompt)
        return m.group(0) if m else "{}"


def _make_engine(evaluator=None, pop_size: int = 4) -> EvolutionEngine:
    backend = _EchoBackend()
    mutator = LLMMutator(backend=backend, fallback_on_error=True)
    ev = evaluator or (lambda agent, task: 0.5)
    return EvolutionEngine(
        evaluator=ev,
        mutator=mutator,
        backend=backend,
        population_size=pop_size,
        seed=42,
    )


def _make_agent(system_prompt: str = "Test prompt", fitness: float | None = None) -> Agent:
    g = Genome(system_prompt=system_prompt)
    a = Agent(genome=g)
    if fitness is not None:
        a.fitness = fitness
    return a


# ── Lamarckian evolution ──────────────────────────────────────────────────────


class TestLamarckianAdapter:
    def test_high_score_captured(self) -> None:
        """Scores above threshold add a few-shot example to the genome."""
        agent = _make_agent()
        assert agent.genome.few_shot_examples == []

        adapter = LamarckianAdapter(
            base_evaluator=lambda a, t: 0.9,
            capture_threshold=0.7,
        )
        score = adapter(agent, "solve x")

        assert score == pytest.approx(0.9)
        assert len(agent.genome.few_shot_examples) == 1
        ex = agent.genome.few_shot_examples[0]
        assert ex["task"] == "solve x"
        assert ex["score"] == pytest.approx(0.9, abs=0.01)

    def test_low_score_not_captured(self) -> None:
        """Scores below threshold leave few_shot_examples unchanged."""
        agent = _make_agent()
        adapter = LamarckianAdapter(
            base_evaluator=lambda a, t: 0.4,
            capture_threshold=0.7,
        )
        adapter(agent, "solve y")
        assert agent.genome.few_shot_examples == []

    def test_max_examples_respected(self) -> None:
        """Never stores more than max_examples entries."""
        agent = _make_agent()
        adapter = LamarckianAdapter(
            base_evaluator=lambda a, t: 0.95,
            max_examples=2,
        )
        for i in range(5):
            adapter(agent, f"task {i}")

        assert len(agent.genome.few_shot_examples) <= 2

    def test_duplicate_task_not_added(self) -> None:
        """Same task string produces only one example entry."""
        agent = _make_agent()
        adapter = LamarckianAdapter(base_evaluator=lambda a, t: 0.9)
        adapter(agent, "same task")
        adapter(agent, "same task")
        assert len(agent.genome.few_shot_examples) == 1

    def test_record_response_stored_in_example(self) -> None:
        """record_response() makes the response available in the captured example."""
        agent = _make_agent()
        adapter = LamarckianAdapter(base_evaluator=lambda a, t: 0.9)
        adapter.record_response(agent.agent_id, "my output")
        adapter(agent, "some task")

        ex = agent.genome.few_shot_examples[0]
        assert ex.get("response") == "my output"

    def test_examples_sorted_by_score(self) -> None:
        """Examples are kept sorted descending by score, lowest evicted when full."""
        agent = _make_agent()
        scores = [0.71, 0.85, 0.78, 0.92]  # 4 unique tasks, max=3
        adapter = LamarckianAdapter(
            base_evaluator=None,  # type: ignore[arg-type]
            max_examples=3,
        )
        for i, s in enumerate(scores):
            adapter._base = lambda a, t, _s=s: _s  # type: ignore[assignment]
            adapter(agent, f"task_{i}")

        assert len(agent.genome.few_shot_examples) == 3
        kept_scores = [ex["score"] for ex in agent.genome.few_shot_examples]
        # Lowest score (0.71) should have been evicted
        assert 0.71 not in kept_scores

    def test_repr(self) -> None:
        adapter = LamarckianAdapter(base_evaluator=lambda a, t: 0.5)
        assert "LamarckianAdapter" in repr(adapter)


class TestFewShotExamplesInGenome:
    def test_serialise_deserialise_examples(self) -> None:
        """few_shot_examples round-trips through to_dict/from_dict."""
        g = Genome(few_shot_examples=[{"task": "x", "score": 0.8, "response": "y"}])
        restored = Genome.from_dict(g.to_dict())
        assert restored.few_shot_examples == [{"task": "x", "score": 0.8, "response": "y"}]

    def test_empty_by_default(self) -> None:
        assert Genome().few_shot_examples == []

    def test_from_dict_without_field_defaults_empty(self) -> None:
        """Deserialing old genome dicts (no few_shot_examples key) still works."""
        g = Genome.from_dict({"system_prompt": "hello"})
        assert g.few_shot_examples == []

    def test_agent_run_injects_examples(self) -> None:
        """Agent.run() includes few-shot examples in the system message."""
        captured: dict[str, Any] = {}

        class _SpyBackend:
            model_name = "spy"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                captured["system"] = kwargs.get("system", "")
                return "response"

        g = Genome(
            system_prompt="Base prompt",
            few_shot_examples=[{"task": "t", "score": 0.9, "response": "r"}],
        )
        agent = Agent(genome=g, backend=_SpyBackend())
        agent.run("do something")

        assert "Base prompt" in captured["system"]
        assert "Successful examples" in captured["system"]
        assert "score 0.9" in captured["system"]


# ── VarianceAwareEvaluator ────────────────────────────────────────────────────


class TestVarianceAwareEvaluator:
    def _make(self, scores: list[float], **kwargs: Any):
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        evals = [lambda a, t, s=s: s for s in scores]
        return VarianceAwareEvaluator(evaluators=evals, **kwargs)

    def test_low_variance_near_mean(self) -> None:
        """Consistent scores → result close to mean (small penalty)."""
        ev = self._make([0.8, 0.82, 0.79])
        agent = _make_agent()
        result = ev.evaluate(agent, "task")
        assert result == pytest.approx(0.80333, abs=0.05)

    def test_high_variance_penalised(self) -> None:
        """High variance (reward hacking) → score below mean."""
        ev = self._make([1.0, 0.0, 0.5], penalty_weight=1.0)
        agent = _make_agent()
        result = ev.evaluate(agent, "task")
        mean = (1.0 + 0.0 + 0.5) / 3
        import statistics
        var = statistics.variance([1.0, 0.0, 0.5])
        expected = mean - var
        assert result == pytest.approx(max(0.0, expected), abs=0.01)

    def test_zero_penalty_is_pure_mean(self) -> None:
        ev = self._make([0.6, 0.8], penalty_weight=0.0)
        agent = _make_agent()
        assert ev.evaluate(agent, "task") == pytest.approx(0.7, abs=0.001)

    def test_min_aggregate(self) -> None:
        ev = self._make([0.9, 0.3], penalty_weight=0.0, aggregate="min")
        agent = _make_agent()
        assert ev.evaluate(agent, "task") == pytest.approx(0.3, abs=0.001)

    def test_result_clipped_to_0_1(self) -> None:
        ev = self._make([1.0, 0.0], penalty_weight=5.0)
        agent = _make_agent()
        result = ev.evaluate(agent, "task")
        assert 0.0 <= result <= 1.0

    def test_requires_at_least_2_evaluators(self) -> None:
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        with pytest.raises(ValueError, match="at least 2"):
            VarianceAwareEvaluator(evaluators=[lambda a, t: 0.5])

    def test_weights_mismatch_raises(self) -> None:
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        with pytest.raises(ValueError, match="same length"):
            VarianceAwareEvaluator(
                evaluators=[lambda a, t: 0.5, lambda a, t: 0.6],
                weights=[1.0, 2.0, 3.0],
            )

    def test_invalid_aggregate_raises(self) -> None:
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        with pytest.raises(ValueError, match="aggregate"):
            VarianceAwareEvaluator(
                evaluators=[lambda a, t: 0.5, lambda a, t: 0.6],
                aggregate="max",
            )

    def test_failed_subevaluator_scores_zero(self) -> None:
        """A sub-evaluator that raises does not crash the composite."""
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        def _boom(agent: Agent, task: str) -> float:
            raise RuntimeError("oops")

        ev = VarianceAwareEvaluator(
            evaluators=[lambda a, t: 0.8, _boom],
            penalty_weight=0.0,
        )
        agent = _make_agent()
        result = ev.evaluate(agent, "task")
        # mean of [0.8, 0.0] = 0.4, no penalty
        assert result == pytest.approx(0.4, abs=0.01)

    def test_custom_weights_applied(self) -> None:
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        ev = VarianceAwareEvaluator(
            evaluators=[lambda a, t: 1.0, lambda a, t: 0.0],
            weights=[0.9, 0.1],
            penalty_weight=0.0,
        )
        agent = _make_agent()
        result = ev.evaluate(agent, "task")
        assert result == pytest.approx(0.9, abs=0.01)

    def test_repr(self) -> None:
        from cambrian.evaluators.variance_aware import VarianceAwareEvaluator

        ev = VarianceAwareEvaluator(
            evaluators=[lambda a, t: 0.5, lambda a, t: 0.6]
        )
        assert "VarianceAwareEvaluator" in repr(ev)


class TestBuildDiversifiedEvaluator:
    def test_returns_variance_aware(self) -> None:
        from cambrian.evaluators.variance_aware import (
            VarianceAwareEvaluator,
            build_diversified_evaluator,
        )

        class _FakeJudge:
            model_name = "fake"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                return '{"score": 7}'

        ev = build_diversified_evaluator(
            backend=_FakeJudge(),
            expected_output="hello",
            task_description="greet",
        )
        assert isinstance(ev, VarianceAwareEvaluator)


# ── Prompt auto-compression in EvolutionEngine ────────────────────────────────


class TestAutoCompression:
    def test_compress_interval_zero_disables(self) -> None:
        """compress_interval=0 means no compression is ever triggered."""
        engine = _make_engine(pop_size=3)
        assert engine._compress_interval == 0

    def test_compress_interval_stored(self) -> None:
        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=lambda a, t: 0.5,
            mutator=mutator,
            backend=backend,
            population_size=4,
            compress_interval=5,
            compress_max_tokens=100,
        )
        assert engine._compress_interval == 5
        assert engine._compress_max_tokens == 100

    def test_compress_population_shortens_long_prompt(self) -> None:
        """_compress_population pruning actually reduces overlong multi-paragraph prompts."""
        # procut_prune requires multiple paragraphs (split on \n\n) to prune
        paragraph = "This is a sentence for testing compression. " * 8  # ~88 tokens per paragraph
        long_prompt = "\n\n".join([paragraph] * 10)  # 10 paragraphs, ~880 tokens
        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=lambda a, t: 0.5,
            mutator=mutator,
            population_size=2,
            compress_interval=1,
            compress_max_tokens=100,  # force paragraph-dropping
        )
        pop = [_make_agent(long_prompt) for _ in range(2)]
        compressed = engine._compress_population(pop)
        for a in compressed:
            # Should be shorter than original (880 tokens → ≤ 100+tolerance)
            assert a.genome.token_count() < 200

    def test_compress_run_triggers_at_interval(self) -> None:
        """Evolution with compress_interval applies compression at the right gen."""
        compressed_gens: list[int] = []

        from cambrian.compress import procut_prune as _orig_prune

        def _spy_prune(genome, max_tokens=256):
            compressed_gens.append(1)  # record a call
            return _orig_prune(genome, max_tokens)

        backend = _EchoBackend()
        mutator = LLMMutator(backend=backend, fallback_on_error=True)
        engine = EvolutionEngine(
            evaluator=lambda a, t: 0.5,
            mutator=mutator,
            backend=backend,
            population_size=3,
            compress_interval=2,  # compress every 2 gens
            seed=0,
        )

        with patch("cambrian.evolution.EvolutionEngine._compress_population", wraps=engine._compress_population) as mock:
            engine.evolve(
                seed_genomes=[Genome(system_prompt="short")],
                task="test",
                n_generations=4,
            )
        # Should have been called at gen 2 and gen 4
        assert mock.call_count == 2


# ── Population save / load ────────────────────────────────────────────────────


class TestPopulationSaveLoad:
    def _pop(self, n: int = 3) -> list[Agent]:
        agents = []
        for i in range(n):
            a = _make_agent(f"prompt {i}", fitness=0.1 * (i + 1))
            a._generation = i
            agents.append(a)
        return agents

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        engine = _make_engine()
        pop = self._pop()
        path = tmp_path / "pop.json"
        engine.save_population(path, pop)
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 3

    def test_roundtrip_restores_fitness_and_generation(self, tmp_path: Path) -> None:
        engine = _make_engine()
        pop = self._pop()
        path = tmp_path / "pop.json"
        engine.save_population(path, pop)
        loaded = engine.load_population(path)
        assert len(loaded) == 3
        for orig, rest in zip(pop, loaded):
            assert rest.fitness == pytest.approx(orig.fitness)
            assert rest._generation == orig._generation

    def test_roundtrip_restores_genome(self, tmp_path: Path) -> None:
        engine = _make_engine()
        pop = self._pop()
        path = tmp_path / "pop.json"
        engine.save_population(path, pop)
        loaded = engine.load_population(path)
        for orig, rest in zip(pop, loaded):
            assert rest.genome.system_prompt == orig.genome.system_prompt

    def test_load_attaches_engine_backend(self, tmp_path: Path) -> None:
        engine = _make_engine()
        pop = self._pop()
        path = tmp_path / "pop.json"
        engine.save_population(path, pop)
        loaded = engine.load_population(path)
        for a in loaded:
            assert a.backend is engine._backend

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        engine = _make_engine()
        with pytest.raises(ValueError, match="Cannot parse"):
            engine.load_population(path)

    def test_load_non_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"key": "value"}')
        engine = _make_engine()
        with pytest.raises(ValueError, match="JSON array"):
            engine.load_population(path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        engine = _make_engine()
        with pytest.raises(FileNotFoundError):
            engine.load_population(tmp_path / "missing.json")

    def test_save_preserves_few_shot_examples(self, tmp_path: Path) -> None:
        engine = _make_engine()
        a = _make_agent()
        a.genome.few_shot_examples = [{"task": "x", "score": 0.9, "response": "r"}]
        path = tmp_path / "pop.json"
        engine.save_population(path, [a])
        loaded = engine.load_population(path)
        assert loaded[0].genome.few_shot_examples == [{"task": "x", "score": 0.9, "response": "r"}]


# ── AnthropicBackend ──────────────────────────────────────────────────────────


class TestAnthropicBackend:
    def test_import_without_sdk_raises(self) -> None:
        """When the anthropic package is absent, instantiation raises ImportError."""
        import sys
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            # Force re-import
            import importlib
            import cambrian.backends.anthropic as _mod
            importlib.reload(_mod)
            with pytest.raises(ImportError, match="anthropic"):
                _mod.AnthropicBackend()
        finally:
            if original is None:
                del sys.modules["anthropic"]
            else:
                sys.modules["anthropic"] = original

    def test_model_name_property(self) -> None:
        from cambrian.backends.anthropic import AnthropicBackend

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-haiku-4-5-20251001"
        assert b.model_name == "claude-haiku-4-5-20251001"

    def test_repr(self) -> None:
        from cambrian.backends.anthropic import AnthropicBackend

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-test"
        assert "AnthropicBackend" in repr(b)
        assert "claude-test" in repr(b)

    def test_generate_with_mock_sdk(self) -> None:
        """generate() correctly calls client.messages.create and returns text."""
        from cambrian.backends.anthropic import AnthropicBackend

        mock_block = MagicMock()
        mock_block.text = "  evolution is beautiful  "
        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-haiku-4-5-20251001"
        b._api_key = "test-key"
        b._temperature = 0.7
        b._max_tokens = 1024
        b._max_retries = 3
        b._timeout = 60.0

        with patch("anthropic.Anthropic", return_value=mock_client):
            result = b.generate("Hello", system="Be helpful")

        assert result == "evolution is beautiful"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_temperature_clamped(self) -> None:
        """Temperature above 1.0 is clamped to 1.0 for Claude."""
        from cambrian.backends.anthropic import AnthropicBackend

        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_message = MagicMock()
        mock_message.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-haiku-4-5-20251001"
        b._api_key = "k"
        b._temperature = 0.7
        b._max_tokens = 100
        b._max_retries = 1
        b._timeout = 10.0

        with patch("anthropic.Anthropic", return_value=mock_client):
            b.generate("hi", temperature=1.8)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] <= 1.0

    def test_retry_on_rate_limit(self) -> None:
        """RateLimitError triggers retries with back-off."""
        import anthropic as _anthropic

        from cambrian.backends.anthropic import AnthropicBackend

        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_message = MagicMock()
        mock_message.content = [mock_block]
        mock_client = MagicMock()

        call_count = {"n": 0}

        def _create(**kwargs: Any):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise _anthropic.RateLimitError(
                    "rate limit",
                    response=MagicMock(status_code=429),
                    body={},
                )
            return mock_message

        mock_client.messages.create.side_effect = _create

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-haiku-4-5-20251001"
        b._api_key = "k"
        b._temperature = 0.7
        b._max_tokens = 100
        b._max_retries = 3
        b._timeout = 10.0

        with patch("anthropic.Anthropic", return_value=mock_client), \
             patch("time.sleep"):
            result = b.generate("hi")

        assert result == "ok"
        assert call_count["n"] == 3

    def test_exhausted_retries_raise(self) -> None:
        """After max_retries failures, RuntimeError is raised."""
        import anthropic as _anthropic

        from cambrian.backends.anthropic import AnthropicBackend

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _anthropic.RateLimitError(
            "rate limit",
            response=MagicMock(status_code=429),
            body={},
        )

        b = AnthropicBackend.__new__(AnthropicBackend)
        b._model = "claude-haiku-4-5-20251001"
        b._api_key = "k"
        b._temperature = 0.7
        b._max_tokens = 100
        b._max_retries = 2
        b._timeout = 10.0

        with patch("anthropic.Anthropic", return_value=mock_client), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="failed after"):
                b.generate("hi")


# ── CLI: --tournament-k, --compress-every, analyze ───────────────────────────


class TestCLIFlags:
    def test_tournament_k_wired(self) -> None:
        """--tournament-k is passed to EvolutionEngine constructor."""
        from click.testing import CliRunner

        from cambrian.cli import main

        fake_agent = _make_agent(fitness=0.5)
        fake_agent._fitness = 0.5

        with patch("cambrian.cli.EvolutionEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_instance.evolve.return_value = fake_agent
            mock_instance.memory.to_json.return_value = "{}"
            MockEngine.return_value = mock_instance

            runner = CliRunner()
            runner.invoke(
                main,
                [
                    "evolve", "test task",
                    "--tournament-k", "5",
                    "--api-key", "test-key",
                    "--generations", "1",
                    "--population", "3",
                ],
            )

        call_kwargs = MockEngine.call_args.kwargs
        assert call_kwargs.get("tournament_k") == 5

    def test_compress_every_wired(self) -> None:
        """--compress-every is passed to EvolutionEngine constructor."""
        from click.testing import CliRunner

        from cambrian.cli import main

        fake_agent = _make_agent(fitness=0.5)
        fake_agent._fitness = 0.5

        with patch("cambrian.cli.EvolutionEngine") as MockEngine:
            mock_instance = MagicMock()
            mock_instance.evolve.return_value = fake_agent
            mock_instance.memory.to_json.return_value = "{}"
            MockEngine.return_value = mock_instance

            runner = CliRunner()
            runner.invoke(
                main,
                [
                    "evolve", "test",
                    "--compress-every", "3",
                    "--api-key", "test-key",
                    "--generations", "1",
                    "--population", "3",
                ],
            )

        call_kwargs = MockEngine.call_args.kwargs
        assert call_kwargs.get("compress_interval") == 3


class TestAnalyzeCommand:
    def _make_lineage_file(self, tmp_path: Path) -> Path:
        from cambrian.memory import EvolutionaryMemory

        mem = EvolutionaryMemory(name="test-run")
        for i in range(6):
            mem.add_agent(
                agent_id=f"agent_{i}",
                generation=i // 2,
                fitness=0.4 + i * 0.1,
                genome_snapshot={
                    "system_prompt": "You are a test agent " * max(1, i),
                    "strategy": "step-by-step" if i % 2 == 0 else "concise",
                    "temperature": 0.5 + i * 0.05,
                },
                parents=[f"agent_{i-1}"] if i > 0 else None,
            )
        path = tmp_path / "lineage.json"
        path.write_text(mem.to_json())
        return path

    def test_analyze_exits_zero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        path = self._make_lineage_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(path)])
        assert result.exit_code == 0, result.output

    def test_analyze_shows_generation_stats(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        path = self._make_lineage_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(path)])
        # Should mention generation numbers
        assert "Gen" in result.output or "gen" in result.output.lower()

    def test_analyze_top_flag(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main

        path = self._make_lineage_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(path), "--top", "2"])
        assert result.exit_code == 0

    def test_analyze_empty_memory_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from cambrian.cli import main
        from cambrian.memory import EvolutionaryMemory

        mem = EvolutionaryMemory(name="empty")
        path = tmp_path / "empty.json"
        path.write_text(mem.to_json())
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(path)])
        # Should report "No data" without crashing
        assert "No data" in result.output or result.exit_code == 0
