# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for the plugin infrastructure: base class, registry, and hook system."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.plugins.base import CambrianPlugin


# -- Fakes (reused from test_evolution.py pattern) ----------------------------


class _EchoBackend:
    """Minimal backend that returns the input genome JSON unchanged."""

    model_name = "echo"

    def generate(self, prompt: str, **kwargs: object) -> str:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", prompt)
        if m:
            return m.group(1)
        m2 = re.search(r"\{[\s\S]+\}", prompt)
        return m2.group(0) if m2 else "{}"


class _ConstantEvaluator:
    """Always returns the same fitness value."""

    def __init__(self, fitness: float = 0.5) -> None:
        self._fitness = fitness

    def __call__(self, agent: Agent, task: str) -> float:
        return self._fitness


def _make_engine(
    evaluator: object | None = None,
    population_size: int = 4,
    seed: int = 42,
) -> EvolutionEngine:
    backend = _EchoBackend()
    mutator = LLMMutator(backend=backend, fallback_on_error=True)  # type: ignore[arg-type]
    return EvolutionEngine(
        evaluator=evaluator or _ConstantEvaluator(),  # type: ignore[arg-type]
        mutator=mutator,
        population_size=population_size,
        mutation_rate=0.8,
        crossover_rate=0.3,
        elite_ratio=0.25,
        tournament_k=2,
        seed=seed,
    )


def _seed_genome(prompt: str = "solve the task") -> Genome:
    return Genome(system_prompt=prompt, temperature=0.7, model="gpt-4o-mini")


# -- Concrete plugin for testing ----------------------------------------------


class DummyPlugin(CambrianPlugin):
    """A concrete plugin used exclusively in tests."""

    name = "dummy"
    description = "A test plugin that records hook calls."
    category = "testing"
    impact = None

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def register(self, engine: EvolutionEngine) -> None:
        """Attach a recording hook to on_generation_end."""
        engine.add_hook(
            "on_generation_end",
            lambda **kwargs: self.calls.append(f"gen_{kwargs['generation']}"),
        )


class AllHooksPlugin(CambrianPlugin):
    """Plugin that registers on all 7 hook points to verify they fire."""

    name = "all_hooks"
    description = "Records every hook invocation."
    category = "testing"

    def __init__(self) -> None:
        super().__init__()
        self.fired: list[str] = []

    def register(self, engine: EvolutionEngine) -> None:
        for hook_name in (
            "pre_mutation",
            "post_mutation",
            "pre_selection",
            "post_selection",
            "pre_evaluation",
            "post_evaluation",
            "on_generation_end",
        ):
            # Capture hook_name in closure via default arg
            engine.add_hook(
                hook_name,
                lambda _name=hook_name, **kwargs: self.fired.append(_name),
            )


# =============================================================================
# Tests: CambrianPlugin base class
# =============================================================================


class TestCambrianPluginBase:
    """Verify the ABC contract and metadata helpers."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            CambrianPlugin()  # type: ignore[abstract]

    def test_metadata_returns_correct_dict(self) -> None:
        plugin = DummyPlugin()
        meta = plugin.metadata()
        assert meta == {
            "name": "dummy",
            "description": "A test plugin that records hook calls.",
            "category": "testing",
            "impact": None,
        }

    def test_activate_deactivate_cycle(self) -> None:
        plugin = DummyPlugin()
        assert not plugin.is_active

        plugin.activate()
        assert plugin.is_active

        plugin.deactivate()
        assert not plugin.is_active

    def test_repr_shows_state(self) -> None:
        plugin = DummyPlugin()
        assert "inactive" in repr(plugin)
        assert "dummy" in repr(plugin)

        plugin.activate()
        assert "active" in repr(plugin)

    def test_default_class_attributes(self) -> None:
        plugin = DummyPlugin()
        assert plugin.name == "dummy"
        assert plugin.description != ""
        assert plugin.category == "testing"


# =============================================================================
# Tests: EvolutionEngine hook system
# =============================================================================


class TestEvolutionEngineHooks:
    """Verify add_hook / _run_hooks on the engine."""

    def test_hooks_dict_initialized_with_seven_keys(self) -> None:
        engine = _make_engine()
        expected_hooks = {
            "pre_mutation",
            "post_mutation",
            "pre_selection",
            "post_selection",
            "pre_evaluation",
            "post_evaluation",
            "on_generation_end",
        }
        assert set(engine._hooks.keys()) == expected_hooks

    def test_hooks_start_empty(self) -> None:
        engine = _make_engine()
        for hook_list in engine._hooks.values():
            assert hook_list == []

    def test_add_hook_appends_callable(self) -> None:
        engine = _make_engine()
        fn = MagicMock()
        engine.add_hook("pre_mutation", fn)
        assert fn in engine._hooks["pre_mutation"]

    def test_add_hook_rejects_invalid_name(self) -> None:
        engine = _make_engine()
        with pytest.raises(ValueError, match="Unknown hook"):
            engine.add_hook("not_a_real_hook", lambda: None)

    def test_run_hooks_invokes_all_registered(self) -> None:
        engine = _make_engine()
        fn1 = MagicMock()
        fn2 = MagicMock()
        engine.add_hook("on_generation_end", fn1)
        engine.add_hook("on_generation_end", fn2)
        engine._run_hooks("on_generation_end", generation=1, population=[])
        fn1.assert_called_once_with(generation=1, population=[])
        fn2.assert_called_once_with(generation=1, population=[])

    def test_run_hooks_noop_when_empty(self) -> None:
        """Hooks with no registered callables should not raise."""
        engine = _make_engine()
        engine._run_hooks("pre_mutation", parent=None, task="t")

    def test_on_generation_end_fires_during_evolve(self) -> None:
        fired: list[int] = []
        engine = _make_engine(population_size=4)
        engine.add_hook(
            "on_generation_end",
            lambda **kw: fired.append(kw["generation"]),
        )
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=3,
        )
        # on_generation_end fires for gens 1, 2, 3 (not gen 0)
        assert fired == [1, 2, 3]

    def test_all_seven_hooks_fire_during_evolve(self) -> None:
        plugin = AllHooksPlugin()
        engine = _make_engine(population_size=4)
        plugin.register(engine)
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=1,
        )
        # Every hook type should have fired at least once
        hook_types_fired = set(plugin.fired)
        assert "pre_evaluation" in hook_types_fired
        assert "post_evaluation" in hook_types_fired
        assert "pre_selection" in hook_types_fired
        assert "post_selection" in hook_types_fired
        assert "pre_mutation" in hook_types_fired or "post_mutation" in hook_types_fired
        assert "on_generation_end" in hook_types_fired

    def test_pre_post_evaluation_hooks_receive_correct_kwargs(self) -> None:
        pre_kwargs: list[dict] = []
        post_kwargs: list[dict] = []
        engine = _make_engine(population_size=2)
        engine.add_hook("pre_evaluation", lambda **kw: pre_kwargs.append(kw))
        engine.add_hook("post_evaluation", lambda **kw: post_kwargs.append(kw))

        pop = engine.initialize_population([_seed_genome()])
        engine.evaluate_population(pop, "my_task")

        assert len(pre_kwargs) == 2
        assert len(post_kwargs) == 2
        # pre_evaluation receives agent and task
        assert "agent" in pre_kwargs[0]
        assert pre_kwargs[0]["task"] == "my_task"
        # post_evaluation receives agent, score, and task
        assert "agent" in post_kwargs[0]
        assert "score" in post_kwargs[0]
        assert post_kwargs[0]["task"] == "my_task"

    def test_hooks_do_not_break_existing_on_generation_callback(self) -> None:
        """The on_generation callback still works alongside hooks."""
        callback_calls: list[int] = []
        hook_calls: list[int] = []

        engine = _make_engine(population_size=4)
        engine.add_hook(
            "on_generation_end",
            lambda **kw: hook_calls.append(kw["generation"]),
        )
        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="test",
            n_generations=2,
            on_generation=lambda gen, pop: callback_calls.append(gen),
        )
        # Callback fires for gen 0 (initial) + 1, 2
        assert callback_calls == [0, 1, 2]
        # Hook fires for gen 1, 2 only
        assert hook_calls == [1, 2]


# =============================================================================
# Tests: PluginRegistry
# =============================================================================


class TestPluginRegistry:
    """Verify discovery, loading, enable/disable, and YAML config."""

    def test_discover_returns_sorted_list(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        names = registry.discover()
        assert isinstance(names, list)
        # 'base' is a reserved name and should be excluded
        assert "base" not in names
        # The list should be sorted
        assert names == sorted(names)

    def test_load_nonexistent_plugin_raises(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        with pytest.raises(ModuleNotFoundError):
            registry.load("nonexistent_plugin_xyz")

    def test_enable_calls_register_and_activate(self) -> None:
        """Enable should call register() then activate() on the plugin."""
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()

        # Manually load a DummyPlugin into the registry
        plugin = DummyPlugin()
        registry._loaded["dummy"] = plugin

        engine = _make_engine()
        registry.enable(["dummy"], engine)

        assert plugin.is_active

    def test_disable_deactivates_plugin(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        plugin = DummyPlugin()
        plugin.activate()
        registry._loaded["dummy"] = plugin

        registry.disable("dummy")
        assert not plugin.is_active

    def test_disable_nonexistent_is_noop(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        registry.disable("nonexistent")  # should not raise

    def test_active_plugins_property(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        p1 = DummyPlugin()
        p1.activate()
        registry._loaded["dummy"] = p1

        p2 = AllHooksPlugin()
        registry._loaded["all_hooks"] = p2

        assert "dummy" in registry.active_plugins
        assert "all_hooks" not in registry.active_plugins

    def test_repr(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        r = repr(registry)
        assert "PluginRegistry" in r

    def test_load_from_yaml_missing_file_raises(self) -> None:
        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()
        engine = _make_engine()
        with pytest.raises(FileNotFoundError):
            registry.load_from_yaml("/nonexistent/path.yaml", engine)

    def test_load_from_yaml_enables_listed_plugins(self, tmp_path: Path) -> None:
        """YAML config with plugins.enabled should enable those plugins."""
        from cambrian.plugin_registry import PluginRegistry

        # Write a minimal YAML config
        config = tmp_path / "config.yaml"
        config.write_text(
            textwrap.dedent("""\
                plugins:
                  enabled:
                    - dummy
            """)
        )

        registry = PluginRegistry()
        # Pre-load the dummy plugin so the registry can find it
        plugin = DummyPlugin()
        registry._loaded["dummy"] = plugin

        engine = _make_engine()
        registry.load_from_yaml(str(config), engine)

        assert plugin.is_active

    def test_load_from_yaml_empty_config_is_noop(self, tmp_path: Path) -> None:
        from cambrian.plugin_registry import PluginRegistry

        config = tmp_path / "config.yaml"
        config.write_text("# empty config\n")

        registry = PluginRegistry()
        engine = _make_engine()
        registry.load_from_yaml(str(config), engine)
        assert registry.active_plugins == []


# =============================================================================
# Tests: Plugin + Engine integration
# =============================================================================


class TestPluginEngineIntegration:
    """End-to-end: plugin registers hooks, engine fires them during evolution."""

    def test_dummy_plugin_records_generation_hooks(self) -> None:
        plugin = DummyPlugin()
        engine = _make_engine(population_size=4)
        plugin.register(engine)
        plugin.activate()

        engine.evolve(
            seed_genomes=[_seed_genome()],
            task="integration test",
            n_generations=3,
        )
        assert plugin.calls == ["gen_1", "gen_2", "gen_3"]
        assert plugin.is_active

    def test_existing_evolution_tests_still_pass(self) -> None:
        """Sanity check: hook additions do not break basic evolution flow."""
        engine = _make_engine(population_size=4)
        best = engine.evolve(
            seed_genomes=[_seed_genome()],
            task="regression test",
            n_generations=2,
        )
        assert isinstance(best, Agent)
        assert best.fitness is not None
        assert best.fitness >= 0.0
        assert engine.generation == 2
