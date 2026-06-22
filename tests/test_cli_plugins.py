# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for CLI plugin features: --enable flag, --config flag, and plugins command.

Verifies that:
- The ``plugins`` command lists available plugins and filters by category.
- ``evolve --enable`` parses comma-separated plugin names and calls ``enable()``.
- ``evolve --config`` loads a YAML config and calls ``load_from_yaml()``.
- ``evolve --help`` exposes the new ``--enable`` and ``--config`` options.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cambrian.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_genome_json(prompt: str = "expert step-by-step analytical prompt") -> str:
    """Return a valid genome JSON string for mocking backend responses."""
    return json.dumps({
        "system_prompt": prompt,
        "strategy": "step-by-step",
        "temperature": 0.7,
        "model": "gpt-4o-mini",
        "tools": [],
        "few_shot_examples": [],
    })


# ---------------------------------------------------------------------------
# plugins command — help & listing
# ---------------------------------------------------------------------------


class TestPluginsHelp:
    """Verify the plugins command --help output."""

    def test_plugins_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "--help"])
        assert result.exit_code == 0

    def test_plugins_help_contains_category_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "--help"])
        assert "--category" in result.output

    def test_plugins_appears_in_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "plugins" in result.output


class TestPluginsCommand:
    """Functional tests for the plugins listing command."""

    def test_plugins_lists_all(self) -> None:
        """plugins command lists all discovered plugins."""
        runner = CliRunner()
        result = runner.invoke(main, ["plugins"])
        assert result.exit_code == 0
        # At least some well-known plugins should appear
        assert "dream" in result.output
        assert "quorum" in result.output
        assert "tabu" in result.output

    def test_plugins_category_filter_memory(self) -> None:
        """plugins --category memory shows only memory plugins."""
        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "--category", "memory"])
        assert result.exit_code == 0
        assert "dream" in result.output

    def test_plugins_category_filter_no_results(self) -> None:
        """plugins --category with an unknown category shows nothing."""
        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "--category", "nonexistent_xyz"])
        assert result.exit_code == 0
        assert "No plugins found" in result.output

    def test_plugins_without_rich(self) -> None:
        """plugins command works when Rich is unavailable (plain text fallback)."""
        runner = CliRunner()
        with patch("cambrian.cli._RICH", False):
            result = runner.invoke(main, ["plugins"])
        assert result.exit_code == 0
        # Plain text header should be present
        assert "Name" in result.output
        assert "Category" in result.output
        assert "Description" in result.output


# ---------------------------------------------------------------------------
# evolve --help — new options
# ---------------------------------------------------------------------------


class TestEvolvePluginOptions:
    """Verify the evolve command exposes --enable and --config."""

    def test_evolve_help_contains_enable(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["evolve", "--help"])
        assert "--enable" in result.output

    def test_evolve_help_contains_config(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["evolve", "--help"])
        assert "--config" in result.output

    def test_evolve_help_enable_description(self) -> None:
        """--enable help text mentions comma-separated plugins."""
        runner = CliRunner()
        result = runner.invoke(main, ["evolve", "--help"])
        assert "Comma-separated" in result.output or "comma-separated" in result.output.lower()

    def test_evolve_help_config_description(self) -> None:
        """--config help text mentions YAML."""
        runner = CliRunner()
        result = runner.invoke(main, ["evolve", "--help"])
        assert "YAML" in result.output


# ---------------------------------------------------------------------------
# evolve --enable functional tests
# ---------------------------------------------------------------------------


class TestEvolveEnableFlag:
    """Functional tests for evolve --enable plugin loading."""

    def test_enable_parses_comma_separated_names(self, tmp_path: Path) -> None:
        """--enable dream,tabu calls PluginRegistry.enable with those names."""
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            # Mock backend
            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            # Mock engine
            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            # Mock registry
            mock_reg = MagicMock()
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--population", "2",
                "--enable", "dream,tabu",
            ])

        assert result.exit_code == 0, result.output
        # Verify enable() was called with the parsed names
        mock_reg.enable.assert_called_once_with(["dream", "tabu"], mock_engine)

    def test_enable_strips_whitespace(self, tmp_path: Path) -> None:
        """--enable ' dream , tabu ' strips whitespace from plugin names."""
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            mock_reg = MagicMock()
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--enable", " dream , tabu ",
            ])

        assert result.exit_code == 0, result.output
        mock_reg.enable.assert_called_once_with(["dream", "tabu"], mock_engine)

    def test_enable_handles_failure_gracefully(self, tmp_path: Path) -> None:
        """--enable prints a warning but does not crash if registry.enable fails."""
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            mock_reg = MagicMock()
            mock_reg.enable.side_effect = ValueError("plugin not found")
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--enable", "nonexistent",
            ])

        assert result.exit_code == 0, result.output
        assert "Warning" in result.output


# ---------------------------------------------------------------------------
# evolve --config functional tests
# ---------------------------------------------------------------------------


class TestEvolveConfigFlag:
    """Functional tests for evolve --config YAML plugin loading."""

    def test_config_loads_yaml_file(self, tmp_path: Path) -> None:
        """--config passes the config file to PluginRegistry.load_from_yaml."""
        config_file = tmp_path / "cambrian.yaml"
        config_file.write_text("plugins:\n  enabled:\n    - dream\n    - tabu\n")

        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            mock_reg = MagicMock()
            mock_reg.active_plugins = ["dream", "tabu"]
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--config", str(config_file),
            ])

        assert result.exit_code == 0, result.output
        mock_reg.load_from_yaml.assert_called_once_with(str(config_file), mock_engine)

    def test_config_nonexistent_file_is_skipped(self, tmp_path: Path) -> None:
        """--config with a missing file is silently skipped (no crash)."""
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--config", str(tmp_path / "nope.yaml"),
            ])

        # Should complete without error -- file doesn't exist so it's skipped
        assert result.exit_code == 0, result.output

    def test_config_handles_load_error_gracefully(self, tmp_path: Path) -> None:
        """--config prints a warning if load_from_yaml raises an exception."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("invalid: yaml: [")

        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            mock_reg = MagicMock()
            mock_reg.load_from_yaml.side_effect = RuntimeError("bad yaml")
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--config", str(config_file),
            ])

        assert result.exit_code == 0, result.output
        assert "Warning" in result.output

    def test_config_and_enable_together(self, tmp_path: Path) -> None:
        """--config and --enable can be used simultaneously."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("plugins:\n  enabled:\n    - dream\n")

        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli.EvolutionEngine") as MockEngine, \
             patch("cambrian.plugin_registry.PluginRegistry") as MockRegistry:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best = Agent(genome=Genome(system_prompt="evolved"))
            best.fitness = 0.5
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best
            MockEngine.return_value = mock_engine

            mock_reg = MagicMock()
            mock_reg.active_plugins = ["dream"]
            MockRegistry.return_value = mock_reg

            result = runner.invoke(main, [
                "evolve", "test task",
                "--api-key", "fake-key",
                "--generations", "1",
                "--config", str(config_file),
                "--enable", "tabu",
            ])

        assert result.exit_code == 0, result.output
        # Both load_from_yaml AND enable should have been called
        mock_reg.load_from_yaml.assert_called_once()
        mock_reg.enable.assert_called_once_with(["tabu"], mock_engine)
