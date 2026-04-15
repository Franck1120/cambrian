# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for Round 8 CLI commands: meta-evolve and tournament.

Verifies that:
- --help flags work and print expected options
- Parameter defaults are correct
- Commands handle mock backends without crashing
- Output files are written correctly
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
    return json.dumps({
        "system_prompt": prompt,
        "strategy": "step-by-step",
        "temperature": 0.7,
        "model": "gpt-4o-mini",
        "tools": [],
        "few_shot_examples": [],
    })


def _mock_backend_cls(return_value: str = "") -> MagicMock:
    backend = MagicMock()
    backend.generate = MagicMock(return_value=return_value or _mock_genome_json())
    cls = MagicMock(return_value=backend)
    return cls


# ---------------------------------------------------------------------------
# --help tests
# ---------------------------------------------------------------------------

class TestMetaEvolveHelp:
    """Verify meta-evolve --help output."""

    def test_meta_evolve_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert result.exit_code == 0

    def test_meta_evolve_help_contains_task(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "TASK" in result.output

    def test_meta_evolve_help_contains_generations(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "--generations" in result.output

    def test_meta_evolve_help_contains_meta_interval(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "--meta-interval" in result.output

    def test_meta_evolve_help_contains_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "--output" in result.output

    def test_meta_evolve_help_contains_population(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "--population" in result.output

    def test_meta_evolve_help_contains_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "--model" in result.output

    def test_meta_evolve_help_shows_default_generations(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "10" in result.output  # default 10 generations

    def test_meta_evolve_help_shows_default_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["meta-evolve", "--help"])
        assert "gpt-4o-mini" in result.output


class TestTournamentHelp:
    """Verify tournament --help output."""

    def test_tournament_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert result.exit_code == 0

    def test_tournament_help_contains_task(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "TASK" in result.output

    def test_tournament_help_contains_population(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "--population" in result.output

    def test_tournament_help_contains_agents_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "--agents-file" in result.output

    def test_tournament_help_contains_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "--output" in result.output

    def test_tournament_help_shows_default_population(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "6" in result.output  # default population

    def test_tournament_help_contains_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["tournament", "--help"])
        assert "--model" in result.output


# ---------------------------------------------------------------------------
# meta-evolve functional tests
# ---------------------------------------------------------------------------

class TestMetaEvolveCommand:
    """Functional tests for the meta-evolve command."""

    def test_meta_evolve_runs_and_writes_output(self, tmp_path: Path) -> None:
        """meta-evolve completes and writes the output JSON."""
        out = tmp_path / "meta_best.json"
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.meta_evolution.MetaEvolutionEngine") as MockMeta:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            # Mock MetaEvolutionEngine so it doesn't call the LLM
            from cambrian.agent import Agent, Genome
            best_agent = Agent(genome=Genome(system_prompt="evolved expert"))
            best_agent.fitness = 0.75
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best_agent
            MockMeta.return_value = mock_engine

            result = runner.invoke(main, [
                "meta-evolve", "test task",
                "--generations", "2",
                "--population", "2",
                "--output", str(out),
            ])

        assert result.exit_code == 0, result.output
        assert out.exists()
        data = json.loads(out.read_text())
        assert "system_prompt" in data

    def test_meta_evolve_default_output_filename(self, tmp_path: Path) -> None:
        """meta-evolve defaults to meta_best.json."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("cambrian.cli._make_backend") as mk_backend, \
                 patch("cambrian.meta_evolution.MetaEvolutionEngine") as MockMeta:

                backend = MagicMock()
                backend.generate = MagicMock(return_value=_mock_genome_json())
                mk_backend.return_value = backend

                from cambrian.agent import Agent, Genome
                best_agent = Agent(genome=Genome(system_prompt="evolved"))
                best_agent.fitness = 0.5
                mock_engine = MagicMock()
                mock_engine.evolve.return_value = best_agent
                MockMeta.return_value = mock_engine

                result = runner.invoke(main, ["meta-evolve", "task"])

        assert result.exit_code == 0

    def test_meta_evolve_prints_fitness(self, tmp_path: Path) -> None:
        """meta-evolve output includes fitness."""
        out = tmp_path / "out.json"
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.meta_evolution.MetaEvolutionEngine") as MockMeta:

            backend = MagicMock()
            backend.generate = MagicMock(return_value=_mock_genome_json())
            mk_backend.return_value = backend

            from cambrian.agent import Agent, Genome
            best_agent = Agent(genome=Genome(system_prompt="evolved"))
            best_agent.fitness = 0.8888
            mock_engine = MagicMock()
            mock_engine.evolve.return_value = best_agent
            MockMeta.return_value = mock_engine

            result = runner.invoke(main, [
                "meta-evolve", "task",
                "--generations", "1",
                "--output", str(out),
            ])

        assert result.exit_code == 0
        assert "0.8888" in result.output


# ---------------------------------------------------------------------------
# tournament functional tests
# ---------------------------------------------------------------------------

class TestTournamentCommand:
    """Functional tests for the tournament command."""

    def test_tournament_runs_with_random_population(self, tmp_path: Path) -> None:
        """tournament runs without agents-file (generates random population)."""
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli._make_evaluator") as mk_eval, \
             patch("cambrian.self_play.SelfPlayEvaluator") as MockSP, \
             patch("cambrian.self_play.run_tournament") as mock_rt:

            backend = MagicMock()
            mk_backend.return_value = backend

            evaluator = MagicMock()
            mk_eval.return_value = evaluator

            sp_eval = MagicMock()
            MockSP.return_value = sp_eval

            from cambrian.self_play import TournamentRecord
            record = TournamentRecord()
            mock_rt.return_value = record

            result = runner.invoke(main, [
                "tournament", "test task",
                "--population", "3",
            ])

        assert result.exit_code == 0

    def test_tournament_saves_output_json(self, tmp_path: Path) -> None:
        """tournament saves results JSON when --output is given."""
        out = tmp_path / "results.json"
        runner = CliRunner()

        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli._make_evaluator") as mk_eval, \
             patch("cambrian.self_play.SelfPlayEvaluator") as MockSP, \
             patch("cambrian.self_play.run_tournament") as mock_rt:

            backend = MagicMock()
            mk_backend.return_value = backend
            evaluator = MagicMock()
            mk_eval.return_value = evaluator
            sp_eval = MagicMock()
            MockSP.return_value = sp_eval

            from cambrian.self_play import TournamentRecord
            record = TournamentRecord()
            mock_rt.return_value = record

            result = runner.invoke(main, [
                "tournament", "test task",
                "--population", "2",
                "--output", str(out),
            ])

        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "task" in data
        assert "agents" in data
        assert data["task"] == "test task"

    def test_tournament_agents_file_invalid_json_structure(self, tmp_path: Path) -> None:
        """tournament raises error if agents-file is not a list."""
        agents_file = tmp_path / "agents.json"
        agents_file.write_text(json.dumps({"not": "a list"}))

        runner = CliRunner()
        with patch("cambrian.cli._make_backend") as mk_backend:
            mk_backend.return_value = MagicMock()
            result = runner.invoke(main, [
                "tournament", "test task",
                "--agents-file", str(agents_file),
            ])

        assert result.exit_code != 0
        assert "list" in result.output.lower() or "Error" in result.output

    def test_tournament_agents_file_valid(self, tmp_path: Path) -> None:
        """tournament loads agents from agents-file correctly."""
        from cambrian.agent import Genome
        agents_data = [
            Genome(system_prompt=f"agent {i}").to_dict()
            for i in range(3)
        ]
        agents_file = tmp_path / "agents.json"
        agents_file.write_text(json.dumps(agents_data))

        runner = CliRunner()
        with patch("cambrian.cli._make_backend") as mk_backend, \
             patch("cambrian.cli._make_evaluator") as mk_eval, \
             patch("cambrian.self_play.SelfPlayEvaluator") as MockSP, \
             patch("cambrian.self_play.run_tournament") as mock_rt:

            backend = MagicMock()
            mk_backend.return_value = backend
            mk_eval.return_value = MagicMock()
            MockSP.return_value = MagicMock()

            from cambrian.self_play import TournamentRecord
            mock_rt.return_value = TournamentRecord()

            result = runner.invoke(main, [
                "tournament", "test task",
                "--agents-file", str(agents_file),
            ])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# General CLI health checks
# ---------------------------------------------------------------------------

class TestCliHealth:
    """Verify all commands appear in --help and have required options."""

    def test_main_help_lists_meta_evolve(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "meta-evolve" in result.output

    def test_main_help_lists_tournament(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "tournament" in result.output

    def test_version_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "Cambrian" in result.output
        assert "1.0.2" in result.output

    def test_evolve_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["evolve", "--help"])
        assert result.exit_code == 0

    def test_forge_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["forge", "--help"])
        assert result.exit_code == 0

    def test_forge_help_contains_mode(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["forge", "--help"])
        assert "--mode" in result.output
        assert "code" in result.output
        assert "pipeline" in result.output

    def test_forge_help_contains_test_case(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["forge", "--help"])
        assert "--test-case" in result.output

    def test_forge_help_contains_generations(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["forge", "--help"])
        assert "--generations" in result.output

    def test_distill_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["distill", "--help"])
        assert result.exit_code == 0

    def test_stats_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0

    def test_run_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0

    def test_compare_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0

    def test_snapshot_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["snapshot", "--help"])
        assert result.exit_code == 0
