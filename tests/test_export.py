# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.export — all four export formats + load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cambrian.agent import Agent, Genome
from cambrian.export import (
    export_api,
    export_genome_json,
    export_mcp,
    export_standalone,
    load_genome_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(prompt: str = "You are a helpful assistant.", fitness: float = 0.75) -> Agent:
    agent = Agent(genome=Genome(system_prompt=prompt))
    agent.fitness = fitness
    return agent


# ---------------------------------------------------------------------------
# export_genome_json / load_genome_json
# ---------------------------------------------------------------------------


class TestExportGenomeJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "genome.json"
        result = export_genome_json(agent, dest)
        assert result == dest
        assert dest.exists()

    def test_file_contains_genome_key(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="expert solver")
        dest = tmp_path / "genome.json"
        export_genome_json(agent, dest)
        data = json.loads(dest.read_text())
        assert "genome" in data

    def test_file_contains_fitness(self, tmp_path: Path) -> None:
        agent = _make_agent(fitness=0.42)
        dest = tmp_path / "genome.json"
        export_genome_json(agent, dest)
        data = json.loads(dest.read_text())
        assert abs(data["fitness"] - 0.42) < 1e-6

    def test_file_contains_agent_id(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "genome.json"
        export_genome_json(agent, dest)
        data = json.loads(dest.read_text())
        assert "agent_id" in data
        assert isinstance(data["agent_id"], str)

    def test_file_contains_version(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "genome.json"
        export_genome_json(agent, dest)
        data = json.loads(dest.read_text())
        assert "cambrian_version" in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "nested" / "dir" / "genome.json"
        export_genome_json(agent, dest)
        assert dest.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        agent = _make_agent()
        result = export_genome_json(agent, tmp_path / "g.json")
        assert isinstance(result, Path)

    def test_system_prompt_preserved(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="unique system prompt xyz")
        dest = tmp_path / "genome.json"
        export_genome_json(agent, dest)
        data = json.loads(dest.read_text())
        assert data["genome"]["system_prompt"] == "unique system prompt xyz"


class TestLoadGenomeJson:
    def test_round_trip(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="round trip test prompt")
        dest = tmp_path / "rt.json"
        export_genome_json(agent, dest)
        genome = load_genome_json(dest)
        assert isinstance(genome, Genome)
        assert genome.system_prompt == "round trip test prompt"

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_genome_json(tmp_path / "nonexistent.json")

    def test_raises_key_error_on_missing_genome_key(self, tmp_path: Path) -> None:
        dest = tmp_path / "bad.json"
        dest.write_text(json.dumps({"no_genome_here": True}))
        with pytest.raises(KeyError):
            load_genome_json(dest)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "str.json"
        export_genome_json(agent, dest)
        genome = load_genome_json(str(dest))
        assert isinstance(genome, Genome)


# ---------------------------------------------------------------------------
# export_standalone
# ---------------------------------------------------------------------------


class TestExportStandalone:
    def test_creates_py_file(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "agent.py"
        result = export_standalone(agent, dest)
        assert result == dest
        assert dest.exists()

    def test_file_is_valid_python_syntax(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="test prompt for standalone")
        dest = tmp_path / "agent.py"
        export_standalone(agent, dest)
        source = dest.read_text()
        compile(source, str(dest), "exec")  # raises SyntaxError on bad syntax

    def test_contains_system_prompt(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="my unique system prompt")
        dest = tmp_path / "agent.py"
        export_standalone(agent, dest)
        source = dest.read_text()
        assert "my unique system prompt" in source

    def test_contains_model(self, tmp_path: Path) -> None:
        agent = _make_agent()
        agent.genome.model = "gpt-4o"
        dest = tmp_path / "agent.py"
        export_standalone(agent, dest)
        source = dest.read_text()
        assert "gpt-4o" in source

    def test_contains_shebang(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "agent.py"
        export_standalone(agent, dest)
        source = dest.read_text()
        assert source.startswith("#!/usr/bin/env python3")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "subdir" / "agent.py"
        export_standalone(agent, dest)
        assert dest.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        agent = _make_agent()
        result = export_standalone(agent, tmp_path / "a.py")
        assert isinstance(result, Path)

    def test_no_fitness_agent_works(self, tmp_path: Path) -> None:
        agent = Agent(genome=Genome(system_prompt="no fitness"))
        dest = tmp_path / "nofitness.py"
        export_standalone(agent, dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# export_mcp
# ---------------------------------------------------------------------------


class TestExportMcp:
    def test_creates_output_dir(self, tmp_path: Path) -> None:
        agent = _make_agent()
        export_mcp(agent, tmp_path / "mcp_server")
        assert (tmp_path / "mcp_server").is_dir()

    def test_creates_manifest_json(self, tmp_path: Path) -> None:
        agent = _make_agent()
        out_dir = tmp_path / "mcp"
        export_mcp(agent, out_dir)
        files = list(out_dir.iterdir())
        json_files = [f for f in files if f.suffix == ".json"]
        assert len(json_files) >= 1

    def test_manifest_is_valid_json(self, tmp_path: Path) -> None:
        agent = _make_agent()
        out_dir = tmp_path / "mcp"
        export_mcp(agent, out_dir)
        for f in out_dir.iterdir():
            if f.suffix == ".json":
                data = json.loads(f.read_text())
                assert isinstance(data, dict)
                break

    def test_creates_server_py(self, tmp_path: Path) -> None:
        agent = _make_agent()
        out_dir = tmp_path / "mcp"
        export_mcp(agent, out_dir)
        py_files = [f for f in out_dir.iterdir() if f.suffix == ".py"]
        assert len(py_files) >= 1

    def test_server_py_valid_syntax(self, tmp_path: Path) -> None:
        agent = _make_agent()
        out_dir = tmp_path / "mcp"
        export_mcp(agent, out_dir)
        for f in out_dir.iterdir():
            if f.suffix == ".py":
                source = f.read_text()
                compile(source, str(f), "exec")

    def test_manifest_has_schema_version(self, tmp_path: Path) -> None:
        agent = _make_agent()
        out_dir = tmp_path / "mcp"
        export_mcp(agent, out_dir)
        for f in out_dir.iterdir():
            if f.suffix == ".json":
                data = json.loads(f.read_text())
                assert "schema_version" in data
                break


# ---------------------------------------------------------------------------
# export_api
# ---------------------------------------------------------------------------


class TestExportApi:
    def test_creates_py_file(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "api_agent.py"
        result = export_api(agent, dest)
        assert result == dest
        assert dest.exists()

    def test_file_is_valid_python_syntax(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "api_agent.py"
        export_api(agent, dest)
        source = dest.read_text()
        compile(source, str(dest), "exec")

    def test_contains_fastapi_import(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "api_agent.py"
        export_api(agent, dest)
        source = dest.read_text()
        assert "FastAPI" in source or "fastapi" in source.lower()

    def test_contains_post_endpoint(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "api_agent.py"
        export_api(agent, dest)
        source = dest.read_text()
        assert "/run" in source or "post" in source.lower()

    def test_contains_system_prompt(self, tmp_path: Path) -> None:
        agent = _make_agent(prompt="api test prompt xyz")
        dest = tmp_path / "api_agent.py"
        export_api(agent, dest)
        source = dest.read_text()
        assert "api test prompt xyz" in source

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        agent = _make_agent()
        dest = tmp_path / "api" / "subdir" / "app.py"
        export_api(agent, dest)
        assert dest.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        agent = _make_agent()
        result = export_api(agent, tmp_path / "api.py")
        assert isinstance(result, Path)
