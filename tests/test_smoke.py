# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Smoke tests — import integrity checks for Cambrian's public surface.

These tests verify that:
- Every symbol listed in cambrian.__all__ is importable at the top level
- The dashboard module is importable without streamlit installed
- Core version metadata is accessible

No logic is exercised here — these are "does it import cleanly?" guards
that catch circular imports, missing __init__ exports, and broken submodules.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import cambrian


# ---------------------------------------------------------------------------
# __all__ importability
# ---------------------------------------------------------------------------


class TestAllSymbolsImportable:
    """Every name in cambrian.__all__ must resolve to a non-None object."""

    @pytest.mark.parametrize("symbol", cambrian.__all__)
    def test_symbol_importable(self, symbol: str) -> None:
        """Each symbol in __all__ must be accessible from the top-level package."""
        obj = getattr(cambrian, symbol, None)
        assert obj is not None, (
            f"cambrian.{symbol} is listed in __all__ but is None or not importable"
        )

    def test_all_is_not_empty(self) -> None:
        assert len(cambrian.__all__) > 0

    def test_version_is_string(self) -> None:
        assert isinstance(cambrian.__version__, str)
        assert len(cambrian.__version__) > 0

    def test_version_format(self) -> None:
        parts = cambrian.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_author_is_string(self) -> None:
        assert isinstance(cambrian.__author__, str)

    def test_core_classes_present(self) -> None:
        """Spot-check: Agent, Genome, EvolutionEngine must always be in __all__."""
        assert "Agent" in cambrian.__all__
        assert "Genome" in cambrian.__all__
        assert "EvolutionEngine" in cambrian.__all__


# ---------------------------------------------------------------------------
# Submodule import smoke tests
# ---------------------------------------------------------------------------


_SUBMODULES = [
    "cambrian.agent",
    "cambrian.evolution",
    "cambrian.mutator",
    "cambrian.evaluator",
    "cambrian.memory",
    "cambrian.dream",
    "cambrian.quorum",
    "cambrian.apoptosis",
    "cambrian.a2a",
    "cambrian.cli_tools",
    "cambrian.export",
    "cambrian.code_genome",
    "cambrian.pipeline",
    "cambrian.self_play",
    "cambrian.meta_evolution",
    "cambrian.safeguards",
    "cambrian.dpo",
    "cambrian.ecosystem",
    "cambrian.metamorphosis",
    "cambrian.fractal",
    "cambrian.stats",
    "cambrian.cache",
    "cambrian.router",
    "cambrian.compress",
    "cambrian.transfer",
    "cambrian.annealing",
    "cambrian.tabu",
    "cambrian.zeitgeber",
    "cambrian.hgt",
    "cambrian.immune_memory",
    "cambrian.neuromodulation",
    "cambrian.transgenerational",
    "cambrian.ensemble",
    "cambrian.moa",
    "cambrian.llm_cascade",
    "cambrian.glossolalia",
    "cambrian.inference_scaling",
    "cambrian.red_team",
]


class TestSubmoduleImports:
    @pytest.mark.parametrize("module", _SUBMODULES)
    def test_submodule_importable(self, module: str) -> None:
        """Every listed submodule must import without errors."""
        mod = importlib.import_module(module)
        assert mod is not None

    def test_dashboard_importable_without_streamlit(self) -> None:
        """cambrian.dashboard must import cleanly even if streamlit is absent.

        The module uses lazy imports for streamlit — only failing at
        runtime when the dashboard is actually started.
        """
        # Remove streamlit from sys.modules to simulate it being absent,
        # then verify dashboard still imports (lazy import pattern).
        st_backup = sys.modules.pop("streamlit", None)
        try:
            # Force reimport
            sys.modules.pop("cambrian.dashboard", None)
            mod = importlib.import_module("cambrian.dashboard")
            assert mod is not None
            # The run_dashboard function must exist
            assert hasattr(mod, "run_dashboard")
        finally:
            # Restore streamlit if it was present
            if st_backup is not None:
                sys.modules["streamlit"] = st_backup

    def test_backends_importable(self) -> None:
        from cambrian.backends.openai_compat import OpenAICompatBackend  # noqa: F401
        from cambrian.backends.base import LLMBackend  # noqa: F401

    def test_evaluators_importable(self) -> None:
        from cambrian.evaluators.code import CodeEvaluator  # noqa: F401
        from cambrian.evaluators.composite import CompositeEvaluator  # noqa: F401
        from cambrian.evaluators.llm_judge import LLMJudgeEvaluator  # noqa: F401

    def test_utils_importable(self) -> None:
        from cambrian.utils.logging import get_logger  # noqa: F401
        from cambrian.utils.sandbox import run_in_sandbox  # noqa: F401
