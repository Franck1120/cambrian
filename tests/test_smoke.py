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


# ---------------------------------------------------------------------------
# Plugin and non-core symbol smoke tests (post-migration)
# ---------------------------------------------------------------------------
# After the plugin system refactor, symbols that were in cambrian.__all__ now
# live at their canonical module paths. Backward-compat stubs in cambrian.*
# keep all existing import paths working.

_STUB_SYMBOLS: list[tuple[str, str]] = [
    # ── Moved to cambrian/plugins/ (backward-compat stubs in cambrian/) ───────
    ("cambrian.dream", "DreamPhase"),
    ("cambrian.quorum", "QuorumSensor"),
    ("cambrian.moa", "MixtureOfAgents"),
    ("cambrian.moa", "QuantumTunneler"),
    ("cambrian.reflexion", "ReflexionEvaluator"),
    ("cambrian.symbiosis", "SymbioticFuser"),
    ("cambrian.symbiosis", "SymbioticPair"),
    ("cambrian.hormesis", "HormesisAdapter"),
    ("cambrian.hormesis", "HormesisEvent"),
    ("cambrian.apoptosis", "ApoptosisController"),
    ("cambrian.apoptosis", "ApoptosisEvent"),
    ("cambrian.catalysis", "CatalysisEngine"),
    ("cambrian.catalysis", "CatalystSelector"),
    ("cambrian.catalysis", "CatalysisEvent"),
    ("cambrian.llm_cascade", "LLMCascade"),
    ("cambrian.llm_cascade", "CascadeLevel"),
    ("cambrian.llm_cascade", "CascadeResult"),
    ("cambrian.llm_cascade", "hedging_confidence"),
    ("cambrian.llm_cascade", "length_confidence"),
    ("cambrian.ensemble", "AgentEnsemble"),
    ("cambrian.ensemble", "BoostingEnsemble"),
    ("cambrian.ensemble", "EnsembleResult"),
    ("cambrian.ensemble", "exact_match_scorer"),
    ("cambrian.ensemble", "substring_scorer"),
    ("cambrian.glossolalia", "GlossaloliaReasoner"),
    ("cambrian.glossolalia", "GlossaloliaEvaluator"),
    ("cambrian.glossolalia", "GlossaloliaResult"),
    ("cambrian.inference_scaling", "BestOfN"),
    ("cambrian.inference_scaling", "BeamSearch"),
    ("cambrian.inference_scaling", "ScalingResult"),
    ("cambrian.inference_scaling", "KeywordScorer"),
    ("cambrian.inference_scaling", "SelfConsistencyScorer"),
    ("cambrian.inference_scaling", "length_scorer"),
    ("cambrian.transfer", "TransferAdapter"),
    ("cambrian.transfer", "TransferBank"),
    ("cambrian.transfer", "TransferRecord"),
    ("cambrian.tabu", "TabuList"),
    ("cambrian.tabu", "TabuMutator"),
    ("cambrian.tabu", "TabuEntry"),
    ("cambrian.annealing", "AnnealingSchedule"),
    ("cambrian.annealing", "AnnealingSelector"),
    ("cambrian.annealing", "AnnealingEvent"),
    ("cambrian.red_team", "RedTeamAgent"),
    ("cambrian.red_team", "RobustnessEvaluator"),
    ("cambrian.red_team", "RedTeamSession"),
    ("cambrian.red_team", "RobustnessReport"),
    ("cambrian.red_team", "AttackResult"),
    ("cambrian.zeitgeber", "ZeitgeberClock"),
    ("cambrian.zeitgeber", "ZeitgeberScheduler"),
    ("cambrian.zeitgeber", "ZeitgeberState"),
    ("cambrian.hgt", "HGTransfer"),
    ("cambrian.hgt", "HGTPool"),
    ("cambrian.hgt", "HGTPlasmid"),
    ("cambrian.hgt", "HGTEvent"),
    ("cambrian.transgenerational", "TransgenerationalRegistry"),
    ("cambrian.transgenerational", "EpigeneMark"),
    ("cambrian.transgenerational", "InheritanceRecord"),
    ("cambrian.immune_memory", "ImmuneCortex"),
    ("cambrian.immune_memory", "BCellMemory"),
    ("cambrian.immune_memory", "TCellMemory"),
    ("cambrian.immune_memory", "MemoryCell"),
    ("cambrian.immune_memory", "RecallResult"),
    ("cambrian.neuromodulation", "NeuromodulatorBank"),
    ("cambrian.neuromodulation", "NeuroState"),
    ("cambrian.neuromodulation", "DopamineModulator"),
    ("cambrian.neuromodulation", "SerotoninModulator"),
    ("cambrian.neuromodulation", "AcetylcholineModulator"),
    ("cambrian.neuromodulation", "NoradrenalineModulator"),
    ("cambrian.metamorphosis", "MetamorphicPhase"),
    ("cambrian.metamorphosis", "PhaseConfig"),
    ("cambrian.metamorphosis", "MorphEvent"),
    ("cambrian.metamorphosis", "MetamorphosisController"),
    ("cambrian.metamorphosis", "MetamorphicPopulation"),
    ("cambrian.ecosystem", "EcologicalRole"),
    ("cambrian.ecosystem", "EcosystemConfig"),
    ("cambrian.ecosystem", "EcosystemEvent"),
    ("cambrian.ecosystem", "EcosystemInteraction"),
    ("cambrian.ecosystem", "EcosystemEvaluator"),
    ("cambrian.fractal", "FractalScale"),
    ("cambrian.fractal", "ScaleConfig"),
    ("cambrian.fractal", "FractalResult"),
    ("cambrian.fractal", "FractalMutator"),
    ("cambrian.fractal", "FractalPopulation"),
    ("cambrian.fractal", "FractalEvolution"),
    ("cambrian.dpo", "DPOPair"),
    ("cambrian.dpo", "DPOSelector"),
    ("cambrian.dpo", "DPOTrainer"),
    ("cambrian.safeguards", "DriftEvent"),
    ("cambrian.safeguards", "GoalDriftDetector"),
    ("cambrian.safeguards", "FitnessAnomalyDetector"),
    ("cambrian.safeguards", "SafeguardController"),
    ("cambrian.code_genome", "CodeGenome"),
    ("cambrian.code_genome", "CodeAgent"),
    ("cambrian.code_genome", "CodeMutator"),
    ("cambrian.code_genome", "CodeEvaluator"),
    ("cambrian.code_genome", "CodeEvolutionEngine"),
    ("cambrian.pipeline", "PipelineStep"),
    ("cambrian.pipeline", "Pipeline"),
    ("cambrian.pipeline", "PipelineMutator"),
    ("cambrian.pipeline", "PipelineEvaluator"),
    ("cambrian.pipeline", "PipelineEvolutionEngine"),
    # ── Non-plugin modules removed from __all__ but still importable ──────────
    ("cambrian.lamarck", "LamarckianAdapter"),
    ("cambrian.archipelago", "Archipelago"),
    ("cambrian.diffcot", "DiffCoTEvaluator"),
    ("cambrian.diffcot", "DiffCoTReasoner"),
    ("cambrian.diffcot", "make_diffcot_evaluator"),
    ("cambrian.causal", "CausalGraph"),
    ("cambrian.causal", "CausalMutator"),
    ("cambrian.causal", "inject_causal_context"),
    ("cambrian.tool_creation", "ToolInventor"),
    ("cambrian.tool_creation", "ToolPopulationRegistry"),
    ("cambrian.self_play", "SelfPlayEvaluator"),
    ("cambrian.self_play", "TournamentRecord"),
    ("cambrian.self_play", "run_tournament"),
    ("cambrian.meta_evolution", "MetaEvolutionEngine"),
    ("cambrian.meta_evolution", "HyperParams"),
    ("cambrian.world_model", "WorldModelEvaluator"),
    ("cambrian.world_model", "WorldModel"),
    ("cambrian.world_model", "world_model_fitness"),
    # ── Plugin system public API ──────────────────────────────────────────────
    ("cambrian.plugin_registry", "PluginRegistry"),
    ("cambrian.plugins.base", "CambrianPlugin"),
]


class TestSymbolsStillImportable:
    """All symbols previously in cambrian.__all__ must still be importable.

    After the plugin system refactor, symbols live at their canonical module
    paths. Backward-compat stubs ensure ``cambrian.dream.DreamPhase`` etc.
    still work unchanged.
    """

    @pytest.mark.parametrize("module_path,symbol", _STUB_SYMBOLS)
    def test_symbol_importable_via_module(self, module_path: str, symbol: str) -> None:
        """Each symbol must be importable from its module path."""
        mod = importlib.import_module(module_path)
        obj = getattr(mod, symbol, None)
        assert obj is not None, (
            f"{module_path}.{symbol} is not importable after plugin migration"
        )
