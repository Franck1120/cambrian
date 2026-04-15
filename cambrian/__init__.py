# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""
Cambrian — Evolutionary AI Agent Framework.

Evolve AI agents autonomously through genetic algorithms and LLM-guided mutation.

Quick start (Evolve mode)::

    from cambrian import Agent, Genome, EvolutionEngine
    from cambrian.mutator import LLMMutator
    from cambrian.backends.openai_compat import OpenAICompatBackend
    from cambrian.evaluators.code import CodeEvaluator

    backend = OpenAICompatBackend(model="gpt-4o-mini")
    engine = EvolutionEngine(
        evaluator=CodeEvaluator(expected_output="hello"),
        mutator=LLMMutator(backend=backend),
        backend=backend,
    )
    best = engine.evolve([Genome(system_prompt="You are a Python expert.")],
                         task="Print hello", n_generations=5)

Quick start (Forge mode — code evolution)::

    from cambrian.code_genome import CodeEvolutionEngine, CodeGenome

    engine = CodeEvolutionEngine(backend=backend, population_size=6)
    best = engine.evolve(
        seed=CodeGenome(description="reverse a string"),
        task="Write a Python function reverse(s: str) -> str",
        test_cases=[{"input": "hello", "expected": "olleh"}],
        n_generations=8,
    )
"""

__version__ = "1.0.3"
__author__ = "Cambrian AI Contributors"

# Core
from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.memory import EvolutionaryMemory
from cambrian.evaluator import Evaluator

# Bio-inspired
from cambrian.lamarck import LamarckianAdapter
from cambrian.archipelago import Archipelago

# Round 7 — Reasoning & Tools
from cambrian.diffcot import DiffCoTEvaluator, DiffCoTReasoner, make_diffcot_evaluator
from cambrian.causal import CausalGraph, CausalMutator, inject_causal_context
from cambrian.tool_creation import ToolInventor, ToolPopulationRegistry

# Round 8 — Competition, Meta-learning, World Model
from cambrian.self_play import SelfPlayEvaluator, TournamentRecord, run_tournament
from cambrian.meta_evolution import MetaEvolutionEngine, HyperParams
from cambrian.world_model import WorldModelEvaluator, WorldModel, world_model_fitness

# Round 9 — Forge mode (code + pipeline evolution)
from cambrian.code_genome import (
    CodeGenome,
    CodeAgent,
    CodeMutator,
    CodeEvaluator as CodeGenomeEvaluator,
    CodeEvolutionEngine,
)
from cambrian.pipeline import (
    PipelineStep,
    Pipeline,
    PipelineMutator,
    PipelineEvaluator,
    PipelineEvolutionEngine,
)

# Round 10 — Dream, Quorum, MoA, Reflexion, Quantum Tunneling
from cambrian.dream import DreamPhase
from cambrian.quorum import QuorumSensor
from cambrian.moa import MixtureOfAgents, QuantumTunneler
from cambrian.reflexion import ReflexionEvaluator

# Tier 3 — Symbiotic Fusion, Hormesis, Apoptosis, Catalysis, LLM Cascade, Ensemble
from cambrian.symbiosis import SymbioticFuser, SymbioticPair
from cambrian.hormesis import HormesisAdapter, HormesisEvent
from cambrian.apoptosis import ApoptosisController, ApoptosisEvent
from cambrian.catalysis import CatalysisEngine, CatalystSelector, CatalysisEvent
from cambrian.llm_cascade import (
    LLMCascade,
    CascadeLevel,
    CascadeResult,
    hedging_confidence,
    length_confidence,
)
from cambrian.ensemble import (
    AgentEnsemble,
    BoostingEnsemble,
    EnsembleResult,
    exact_match_scorer,
    substring_scorer,
)
from cambrian.glossolalia import (
    GlossaloliaReasoner,
    GlossaloliaEvaluator,
    GlossaloliaResult,
)
from cambrian.inference_scaling import (
    BestOfN,
    BeamSearch,
    ScalingResult,
    KeywordScorer,
    SelfConsistencyScorer,
    length_scorer,
)

# Tier 4 — Transfer Learning, Tabu, Annealing, Red Teaming, Zeitgeber, HGT, ...
from cambrian.transfer import TransferAdapter, TransferBank, TransferRecord
from cambrian.tabu import TabuList, TabuMutator, TabuEntry
from cambrian.annealing import AnnealingSchedule, AnnealingSelector, AnnealingEvent
from cambrian.red_team import (
    RedTeamAgent,
    RobustnessEvaluator,
    RedTeamSession,
    RobustnessReport,
    AttackResult,
)
from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler, ZeitgeberState
from cambrian.hgt import HGTransfer, HGTPool, HGTPlasmid, HGTEvent
from cambrian.transgenerational import (
    TransgenerationalRegistry,
    EpigeneMark,
    InheritanceRecord,
)
from cambrian.immune_memory import (
    ImmuneCortex,
    BCellMemory,
    TCellMemory,
    MemoryCell,
    RecallResult,
)
from cambrian.neuromodulation import (
    NeuromodulatorBank,
    NeuroState,
    DopamineModulator,
    SerotoninModulator,
    AcetylcholineModulator,
    NoradrenalineModulator,
)

# Tier 5 — DPO selection, Safeguards
from cambrian.dpo import DPOPair, DPOSelector, DPOTrainer
from cambrian.safeguards import (
    DriftEvent,
    FitnessAnomalyDetector,
    GoalDriftDetector,
    SafeguardController,
)

# Tier 5 — Metamorphosis, Ecosystem, Fractal Evolution
from cambrian.metamorphosis import (
    MetamorphicPhase,
    PhaseConfig,
    MorphEvent,
    MetamorphosisController,
    MetamorphicPopulation,
)
from cambrian.ecosystem import (
    EcologicalRole,
    EcosystemConfig,
    EcosystemEvent,
    EcosystemInteraction,
    EcosystemEvaluator,
)
from cambrian.fractal import (
    FractalScale,
    ScaleConfig,
    FractalResult,
    FractalMutator,
    FractalPopulation,
    FractalEvolution,
)

__all__ = [
    # Core
    "Agent",
    "Genome",
    "EvolutionEngine",
    "LLMMutator",
    "EvolutionaryMemory",
    "Evaluator",
    # Bio-inspired
    "LamarckianAdapter",
    "Archipelago",
    # Reasoning & Tools
    "DiffCoTEvaluator",
    "DiffCoTReasoner",
    "make_diffcot_evaluator",
    "CausalGraph",
    "CausalMutator",
    "inject_causal_context",
    "ToolInventor",
    "ToolPopulationRegistry",
    # Competition & Meta-learning
    "SelfPlayEvaluator",
    "TournamentRecord",
    "run_tournament",
    "MetaEvolutionEngine",
    "HyperParams",
    # World Model
    "WorldModelEvaluator",
    "WorldModel",
    "world_model_fitness",
    # Forge mode — Code evolution
    "CodeGenome",
    "CodeAgent",
    "CodeMutator",
    "CodeGenomeEvaluator",
    "CodeEvolutionEngine",
    # Forge mode — Pipeline evolution
    "PipelineStep",
    "Pipeline",
    "PipelineMutator",
    "PipelineEvaluator",
    "PipelineEvolutionEngine",
    # Dream, Quorum, MoA, Reflexion, Quantum Tunneling
    "DreamPhase",
    "QuorumSensor",
    "MixtureOfAgents",
    "QuantumTunneler",
    "ReflexionEvaluator",
    # Tier 3 — Symbiotic Fusion, Hormesis, Apoptosis, Catalysis, LLM Cascade, Ensemble
    "SymbioticFuser",
    "SymbioticPair",
    "HormesisAdapter",
    "HormesisEvent",
    "ApoptosisController",
    "ApoptosisEvent",
    "CatalysisEngine",
    "CatalystSelector",
    "CatalysisEvent",
    "LLMCascade",
    "CascadeLevel",
    "CascadeResult",
    "hedging_confidence",
    "length_confidence",
    "AgentEnsemble",
    "BoostingEnsemble",
    "EnsembleResult",
    "exact_match_scorer",
    "substring_scorer",
    "GlossaloliaReasoner",
    "GlossaloliaEvaluator",
    "GlossaloliaResult",
    "BestOfN",
    "BeamSearch",
    "ScalingResult",
    "KeywordScorer",
    "SelfConsistencyScorer",
    "length_scorer",
    # Tier 4
    "TransferAdapter",
    "TransferBank",
    "TransferRecord",
    "TabuList",
    "TabuMutator",
    "TabuEntry",
    "AnnealingSchedule",
    "AnnealingSelector",
    "AnnealingEvent",
    "RedTeamAgent",
    "RobustnessEvaluator",
    "RedTeamSession",
    "RobustnessReport",
    "AttackResult",
    "ZeitgeberClock",
    "ZeitgeberScheduler",
    "ZeitgeberState",
    "HGTransfer",
    "HGTPool",
    "HGTPlasmid",
    "HGTEvent",
    "TransgenerationalRegistry",
    "EpigeneMark",
    "InheritanceRecord",
    # Tier 4 — B/T-cell Immune Memory
    "ImmuneCortex",
    "BCellMemory",
    "TCellMemory",
    "MemoryCell",
    "RecallResult",
    # Tier 4 — Neuromodulation
    "NeuromodulatorBank",
    "NeuroState",
    "DopamineModulator",
    "SerotoninModulator",
    "AcetylcholineModulator",
    "NoradrenalineModulator",
    # Tier 5 — Metamorphosis
    "MetamorphicPhase",
    "PhaseConfig",
    "MorphEvent",
    "MetamorphosisController",
    "MetamorphicPopulation",
    # Tier 5 — Ecosystem
    "EcologicalRole",
    "EcosystemConfig",
    "EcosystemEvent",
    "EcosystemInteraction",
    "EcosystemEvaluator",
    # Tier 5 — Fractal Evolution
    "FractalScale",
    "ScaleConfig",
    "FractalResult",
    "FractalMutator",
    "FractalPopulation",
    "FractalEvolution",
    # Tier 5 — DPO & Safeguards
    "DPOPair",
    "DPOSelector",
    "DPOTrainer",
    "DriftEvent",
    "GoalDriftDetector",
    "FitnessAnomalyDetector",
    "SafeguardController",
    # Meta
    "__version__",
]
