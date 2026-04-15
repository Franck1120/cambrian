"""
Cambrian — Evolutionary AI Agent Framework.

Evolve AI agents autonomously through genetic algorithms and LLM-guided mutation.

Quick start::

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
"""

__version__ = "0.9.0"
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

# Round 9 — Forge mode: Code Evolution + Pipeline Composition
from cambrian.code_genome import (
    CodeGenome,
    CodeMutator,
    CodeEvaluator,
    CodeEvaluationResult,
    CodeEvolutionEngine,
    TestCase,
)
from cambrian.pipeline import (
    Pipeline,
    PipelineStep,
    PipelineRunner,
    PipelineEvaluator,
    PipelineMutator,
    PipelineEvolutionEngine,
)

# Round 9 — Adaptive Pressures: Dream Phase + Quorum Sensing
from cambrian.dream import DreamPhase, DreamScenario, Experience
from cambrian.quorum import QuorumSensor, QuorumState

# Round 9 — Ensemble & Reflexion
from cambrian.moa import MixtureOfAgents, MoAResult, QuantumTunneler, TunnelEvent
from cambrian.reflexion import (
    ReflexionAgent,
    ReflexionEvaluator,
    ReflexionResult,
    ReflexionRound,
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
    # Code Evolution (Forge)
    "CodeGenome",
    "CodeMutator",
    "CodeEvaluator",
    "CodeEvaluationResult",
    "CodeEvolutionEngine",
    "TestCase",
    # Pipeline Composition (Forge)
    "Pipeline",
    "PipelineStep",
    "PipelineRunner",
    "PipelineEvaluator",
    "PipelineMutator",
    "PipelineEvolutionEngine",
    # Adaptive Pressures
    "DreamPhase",
    "DreamScenario",
    "Experience",
    "QuorumSensor",
    "QuorumState",
    # Ensemble & Reflexion
    "MixtureOfAgents",
    "MoAResult",
    "QuantumTunneler",
    "TunnelEvent",
    "ReflexionAgent",
    "ReflexionEvaluator",
    "ReflexionResult",
    "ReflexionRound",
    # Meta
    "__version__",
]
