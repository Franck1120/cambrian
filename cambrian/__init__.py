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

    from cambrian.plugins.code_genome import CodeEvolutionEngine, CodeGenome

    engine = CodeEvolutionEngine(backend=backend, population_size=6)
    best = engine.evolve(
        seed=CodeGenome(description="reverse a string"),
        task="Write a Python function reverse(s: str) -> str",
        test_cases=[{"input": "hello", "expected": "olleh"}],
        n_generations=8,
    )

Plugin system::

    from cambrian.plugin_registry import PluginRegistry

    registry = PluginRegistry()
    print(registry.discover())   # list all available plugins
    registry.enable(["dream", "quorum"], engine)

Access plugins directly::

    from cambrian.plugins.dream import DreamPhase
    from cambrian.plugins.ensemble import AgentEnsemble
    # ... or via backward-compat stubs:
    from cambrian.dream import DreamPhase  # still works
"""

__version__ = "1.0.4"
__author__ = "Cambrian AI Contributors"

# ── Core API ──────────────────────────────────────────────────────────────────
from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.memory import EvolutionaryMemory
from cambrian.evaluator import Evaluator
from cambrian.plugin_registry import PluginRegistry

__all__ = [
    # Runtime metadata
    "__version__",
    "__author__",
    # Agent & genome
    "Agent",
    "Genome",
    # Evolution engine
    "EvolutionEngine",
    # Mutation
    "LLMMutator",
    # Memory
    "EvolutionaryMemory",
    # Base evaluator
    "Evaluator",
    # Plugin system
    "PluginRegistry",
]
