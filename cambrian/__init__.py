"""
Cambrian — Evolutionary AI Agent Framework.

Evolve AI agents autonomously through genetic algorithms and LLM-guided mutation.
"""

__version__ = "0.1.0"
__author__ = "Cambrian AI Contributors"

from cambrian.agent import Agent, Genome
from cambrian.evolution import EvolutionEngine

__all__ = ["Agent", "Genome", "EvolutionEngine", "__version__"]
