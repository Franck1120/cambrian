"""LLMMutator — LLM-guided genome mutation and crossover.

Instead of blind random mutations (flip a bit, tweak a parameter), Cambrian
uses an LLM to *understand* the genome and suggest targeted improvements.
This turns the evolutionary search from a random walk into a directed search
guided by the LLM's knowledge about what makes prompts and strategies effective.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend


_MUTATE_SYSTEM = """You are an expert AI prompt engineer and evolutionary algorithm researcher.
Your task is to improve an AI agent's genome (its configuration) to make it perform better.
Analyze the genome and suggest concrete, specific improvements.
Return ONLY a valid JSON object with the same structure as the input genome."""

_MUTATE_TEMPLATE = """Current agent genome:
{genome_json}

This agent achieved fitness score: {fitness}
Task the agent must solve: {task}

Analyze this genome and return an improved version as JSON.
Focus on:
1. Making the system_prompt more effective and specific for the task
2. Adjusting the strategy to better match the task type
3. Fine-tuning the temperature (lower for factual, higher for creative tasks)
4. Keeping the same model unless there's a strong reason to change it

Return ONLY the JSON object, no explanation."""


_CROSSOVER_SYSTEM = """You are an expert AI researcher performing genetic crossover between two AI agent genomes.
Combine the best traits from both parents to create a superior offspring.
Return ONLY a valid JSON object."""

_CROSSOVER_TEMPLATE = """Parent A genome (fitness: {fitness_a}):
{genome_a}

Parent B genome (fitness: {fitness_b}):
{genome_b}

Task: {task}

Create an offspring that combines the strongest elements from both parents.
Rules:
- Take the better system_prompt as the base, then add key phrases from the other
- Choose the strategy that better fits the task
- Average the temperatures with a bias toward the higher-performing parent
- Keep the same model as the higher-fitness parent

Return ONLY the JSON object."""


class LLMMutator:
    """Mutates and crosses over agent genomes using an LLM.

    Args:
        backend: The LLM backend used to generate mutations.
        mutation_temperature: Temperature for the mutation LLM call. Low
            values (0.3–0.5) produce conservative mutations; high values
            (0.8–1.2) produce creative ones. Default ``0.6``.
        fallback_on_error: If ``True`` (default), returns the original genome
            when the LLM produces invalid JSON. If ``False``, raises.
    """

    def __init__(
        self,
        backend: LLMBackend,
        mutation_temperature: float = 0.6,
        fallback_on_error: bool = True,
    ) -> None:
        self._backend = backend
        self._mut_temp = mutation_temperature
        self._fallback = fallback_on_error

    def mutate(self, agent: Agent, task: str = "") -> Agent:
        """Return a new agent with an LLM-improved genome.

        Args:
            agent: The agent whose genome to mutate.
            task: The task description, used to focus the mutation.

        Returns:
            A new :class:`Agent` with the mutated genome. The original is
            not modified.
        """
        genome_json = json.dumps(agent.genome.to_dict(), indent=2)
        prompt = _MUTATE_TEMPLATE.format(
            genome_json=genome_json,
            fitness=f"{agent.fitness:.4f}" if agent.fitness is not None else "unknown",
            task=task or "general problem solving",
        )

        try:
            raw = self._backend.generate(
                prompt,
                system=_MUTATE_SYSTEM,
                temperature=self._mut_temp,
            )
            new_genome = self._parse_genome(raw, agent.genome)
        except Exception:
            if self._fallback:
                new_genome = self._random_tweak(agent.genome)
            else:
                raise

        new_agent = agent.clone()
        new_agent.genome = new_genome
        new_agent._fitness = None  # reset fitness after mutation
        return new_agent

    def crossover(self, parent_a: Agent, parent_b: Agent, task: str = "") -> Agent:
        """Combine two parent genomes to produce an offspring.

        Uses the LLM to intelligently select and combine traits from both
        parents rather than random segment swapping.

        Args:
            parent_a: First parent agent.
            parent_b: Second parent agent.
            task: The task description used to focus the crossover.

        Returns:
            A new :class:`Agent` with the combined genome.
        """
        prompt = _CROSSOVER_TEMPLATE.format(
            genome_a=json.dumps(parent_a.genome.to_dict(), indent=2),
            fitness_a=f"{parent_a.fitness:.4f}" if parent_a.fitness is not None else "0",
            genome_b=json.dumps(parent_b.genome.to_dict(), indent=2),
            fitness_b=f"{parent_b.fitness:.4f}" if parent_b.fitness is not None else "0",
            task=task or "general problem solving",
        )

        # Choose the higher-fitness parent as the base for the offspring
        base_parent = parent_a if (parent_a.fitness or 0) >= (parent_b.fitness or 0) else parent_b

        try:
            raw = self._backend.generate(
                prompt,
                system=_CROSSOVER_SYSTEM,
                temperature=self._mut_temp,
            )
            child_genome = self._parse_genome(raw, base_parent.genome)
        except Exception:
            if self._fallback:
                child_genome = self._deterministic_crossover(
                    parent_a.genome, parent_b.genome
                )
            else:
                raise

        child = base_parent.clone()
        child.genome = child_genome
        child._fitness = None
        return child

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_genome(raw: str, fallback_genome: Genome) -> Genome:
        """Parse a JSON-encoded genome from the LLM response."""
        # Try to extract JSON from markdown code fences
        json_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if json_match:
            raw = json_match.group(1)

        try:
            data = json.loads(raw.strip())
            return Genome.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return Genome.from_dict(fallback_genome.to_dict())

    @staticmethod
    def _random_tweak(genome: Genome) -> Genome:
        """Minimal random mutation when LLM call fails."""
        data = genome.to_dict()
        # Randomly adjust temperature by ±0.1 clamped to [0.1, 1.5]
        delta = random.uniform(-0.1, 0.1)
        data["temperature"] = max(0.1, min(1.5, data["temperature"] + delta))
        return Genome.from_dict(data)

    @staticmethod
    def _deterministic_crossover(g_a: Genome, g_b: Genome) -> Genome:
        """Fallback crossover: sentence-level mix + temperature average."""
        # Split system prompts into sentences and interleave
        sents_a = re.split(r"(?<=[.!?])\s+", g_a.system_prompt)
        sents_b = re.split(r"(?<=[.!?])\s+", g_b.system_prompt)
        mixed: list[str] = []
        for i, (sa, sb) in enumerate(zip(sents_a, sents_b)):
            mixed.append(sa if i % 2 == 0 else sb)
        # Append any remaining sentences from the longer parent
        mixed.extend(sents_a[len(sents_b):])
        mixed.extend(sents_b[len(sents_a):])

        return Genome(
            system_prompt=" ".join(mixed),
            tools=list(set(g_a.tools + g_b.tools)),
            strategy=g_a.strategy,  # take from parent A
            temperature=(g_a.temperature + g_b.temperature) / 2,
            model=g_a.model,
        )
