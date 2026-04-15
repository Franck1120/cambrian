# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Shared pytest fixtures for the Cambrian test suite.

Provides reusable fixtures for:
- Mock LLM backend (no API key required)
- Sample Genome objects
- Sample Agent objects
- Sample populations
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOOD_GENOME_DICT = {
    "system_prompt": "You are an expert step-by-step systematic solver.",
    "strategy": "step-by-step",
    "temperature": 0.7,
    "model": "gpt-4o-mini",
    "tools": [],
    "few_shot_examples": [],
}

_GOOD_GENOME_JSON = json.dumps(_GOOD_GENOME_DICT)


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MagicMock:
    """A mock LLM backend that returns a valid genome JSON on every call.

    Use this for evolution / mutator tests where the backend response must
    parse as a Genome.
    """
    backend = MagicMock()
    backend.generate = MagicMock(return_value=_GOOD_GENOME_JSON)
    return backend


@pytest.fixture
def mock_backend_text() -> MagicMock:
    """A mock LLM backend that returns plain text (not JSON).

    Use this for agent.run() tests or A2A delegation tests.
    """
    backend = MagicMock()
    backend.generate = MagicMock(return_value="mock agent response")
    return backend


# ---------------------------------------------------------------------------
# Genome fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_genome() -> Genome:
    """A single Genome with a clear system prompt."""
    return Genome(system_prompt="You are a helpful assistant.")


@pytest.fixture
def expert_genome() -> Genome:
    """A Genome loaded with high-value keywords (scores well on keyword evaluators)."""
    return Genome(
        system_prompt="You are an expert step-by-step systematic analytical solver.",
        strategy="step-by-step",
        temperature=0.5,
    )


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_agent(mock_backend_text: MagicMock) -> Agent:
    """A single Agent with mock backend, no fitness set."""
    return Agent(
        genome=Genome(system_prompt="You are a helpful assistant."),
        backend=mock_backend_text,
    )


@pytest.fixture
def scored_agent(mock_backend_text: MagicMock) -> Agent:
    """An Agent with fitness=0.8, useful for ranking/selection tests."""
    agent = Agent(
        genome=Genome(system_prompt="Expert analytical solver."),
        backend=mock_backend_text,
    )
    agent.fitness = 0.8
    return agent


# ---------------------------------------------------------------------------
# Population fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_population(mock_backend_text: MagicMock) -> list[Agent]:
    """A population of 6 agents with varied fitness values."""
    prompts = [
        ("You are a helpful assistant.", 0.1),
        ("You are an expert solver.", 0.5),
        ("Analytical step-by-step reasoner.", 0.7),
        ("Precise and structured expert.", 0.8),
        ("Systematic rigorous validator.", 0.9),
        ("Methodical analytical verifier.", 0.6),
    ]
    population = []
    for prompt, fit in prompts:
        agent = Agent(
            genome=Genome(system_prompt=prompt),
            backend=mock_backend_text,
        )
        agent.fitness = fit
        population.append(agent)
    return population
