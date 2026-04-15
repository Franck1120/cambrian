# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Direct Preference Optimization (DPO) selector and trainer for Cambrian.

This module provides two complementary tools for applying DPO-style preference
learning within an evolutionary population of :class:`~cambrian.agent.Agent`
objects:

- :class:`DPOSelector` — a lightweight, fitness-only approach that adjusts each
  agent's fitness score using a reward derived from pairwise preference margins.
- :class:`DPOTrainer` — a more expensive, backend-driven approach that actually
  rewrites the genome of lower-performing agents to prefer patterns seen in
  top-performing (``chosen``) agents and avoid patterns from poor-performing
  (``rejected``) agents.

Neither class depends on gradient descent; the "DPO" framing is used loosely to
mean *explicit pairwise comparison driving optimisation*, analogous to the spirit
of the DPO objective in RLHF literature.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from cambrian.agent import Agent
from cambrian.backends.base import LLMBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLAMP_LOW: float = 0.0
_CLAMP_HIGH: float = 1.0


def _fitness_of(agent: Agent) -> float:
    """Return the agent's fitness, treating ``None`` as ``0.0``.

    Args:
        agent: The agent whose fitness is requested.

    Returns:
        The agent's fitness as a float, defaulting to ``0.0`` when unset.
    """
    return agent.fitness if agent.fitness is not None else 0.0


def _clamp(value: float, low: float = _CLAMP_LOW, high: float = _CLAMP_HIGH) -> float:
    """Clamp *value* to [*low*, *high*].

    Args:
        value: The value to clamp.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        The clamped value.
    """
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# DPOPair
# ---------------------------------------------------------------------------


@dataclass
class DPOPair:
    """A single preference pair used by DPO-style optimisation.

    Attributes:
        chosen: The preferred agent (higher fitness).
        rejected: The dispreferred agent (lower fitness).
        task: The task description that was used to evaluate both agents.
        margin: Fitness difference ``chosen.fitness - rejected.fitness``,
            pre-computed at pair construction time.  Always non-negative.
    """

    chosen: Agent
    rejected: Agent
    task: str
    margin: float


# ---------------------------------------------------------------------------
# DPOSelector
# ---------------------------------------------------------------------------


class DPOSelector:
    """Builds preference pairs from a population and applies a DPO fitness bonus.

    The selector does **not** call any LLM backend — it works entirely with the
    existing fitness scores of agents in the population.

    Algorithm:

    1. Sort agents by fitness (descending).
    2. Build preferred/rejected pairs according to *pair_strategy*:
       - ``"adjacent"``: pair rank-0 with rank-1, rank-2 with rank-3, etc.
       - ``"random"``: shuffle the sorted list, then pair consecutive agents.
    3. For each pair compute a DPO reward and add it to both agents' fitness
       (``chosen`` gets the full reward, ``rejected`` gets no bonus so the gap
       widens relative to the base fitness scale).

    Args:
        beta: Controls the strength of the DPO reward.  Larger values amplify
            the bonus; typical range is [0.01, 1.0].  Default: ``0.1``.
        pair_strategy: Either ``"adjacent"`` or ``"random"``.  Default:
            ``"adjacent"``.

    Raises:
        ValueError: If *pair_strategy* is not one of the accepted values.
    """

    _VALID_STRATEGIES: frozenset[str] = frozenset({"adjacent", "random"})

    def __init__(self, beta: float = 0.1, pair_strategy: str = "adjacent") -> None:
        if pair_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"pair_strategy must be one of {sorted(self._VALID_STRATEGIES)}, "
                f"got {pair_strategy!r}"
            )
        self.beta = beta
        self.pair_strategy = pair_strategy

    def build_pairs(self, population: list[Agent], task: str) -> list[DPOPair]:
        """Construct preference pairs from *population*.

        Agents are sorted by fitness (descending) before pairing.  If the
        population has an odd number of agents the last agent is left unpaired.

        Args:
            population: The current population.  May be empty — returns ``[]``
                in that case.
            task: The task description associated with this evaluation round.

        Returns:
            A list of :class:`DPOPair` objects.  Length is
            ``len(population) // 2``.
        """
        if not population:
            return []

        sorted_pop = sorted(population, key=_fitness_of, reverse=True)

        if self.pair_strategy == "random":
            random.shuffle(sorted_pop)

        pairs: list[DPOPair] = []
        for i in range(0, len(sorted_pop) - 1, 2):
            a = sorted_pop[i]
            b = sorted_pop[i + 1]
            # Ensure chosen is always the higher-fitness agent of the two.
            if _fitness_of(a) >= _fitness_of(b):
                chosen, rejected = a, b
            else:
                chosen, rejected = b, a
            margin = _fitness_of(chosen) - _fitness_of(rejected)
            pairs.append(DPOPair(chosen=chosen, rejected=rejected, task=task, margin=margin))

        return pairs

    def compute_dpo_reward(self, pair: DPOPair) -> float:
        """Compute the DPO reward for a single preference pair.

        The reward is ``pair.margin * self.beta``, clamped to ``[0.0, 1.0]``.

        Args:
            pair: The preference pair to evaluate.

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        return _clamp(pair.margin * self.beta)

    def apply(self, population: list[Agent], task: str) -> list[Agent]:
        """Apply DPO fitness bonuses to all agents in *population*.

        For each preference pair the ``chosen`` agent receives a fitness bonus
        equal to ``compute_dpo_reward(pair)``.  The ``rejected`` agent receives
        no bonus.  Agents not part of any pair (e.g. the last agent in an
        odd-sized population) are left unchanged.

        Fitness is modified **in-place** on each :class:`~cambrian.agent.Agent`.
        Agents whose fitness was ``None`` are treated as ``0.0`` before the
        bonus is applied.

        Args:
            population: The current population (may be empty).
            task: The task description for this evaluation round.

        Returns:
            The same *population* list (modified in-place for convenience).
        """
        pairs = self.build_pairs(population, task)
        for pair in pairs:
            reward = self.compute_dpo_reward(pair)
            pair.chosen.fitness = _clamp(_fitness_of(pair.chosen) + reward)
            logger.debug(
                "DPOSelector: chosen=%s reward=%.4f  rejected=%s (no bonus)",
                pair.chosen.agent_id,
                reward,
                pair.rejected.agent_id,
            )
        return population


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------


class DPOTrainer:
    """Simulates a DPO training loop by refining low-performing agents via LLM.

    Unlike :class:`DPOSelector`, this class calls *backend* to rewrite the
    genome of each bottom-50% agent so that it more closely resembles the
    ``chosen`` agents in the collected pairs and avoids the ``rejected`` agents.

    If the backend call fails for any agent, that agent is silently cloned
    from its original state (i.e. its genome is preserved but a fresh
    ``agent_id`` is assigned) and a warning is logged.

    Args:
        backend: The LLM backend used to generate refined system prompts.
        beta: DPO reward strength passed to the internal :class:`DPOSelector`.
            Default: ``0.1``.
        n_refinements: Number of independent refinement passes per agent.
            The best-scoring rewrite is selected if fitness information is
            available; otherwise the last rewrite is used.  Default: ``3``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        beta: float = 0.1,
        n_refinements: int = 3,
    ) -> None:
        self.backend = backend
        self.beta = beta
        self.n_refinements = max(1, n_refinements)
        self._selector = DPOSelector(beta=beta, pair_strategy="adjacent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_pairs(self, population: list[Agent], task: str) -> list[DPOPair]:
        """Collect top-N vs bottom-N preference pairs from *population*.

        The population is sorted by fitness.  The top half becomes ``chosen``
        candidates and the bottom half becomes ``rejected`` candidates.  Pairs
        are formed by zipping the two halves (best chosen vs worst rejected,
        etc.).

        If the population has fewer than 2 agents, an empty list is returned.

        Args:
            population: The current population.
            task: The task description for this round.

        Returns:
            A list of :class:`DPOPair` objects of length
            ``len(population) // 2``.
        """
        if len(population) < 2:
            return []

        sorted_pop = sorted(population, key=_fitness_of, reverse=True)
        midpoint = len(sorted_pop) // 2
        top_half = sorted_pop[:midpoint]
        bottom_half = sorted_pop[midpoint:]

        pairs: list[DPOPair] = []
        for chosen, rejected in zip(top_half, bottom_half):
            margin = _fitness_of(chosen) - _fitness_of(rejected)
            pairs.append(DPOPair(chosen=chosen, rejected=rejected, task=task, margin=margin))

        return pairs

    def refine(self, agent: Agent, pairs: list[DPOPair], task: str) -> Agent:
        """Rewrite *agent*'s genome by learning from preference pairs.

        Constructs a refinement prompt that describes the patterns found in
        ``chosen`` agents' system prompts and asks the backend to rewrite
        *agent*'s system prompt to incorporate those patterns while avoiding
        the ``rejected`` patterns.

        If *pairs* is empty, returns a clone of *agent* unchanged.  If the
        backend raises any exception, a warning is logged and a clone of
        *agent* is returned as a fallback.

        Args:
            agent: The agent whose genome should be refined.
            pairs: The preference pairs to learn from.
            task: The task that was evaluated.

        Returns:
            A new :class:`Agent` with a potentially improved genome.
        """
        if not pairs:
            logger.debug("DPOTrainer.refine: no pairs — cloning agent %s", agent.agent_id)
            return agent.clone()

        chosen_prompts = "\n---\n".join(p.chosen.genome.system_prompt for p in pairs)
        rejected_prompts = "\n---\n".join(p.rejected.genome.system_prompt for p in pairs)

        refinement_prompt = (
            f"You are an expert AI prompt engineer.\n\n"
            f"Task: {task}\n\n"
            f"Below are PREFERRED system prompts that led to high performance:\n"
            f"<chosen>\n{chosen_prompts}\n</chosen>\n\n"
            f"Below are REJECTED system prompts that led to poor performance:\n"
            f"<rejected>\n{rejected_prompts}\n</rejected>\n\n"
            f"Current system prompt to refine:\n"
            f"<current>\n{agent.genome.system_prompt}\n</current>\n\n"
            f"Rewrite the current system prompt to incorporate the strengths of the "
            f"PREFERRED examples and avoid the weaknesses of the REJECTED examples. "
            f"Output ONLY the new system prompt text, no explanation, no XML tags."
        )

        best_prompt: str = agent.genome.system_prompt
        for attempt in range(self.n_refinements):
            try:
                result = self.backend.generate(refinement_prompt)
                result = result.strip()
                if result:
                    best_prompt = result
                    logger.debug(
                        "DPOTrainer.refine: attempt %d/%d succeeded for agent %s",
                        attempt + 1,
                        self.n_refinements,
                        agent.agent_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DPOTrainer.refine: backend error on attempt %d for agent %s: %s",
                    attempt + 1,
                    agent.agent_id,
                    exc,
                )
                break

        refined = agent.clone()
        refined.genome.system_prompt = best_prompt
        return refined

    def train(
        self,
        population: list[Agent],
        task: str,
        n_pairs: int = 3,
    ) -> list[Agent]:
        """Run one DPO training pass over *population*.

        Steps:

        1. Collect up to *n_pairs* preference pairs from the population.
        2. Identify the bottom 50% of agents (by fitness).
        3. Refine each bottom-50% agent using :meth:`refine`.
        4. Return the updated population (top-50% unchanged, bottom-50%
           replaced by their refined counterparts).

        If the population is empty, returns an empty list immediately.

        Args:
            population: The current population.
            task: The task description for this round.
            n_pairs: Maximum number of preference pairs to collect.  Default: ``3``.

        Returns:
            A new list containing the top-50% agents (unchanged) plus the
            refined bottom-50% agents.
        """
        if not population:
            return []

        all_pairs = self.collect_pairs(population, task)
        selected_pairs = all_pairs[:n_pairs] if len(all_pairs) > n_pairs else all_pairs

        sorted_pop = sorted(population, key=_fitness_of, reverse=True)
        midpoint = len(sorted_pop) // 2

        top_half = sorted_pop[:midpoint]
        bottom_half = sorted_pop[midpoint:]

        refined_bottom: list[Agent] = []
        for agent in bottom_half:
            refined = self.refine(agent, selected_pairs, task)
            refined_bottom.append(refined)

        result = list(top_half) + refined_bottom
        logger.info(
            "DPOTrainer.train: refined %d/%d agents using %d pairs",
            len(refined_bottom),
            len(population),
            len(selected_pairs),
        )
        return result
