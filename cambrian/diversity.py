"""MAP-Elites — quality-diversity algorithm for agent populations.

MAP-Elites (Mouret & Clune, 2015) maintains a 2D grid where each cell holds
the single best agent for a particular (behaviour dimension 1, behaviour
dimension 2) combination. This ensures the population remains behaviourally
diverse even when the fitness landscape is dominated by one phenotype.

Behaviour dimensions used here:
- Axis 0: Prompt length bucket (short / medium / long)
- Axis 1: Temperature bucket (cold / warm / hot)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cambrian.agent import Agent


# Bucket boundaries for the two MAP-Elites dimensions
_PROMPT_LENGTH_BUCKETS = [0, 100, 300, float("inf")]   # tokens (rough chars/4)
_TEMPERATURE_BUCKETS = [0.0, 0.4, 0.8, 2.1]            # temperature ranges


def _prompt_bucket(system_prompt: str) -> int:
    """Return the 0-indexed bucket for a given prompt character count."""
    length = len(system_prompt) // 4  # rough token estimate
    for i, boundary in enumerate(_PROMPT_LENGTH_BUCKETS[1:]):
        if length < boundary:
            return i
    return len(_PROMPT_LENGTH_BUCKETS) - 2


def _temp_bucket(temperature: float) -> int:
    """Return the 0-indexed bucket for a given temperature."""
    for i, boundary in enumerate(_TEMPERATURE_BUCKETS[1:]):
        if temperature < boundary:
            return i
    return len(_TEMPERATURE_BUCKETS) - 2


class MAPElites:
    """Quality-diversity grid for Cambrian agent populations.

    Each cell in the 3×3 grid holds at most one agent — the best-fitness
    agent observed for that (prompt_length_bucket, temperature_bucket) pair.

    Agents with ``None`` fitness are not inserted.

    Args:
        n_prompt_buckets: Number of prompt-length buckets. Default 3.
        n_temp_buckets: Number of temperature buckets. Default 3.
    """

    def __init__(
        self,
        n_prompt_buckets: int = 3,
        n_temp_buckets: int = 3,
    ) -> None:
        self._n_p = n_prompt_buckets
        self._n_t = n_temp_buckets
        # grid[p_bucket][t_bucket] → Agent | None
        self._grid: list[list["Agent | None"]] = [
            [None] * n_temp_buckets for _ in range(n_prompt_buckets)
        ]
        self._total_added = 0

    def add(self, agent: "Agent") -> bool:
        """Attempt to insert *agent* into the MAP-Elites grid.

        Inserts if the cell is empty or if *agent* has higher fitness than
        the current occupant.

        Args:
            agent: The agent to insert. Must have a non-None ``fitness``.

        Returns:
            ``True`` if the agent was inserted (new elite), ``False`` otherwise.
        """
        if agent.fitness is None:
            return False

        p = min(_prompt_bucket(agent.genome.system_prompt), self._n_p - 1)
        t = min(_temp_bucket(agent.genome.temperature), self._n_t - 1)

        current = self._grid[p][t]
        if current is None or (current.fitness is None) or agent.fitness > current.fitness:
            self._grid[p][t] = agent
            self._total_added += 1
            return True
        return False

    def get_diverse_population(self) -> list["Agent"]:
        """Return one agent per occupied cell — maximally diverse set.

        Returns:
            List of elite agents (one per non-empty grid cell), sorted by
            descending fitness.
        """
        elites = [cell for row in self._grid for cell in row if cell is not None]
        elites.sort(key=lambda a: a.fitness or 0.0, reverse=True)
        return elites

    @property
    def occupancy(self) -> int:
        """Number of non-empty grid cells."""
        return sum(
            1 for row in self._grid for cell in row if cell is not None
        )

    @property
    def capacity(self) -> int:
        """Total number of grid cells."""
        return self._n_p * self._n_t

    def coverage(self) -> float:
        """Fraction of grid cells that are occupied (0.0–1.0)."""
        return self.occupancy / self.capacity

    def best(self) -> "Agent | None":
        """Return the elite agent with the highest fitness."""
        elites = self.get_diverse_population()
        return elites[0] if elites else None

    def __repr__(self) -> str:
        return (
            f"MAPElites(occupancy={self.occupancy}/{self.capacity}, "
            f"coverage={self.coverage():.1%})"
        )
