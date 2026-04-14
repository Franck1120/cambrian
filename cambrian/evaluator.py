"""Abstract base class for agent evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cambrian.agent import Agent


class Evaluator(ABC):
    """Abstract evaluator that scores an agent's response to a task.

    Concrete subclasses must implement :meth:`evaluate` and return a scalar
    fitness score in the range ``[0.0, 1.0]``.

    The fitness score is the primary signal driving evolutionary selection:
    higher is better.
    """

    @abstractmethod
    def evaluate(self, agent: "Agent", task: str) -> float:
        """Evaluate *agent* on *task* and return a fitness score.

        Args:
            agent: The agent to evaluate. Its :meth:`~Agent.run` method will be
                called to obtain the agent's response.
            task: Natural-language task description.

        Returns:
            Fitness score in ``[0.0, 1.0]``.  Higher = better.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
