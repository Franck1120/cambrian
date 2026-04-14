"""Lamarckian evolution — write acquired behaviours back into the genome.

In standard Darwinian evolution the genome is fixed at birth; only offspring
inherit changes. Lamarck's theory (historically wrong for biology, useful for
AI) allows characteristics acquired *during an organism's lifetime* to be
passed to descendants.

In Cambrian's context, a "lifetime" is one evaluation episode. If an agent
achieves high fitness on a task, we record the (task, response, score) triple
as a *few-shot example* and embed it into the genome. Offspring produced by
mutating or crossing over that genome inherit the examples, giving the LLM
explicit proof of what the prompt can do — a form of in-context learning that
compounds across generations.
"""

from __future__ import annotations

from typing import Any, Callable

from cambrian.agent import Agent


class LamarckianAdapter:
    """Wraps an evaluator to capture successful (task, response, score) triples.

    When an agent's fitness exceeds *capture_threshold*, the adapter attempts
    to record its most recent task/response pair as a few-shot example in the
    genome.  Up to *max_examples* examples are retained; lower-scoring examples
    are evicted when the list is full.

    Args:
        base_evaluator: The underlying ``(agent, task) -> float`` callable.
        capture_threshold: Minimum fitness to qualify as a "success" worth
            recording.  Default ``0.7``.
        max_examples: Maximum few-shot examples kept per genome. Default ``3``.

    Example::

        from cambrian.lamarck import LamarckianAdapter
        from cambrian.evaluators.code import CodeEvaluator

        inner = CodeEvaluator(expected_output="Hello, world!")
        lamarck = LamarckianAdapter(inner, capture_threshold=0.8)
        engine = EvolutionEngine(evaluator=lamarck, ...)
    """

    def __init__(
        self,
        base_evaluator: Callable[[Agent, str], float],
        capture_threshold: float = 0.7,
        max_examples: int = 3,
    ) -> None:
        self._base = base_evaluator
        self._threshold = capture_threshold
        self._max = max_examples
        # Track last response per agent for capture; populated by run-wrapping
        self._last_response: dict[str, str] = {}

    # ── Public helpers ─────────────────────────────────────────────────────────

    def record_response(self, agent_id: str, response: str) -> None:
        """Store the most recent response for *agent_id* so it can be captured.

        Call this after ``agent.run(task)`` if you want responses to be
        included in few-shot examples.  The evaluator alone can only record
        task + score; the response must be supplied externally.

        Args:
            agent_id: The agent whose response to record.
            response: The text response produced by the agent.
        """
        self._last_response[agent_id] = response[:500]  # cap length

    def __call__(self, agent: Agent, task: str) -> float:
        """Evaluate *agent* and, if successful, write the example into its genome.

        Args:
            agent: The agent to evaluate.
            task: The task description.

        Returns:
            The fitness score from the base evaluator.
        """
        score = self._base(agent, task)

        if score >= self._threshold:
            self._capture(agent, task, score)

        return score

    # ── Internals ──────────────────────────────────────────────────────────────

    def _capture(self, agent: Agent, task: str, score: float) -> None:
        """Write a successful (task, response, score) example into the genome."""
        response = self._last_response.get(agent.agent_id, "")
        example: dict[str, Any] = {
            "task": task[:200],
            "score": round(score, 4),
            "response": response,
        }

        examples = list(agent.genome.few_shot_examples)

        # Skip exact duplicate tasks
        if any(ex.get("task") == example["task"] for ex in examples):
            return

        examples.append(example)
        # Keep only the highest-scoring ones up to max_examples
        examples.sort(key=lambda e: e.get("score", 0.0), reverse=True)
        agent.genome.few_shot_examples = examples[: self._max]

    def __repr__(self) -> str:
        return (
            f"LamarckianAdapter(threshold={self._threshold}, "
            f"max_examples={self._max}, base={self._base!r})"
        )
