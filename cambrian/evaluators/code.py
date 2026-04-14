"""CodeEvaluator — runs agent-generated code in a sandbox and scores it.

Scoring heuristic:
- Code runs without error AND produces correct output → 1.0
- Code runs without error, wrong output → 0.5
- Code raises exception → 0.1
- Timed out → 0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator
from cambrian.utils.sandbox import run_in_sandbox, extract_python_code

if TYPE_CHECKING:
    from cambrian.agent import Agent


class CodeEvaluator(Evaluator):
    """Evaluates agents that produce executable Python code.

    The agent's response is parsed for a Python code block, executed in a
    subprocess sandbox with a hard timeout, and scored based on whether the
    code exits cleanly and (optionally) produces expected output.

    Args:
        expected_output: If provided, the sandbox stdout is compared against
            this string (stripped). Matching boosts the score to 1.0.
        timeout: Maximum seconds the subprocess may run. Default ``10``.
        partial_match: When True, a substring match on expected_output counts
            as a correct answer (score 1.0). Default ``False``.
    """

    def __init__(
        self,
        expected_output: str | None = None,
        timeout: float = 10.0,
        partial_match: bool = False,
    ) -> None:
        self._expected = expected_output.strip() if expected_output else None
        self._timeout = timeout
        self._partial_match = partial_match

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Run *agent* on *task*, execute the produced code, return a score.

        Scoring table:

        +---------------------+-------+
        | Outcome             | Score |
        +=====================+=======+
        | Timeout             | 0.0   |
        | Exception / crash   | 0.1   |
        | Runs, no expected   | 0.8   |
        | Partial line match  | 0.3–0.7 |
        | Exact match         | 1.0   |
        +---------------------+-------+

        Args:
            agent: Agent to evaluate. Its :meth:`~cambrian.agent.Agent.run`
                method is called with *task* to produce code.
            task: The coding task description.

        Returns:
            Fitness score in ``[0.0, 1.0]``.
        """
        response = agent.run(task)
        code = extract_python_code(response)

        if not code.strip():
            return 0.0

        result = run_in_sandbox(code, timeout=self._timeout)

        if result.timed_out:
            return 0.0

        if result.returncode != 0:
            # Penalise exceptions but give a small reward for trying
            return 0.1

        # Execution succeeded — check output if expected
        if self._expected is None:
            return 0.8  # Can't verify correctness, but it ran

        actual = result.stdout.strip()

        if self._partial_match:
            if self._expected in actual:
                return 1.0
        else:
            if actual == self._expected:
                return 1.0

        # Output didn't match
        # Partial credit: number of matching lines / total expected lines
        expected_lines = self._expected.splitlines()
        actual_lines = actual.splitlines()
        matching = sum(
            1 for e, a in zip(expected_lines, actual_lines) if e.strip() == a.strip()
        )
        if expected_lines:
            return 0.3 + 0.4 * (matching / len(expected_lines))
        return 0.5

    def __repr__(self) -> str:
        return (
            f"CodeEvaluator(expected={self._expected!r}, timeout={self._timeout})"
        )
