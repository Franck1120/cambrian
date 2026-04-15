# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""LLMJudgeEvaluator — uses an LLM to score agent responses on a 0-10 scale.

The judge is given the original task, the agent's response, and a rubric
prompt, then parses the numeric rating from the LLM's output.

This evaluator is model-agnostic: it accepts any :class:`~cambrian.backends.base.LLMBackend`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cambrian.evaluator import Evaluator
from cambrian.backends.base import LLMBackend

if TYPE_CHECKING:
    from cambrian.agent import Agent


_JUDGE_SYSTEM_PROMPT = """You are an impartial AI evaluator. Your job is to score an AI agent's
response to a given task on a scale from 0 to 10.

Scoring rubric:
- 10: Perfect. Completely solves the task, no errors, excellent quality.
- 8-9: Very good. Solves the task with minor issues.
- 6-7: Adequate. Mostly correct but has notable gaps or errors.
- 4-5: Partial. Attempts the task but significant issues remain.
- 2-3: Poor. Barely addresses the task.
- 0-1: Fail. Wrong, harmful, or completely off-topic.

Respond with ONLY a JSON object: {"score": <0-10>, "reason": "<one sentence>"}"""

_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+(?:\.\d+)?)')


class LLMJudgeEvaluator(Evaluator):
    """Evaluator that uses an LLM to judge agent output quality.

    Useful when "correct" output cannot be determined algorithmically (e.g.
    open-ended writing, reasoning tasks, strategy explanations).

    Args:
        judge_backend: The LLM backend to use for judging. Should be a
            capable model (e.g. GPT-4o) to ensure reliable scoring.
        rubric_extension: Optional additional rubric instructions appended to
            the default judge system prompt.
        temperature: Judge temperature. Low values (0.1–0.2) reduce scoring
            variance. Default ``0.1``.
    """

    def __init__(
        self,
        judge_backend: LLMBackend,
        rubric_extension: str = "",
        temperature: float = 0.1,
    ) -> None:
        self._judge = judge_backend
        self._rubric_extension = rubric_extension
        self._temperature = temperature

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Score the agent's response on *task* using the judge LLM.

        Returns a normalised score in ``[0.0, 1.0]``.
        """
        response = agent.run(task)

        system = _JUDGE_SYSTEM_PROMPT
        if self._rubric_extension:
            system += f"\n\nAdditional criteria:\n{self._rubric_extension}"

        prompt = (
            f"TASK:\n{task}\n\n"
            f"AGENT RESPONSE:\n{response}\n\n"
            "Rate this response using the rubric above. "
            'Return ONLY JSON: {"score": <0-10>, "reason": "<one sentence>"}'
        )

        try:
            raw = self._judge.generate(
                prompt,
                system=system,
                temperature=self._temperature,
                max_tokens=128,
            )
        except Exception:
            return 0.0

        match = _SCORE_RE.search(raw)
        if not match:
            # Try to extract any standalone integer 0–10
            fallback = re.search(r"\b([0-9]|10)\b", raw)
            if fallback:
                score = float(fallback.group(1))
            else:
                return 0.0
        else:
            score = float(match.group(1))

        return min(max(score / 10.0, 0.0), 1.0)

    def __repr__(self) -> str:
        return f"LLMJudgeEvaluator(judge={self._judge!r})"
