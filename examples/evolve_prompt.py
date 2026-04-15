# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Evolve a system prompt for a specific persona or reasoning style.

Unlike code-evaluation examples, this one has no "correct" answer to
check with a sandbox. Instead it uses an LLM judge that scores the
agent's response on a custom rubric — a fully open-ended optimisation.

Scenario
--------
You want to build a "Socratic tutor" system prompt that helps students
discover answers themselves instead of giving them outright.  The judge
scores responses on three criteria:

  1. **Questioning quality** — does the response ask clarifying questions?
  2. **No direct answers** — does it avoid giving the answer away?
  3. **Engagement** — is the response encouraging and educational?

After evolution, you have an optimised system prompt you can drop into
your own product.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python examples/evolve_prompt.py

    # Save the winner:
    python examples/evolve_prompt.py --output tutor_prompt.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluator import Evaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Agent

logger = get_logger(__name__)


# ── Rubric ───────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are an expert in Socratic pedagogy. You rate AI tutor responses
on a scale from 0 to 10.

Scoring rubric:
  10  — Guides the student with probing questions, zero direct answers, highly encouraging.
  7-9 — Mostly Socratic, perhaps one minor direct hint, good engagement.
  5-6 — Mix of guidance and direct answers, acceptable but improvable.
  3-4 — Mostly gives answers directly, little questioning.
  0-2 — Entirely non-Socratic, lectures or ignores the student.

Return ONLY JSON: {"score": <0-10>, "reason": "<one sentence>"}"""

_JUDGE_TEMPLATE = """STUDENT QUESTION:
{student_question}

TUTOR RESPONSE:
{tutor_response}

Rate the tutor response using the Socratic pedagogy rubric."""

# Student questions used to probe the agent during evaluation
_STUDENT_QUESTIONS = [
    "What is the derivative of x²?",
    "Why does water boil at 100°C?",
    "Can you solve 2x + 4 = 10 for me?",
    "What causes inflation?",
    "How do I prove that √2 is irrational?",
]


# ── Custom evaluator ──────────────────────────────────────────────────────────

class SocraticEvaluator(Evaluator):
    """Evaluates agents on their Socratic tutoring quality.

    Sends several student questions to the agent and asks a judge LLM to
    score each response against a Socratic rubric.  The final score is the
    mean of all question-level scores.

    Args:
        judge_backend: LLM used as the judge (should be a capable model).
        n_questions: Number of student questions to sample per evaluation.
            Default ``3``.
        temperature: Temperature for the judge. Low values reduce variance.
    """

    def __init__(
        self,
        judge_backend: "OpenAICompatBackend",
        n_questions: int = 3,
        temperature: float = 0.1,
    ) -> None:
        self._judge = judge_backend
        self._n_q = min(n_questions, len(_STUDENT_QUESTIONS))
        self._temperature = temperature

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Score *agent* by judging its answers to *_n_q* student questions.

        Args:
            agent: Agent whose system prompt is being evaluated.
            task: Unused (task is determined by the student questions).

        Returns:
            Mean Socratic quality score normalised to ``[0.0, 1.0]``.
        """
        import re
        import random

        questions = random.sample(_STUDENT_QUESTIONS, self._n_q)
        scores: list[float] = []

        for q in questions:
            try:
                tutor_response = agent.run(q)
            except Exception:
                scores.append(0.0)
                continue

            prompt = _JUDGE_TEMPLATE.format(
                student_question=q, tutor_response=tutor_response
            )
            try:
                raw = self._judge.generate(
                    prompt,
                    system=_JUDGE_SYSTEM,
                    temperature=self._temperature,
                    max_tokens=128,
                )
            except Exception:
                scores.append(0.0)
                continue

            # Extract score from JSON or fallback to bare integer
            match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
            if not match:
                match = re.search(r'\b([0-9]|10)\b', raw)
            score = float(match.group(1)) / 10.0 if match else 0.0
            scores.append(min(max(score, 0.0), 1.0))
            logger.debug("  question=%r score=%.2f", q[:40], score)

        return sum(scores) / len(scores) if scores else 0.0


# ── Seed prompts (diverse starting population) ────────────────────────────────

_SEED_PROMPTS = [
    (
        "You are a Socratic tutor. Never give direct answers. "
        "Instead, ask questions that lead the student to discover the answer themselves. "
        "Be encouraging and patient."
    ),
    (
        "You are a thoughtful teacher who believes students learn best through guided discovery. "
        "Respond to questions with counter-questions, analogies, and hints. "
        "Celebrate curiosity. Never do the work for them."
    ),
    (
        "You are an educational coach using the Socratic method. "
        "Your responses should challenge assumptions, probe understanding, "
        "and guide students step-by-step without revealing solutions. "
        "Tone: warm, curious, encouraging."
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Run Socratic tutor system prompt evolution."""
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    agent_backend = OpenAICompatBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=60,
        max_retries=2,
    )
    judge_backend = OpenAICompatBackend(
        model=args.judge_model or args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=60,
        max_retries=2,
    )

    mutator = LLMMutator(
        backend=agent_backend,
        mutation_temperature=0.7,  # higher → more creative prompt mutations
    )
    evaluator = SocraticEvaluator(
        judge_backend=judge_backend,
        n_questions=args.n_questions,
    )

    engine = EvolutionEngine(
        evaluator=evaluator,
        mutator=mutator,
        backend=agent_backend,
        population_size=args.population,
        mutation_rate=0.9,
        crossover_rate=0.3,
        elite_ratio=0.2,
        tournament_k=3,
        seed=args.seed,
    )

    seed_genomes = [
        Genome(
            system_prompt=p,
            strategy="socratic",
            temperature=0.6,
            model=args.model,
        )
        for p in _SEED_PROMPTS
    ]

    task = "Answer student questions using the Socratic method."

    print("\nCambrian — Socratic Tutor Prompt Evolution")
    print(f"Agent model  : {args.model}")
    print(f"Judge model  : {args.judge_model or args.model}")
    print(f"Generations  : {args.generations}  |  Population: {args.population}")
    print(f"Questions/eval: {args.n_questions}")
    print()

    def _on_gen(gen: int, population: list) -> None:
        scores = [a.fitness or 0.0 for a in population]
        best = max(scores)
        mean = sum(scores) / len(scores)
        print(f"  Gen {gen:2d}  best={best:.4f}  mean={mean:.4f}")

    best = engine.evolve(
        seed_genomes=seed_genomes,
        task=task,
        n_generations=args.generations,
        on_generation=_on_gen,
    )

    print("\n" + "=" * 64)
    print("Evolution complete!")
    print(f"Best Socratic quality: {best.fitness:.4f} / 1.0")
    print(f"Model        : {best.genome.model}")
    print(f"Temperature  : {best.genome.temperature:.2f}")
    print("\nOptimised system prompt:\n")
    print(best.genome.system_prompt)
    print("=" * 64)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(best.genome.to_dict(), indent=2))
        print(f"\nBest genome saved to {out}")
        print("Usage:")
        print("  from cambrian.agent import Genome")
        print(f"  g = Genome.from_dict(json.load(open('{out}')))")
        print("  # Pass g.system_prompt to your chat API as the system message.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve a Socratic tutor system prompt"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Agent model")
    parser.add_argument("--judge-model", default=None, help="Judge model (defaults to --model)")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAMBRIAN_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--n-questions", type=int, default=3,
                        help="Student questions sampled per evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None, help="Save best genome JSON here")
    main(parser.parse_args())
