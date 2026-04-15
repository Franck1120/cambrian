"""Dream phase — GraphRAG-style recombination of lineage experiences.

Inspired by the role of REM sleep in memory consolidation and creative insight,
the dream phase periodically replays and recombines the highest-fitness genome
experiences stored in the :class:`~cambrian.memory.EvolutionaryMemory` lineage
graph.  This lets evolution escape local optima by synthesising novel prompts
that blend the best elements from diverse ancestral lines.

Algorithm
---------
1. Pull the top-N genome snapshots from the lineage graph (by fitness).
2. Build a *dream context*: a structured summary of what worked (key prompt
   phrases, strategies, temperatures) and what generation each insight came from.
3. Ask the LLM to generate *n_offspring* novel genomes inspired by the context.
4. Return the offspring — the caller injects them into the next population.

Integration
-----------
Call :meth:`DreamPhase.should_dream` in the generational loop to decide when
to dream.  A typical interval is every 5 generations::

    dream = DreamPhase(backend=backend, memory=memory)
    for gen in range(1, n_generations + 1):
        population = engine._next_generation(population, task)
        if dream.should_dream(gen):
            offspring = dream.dream(task, n_offspring=2)
            # Replace the two worst agents with dream offspring
            population.sort(key=lambda a: a.fitness or 0.0)
            for i, o in enumerate(offspring):
                population[i] = o
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.agent import Genome
    from cambrian.backends.base import LLMBackend
    from cambrian.memory import EvolutionaryMemory

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_DREAM_SYSTEM = """\
You are a creative AI researcher performing a "dream phase" — an offline
recombination of successful agent experiences to discover novel solutions.

You will receive a set of high-performing genome snapshots from a lineage graph.
Each snapshot contains a system prompt, strategy, temperature, and fitness score.

Your task: synthesise these experiences to generate NEW genome variants that
blend the best patterns from the ancestral line while exploring fresh directions.

For each new genome, return a JSON object in this exact schema:
{
  "system_prompt": "<the new system prompt>",
  "strategy": "<strategy hint: e.g. chain-of-thought, step-by-step, adversarial>",
  "temperature": <float in [0.0, 2.0]>
}

Return a JSON array of genome objects. Nothing else — no explanation, no markdown.
"""

_DREAM_TEMPLATE = """\
Task: {task}

Ancestral experiences (sorted by fitness, highest first):
{experiences}

Generate {n_offspring} novel genome variants inspired by these experiences.
Focus on what made the high-fitness agents successful and extrapolate further.
"""


# ─────────────────────────────────────────────────────────────────────────────
# DreamPhase
# ─────────────────────────────────────────────────────────────────────────────


class DreamPhase:
    """Recombines top lineage experiences to generate novel genome offspring.

    Args:
        backend: LLM backend for the dream generation call.
        memory: The :class:`~cambrian.memory.EvolutionaryMemory` lineage graph.
        top_n: Number of top ancestors to include in the dream context. Default ``5``.
        temperature: Sampling temperature for the dream call. Default ``0.9``.
        interval: How many generations between dream phases. Default ``5``.
        min_fitness: Only ancestors above this fitness are included. Default ``0.0``.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        memory: "EvolutionaryMemory",
        top_n: int = 5,
        temperature: float = 0.9,
        interval: int = 5,
        min_fitness: float = 0.0,
    ) -> None:
        self._backend = backend
        self._memory = memory
        self._top_n = top_n
        self._temp = temperature
        self._interval = interval
        self._min_fitness = min_fitness
        self._dream_count = 0

    def should_dream(self, generation: int) -> bool:
        """Return ``True`` if a dream phase should run at *generation*.

        A dream phase runs every :attr:`interval` generations starting from
        generation ``interval`` (never at generation 0).

        Args:
            generation: Current generation number (1-based).

        Returns:
            ``True`` when ``generation > 0`` and ``generation % interval == 0``.
        """
        return generation > 0 and self._interval > 0 and generation % self._interval == 0

    def dream(self, task: str, n_offspring: int = 3) -> "list[Genome]":
        """Generate novel genomes by recombining the top lineage experiences.

        Args:
            task: Task description used to contextualise the recombination.
            n_offspring: Number of new genomes to generate. Default ``3``.

        Returns:
            List of new :class:`~cambrian.agent.Genome` objects (not yet evaluated).
            Returns an empty list if the lineage is empty or the LLM call fails.
        """
        from cambrian.agent import Genome as _Genome

        top = self._memory.get_top_ancestors(n=self._top_n, min_fitness=self._min_fitness)
        if not top:
            logger.debug("DreamPhase: no ancestors in lineage — skipping")
            return []

        experiences = self._build_experience_text(top)
        prompt = _DREAM_TEMPLATE.format(
            task=task,
            experiences=experiences,
            n_offspring=n_offspring,
        )

        try:
            raw = self._backend.generate(prompt, system=_DREAM_SYSTEM, temperature=self._temp)
            offspring_dicts = self._parse_offspring(raw, n_offspring)
        except Exception as exc:
            logger.warning("DreamPhase LLM call failed: %s", exc)
            return []

        self._dream_count += 1
        genomes: list[_Genome] = []
        for d in offspring_dicts:
            g = _Genome(
                system_prompt=str(d.get("system_prompt", "")),
                strategy=str(d.get("strategy", "direct")),
                temperature=max(0.0, min(2.0, float(d.get("temperature", 0.7)))),
            )
            genomes.append(g)

        logger.info(
            "DreamPhase #%d: generated %d offspring from %d ancestors",
            self._dream_count, len(genomes), len(top),
        )
        return genomes

    @property
    def dream_count(self) -> int:
        """Total number of dream phases that have run."""
        return self._dream_count

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_experience_text(top: list[dict[str, Any]]) -> str:
        """Format the top ancestors into a human-readable experience log."""
        lines: list[str] = []
        for rank, ancestor in enumerate(top, 1):
            fitness = ancestor.get("fitness") or 0.0
            generation = ancestor.get("generation", "?")
            genome = ancestor.get("genome", {})
            prompt_preview = str(genome.get("system_prompt", "")).strip()[:200]
            strategy = genome.get("strategy", "")
            temperature = genome.get("temperature", "")
            lines.append(
                f"[{rank}] gen={generation} fitness={fitness:.4f} strategy={strategy!r} "
                f"temp={temperature}\n"
                f"    prompt: {prompt_preview!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_offspring(raw: str, n_offspring: int) -> list[dict[str, Any]]:
        """Parse a JSON array of genome dicts from the LLM response.

        Strips markdown fences if present.  Falls back to an empty list on
        any parse error.

        Args:
            raw: Raw LLM response string.
            n_offspring: Expected number of offspring (used for logging).

        Returns:
            List of genome dicts (may be shorter than *n_offspring*).
        """
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end])
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except Exception:
                pass

        # Fallback: try wrapping single object in a list
        start_obj = text.find("{")
        end_obj = text.rfind("}") + 1
        if start_obj != -1 and end_obj > start_obj:
            try:
                data = json.loads(text[start_obj:end_obj])
                if isinstance(data, dict):
                    return [data]
            except Exception:
                pass

        logger.warning("DreamPhase: failed to parse %d offspring from LLM response", n_offspring)
        return []
