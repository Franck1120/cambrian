"""evolve_researcher.py — Advanced example: evolving a research agent.

This example evolves an AI agent specialised in researching a topic and
producing structured reports.  It demonstrates:

- Custom LLM-judge evaluator with a domain-specific rubric
- Lamarckian adapter that captures successful research patterns
- Stigmergy: high-scoring agents deposit pheromone traces
- Epigenetic layer adapted for research tasks
- Multi-objective Pareto: accuracy vs. conciseness trade-off
- Saving the best agent and exporting it as a standalone script

Usage:
    # Minimal — uses OPENAI_API_KEY from environment
    python examples/evolve_researcher.py

    # Full options
    python examples/evolve_researcher.py \\
        --topic "quantum computing applications in cryptography" \\
        --generations 15 \\
        --population 12 \\
        --model gpt-4o-mini \\
        --output researcher.json

Prerequisites:
    pip install cambrian
    export OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Bootstrap: allow running from repo root without installing ─────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from cambrian.agent import Agent, Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.epigenetics import EpigenomicContext, make_standard_layer
from cambrian.evaluators.composite import CompositeEvaluator
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evaluators.variance_aware import VarianceAwareEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.export import export_genome_json, export_standalone
from cambrian.lamarck import LamarckianAdapter
from cambrian.memory import EvolutionaryMemory
from cambrian.mutator import LLMMutator
from cambrian.pareto import ObjectiveVector, brevity_objective, fitness_objective, nsga2_select
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── Rubric ─────────────────────────────────────────────────────────────────────

RESEARCH_RUBRIC = """
You are evaluating a research report on the topic: {topic}

Score the report on a scale of 0.0 to 1.0 based on:
- Accuracy and factual correctness (30%)
- Depth and comprehensiveness of coverage (25%)
- Logical structure and clarity (20%)
- Citation of relevant concepts and sub-topics (15%)
- Conciseness and avoidance of padding (10%)

REPORT:
{response}

Respond with ONLY a decimal number between 0.0 and 1.0.
""".strip()


# ── Seed genomes ───────────────────────────────────────────────────────────────

SEED_GENOMES = [
    Genome(
        system_prompt=(
            "You are an expert research analyst. Given a topic, you produce "
            "a structured report covering: background, key concepts, current "
            "state of the art, challenges, and future directions. "
            "Be accurate, comprehensive, and concise."
        ),
        strategy="structured-report",
        temperature=0.4,
    ),
    Genome(
        system_prompt=(
            "You are a scientific writer specialising in technical synthesis. "
            "When given a research topic, you identify the most important "
            "sub-fields, key papers, and open problems. "
            "Use precise terminology and avoid vague generalisations."
        ),
        strategy="technical-synthesis",
        temperature=0.3,
    ),
    Genome(
        system_prompt=(
            "You are a generalist researcher. You produce clear, well-organised "
            "reports that are accessible to a technical audience. "
            "Start with a one-paragraph executive summary, then elaborate."
        ),
        strategy="executive-summary-first",
        temperature=0.5,
    ),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve a research agent.")
    parser.add_argument(
        "--topic", default="machine learning for drug discovery",
        help="Research topic to optimise for.",
    )
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default="researcher_best.json")
    parser.add_argument("--memory-out", default="researcher_lineage.json")
    parser.add_argument("--export-script", default=None,
                        help="Also export as a standalone Python script.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    topic: str = args.topic
    task = f"Research and write a comprehensive report on: {topic}"

    # ── Backends ──────────────────────────────────────────────────────────────
    backend = OpenAICompatBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=90,
        max_retries=3,
    )

    # ── Evaluator stack ───────────────────────────────────────────────────────
    # 1. LLM judge with domain-specific rubric
    judge_rubric = RESEARCH_RUBRIC.format(topic=topic, response="{response}")
    base_judge = LLMJudgeEvaluator(
        judge_backend=backend,
        rubric=judge_rubric,
    )

    # 2. Variance-aware: penalise inconsistent agents (anti-reward-hacking)
    variance_judge = VarianceAwareEvaluator(
        base_evaluator=base_judge,
        n_trials=2,
        variance_penalty=0.1,
    )

    # 3. Lamarckian: capture successful research patterns in the genome
    lamarck_evaluator = LamarckianAdapter(
        base_evaluator=variance_judge,
        capture_threshold=0.75,
        max_examples=3,
    )

    # ── Memory + stigmergy ────────────────────────────────────────────────────
    memory = EvolutionaryMemory(name=f"researcher_{topic[:20]}")

    # ── Mutator with stigmergy ────────────────────────────────────────────────
    mutator = LLMMutator(
        backend=backend,
        mutation_temperature=0.6,
        memory=memory,
        stigmergy_traces=3,
    )

    # ── Epigenetic layer ─────────────────────────────────────────────────────
    epi_layer = make_standard_layer()

    # ── Evolution engine ──────────────────────────────────────────────────────
    engine = EvolutionEngine(
        evaluator=lamarck_evaluator,
        mutator=mutator,
        backend=backend,
        population_size=args.population,
        elite_n=2,
        mutation_rate=0.8,
        crossover_rate=0.3,
        tournament_k=3,
        seed=args.seed,
        memory=memory,
    )

    # Track multi-objective vectors for post-hoc analysis
    objective_vectors: list[ObjectiveVector] = []

    def _on_generation(gen: int, population: list[Agent]) -> None:
        """Per-generation hook: apply epigenetics + collect Pareto vectors."""
        alive = [a for a in population if a.fitness is not None]
        best_fit = max((a.fitness or 0.0) for a in alive) if alive else 0.0
        mean_fit = sum(a.fitness or 0.0 for a in alive) / max(len(alive), 1)

        ctx = EpigenomicContext(
            generation=gen,
            task=topic,
            population_mean_fitness=mean_fit,
            population_best_fitness=best_fit,
            total_generations=args.generations,
        )

        # Apply epigenetic annotations (mutates runtime context, not genome)
        for agent in population:
            epi_layer.apply(agent, ctx)

        # Collect multi-objective vectors (fitness × brevity)
        for agent in population:
            if agent.fitness is not None:
                vec = ObjectiveVector(
                    agent_id=agent.id,
                    scores={
                        "fitness": fitness_objective(agent),
                        "brevity": brevity_objective(agent, max_tokens=500),
                    },
                )
                objective_vectors.append(vec)

        print(
            f"  Gen {gen:3d}  best={best_fit:.4f}  mean={mean_fit:.4f}  "
            f"pop={len(alive)}"
        )

    # ── Run evolution ─────────────────────────────────────────────────────────
    print(f"\nCambrian Research Agent Evolution")
    print(f"Topic     : {topic}")
    print(f"Model     : {args.model}")
    print(f"Generations: {args.generations}  Population: {args.population}")
    print("-" * 60)

    best = engine.evolve(
        seed_genomes=SEED_GENOMES,
        task=task,
        n_generations=args.generations,
        on_generation=_on_generation,
    )

    # ── Post-evolution Pareto analysis ────────────────────────────────────────
    print("\n--- Multi-objective Pareto Analysis ---")
    # Get current population from engine (last generation)
    last_pop = engine._population  # type: ignore[attr-defined]
    last_vecs = [
        ObjectiveVector(
            agent_id=a.id,
            scores={
                "fitness": fitness_objective(a),
                "brevity": brevity_objective(a, max_tokens=500),
            },
        )
        for a in last_pop if a.fitness is not None
    ]
    pareto_agents = nsga2_select(last_pop, last_vecs, target_size=3)
    print(f"Pareto-optimal agents (fitness × brevity): {len(pareto_agents)}")
    for pa in pareto_agents:
        print(f"  {pa.id[:10]}  fitness={pa.fitness:.4f}  "
              f"prompt_len={len(pa.genome.system_prompt)}")

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n--- Best Agent ---")
    print(f"Fitness  : {best.fitness:.4f}")
    print(f"Model    : {best.genome.model}")
    print(f"Strategy : {best.genome.strategy}")
    print(f"Prompt   :\n{best.genome.system_prompt[:300]}")

    # Save genome JSON
    export_genome_json(best, args.output)
    print(f"\nGenome saved to {args.output}")

    # Save lineage
    Path(args.memory_out).write_text(memory.to_json())
    print(f"Lineage saved to {args.memory_out}")

    # Optionally export standalone script
    if args.export_script:
        export_standalone(best, args.export_script)
        print(f"Standalone script exported to {args.export_script}")

    # ── Test the best agent ───────────────────────────────────────────────────
    print(f"\n--- Running best agent on task ---")
    response = best.run(task)
    print(response[:500] + ("..." if len(response) > 500 else ""))


if __name__ == "__main__":
    main()
