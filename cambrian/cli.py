"""Cambrian CLI — command-line interface for running evolutionary experiments.

Commands
--------
evolve      Run an evolutionary search for a given task.
dashboard   Show live stats for a running or completed run.
distill     Extract and display the best genome from a lineage JSON file.
version     Print the Cambrian version.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

from cambrian import __version__
from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)
console = Console() if _RICH else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_backend(model: str, base_url: str, api_key: str) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=60,
        max_retries=3,
    )


def _make_evaluator(
    task: str, judge_backend: OpenAICompatBackend | None = None
) -> Any:
    """Return a simple evaluator: LLMJudge if judge backend available, else stub."""
    if judge_backend is not None:
        from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
        return LLMJudgeEvaluator(backend=judge_backend, task_description=task)

    # Minimal stub — returns 0 so the loop still runs during demos / tests
    def _stub(agent: Any, _task: str) -> float:  # noqa: ANN001
        logger.warning("No evaluator configured — returning 0.0 fitness.")
        return 0.0

    return _stub


def _rich_gen_table(gen: int, population: list[Any]) -> "Table":
    """Build a Rich table summarising a generation."""
    table = Table(title=f"Generation {gen}", show_header=True, header_style="bold cyan")
    table.add_column("Agent", style="dim", width=10)
    table.add_column("Fitness", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("Strategy", width=14)
    table.add_column("Prompt (preview)", width=40)

    for a in sorted(population, key=lambda x: x.fitness or 0.0, reverse=True)[:8]:
        fitness_str = f"{a.fitness:.4f}" if a.fitness is not None else "–"
        prompt_preview = (a.genome.system_prompt[:37] + "…") if len(
            a.genome.system_prompt
        ) > 40 else a.genome.system_prompt
        table.add_row(
            a.id[:8],
            fitness_str,
            f"{a.genome.temperature:.2f}",
            a.genome.strategy,
            prompt_preview,
        )
    return table


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
def main() -> None:
    """Cambrian — LLM-guided evolutionary agent optimisation."""


# ── evolve ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("task")
@click.option(
    "--model", "-m",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model used for agents and mutator.",
)
@click.option(
    "--judge-model",
    default=None,
    help="Separate model for the LLM judge evaluator. Defaults to --model.",
)
@click.option(
    "--base-url",
    default="https://api.openai.com/v1",
    show_default=True,
    envvar="CAMBRIAN_BASE_URL",
    help="OpenAI-compatible API base URL.",
)
@click.option(
    "--api-key",
    default=None,
    envvar="OPENAI_API_KEY",
    help="API key. Falls back to OPENAI_API_KEY env var.",
)
@click.option(
    "--generations", "-g", default=10, show_default=True,
    help="Number of generations to run.",
)
@click.option(
    "--population", "-p", default=8, show_default=True,
    help="Population size per generation.",
)
@click.option(
    "--mutation-rate", default=0.8, show_default=True,
    help="Probability of mutating each non-elite agent.",
)
@click.option(
    "--crossover-rate", default=0.3, show_default=True,
    help="Probability of crossover vs. direct mutation.",
)
@click.option(
    "--mutation-temp", default=0.6, show_default=True,
    help="Temperature used by the mutator LLM call.",
)
@click.option(
    "--seed-prompt", default=None,
    help="Initial system prompt. Defaults to a generic problem-solver.",
)
@click.option(
    "--output", "-o", default=None,
    help="Path to write the best genome JSON at the end.",
)
@click.option(
    "--memory-out", default=None,
    help="Path to write the full lineage graph JSON.",
)
@click.option(
    "--seed", default=None, type=int,
    help="Random seed for reproducibility.",
)
def evolve(
    task: str,
    model: str,
    judge_model: str | None,
    base_url: str,
    api_key: str | None,
    generations: int,
    population: int,
    mutation_rate: float,
    crossover_rate: float,
    mutation_temp: float,
    seed_prompt: str | None,
    output: str | None,
    memory_out: str | None,
    seed: int | None,
) -> None:
    """Run evolutionary optimisation for TASK.

    TASK is a natural-language description of the problem the agents must solve.

    \b
    Example:
        cambrian evolve "Write a Python function that returns the nth Fibonacci number"
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        click.echo(
            "Error: no API key. Set OPENAI_API_KEY or pass --api-key.", err=True
        )
        sys.exit(1)

    backend = _make_backend(model, base_url, api_key)
    judge_backend = _make_backend(
        judge_model or model, base_url, api_key
    )

    mutator = LLMMutator(backend=backend, mutation_temperature=mutation_temp)
    evaluator = _make_evaluator(task, judge_backend)

    engine = EvolutionEngine(
        evaluator=evaluator,
        mutator=mutator,
        population_size=population,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        seed=seed,
    )

    initial_prompt = seed_prompt or (
        "You are a precise and knowledgeable AI assistant. "
        "You solve problems step-by-step and verify your answers."
    )
    seed_genome = Genome(
        system_prompt=initial_prompt,
        tools=[],
        strategy="chain-of-thought",
        temperature=0.7,
        model=model,
    )

    live_table: Any = None

    def _on_gen(gen: int, pop: list[Any]) -> None:
        nonlocal live_table
        if _RICH and console is not None:
            console.print(_rich_gen_table(gen, pop))

    click.echo(f"Cambrian v{__version__} — evolving for {generations} generations")
    click.echo(f"Task: {task[:80]}")
    click.echo(f"Model: {model}  |  Population: {population}  |  Seed: {seed}")
    click.echo("")

    best = engine.evolve(
        seed_genomes=[seed_genome],
        task=task,
        n_generations=generations,
        on_generation=_on_gen,
    )

    if _RICH and console is not None:
        console.rule("[bold green]Evolution complete")
        console.print(Panel(
            f"[bold]Best fitness:[/bold] {best.fitness:.4f}\n"
            f"[bold]Model:[/bold] {best.genome.model}\n"
            f"[bold]Temperature:[/bold] {best.genome.temperature}\n"
            f"[bold]Strategy:[/bold] {best.genome.strategy}\n\n"
            f"[bold]System prompt:[/bold]\n{best.genome.system_prompt}",
            title="Best Genome",
            border_style="green",
        ))
    else:
        click.echo(f"\nBest fitness: {best.fitness:.4f}")
        click.echo(f"System prompt: {best.genome.system_prompt[:200]}")

    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(best.genome.to_dict(), indent=2))
        click.echo(f"Best genome written to {out_path}")

    if memory_out:
        mem_path = Path(memory_out)
        mem_path.write_text(engine.memory.to_json())
        click.echo(f"Lineage graph written to {mem_path}")


# ── dashboard ─────────────────────────────────────────────────────────────────

@main.command()
@click.argument("memory_file", type=click.Path(exists=True))
def dashboard(memory_file: str) -> None:
    """Display statistics from a saved lineage JSON file.

    MEMORY_FILE is the path written by --memory-out during an evolve run.
    """
    from cambrian.memory import EvolutionaryMemory

    data = Path(memory_file).read_text()
    mem = EvolutionaryMemory.from_json(data)
    stats = mem.generation_stats()

    if _RICH and console is not None:
        table = Table(title=f"Run: {mem.name}", show_header=True, header_style="bold magenta")
        table.add_column("Generation", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Bar")

        for gen, s in sorted(stats.items()):
            bar_len = int(s["best"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            table.add_row(
                str(gen),
                str(int(s["count"])),
                f"{s['best']:.4f}",
                f"{s['mean']:.4f}",
                bar,
            )
        console.print(table)

        top = mem.get_top_ancestors(n=3)
        if top:
            console.print(Panel(
                "\n".join(
                    f"[bold]{a['agent_id'][:10]}[/bold]  fitness={a.get('fitness', 0):.4f}"
                    f"  gen={a.get('generation', '?')}"
                    for a in top
                ),
                title="Top 3 Ancestors",
                border_style="cyan",
            ))
    else:
        for gen, s in sorted(stats.items()):
            click.echo(f"Gen {gen:3d}  count={int(s['count'])}  best={s['best']:.4f}  mean={s['mean']:.4f}")


# ── distill ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("genome_file", type=click.Path(exists=True))
def distill(genome_file: str) -> None:
    """Pretty-print a saved genome JSON file.

    GENOME_FILE is the path written by --output during an evolve run.
    """
    data = json.loads(Path(genome_file).read_text())
    genome = Genome.from_dict(data)

    if _RICH and console is not None:
        console.print(Panel(
            f"[bold]Model:[/bold] {genome.model}\n"
            f"[bold]Temperature:[/bold] {genome.temperature}\n"
            f"[bold]Strategy:[/bold] {genome.strategy}\n"
            f"[bold]Tools:[/bold] {', '.join(genome.tools) or 'none'}\n\n"
            f"[bold]System Prompt:[/bold]\n{genome.system_prompt}",
            title=f"Genome — {genome_file}",
            border_style="blue",
        ))
    else:
        click.echo(json.dumps(data, indent=2))


# ── version ───────────────────────────────────────────────────────────────────

@main.command()
def version() -> None:
    """Print Cambrian version."""
    click.echo(f"Cambrian {__version__}")
