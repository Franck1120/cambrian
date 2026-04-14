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
        return LLMJudgeEvaluator(judge_backend=judge_backend)

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
@click.option(
    "--tournament-k", default=3, show_default=True,
    help="Tournament size for parent selection.",
)
@click.option(
    "--compress-every", default=0, show_default=True,
    help="Apply prompt compression every N generations (0 = off).",
)
@click.option(
    "--compress-tokens", default=256, show_default=True,
    help="Max token budget when compressing prompts.",
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
    tournament_k: int,
    compress_every: int,
    compress_tokens: int,
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
        backend=backend,
        population_size=population,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_k=tournament_k,
        seed=seed,
        compress_interval=compress_every,
        compress_max_tokens=compress_tokens,
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


# ── stats (legacy text summary) ───────────────────────────────────────────────

@main.command()
@click.argument("memory_file", type=click.Path(exists=True))
def stats(memory_file: str) -> None:
    """Display text statistics from a saved lineage JSON file.

    MEMORY_FILE is the path written by --memory-out during an evolve run.
    For a live Streamlit dashboard use: cambrian dashboard --log-file FILE
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


# ── analyze ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("memory_file", type=click.Path(exists=True))
@click.option(
    "--top", default=5, show_default=True,
    help="Number of top agents to show in the lineage section.",
)
def analyze(memory_file: str, top: int) -> None:
    """Deep analysis of an evolution run from a lineage JSON file.

    MEMORY_FILE is the path written by --memory-out during an evolve run.

    Shows:
    \b
      - Per-generation fitness trajectory (best + mean)
      - Diversity metrics (unique strategies, temperature spread)
      - Full lineage of the best agent
      - Stagnation detection
    """
    import statistics as _stats

    from cambrian.memory import EvolutionaryMemory

    data = Path(memory_file).read_text()
    mem = EvolutionaryMemory.from_json(data)
    gen_stats = mem.generation_stats()

    if not gen_stats:
        click.echo("No data in lineage file.", err=True)
        return

    # ── Fitness trajectory ────────────────────────────────────────────────────
    if _RICH and console is not None:
        tbl = Table(
            title=f"Evolution Analysis — {mem.name}",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Gen", justify="right")
        tbl.add_column("Best", justify="right")
        tbl.add_column("Mean", justify="right")
        tbl.add_column("Delta", justify="right")
        tbl.add_column("Agents", justify="right")
        tbl.add_column("Trend")

        prev_best = 0.0
        for gen, s in sorted(gen_stats.items()):
            best = s["best"]
            delta = best - prev_best
            delta_str = f"[green]+{delta:.4f}[/]" if delta > 0 else f"[red]{delta:.4f}[/]"
            bar_len = int(best * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            tbl.add_row(
                str(gen),
                f"{best:.4f}",
                f"{s['mean']:.4f}",
                delta_str,
                str(int(s["count"])),
                bar,
            )
            prev_best = best

        console.print(tbl)
    else:
        click.echo(f"\nEvolution Analysis — {mem.name}")
        click.echo("-" * 56)
        prev_best = 0.0
        for gen, s in sorted(gen_stats.items()):
            delta = s["best"] - prev_best
            click.echo(
                f"Gen {gen:3d}  best={s['best']:.4f}  mean={s['mean']:.4f}"
                f"  delta={delta:+.4f}  n={int(s['count'])}"
            )
            prev_best = s["best"]

    # ── Diversity metrics ─────────────────────────────────────────────────────
    all_agents = list(mem._graph.nodes(data=True))
    strategies: list[str] = []
    temperatures: list[float] = []
    prompt_lengths: list[int] = []

    for _, attrs in all_agents:
        g = attrs.get("genome") or {}
        if isinstance(g, dict):
            s = g.get("strategy", "")
            if s:
                strategies.append(s)
            t = g.get("temperature")
            if t is not None:
                temperatures.append(float(t))
            p = g.get("system_prompt", "")
            if p:
                prompt_lengths.append(len(p) // 4)

    diversity_lines = []
    unique_strats = set(strategies)
    diversity_lines.append(f"Unique strategies : {', '.join(sorted(unique_strats)) or 'n/a'}")
    if temperatures:
        diversity_lines.append(
            f"Temperature range : {min(temperatures):.2f} – {max(temperatures):.2f}"
            f"  (mean={_stats.mean(temperatures):.2f}"
            + (f", stdev={_stats.stdev(temperatures):.2f}" if len(temperatures) > 1 else "")
            + ")"
        )
    if prompt_lengths:
        diversity_lines.append(
            f"Prompt tokens     : {min(prompt_lengths)} – {max(prompt_lengths)}"
            f"  (mean={int(_stats.mean(prompt_lengths))})"
        )

    # Stagnation: count gens where best didn't improve by >0.001
    gens_sorted = sorted(gen_stats.items())
    stagnant = sum(
        1 for i in range(1, len(gens_sorted))
        if gens_sorted[i][1]["best"] - gens_sorted[i - 1][1]["best"] < 0.001
    )
    diversity_lines.append(
        f"Stagnant gens     : {stagnant}/{len(gens_sorted) - 1}"
        + (" [WARNING: consider increasing mutation rate]" if stagnant > len(gens_sorted) // 2 else "")
    )

    if _RICH and console is not None:
        console.print(Panel("\n".join(diversity_lines), title="Diversity Metrics", border_style="yellow"))
    else:
        click.echo("\nDiversity Metrics")
        click.echo("-" * 40)
        for line in diversity_lines:
            click.echo(f"  {line}")

    # ── Lineage of best agent ─────────────────────────────────────────────────
    top_agents = mem.get_top_ancestors(n=top)
    if top_agents:
        best_id = top_agents[0]["agent_id"]
        lineage = mem.get_lineage(best_id)
        lineage_str = " → ".join(aid[:8] for aid in lineage)

        if _RICH and console is not None:
            console.print(Panel(
                f"[bold]Best agent:[/bold] {best_id[:10]}  "
                f"fitness={top_agents[0].get('fitness', 0):.4f}\n\n"
                f"[bold]Lineage ({len(lineage)} ancestors):[/bold]\n{lineage_str}",
                title=f"Top {top} Agents & Lineage",
                border_style="green",
            ))

            top_tbl = Table(show_header=True, header_style="bold magenta")
            top_tbl.add_column("Rank")
            top_tbl.add_column("Agent ID")
            top_tbl.add_column("Fitness", justify="right")
            top_tbl.add_column("Generation", justify="right")
            for rank, a in enumerate(top_agents, 1):
                top_tbl.add_row(
                    str(rank),
                    a["agent_id"][:12],
                    f"{a.get('fitness', 0):.4f}",
                    str(a.get("generation", "?")),
                )
            console.print(top_tbl)
        else:
            click.echo(f"\nBest agent: {best_id[:10]}  fitness={top_agents[0].get('fitness', 0):.4f}")
            click.echo(f"Lineage ({len(lineage)} ancestors): {lineage_str}")
            click.echo("\nTop agents:")
            for rank, a in enumerate(top_agents, 1):
                click.echo(
                    f"  {rank}. {a['agent_id'][:10]}  "
                    f"fitness={a.get('fitness', 0):.4f}  "
                    f"gen={a.get('generation', '?')}"
                )


# ── distill-agent ─────────────────────────────────────────────────────────────

@main.command("distill-agent")
@click.option(
    "--agent", "agent_file", required=True, type=click.Path(exists=True),
    help="Path to the evolved genome JSON (written by --output).",
)
@click.option(
    "--target", required=True,
    help="Target model identifier for the distilled agent (e.g. gemma-4-12b, llama3.2).",
)
@click.option(
    "--base-url",
    default="https://api.openai.com/v1",
    show_default=True,
    envvar="CAMBRIAN_BASE_URL",
    help="OpenAI-compatible API base URL for the distillation backend.",
)
@click.option(
    "--api-key", default=None, envvar="OPENAI_API_KEY",
    help="API key for the distillation backend.",
)
@click.option(
    "--output", "-o", default=None,
    help="Path to save the distilled genome JSON. Defaults to <agent_file>.distilled.json",
)
@click.option(
    "--max-tokens", default=150, show_default=True,
    help="Target prompt length (in tokens) for the distilled agent.",
)
def distill_agent(
    agent_file: str,
    target: str,
    base_url: str,
    api_key: str | None,
    output: str | None,
    max_tokens: int,
) -> None:
    """Distil an evolved genome to run efficiently on a smaller target model.

    Reads the agent genome from AGENT_FILE, uses an LLM to compress and
    adapt the system prompt for the TARGET model's capabilities, then saves
    the distilled genome.

    \\b
    Example:
        cambrian distill-agent --agent best.json --target gemma-4-12b --max-tokens 120
    """
    from cambrian.compress import caveman_compress, procut_prune

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    data = json.loads(Path(agent_file).read_text())
    from cambrian.agent import Genome as _Genome
    source_genome = _Genome.from_dict(data)

    click.echo(f"Source genome: model={source_genome.model}  tokens={source_genome.token_count()}")
    click.echo(f"Target model : {target}  max_tokens={max_tokens}")
    click.echo("")

    distilled_genome = _Genome.from_dict(source_genome.to_dict())
    distilled_genome.model = target
    distilled_genome.genome_id = source_genome.genome_id + "-d"

    # Step 1: structural compression (remove stopwords + filler)
    compressed_prompt = caveman_compress(source_genome.system_prompt)
    distilled_genome.system_prompt = compressed_prompt

    # Step 2: paragraph pruning to budget
    distilled_genome = procut_prune(distilled_genome, max_tokens=max_tokens)

    # Step 3: LLM-assisted adaptation (if backend available)
    if api_key:
        try:
            backend = _make_backend(target, base_url, api_key)
            adapt_prompt = (
                f"You are adapting a system prompt originally written for a large LLM "
                f"({source_genome.model}) to work well on a smaller model ({target}).\n\n"
                f"Original prompt:\n{source_genome.system_prompt}\n\n"
                f"Compressed draft:\n{distilled_genome.system_prompt}\n\n"
                f"Requirements:\n"
                f"- Keep under {max_tokens} tokens (~{max_tokens * 4} characters)\n"
                f"- Preserve the core intent and instructions\n"
                f"- Use simple, direct language suitable for a smaller model\n"
                f"- Remove meta-commentary, verbose explanations, nested conditionals\n\n"
                "Output ONLY the adapted system prompt, no preamble."
            )
            adapted = backend.generate(adapt_prompt, temperature=0.2)
            if adapted and len(adapted) // 4 <= max_tokens * 1.5:
                distilled_genome.system_prompt = adapted.strip()
                click.echo("LLM-assisted adaptation applied.")
        except Exception as exc:
            click.echo(f"LLM adaptation skipped ({exc.__class__.__name__}): {exc}", err=True)
    else:
        click.echo("No API key: using structural compression only.")

    # Clamp temperature for smaller models (they're more sensitive)
    if distilled_genome.temperature > 0.9:
        distilled_genome.temperature = min(distilled_genome.temperature, 0.8)

    final_tokens = distilled_genome.token_count()
    reduction = (1 - final_tokens / max(source_genome.token_count(), 1)) * 100

    if _RICH and console is not None:
        console.print(Panel(
            f"[bold]Source:[/bold] {source_genome.model}  {source_genome.token_count()} tokens\n"
            f"[bold]Target:[/bold] {target}  {final_tokens} tokens  "
            f"([green]-{reduction:.0f}%[/green] reduction)\n\n"
            f"[bold]Distilled prompt:[/bold]\n{distilled_genome.system_prompt}",
            title="Distillation Complete",
            border_style="green",
        ))
    else:
        click.echo(f"Tokens: {source_genome.token_count()} → {final_tokens} (-{reduction:.0f}%)")
        click.echo(f"Distilled prompt: {distilled_genome.system_prompt[:200]}")

    out_path = output or agent_file.replace(".json", f".distilled.{target}.json")
    Path(out_path).write_text(json.dumps(distilled_genome.to_dict(), indent=2))
    click.echo(f"\nDistilled genome saved to {out_path}")


# ── dashboard ─────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--port", default=8501, show_default=True,
    help="TCP port for the Streamlit server.",
)
@click.option(
    "--log-file", default="cambrian_log.json", show_default=True,
    help="Path to the evolution log JSON written by the evolution engine.",
)
@click.option(
    "--no-browser", is_flag=True, default=False,
    help="Do not open a browser tab automatically.",
)
def dashboard(port: int, log_file: str, no_browser: bool) -> None:
    """Launch the Streamlit live evolution dashboard.

    Reads LOG_FILE (a JSON log written by the evolution engine) and displays
    fitness trajectory, top agents, fitness landscape, and more.

    \\b
    Example:
        cambrian dashboard --port 8501 --log-file my_run.json
    """
    try:
        from cambrian.dashboard import run_dashboard
        click.echo(f"Starting dashboard on http://localhost:{port} ...")
        run_dashboard(port=port, log_file=log_file, open_browser=not no_browser)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


# ── run ───────────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--agent", "agent_file", required=True, type=click.Path(exists=True),
    help="Path to an evolved genome JSON file (written by --output or export_genome_json).",
)
@click.argument("task")
@click.option(
    "--base-url",
    default="https://api.openai.com/v1",
    show_default=True,
    envvar="CAMBRIAN_BASE_URL",
    help="OpenAI-compatible API base URL.",
)
@click.option(
    "--api-key", default=None, envvar="OPENAI_API_KEY",
    help="API key. Falls back to OPENAI_API_KEY env var.",
)
@click.option(
    "--model", default=None,
    help="Override the genome's model (e.g. to route to a different backend).",
)
@click.option(
    "--temperature", default=None, type=float,
    help="Override the genome's sampling temperature.",
)
@click.option(
    "--max-tokens", default=1024, show_default=True,
    help="Max tokens for the agent response.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format: plain text or JSON envelope.",
)
def run(
    agent_file: str,
    task: str,
    base_url: str,
    api_key: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int,
    output_format: str,
) -> None:
    """Run an evolved agent on TASK.

    Loads the genome from AGENT_FILE and runs it on TASK using the configured
    backend.  Prints the agent's response to stdout.

    \\b
    Example:
        cambrian run --agent best.json "What is the Riemann hypothesis?"
        cambrian run --agent best.json --format json "Explain entropy."
    """
    from cambrian.export import load_genome_json

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        click.echo("Error: no API key. Set OPENAI_API_KEY or pass --api-key.", err=True)
        sys.exit(1)

    genome = load_genome_json(agent_file)
    if model:
        genome.model = model
    if temperature is not None:
        genome.temperature = temperature

    click.echo(f"Agent: {genome.genome_id}  model={genome.model}  temp={genome.temperature}", err=True)

    backend = _make_backend(genome.model, base_url, api_key)
    from cambrian.agent import Agent as _Agent
    agent_obj = _Agent(genome=genome, backend=backend)

    t0 = time.monotonic()
    try:
        result = agent_obj.run(task)
    except Exception as exc:
        click.echo(f"Error running agent: {exc}", err=True)
        sys.exit(1)
    latency = (time.monotonic() - t0) * 1000

    if output_format == "json":
        payload = {
            "result": result,
            "task": task,
            "agent_id": agent_obj.id,
            "genome_id": genome.genome_id,
            "model": genome.model,
            "latency_ms": round(latency, 2),
        }
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        click.echo(result)


# ── version ───────────────────────────────────────────────────────────────────

@main.command()
def version() -> None:
    """Print Cambrian version."""
    click.echo(f"Cambrian {__version__}")
