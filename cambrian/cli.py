# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
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
    from rich.panel import Panel
    from rich.table import Table
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
@click.version_option(version=__version__, prog_name="Cambrian")
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
    help="Path to the Evolve log JSON written by the evolution engine.",
)
@click.option(
    "--forge-log-file", default="forge_log.json", show_default=True,
    help="Path to the Forge log JSON written by forge runs.",
)
@click.option(
    "--no-browser", is_flag=True, default=False,
    help="Do not open a browser tab automatically.",
)
def dashboard(port: int, log_file: str, forge_log_file: str, no_browser: bool) -> None:
    """Launch the Streamlit live evolution dashboard (2 tabs: Evolve + Forge).

    Reads LOG_FILE (Evolve mode) and FORGE_LOG_FILE (Forge mode) and displays
    fitness trajectory, top agents, fitness landscape, code/pipeline viewers.

    \\b
    Example:
        cambrian dashboard --port 8501 --log-file evolve_run.json
    """
    try:
        from cambrian.dashboard import run_dashboard
        click.echo(f"Starting dashboard on http://localhost:{port} ...")
        run_dashboard(
            port=port,
            log_file=log_file,
            forge_log_file=forge_log_file,
            open_browser=not no_browser,
        )
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

    if output_format != "json":
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

# ── snapshot ──────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--memory", "memory_file", required=True, type=click.Path(exists=True),
    help="Path to a lineage JSON file written by --memory-out.",
)
@click.option(
    "--generation", "-g", required=True, type=int,
    help="Generation number to display.",
)
@click.option(
    "--top", default=5, show_default=True,
    help="Number of top agents to show.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text", show_default=True,
    help="Output format.",
)
def snapshot(memory_file: str, generation: int, top: int, output_format: str) -> None:
    """Show the population state at a specific generation.

    Reads MEMORY (a lineage JSON written by --memory-out) and displays the
    agents alive at GENERATION: their fitness, strategy, temperature, and
    genome IDs.

    \\b
    Example:
        cambrian snapshot --memory run.json --generation 5
        cambrian snapshot --memory run.json --generation 10 --format json
    """
    import statistics as _stats

    from cambrian.memory import EvolutionaryMemory

    mem = EvolutionaryMemory.from_json(Path(memory_file).read_text())

    # Collect all agents at the requested generation
    nodes = [
        (nid, attrs)
        for nid, attrs in mem._graph.nodes(data=True)
        if attrs.get("generation") == generation
    ]

    if not nodes:
        # Fall back: show available generations
        available = sorted({
            attrs.get("generation")
            for _, attrs in mem._graph.nodes(data=True)
            if attrs.get("generation") is not None
        })
        click.echo(
            f"No agents found at generation {generation}. "
            f"Available: {available}", err=True
        )
        sys.exit(1)

    # Sort by fitness descending
    nodes.sort(
        key=lambda x: x[1].get("fitness") or 0.0,
        reverse=True,
    )

    if output_format == "json":
        payload = {
            "generation": generation,
            "total_agents": len(nodes),
            "agents": [
                {
                    "agent_id": nid,
                    "fitness": attrs.get("fitness"),
                    "genome": attrs.get("genome"),
                }
                for nid, attrs in nodes[:top]
            ],
        }
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # Text output
    fitnesses = [attrs.get("fitness") or 0.0 for _, attrs in nodes]
    mean_fit = _stats.mean(fitnesses) if fitnesses else 0.0
    best_fit = max(fitnesses) if fitnesses else 0.0

    if _RICH and console is not None:
        from rich.table import Table
        tbl = Table(
            title=f"Population Snapshot — Generation {generation}",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Rank", width=5)
        tbl.add_column("Agent ID", width=12)
        tbl.add_column("Fitness", justify="right")
        tbl.add_column("Model", width=12)
        tbl.add_column("Temp", justify="right")
        tbl.add_column("Strategy", width=14)
        tbl.add_column("Prompt (preview)", width=38)

        for rank, (nid, attrs) in enumerate(nodes[:top], 1):
            genome = attrs.get("genome") or {}
            fit = attrs.get("fitness")
            fit_str = f"{fit:.4f}" if fit is not None else "–"
            prompt = str(genome.get("system_prompt", ""))
            preview = (prompt[:35] + "…") if len(prompt) > 38 else prompt
            tbl.add_row(
                str(rank),
                nid[:10],
                fit_str,
                str(genome.get("model", "")),
                f"{genome.get('temperature', 0):.2f}",
                str(genome.get("strategy", "")),
                preview,
            )
        console.print(tbl)
        console.print(
            f"[dim]Total agents: {len(nodes)}  "
            f"Best: {best_fit:.4f}  Mean: {mean_fit:.4f}[/dim]"
        )
    else:
        click.echo(f"Generation {generation}  |  agents={len(nodes)}  "
                   f"best={best_fit:.4f}  mean={mean_fit:.4f}")
        click.echo("-" * 70)
        for rank, (nid, attrs) in enumerate(nodes[:top], 1):
            genome = attrs.get("genome") or {}
            fit = attrs.get("fitness")
            fit_str = f"{fit:.4f}" if fit is not None else "–"
            prompt = str(genome.get("system_prompt", ""))[:50]
            click.echo(
                f"{rank:2d}. {nid[:10]}  fit={fit_str}  "
                f"model={genome.get('model', '?')}  "
                f"temp={genome.get('temperature', 0):.2f}  "
                f"prompt={prompt!r}"
            )


# ── version ───────────────────────────────────────────────────────────────────


@main.command()
@click.argument("run1", metavar="RUN1")
@click.argument("run2", metavar="RUN2")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.option("--metric", default="best_fitness", help="Metric to compare (default: best_fitness).")
def compare(run1: str, run2: str, output_format: str, metric: str) -> None:
    """Compare two evolution run logs written by JSONLogger.

    \b
    RUN1 and RUN2 are NDJSON log files (one JSON object per line).

    Example:
        cambrian compare run_a.json run_b.json
        cambrian compare run_a.json run_b.json --metric mean_fitness --format json
    """
    from cambrian.utils.logging import load_json_log

    try:
        entries_a = load_json_log(run1)
        entries_b = load_json_log(run2)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not entries_a or not entries_b:
        click.echo("Error: one or both log files are empty.", err=True)
        sys.exit(1)

    # Filter to generation entries only
    gens_a = [e for e in entries_a if "generation" in e and "event" not in e]
    gens_b = [e for e in entries_b if "generation" in e and "event" not in e]

    run_id_a = entries_a[0].get("run_id", Path(run1).stem)
    run_id_b = entries_b[0].get("run_id", Path(run2).stem)

    def _extract(entries: list[dict[str, Any]], key: str) -> list[float]:
        return [float(e.get(key, 0.0)) for e in entries]

    vals_a = _extract(gens_a, metric)
    vals_b = _extract(gens_b, metric)

    final_a = vals_a[-1] if vals_a else 0.0
    final_b = vals_b[-1] if vals_b else 0.0
    best_a = max(vals_a) if vals_a else 0.0
    best_b = max(vals_b) if vals_b else 0.0
    winner = run_id_a if best_a >= best_b else run_id_b
    delta = abs(best_a - best_b)

    if output_format == "json":
        payload = {
            "metric": metric,
            "run_a": {
                "id": run_id_a,
                "file": run1,
                "generations": len(gens_a),
                "final": round(final_a, 6),
                "best": round(best_a, 6),
                "values": [round(v, 6) for v in vals_a],
            },
            "run_b": {
                "id": run_id_b,
                "file": run2,
                "generations": len(gens_b),
                "final": round(final_b, 6),
                "best": round(best_b, 6),
                "values": [round(v, 6) for v in vals_b],
            },
            "winner": winner,
            "delta": round(delta, 6),
        }
        click.echo(json.dumps(payload, indent=2))
        return

    # Text output
    if _RICH and console:
        table = Table(title=f"Evolution Comparison — {metric}", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column(run_id_a, style="cyan")
        table.add_column(run_id_b, style="magenta")
        table.add_row("File", run1, run2)
        table.add_row("Generations", str(len(gens_a)), str(len(gens_b)))
        table.add_row(f"Final {metric}", f"{final_a:.4f}", f"{final_b:.4f}")
        table.add_row(f"Best {metric}", f"{best_a:.4f}", f"{best_b:.4f}")
        table.add_row("Winner", "✓" if run_id_a == winner else "", "✓" if run_id_b == winner else "")
        table.add_row("Δ (best)", f"{delta:.4f}", "")
        console.print(table)

        # Per-generation breakdown
        gen_table = Table(title="Per-Generation Values", show_lines=False)
        gen_table.add_column("Gen", style="dim")
        gen_table.add_column(run_id_a, style="cyan")
        gen_table.add_column(run_id_b, style="magenta")
        for i in range(max(len(vals_a), len(vals_b))):
            a_val = f"{vals_a[i]:.4f}" if i < len(vals_a) else "-"
            b_val = f"{vals_b[i]:.4f}" if i < len(vals_b) else "-"
            gen_table.add_row(str(i), a_val, b_val)
        console.print(gen_table)
    else:
        click.echo(f"\nComparison — {metric}")
        click.echo(f"{'':20s}  {run_id_a:>12s}  {run_id_b:>12s}")
        click.echo("-" * 50)
        click.echo(f"{'Generations':20s}  {len(gens_a):>12d}  {len(gens_b):>12d}")
        click.echo(f"{'Final ' + metric:20s}  {final_a:>12.4f}  {final_b:>12.4f}")
        click.echo(f"{'Best ' + metric:20s}  {best_a:>12.4f}  {best_b:>12.4f}")
        click.echo(f"\nWinner: {winner}  (Δ={delta:.4f})")


@main.command()
def version() -> None:
    """Print Cambrian version."""
    click.echo(f"Cambrian {__version__}")


# ── meta-evolve ───────────────────────────────────────────────────────────────


@main.command("meta-evolve")
@click.argument("task")
@click.option("--model", "-m", default="gpt-4o-mini", show_default=True,
              help="LLM model identifier.")
@click.option("--base-url", default="https://api.openai.com/v1", show_default=True,
              envvar="CAMBRIAN_BASE_URL", help="OpenAI-compatible API base URL.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY",
              help="API key (falls back to OPENAI_API_KEY env var).")
@click.option("--generations", "-g", default=10, show_default=True,
              help="Total number of generations.")
@click.option("--population", "-p", default=6, show_default=True,
              help="Population size.")
@click.option("--meta-interval", default=2, show_default=True,
              help="Run a hyperparameter meta-step every N generations.")
@click.option("--output", "-o", default="meta_best.json", show_default=True,
              help="Path to save the best genome JSON.")
@click.option("--memory-out", default=None,
              help="Path to save lineage JSON (optional).")
def meta_evolve(
    task: str,
    model: str,
    base_url: str,
    api_key: str | None,
    generations: int,
    population: int,
    meta_interval: int,
    output: str,
    memory_out: str | None,
) -> None:
    """Run meta-evolution: co-evolve agents AND hyperparameters (MAML-inspired).

    Outer loop adapts mutation_rate, crossover_rate, temperature, and
    tournament_k automatically. Use when you want Cambrian to tune itself
    while searching.

    \b
    Example:
        cambrian meta-evolve "Summarise text in one sentence" \\
            --generations 20 --population 8 --output meta_best.json
    """
    from cambrian.meta_evolution import MetaEvolutionEngine, HyperParams
    from cambrian.evaluator import Evaluator
    from cambrian.agent import Agent

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    backend = _make_backend(model, base_url, key)
    mutator = LLMMutator(backend=backend)

    class _LLMJudgeEval(Evaluator):
        def evaluate(self, agent: Agent, task: str) -> float:
            prompt = (
                f"Rate how well this system prompt helps with: {task}\n"
                f"System prompt: {agent.genome.system_prompt}\n"
                f"Reply with a single float between 0.0 and 1.0. Nothing else."
            )
            try:
                raw = backend.generate(prompt)
                return max(0.0, min(1.0, float(raw.strip().split()[0])))
            except Exception:
                return 0.0

    evaluator = _LLMJudgeEval()
    engine = MetaEvolutionEngine(
        evaluator=evaluator,
        mutator=mutator,
        backend=backend,
        population_size=population,
    )
    seeds = [Genome(system_prompt=f"Agent {i}: help with {task}") for i in range(population)]

    if _RICH and console is not None:
        console.rule(f"[bold cyan]Meta-Evolution[/bold cyan]  task={task!r}")

    def _on_gen(gen: int, pop: list[Any], hp: HyperParams) -> None:
        scores = [a.fitness or 0.0 for a in pop]
        best_s = max(scores) if scores else 0.0
        mean_s = sum(scores) / max(len(scores), 1)
        if _RICH and console is not None:
            console.print(
                f"  Gen [bold]{gen:3d}[/bold]  best=[green]{best_s:.4f}[/green]"
                f"  mean={mean_s:.4f}  mut={hp.mutation_rate:.2f}"
                f"  temp={hp.temperature:.2f}"
            )
        else:
            click.echo(
                f"  Gen {gen:3d}  best={best_s:.4f}  mean={mean_s:.4f}"
                f"  mut={hp.mutation_rate:.2f}  temp={hp.temperature:.2f}"
            )

    click.echo(f"Meta-evolving {population} agents for {generations} generations ...")
    best = engine.evolve(
        seed_genomes=seeds,
        task=task,
        n_generations=generations,
        meta_interval=meta_interval,
        on_generation=_on_gen,
    )

    out_path = Path(output)
    out_path.write_text(json.dumps(best.genome.to_dict(), indent=2))

    if memory_out:
        mem_path = Path(memory_out)
        mem_path.write_text(engine.memory.to_json())
        click.echo(f"Lineage graph written to {mem_path}")

    if _RICH and console is not None:
        console.rule("[bold green]Done[/bold green]")
        console.print(f"Best fitness : [bold green]{best.fitness or 0.0:.4f}[/bold green]")
        console.print(f"Genome saved : {out_path}")
    else:
        click.echo(f"\nBest fitness: {best.fitness or 0.0:.4f}")
        click.echo(f"Genome saved: {out_path}")


# ── tournament ────────────────────────────────────────────────────────────────


@main.command()
@click.argument("task")
@click.option("--model", "-m", default="gpt-4o-mini", show_default=True,
              help="LLM model identifier.")
@click.option("--base-url", default="https://api.openai.com/v1", show_default=True,
              envvar="CAMBRIAN_BASE_URL", help="OpenAI-compatible API base URL.")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY",
              help="API key (falls back to OPENAI_API_KEY env var).")
@click.option("--agents-file", "-a", default=None, type=click.Path(exists=True),
              help="JSON file containing a list of genome dicts to load as agents.")
@click.option("--population", "-p", default=6, show_default=True,
              help="Random population size (used when --agents-file is not given).")
@click.option("--output", "-o", default=None,
              help="Path to save tournament record JSON (optional).")
def tournament(
    task: str,
    model: str,
    base_url: str,
    api_key: str | None,
    agents_file: str | None,
    population: int,
    output: str | None,
) -> None:
    """Run a round-robin tournament between agents for a given task.

    Each pair of agents competes head-to-head; fitness is updated in-place.
    Prints a ranked leaderboard after all matches.

    \b
    Example:
        cambrian tournament "Explain quantum entanglement" \\
            --agents-file team.json --output results.json
    """
    from cambrian.self_play import SelfPlayEvaluator, run_tournament
    from cambrian.agent import Agent

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    backend = _make_backend(model, base_url, key)
    evaluator = _make_evaluator(task, backend)

    if agents_file:
        data = json.loads(Path(agents_file).read_text())
        if isinstance(data, list):
            agents = [Agent(genome=Genome.from_dict(d)) for d in data]
        else:
            raise click.ClickException("agents-file must contain a JSON list of genome dicts")
    else:
        agents = [Agent(genome=Genome(system_prompt=f"Agent {i}: help with {task}"))
                  for i in range(population)]

    sp_eval = SelfPlayEvaluator(base_evaluator=evaluator, win_bonus=0.1, loss_penalty=0.05)

    click.echo(f"Running tournament: {len(agents)} agents, task={task!r}")
    record = run_tournament(agents, sp_eval, task)

    # Build ranked table
    ranked = sorted(
        agents,
        key=lambda a: a.fitness or 0.0,
        reverse=True,
    )

    if _RICH and console is not None:
        table = Table(title="Tournament Leaderboard", show_header=True)
        table.add_column("Rank", style="bold", width=6)
        table.add_column("Agent ID", width=36)
        table.add_column("Fitness", justify="right")
        table.add_column("W", justify="right", style="green")
        table.add_column("L", justify="right", style="red")
        table.add_column("D", justify="right")
        for rank, agent in enumerate(ranked, 1):
            aid = agent.id
            table.add_row(
                str(rank),
                aid[:32] + "…" if len(aid) > 32 else aid,
                f"{agent.fitness or 0.0:.4f}",
                str(record.wins.get(aid, 0)),
                str(record.losses.get(aid, 0)),
                str(record.draws.get(aid, 0)),
            )
        console.print(table)
    else:
        click.echo(f"\n{'Rank':>4}  {'Agent':36}  {'Fitness':>8}  W  L  D")
        click.echo("-" * 64)
        for rank, agent in enumerate(ranked, 1):
            aid = agent.id[:32]
            click.echo(
                f"{rank:>4}  {aid:36}  {agent.fitness or 0.0:8.4f}"
                f"  {record.wins.get(agent.id, 0)}"
                f"  {record.losses.get(agent.id, 0)}"
                f"  {record.draws.get(agent.id, 0)}"
            )

    if output:
        out_data = {
            "task": task,
            "agents": [
                {
                    "id": a.id,
                    "fitness": a.fitness or 0.0,
                    "wins": record.wins.get(a.id, 0),
                    "losses": record.losses.get(a.id, 0),
                    "draws": record.draws.get(a.id, 0),
                    "genome": a.genome.to_dict(),
                }
                for a in ranked
            ],
            "total_matches": len(record.matches),
        }
        Path(output).write_text(json.dumps(out_data, indent=2))
        click.echo(f"\nResults saved: {output}")


# ── forge ─────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("task")
@click.option(
    "--model", "-m",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model for code mutation and evaluation.",
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
    "--generations", "-g",
    default=8,
    show_default=True,
    help="Number of evolutionary generations.",
)
@click.option(
    "--population", "-p",
    default=6,
    show_default=True,
    help="Population size per generation.",
)
@click.option(
    "--mode",
    default="code",
    show_default=True,
    type=click.Choice(["code", "pipeline"], case_sensitive=False),
    help="Forge mode: 'code' evolves Python code, 'pipeline' evolves step chains.",
)
@click.option(
    "--entry-point",
    default="solution",
    show_default=True,
    help="[code mode] Name of the Python function to evolve.",
)
@click.option(
    "--test-case",
    "test_cases",
    multiple=True,
    help=(
        "[code mode] Test case in 'INPUT:EXPECTED' format. "
        "Repeat for multiple cases."
    ),
)
@click.option(
    "--seed-code",
    default=None,
    help="[code mode] Path to a Python file used as the initial genome.",
)
@click.option(
    "--seed-pipeline",
    default=None,
    type=click.Path(exists=True),
    help="[pipeline mode] Path to a Pipeline JSON file used as the seed.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output path: forge_best.py (code) or forge_best.json (pipeline).",
)
@click.option(
    "--temperature",
    default=0.6,
    show_default=True,
    help="Mutation temperature.",
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    help="[code mode] Sandbox timeout per test-case (seconds).",
)
def forge(
    task: str,
    model: str,
    base_url: str,
    api_key: str | None,
    generations: int,
    population: int,
    mode: str,
    entry_point: str,
    test_cases: tuple[str, ...],
    seed_code: str | None,
    seed_pipeline: str | None,
    output: str | None,
    temperature: float,
    timeout: float,
) -> None:
    """Synthesise and evolve executable Python code or a multi-step agent pipeline.

    TASK is a natural-language description of the problem to solve.

    \b
    Examples
    --------
    cambrian forge "Write reverse(s: str) -> str" --test-case "hello:olleh"
    cambrian forge "Summarise text" --mode pipeline --generations 5
    """
    from cambrian.code_genome import CodeEvolutionEngine, CodeGenome
    from cambrian.pipeline import Pipeline, PipelineEvolutionEngine, PipelineStep

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    backend = OpenAICompatBackend(model=model, base_url=base_url, api_key=resolved_key)

    if _RICH and console is not None:
        console.rule(f"[bold cyan]Cambrian Forge — {mode} mode[/bold cyan]")
        console.print(f"[dim]Task:[/dim] {task}")
        console.print(f"[dim]Model:[/dim] {model}  [dim]Gens:[/dim] {generations}  [dim]Pop:[/dim] {population}")

    if mode == "code":
        # --- Code mode ---
        parsed_tests: list[dict[str, str]] = []
        for tc_str in test_cases:
            if ":" in tc_str:
                inp, _, exp = tc_str.partition(":")
                parsed_tests.append({"input": inp, "expected": exp})
            else:
                logger.warning("Ignoring malformed --test-case %r (expected 'INPUT:EXPECTED')", tc_str)

        seed_src = ""
        if seed_code:
            seed_src = Path(seed_code).read_text()

        seed = CodeGenome(
            code=seed_src,
            entry_point=entry_point,
            description=task,
            test_cases=parsed_tests,
        )

        engine = CodeEvolutionEngine(
            backend=backend,
            population_size=population,
            mutation_temperature=temperature,
            timeout=timeout,
        )

        generation_count = [0]

        def _on_gen(gen: int, pop: list[Any]) -> None:
            generation_count[0] = gen
            scores = [a.fitness or 0.0 for a in pop]
            best_score = max(scores) if scores else 0.0
            mean_score = sum(scores) / max(len(scores), 1)
            if _RICH and console is not None:
                console.print(
                    f"  Gen [bold]{gen:3d}[/bold]  best=[green]{best_score:.4f}[/green]  mean={mean_score:.4f}"
                )
            else:
                click.echo(f"  Gen {gen:3d}  best={best_score:.4f}  mean={mean_score:.4f}")

        click.echo(f"Evolving code for {generations} generations (population={population}) ...")
        best = engine.evolve(
            seed=seed,
            task=task,
            n_generations=generations,
            on_generation=_on_gen,
        )

        out_path = Path(output) if output else Path("forge_best.py")
        out_path.write_text(best.genome.code)

        if _RICH and console is not None:
            console.rule("[bold green]Done[/bold green]")
            console.print(f"Best fitness: [bold green]{best.fitness or 0.0:.4f}[/bold green]")
            console.print(f"Output: {out_path}")
        else:
            click.echo(f"\nBest fitness: {best.fitness or 0.0:.4f}")
            click.echo(f"Output: {out_path}")

    else:
        # --- Pipeline mode ---
        import json as _json

        if seed_pipeline:
            data = _json.loads(Path(seed_pipeline).read_text())
            seed_pl = Pipeline.from_dict(data)
        else:
            seed_pl = Pipeline(
                name="forge-pipeline",
                steps=[
                    PipelineStep(
                        name="analyser",
                        system_prompt="Analyse the task and identify key requirements.",
                        role="extractor",
                    ),
                    PipelineStep(
                        name="solver",
                        system_prompt="Solve the task based on the analysis.",
                        role="transformer",
                    ),
                    PipelineStep(
                        name="validator",
                        system_prompt="Review and improve the solution for correctness.",
                        role="validator",
                    ),
                ],
            )

        engine_pl = PipelineEvolutionEngine(
            backend=backend,
            population_size=population,
            temperature=temperature,
        )

        def _on_pl_gen(gen: int, pop: list[Any]) -> None:
            scores = [p.fitness or 0.0 for p in pop]
            best_score = max(scores) if scores else 0.0
            mean_score = sum(scores) / max(len(scores), 1)
            if _RICH and console is not None:
                console.print(
                    f"  Gen [bold]{gen:3d}[/bold]  best=[green]{best_score:.4f}[/green]  mean={mean_score:.4f}"
                )
            else:
                click.echo(f"  Gen {gen:3d}  best={best_score:.4f}  mean={mean_score:.4f}")

        click.echo(f"Evolving pipeline for {generations} generations (population={population}) ...")
        best_pl = engine_pl.evolve(
            seed=seed_pl,
            task=task,
            n_generations=generations,
            on_generation=_on_pl_gen,
        )

        out_path = Path(output) if output else Path("forge_pipeline.json")
        out_path.write_text(_json.dumps(best_pl.to_dict(), indent=2))

        if _RICH and console is not None:
            console.rule("[bold green]Done[/bold green]")
            console.print(f"Best fitness: [bold green]{best_pl.fitness or 0.0:.4f}[/bold green]")
            console.print(f"Steps: {len(best_pl.steps)}")
            console.print(f"Output: {out_path}")
        else:
            click.echo(f"\nBest fitness: {best_pl.fitness or 0.0:.4f}")
            click.echo(f"Steps: {len(best_pl.steps)}")
            click.echo(f"Output: {out_path}")
