"""Streamlit-based live evolution dashboard for Cambrian.

Displays real-time and post-run visualisations of an evolutionary experiment:

- **Fitness trajectory** — best and mean fitness across generations.
- **Diversity timeline** — strategy entropy, temperature std, unique strategies.
- **Top agents** — best genome, fitness, generation, token count.
- **Fitness landscape** — 2D heatmap of temperature × prompt length.
- **Pareto front** — non-dominated agents highlighted.

The dashboard reads from a *log file* written by the evolution engine.  You
can point it at a live file (Streamlit auto-refreshes every few seconds) or
at a completed run file.

Log format
----------
The log file is a JSON file with this structure (written by
:func:`~cambrian.evolution.EvolutionEngine.save_population` or manually)::

    [
        {
            "generation": 0,
            "agents": [
                {"id": "...", "fitness": 0.62, "genome": {...}},
                ...
            ]
        },
        ...
    ]

Usage
-----
From the CLI::

    cambrian dashboard --port 8501 --log-file evolution_log.json

Programmatically::

    from cambrian.dashboard import run_dashboard
    run_dashboard(port=8501, log_file="evolution_log.json")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ── Dashboard logic (requires streamlit) ─────────────────────────────────────


def _build_app(log_file: str) -> None:
    """Build and run the Streamlit app.  Must be called from within Streamlit."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the dashboard. "
            "Install it with: pip install streamlit"
        ) from exc

    st.set_page_config(
        page_title="Cambrian — Evolution Dashboard",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Cambrian Evolution Dashboard")

    # ── Load data ─────────────────────────────────────────────────────────────

    log_path = Path(log_file)
    if not log_path.exists():
        st.error(f"Log file not found: {log_file}")
        st.stop()
        return

    try:
        raw = log_path.read_text(encoding="utf-8")
        generations: list[dict[str, Any]] = json.loads(raw)
    except Exception as exc:
        st.error(f"Failed to parse log file: {exc}")
        st.stop()
        return

    if not generations:
        st.warning("Log file is empty. Start an evolution run to populate the dashboard.")
        st.stop()
        return

    # ── Sidebar controls ──────────────────────────────────────────────────────

    st.sidebar.header("Controls")
    refresh_interval = st.sidebar.slider(
        "Auto-refresh (seconds)", min_value=2, max_value=60, value=5
    )
    st.sidebar.caption(f"Watching: `{log_file}`")
    st.sidebar.caption(f"Generations loaded: **{len(generations)}**")

    # Auto-refresh via meta-refresh
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_interval}">',
        unsafe_allow_html=True,
    )

    # ── Flatten data ──────────────────────────────────────────────────────────

    gen_numbers: list[int] = []
    best_fitness: list[float] = []
    mean_fitness: list[float] = []
    all_agents: list[dict[str, Any]] = []

    for gen_data in generations:
        gen_idx = int(gen_data.get("generation", len(gen_numbers)))
        agents = gen_data.get("agents", [])
        if not agents:
            continue
        fitnesses = [float(a.get("fitness") or 0.0) for a in agents]
        gen_numbers.append(gen_idx)
        best_fitness.append(max(fitnesses))
        mean_fitness.append(sum(fitnesses) / len(fitnesses))
        for a in agents:
            a = dict(a)
            a["_generation"] = gen_idx
            all_agents.append(a)

    if not gen_numbers:
        st.warning("No agent data found in log file.")
        st.stop()
        return

    # ── Metrics row ───────────────────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations", len(gen_numbers))
    col2.metric("Best fitness", f"{max(best_fitness):.4f}")
    col3.metric("Latest mean", f"{mean_fitness[-1]:.4f}")
    col4.metric("Total agents", len(all_agents))

    st.divider()

    # ── Fitness trajectory ────────────────────────────────────────────────────

    st.subheader("Fitness Trajectory")
    try:
        import altair as alt
        import pandas as pd

        df_fitness = pd.DataFrame({
            "Generation": gen_numbers + gen_numbers,
            "Fitness": best_fitness + mean_fitness,
            "Series": ["Best"] * len(gen_numbers) + ["Mean"] * len(gen_numbers),
        })
        chart = (
            alt.Chart(df_fitness)
            .mark_line(point=True)
            .encode(
                x=alt.X("Generation:Q", title="Generation"),
                y=alt.Y("Fitness:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Series:N"),
                tooltip=["Generation:Q", "Fitness:Q", "Series:N"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        # Fallback: simple line_chart
        try:
            import pandas as pd
            df = pd.DataFrame({"Best": best_fitness, "Mean": mean_fitness}, index=gen_numbers)
            st.line_chart(df)
        except ImportError:
            st.write("Install `pandas` for charts.")

    # ── Top agents table ──────────────────────────────────────────────────────

    st.subheader("Top Agents")
    top_n = st.slider("Show top N agents", min_value=3, max_value=20, value=5)

    sorted_agents = sorted(
        all_agents,
        key=lambda a: float(a.get("fitness") or 0.0),
        reverse=True,
    )[:top_n]

    try:
        import pandas as pd

        rows = []
        for a in sorted_agents:
            genome = a.get("genome", {})
            rows.append({
                "ID": str(a.get("id", ""))[:8],
                "Gen": a.get("_generation", "?"),
                "Fitness": round(float(a.get("fitness") or 0.0), 4),
                "Model": genome.get("model", ""),
                "Temp": round(float(genome.get("temperature", 0.7)), 2),
                "Tokens": len(genome.get("system_prompt", "")) // 4,
                "Strategy": genome.get("strategy", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for a in sorted_agents:
            st.json(a)

    # ── Best genome ───────────────────────────────────────────────────────────

    if sorted_agents:
        st.subheader("Best Genome")
        best = sorted_agents[0]
        genome = best.get("genome", {})
        with st.expander(
            f"Agent {str(best.get('id',''))[:8]}  "
            f"fitness={float(best.get('fitness') or 0):.4f}  "
            f"gen={best.get('_generation','?')}"
        ):
            st.markdown("**System prompt:**")
            st.code(genome.get("system_prompt", "(empty)"), language="text")
            st.markdown(
                f"**Model:** `{genome.get('model','')}`  "
                f"**Temp:** `{genome.get('temperature',0.7)}`  "
                f"**Strategy:** `{genome.get('strategy','')}`"
            )

    # ── Diversity heatmap (temperature × tokens) ──────────────────────────────

    st.subheader("Fitness Landscape (Temperature × Prompt Length)")
    try:
        import pandas as pd

        heatmap_rows = []
        for a in all_agents:
            genome = a.get("genome", {})
            temp = float(genome.get("temperature", 0.7))
            tokens = len(genome.get("system_prompt", "")) // 4
            fitness = float(a.get("fitness") or 0.0)
            temp_bucket = round(temp * 2) / 2  # bins: 0.0, 0.5, 1.0, ...
            token_bucket = (tokens // 100) * 100  # bins: 0, 100, 200, ...
            heatmap_rows.append({
                "Temp bucket": f"{temp_bucket:.1f}",
                "Token bucket": f"{token_bucket}–{token_bucket+99}",
                "Fitness": fitness,
            })
        if heatmap_rows:
            df_heat = pd.DataFrame(heatmap_rows)
            pivot = df_heat.groupby(["Temp bucket", "Token bucket"])["Fitness"].mean().reset_index()
            try:
                import altair as alt
                heat_chart = (
                    alt.Chart(pivot)
                    .mark_rect()
                    .encode(
                        x=alt.X("Temp bucket:O", title="Temperature"),
                        y=alt.Y("Token bucket:O", title="Prompt tokens"),
                        color=alt.Color(
                            "Fitness:Q",
                            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                        ),
                        tooltip=["Temp bucket:O", "Token bucket:O", "Fitness:Q"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(heat_chart, use_container_width=True)
            except ImportError:
                st.dataframe(pivot, use_container_width=True)
    except ImportError:
        st.info("Install `pandas` and `altair` for visualisations.")

    st.caption(
        f"Cambrian Evolution Dashboard · log: `{log_file}` · "
        f"auto-refresh every {refresh_interval}s"
    )


# ── run_dashboard (public API) ────────────────────────────────────────────────


def run_dashboard(
    port: int = 8501,
    log_file: str = "cambrian_log.json",
    open_browser: bool = True,
) -> None:
    """Launch the Streamlit dashboard in a subprocess.

    This function spawns ``streamlit run`` pointing at a temporary launcher
    script that calls :func:`_build_app`.

    Args:
        port: TCP port for the Streamlit server. Default ``8501``.
        log_file: Path to the evolution log JSON file. Default
            ``"cambrian_log.json"``.
        open_browser: If ``True``, Streamlit opens a browser tab automatically.

    Raises:
        ImportError: If ``streamlit`` is not installed.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the dashboard. "
            "Install it with: pip install streamlit"
        ) from exc

    import subprocess
    import tempfile
    import textwrap

    # Write a tiny launcher that calls _build_app with the correct log_file
    launcher_code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
        from cambrian.dashboard import _build_app
        _build_app({log_file!r})
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_cambrian_dashboard.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(launcher_code)
        launcher_path = f.name

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        launcher_path,
        "--server.port", str(port),
    ]
    if not open_browser:
        cmd += ["--server.headless", "true"]

    subprocess.run(cmd, check=False)
