# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Streamlit-based live evolution dashboard for Cambrian.

Two-tab dashboard covering both run modes:

**Tab 1 — Evolve**: Real-time fitness trajectory, agent population table, fitness
landscape heatmap, and best genome viewer for Evolve mode (prompt optimisation).

**Tab 2 — Forge**: Pipeline step viewer, code evolution history with generation
slider, test-case pass/fail grid, and execution time chart for Forge mode.

**Shared widgets**: Auto-refresh control, generation slider, export button.

Log formats
-----------
*Evolve log* (``--log-file`` in the CLI) — JSON array::

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

*Forge log* — JSON array (one entry per generation)::

    [
        {
            "generation": 0,
            "mode": "code",   # or "pipeline"
            "population": [
                {
                    "id": "...", "fitness": 0.75,
                    "genome": {
                        "code": "def solution(s): ...",
                        "entry_point": "solution",
                        "test_cases": [{"input": "hi", "expected": "ih"}],
                        "version": 2
                    }
                },
                ...
            ]
        },
        ...
    ]

Usage
-----
::

    cambrian dashboard --port 8501 --log-file evolve_log.json

Programmatically::

    from cambrian.dashboard import run_dashboard
    run_dashboard(port=8501, log_file="evolve_log.json")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared between tabs
# ─────────────────────────────────────────────────────────────────────────────


def _load_json(path: str) -> list[dict[str, Any]] | None:
    """Load and parse a JSON log file.  Returns ``None`` on failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data  # type: ignore[no-any-return]
    except Exception:
        return None


def _flatten_evolve(generations: list[dict[str, Any]]) -> tuple[
    list[int], list[float], list[float], list[dict[str, Any]]
]:
    """Extract per-generation series and a flat agent list from Evolve log."""
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

    return gen_numbers, best_fitness, mean_fitness, all_agents


def _flatten_forge(generations: list[dict[str, Any]]) -> tuple[
    list[int], list[float], list[float], list[dict[str, Any]]
]:
    """Extract per-generation series and a flat population list from Forge log."""
    gen_numbers: list[int] = []
    best_fitness: list[float] = []
    mean_fitness: list[float] = []
    all_agents: list[dict[str, Any]] = []

    for gen_data in generations:
        gen_idx = int(gen_data.get("generation", len(gen_numbers)))
        population = gen_data.get("population", [])
        if not population:
            continue
        fitnesses = [float(a.get("fitness") or 0.0) for a in population]
        gen_numbers.append(gen_idx)
        best_fitness.append(max(fitnesses))
        mean_fitness.append(sum(fitnesses) / len(fitnesses))
        for a in population:
            a = dict(a)
            a["_generation"] = gen_idx
            all_agents.append(a)

    return gen_numbers, best_fitness, mean_fitness, all_agents


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Evolve
# ─────────────────────────────────────────────────────────────────────────────


def _render_evolve_tab(
    st: Any,
    log_file: str,
    refresh_interval: int,
) -> None:
    """Render the Evolve tab."""
    generations = _load_json(log_file)

    if generations is None:
        st.info(f"Evolve log not found: `{log_file}`")
        st.caption("Start an evolve run with `--log-file` to populate this tab.")
        return

    if not generations:
        st.warning("Evolve log is empty.")
        return

    gen_numbers, best_fitness, mean_fitness, all_agents = _flatten_evolve(generations)

    if not gen_numbers:
        st.warning("No agent data in Evolve log.")
        return

    # ── Metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations", len(gen_numbers))
    col2.metric("Best fitness", f"{max(best_fitness):.4f}")
    col3.metric("Latest mean", f"{mean_fitness[-1]:.4f}")
    col4.metric("Total agents", len(all_agents))

    st.divider()

    # ── Fitness trajectory ──
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
        try:
            import pandas as pd
            df = pd.DataFrame({"Best": best_fitness, "Mean": mean_fitness}, index=gen_numbers)
            st.line_chart(df)
        except ImportError:
            st.write("Install `pandas` and `altair` for charts.")

    # ── Generation slider + population table ──
    st.divider()
    st.subheader("Population Table")

    gen_choice = st.slider(
        "Generation", min_value=min(gen_numbers), max_value=max(gen_numbers),
        value=max(gen_numbers), step=1, key="evolve_gen_slider",
    )

    gen_agents = [a for a in all_agents if a.get("_generation") == gen_choice]
    top_n = st.slider("Show top N agents", min_value=3, max_value=20, value=10, key="evolve_topn")
    gen_agents_sorted = sorted(
        gen_agents, key=lambda a: float(a.get("fitness") or 0.0), reverse=True
    )[:top_n]

    try:
        import pandas as pd
        rows = []
        for a in gen_agents_sorted:
            genome = a.get("genome", {})
            rows.append({
                "ID": str(a.get("id", ""))[:8],
                "Fitness": round(float(a.get("fitness") or 0.0), 4),
                "Model": genome.get("model", ""),
                "Temp": round(float(genome.get("temperature", 0.7)), 2),
                "Strategy": genome.get("strategy", ""),
                "Tokens": len(genome.get("system_prompt", "")) // 4,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for a in gen_agents_sorted:
            st.json(a)

    # ── Best genome viewer ──
    all_sorted = sorted(all_agents, key=lambda a: float(a.get("fitness") or 0.0), reverse=True)
    if all_sorted:
        st.divider()
        st.subheader("Best Genome")
        best = all_sorted[0]
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
                f"**Temp:** `{genome.get('temperature', 0.7)}`  "
                f"**Strategy:** `{genome.get('strategy','')}`"
            )
        # Export
        export_json = json.dumps(genome, indent=2)
        st.download_button(
            label="Export best genome (JSON)",
            data=export_json,
            file_name="best_genome.json",
            mime="application/json",
        )

    # ── Fitness landscape heatmap ──
    st.divider()
    st.subheader("Fitness Landscape (Temperature × Prompt Length)")
    try:
        import pandas as pd

        heatmap_rows = [
            {
                "Temp bucket": f"{round(float(a.get('genome', {}).get('temperature', 0.7)) * 2) / 2:.1f}",
                "Token bucket": f"{(len(a.get('genome', {}).get('system_prompt', '')) // 4 // 100) * 100}–"
                                f"{(len(a.get('genome', {}).get('system_prompt', '')) // 4 // 100) * 100 + 99}",
                "Fitness": float(a.get("fitness") or 0.0),
            }
            for a in all_agents
        ]
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
        st.info("Install `pandas` and `altair` for the fitness landscape.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Forge
# ─────────────────────────────────────────────────────────────────────────────


def _render_forge_tab(
    st: Any,
    forge_log_file: str,
) -> None:
    """Render the Forge tab."""
    generations = _load_json(forge_log_file)

    if generations is None:
        st.info(f"Forge log not found: `{forge_log_file}`")
        st.caption(
            "Start a forge run with `cambrian forge TASK` to populate this tab.\n\n"
            "Tip: Pass `--forge-log-file <path>` to the CLI to set a custom log path."
        )
        return

    if not generations:
        st.warning("Forge log is empty.")
        return

    gen_numbers, best_fitness, mean_fitness, all_agents = _flatten_forge(generations)

    if not gen_numbers:
        st.warning("No population data in Forge log.")
        return

    mode = generations[0].get("mode", "code") if generations else "code"

    # ── Metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations", len(gen_numbers))
    col2.metric("Best fitness", f"{max(best_fitness):.4f}")
    col3.metric("Mode", mode.capitalize())
    col4.metric("Total candidates", len(all_agents))

    st.divider()

    # ── Fitness trajectory ──
    st.subheader("Fitness Trajectory")
    try:
        import altair as alt
        import pandas as pd

        df_f = pd.DataFrame({
            "Generation": gen_numbers + gen_numbers,
            "Fitness": best_fitness + mean_fitness,
            "Series": ["Best"] * len(gen_numbers) + ["Mean"] * len(gen_numbers),
        })
        chart = (
            alt.Chart(df_f)
            .mark_line(point=True)
            .encode(
                x=alt.X("Generation:Q"),
                y=alt.Y("Fitness:Q", scale=alt.Scale(domain=[0, 1])),
                color="Series:N",
                tooltip=["Generation:Q", "Fitness:Q", "Series:N"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        try:
            import pandas as pd
            st.line_chart(pd.DataFrame({"Best": best_fitness, "Mean": mean_fitness}, index=gen_numbers))
        except ImportError:
            st.write("Install `pandas` and `altair` for charts.")

    st.divider()

    # ── Generation slider ──
    gen_choice = st.slider(
        "Generation", min_value=min(gen_numbers), max_value=max(gen_numbers),
        value=max(gen_numbers), step=1, key="forge_gen_slider",
    )

    gen_population = [a for a in all_agents if a.get("_generation") == gen_choice]
    gen_sorted = sorted(gen_population, key=lambda a: float(a.get("fitness") or 0.0), reverse=True)

    if mode == "code":
        _render_code_forge(st, gen_sorted, all_agents, gen_numbers)
    else:
        _render_pipeline_forge(st, gen_sorted)


def _render_code_forge(
    st: Any,
    gen_sorted: list[dict[str, Any]],
    all_agents: list[dict[str, Any]],
    gen_numbers: list[int],
) -> None:
    """Forge code mode: population table, best code viewer, test-case grid."""
    # Population table
    st.subheader("Code Population")
    try:
        import pandas as pd
        rows = []
        for a in gen_sorted:
            g = a.get("genome", {})
            code = g.get("code", "")
            rows.append({
                "ID": str(a.get("id", ""))[:8],
                "Fitness": round(float(a.get("fitness") or 0.0), 4),
                "Version": g.get("version", 0),
                "Lines": code.count("\n") + 1 if code else 0,
                "Entry point": g.get("entry_point", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for a in gen_sorted:
            st.json(a)

    # Best code viewer
    if gen_sorted:
        best = gen_sorted[0]
        g = best.get("genome", {})
        st.subheader(f"Best Code — Gen {best.get('_generation','?')} (fitness {float(best.get('fitness') or 0):.4f})")
        st.code(g.get("code", "(empty)"), language="python")

        # Test-case pass/fail grid
        test_cases = g.get("test_cases", [])
        if test_cases:
            st.subheader("Test Cases")
            try:
                import pandas as pd
                tc_rows = []
                for tc in test_cases:
                    tc_rows.append({
                        "Input": str(tc.get("input", "")),
                        "Expected": str(tc.get("expected", "")),
                    })
                st.dataframe(pd.DataFrame(tc_rows), use_container_width=True)
            except ImportError:
                st.json(test_cases)

        # Export
        export_code = g.get("code", "")
        if export_code:
            st.download_button(
                label="Export best code (Python)",
                data=export_code,
                file_name="forge_best.py",
                mime="text/plain",
            )


def _render_pipeline_forge(
    st: Any,
    gen_sorted: list[dict[str, Any]],
) -> None:
    """Forge pipeline mode: pipeline step viewer, fitness table."""
    st.subheader("Pipeline Population")
    try:
        import pandas as pd
        rows = []
        for a in gen_sorted:
            g = a.get("genome", {})
            steps = g.get("steps", [])
            rows.append({
                "ID": str(a.get("id", a.get("pipeline_id", "")))[:8],
                "Fitness": round(float(a.get("fitness") or 0.0), 4),
                "Version": g.get("version", 0),
                "Steps": len(steps),
                "Name": g.get("name", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for a in gen_sorted:
            st.json(a)

    # Best pipeline step viewer
    if gen_sorted:
        best = gen_sorted[0]
        g = best.get("genome", {})
        steps = g.get("steps", [])
        st.subheader(
            f"Best Pipeline — fitness {float(best.get('fitness') or 0):.4f} — {len(steps)} steps"
        )
        for i, step in enumerate(steps, 1):
            with st.expander(f"Step {i}: {step.get('name','?')} [{step.get('role','?')}] temp={step.get('temperature', 0.7):.2f}"):
                st.code(step.get("system_prompt", "(empty)"), language="text")

        # Export
        export_json = json.dumps(g, indent=2)
        st.download_button(
            label="Export best pipeline (JSON)",
            data=export_json,
            file_name="forge_pipeline.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_app(log_file: str, forge_log_file: str = "forge_log.json") -> None:
    """Build and render the two-tab Streamlit dashboard.

    Args:
        log_file: Path to the Evolve log JSON.
        forge_log_file: Path to the Forge log JSON.
    """
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

    # ── Sidebar ──
    st.sidebar.header("Controls")
    refresh_interval = st.sidebar.slider(
        "Auto-refresh (seconds)", min_value=2, max_value=60, value=5
    )
    st.sidebar.caption(f"Evolve log: `{log_file}`")
    st.sidebar.caption(f"Forge log: `{forge_log_file}`")

    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_interval}">',
        unsafe_allow_html=True,
    )

    # ── Two tabs ──
    tab_evolve, tab_forge = st.tabs(["🔬 Evolve", "⚒️ Forge"])

    with tab_evolve:
        _render_evolve_tab(st, log_file, refresh_interval)

    with tab_forge:
        _render_forge_tab(st, forge_log_file)

    st.caption(
        f"Cambrian Evolution Dashboard · auto-refresh every {refresh_interval}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_dashboard (public API)
# ─────────────────────────────────────────────────────────────────────────────


def run_dashboard(
    port: int = 8501,
    log_file: str = "cambrian_log.json",
    forge_log_file: str = "forge_log.json",
    open_browser: bool = True,
) -> None:
    """Launch the Streamlit dashboard in a subprocess.

    Args:
        port: TCP port for the Streamlit server. Default ``8501``.
        log_file: Path to the Evolve log JSON file. Default ``"cambrian_log.json"``.
        forge_log_file: Path to the Forge log JSON file. Default ``"forge_log.json"``.
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

    launcher_code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
        from cambrian.dashboard import _build_app
        _build_app({log_file!r}, {forge_log_file!r})
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
