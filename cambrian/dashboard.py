"""Streamlit live evolution dashboard for Cambrian — Evolve + Forge modes.

Supports two display modes:
- **Evolve tab**: real-time fitness chart, population table, diversity metrics,
  lineage graph preview, best genome viewer.
- **Forge tab**: CodeGenome code viewer, test-case pass/fail grid, LOC + runtime
  charts, pipeline step viewer.
- **Shared**: generation slider, export button, live NDJSON log panel.

The dashboard reads from a log file written by the evolution engine in NDJSON
format (one JSON object per line) or from a legacy JSON array.

Usage
-----
From the CLI::

    cambrian dashboard --log-file run.json

Programmatically::

    from cambrian.dashboard import run_dashboard
    run_dashboard(port=8501, log_file="run.json")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_log(log_file: str) -> list[dict[str, Any]]:
    """Load a log file in NDJSON or legacy JSON array format."""
    path = Path(log_file)
    raw = path.read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        return []

    # Try NDJSON first (one JSON object per line — dicts only)
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Accept only dicts (skip bare arrays that appear in legacy JSON)
            if isinstance(obj, dict):
                entries.append(obj)
        except json.JSONDecodeError:
            continue

    if entries:
        return entries

    # Fallback: legacy JSON array (entire file is a JSON array)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            # Flatten one level: each element should be a dict (generation record)
            result: list[dict[str, Any]] = []
            for item in data:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, list):
                    # Legacy format: list of agent lists per generation
                    for sub in item:
                        if isinstance(sub, dict):
                            result.append(sub)
            return result
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    return []


def _extract_generations(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract only generation entries (event == 'generation' or has 'generation' key)."""
    return [
        e for e in entries
        if e.get("event") == "generation" or (
            "generation" in e and e.get("event") != "run_complete"
        )
    ]


def _extract_forge_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract Forge-specific entries (event == 'forge_generation')."""
    return [e for e in entries if e.get("event") == "forge_generation"]


# ── Main app ──────────────────────────────────────────────────────────────────


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
        page_title="Cambrian Dashboard",
        page_icon="\U0001f9ec",
        layout="wide",
    )

    st.title("\U0001f9ec Cambrian Dashboard")

    # ── Sidebar ───────────────────────────────────────────────────────────────

    st.sidebar.header("\U0001f9ec Cambrian")
    refresh_s = st.sidebar.slider("Auto-refresh (s)", 2, 60, 5)
    st.sidebar.caption(f"Log: `{log_file}`")

    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_s}">',
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────

    log_path = Path(log_file)
    if not log_path.exists():
        st.error(f"Log file not found: `{log_file}`")
        st.info("Start an evolution run with `cambrian evolve` or `cambrian forge`.")
        st.stop()
        return

    entries = _load_log(log_file)
    if not entries:
        st.warning("Log file is empty or contains no valid JSON entries.")
        st.info("Waiting for data…")
        st.stop()
        return

    gen_entries = _extract_generations(entries)
    forge_entries = _extract_forge_entries(entries)
    has_evolve = bool(gen_entries)
    has_forge = bool(forge_entries)

    # Summary metrics from run_complete entry if present
    run_complete = next(
        (e for e in entries if e.get("event") == "run_complete"), None
    )

    st.sidebar.caption(f"Entries: **{len(entries)}**")
    if run_complete:
        st.sidebar.success(f"Run complete · best={run_complete.get('best_fitness', '?'):.4f}")

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab_labels = []
    if has_evolve or not has_forge:
        tab_labels.append("\U0001f9ec Evolve")
    if has_forge:
        tab_labels.append("\U0001f527 Forge")
    tab_labels.append("\U0001f4cb Logs")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── EVOLVE TAB ────────────────────────────────────────────────────────────

    if has_evolve or not has_forge:
        with tabs[tab_idx]:
            tab_idx += 1
            _render_evolve_tab(gen_entries, run_complete)

    # ── FORGE TAB ─────────────────────────────────────────────────────────────

    if has_forge:
        with tabs[tab_idx]:
            tab_idx += 1
            _render_forge_tab(forge_entries)

    # ── LOGS TAB ─────────────────────────────────────────────────────────────

    with tabs[tab_idx]:
        _render_logs_tab(entries, log_file)

    st.caption(
        f"Cambrian Dashboard · `{log_file}` · "
        f"auto-refresh every {refresh_s}s"
    )


# ── Evolve tab ────────────────────────────────────────────────────────────────


def _render_evolve_tab(
    gen_entries: list[dict[str, Any]],
    run_complete: "dict[str, Any] | None",
) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    st.header("\U0001f9ec Evolve Mode")

    if not gen_entries:
        st.info("No generation data yet. Waiting for first generation…")
        return

    # Flatten generation data
    gen_numbers: list[int] = []
    best_fitness_series: list[float] = []
    mean_fitness_series: list[float] = []
    std_fitness_series: list[float] = []
    all_agents: list[dict[str, Any]] = []

    for e in gen_entries:
        gen = int(e.get("generation", len(gen_numbers)))
        best_f = float(e.get("best_fitness", 0.0))
        mean_f = float(e.get("mean_fitness", 0.0))
        std_f = float(e.get("std_fitness", 0.0))
        gen_numbers.append(gen)
        best_fitness_series.append(best_f)
        mean_fitness_series.append(mean_f)
        std_fitness_series.append(std_f)
        for a in e.get("agents", []):
            a = dict(a)
            a["_generation"] = gen
            all_agents.append(a)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations", len(gen_numbers))
    col2.metric("Best fitness", f"{max(best_fitness_series):.4f}")
    col3.metric("Latest mean", f"{mean_fitness_series[-1]:.4f}")
    col4.metric("Total agents evaluated", len(all_agents))

    st.divider()

    # Generation slider
    if len(gen_numbers) > 1:
        selected_gen = st.slider(
            "Select generation",
            min_value=min(gen_numbers),
            max_value=max(gen_numbers),
            value=max(gen_numbers),
        )
        gen_agents = [a for a in all_agents if a.get("_generation") == selected_gen]
    else:
        selected_gen = gen_numbers[0] if gen_numbers else 0
        gen_agents = all_agents

    # Fitness chart
    st.subheader("Fitness Trajectory")
    try:
        import pandas as pd

        df_fitness = pd.DataFrame({
            "Generation": gen_numbers * 2,
            "Fitness": best_fitness_series + mean_fitness_series,
            "Series": ["Best"] * len(gen_numbers) + ["Mean"] * len(gen_numbers),
        })
        try:
            import altair as alt
            chart = (
                alt.Chart(df_fitness)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Generation:Q"),
                    y=alt.Y("Fitness:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Series:N"),
                    tooltip=["Generation:Q", "Fitness:Q", "Series:N"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            df_line = pd.DataFrame(
                {"Best": best_fitness_series, "Mean": mean_fitness_series},
                index=gen_numbers,
            )
            st.line_chart(df_line)
    except ImportError:
        st.write("Install `pandas` for charts.")

    # Diversity metrics
    if std_fitness_series:
        with st.expander("Diversity Metrics"):
            try:
                import pandas as pd
                df_div = pd.DataFrame({
                    "Generation": gen_numbers,
                    "Std Dev": std_fitness_series,
                })
                st.line_chart(df_div.set_index("Generation"))
            except ImportError:
                pass

    # Population table for selected generation
    st.subheader(f"Population at Generation {selected_gen}")
    top_n = st.slider("Show top N", 3, 20, 8, key="evolve_top_n")
    sorted_agents = sorted(
        gen_agents,
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
                "Fitness": round(float(a.get("fitness") or 0.0), 4),
                "Model": genome.get("model", ""),
                "Temp": round(float(genome.get("temperature", 0.7)), 2),
                "Strategy": genome.get("strategy", ""),
                "Tokens": len(genome.get("system_prompt", "")) // 4,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        for a in sorted_agents[:3]:
            st.json(a)

    # Best genome viewer
    if sorted_agents:
        st.subheader("Best Genome")
        best = sorted_agents[0]
        genome = best.get("genome", {})
        with st.expander(
            f"Agent {str(best.get('id',''))[:8]}  "
            f"fitness={float(best.get('fitness') or 0):.4f}  "
            f"gen={best.get('_generation','?')}",
            expanded=True,
        ):
            st.markdown("**System prompt:**")
            st.code(genome.get("system_prompt", "(empty)"), language="text")
            _cols = st.columns(3)
            _cols[0].metric("Model", genome.get("model", ""))
            _cols[1].metric("Temperature", f"{genome.get('temperature', 0.7):.2f}")
            _cols[2].metric("Strategy", genome.get("strategy", ""))

    # Export button
    if sorted_agents:
        best_genome = sorted_agents[0].get("genome", {})
        st.download_button(
            "Export best genome (JSON)",
            data=json.dumps(best_genome, indent=2),
            file_name="best_genome.json",
            mime="application/json",
        )


# ── Forge tab ─────────────────────────────────────────────────────────────────


def _render_forge_tab(forge_entries: list[dict[str, Any]]) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    st.header("\U0001f527 Forge Mode")

    if not forge_entries:
        st.info("No Forge data yet. Run `cambrian forge TASK` to populate.")
        return

    forge_mode = forge_entries[0].get("mode", "code") if forge_entries else "code"

    # Metrics
    last = forge_entries[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Generations", len(forge_entries))
    col2.metric("Best fitness", f"{last.get('best_fitness', 0.0):.4f}")
    col3.metric("Mode", forge_mode)

    st.divider()

    # Generation slider
    if len(forge_entries) > 1:
        gen_idx = st.slider(
            "Select Forge generation",
            min_value=0,
            max_value=len(forge_entries) - 1,
            value=len(forge_entries) - 1,
            key="forge_gen",
        )
        entry = forge_entries[gen_idx]
    else:
        entry = forge_entries[0]
        gen_idx = 0

    if forge_mode == "code":
        _render_forge_code_tab(forge_entries, entry, gen_idx)
    else:
        _render_forge_pipeline_tab(forge_entries, entry, gen_idx)


def _render_forge_code_tab(
    all_entries: list[dict[str, Any]],
    entry: dict[str, Any],
    gen_idx: int,
) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    # Fitness + LOC + runtime over time
    gens = list(range(len(all_entries)))
    fitnesses = [e.get("best_fitness", 0.0) for e in all_entries]
    locs = [e.get("best_loc", 0) for e in all_entries]
    runtimes = [e.get("best_runtime_s", 0.0) for e in all_entries]

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Fitness + LOC over generations")
        try:
            import pandas as pd
            df = pd.DataFrame({
                "Generation": gens * 2,
                "Value": fitnesses + [loc_val / max(max(locs), 1) for loc_val in locs],
                "Series": ["Fitness"] * len(gens) + ["LOC (norm)"] * len(gens),
            })
            try:
                import altair as alt
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(
                        x="Generation:Q",
                        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
                        color="Series:N",
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.line_chart(pd.DataFrame({"Fitness": fitnesses, "LOC (norm)": [loc_val / max(max(locs), 1) for loc_val in locs]}, index=gens))
        except ImportError:
            st.write("Install pandas for charts.")

    with col_right:
        st.subheader("Execution time (s)")
        try:
            import pandas as pd
            df_rt = pd.DataFrame({"Runtime (s)": runtimes}, index=gens)
            st.bar_chart(df_rt)
        except ImportError:
            pass

    # Best code viewer
    st.subheader(f"Best Code — Generation {gen_idx}")
    best_code = entry.get("best_code", "# (no code yet)")
    st.code(best_code, language="python")

    # Test case pass/fail grid
    test_results = entry.get("test_results", [])
    if test_results:
        st.subheader("Test Cases")
        cols = st.columns(min(len(test_results), 5))
        for i, tr in enumerate(test_results):
            passed = tr.get("passed", False)
            label = tr.get("label", f"Test {i+1}")
            with cols[i % len(cols)]:
                if passed:
                    st.success(label[:20])
                else:
                    st.error(label[:20])

    # Export code
    if best_code and best_code != "# (no code yet)":
        st.download_button(
            "Export best code (.py)",
            data=best_code,
            file_name="forge_best.py",
            mime="text/x-python",
        )


def _render_forge_pipeline_tab(
    all_entries: list[dict[str, Any]],
    entry: dict[str, Any],
    gen_idx: int,
) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader(f"Pipeline — Generation {gen_idx}")

    steps = entry.get("best_steps", [])
    if not steps:
        st.info("No pipeline step data in log.")
        return

    for i, step in enumerate(steps):
        with st.expander(f"Step {i+1}: [{step.get('role', '?')}]", expanded=True):
            st.code(step.get("system_prompt", ""), language="text")
            st.caption(f"Temperature: {step.get('temperature', 0.7):.2f}")

    # Fitness over generations
    gens = list(range(len(all_entries)))
    fitnesses = [e.get("best_fitness", 0.0) for e in all_entries]
    step_counts = [len(e.get("best_steps", [])) for e in all_entries]

    st.subheader("Pipeline Evolution")
    try:
        import pandas as pd
        df = pd.DataFrame({"Fitness": fitnesses, "Steps": step_counts}, index=gens)
        st.line_chart(df)
    except ImportError:
        pass

    # Export pipeline
    if steps:
        st.download_button(
            "Export best pipeline (JSON)",
            data=json.dumps({"steps": steps}, indent=2),
            file_name="forge_pipeline.json",
            mime="application/json",
        )


# ── Logs tab ──────────────────────────────────────────────────────────────────


def _render_logs_tab(entries: list[dict[str, Any]], log_file: str) -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    st.header("\U0001f4cb Live Logs")
    st.caption(f"Showing last 50 entries from `{log_file}`")

    tail = entries[-50:]
    for entry in reversed(tail):
        event = entry.get("event", "generation")
        gen = entry.get("generation", "?")
        run_id = entry.get("run_id", "")
        best = entry.get("best_fitness", "")
        label = f"[{event}] gen={gen}"
        if run_id:
            label += f" run={run_id}"
        if best != "":
            label += f" best={best:.4f}"

        with st.expander(label):
            st.json(entry)


# ── run_dashboard (public API) ────────────────────────────────────────────────


def run_dashboard(
    port: int = 8501,
    log_file: str = "cambrian_log.json",
    open_browser: bool = True,
) -> None:
    """Launch the Streamlit dashboard in a subprocess.

    Args:
        port: TCP port for the Streamlit server. Default ``8501``.
        log_file: Path to the evolution log JSON/NDJSON file.
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
        sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
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
