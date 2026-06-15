# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Cambrian FastAPI server — REST + WebSocket backend for the React UI.

Endpoints
---------
GET  /api/health
GET  /api/plugins
GET  /api/plugins/{name}
POST /api/plugins/{name}/enable
POST /api/plugins/{name}/disable
GET  /api/models
GET  /api/runs
GET  /api/runs/{run_id}
DELETE /api/runs/{run_id}
POST /api/evolve  → 201, {run_id, status}
GET  /api/runs/{run_id}/agents
GET  /api/agents/{agent_id}
WS   /ws/evolve/{run_id}

Usage::

    cambrian serve --port 8000

Backend selection
-----------------
``POST /api/evolve`` runs the **real** :class:`~cambrian.evolution.EvolutionEngine`.
The LLM backend is chosen by the ``CAMBRIAN_BACKEND`` environment variable:

* ``gemini`` (default) — Google Gemini Flash. Requires ``GEMINI_API_KEY``
  (or ``GOOGLE_API_KEY``). Free tier: ~1500 requests/day.
* ``mock`` — offline deterministic backend (no inference). For demos/CI.

If ``gemini`` is selected but no key is present, the server logs a warning and
falls back to the mock backend, tagging the run ``backend="mock"`` so the UI
can show that the numbers are simulated, not real model output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cambrian.plugin_registry import PluginRegistry

logger = logging.getLogger("cambrian.api")

# ── Global state ──────────────────────────────────────────────────────────────

DB_PATH: Path = Path("cambrian.db")

#: plugin name → enabled flag
_plugin_states: dict[str, bool] = {}

#: run_id → thread-safe event queue consumed by WebSocket handler
_run_queues: dict[str, queue.Queue[dict[str, Any]]] = {}

_registry = PluginRegistry()

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class PluginOut(BaseModel):
    name: str
    enabled: bool
    description: str
    category: str
    impact: str | None = None


class ModelOut(BaseModel):
    id: str
    name: str
    provider: str
    context_window: int
    speed: str
    cost_per_1k: float
    status: str


class RunOut(BaseModel):
    id: str
    task: str
    generations: int
    population: int
    best_fitness: float
    created_at: str
    duration_s: float
    status: str


class AgentOut(BaseModel):
    id: str
    run_id: str
    generation: int
    fitness: float
    temperature: float
    strategy: str
    prompt_tokens: int
    status: str
    parent_id: str | None
    genome: str
    fitness_history: list[float]
    plugins_active: list[str]


class EvolutionConfigIn(BaseModel):
    task: str
    generations: int = 10
    population: int = 8
    model_id: str = "gpt-4o-mini"
    plugins: list[str] = []


class EvolveOut(BaseModel):
    run_id: str
    status: str
    backend: str = "gemini"
    #: Set when the requested backend was unavailable and a fallback was used.
    warning: str | None = None


# ── Built-in model catalogue (no registry file needed) ────────────────────────

_MODELS: list[ModelOut] = [
    ModelOut(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        provider="openai",
        context_window=128_000,
        speed="fast",
        cost_per_1k=0.15,
        status="online",
    ),
    ModelOut(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_window=128_000,
        speed="medium",
        cost_per_1k=2.50,
        status="online",
    ),
    ModelOut(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        context_window=200_000,
        speed="medium",
        cost_per_1k=3.00,
        status="online",
    ),
    ModelOut(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_window=200_000,
        speed="fast",
        cost_per_1k=0.25,
        status="online",
    ),
    ModelOut(
        id="llama-3.1-70b-versatile",
        name="Llama 3.1 70B",
        provider="groq",
        context_window=131_072,
        speed="fast",
        cost_per_1k=0.59,
        status="online",
    ),
    ModelOut(
        id="mixtral-8x7b-32768",
        name="Mixtral 8x7B",
        provider="groq",
        context_window=32_768,
        speed="fast",
        cost_per_1k=0.24,
        status="online",
    ),
    ModelOut(
        id="local-ollama",
        name="Local (Ollama)",
        provider="local",
        context_window=8_192,
        speed="slow",
        cost_per_1k=0.00,
        status="offline",
    ),
]

# ── SQLite helpers ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    task        TEXT    NOT NULL,
    generations INTEGER NOT NULL,
    population  INTEGER NOT NULL,
    best_fitness REAL   DEFAULT 0.0,
    created_at  TEXT    NOT NULL,
    duration_s  REAL    DEFAULT 0.0,
    status      TEXT    DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    run_id        TEXT    NOT NULL REFERENCES runs(id),
    generation    INTEGER NOT NULL,
    fitness       REAL    DEFAULT 0.0,
    temperature   REAL    DEFAULT 0.7,
    strategy      TEXT    DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    status        TEXT    DEFAULT 'active',
    parent_id     TEXT,
    genome        TEXT    DEFAULT '',
    fitness_history TEXT  DEFAULT '[]',
    plugins_active  TEXT  DEFAULT '[]'
);
"""


def _init_db(path: Path) -> None:
    """Create tables if they don't exist."""
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA)


def _get_conn(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── DB operations ─────────────────────────────────────────────────────────────


def _insert_run(run_id: str, task: str, generations: int, population: int, path: Path | None = None) -> None:
    with _get_conn(path) as conn:
        conn.execute(
            "INSERT INTO runs (id, task, generations, population, created_at, status) VALUES (?,?,?,?,?,?)",
            (run_id, task, generations, population, _now(), "running"),
        )


def _fetch_run(run_id: str, path: Path | None = None) -> sqlite3.Row | None:
    with _get_conn(path) as conn:
        return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def _fetch_all_runs(path: Path | None = None) -> list[sqlite3.Row]:
    with _get_conn(path) as conn:
        return conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()


def _update_run(
    run_id: str, best_fitness: float, duration_s: float, status: str, path: Path | None = None
) -> None:
    with _get_conn(path) as conn:
        conn.execute(
            "UPDATE runs SET best_fitness=?, duration_s=?, status=? WHERE id=?",
            (best_fitness, duration_s, status, run_id),
        )


def _delete_run(run_id: str, path: Path | None = None) -> int:
    with _get_conn(path) as conn:
        conn.execute("DELETE FROM agents WHERE run_id=?", (run_id,))
        cur = conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return cur.rowcount


def _insert_agent(agent: dict[str, Any], path: Path | None = None) -> None:
    with _get_conn(path) as conn:
        conn.execute(
            """INSERT INTO agents
               (id, run_id, generation, fitness, temperature, strategy,
                prompt_tokens, status, parent_id, genome,
                fitness_history, plugins_active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                agent["id"],
                agent["run_id"],
                agent["generation"],
                agent["fitness"],
                agent["temperature"],
                agent["strategy"],
                agent["prompt_tokens"],
                agent["status"],
                agent["parent_id"],
                agent["genome"],
                json.dumps(agent["fitness_history"]),
                json.dumps(agent["plugins_active"]),
            ),
        )


def _update_agent_status(agent_id: str, status: str, path: Path | None = None) -> None:
    with _get_conn(path) as conn:
        conn.execute("UPDATE agents SET status=? WHERE id=?", (status, agent_id))


def _fetch_agents_for_run(run_id: str, path: Path | None = None) -> list[sqlite3.Row]:
    with _get_conn(path) as conn:
        return conn.execute(
            "SELECT * FROM agents WHERE run_id=? ORDER BY generation, fitness DESC",
            (run_id,),
        ).fetchall()


def _fetch_agent(agent_id: str, path: Path | None = None) -> sqlite3.Row | None:
    with _get_conn(path) as conn:
        return conn.execute(
            "SELECT * FROM agents WHERE id=?", (agent_id,)
        ).fetchone()


# ── Conversion helpers ────────────────────────────────────────────────────────


def _row_to_run(row: sqlite3.Row) -> RunOut:
    return RunOut(
        id=row["id"],
        task=row["task"],
        generations=row["generations"],
        population=row["population"],
        best_fitness=row["best_fitness"],
        created_at=row["created_at"],
        duration_s=row["duration_s"],
        status=row["status"],
    )


def _row_to_agent(row: sqlite3.Row) -> AgentOut:
    return AgentOut(
        id=row["id"],
        run_id=row["run_id"],
        generation=row["generation"],
        fitness=row["fitness"],
        temperature=row["temperature"],
        strategy=row["strategy"],
        prompt_tokens=row["prompt_tokens"],
        status=row["status"],
        parent_id=row["parent_id"],
        genome=row["genome"],
        fitness_history=json.loads(row["fitness_history"]),
        plugins_active=json.loads(row["plugins_active"]),
    )


# ── Backend resolution ────────────────────────────────────────────────────────


def _resolve_backend() -> tuple[Any, str, bool, str | None]:
    """Pick the LLM backend from ``CAMBRIAN_BACKEND`` (default ``gemini``).

    Returns ``(backend, label, is_mock, warning)``. Falls back to the offline
    :class:`~cambrian.backends.mock.MockBackend` (with a warning) when Gemini
    is requested but no API key is configured.
    """
    from cambrian.backends.mock import MockBackend

    choice = os.getenv("CAMBRIAN_BACKEND", "gemini").strip().lower()
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if choice == "mock":
        return MockBackend(), "mock", True, None

    if choice == "gemini":
        if not has_key:
            warning = (
                "CAMBRIAN_BACKEND=gemini but no GEMINI_API_KEY/GOOGLE_API_KEY set — "
                "using mock backend (results are simulated, NOT real LLM output). "
                "Set GEMINI_API_KEY for a real run."
            )
            logger.warning(warning)
            return MockBackend(), "mock", True, warning
        from cambrian.backends.gemini import GeminiBackend

        return GeminiBackend(model="gemini-2.0-flash"), "gemini-2.0-flash", False, None

    # Unknown value — fail safe to mock with a warning rather than crash.
    warning = f"Unknown CAMBRIAN_BACKEND={choice!r}; using mock backend."
    logger.warning(warning)
    return MockBackend(), "mock", True, warning


def _mock_evaluator(agent: Any, task: str) -> float:
    """Deterministic offline fitness over the genome (no LLM call).

    Used only with the mock backend so the *real* EvolutionEngine still has a
    landscape to climb (exercising selection/mutation/elitism for real). The
    score is a reproducible function of the genome — it measures the search
    machinery, NOT model capability. Runs tagged ``backend="mock"`` advertise
    this so the UI never presents these numbers as real.
    """
    genome = agent.genome
    prompt = genome.system_prompt.lower()
    # Reward task-relevant keywords + reasonable length + moderate temperature.
    keywords = [w for w in task.lower().split() if len(w) > 3]
    hits = sum(1 for kw in set(keywords) if kw in prompt)
    kw_score = hits / max(1, len(set(keywords)))
    len_score = min(1.0, len(prompt) / 600.0)
    temp_score = 1.0 - abs(genome.temperature - 0.7) / 1.4
    return round(0.5 * kw_score + 0.3 * len_score + 0.2 * temp_score, 4)


def _run_evolution(
    run_id: str,
    task: str,
    generations: int,
    population: int,
    plugins: list[str],
    db_path: Path,
    backend: Any,
    backend_label: str,
    is_mock: bool,
) -> None:
    """Run the real :class:`EvolutionEngine` and stream events to the WS queue.

    Pushes WSMessage-compatible dicts into ``_run_queues[run_id]`` and persists
    agents + final run stats to SQLite. *db_path* is captured at call time so
    daemon threads don't pick up a stale global if a test resets ``DB_PATH``.
    """
    from cambrian.agent import Genome
    from cambrian.evolution import EvolutionEngine
    from cambrian.mutator import LLMMutator

    q = _run_queues[run_id]
    start = time.monotonic()

    if is_mock:
        evaluator: Any = _mock_evaluator
    else:
        from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

        evaluator = LLMJudgeEvaluator(judge_backend=backend)

    # Deterministic seed per run for reproducibility.
    seed = int(uuid.UUID(run_id).int % (2**32))
    mutator = LLMMutator(backend=backend, mutation_temperature=0.6)
    engine = EvolutionEngine(
        evaluator=evaluator,
        mutator=mutator,
        backend=backend,
        population_size=population,
        seed=seed,
    )

    # ── Plugin loading: register requested operators on the engine hooks ──────
    active_plugins: list[str] = []
    if plugins:
        try:
            registry = PluginRegistry()
            registry.enable(plugins, engine)
            active_plugins = registry.active_plugins
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable plugins %s: %s", plugins, exc)

    persisted_ids: set[str] = set()

    def _persist_and_stream(gen: int, pop: list[Any]) -> None:
        gen_agents: list[dict[str, Any]] = []
        for a in pop:
            fitness = round(float(a.fitness or 0.0), 4)
            agent: dict[str, Any] = {
                "id": a.id,
                "run_id": run_id,
                "generation": gen,
                "fitness": fitness,
                "temperature": round(float(a.genome.temperature), 2),
                "strategy": a.genome.strategy,
                "prompt_tokens": a.genome.token_count(),
                "status": "active",
                "parent_id": None,
                "genome": a.genome.system_prompt,
                "fitness_history": [fitness],
                "plugins_active": active_plugins[:3],
            }
            if a.id not in persisted_ids:
                _insert_agent(agent, db_path)
                persisted_ids.add(a.id)
            gen_agents.append(agent)
            q.put({
                "type": "agent_update",
                "payload": {k: v for k, v in agent.items() if k != "run_id"},
            })

        best_gen = max(gen_agents, key=lambda x: x["fitness"])
        avg_fitness = round(sum(x["fitness"] for x in gen_agents) / len(gen_agents), 4)
        q.put({
            "type": "generation_complete",
            "payload": {
                "generation": gen,
                "best_fitness": best_gen["fitness"],
                "avg_fitness": avg_fitness,
                "backend": backend_label,
                "agent_ids": [x["id"] for x in gen_agents],
            },
        })

    try:
        seed_genome = Genome(
            system_prompt="You are a precise, knowledgeable AI assistant.",
            model=backend_label,
        )
        best = engine.evolve(
            seed_genomes=[seed_genome],
            task=task,
            n_generations=generations,
            on_generation=_persist_and_stream,
        )

        # Mark top 20 % across all persisted agents as elite.
        rows = _fetch_agents_for_run(run_id, db_path)
        ranked = sorted(rows, key=lambda r: r["fitness"], reverse=True)
        elite_n = max(1, len(ranked) // 5)
        for r in ranked[:elite_n]:
            _update_agent_status(r["id"], "elite", db_path)

        best_fitness = round(float(best.fitness or 0.0), 4) if best else 0.0
        duration = round(time.monotonic() - start, 3)
        _update_run(run_id, best_fitness, duration, "completed", db_path)

        q.put({
            "type": "run_complete",
            "payload": {
                "run_id": run_id,
                "best_fitness": best_fitness,
                "best_agent_id": best.id if best else None,
                "duration_s": duration,
                "total_agents": len(persisted_ids),
                "backend": backend_label,
                "simulated": is_mock,
            },
        })

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        logger.exception("Evolution run %s failed", run_id)
        duration = round(time.monotonic() - start, 3)
        try:
            _update_run(run_id, 0.0, duration, "failed", db_path)
        except Exception:  # noqa: BLE001
            pass
        q.put({"type": "error", "payload": {"message": err_msg, "backend": backend_label}})


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ANN001
    _init_db(DB_PATH)
    yield


app = FastAPI(
    title="Cambrian API",
    description="REST + WebSocket backend for the Cambrian evolutionary AI framework.",
    version="1.0.0",
    lifespan=_lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


# ── Plugins ────────────────────────────────────────────────────────────────────


def _all_plugin_meta() -> list[dict[str, Any]]:
    return _registry.list_all()


def _plugin_meta(name: str) -> dict[str, Any] | None:
    for p in _all_plugin_meta():
        if p["name"] == name:
            return p
    return None


def _to_plugin_out(meta: dict[str, Any]) -> PluginOut:
    return PluginOut(
        name=meta["name"],
        enabled=_plugin_states.get(meta["name"], False),
        description=meta["description"],
        category=meta["category"],
        impact=meta.get("impact"),
    )


@app.get("/api/plugins", response_model=list[PluginOut])
def list_plugins() -> list[PluginOut]:
    return [_to_plugin_out(m) for m in _all_plugin_meta()]


@app.get("/api/plugins/{name}", response_model=PluginOut)
def get_plugin(name: str) -> PluginOut:
    meta = _plugin_meta(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")
    return _to_plugin_out(meta)


@app.post("/api/plugins/{name}/enable", response_model=PluginOut)
def enable_plugin(name: str) -> PluginOut:
    meta = _plugin_meta(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")
    _plugin_states[name] = True
    return _to_plugin_out(meta)


@app.post("/api/plugins/{name}/disable", response_model=PluginOut)
def disable_plugin(name: str) -> PluginOut:
    meta = _plugin_meta(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")
    _plugin_states[name] = False
    return _to_plugin_out(meta)


# ── Models ─────────────────────────────────────────────────────────────────────


@app.get("/api/models", response_model=list[ModelOut])
def list_models() -> list[ModelOut]:
    return _MODELS


# ── Runs ───────────────────────────────────────────────────────────────────────


@app.get("/api/runs", response_model=list[RunOut])
def list_runs() -> list[RunOut]:
    return [_row_to_run(r) for r in _fetch_all_runs()]


@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str) -> RunOut:
    row = _fetch_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _row_to_run(row)


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    deleted = _delete_run(run_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    _run_queues.pop(run_id, None)


# ── Evolve ─────────────────────────────────────────────────────────────────────


@app.post("/api/evolve", response_model=EvolveOut, status_code=201)
def start_evolution(config: EvolutionConfigIn) -> EvolveOut:
    """Create a run record and launch a real EvolutionEngine in a background thread."""
    run_id = str(uuid.uuid4())

    # Capture DB_PATH at request time so the daemon thread is not affected
    # if the module-level variable changes later (e.g. in tests).
    db_path = DB_PATH
    _init_db(db_path)
    _insert_run(run_id, config.task, config.generations, config.population, db_path)

    backend, label, is_mock, warning = _resolve_backend()

    q: queue.Queue[dict[str, Any]] = queue.Queue()
    _run_queues[run_id] = q

    thread = threading.Thread(
        target=_run_evolution,
        args=(
            run_id,
            config.task,
            config.generations,
            config.population,
            config.plugins,
            db_path,
            backend,
            label,
            is_mock,
        ),
        daemon=True,
        name=f"evo-{run_id[:8]}",
    )
    thread.start()

    return EvolveOut(run_id=run_id, status="running", backend=label, warning=warning)


# ── Agents ─────────────────────────────────────────────────────────────────────


@app.get("/api/runs/{run_id}/agents", response_model=list[AgentOut])
def get_run_agents(run_id: str) -> list[AgentOut]:
    if _fetch_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    # Block briefly to let the evolution thread produce at least one agent
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        rows = _fetch_agents_for_run(run_id)
        if rows:
            return [_row_to_agent(r) for r in rows]
        time.sleep(0.05)
    return []


@app.get("/api/agents/{agent_id}", response_model=AgentOut)
def get_agent(agent_id: str) -> AgentOut:
    row = _fetch_agent(agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _row_to_agent(row)


# ── WebSocket ──────────────────────────────────────────────────────────────────


@app.websocket("/ws/evolve/{run_id}")
async def ws_evolve(websocket: WebSocket, run_id: str) -> None:
    """Stream live evolution events to the client.

    Messages conform to the ``WSMessage`` TypeScript interface::

        { type: 'agent_update' | 'generation_complete' | 'run_complete' | 'error',
          payload: unknown }
    """
    if run_id not in _run_queues:
        await websocket.close(code=1008)
        return

    q = _run_queues[run_id]
    await websocket.accept()

    try:
        while True:
            # Non-blocking poll; yield to event loop between attempts so other
            # coroutines (e.g. send_text) are not starved.
            try:
                event = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            await websocket.send_text(json.dumps(event))

            if event.get("type") in ("run_complete", "error"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        # Explicit close signals EOF to the client so receive_text() raises
        # WebSocketDisconnect instead of blocking indefinitely.
        try:
            await websocket.close()
        except Exception:
            pass


# ── Utilities ──────────────────────────────────────────────────────────────────


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()
