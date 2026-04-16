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
"""

from __future__ import annotations

import asyncio
import json
import queue
import random
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


# ── Stub evolution (runs in background thread) ────────────────────────────────

_STRATEGIES = ["reflexion", "chain-of-thought", "tree-of-thought", "react", "direct", "moa"]


def _stub_evolution(
    run_id: str,
    task: str,
    generations: int,
    population: int,
    plugins: list[str],
    db_path: Path,
) -> None:
    """Simulate an evolutionary run without a real LLM backend.

    Pushes WSMessage-compatible dicts into ``_run_queues[run_id]``.
    Persists agents and final run stats to SQLite.

    *db_path* is captured at call time so daemon threads don't pick up
    a stale global if the caller (e.g. a test) resets ``DB_PATH`` later.
    """
    q = _run_queues[run_id]
    start = time.monotonic()
    rng = random.Random(run_id)

    all_agents: list[dict[str, Any]] = []
    base_fitness = rng.uniform(0.25, 0.45)

    try:
        for gen in range(generations):
            gen_agents: list[dict[str, Any]] = []

            for i in range(population):
                agent_id = str(uuid.uuid4())
                # Fitness gradually improves, with noise
                noise = rng.uniform(-0.03, 0.05)
                fitness = min(0.99, base_fitness + gen * rng.uniform(0.04, 0.10) + noise)
                fitness = max(0.0, round(fitness, 4))

                parent_id = all_agents[rng.randrange(len(all_agents))]["id"] if all_agents else None
                strategy = rng.choice(_STRATEGIES)
                temperature = round(rng.uniform(0.5, 1.2), 2)
                prompt_tokens = rng.randint(120, 480)

                fitness_history = [
                    round(max(0.0, fitness - (gen - j) * rng.uniform(0.04, 0.08)), 4)
                    for j in range(gen + 1)
                ]

                agent: dict[str, Any] = {
                    "id": agent_id,
                    "run_id": run_id,
                    "generation": gen,
                    "fitness": fitness,
                    "temperature": temperature,
                    "strategy": strategy,
                    "prompt_tokens": prompt_tokens,
                    "status": "active",
                    "parent_id": parent_id,
                    "genome": (
                        f"You are an AI agent evolving to solve: {task}\n\n"
                        f"Strategy: {strategy} | Gen: {gen} | Temp: {temperature}"
                    ),
                    "fitness_history": fitness_history,
                    "plugins_active": plugins[:3],
                }

                _insert_agent(agent, db_path)
                gen_agents.append(agent)
                all_agents.append(agent)

                q.put({
                    "type": "agent_update",
                    "payload": {k: v for k, v in agent.items() if k != "run_id"},
                })

            best_gen = max(gen_agents, key=lambda a: a["fitness"])
            avg_fitness = round(sum(a["fitness"] for a in gen_agents) / len(gen_agents), 4)

            q.put({
                "type": "generation_complete",
                "payload": {
                    "generation": gen,
                    "best_fitness": best_gen["fitness"],
                    "avg_fitness": avg_fitness,
                    "diversity": round(rng.uniform(0.3, 0.8), 3),
                    "agent_ids": [a["id"] for a in gen_agents],
                },
            })

        # Mark top 20 % as elite
        sorted_all = sorted(all_agents, key=lambda a: a["fitness"], reverse=True)
        elite_n = max(1, len(sorted_all) // 5)
        for a in sorted_all[:elite_n]:
            _update_agent_status(a["id"], "elite", db_path)

        best_overall = sorted_all[0]
        duration = round(time.monotonic() - start, 3)
        _update_run(run_id, best_overall["fitness"], duration, "completed", db_path)

        q.put({
            "type": "run_complete",
            "payload": {
                "run_id": run_id,
                "best_fitness": best_overall["fitness"],
                "best_agent_id": best_overall["id"],
                "duration_s": duration,
                "total_agents": len(all_agents),
            },
        })

    except Exception as exc:
        err_msg = str(exc)
        duration = round(time.monotonic() - start, 3)
        try:
            _update_run(run_id, 0.0, duration, "failed", db_path)
        except Exception:
            pass
        q.put({"type": "error", "payload": {"message": err_msg}})


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
    """Create a run record and launch stub evolution in a background thread."""
    run_id = str(uuid.uuid4())

    # Capture DB_PATH at request time so the daemon thread is not affected
    # if the module-level variable changes later (e.g. in tests).
    db_path = DB_PATH
    _init_db(db_path)
    _insert_run(run_id, config.task, config.generations, config.population, db_path)

    q: queue.Queue[dict[str, Any]] = queue.Queue()
    _run_queues[run_id] = q

    thread = threading.Thread(
        target=_stub_evolution,
        args=(run_id, config.task, config.generations, config.population, config.plugins, db_path),
        daemon=True,
        name=f"evo-{run_id[:8]}",
    )
    thread.start()

    return EvolveOut(run_id=run_id, status="running")


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
