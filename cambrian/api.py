# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""FastAPI REST + WebSocket server for the Cambrian web UI.

Run with:
    uvicorn cambrian.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cambrian.plugin_registry import PluginRegistry

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Cambrian API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_registry = PluginRegistry()

# plugin_name -> enabled bool
_plugin_enabled: dict[str, bool] = {}

# run_id -> EvolutionRun dict
_runs: dict[str, dict[str, Any]] = {}

# agent_id -> Agent dict  (keyed across all runs)
_agents: dict[str, dict[str, Any]] = {}

# run_id -> list of agent_ids
_run_agents: dict[str, list[str]] = {}

# ---------------------------------------------------------------------------
# Static model catalogue (mirrors web/src/data/models.ts)
# ---------------------------------------------------------------------------

_MODELS: list[dict[str, Any]] = [
    {"id": "llama-3.3-70b", "name": "Llama 3.3 70B", "provider": "groq", "context_window": 131072, "speed": "fast", "cost_per_1k": 0.59, "status": "online"},
    {"id": "llama-3.1-8b", "name": "Llama 3.1 8B", "provider": "groq", "context_window": 131072, "speed": "fast", "cost_per_1k": 0.05, "status": "online"},
    {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "provider": "groq", "context_window": 32768, "speed": "fast", "cost_per_1k": 0.27, "status": "online"},
    {"id": "gemma2-9b", "name": "Gemma 2 9B", "provider": "groq", "context_window": 8192, "speed": "fast", "cost_per_1k": 0.20, "status": "online"},
    {"id": "qwen-qwq-32b", "name": "QwQ 32B", "provider": "groq", "context_window": 131072, "speed": "medium", "cost_per_1k": 0.29, "status": "online"},
    {"id": "deepseek-r1-distill-70b", "name": "DeepSeek R1 Distill 70B", "provider": "groq", "context_window": 131072, "speed": "medium", "cost_per_1k": 0.75, "status": "rate-limited"},
    {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "provider": "anthropic", "context_window": 200000, "speed": "fast", "cost_per_1k": 0.25, "status": "online"},
    {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "provider": "anthropic", "context_window": 200000, "speed": "medium", "cost_per_1k": 3.00, "status": "online"},
    {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "provider": "anthropic", "context_window": 200000, "speed": "slow", "cost_per_1k": 15.00, "status": "online"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai", "context_window": 128000, "speed": "fast", "cost_per_1k": 0.15, "status": "online"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai", "context_window": 128000, "speed": "medium", "cost_per_1k": 2.50, "status": "online"},
    {"id": "o1-mini", "name": "o1-mini", "provider": "openai", "context_window": 128000, "speed": "slow", "cost_per_1k": 3.00, "status": "rate-limited"},
    {"id": "ollama-llama3", "name": "Llama 3 (Ollama)", "provider": "local", "context_window": 4096, "speed": "slow", "cost_per_1k": 0.00, "status": "online"},
    {"id": "ollama-qwen2.5", "name": "Qwen 2.5 7B (Ollama)", "provider": "local", "context_window": 32768, "speed": "slow", "cost_per_1k": 0.00, "status": "offline"},
    {"id": "cli-proxy-api", "name": "CLI Proxy API", "provider": "proxy", "context_window": 200000, "speed": "medium", "cost_per_1k": 1.00, "status": "online"},
]

# ---------------------------------------------------------------------------
# Startup — discover plugins
# ---------------------------------------------------------------------------


@app.on_event("startup")  # type: ignore[misc]
def _init_plugins() -> None:
    for name in _registry.discover():
        _plugin_enabled.setdefault(name, False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PluginToggle(BaseModel):
    enabled: bool


class EvolveRequest(BaseModel):
    task: str
    generations: int = 8
    population: int = 6
    model_id: str = "llama-3.3-70b"
    plugins: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRATEGIES = ["reflexion", "chain-of-thought", "tree-of-thought", "react", "direct", "moa"]


def _make_agent(
    run_id: str,
    generation: int,
    base_fitness: float,
    parent_id: str | None,
) -> dict[str, Any]:
    agent_id = f"{run_id[:6]}-g{generation}-{uuid.uuid4().hex[:4]}"
    fitness = min(1.0, base_fitness + random.uniform(-0.05, 0.05))
    fitness_history = [
        min(1.0, (base_fitness * (i + 1) / generation) + random.uniform(-0.03, 0.03))
        for i in range(generation)
    ]
    agent: dict[str, Any] = {
        "id": agent_id,
        "generation": generation,
        "fitness": round(fitness, 4),
        "temperature": round(random.uniform(0.4, 1.0), 2),
        "strategy": random.choice(_STRATEGIES),
        "prompt_tokens": random.randint(80, 450),
        "status": "active",
        "parent_id": parent_id,
        "genome": f"Agent genome for generation {generation}",
        "fitness_history": [round(f, 4) for f in fitness_history],
        "plugins_active": [],
        "run_logs": [],
    }
    return agent


def _build_run_summary(run_id: str) -> dict[str, Any]:
    run = _runs[run_id]
    agent_list = [_agents[aid] for aid in _run_agents.get(run_id, []) if aid in _agents]
    best = max((a["fitness"] for a in agent_list), default=0.0)
    return {**run, "best_fitness": round(best, 4)}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


@app.get("/api/plugins")
def list_plugins() -> list[dict[str, Any]]:
    result = []
    for i, meta in enumerate(_registry.list_all()):
        name = meta["name"] or ""
        result.append({
            "id": name,
            "name": name,
            "description": meta.get("description") or "",
            "category": meta.get("category") or "mutation",
            "impact": None,
            "enabled": _plugin_enabled.get(name, False),
            "order": i,
            "tags": [],
            "icon": "",
        })
    return result


@app.get("/api/plugins/{name}")
def get_plugin(name: str) -> dict[str, Any]:
    for i, meta in enumerate(_registry.list_all()):
        if meta["name"] == name:
            return {
                "id": name,
                "name": name,
                "description": meta.get("description") or "",
                "category": meta.get("category") or "mutation",
                "impact": None,
                "enabled": _plugin_enabled.get(name, False),
                "order": i,
                "tags": [],
                "icon": "",
            }
    return {"error": "not found"}


@app.post("/api/plugins/{name}")
def toggle_plugin(name: str, body: PluginToggle) -> dict[str, Any]:
    _plugin_enabled[name] = body.enabled
    return {
        "id": name,
        "name": name,
        "enabled": body.enabled,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    return _MODELS


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return [_build_run_summary(rid) for rid in _runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    if run_id not in _runs:
        return {"error": "not found"}
    return _build_run_summary(run_id)


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, str]:
    _runs.pop(run_id, None)
    for aid in _run_agents.pop(run_id, []):
        _agents.pop(aid, None)
    return {"status": "deleted"}


@app.get("/api/runs/{run_id}/agents")
def get_run_agents(run_id: str) -> list[dict[str, Any]]:
    return [_agents[aid] for aid in _run_agents.get(run_id, []) if aid in _agents]


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str) -> dict[str, Any]:
    if agent_id not in _agents:
        return {"error": "not found"}
    return _agents[agent_id]


# ---------------------------------------------------------------------------
# Evolve — POST starts a run, WS streams live updates
# ---------------------------------------------------------------------------


@app.post("/api/evolve")
def start_evolve(req: EvolveRequest) -> dict[str, str]:
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "id": run_id,
        "task": req.task,
        "generations": req.generations,
        "population": req.population,
        "best_fitness": 0.0,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "duration_s": 0,
        "status": "pending",
    }
    _run_agents[run_id] = []
    return {"run_id": run_id}


@app.websocket("/ws/evolve/{run_id}")
async def ws_evolve(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    if run_id not in _runs:
        await websocket.send_json({"type": "error", "payload": {"message": "Run not found"}})
        await websocket.close()
        return

    run = _runs[run_id]
    generations: int = run["generations"]
    population: int = run["population"]
    start_time = time.time()
    parent_ids: list[str | None] = [None] * population

    run["status"] = "running"

    try:
        for gen in range(1, generations + 1):
            gen_agents: list[dict[str, Any]] = []
            base_fitness = 0.3 + (gen / generations) * 0.65

            for i in range(population):
                agent = _make_agent(run_id, gen, base_fitness, parent_ids[i % len(parent_ids)])
                # Mark top 2 as elite in last generation
                if gen == generations and i < 2:
                    agent["status"] = "elite"

                _agents[agent["id"]] = agent
                _run_agents[run_id].append(agent["id"])
                gen_agents.append(agent)

                await websocket.send_json({
                    "type": "agent_update",
                    "payload": agent,
                })
                await asyncio.sleep(0.05)

            # Update parent pool for next generation
            gen_agents.sort(key=lambda a: a["fitness"], reverse=True)
            parent_ids = [a["id"] for a in gen_agents]

            best_gen = gen_agents[0]["fitness"]
            avg_gen = sum(a["fitness"] for a in gen_agents) / len(gen_agents)
            diversity = max(0.0, 1.0 - (gen / generations) * 0.6)

            gen_stat = {
                "generation": gen,
                "best_fitness": round(best_gen, 4),
                "avg_fitness": round(avg_gen, 4),
                "diversity": round(diversity, 4),
            }

            await websocket.send_json({
                "type": "generation_complete",
                "payload": gen_stat,
            })

            await asyncio.sleep(0.2)

        duration = int(time.time() - start_time)
        all_run_agents = [_agents[aid] for aid in _run_agents.get(run_id, []) if aid in _agents]
        best_fitness = max((a["fitness"] for a in all_run_agents), default=0.0)

        run["status"] = "complete"
        run["duration_s"] = duration
        run["best_fitness"] = round(best_fitness, 4)

        await websocket.send_json({
            "type": "run_complete",
            "payload": {
                "run_id": run_id,
                "best_fitness": round(best_fitness, 4),
                "duration_s": duration,
                "total_agents": len(all_run_agents),
            },
        })

    except WebSocketDisconnect:
        run["status"] = "aborted"
    except Exception as exc:
        run["status"] = "error"
        try:
            await websocket.send_json({"type": "error", "payload": {"message": str(exc)}})
        except Exception:  # noqa: BLE001
            pass
