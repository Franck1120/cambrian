# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for cambrian.api — FastAPI REST + WebSocket server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient backed by a fresh per-test SQLite DB."""
    import cambrian.api as api_mod

    db_file = tmp_path / "test_cambrian.db"
    monkeypatch.setattr(api_mod, "DB_PATH", db_file)
    monkeypatch.setattr(api_mod, "_plugin_states", {})
    monkeypatch.setattr(api_mod, "_run_queues", {})
    api_mod._init_db(db_file)

    return TestClient(api_mod.app, raise_server_exceptions=True)


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Plugins ───────────────────────────────────────────────────────────────────


def test_get_plugins_returns_list(client: TestClient) -> None:
    resp = client.get("/api/plugins")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Each plugin has required fields
    for p in data:
        assert "name" in p
        assert "enabled" in p
        assert "description" in p
        assert "category" in p


def test_enable_plugin(client: TestClient) -> None:
    resp = client.post("/api/plugins/dream/enable")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert resp.json()["name"] == "dream"


def test_disable_plugin(client: TestClient) -> None:
    client.post("/api/plugins/dream/enable")
    resp = client.post("/api/plugins/dream/disable")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_enable_unknown_plugin_returns_404(client: TestClient) -> None:
    resp = client.post("/api/plugins/totally_fake_plugin_xyz/enable")
    assert resp.status_code == 404


def test_get_single_plugin(client: TestClient) -> None:
    resp = client.get("/api/plugins/dream")
    assert resp.status_code == 200
    assert resp.json()["name"] == "dream"


def test_get_unknown_plugin_returns_404(client: TestClient) -> None:
    resp = client.get("/api/plugins/nonexistent_xyz")
    assert resp.status_code == 404


# ── Models ────────────────────────────────────────────────────────────────────


def test_get_models_returns_list(client: TestClient) -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for m in data:
        assert "id" in m
        assert "name" in m
        assert "provider" in m


# ── Runs ──────────────────────────────────────────────────────────────────────


def test_list_runs_empty(client: TestClient) -> None:
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_run_via_evolve(client: TestClient) -> None:
    payload = {
        "task": "Write a Python hello-world function",
        "generations": 2,
        "population": 4,
        "model_id": "gpt-4o-mini",
        "plugins": [],
    }
    resp = client.post("/api/evolve", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert "run_id" in data
    assert data["status"] in ("running", "completed")


def test_get_run_after_create(client: TestClient) -> None:
    run_id = client.post(
        "/api/evolve", json={"task": "Add two numbers", "generations": 1}
    ).json()["run_id"]

    resp = client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == run_id
    assert data["task"] == "Add two numbers"
    assert "status" in data


def test_get_run_not_found(client: TestClient) -> None:
    resp = client.get("/api/runs/does-not-exist-00000")
    assert resp.status_code == 404


def test_list_runs_after_two_creates(client: TestClient) -> None:
    client.post("/api/evolve", json={"task": "Task A", "generations": 1})
    client.post("/api/evolve", json={"task": "Task B", "generations": 1})
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_run(client: TestClient) -> None:
    run_id = client.post(
        "/api/evolve", json={"task": "Temporary", "generations": 1}
    ).json()["run_id"]
    resp = client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 204
    # Confirm gone
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_delete_run_not_found(client: TestClient) -> None:
    resp = client.delete("/api/runs/ghost-run-id")
    assert resp.status_code == 404


# ── Agents ────────────────────────────────────────────────────────────────────


def test_get_run_agents_returns_list(client: TestClient) -> None:
    run_id = client.post(
        "/api/evolve", json={"task": "T", "generations": 1, "population": 3}
    ).json()["run_id"]
    resp = client.get(f"/api/runs/{run_id}/agents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_agent_detail(client: TestClient) -> None:
    run_id = client.post(
        "/api/evolve", json={"task": "T", "generations": 1, "population": 2}
    ).json()["run_id"]
    agents = client.get(f"/api/runs/{run_id}/agents").json()
    if agents:
        agent_id = agents[0]["id"]
        resp = client.get(f"/api/agents/{agent_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == agent_id
        assert "genome" in resp.json()
        assert "fitness_history" in resp.json()


def test_get_agent_not_found(client: TestClient) -> None:
    resp = client.get("/api/agents/nonexistent-agent-id")
    assert resp.status_code == 404


# ── WebSocket ─────────────────────────────────────────────────────────────────


def test_websocket_evolve_connect_and_messages(client: TestClient) -> None:
    """WebSocket /ws/evolve/{run_id} delivers at least one message.

    Starlette's WebSocketTestSession.receive_text() does not support a
    ``timeout`` argument, so we rely on the evolution thread finishing
    quickly (1 generation × 2 agents → 4 events) before the handler closes.
    """
    run_id = client.post(
        "/api/evolve", json={"task": "WS task", "generations": 1, "population": 2}
    ).json()["run_id"]

    messages: list[dict] = []
    with client.websocket_connect(f"/ws/evolve/{run_id}") as ws:
        # Read until server closes the connection (WebSocketDisconnect) or 8 msgs
        for _ in range(8):
            try:
                raw = ws.receive_text()
                messages.append(json.loads(raw))
            except Exception:
                break

    assert len(messages) > 0, "Expected at least one WSMessage from the server"
    # Every message must conform to the WSMessage interface
    for msg in messages:
        assert "type" in msg
        assert "payload" in msg


def test_websocket_run_complete_message(client: TestClient) -> None:
    """At least one message of type agent_update/generation_complete/run_complete."""
    run_id = client.post(
        "/api/evolve", json={"task": "WS task 2", "generations": 1, "population": 2}
    ).json()["run_id"]

    received_types: set[str] = set()
    with client.websocket_connect(f"/ws/evolve/{run_id}") as ws:
        for _ in range(10):
            try:
                msg = json.loads(ws.receive_text())
                received_types.add(msg["type"])
                if msg["type"] == "run_complete":
                    break
            except Exception:
                break

    assert received_types & {"generation_complete", "run_complete", "agent_update"}
