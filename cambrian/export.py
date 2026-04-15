# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Export evolved agents in standard formats for deployment.

After evolution, the best genome is typically used to create a deployable
agent.  This module supports three export formats:

1. **Standalone script** — a self-contained Python script that loads the
   genome and runs the agent without the Cambrian framework.

2. **MCP server** — an MCP-compatible JSON server stub that exposes the
   agent as a tool endpoint (``tools/call`` protocol).

3. **REST API** — a minimal FastAPI application that wraps the agent in an
   HTTP endpoint (``POST /run``).

Architecture
------------

:func:`export_standalone`
    Write a Python script that wraps the genome and runs it via an
    OpenAI-compatible backend.

:func:`export_mcp`
    Write an MCP JSON manifest + server stub.

:func:`export_api`
    Write a FastAPI application.

:func:`export_genome_json`
    Write the raw genome to JSON (for ``cambrian run --agent``).

:func:`load_genome_json`
    Load a genome from a JSON file.

Usage::

    from cambrian.export import export_standalone, export_genome_json
    from cambrian.agent import Agent

    # Save the best genome
    export_genome_json(best_agent, "best_genome.json")

    # Export as a standalone script
    export_standalone(best_agent, "my_agent.py")

    # Export as an MCP server stub
    export_mcp(best_agent, output_dir="mcp_server/")

    # Export as a FastAPI app
    export_api(best_agent, "api_agent.py")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cambrian.agent import Agent, Genome
from cambrian.utils.logging import get_logger

logger = get_logger(__name__)


# ── JSON serialisation ─────────────────────────────────────────────────────────


def export_genome_json(agent: Agent, path: str | Path) -> Path:
    """Serialise the agent's genome to a JSON file.

    Args:
        agent: Agent whose genome to export.
        path: Destination file path.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "genome": agent.genome.to_dict(),
        "fitness": agent.fitness,
        "agent_id": agent.id,
        "cambrian_version": "0.18.0",
    }
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Genome exported to %s", dest)
    return dest


def load_genome_json(path: str | Path) -> Genome:
    """Load a genome from a JSON file written by :func:`export_genome_json`.

    Args:
        path: Path to the JSON file.

    Returns:
        Reconstructed :class:`~cambrian.agent.Genome`.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If the JSON is missing the ``"genome"`` key.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Genome file not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    return Genome.from_dict(data["genome"])


# ── Standalone script ──────────────────────────────────────────────────────────

_STANDALONE_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated standalone agent — exported by Cambrian {version}.

Genome ID : {genome_id}
Model     : {model}
Fitness   : {fitness}
"""

import os
import httpx

SYSTEM_PROMPT = """{system_prompt}"""
STRATEGY      = "{strategy}"
MODEL         = "{model}"
TEMPERATURE   = {temperature}

API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("OPENAI_API_KEY", "")


def run(task: str, max_tokens: int = 1024) -> str:
    """Run the agent on *task* and return its response."""
    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT + "\\n\\n" + STRATEGY}},
        {{"role": "user",   "content": task}},
    ]
    headers = {{
        "Authorization": f"Bearer {{API_KEY}}",
        "Content-Type":  "application/json",
    }}
    payload = {{
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }}
    resp = httpx.post(f"{{API_BASE}}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "Hello, who are you?"
    print(run(task))
'''


def export_standalone(agent: Agent, path: str | Path) -> Path:
    """Export the agent as a self-contained Python script.

    Args:
        agent: Agent to export.
        path: Destination ``.py`` file.

    Returns:
        Resolved path of the written file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    genome = agent.genome
    content = _STANDALONE_TEMPLATE.format(
        version="0.18.0",
        genome_id=genome.genome_id,
        model=genome.model,
        fitness=round(agent.fitness or 0.0, 4),
        system_prompt=genome.system_prompt.replace('"', '\\"'),
        strategy=genome.strategy.replace('"', '\\"'),
        temperature=genome.temperature,
    )
    dest.write_text(content, encoding="utf-8")
    logger.info("Standalone agent exported to %s", dest)
    return dest


# ── MCP server stub ────────────────────────────────────────────────────────────

_MCP_MANIFEST_TEMPLATE = """\
{{
  "schema_version": "v1",
  "name": "cambrian-agent-{genome_id}",
  "version": "1.0.0",
  "description": "Evolved Cambrian agent (model={model}, fitness={fitness})",
  "tools": [
    {{
      "name": "run_agent",
      "description": "Run the evolved agent on a task.",
      "inputSchema": {{
        "type": "object",
        "properties": {{
          "task": {{"type": "string", "description": "The task for the agent."}}
        }},
        "required": ["task"]
      }}
    }}
  ]
}}
"""

_MCP_SERVER_TEMPLATE = '''\
#!/usr/bin/env python3
"""MCP server stub for a Cambrian-evolved agent.

Start with: python mcp_server.py
Protocol   : MCP tools/call over stdio (JSON-RPC 2.0)
"""
import json
import sys
import os

SYSTEM_PROMPT = """{system_prompt}"""
MODEL         = "{model}"
TEMPERATURE   = {temperature}

API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("OPENAI_API_KEY", "")


def _run_agent(task: str) -> str:
    import httpx
    resp = httpx.post(
        f"{{API_BASE}}/chat/completions",
        headers={{"Authorization": f"Bearer {{API_KEY}}", "Content-Type": "application/json"}},
        json={{
            "model": MODEL, "temperature": TEMPERATURE,
            "messages": [
                {{"role": "system", "content": SYSTEM_PROMPT}},
                {{"role": "user",   "content": task}},
            ],
            "max_tokens": 1024,
        }},
        timeout=60,
    )
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def handle(request: dict) -> dict:
    method = request.get("method", "")
    if method == "tools/list":
        return {{"result": {{"tools": [{{"name": "run_agent", "description": "Run the agent."}}]}}}}
    if method == "tools/call":
        args = request.get("params", {{}}).get("arguments", {{}})
        output = _run_agent(args.get("task", ""))
        return {{"result": {{"content": [{{"type": "text", "text": output}}]}}}}
    return {{"error": {{"code": -32601, "message": "Method not found"}}}}


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            req = json.loads(line)
            resp = handle(req)
            resp["id"] = req.get("id")
            print(json.dumps(resp), flush=True)
        except Exception as exc:
            print(json.dumps({{"error": str(exc)}}), flush=True)
'''


def export_mcp(agent: Agent, output_dir: str | Path) -> Path:
    """Export an MCP server stub for the agent.

    Creates two files in *output_dir*:
    - ``manifest.json`` — MCP capability manifest
    - ``mcp_server.py`` — server script

    Args:
        agent: Agent to export.
        output_dir: Directory to write files into.

    Returns:
        Path to *output_dir*.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    genome = agent.genome

    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        _MCP_MANIFEST_TEMPLATE.format(
            genome_id=genome.genome_id,
            model=genome.model,
            fitness=round(agent.fitness or 0.0, 4),
        ),
        encoding="utf-8",
    )

    server_path = dest / "mcp_server.py"
    server_path.write_text(
        _MCP_SERVER_TEMPLATE.format(
            system_prompt=genome.system_prompt.replace('"', '\\"'),
            model=genome.model,
            temperature=genome.temperature,
        ),
        encoding="utf-8",
    )

    logger.info("MCP server stub exported to %s", dest)
    return dest


# ── FastAPI REST app ───────────────────────────────────────────────────────────

_API_TEMPLATE = '''\
#!/usr/bin/env python3
"""FastAPI REST wrapper for a Cambrian-evolved agent.

Install:  pip install fastapi uvicorn httpx
Run:      uvicorn api_agent:app --port 8000
Endpoint: POST /run  {{"task": "..."}}
"""
import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

SYSTEM_PROMPT = """{system_prompt}"""
MODEL         = "{model}"
TEMPERATURE   = {temperature}

API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("OPENAI_API_KEY", "")

app = FastAPI(title="Cambrian Agent", version="1.0.0")


class RunRequest(BaseModel):
    task: str
    max_tokens: int = 1024


class RunResponse(BaseModel):
    result: str
    model: str
    fitness: float


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    resp = httpx.post(
        f"{{API_BASE}}/chat/completions",
        headers={{"Authorization": f"Bearer {{API_KEY}}", "Content-Type": "application/json"}},
        json={{
            "model": MODEL, "temperature": TEMPERATURE,
            "messages": [
                {{"role": "system", "content": SYSTEM_PROMPT}},
                {{"role": "user",   "content": req.task}},
            ],
            "max_tokens": req.max_tokens,
        }},
        timeout=60,
    )
    resp.raise_for_status()
    content = str(resp.json()["choices"][0]["message"]["content"]).strip()
    return RunResponse(result=content, model=MODEL, fitness={fitness})


@app.get("/health")
def health() -> dict:
    return {{"status": "ok", "model": MODEL}}
'''


def export_api(agent: Agent, path: str | Path) -> Path:
    """Export the agent as a FastAPI application.

    Args:
        agent: Agent to export.
        path: Destination ``.py`` file.

    Returns:
        Resolved path of the written file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    genome = agent.genome
    content = _API_TEMPLATE.format(
        system_prompt=genome.system_prompt.replace('"', '\\"'),
        model=genome.model,
        temperature=genome.temperature,
        fitness=round(agent.fitness or 0.0, 4),
    )
    dest.write_text(content, encoding="utf-8")
    logger.info("FastAPI agent exported to %s", dest)
    return dest
