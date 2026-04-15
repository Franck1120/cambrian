# Environment Variables Reference

Cambrian reads configuration from environment variables. All are optional
unless stated otherwise.

---

## Core API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(none)* | API key for your LLM provider. Required for any command that calls an LLM. Accepted by all major providers (OpenAI, Anthropic, Groq, Together.ai). For local Ollama, set to any non-empty string (e.g. `ollama`). |
| `CAMBRIAN_BASE_URL` | `https://api.openai.com/v1` | Base URL of the OpenAI-compatible API endpoint. Override to use any provider. |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | Alias used in exported agent scripts (standalone, API, MCP). |

### Provider Examples

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export CAMBRIAN_BASE_URL="https://api.openai.com/v1"   # (default, can omit)

# Anthropic Claude via OpenAI-compatible proxy
export OPENAI_API_KEY="sk-ant-..."
export CAMBRIAN_BASE_URL="https://api.anthropic.com/v1"

# Groq
export OPENAI_API_KEY="gsk_..."
export CAMBRIAN_BASE_URL="https://api.groq.com/openai/v1"

# Together.ai
export OPENAI_API_KEY="..."
export CAMBRIAN_BASE_URL="https://api.together.xyz/v1"

# Local Ollama (no real key needed)
export OPENAI_API_KEY="ollama"
export CAMBRIAN_BASE_URL="http://localhost:11434/v1"

# LM Studio
export OPENAI_API_KEY="lm-studio"
export CAMBRIAN_BASE_URL="http://localhost:1234/v1"
```

---

## CLI Overrides

All environment variables can be overridden with explicit CLI flags:

| Env var | CLI flag |
|---------|----------|
| `OPENAI_API_KEY` | `--api-key` |
| `CAMBRIAN_BASE_URL` | `--base-url` |

CLI flags take precedence over environment variables.

---

## Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMBRIAN_LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `PYTHONDONTWRITEBYTECODE` | *(unset)* | Standard Python: prevents `.pyc` generation in sandboxed code evaluation. |

---

## Sandbox Security

The `CodeEvaluator` runs agent-generated code in a subprocess sandbox.
The sandbox inherits only these environment variables from the parent:

| Variable | Purpose |
|----------|---------|
| `PATH` | Required for subprocess execution |
| `PYTHONPATH` | Module resolution in sandboxed code |
| `HOME` | UNIX home directory |
| `USERPROFILE` | Windows home directory |
| `TEMP` / `TMP` | Temporary file paths |

All other variables (including `OPENAI_API_KEY`) are **stripped** from
the sandbox. Agent-generated code cannot exfiltrate credentials.

---

## Full Example: `.env` file

Cambrian does not load `.env` files automatically (no `python-dotenv`
dependency). Use your shell or a tool like `direnv`:

```bash
# .env (load with: source .env or eval $(cat .env | xargs))
OPENAI_API_KEY=sk-...
CAMBRIAN_BASE_URL=https://api.openai.com/v1
```

Or use direnv's `.envrc`:

```bash
# .envrc
export OPENAI_API_KEY="sk-..."
export CAMBRIAN_BASE_URL="https://api.openai.com/v1"
```

---

*Cambrian v1.0.2 · [GitHub](https://github.com/Franck1120/cambrian)*
