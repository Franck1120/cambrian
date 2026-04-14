# Cambrian Security Guidelines

> Version: 0.18.0 · Last updated: 2026-04-14

This document covers the security model, sandboxing requirements, and safe
practices for running Cambrian in production or shared environments.

---

## Core Risk: LLM-Generated Execution

Cambrian evolves AI agents by having LLMs generate and mutate **system prompts
and strategies**.  In normal operation, prompts are text-only and pose minimal
risk.  However, two features require special attention:

| Feature | Risk | Mitigation |
|---------|------|-----------|
| `CLITool` / `CLIToolkit` | Agents can invoke shell commands | Allowlist, timeout, no shell=True |
| `ToolInventor` | LLM invents new CLI tool specs | Validate name regex, dry-run before use |
| `DiffCoTReasoner` | Multi-step LLM calls, more cost | Rate limit; use short `n_steps` |
| `evolve` command | Calls external LLM API repeatedly | Set `--generations` limit |

---

## 1. Subprocess Sandboxing

### CLITool safety rules

`CLITool` wraps shell commands with `{input}` substitution.  By default:

```python
# Safe: shell=False (no shell expansion), fixed timeout
tool = CLITool(
    name="word_count",
    command_template="wc -w {input}",
    shell=False,      # NEVER set True unless you control the input
    timeout=10.0,     # Always set a timeout
)
```

**Never** set `shell=True` with user-controlled `{input}` — this enables shell
injection.  If you need shell features, use a fixed script:

```python
# Bad — shell injection risk if input comes from untrusted source
CLITool(name="run", command_template="bash -c {input!r}", shell=True)

# Better — fixed script, only accepts positional argument
CLITool(name="run", command_template="/opt/sandbox/run.sh {input}", shell=False)
```

### Recommended subprocess restrictions (Linux)

Run Cambrian agents in an isolated environment:

```bash
# Use firejail for filesystem isolation
firejail --noprofile --net=none --nosound --noroot \
  python -m cambrian evolve "..."

# Or use Docker with resource limits
docker run --rm --network=none --memory=512m --cpus=1 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  cambrian:latest python -m cambrian evolve "..."
```

### Timeout enforcement

Always configure timeouts at every layer:

```python
# subprocess level (CLITool)
tool = CLITool(name="grep", command_template="grep -r {input} .", timeout=5.0)

# backend level (API calls)
backend = OpenAICompatBackend(timeout=60, max_retries=3)
```

---

## 2. API Key Security

- Store all API keys in **environment variables**, never in code or config files.
- Use separate API keys for evolution runs and production agents.
- Set **spending limits** on your API provider accounts.
- Rotate keys if a run log is accidentally committed.

```bash
# Good
export OPENAI_API_KEY="sk-..."
cambrian evolve "task"

# Never — visible in shell history
cambrian evolve "task" --api-key "sk-..."
```

Required environment variables:

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI-compatible backends |
| `CAMBRIAN_BASE_URL` | Override API endpoint |
| `ANTHROPIC_API_KEY` | Anthropic backend |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini backend |

---

## 3. Prompt Injection

Evolved system prompts may inadvertently contain instruction-following patterns
that interfere with safety measures.

### Mitigation

1. **Constitutional AI** (`cambrian/constitutional.py`): run critique-revise
   cycles before deploying evolved prompts.

2. **Output validation**: always validate agent responses at system boundaries.
   Never pass raw LLM output to database queries or system calls without
   sanitisation.

3. **Prompt review**: inspect the final evolved system prompt before deployment:

```bash
cambrian distill best.json  # pretty-print the genome
```

4. **Red-teaming**: use adversarial co-evolution (`CoEvolutionEngine`) to test
   whether evolved agents can be jailbroken.

---

## 4. ToolInventor Safety

When `ToolInventor` is enabled, an LLM invents new CLI tool specs.  The
invented tools are **dry-run tested** before being registered, but you should:

1. **Review invented tools** before deploying them in production.
2. **Use `max_tools_per_agent`** to cap how many tools an agent accumulates.
3. **Validate command templates** — only allow a safe command allowlist in
   production:

```python
ALLOWED_COMMANDS = {"echo", "wc", "grep", "head", "tail", "sort", "uniq"}

def validate_tool(spec: ToolSpec) -> bool:
    first_word = spec.command_template.split()[0]
    return first_word in ALLOWED_COMMANDS
```

---

## 5. Lineage File Security

Lineage JSON files (`--memory-out`) contain full genome snapshots including all
system prompts.  These may contain sensitive task descriptions.

- Treat lineage files as **internal development artifacts**, not for sharing.
- Do not commit lineage files to public repositories.
- Add `lineage_*.json` to `.gitignore`.

---

## 6. Network Isolation

For maximum safety, run evolution with network isolation except for the LLM API:

```bash
# Docker approach with explicit network policy
docker run --rm --network=restricted --memory=512m \
  cambrian:latest cambrian evolve "task"
```

---

## 7. Rate Limiting and Cost Controls

Unbounded evolution runs can generate large API bills:

```bash
# Cap generation count
cambrian evolve "task" --generations 20 --population 8

# Use a cheaper model for evolution, expensive model for evaluation
cambrian evolve "task" --model gpt-4o-mini --judge-model gpt-4o
```

The `DiffCoTReasoner` multiplies LLM calls by `n_steps`.  Keep `n_steps` low
(2–3) in cost-sensitive environments.

---

## 8. Reporting Vulnerabilities

If you discover a security vulnerability in Cambrian, please report it by
opening a GitHub issue tagged **[SECURITY]** or by contacting the maintainers
directly.  Do not post proof-of-concept exploits in public issues.

---

## Checklist for Production Deployments

- [ ] API keys in environment variables, not in code
- [ ] Spending limits configured on API provider
- [ ] `CLITool` uses `shell=False` and explicit `timeout`
- [ ] `ToolInventor` disabled or using a command allowlist
- [ ] Evolution runs inside Docker/firejail with network restrictions
- [ ] Lineage files excluded from version control
- [ ] Constitutional AI critique-revise cycles run before deployment
- [ ] Final evolved prompt reviewed by a human before production use
- [ ] Rate limiting configured (generations, population size)
