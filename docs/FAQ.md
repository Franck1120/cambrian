# Frequently Asked Questions

---

## General

### What is Cambrian?

Cambrian is an LLM-guided evolutionary framework that automatically optimises
AI agent system prompts, strategies, and hyperparameters.  Instead of hand-
tuning prompts, you define a task and an evaluator, then let Cambrian evolve
the best agent configuration over multiple generations.

### How is this different from prompt engineering?

Manual prompt engineering is a manual, single-pass process.  Cambrian treats
prompts as evolvable genomes and runs a directed search guided by the LLM's
own understanding of what makes prompts effective — combined with actual
performance on your evaluation task.

### What LLMs are supported?

| Backend | Models |
|---------|--------|
| `OpenAICompatBackend` | GPT-4o, GPT-4o-mini, Ollama, Groq, vLLM, any OpenAI-compatible API |
| `AnthropicBackend` | Claude 3.5, Claude 4 series |
| `GeminiBackend` | Gemini 1.5, Gemini 2.0 |

Set the `CAMBRIAN_BASE_URL` environment variable to point at any
OpenAI-compatible endpoint (Ollama, Groq, etc.).

### Does it work with local models?

Yes.  Point `CAMBRIAN_BASE_URL` at your Ollama or vLLM endpoint:

```bash
export CAMBRIAN_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"   # placeholder — Ollama ignores the key
cambrian evolve "your task" --model llama3.2
```

---

## Performance & Cost

### How many API calls does one evolution run make?

Approximately `population_size × n_generations × (1 eval call + 1 mutation call)`.
For `--population 8 --generations 10` that is roughly 160 calls.

Use `--model gpt-4o-mini` for evolution and only switch to a more expensive
model for final evaluation.

### How can I reduce cost?

- Use `gpt-4o-mini` or a local Ollama model for mutation.
- Reduce `--population` and `--generations` during exploration.
- Enable `SpeculativeMutator` only when you have budget — it multiplies calls
  by `k_candidates`.
- Set spending limits on your API provider.
- Use `DiffCoTReasoner` with `n_steps=2` max in cost-sensitive settings.

### Will fitness always improve monotonically?

No — evolution is stochastic.  Fitness can plateau or temporarily decrease.
The engine uses elitism (top agents are always carried forward), so the
*best* fitness never decreases, but the *mean* can oscillate.

---

## Evaluators

### Can I write a custom evaluator?

Yes.  Subclass `Evaluator` and implement `evaluate(agent, task) -> float`:

```python
from cambrian.evaluator import Evaluator
from cambrian.agent import Agent

class MyEvaluator(Evaluator):
    def evaluate(self, agent: Agent, task: str) -> float:
        response = agent.run(task)
        # Score the response however you like
        return 1.0 if "correct answer" in response.lower() else 0.0
```

### What is the CodeEvaluator?

`CodeEvaluator` runs the agent's code output in a subprocess sandbox and
compares it to `expected_output`.  It awards partial credit for partially
correct outputs (e.g., right format, wrong value).

### What is the LLMJudgeEvaluator?

`LLMJudgeEvaluator` passes the agent's response to a second "judge" LLM with
a custom rubric.  The judge returns a score in `[0.0, 1.0]`.  This is useful
when correctness is subjective or hard to test programmatically.

---

## Features

### What is Lamarckian adaptation?

When an agent scores above a threshold, `LamarckianAdapter` captures the
(task, response) pair as a few-shot example and stores it in the genome.
Future mutations can build on these proven patterns.

### What is stigmergy?

Stigmergy is a collective memory mechanism: high-scoring agents deposit
"pheromone traces" in a shared memory.  The `LLMMutator` injects these
traces into the mutation prompt, biasing the search toward regions of the
prompt space that have worked before.

### What is the NSGA-II selection?

NSGA-II (Non-dominated Sorting Genetic Algorithm II) allows evolution across
multiple objectives simultaneously — e.g., maximising fitness while minimising
prompt length.  Agents on the Pareto front are preferred.

### What is meta-evolution?

`MetaEvolutionEngine` evolves the hyperparameters (mutation rate, crossover
rate, temperature) alongside the agent genomes.  Every `meta_interval`
generations, the engine tries perturbed configurations and keeps whichever
improves mean population fitness.

### What is the world model?

`WorldModelEvaluator` gives each agent a lightweight predictor of its own
performance.  Agents that predict their scores accurately get a blended fitness
bonus.  This selects for self-aware agents that know when they are likely to
succeed.

---

## Deployment

### How do I deploy an evolved agent?

```bash
# Self-contained script
cambrian run --agent best.json "your task"

# FastAPI REST API
from cambrian.export import export_api
export_api(best, "my_agent_api.py")
uvicorn my_agent_api:app

# MCP server
from cambrian.export import export_mcp
export_mcp(best, "mcp_server/")
```

### Is it safe to run CLITool in production?

Only if you use `shell=False`, set an explicit `timeout`, and maintain a
command allowlist.  See [`SECURITY.md`](../SECURITY.md) for the full checklist.

### Should I commit lineage files?

No.  Lineage JSON files contain full genome snapshots including all system
prompts.  Add `lineage_*.json` to `.gitignore` and treat them as internal
development artifacts.

---

## Troubleshooting

### Evolution is stuck — fitness not improving

1. Increase `mutation_rate` (try `0.9`).
2. Increase population size to explore more of the prompt space.
3. Try `SpeculativeMutator(k_candidates=5)` to evaluate more candidates per step.
4. Check your evaluator — if it returns the same score for very different
   responses, evolution has no signal to work with.

### mypy errors after importing Cambrian

Cambrian targets `mypy --strict`.  If you see errors in your own code,
check that you are using the public API only.  Internal classes (prefixed with
`_`) are not part of the public API.

### The agent's response is always the same

The base temperature in the genome may be too low.  Try `Genome(temperature=0.7)`.
The `DiffCoTReasoner` also anneals to near-zero at the final step — use
`temperature_schedule="linear"` if you want to preserve some randomness.

---

## Contributing

Contributions are welcome!  Please open an issue or PR on
[GitHub](https://github.com/Franck1120/cambrian).  Run the full test suite
before submitting:

```bash
pytest tests/ -q
mypy cambrian/ --strict --ignore-missing-imports
ruff check cambrian/
```
