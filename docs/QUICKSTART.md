# Cambrian Quickstart

Get from zero to your first evolved agent in under 5 minutes.

---

## 1. Install

```bash
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"
```

Requires **Python 3.11+** and an API key for any supported backend.

---

## 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI or compatible
# or
export ANTHROPIC_API_KEY="sk-ant-..."   # Anthropic Claude
# or
export GEMINI_API_KEY="..."             # Google Gemini
```

---

## 3. Evolve your first agent

```bash
cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini \
    --generations 5 \
    --population 6 \
    --output best.json
```

Cambrian runs 5 generations of 6 agents, evaluates each one, and saves the
best evolved agent to `best.json`.  You'll see live output like:

```
Gen   1  best=0.6200  mean=0.4800  pop=6
Gen   2  best=0.7400  mean=0.5900  pop=6
Gen   3  best=0.8100  mean=0.6700  pop=6
...
```

---

## 4. Run the evolved agent

```bash
cambrian run --agent best.json "Reverse the string 'hello world'"
```

Or get JSON output for programmatic use:

```bash
cambrian run --agent best.json --format json "Reverse 'cambrian'"
```

---

## 5. Inspect the evolved genome

```bash
cambrian distill best.json
```

This prints the system prompt, strategy, temperature, and any few-shot
examples the agent acquired via Lamarckian learning.

---

## 6. Python API — minimal example

```python
from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator

backend = OpenAICompatBackend(model="gpt-4o-mini", api_key="sk-...")
mutator = LLMMutator(backend=backend)
evaluator = CodeEvaluator(expected_output="Hello, world!")

engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    backend=backend,
    population_size=6,
    mutation_rate=0.8,
)

seed = Genome(system_prompt="You are a Python expert. Output code only.")
best = engine.evolve(seed_genomes=[seed], task="Print Hello, world!", n_generations=5)
print(f"Best fitness: {best.fitness:.4f}")
print(best.genome.system_prompt)
```

---

## 7. What to try next

| Goal | How |
|------|-----|
| Multi-objective evolution | See `examples/evolve_researcher.py` |
| LLM judge instead of code eval | `LLMJudgeEvaluator` with a custom rubric |
| Faster search via parallel mutations | `SpeculativeMutator(k_candidates=5)` |
| Run multiple isolated populations | `Archipelago(n_islands=4, topology="ring")` |
| Export as a FastAPI app | `export_api(best, "my_agent_api.py")` |
| Export as an MCP server | `export_mcp(best, "mcp_server/")` |
| Evolve Python code (Forge) | See `examples/evolve_forge_code.py` |
| Auto-regulate diversity | See `examples/evolve_with_quorum.py` |
| Memory consolidation | See `examples/evolve_with_dream.py` |
| Ensemble synthesis | See `examples/evolve_with_moa.py` |
| Improve with self-critique | `ReflexionAgent(agent, n_rounds=2)` |

---

## 8. Forge Mode — evolve Python code

Use `cambrian forge` when you want to evolve **executable code** rather than
prompt text.

```bash
# Evolve a Python script that sums two integers from stdin
cambrian forge "Read two integers from stdin and print their sum" \
    --mode code \
    --model gpt-4o-mini \
    --test-case "3 4|7" \
    --test-case "10 20|30" \
    --test-case "-1 1|0" \
    --generations 8 \
    --output best_code.json
```

Python API equivalent:

```python
from cambrian.code_genome import CodeGenome, CodeEvaluator, CodeEvolutionEngine, TestCase
from cambrian.backends.openai_compat import OpenAICompatBackend

backend = OpenAICompatBackend(model="gpt-4o-mini")
test_cases = [
    TestCase("3 4", "7"),
    TestCase("10 20", "30"),
    TestCase("-1 1", "0"),
]
engine = CodeEvolutionEngine(
    backend=backend,
    evaluator=CodeEvaluator(test_cases),
    population_size=6,
)
best = engine.evolve(
    CodeGenome(description="sum two integers from stdin"),
    task="sum two integers",
    n_generations=8,
)
print(best.code)
```

---

## 9. Useful CLI commands

```bash
# View population at a specific generation
cambrian snapshot --memory lineage.json --generation 3

# Compare two evolution runs
cambrian compare run_a.json run_b.json

# Show statistics from a lineage file
cambrian stats lineage.json

# Deep analysis with top-N agents
cambrian analyze lineage.json --top 5

# Live dashboard (requires streamlit)
cambrian dashboard --log-file run.json --port 8501
```

---

## Troubleshooting

**`Error: set OPENAI_API_KEY or pass --api-key`**
→ Export your API key before running.

**Slow evolution**
→ Reduce `--population` or `--generations`.  Use a faster model like `gpt-4o-mini`.

**Agents not improving**
→ Try increasing `--mutation-rate` to `0.9` or use `SpeculativeMutator` to evaluate more candidates per generation.

**Out of budget**
→ Set spending limits on your API provider. Use `--generations 5 --population 4` for exploration runs.

---

## Next steps

- Read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for a complete component reference.
- Read [`docs/METHODOLOGY.md`](METHODOLOGY.md) for academic context behind each technique.
- Read [`SECURITY.md`](../SECURITY.md) before deploying evolved agents to production.
