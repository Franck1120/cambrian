# Cambrian — Step-by-Step Tutorial

This tutorial walks a new user through the full Cambrian workflow:
installation → first evolution run → Forge mode → analysing results → exporting an agent.

---

## Prerequisites

- Python 3.11 or 3.12
- An OpenAI-compatible API key (any provider — OpenAI, Anthropic, Groq, local Ollama)

---

## 1. Installation

```bash
# From PyPI (stable release)
pip install cambrian-ai

# For development / latest main
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"
```

Verify the installation:

```bash
cambrian version
# Cambrian 1.0.1
```

---

## 2. First Evolution Run (Evolve Mode)

Evolve mode searches for the best **system prompt** for a given task.

### 2.1 Set your API key

```bash
export OPENAI_API_KEY="sk-..."         # OpenAI
# or
export OPENAI_API_KEY="ollama"         # local Ollama (no real key needed)
export CAMBRIAN_BASE_URL="http://localhost:11434/v1"
```

### 2.2 Run a minimal evolution

```bash
cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini \
    --generations 5 \
    --population 4 \
    --output best.json \
    --memory-out lineage.json
```

You will see a live table:

```
  Gen   0  best=0.1200  mean=0.0950
  Gen   1  best=0.4800  mean=0.3100
  Gen   2  best=0.6500  mean=0.5200
  ...
```

At the end, `best.json` holds the winning genome and `lineage.json` holds the full ancestry graph.

### 2.3 Inspect the winner

```bash
cambrian distill best.json
```

Expected output:

```
┌─────────────────── Best Genome ────────────────────┐
│ System prompt : You are a Python expert who …      │
│ Strategy      : step-by-step                       │
│ Temperature   : 0.7                                │
│ Model         : gpt-4o-mini                        │
│ Fitness       : 0.6500                             │
└────────────────────────────────────────────────────┘
```

### 2.4 Run the evolved agent on a new task

```bash
cambrian run --agent best.json "Reverse 'hello world'"
```

---

## 3. Python API — First Evolution

The same run from code:

```python
from cambrian import Agent, Genome, EvolutionEngine, LLMMutator
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.code import CodeEvaluator

backend = OpenAICompatBackend(model="gpt-4o-mini")
evaluator = CodeEvaluator(
    entry_point="reverse",
    test_cases=[{"input": "hello", "expected": "olleh"}],
)
mutator = LLMMutator(backend=backend)

engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    backend=backend,
    population_size=6,
    elite_ratio=0.25,
)

seeds = [Genome(system_prompt="You are a Python expert.")]
best = engine.evolve(seed_genomes=seeds, task="Write reverse(s: str) -> str", n_generations=8)

print(f"Best fitness : {best.fitness:.4f}")
print(f"Best prompt  : {best.genome.system_prompt}")
```

### Tracking progress with a callback

```python
def on_generation(gen: int, population: list[Agent]) -> None:
    scores = [a.fitness or 0.0 for a in population]
    print(f"Gen {gen:3d}  best={max(scores):.4f}  mean={sum(scores)/len(scores):.4f}")

best = engine.evolve(
    seed_genomes=seeds,
    task="Write reverse(s: str) -> str",
    n_generations=10,
    on_generation=on_generation,
)
```

---

## 4. Forge Mode — Evolving Python Code

Forge mode evolves **executable Python code**, not just prompts.

### 4.1 CLI

```bash
cambrian forge "Write a function reverse(s: str) -> str" \
    --test-case "hello:olleh" \
    --test-case "world:dlrow" \
    --test-case "abc:cba" \
    --generations 8 \
    --population 6 \
    --output forge_best.py
```

The output `forge_best.py` is runnable Python.

### 4.2 Python API

```python
from cambrian.code_genome import CodeEvolutionEngine, CodeGenome
from cambrian.backends.openai_compat import OpenAICompatBackend

backend = OpenAICompatBackend(model="gpt-4o-mini")
engine = CodeEvolutionEngine(backend=backend, population_size=6)

best = engine.evolve(
    seed=CodeGenome(description="reverse a string"),
    task="Write a Python function reverse(s: str) -> str",
    test_cases=[
        {"input": "hello", "expected": "olleh"},
        {"input": "world", "expected": "dlrow"},
    ],
    n_generations=8,
)

print(best.genome.code)
```

---

## 5. Analysing Results

### 5.1 View stats

```bash
cambrian stats lineage.json
```

### 5.2 Snapshot a generation

```bash
# Top 5 agents at generation 5
cambrian snapshot --memory lineage.json --generation 5 --top 5

# As JSON
cambrian snapshot --memory lineage.json --generation 10 --format json > gen10.json
```

### 5.3 Deep analysis

```bash
cambrian analyze lineage.json --top 5
```

Output includes:
- Fitness trajectory (mean, best, variance per generation)
- Diversity index (genome Jaccard distances)
- Lineage tree (ancestry of the best agent)
- Top-N genomes with their prompts

### 5.4 Compare two runs

```bash
cambrian compare run_a.json run_b.json
cambrian compare run_a.json run_b.json --metric mean_fitness --format json
```

### 5.5 Live dashboard

```bash
cambrian dashboard --port 8501 --log-file lineage.json
```

Opens a Streamlit app with interactive fitness curves, diversity plots, and genome viewer.

---

## 6. Advanced — Meta-Evolution

Meta-evolution automatically tunes the hyperparameters (mutation rate, crossover rate,
temperature, tournament size) while evolving agents.

```bash
cambrian meta-evolve "Summarise any text in one sentence" \
    --generations 20 \
    --population 8 \
    --output meta_best.json
```

In Python:

```python
from cambrian.meta_evolution import MetaEvolutionEngine, HyperParams
from cambrian import LLMMutator
from cambrian.backends.openai_compat import OpenAICompatBackend

backend = OpenAICompatBackend(model="gpt-4o-mini")
engine = MetaEvolutionEngine(
    evaluator=my_evaluator,
    mutator=LLMMutator(backend=backend),
    backend=backend,
    population_size=8,
    initial_hp=HyperParams(mutation_rate=0.8, temperature=0.7),
)
best = engine.evolve(seed_genomes=seeds, task="...", n_generations=20, meta_interval=2)
```

---

## 7. Advanced — Round-Robin Tournament

Run agents head-to-head and rank them by win/loss record.

```bash
cambrian tournament "Explain quantum entanglement" \
    --population 6 \
    --output tournament_results.json
```

In Python:

```python
from cambrian.self_play import SelfPlayEvaluator, run_tournament
from cambrian import Agent, Genome

agents = [Agent(genome=Genome(system_prompt=p)) for p in my_prompts]
sp_eval = SelfPlayEvaluator(base_evaluator=my_evaluator, win_bonus=0.1)
record = run_tournament(agents, sp_eval, task="Explain quantum entanglement")

print(record.wins)   # {agent_id: win_count, ...}
```

---

## 8. Exporting an Evolved Agent

### As a standalone Python script

```bash
cambrian distill-agent --agent best.json --output agent_script.py
```

Or from Python:

```python
from cambrian.export import export_standalone
export_standalone(best_agent, "agent_script.py")
```

### As a FastAPI REST endpoint

```python
from cambrian.export import export_api
export_api(best_agent, "agent_api.py")
# cd to that directory and run: uvicorn agent_api:app
```

### As an MCP server

```python
from cambrian.export import export_mcp
export_mcp(best_agent, "agent_mcp.py")
```

---

## 9. Mock Backend (No API Key Required)

For testing or demos, use a `MagicMock` backend:

```python
import json
from unittest.mock import MagicMock
from cambrian import Agent, Genome, EvolutionEngine, LLMMutator
from cambrian.evaluator import Evaluator

def mock_backend() -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=json.dumps({
        "system_prompt": "You are an expert assistant",
        "strategy": "step-by-step",
        "temperature": 0.7,
        "model": "gpt-4o-mini",
        "tools": [],
        "few_shot_examples": [],
    }))
    return b

class KeywordEvaluator(Evaluator):
    KEYWORDS = ["expert", "step-by-step", "systematic", "analytical"]

    def evaluate(self, agent: Agent, task: str) -> float:
        hits = sum(1 for kw in self.KEYWORDS if kw in agent.genome.system_prompt.lower())
        return min(1.0, hits * 0.25)

backend = mock_backend()
engine = EvolutionEngine(
    evaluator=KeywordEvaluator(),
    mutator=LLMMutator(backend=backend),
    backend=backend,
    population_size=4,
)
seeds = [Genome(system_prompt="simple prompt")]
best = engine.evolve(seed_genomes=seeds, task="demo task", n_generations=5)
print(f"Best fitness: {best.fitness:.4f}")
```

Run the included demo without any API key:

```bash
python examples/demo_end_to_end.py
```

---

## 10. Next Steps

| Goal | Resource |
|------|----------|
| All public classes | `docs/API_REFERENCE.md` |
| Architecture deep-dive | `docs/ARCHITECTURE.md` |
| All 50 techniques | `docs/METHODOLOGY.md` |
| Changelog | `CHANGELOG.md` |
| Full CLI reference | `cambrian --help` / `cambrian <command> --help` |
| Run tests | `pytest tests/ -v` |
| Type-check | `mypy cambrian/` |
| Lint | `ruff check cambrian/` |

---

*Cambrian v1.0.1 — [GitHub](https://github.com/Franck1120/cambrian)*
