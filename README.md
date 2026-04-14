# Cambrian

**LLM-guided evolutionary optimisation of AI agent genomes.**

> Instead of hand-tuning prompts, Cambrian *evolves* them — running a directed, LLM-guided genetic search that finds high-performing system prompts automatically.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)](tests/)

---

## How it works

```
                          ┌─────────────────────────────────────────────────┐
                          │                 Cambrian Loop                   │
                          │                                                 │
  Seed genomes ──────────►│  POPULATION  (N agents, each with a Genome)    │
  (system prompts,        │      │                                          │
   temperature, model)   │      ▼                                          │
                          │  EVALUATE  ──► CodeEvaluator  (sandbox exec)   │
                          │              ► LLMJudgeEvaluator (0-10 rubric) │
                          │              ► CompositeEvaluator (weighted)    │
                          │      │                                          │
                          │      ▼                                          │
                          │  MAP-ELITES ARCHIVE  (diversity grid)           │
                          │      │                                          │
                          │      ▼                                          │
                          │  SELECT  ◄── Tournament selection (k=3)         │
                          │      │       Elitism (top N% survive unchanged) │
                          │      ▼                                          │
                          │  MUTATE / CROSSOVER  ◄── LLMMutator            │
                          │  (LLM improves prompts;  fallback: deterministic│
                          │   sentence-interleave + temperature tweak)      │
                          │      │                                          │
                          │      └──────────────► next generation ─────────┘
                          └─────────────────────────────────────────────────┘
                                  │
                                  ▼
                          Best agent + lineage graph (JSON)
```

Every generation, the LLM mutator *reads* the current prompt, understands why it scored well or poorly, and *writes* an improved version. This turns the genetic search from a random walk into a directed hill-climb powered by the LLM's knowledge.

---

## Features

| Component | What it does |
|-----------|-------------|
| `EvolutionEngine` | Full generational loop: tournament selection, elitism, crossover, mutation, MAP-Elites archiving |
| `LLMMutator` | Sends the current genome + fitness to an LLM, receives an improved genome back |
| `MAPElites` | 3×3 diversity grid keyed on (prompt-length, temperature) — prevents population collapse |
| `EvolutionaryMemory` | NetworkX directed-graph lineage: trace any agent's ancestry, export to JSON |
| `CodeEvaluator` | Runs agent-generated code in a subprocess sandbox; partial-credit line scoring |
| `LLMJudgeEvaluator` | Judge LLM scores responses 0–10 against a custom rubric |
| `CompositeEvaluator` | Combine multiple evaluators (weighted mean or min) to resist reward hacking |
| `ModelRouter` | Routes tasks to cheap / medium / premium model tiers by complexity |
| `SemanticCache` | SHA-256 LRU cache with TTL — avoid redundant API calls across generations |
| `OpenAICompatBackend` | Works with OpenAI, Ollama, Groq, LM Studio, vLLM — any OAI-compat API |
| CLI (`cambrian`) | `evolve`, `dashboard`, `distill`, `version` commands with Rich terminal UI |

---

## Getting started

### Install

```bash
# From PyPI (once published):
pip install cambrian-ai

# From source (recommended for development):
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- An OpenAI-compatible API key (or a local Ollama instance)

---

## Quickstart

### 1. Evolve via the CLI

```bash
export OPENAI_API_KEY=sk-...

cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini \
    --generations 8 \
    --population 6 \
    --output best_genome.json \
    --memory-out lineage.json
```

Watch the generation table update in real time:

```
  Gen  1  best=0.4800  mean=0.3200
  Gen  2  best=0.6400  mean=0.4750
  Gen  3  best=0.8000  mean=0.6300
  ...
```

### 2. Inspect results

```bash
# Pretty-print the winning genome
cambrian distill best_genome.json

# Per-generation fitness table from lineage
cambrian dashboard lineage.json
```

### 3. Use a local Ollama model

```bash
CAMBRIAN_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
    cambrian evolve "Explain recursion to a 10-year-old" \
    --model llama3.2 --generations 5 --population 4
```

---

## Python API

### Minimal example

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
    backend=backend,          # agents need backend to call agent.run()
    population_size=6,
    mutation_rate=0.8,
    seed=42,                  # reproducibility
)

seed = Genome(system_prompt="You are a Python expert. Output code only.")
best = engine.evolve(seed_genomes=[seed], task="Print Hello, world!", n_generations=5)

print(f"Best fitness  : {best.fitness:.4f}")
print(f"Best prompt   : {best.genome.system_prompt}")
```

### With a generation callback

```python
def log_gen(gen: int, population: list) -> None:
    scores = [a.fitness or 0 for a in population]
    print(f"Gen {gen}  best={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")

engine.evolve(seed_genomes=[seed], task="...", n_generations=10, on_generation=log_gen)
```

### CompositeEvaluator (anti-reward-hacking)

```python
from cambrian.evaluators.composite import CompositeEvaluator
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator

evaluator = CompositeEvaluator(
    evaluators=[
        CodeEvaluator(expected_output="FizzBuzz output..."),
        LLMJudgeEvaluator(judge_backend=backend, rubric_extension="Prefer clean code."),
    ],
    weights=[0.7, 0.3],   # normalised automatically
    aggregate="mean",
)
```

### Load a saved genome

```python
import json
from cambrian.agent import Genome

genome = Genome.from_dict(json.load(open("best_genome.json")))
print(genome.system_prompt)
```

---

## Examples

| Script | What it demonstrates |
|--------|---------------------|
| [`examples/evolve_fizzbuzz.py`](examples/evolve_fizzbuzz.py) | FizzBuzz with custom partial-credit scoring |
| [`examples/evolve_coding.py`](examples/evolve_coding.py) | 5-challenge composite coding benchmark |
| [`examples/evolve_prompt.py`](examples/evolve_prompt.py) | Open-ended Socratic tutor prompt optimisation |

```bash
python examples/evolve_coding.py --generations 6 --population 8
python examples/evolve_prompt.py --generations 5 --output tutor.json
```

---

## Architecture

```
cambrian/
├── __init__.py              Public exports: Agent, Genome, EvolutionEngine
├── agent.py                 Genome dataclass + Agent (run, clone, fitness, to_dict)
├── evaluator.py             Abstract Evaluator base class
├── evolution.py             EvolutionEngine — the main evolutionary loop
├── mutator.py               LLMMutator — mutate + crossover + fallbacks
├── diversity.py             MAPElites — 3x3 quality-diversity archive
├── memory.py                EvolutionaryMemory — NetworkX lineage graph
├── cache.py                 SemanticCache — SHA-256 LRU with TTL
├── router.py                ModelRouter — complexity-based tier routing
├── compress.py              caveman_compress + procut_prune
├── cli.py                   Click CLI (evolve / dashboard / distill / version)
├── backends/
│   ├── base.py              Abstract LLMBackend (typed generate signature)
│   └── openai_compat.py     OpenAI-compatible httpx backend, exponential back-off
├── evaluators/
│   ├── code.py              CodeEvaluator — sandbox execution + partial scoring
│   ├── llm_judge.py         LLMJudgeEvaluator — 0-10 rubric, JSON parse + fallback
│   └── composite.py         CompositeEvaluator — weighted mean / min-aggregate
└── utils/
    ├── logging.py           Rich / plain adaptive logging
    └── sandbox.py           run_in_sandbox + extract_python_code

examples/
├── evolve_fizzbuzz.py
├── evolve_coding.py
└── evolve_prompt.py

scripts/
└── benchmark.py             Per-generation wall-clock timing (no API calls)

tests/
├── test_agent.py            Genome + Agent (11 tests)
├── test_evaluator.py        LLMJudgeEvaluator + CompositeEvaluator (10 tests)
├── test_evolution.py        EvolutionEngine integration (13 tests)
└── test_modules.py          Cache / compress / router / diversity / memory (42 tests)
```

---

## Configuration

| Environment variable | Default | Purpose |
|---------------------|---------|---------|
| `OPENAI_API_KEY` | — | API key (also `CAMBRIAN_API_KEY`) |
| `CAMBRIAN_BASE_URL` | `https://api.openai.com/v1` | Override API endpoint |
| `CAMBRIAN_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `WARNING`, …) |

---

## Roadmap

- [ ] Async evaluation (parallel agent runs per generation)
- [ ] Multi-objective evolution (Pareto front)
- [ ] Tool-use genomes (function-calling agents)
- [ ] Web UI dashboard (live generation chart)
- [ ] Anthropic Claude backend
- [ ] Google Gemini backend
- [ ] Distributed evaluation (Ray / Celery)
- [ ] Prompt distillation: automatically compress evolved prompts

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome — new backends, evaluators, examples, and tests especially appreciated.

---

## Development

```bash
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"
pytest tests/ -v           # run test suite (76 tests, ~1s)
```

---

## License

MIT — see [LICENSE](LICENSE).
