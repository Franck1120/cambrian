# Cambrian

**LLM-guided evolutionary optimisation of AI agent genomes.**

Instead of tuning prompts by hand, Cambrian evolves them. It maintains a population of agents, evaluates each on your task, then uses an LLM to intelligently mutate and recombine the best-performing genomes — turning the search into a directed hill-climb rather than a random walk.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cambrian                                │
│                                                                 │
│  ┌──────────┐   evaluate   ┌───────────┐   select   ┌────────┐ │
│  │  Genome  │ ──────────► │ Evaluator │ ──────────► │ MAP-   │ │
│  │ (prompt, │             │ (code /   │             │ Elites │ │
│  │  model,  │             │  LLM /    │             │ archive│ │
│  │  temp,   │             │  composite│             └────────┘ │
│  │  tools)  │             └───────────┘                │       │
│  └──────────┘                                          │       │
│       ▲                                                ▼       │
│       │              ┌───────────────────────────────────────┐ │
│       └─── mutate ── │  LLMMutator (mutate / crossover)      │ │
│                      │  fallback: random tweak / sentence mix │ │
│                      └───────────────────────────────────────┘ │
│                                                                 │
│  EvolutionEngine: tournament selection → elitism → N gens      │
│  EvolutionaryMemory: NetworkX lineage graph (JSON export)       │
│  ModelRouter: cheap → medium → premium tier routing            │
│  SemanticCache: SHA-256 LRU, TTL, hit-rate tracking            │
│  Compress: caveman stopword strip + procut paragraph prune      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Component | Description |
|-----------|-------------|
| `EvolutionEngine` | Full generational loop with elitism, crossover, mutation |
| `LLMMutator` | LLM-guided genome mutation and intelligent crossover |
| `MAPElites` | Quality-diversity archive (prompt-length × temperature grid) |
| `EvolutionaryMemory` | NetworkX lineage graph with JSON serialisation |
| `CodeEvaluator` | Sandboxed subprocess execution with partial-credit scoring |
| `LLMJudgeEvaluator` | LLM-as-judge on a 0-10 rubric, normalised to [0, 1] |
| `CompositeEvaluator` | Weighted mean or min-aggregate over multiple evaluators |
| `ModelRouter` | Routes tasks to cheap/medium/premium tiers by complexity |
| `SemanticCache` | SHA-256 keyed LRU with TTL and hit-rate introspection |
| `OpenAICompatBackend` | Works with OpenAI, Ollama, Groq, Together, any OAI-compat API |
| CLI (`cambrian`) | `evolve`, `dashboard`, `distill`, `version` commands + Rich UI |

---

## Quick Start

### Install

```bash
pip install cambrian-ai
# or from source:
pip install -e ".[dev]"
```

### Run the FizzBuzz example

```bash
export OPENAI_API_KEY=sk-...
python examples/evolve_fizzbuzz.py --generations 5 --population 6
```

### Evolve via CLI

```bash
cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini \
    --generations 10 \
    --population 8 \
    --output best_genome.json \
    --memory-out lineage.json
```

### Use Ollama locally

```bash
CAMBRIAN_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
    cambrian evolve "Explain Newton's second law in one sentence" \
    --model llama3.2 --generations 5
```

### Inspect results

```bash
# Pretty-print best genome
cambrian distill best_genome.json

# Show per-generation stats
cambrian dashboard lineage.json
```

---

## Python API

```python
from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator

backend = OpenAICompatBackend(model="gpt-4o-mini", api_key="sk-...")
mutator = LLMMutator(backend=backend)
evaluator = CodeEvaluator(expected_output="hello world\n")

engine = EvolutionEngine(
    evaluator=evaluator,
    mutator=mutator,
    population_size=8,
    mutation_rate=0.8,
)

seed = Genome(system_prompt="You are a Python expert. Output code only.")
best = engine.evolve(seed_genomes=[seed], task="Print hello world", n_generations=5)

print(f"Best fitness: {best.fitness:.4f}")
print(f"Best prompt: {best.genome.system_prompt}")
```

---

## Architecture

```
cambrian/
├── __init__.py              # Public exports: Agent, Genome, EvolutionEngine
├── agent.py                 # Genome dataclass + Agent (run, clone, fitness)
├── evaluator.py             # Abstract Evaluator base class
├── evolution.py             # EvolutionEngine (main loop)
├── mutator.py               # LLMMutator (mutate + crossover + fallbacks)
├── diversity.py             # MAPElites quality-diversity archive
├── memory.py                # EvolutionaryMemory (NetworkX lineage graph)
├── cache.py                 # SemanticCache (SHA-256 LRU + TTL)
├── router.py                # ModelRouter (complexity-based tier routing)
├── compress.py              # caveman_compress + procut_prune
├── cli.py                   # Click CLI (evolve/dashboard/distill/version)
├── backends/
│   ├── base.py              # Abstract LLMBackend
│   └── openai_compat.py     # OpenAI-compatible httpx backend
├── evaluators/
│   ├── code.py              # CodeEvaluator (sandbox execution)
│   ├── llm_judge.py         # LLMJudgeEvaluator (0-10 rubric)
│   └── composite.py         # CompositeEvaluator (weighted mean / min)
└── utils/
    ├── logging.py           # Rich + plain logging, log_generation_summary()
    └── sandbox.py           # run_in_sandbox(), SandboxResult, extract_python_code()

examples/
└── evolve_fizzbuzz.py       # End-to-end FizzBuzz evolution demo

tests/
├── test_agent.py            # Genome + Agent unit tests
├── test_evolution.py        # EvolutionEngine integration tests
└── test_evaluator.py        # Evaluator unit tests
```

---

## Roadmap

- [ ] Async evaluation (parallel agent runs per generation)
- [ ] Multi-objective evolution (Pareto front)
- [ ] Tool-use genomes (function-calling agents)
- [ ] Web UI dashboard (live generation chart)
- [ ] Anthropic Claude backend
- [ ] Google Gemini backend
- [ ] Distributed evaluation (Ray / Celery)
- [ ] Prompt distillation: compress evolved prompts with caveman + procut

---

## Development

```bash
git clone https://github.com/youruser/cambrian
cd cambrian
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE).
