# Cambrian

> **Stop tuning prompts. Start evolving agents.**

Cambrian runs a genetic algorithm over LLM agent genomes — system prompts, temperature, strategy, few-shot examples, tools — guided by an LLM mutator. One command. No manual tweaking.

[![PyPI](https://img.shields.io/pypi/v/cambrian-ai)](https://pypi.org/project/cambrian-ai/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2242%20passing-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-1.0.4-blue)](CHANGELOG.md)
[![mypy](https://img.shields.io/badge/mypy-0%20errors-blue)](https://mypy-lang.org)

---

## Quickstart

```bash
pip install cambrian-ai

# Evolve a prompt for a coding task — 10 generations, 8 agents, no manual work
cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini --generations 10 --population 8 --output best.json

# Run the best evolved agent
cambrian run --agent best.json "Reverse 'hello world'"

# Evolve executable code (Forge mode) — no prompt engineering needed
cambrian forge "reverse(s: str) -> str" --test-case "hello:olleh"
```

**No API key? No problem:**

```bash
python examples/demo_end_to_end.py   # runs entirely with a mock backend
```

---

## How it works

```
Seed genomes → EVALUATE → SELECT → MUTATE (LLM rewrites genome) → next generation
                   ↑                                                      │
                   └──────────────────────────────────────────────────────┘
```

The LLM mutator reads the current genome and its fitness score, then writes an improved version. Tournament selection + elitism + optional crossover. No gradients. No labelled datasets. Just fitness signal.

---

## What Cambrian evolves

| Mode | What it optimises | Genome |
|------|------------------|--------|
| **Evolve** | System prompts, strategy, temperature, few-shot examples | `Genome` |
| **Forge (code)** | Python code solutions with test-case evaluation | `CodeGenome` |
| **Forge (pipeline)** | Multi-step agent pipelines (transformer → extractor → validator) | `Pipeline` |

---

## Key features

**50 bio-inspired operators** — LamarckianAdapter, EpigeneticLayer, ImmuneMemory, ApoptosisController, DreamPhase, QuorumSensor, HorizontalGeneTransfer, ZeitgeberScheduler, NeuromodulatorBank, MetamorphosisController, EcosystemInteraction, FractalEvolution, and more.

**Production-grade evaluation** — LLMJudge, CodeEvaluator (sandboxed subprocess), CompositeEvaluator, VarianceAwareEvaluator (anti-reward-hacking), BaldwinEvaluator, DiffCoTEvaluator, ConstitutionalWrapper.

**Fleet coordination** — Archipelago (island model with ring/all-to-all migration), MetaEvolutionEngine (MAML-style hyperparameter co-evolution), SelfPlayEvaluator (head-to-head tournaments), AgentNetwork (A2A delegation, broadcast, chain).

**Safety** — GoalDriftDetector, FitnessAnomalyDetector, SafeguardController, DPOSelector, subprocess sandboxing.

**Export anywhere** — `export_standalone` (self-contained script), `export_mcp` (MCP server stub), `export_api` (FastAPI app), `export_genome_json` (reload/share).

---

## Python API

```python
from cambrian.agent import Genome
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.llm_judge import LLMJudgeEvaluator
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator

backend  = OpenAICompatBackend(model="gpt-4o-mini")
engine   = EvolutionEngine(
    evaluator=LLMJudgeEvaluator(backend=backend, rubric="Clarity and accuracy"),
    mutator=LLMMutator(backend=backend),
    backend=backend,
    population_size=8,
)
best = engine.evolve(
    seed_genomes=[Genome(system_prompt="You are a helpful assistant.")],
    task="Explain quantum entanglement to a 10-year-old",
    n_generations=10,
)
print(f"Best fitness: {best.fitness:.4f}")
print(best.genome.system_prompt)
```

---

## Cambrian vs the field

| | Cambrian | DSPy | DGM | AVO | TextGrad |
|-|----------|------|-----|-----|----------|
| Full evolutionary loop | ✅ | ❌ | ✅ | ❌ | ❌ |
| LLM-guided mutation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gradient-free | ✅ | ✅ | ✅ | ✅ | ❌ |
| Code evolution | ✅ | ❌ | ✅ | ❌ | ❌ |
| Island / Archipelago | ✅ | ❌ | ❌ | ❌ | ❌ |
| Meta-evolution (auto-HP) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-agent tournament | ✅ | ❌ | ❌ | ❌ | ❌ |
| 50 bio-inspired operators | ✅ | ❌ | ❌ | ❌ | ❌ |
| Safeguards (drift + anomaly) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Any OpenAI-compatible API | ✅ | ✅ | ✅ | partial | ✅ |

---

## All CLI commands

```bash
cambrian evolve      "task"   # evolve a prompt
cambrian forge       "task"   # evolve code or a pipeline
cambrian run         "task"   # run an evolved agent
cambrian meta-evolve "task"   # co-evolve agents + hyperparameters
cambrian tournament  "task"   # round-robin competition
cambrian analyze     log.json # deep trajectory analysis
cambrian compare     a.json b.json
cambrian snapshot    --memory lineage.json --generation 5
cambrian stats       lineage.json
cambrian distill     best.json
cambrian distill-agent --agent best.json --target gemma-4-12b
cambrian dashboard   --port 8501  # Streamlit live dashboard
cambrian version
```

---

## Documentation

| | |
|-|-|
| [Tutorial](docs/TUTORIAL.md) | New-user guide: install → evolve → forge → export |
| [API Reference](docs/API_REFERENCE.md) | All 126 public symbols |
| [Architecture](docs/ARCHITECTURE.md) | Component diagram + data flows |
| [Comparison](docs/COMPARISON.md) | Deep dive: Cambrian vs DSPy, DGM, AVO, EvoAgent |
| [Deployment](docs/DEPLOYMENT.md) | Render, Docker, Kubernetes |
| [ENV_VARS](docs/ENV_VARS.md) | All environment variables |
| [VISION](VISION.md) | Where Cambrian is going |
| [CHANGELOG](CHANGELOG.md) | What changed |

---

## Development

```bash
git clone https://github.com/Franck1120/cambrian.git
cd cambrian && pip install -e ".[dev]"

pytest tests/ -q          # 1973 tests, ~10s
mypy cambrian/ --ignore-missing-imports
ruff check cambrian/
```

---

## License

MIT — see [LICENSE](LICENSE).
