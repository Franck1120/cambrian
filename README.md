# Cambrian

> **Stop tuning prompts. Start evolving agents.**

Cambrian runs a genetic algorithm over LLM agent genomes — system prompts, temperature, strategy, few-shot examples, tools — guided by an LLM mutator. One command. No manual tweaking.

[![CI](https://github.com/Franck1120/cambrian/actions/workflows/ci.yml/badge.svg)](https://github.com/Franck1120/cambrian/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2347%20passing-brightgreen)](tests/)
[![Status](https://img.shields.io/badge/status-v0.1.0--dev-orange)](#status)

> **Status:** `v0.1.0-dev` — **not yet published to PyPI**, empirical validation pending.
> The core CLI + evolutionary engine are real and tested; convergence vs. baselines
> has not yet been benchmarked on a public task. See [Status](#status) for the honest state.

---

## Quickstart

```bash
# Not on PyPI yet — install from source:
git clone https://github.com/Franck1120/cambrian.git
cd cambrian && pip install -e .

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

**30 bio-inspired operators** — LamarckianAdapter, EpigeneticLayer, ImmuneMemory, ApoptosisController, DreamPhase, QuorumSensor, HorizontalGeneTransfer, ZeitgeberScheduler, NeuromodulatorBank, MetamorphosisController, EcosystemInteraction, FractalEvolution, and more. About half are genuine algorithms (annealing, tabu, apoptosis, HGT, neuromodulation); the rest are LLM-prompting pipelines under a biological name. A handful are wired into the engine hooks by default (see [Operators](#operators-engine-integration)); the others run standalone.

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
| 30 bio-inspired operators | ✅ | ❌ | ❌ | ❌ | ❌ |
| Safeguards (drift + anomaly) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Any OpenAI-compatible API | ✅ | ✅ | ✅ | partial | ✅ |

> **Honest caveat:** this table shows *feature breadth*, not measured superiority.
> Cambrian has **not** been benchmarked against these systems on any public task.
> DSPy/GEPA are also gradient-free; [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026)
> is the closest prior art to Cambrian's LLM-guided mutation and is already published
> and benchmarked. Cambrian's honest claim today is **breadth + clean engineering**,
> not a proven performance edge. See [Status](#status).

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
| [API Reference](docs/API_REFERENCE.md) | Public API surface |
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

## Operators (engine integration)

The 30 operators split into two honest categories:

**Engine-integrated by default** (genuinely alter the evolutionary dynamics via
engine hooks — enable with `--enable tabu,apoptosis`):

| Operator | Hook | Real effect |
|----------|------|-------------|
| `tabu` | wraps the mutator + `on_generation_end` | rejects mutations that revisit recently-seen genomes; retries up to N times |
| `apoptosis` | `post_evaluation` + `on_generation_end` | re-seeds chronically stagnant / sub-floor agents from the current best |

**Standalone** (the other 28) — used directly in your own code, not auto-wired
into the engine loop. About half are real algorithms; the rest are LLM-prompting
pipelines under a biological name. The honest mapping for the LLM-wrapper ones:

| Operator | What it actually is |
|----------|---------------------|
| `glossolalia` | one high-temperature LLM call followed by a low-temperature refine |
| `dream` | offline LLM "replay" pass that proposes genome variants |
| `reflexion` | LLM self-critique of a trajectory, fed back as a score |
| `symbiosis` / `metamorphosis` | multi-step LLM prompting chains |

If you enable a standalone operator via `--enable`, its `register()` is a no-op
(it does not attach to the loop) — that is intentional and documented, not a bug.

---

## Status

**`v0.1.0-dev` — honest state as of 2026-06-15:**

- ✅ **Real:** CLI, `EvolutionEngine` (tournament + elitism + crossover + MAP-Elites),
  `LLMMutator`, NSGA-II (`pareto.py`), sandboxed `CodeEvaluator`, Gemini/OpenAI/Anthropic
  backends, 2347 passing tests (~90% line coverage).
- ✅ **API/WebUI:** now backed by the **real** `EvolutionEngine` (set `CAMBRIAN_BACKEND=gemini`
  + `GEMINI_API_KEY`). Without a key it runs the mock backend and tags runs `backend="mock"`
  so the UI shows the numbers are simulated.
- ⚠️ **Not yet done:** **no convergence benchmark** vs. random search / hill-climbing has
  been run on a real task yet (harness lives in `benchmarks/humaneval_real.py`, awaiting a
  Gemini key). Not on PyPI. 8 open `mypy` errors. ~28/30 operators are standalone.
- ❌ **Not claimed:** any measured performance edge over DSPy / GEPA / DGM.

The one thing that would turn "clean GA framework" into "defensible product" is the
benchmark — run `benchmarks/humaneval_real.py` with a real key (see its header).

---

## License

MIT — see [LICENSE](LICENSE).
