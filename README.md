# Cambrian

**LLM-guided evolutionary optimisation of AI agent genomes.**

> Instead of hand-tuning prompts, Cambrian *evolves* them — running a directed, LLM-guided genetic search that finds high-performing system prompts automatically.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1494%20passing-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org)

---

## How it works

```
  Seed genomes ──────►  POPULATION (N agents, each with a Genome)
  (prompts, temp,              │
   model, tools)               ▼
                          EVALUATE  ── CodeEvaluator / LLMJudge / Composite /
                                       Baldwin / DiffCoT / Constitutional
                               │
                               ▼
                          BIO-PRESSURES
                          ├── Epigenetics   (context-dependent expression)
                          ├── Stigmergy     (pheromone traces → mutation bias)
                          ├── Immune System (suppress barren genome regions)
                          └── Lamarck       (capture successful examples)
                               │
                               ▼
                          SELECT  ─── Tournament (k=3) + Elitism
                          or NSGA-II (multi-objective Pareto)
                          or MCTS   (UCB1-guided tree search)
                               │
                               ▼
                          MUTATE / CROSSOVER ─── LLMMutator
                          ├── Speculative execution (K parallel candidates)
                          ├── Reward shaping (clip / normalise / potential)
                          └── Tool invention (agents invent new CLI tools)
                               │
                               └──────────────► next generation
```

---

## Features

### Core Evolution
| Component | What it does |
|-----------|-------------|
| `EvolutionEngine` | Full generational loop: tournament selection, elitism, crossover, mutation |
| `LLMMutator` | LLM reads the genome + fitness and writes an improved version |
| `EvolutionaryMemory` | NetworkX lineage graph — trace ancestry, export/import JSON |
| `SpeculativeMutator` | Generate K mutations in parallel (asyncio), keep the best |
| `Archipelago` | Multi-island evolution with ring / all-to-all / random migration |

### Evaluators
| Component | What it does |
|-----------|-------------|
| `CodeEvaluator` | Runs agent code in a subprocess sandbox; partial-credit scoring |
| `LLMJudgeEvaluator` | Judge LLM scores responses against a custom rubric |
| `CompositeEvaluator` | Weighted average of multiple evaluators |
| `VarianceAwareEvaluator` | Penalise inconsistent agents (anti-reward-hacking) |
| `BaldwinEvaluator` | Multi-trial evaluation with in-context learning bonus |
| `DiffCoTEvaluator` | Iterative chain-of-thought denoising before scoring |

### Bio-Inspired Modules
| Component | What it does |
|-----------|-------------|
| `LamarckianAdapter` | Capture successful examples into genome's few-shot examples |
| `EpigeneticLayer` | Context-dependent system prompt annotations at runtime |
| `ImmuneMemory` | SHA-256 fingerprinting; suppress re-evaluation of barren regions |
| `MCTSSelector` | UCB1-guided tree search over the mutation tree |
| `CoEvolutionEngine` | Adversarial generator/adversary arms race |
| `CurriculumScheduler` | Task difficulty progression with thresholds |
| `ConstitutionalWrapper` | Critique-revise safety cycles before scoring |

### Reasoning Modules
| Component | What it does |
|-----------|-------------|
| `DiffCoTReasoner` | Diffusion-inspired iterative chain-of-thought denoising |
| `CausalGraph` | Explicit cause-effect relationship representation in strategies |
| `CausalStrategyExtractor` | LLM-based extraction of IF-THEN causal relations |
| `CausalMutator` | Evolves causal graphs alongside genomes |

### Self-Play & Competition
| Component | What it does |
|-----------|-------------|
| `SelfPlayEvaluator` | Head-to-head agent competition; win/loss/draw bonuses applied to fitness |
| `SelfPlayResult` | Single match result: scores, winner, margin, draw detection |
| `TournamentRecord` | Win/loss/draw ledger across a round-robin tournament |
| `run_tournament` | Round-robin: all agent pairs compete; returns ranked `TournamentRecord` |

### Meta-Evolution
| Component | What it does |
|-----------|-------------|
| `MetaEvolutionEngine` | MAML-inspired outer loop: evolves hyperparameters alongside genomes |
| `HyperParams` | Mutable bundle of `mutation_rate`, `crossover_rate`, `temperature`, `tournament_k`, `elite_ratio` |

### World Model
| Component | What it does |
|-----------|-------------|
| `WorldModelEvaluator` | Wraps any evaluator; rewards agents for accurate self-prediction |
| `WorldModel` | Per-agent experience buffer with weighted nearest-neighbour prediction |
| `WorldModelPrediction` | Predicted score + confidence + n_similar |
| `world_model_fitness` | Blend raw score with prediction accuracy into a single metric |

### Structured Logging
| Component | What it does |
|-----------|-------------|
| `JSONLogger` | NDJSON per-generation logging; context-manager-safe; `log_generation`, `log_run_summary` |
| `load_json_log` | Read NDJSON evolution log files; skips malformed lines |

### Multi-Objective
| Component | What it does |
|-----------|-------------|
| `ParetoFront` | Incremental non-dominated set |
| `fast_non_dominated_sort` | NSGA-II O(M·N²) dominance ranking |
| `nsga2_select` | Full NSGA-II selection with crowding distance |
| `ObjectiveVector` | Per-agent multi-objective scores (fitness, brevity, diversity) |

### Reward Shaping
| Component | What it does |
|-----------|-------------|
| `ClipShaper` | Clamp fitness to [min, max] |
| `NormalisationShaper` | Online z-score or min-max normalisation |
| `PotentialShaper` | Potential-based shaping (Ng 1999) |
| `RankShaper` | Convert scores to fractional rank |
| `CuriosityShaper` | Intrinsic motivation bonus for novel genomes |

### Agent-to-Agent (A2A)
| Component | What it does |
|-----------|-------------|
| `AgentNetwork` | Route, delegate, broadcast, chain tasks across agent populations |
| `AgentCard` | Capability descriptor for domain-based routing |
| `A2AMessage` | Structured request/response envelope |

### Tools
| Component | What it does |
|-----------|-------------|
| `CLITool` / `CLIToolkit` | Wrap any shell command as an LLM-callable tool |
| `ToolInventor` | LLM invents new tool specs during evolution |
| `ToolPopulationRegistry` | Shared registry of population-invented tools |
| `ToolSpec` | Typed tool specification stored in `Genome.tool_specs` |

### Export & Deployment
| Component | What it does |
|-----------|-------------|
| `export_genome_json` | Save/load evolved genome |
| `export_standalone` | Self-contained Python script |
| `export_mcp` | MCP server stub (manifest + handler) |
| `export_api` | FastAPI REST application |

### Analytics
| Component | What it does |
|-----------|-------------|
| `ParetoFront` | Non-dominated Pareto archive |
| `DiversityTracker` | Per-generation entropy, temperature and prompt std |
| `FitnessLandscape` | 2D fitness grid (temperature × token-length) |

### Tier 3 — Advanced Bio-Inspired Techniques
| Component | What it does |
|-----------|-------------|
| `SymbioticFuser` | LLM-guided endosymbiosis: merges genomes of compatible (high-fitness + high-distance) agents |
| `HormesisAdapter` | Graduated stress response: mild/moderate/severe stimulation based on fitness gap |
| `ApoptosisController` | Programmed removal of chronically poor agents; optional clone replacement from best survivor |
| `CatalysisEngine` | Catalyst agent injects strategy context into peer mutation prompts |
| `LLMCascade` | Intelligent tiered routing across model sizes; escalates when confidence is below threshold |
| `AgentEnsemble` | Weighted majority vote across agent population; `BoostingEnsemble` adds AdaBoost-style weight updates |
| `GlossaloliaReasoner` | Two-phase latent monologue → structured synthesis (high-temp unconstrained → low-temp harvest) |
| `BestOfN` / `BeamSearch` | Inference-time compute scaling: best-of-N sampling and tree-based beam search |

### Tier 4 — Cutting-Edge Bio-Inspired Techniques
| Component | What it does |
|-----------|-------------|
| `TransferAdapter` | Adapts a source genome to a new task domain at light/medium/heavy intensity |
| `TabuList` / `TabuMutator` | Bi-gram fingerprint prevents revisiting recent genome regions (tabu search) |
| `AnnealingSchedule` / `AnnealingSelector` | Metropolis acceptance with linear/exponential/cosine cooling |
| `RedTeamSession` | Adversarial attack generation + `RobustnessEvaluator` combined score |
| `ZeitgeberScheduler` | Sinusoidal circadian oscillator modulates mutation rate and acceptance threshold |
| `HGTransfer` / `HGTPool` | Horizontal Gene Transfer: sentence-level genome fragments shared across domain pool |
| `TransgenerationalRegistry` | Heritable `EpigeneMark`s that decay per generation and inject into offspring genomes |
| `ImmuneCortex` | B-cell (fast exact recall) + T-cell (adaptive seed) memory for high-fitness genomes |
| `NeuromodulatorBank` | Four biologically-inspired modulators (dopamine, serotonin, acetylcholine, noradrenaline) dynamically adjust mutation rate and selection pressure |

### Tier 5 — Lifecycle, Ecology, and Fractal Evolution
| Component | What it does |
|-----------|-------------|
| `MetamorphosisController` | Holometabolous lifecycle: LARVA (high exploration) -> CHRYSALIS (LLM reorganisation) -> IMAGO (exploit) |
| `MetamorphicPopulation` | Orchestrates per-agent phase advancement and chrysalis reorganisation across a generation |
| `EcosystemInteraction` | 4-role ecological dynamics: HERBIVORE (diversity bonus), PREDATOR (hunt weak), DECOMPOSER (recycle failed), PARASITE (drain strong) |
| `EcosystemEvaluator` | Blends base evaluator score with ecological fitness delta (configurable weight) |
| `FractalEvolution` | Recursive multi-scale evolution: macro populations seed meso, meso seed micro; results bubble back up |
| `FractalMutator` | Scale-aware mutation: MACRO stays broad, MICRO highly localised |
| `FractalPopulation` | Self-contained population at one fractal scale with its own evaluator and mutator |

### Backends
| Backend | Class | SDK |
|---------|-------|-----|
| OpenAI / Ollama / Groq / vLLM | `OpenAICompatBackend` | httpx |
| Anthropic Claude | `AnthropicBackend` | `anthropic` |
| Google Gemini | `GeminiBackend` | `google-genai` |

---

## Getting started

### Install

```bash
# From source (recommended):
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"
```

### Requirements
- Python 3.11+
- An OpenAI-compatible API key (or a local Ollama instance)

---

## CLI Commands

```bash
# ── Evolve mode ──────────────────────────────────────────────────────────────

# Evolve — run evolutionary search (prompt optimisation)
cambrian evolve "Write a Python function that reverses a string" \
    --model gpt-4o-mini --generations 10 --population 8 \
    --output best.json --memory-out lineage.json

# Run — load an evolved agent and run it on a task
cambrian run --agent best.json "What is the Riemann hypothesis?"
cambrian run --agent best.json --format json "Explain entropy."

# ── Forge mode ───────────────────────────────────────────────────────────────

# Forge — evolve executable Python code (code mode)
cambrian forge "Write reverse(s: str) -> str" \
    --test-case "hello:olleh" --test-case "abc:cba" \
    --generations 8 --population 6 --output forge_best.py

# Forge — evolve a multi-step agent pipeline (pipeline mode)
cambrian forge "Summarise text concisely" \
    --mode pipeline --generations 5 --output forge_pipeline.json

# ── Analysis ─────────────────────────────────────────────────────────────────

# Snapshot — show population state at a specific generation
cambrian snapshot --memory lineage.json --generation 5
cambrian snapshot --memory lineage.json --generation 10 --format json

# Stats — text summary of a lineage file
cambrian stats lineage.json

# Analyze — deep trajectory + diversity + lineage analysis
cambrian analyze lineage.json --top 5

# Dashboard — Streamlit live evolution dashboard (2 tabs: Evolve + Forge)
cambrian dashboard --port 8501 --log-file run.json

# Distill — pretty-print a saved genome
cambrian distill best.json

# Compare — compare two NDJSON evolution run logs
cambrian compare run_a.json run_b.json
cambrian compare run_a.json run_b.json --metric mean_fitness --format json

# Distill-agent — compress genome for a smaller model
cambrian distill-agent --agent best.json --target gemma-4-12b --max-tokens 120

# Version
cambrian version
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
    backend=backend,
    population_size=6,
    mutation_rate=0.8,
    seed=42,
)

seed = Genome(system_prompt="You are a Python expert. Output code only.")
best = engine.evolve(seed_genomes=[seed], task="Print Hello, world!", n_generations=5)
print(f"Best fitness: {best.fitness:.4f}")
```

### Multi-objective evolution (NSGA-II)

```python
from cambrian.pareto import ObjectiveVector, brevity_objective, fitness_objective, nsga2_select

vectors = [
    ObjectiveVector(a.id, scores={
        "perf": fitness_objective(a),
        "brevity": brevity_objective(a),
    })
    for a in population
]
selected = nsga2_select(population, vectors, target_size=50)
```

### Agent-to-Agent delegation

```python
from cambrian.a2a import AgentCard, AgentNetwork

network = AgentNetwork()
network.register(code_agent, AgentCard(domains=["python", "code"], confidence=0.9))
network.register(math_agent, AgentCard(domains=["math", "logic"], confidence=0.85))

result = network.delegate("Implement binary search in Python.")
print(result.result)
```

### Reward shaping

```python
from cambrian.reward_shaping import build_shaped_evaluator

shaped = build_shaped_evaluator(my_evaluator, "clip+normalise+curiosity")
```

### DiffCoT iterative reasoning

```python
from cambrian.diffcot import DiffCoTConfig, make_diffcot_evaluator

config = DiffCoTConfig(n_steps=3, noise_level=0.2, temperature_schedule="cosine")
diffcot_eval = make_diffcot_evaluator(base_evaluator, backend, n_steps=3)
```

### Causal reasoning

```python
from cambrian.causal import CausalGraph, inject_causal_context

graph = CausalGraph.from_text(
    "IF the problem is complex THEN use step-by-step reasoning. "
    "IF output is code THEN include docstrings."
)
enriched_genome = inject_causal_context(agent.genome, graph)
```

### Tool invention

```python
from cambrian.tool_creation import ToolInventor, ToolPopulationRegistry

inventor = ToolInventor(backend, max_tools_per_agent=3)
registry = ToolPopulationRegistry()

result = inventor.invent_tool(agent, task="count words in text")
if result and result.success:
    registry.register(result.tool_spec)
```

### Export evolved agent

```python
from cambrian.export import export_genome_json, export_standalone, export_mcp

export_genome_json(best, "best.json")       # reload with cambrian run
export_standalone(best, "my_agent.py")      # self-contained script
export_mcp(best, "mcp_server/")            # MCP server stub
```

### Island model (Archipelago)

```python
from cambrian.archipelago import Archipelago

arch = Archipelago(
    engine_factory=lambda: EvolutionEngine(...),
    n_islands=4,
    island_size=20,
    migration_interval=5,
    topology="ring",
)
best = arch.evolve(seed_genomes=seeds, task="...", n_generations=40)
```

---

## Examples

| Script | What it demonstrates |
|--------|---------------------|
| [`examples/evolve_fizzbuzz.py`](examples/evolve_fizzbuzz.py) | FizzBuzz with custom partial-credit scoring |
| [`examples/evolve_coding.py`](examples/evolve_coding.py) | 5-challenge composite coding benchmark |
| [`examples/evolve_prompt.py`](examples/evolve_prompt.py) | Open-ended Socratic tutor prompt optimisation |
| [`examples/evolve_researcher.py`](examples/evolve_researcher.py) | Research agent: LLM judge + Lamarck + Pareto |

```bash
python examples/evolve_researcher.py \
    --topic "quantum computing applications in cryptography" \
    --generations 15 --population 12
```

---

## Architecture

```
cambrian/
├── agent.py             Genome (+ ToolSpec), Agent
├── evolution.py         EvolutionEngine (main loop)
├── mutator.py           LLMMutator (mutation + stigmergy)
├── evaluator.py         Evaluator ABC
├── memory.py            EvolutionaryMemory + StigmergyTrace
├── compress.py          caveman_compress, procut_prune
├── lamarck.py           LamarckianAdapter
├── epigenetics.py       EpigeneticLayer, EpigenomicContext
├── immune.py            ImmuneMemory, fingerprint()
├── mcts.py              MCTSNode, MCTSSelector
├── coevolution.py       CoEvolutionEngine
├── curriculum.py        CurriculumScheduler, CurriculumStage
├── constitutional.py    ConstitutionalWrapper
├── stats.py             ParetoFront, DiversityTracker, FitnessLandscape
├── a2a.py               AgentNetwork, AgentCard, A2AMessage
├── cli_tools.py         CLITool, CLIToolkit
├── tool_creation.py     ToolInventor, ToolPopulationRegistry
├── pareto.py            NSGA-II, ObjectiveVector, ParetoFront
├── archipelago.py       Island, Archipelago
├── speculative.py       speculate(), SpeculativeMutator
├── reward_shaping.py    Clip/Normalise/Potential/Rank/Curiosity shapers
├── diffcot.py           DiffCoTReasoner, DiffCoTEvaluator
├── causal.py            CausalGraph, CausalStrategyExtractor
├── tool_creation.py     ToolInventor, ToolPopulationRegistry
├── self_play.py         SelfPlayEvaluator, run_tournament
├── meta_evolution.py    MetaEvolutionEngine, HyperParams
├── world_model.py       WorldModelEvaluator, WorldModel
├── export.py            export_genome_json/standalone/mcp/api
├── dashboard.py         Streamlit live dashboard
├── cli.py               Click CLI entry point
├── backends/
│   ├── base.py          LLMBackend ABC
│   ├── openai_compat.py OpenAICompatBackend
│   ├── anthropic.py     AnthropicBackend
│   └── gemini.py        GeminiBackend
├── evaluators/
│   ├── code.py          CodeEvaluator
│   ├── llm_judge.py     LLMJudgeEvaluator
│   ├── composite.py     CompositeEvaluator
│   ├── variance_aware.py VarianceAwareEvaluator
│   └── baldwin.py       BaldwinEvaluator
└── utils/
    └── logging.py       get_logger
```

---

## Configuration

| Environment variable | Purpose |
|---------------------|---------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible backends |
| `CAMBRIAN_BASE_URL` | Override API base URL (Ollama, Groq, etc.) |
| `ANTHROPIC_API_KEY` | Anthropic Claude backend |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini backend |
| `CAMBRIAN_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, `WARNING`) |

---

## Security

See [SECURITY.md](SECURITY.md) for:
- Subprocess sandboxing guidelines
- API key management
- Prompt injection mitigations
- `ToolInventor` safe deployment
- Production deployment checklist

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full component diagram and reference |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Academic references for all 17 techniques |
| [`SECURITY.md`](SECURITY.md) | Security model and sandboxing guidelines |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Development

```bash
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
pip install -e ".[dev]"

# Run tests
pytest tests/ -q                         # 1194 tests, ~8s

# Type check
mypy cambrian/ --strict --ignore-missing-imports

# Lint
ruff check cambrian/
```

---

## License

MIT — see [LICENSE](LICENSE).
