# Cambrian — Product Requirements Document

> Version: 0.9.0 · Status: Active · Updated: 2026-04-15

---

## 1. Vision

Cambrian is an **autonomous evolutionary AI lab** — a framework that finds,
improves, and deploys high-performing AI agents without human prompt engineering.
Users describe a problem; Cambrian builds, scores, breeds, and refines agents
until one is good enough to ship.

---

## 2. Two Modes

### 2.1 Evolve Mode (prompt optimisation)

**Goal**: Optimise a natural-language system prompt for a given task.

```
cambrian evolve "Write a Python function that reverses a string"
```

The engine maintains a population of `Genome` objects (system prompt + strategy
+ temperature + model). Each generation:
1. Agents are scored by an evaluator (code sandbox / LLM judge / composite).
2. Tournament selection picks parents.
3. LLMMutator rewrites genomes.
4. Elites survive unchanged.

**Output**: A `best.json` genome + lineage graph.

### 2.2 Forge Mode (code / pipeline synthesis)

**Goal**: Synthesise and evolve executable Python code or multi-step agent
pipelines that solve a task.

```
cambrian forge "Parse a CSV, compute the mean of column 'price', return it"
```

Forge operates on `CodeGenome` (executable Python) or `Pipeline` (chain of
specialised agents). The mutation operator asks the LLM to rewrite or restructure
the code. The evaluator executes the code in a sandbox and scores by:
- Correctness (test-case matching)
- Efficiency (runtime, LOC)
- Robustness (edge-case handling)

**Output**: A `forge_best.py` (standalone Python file) or `forge_pipeline.json`.

---

## 3. User Interface Specification

### 3.1 CLI

| Command | Description |
|---------|-------------|
| `cambrian evolve TASK` | Run Evolve mode (prompt optimisation) |
| `cambrian forge TASK` | Run Forge mode (code/pipeline synthesis) |
| `cambrian run --agent FILE TASK` | Load evolved genome and run on a task |
| `cambrian analyze MEMORY` | Deep trajectory + lineage analysis |
| `cambrian snapshot --memory FILE -g N` | Show population at generation N |
| `cambrian compare RUN1 RUN2` | Compare two NDJSON evolution logs |
| `cambrian dashboard` | Launch Streamlit live dashboard |
| `cambrian distill GENOME` | Pretty-print a saved genome |
| `cambrian version` | Print version |

### 3.2 Dashboard (Streamlit)

**Tab 1 — Evolve**: Real-time fitness chart, agent population table, lineage
graph preview, diversity metrics.

**Tab 2 — Forge**: Pipeline step viewer, code diff between generations,
test-case pass/fail grid, execution time chart.

**Shared widgets**:
- Generation slider
- Population table (sortable by fitness, temperature, strategy)
- Export button (JSON / standalone Python)
- Live logs panel (NDJSON stream)

### 3.3 Python API

Minimal example (Evolve):
```python
from cambrian import EvolutionEngine, Genome, LLMMutator
from cambrian.backends.openai_compat import OpenAICompatBackend
from cambrian.evaluators.code import CodeEvaluator

backend = OpenAICompatBackend(model="gpt-4o-mini")
engine = EvolutionEngine(
    evaluator=CodeEvaluator(expected_output="hello"),
    mutator=LLMMutator(backend=backend),
    backend=backend,
)
best = engine.evolve([Genome()], task="Print hello", n_generations=10)
```

Minimal example (Forge):
```python
from cambrian.code_genome import CodeEvolutionEngine, CodeGenome

engine = CodeEvolutionEngine(backend=backend, population_size=6)
best = engine.evolve(
    seed=CodeGenome(description="reverse a string"),
    task="Write a Python function reverse(s: str) -> str",
    n_generations=8,
)
```

---

## 4. The 57 Techniques

### 4.1 Core Evolution (8)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 1 | LLM-guided mutation | `mutator.py` | LLM reads genome + fitness → writes improved version |
| 2 | Tournament selection | `evolution.py` | Sample k agents, pick highest fitness |
| 3 | Elitism | `evolution.py` | Top-N agents survive unchanged each generation |
| 4 | Crossover | `mutator.py` | LLM combines two parent genomes; fallback: sentence interleave |
| 5 | MAP-Elites archive | `diversity.py` | Diversity archive by temperature × token-length bins |
| 6 | Generational loop | `evolution.py` | Evaluate → Select → Mutate → repeat for N generations |
| 7 | Population initialisation | `evolution.py` | Seed + temperature jitter to fill population |
| 8 | Fitness-proportionate bias | `evolution.py` | Higher-fitness parent used as crossover base |

### 4.2 Bio-Inspired Pressures (9)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 9 | Lamarckian evolution | `lamarck.py` | Successful (task, response, score) tuples added to genome |
| 10 | Stigmergy | `memory.py` | High-scoring pheromone traces bias future mutations |
| 11 | Epigenetics | `epigenetics.py` | Context-dependent runtime prompt annotations |
| 12 | Immune memory | `immune.py` | SHA-256 fingerprinting; suppress barren genome regions |
| 13 | Co-evolution | `coevolution.py` | Generator + adversary populations evolve together |
| 14 | Curriculum learning | `curriculum.py` | Task difficulty progression with fitness thresholds |
| 15 | Quorum sensing | `quorum.py` | Shannon entropy → auto-adjust mutation_rate |
| 16 | Dream phase | `dream.py` | GraphRAG recombination of experiences from lineage |
| 17 | Quantum tunneling | `moa.py` | Occasional random large jumps in genome space |

### 4.3 Advanced Selection (5)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 18 | MCTS selection | `mcts.py` | UCB1-guided tree search over mutation tree |
| 19 | NSGA-II | `pareto.py` | O(M·N²) multi-objective dominance sorting |
| 20 | Crowding distance | `pareto.py` | Diversity-preserving tiebreaker in Pareto front |
| 21 | Pareto front tracking | `pareto.py` | Incremental non-dominated archive |
| 22 | Meta-evolution | `meta_evolution.py` | MAML-inspired hyperparameter self-tuning |

### 4.4 Evaluators (9)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 23 | Code evaluation | `evaluators/code.py` | Subprocess sandbox with API-key isolation |
| 24 | LLM-as-judge | `evaluators/llm_judge.py` | LLM grades response against rubric |
| 25 | Composite evaluator | `evaluators/composite.py` | Weighted average of sub-evaluators |
| 26 | Variance-aware | `evaluators/variance_aware.py` | mean − penalty×variance (anti-reward-hacking) |
| 27 | Baldwin evaluator | `evaluators/baldwin.py` | Multi-trial + in-context learning bonus |
| 28 | DiffCoT evaluator | `diffcot.py` | Denoising reasoning before base evaluation |
| 29 | World model evaluator | `world_model.py` | Reward accurate self-prediction |
| 30 | Self-play evaluator | `self_play.py` | Head-to-head competition; win/loss fitness delta |
| 31 | Constitutional evaluator | `constitutional.py` | Critique-revise cycles before scoring |

### 4.5 Reasoning Modules (7)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 32 | DiffCoT reasoning | `diffcot.py` | Cosine-annealed iterative chain-of-thought |
| 33 | Causal graph extraction | `causal.py` | IF/THEN, arrow notation → CausalGraph |
| 34 | Causal mutation | `causal.py` | Evolve causal graphs alongside genomes |
| 35 | Reflexion | `reflexion.py` | Self-critique loop: generate → reflect → revise |
| 36 | Mixture of Agents | `moa.py` | Ensemble: N agents answer, aggregator combines |
| 37 | Constitutional AI | `constitutional.py` | Principle-based critique-revise wrapper |
| 38 | Tool invention | `tool_creation.py` | LLM invents CLI tools; registry deduplicates |

### 4.6 Code Evolution (5, Forge mode)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 39 | CodeGenome | `code_genome.py` | Executable Python code as evolvable genome |
| 40 | Code mutation | `code_genome.py` | LLM rewrites code to improve correctness/efficiency |
| 41 | Sandbox execution | `utils/sandbox.py` | API-key-safe subprocess with timeout |
| 42 | Test-case scoring | `code_genome.py` | Partial credit based on test-case pass rate |
| 43 | Incremental improvement | `code_genome.py` | Version counter; mutation must beat or match parent |

### 4.7 Pipeline Evolution (5, Forge mode)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 44 | Pipeline step structure | `pipeline.py` | Named steps with system prompt + role |
| 45 | Step mutation | `pipeline.py` | LLM adds, removes, or reorders pipeline steps |
| 46 | Sequential evaluation | `pipeline.py` | Input flows through all steps; final output scored |
| 47 | Pipeline crossover | `pipeline.py` | Combine step lists from two parent pipelines |
| 48 | Forge CLI | `cli.py` | `cambrian forge TASK` entry point |

### 4.8 Architecture & Infrastructure (9)

| # | Technique | Module | Description |
|---|-----------|--------|-------------|
| 49 | Archipelago (ring) | `archipelago.py` | Islands exchange migrants in ring topology |
| 50 | Archipelago (all-to-all) | `archipelago.py` | Every island sends migrants to every other |
| 51 | Archipelago (random) | `archipelago.py` | Random island pairing for migration |
| 52 | Speculative execution | `speculative.py` | K mutations in parallel (asyncio); keep best |
| 53 | A2A routing | `a2a.py` | Domain-based agent routing + broadcasting |
| 54 | Reward shaping | `reward_shaping.py` | Clip, normalise, potential-based, rank, curiosity |
| 55 | Knowledge distillation | `compress.py` + CLI | Compress evolved prompt for smaller models |
| 56 | Structured NDJSON logging | `utils/logging.py` | Per-generation JSON log for analysis + compare |
| 57 | Structured population stats | `stats.py` | ParetoAnalyzer, DiversityTracker, FitnessLandscape |

---

## 5. Data Model

### 5.1 Genome (Evolve mode)
```
Genome
 ├── system_prompt: str      # evolvable
 ├── strategy: str           # evolvable
 ├── temperature: float      # evolvable [0.0, 2.0]
 ├── model: str              # evolvable
 ├── few_shot_examples: list # Lamarckian memory
 ├── tool_specs: list        # invented tools
 └── genome_id: str          # immutable UUID prefix
```

### 5.2 CodeGenome (Forge mode)
```
CodeGenome
 ├── code: str               # evolvable Python source
 ├── entry_point: str        # function name to call
 ├── description: str        # intent (used in mutation prompt)
 ├── language: str           # "python" (future: "js", "bash")
 ├── test_cases: list[dict]  # {input, expected_output}
 ├── version: int            # monotonically increasing
 └── genome_id: str          # immutable UUID prefix
```

### 5.3 Pipeline (Forge mode)
```
Pipeline
 ├── name: str
 ├── steps: list[PipelineStep]
 │    └── PipelineStep
 │         ├── name: str
 │         ├── system_prompt: str   # evolvable
 │         ├── role: str            # "transformer"|"extractor"|"validator"
 │         └── temperature: float  # evolvable
 └── version: int
```

---

## 6. Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Rounds 1–4: Core engine + bio-pressures | 2026-Q1 | ✅ Done |
| Rounds 5–7: Reasoning + tools + CLI | 2026-Q1 | ✅ Done |
| Round 8: Self-play, meta, world model | 2026-Q1 | ✅ Done |
| Round 9: Forge mode (CodeGenome + Pipeline) | 2026-Q2 | 🔄 Active |
| Round 10: Dream + Quorum + MoA + Reflexion | 2026-Q2 | 🔄 Active |
| Round 11: Full UI (Streamlit 2-tab) | 2026-Q2 | 📋 Planned |
| Round 12: Benchmarks (HumanEval, SWE-bench) | 2026-Q2 | 📋 Planned |
| v1.0 public release | 2026-Q3 | 📋 Planned |

---

## 7. Cost Model

### 7.1 API call estimate per evolution run

| Config | Evaluations | Mutations | Crossovers | Est. API calls |
|--------|-------------|-----------|------------|----------------|
| Quick (pop=5, gen=5) | 25 | ~20 | ~5 | ~55 |
| Default (pop=8, gen=10) | 80 | ~64 | ~16 | ~165 |
| Production (pop=12, gen=20) | 240 | ~190 | ~50 | ~490 |
| Large (pop=20, gen=30) | 600 | ~480 | ~120 | ~1,215 |

### 7.2 Cost by model (gpt-4o-mini @ $0.15/1M input tokens)

| Config | Est. tokens | Est. cost |
|--------|-------------|-----------|
| Quick | ~55k | ~$0.01 |
| Default | ~165k | ~$0.03 |
| Production | ~490k | ~$0.07 |
| Large | ~1.2M | ~$0.18 |

*Mutator prompts ~1,500 tokens each; evaluator varies by task.*

### 7.3 Cost optimisation levers

- Use `gpt-4o-mini` for mutation, reserve `gpt-4o` for LLM-judge evaluation.
- Enable `SpeculativeMutator` to amortise parallel mutations.
- Enable `ImmuneMemory` to skip re-evaluating barren genome regions.
- Enable prompt compression (`--compress-every 5`) to prevent prompt bloat.
- Use `MetaEvolutionEngine` to find optimal hyperparameters in fewer total calls.

---

## 8. Non-Goals (v1.0)

- Multi-modal agents (image/audio/video prompts) — deferred to v2.
- Real-time streaming evaluation — deferred to v2.
- Gradient-based joint training — out of scope (this is gradient-free by design).
- Distributed cluster execution — single-machine focus for v1.

---

## 9. Open Questions

1. Should Forge mode output standalone `.py` or a `CodeGenome` JSON that requires Cambrian to run?
2. Should the Pipeline evaluator support branching (if/else step routing)?
3. What safety guardrails are needed before public release of ToolInventor?
4. Should the dream phase write back to the main population or run as a side branch?
