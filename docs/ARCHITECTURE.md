# Cambrian — Architecture Reference

> Version: 0.9.0 · Last updated: 2026-04-15

---

## Overview

Cambrian is a **bio-inspired evolutionary framework** for automatically
optimising AI agent prompts and hyperparameters.  A population of agents is
evolved over multiple generations using LLM-guided mutation, multi-objective
fitness evaluation, and a growing library of bio-inspired pressures.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CAMBRIAN ARCHITECTURE                                │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    EvolutionEngine / CodeEvolutionEngine / PipelineEngine  │  │
│  │                                                                           │  │
│  │   seeds ──▶ [Evaluate] ──▶ [Select] ──▶ [Mutate/Crossover] ──▶ loop      │  │
│  │                │                │              │                          │  │
│  │                ▼                ▼              ▼                          │  │
│  │           Evaluator      Tournament       LLMMutator                      │  │
│  │           (fitness)      Selection        (mutator.py)                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Backends   │  │  Evaluators  │  │  Bio Pressures│  │   Analytics  │       │
│  │              │  │              │  │               │  │              │       │
│  │ OpenAICompat │  │ CodeEvaluator│  │  Lamarck      │  │ ParetoFront  │       │
│  │ AnthropicBack│  │ LLMJudge     │  │  Stigmergy    │  │ DiversityTrk │       │
│  │ GeminiBackend│  │ Composite    │  │  Epigenetics  │  │ FitnessLand. │       │
│  └──────────────┘  │ VarianceAware│  │  ImmuneMemory │  └──────────────┘       │
│                    │ BaldwinEval  │  │  MCTS         │                          │
│                    │ DiffCoTEval  │  │  Co-Evolution │  ┌──────────────┐       │
│                    │ WorldModelEv.│  │  Curriculum   │  │ Memory/Graph │       │
│                    │ ReflexionEv. │  │  Constitutional│  │ (memory.py)  │       │
│                    └──────────────┘  │  Self-Play    │  └──────────────┘       │
│                                      │  MetaEvolution│                          │
│  ┌──────────────┐                    │  DreamPhase   │  ┌──────────────┐       │
│  │  Reasoning   │                    │  QuorumSensor │  │  Structured  │       │
│  │              │                    └──────────────┘  │  Logging     │       │
│  │ DiffCoT      │  ┌──────────────┐                    │ (JSONLogger) │       │
│  │ CausalGraph  │  │  Tools       │  ┌──────────────┐  └──────────────┘       │
│  │ CausalMutator│  │ CLITool      │  │  Ensemble    │                          │
│  │ ReflexionAgt │  │ ToolInventor │  │ MixtureOfAgt │                          │
│  └──────────────┘  └──────────────┘  │ QuantumTunnl │                          │
│                                      └──────────────┘                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Forge Mode — Code & Pipeline Evolution                                   │   │
│  │  CodeGenome + CodeEvaluator + CodeMutator + CodeEvolutionEngine            │   │
│  │  Pipeline + PipelineRunner + PipelineEvaluator + PipelineMutator           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Data Model

### `Genome` (`cambrian/agent.py`)

The evolvable specification of an AI agent.

| Field              | Type              | Description                                |
|--------------------|-------------------|--------------------------------------------|
| `system_prompt`    | `str`             | System-level LLM instructions              |
| `strategy`         | `str`             | Reasoning hint (e.g. `"step-by-step"`)     |
| `temperature`      | `float`           | Sampling temperature (0.0–2.0)             |
| `model`            | `str`             | LLM model identifier                       |
| `tools`            | `list[str]`       | Allowed tool names (metadata)              |
| `few_shot_examples`| `list[dict]`      | Lamarckian examples injected at inference  |
| `genome_id`        | `str`             | Unique 8-char hex ID                       |

### `Agent` (`cambrian/agent.py`)

Wraps a `Genome` with a `LLMBackend` and tracks fitness.

```
Agent
 ├── genome: Genome          # evolvable configuration
 ├── backend: LLMBackend     # inference engine
 ├── agent_id: str           # UUID prefix
 ├── fitness: float | None   # last evaluation score
 └── _generation: int        # birth generation
```

Key methods: `run(task)`, `clone()`, `to_dict()`.

---

## Evolution Engine (`cambrian/evolution.py`)

The main driver.  One `EvolutionEngine` per run.

```
EvolutionEngine
 ├── evaluator: Callable[[Agent, str], float]
 ├── mutator: LLMMutator
 ├── backend: LLMBackend          # default backend for new agents
 ├── population_size: int
 ├── elite_n: int                 # elites that survive unchanged
 ├── mutation_rate: float
 ├── crossover_rate: float
 ├── tournament_k: int            # tournament selection pool size
 ├── compress_interval: int       # auto-compress every N gens
 ├── compress_max_tokens: int
 └── memory: EvolutionaryMemory   # lineage graph
```

**`evolve(seed_genomes, task, n_generations, on_generation)`**

Main loop:
1. Seed → initial population
2. For each generation:
   a. Evaluate all agents (parallel if backend is thread-safe)
   b. Sort by fitness; keep elites
   c. Fill remaining slots via tournament selection + mutate / crossover
   d. Optionally compress prompts
   e. Call `on_generation` callback
3. Return best agent

**Population save/load**: `save_population(path)` / `load_population(path)` — JSON serialisation of the full population.

---

## Mutator (`cambrian/mutator.py`)

```
LLMMutator
 ├── backend: LLMBackend
 ├── mutation_temperature: float
 ├── fallback_on_error: bool
 ├── memory: EvolutionaryMemory | None   # for stigmergy traces
 └── stigmergy_traces: int               # how many traces to inject
```

`mutate(agent, task)` → new Agent with LLM-improved genome.
`crossover(parent_a, parent_b, task)` → offspring via LLM combination.

Stigmergy: if `memory` is wired in, top-N pheromone traces are appended
to the mutation prompt, biasing offspring toward proven prompt patterns.

---

## Backends (`cambrian/backends/`)

All backends implement `LLMBackend.generate(prompt, **kwargs) → str`.

| Backend              | Class                  | SDK / Protocol         |
|----------------------|------------------------|------------------------|
| OpenAI-compatible    | `OpenAICompatBackend`  | httpx + JSON REST      |
| Anthropic Claude     | `AnthropicBackend`     | `anthropic` SDK        |
| Google Gemini        | `GeminiBackend`        | `google-genai` SDK     |

Environment variables:
- `OPENAI_API_KEY` / `CAMBRIAN_BASE_URL` for OpenAI-compatible
- `ANTHROPIC_API_KEY` for Anthropic
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini

All backends include exponential-backoff retry on rate-limit / server errors.

---

## Evaluators (`cambrian/evaluators/` + `cambrian/evaluator.py`)

```
Evaluator (ABC)
 ├── evaluate(agent, task) → float        # abstract
 └── __call__(agent, task) → float        # delegates to evaluate

Concrete evaluators:
 ├── CodeEvaluator          # executes code, checks output
 ├── LLMJudgeEvaluator      # LLM-as-judge with rubric
 ├── CompositeEvaluator     # weighted average of sub-evaluators
 ├── VarianceAwareEvaluator # mean - penalty*variance (anti-reward-hacking)
 └── BaldwinEvaluator       # multi-trial + improvement bonus
```

### Baldwin Effect (`cambrian/evaluators/baldwin.py`)

Evaluates the agent N times, prepending a feedback hint after each trial.
Fitness = base_score + `baldwin_bonus` × (last_score − first_score).

---

## Bio-Inspired Modules

### Lamarckian Evolution (`cambrian/lamarck.py`)

`LamarckianAdapter` wraps any evaluator.  When an agent scores above
`capture_threshold`, its (task, response, score) is added to the genome's
`few_shot_examples`.  These are injected into the system context at inference,
giving offspring access to proven solutions.

### Stigmergy (`cambrian/memory.py`)

Agents deposit `StigmergyTrace` objects (text + score) into the shared
`EvolutionaryMemory`.  `LLMMutator` retrieves the top-N traces and appends
them to the mutation prompt, biasing offspring toward high-fitness patterns.

Data flow:
```
evaluate agent → high score → add_trace(content, score)
                                      ↓
mutate(agent) → get_traces() → inject into mutation prompt
```

### Epigenetic Layer (`cambrian/epigenetics.py`)

```
EpigenomicContext          EpigeneticLayer
 ├── generation              ├── rules: list[EpigeneticRule]
 ├── task                    ├── express(genome, ctx) → str
 ├── population_mean_fitness └── apply(agent, ctx) → Agent (clone)
 ├── population_best_fitness
 └── extra: dict

make_standard_layer() → 4 rules:
  • generation_pressure  (explore/mid/exploit phase signal)
  • fitness_signal       (population standing)
  • task_mode            (coding / reasoning / creative hint)
  • population_pressure  (diversity collapse warning)
```

The genome is never mutated; only the runtime system prompt is modified.

### Artificial Immune System (`cambrian/immune.py`)

```
ImmuneMemory
 ├── register(agent)           # store fingerprint → best_fitness
 ├── is_suppressed(agent) → bool  # True if below threshold
 ├── recall_score(agent) → float | None
 └── suppression_rate(pop) → float

fingerprint(agent) → 16-char SHA-256 hex
  (normalised_prompt | strategy | temp_bucket | model)
```

Prevents re-evaluation of barren genome regions.
Suppression requires `min_evals_before_suppress` hits to avoid false positives.

### MCTS Selection (`cambrian/mcts.py`)

Monte Carlo Tree Search over the mutation tree:

```
MCTSNode
 ├── agent: Agent
 ├── visits, total_reward, depth
 └── ucb1() → float          # UCB1 = exploit + C*sqrt(ln(parent)/visits)

MCTSSelector
 ├── register(agent) → MCTSNode
 ├── select(population) → Agent     # highest UCB1
 ├── expand(agent, task) → list[Agent]
 ├── backpropagate(agent_id, reward)
 ├── best_path(root_id) → list[MCTSNode]
 └── prune_stale_roots(active_ids)
```

### Adversarial Co-Evolution (`cambrian/coevolution.py`)

Two populations evolve simultaneously:

```
CoEvolutionEngine
 ├── generator_population   # tries to solve the task
 ├── adversary_population   # tries to break generator solutions
 └── evolve(...) → (best_generator, best_adversary)

Fitness:
  generator = base_score × (1 - penalty × break_rate)
  adversary  = mean(break_rate across generators)
```

### Curriculum Learning (`cambrian/curriculum.py`)

```
CurriculumStage(task, difficulty, threshold, max_generations)

CurriculumScheduler
 ├── stages: list[CurriculumStage]
 ├── advance(fitness_values) → bool   # advance if threshold met
 ├── current_task() → str
 └── is_complete → bool

Factories:
  make_coding_curriculum()    # 4 stages: hello world → binary search
  make_reasoning_curriculum() # 5 stages: arithmetic → Tower of Hanoi
```

### DiffCoT Reasoning (`cambrian/diffcot.py`)

Diffusion-inspired iterative chain-of-thought denoising:

```
DiffCoTConfig
 ├── n_steps: int                    # denoising iterations (default 3)
 ├── noise_level: float              # perturbation intensity [0.0, 1.0]
 └── temperature_schedule: str       # "cosine" | "linear" | "constant"

DiffCoTReasoner
 └── reason(agent, task, backend) → str

DiffCoTEvaluator(base_evaluator, reasoner)
 └── evaluate(agent, task) → float   # runs DiffCoT, then base evaluator

make_diffcot_evaluator(base, backend, n_steps) → DiffCoTEvaluator
```

The denoiser starts at high temperature (exploration) and anneals toward
zero, producing progressively cleaner chain-of-thought reasoning.

### Causal Reasoning (`cambrian/causal.py`)

Explicit cause-effect representation in agent strategies:

```
CausalEdge(cause, effect, strength, confidence)

CausalGraph
 ├── from_text(text) → CausalGraph   # parse IF/THEN, →, "causes" patterns
 ├── edges: list[CausalEdge]
 └── to_prompt_block() → str

CausalStrategyExtractor(backend)
 └── extract(strategy_text) → CausalGraph

CausalMutator(backend)
 └── mutate_with_causal(agent, task) → Agent

inject_causal_context(genome, graph) → Genome
```

### Tool Creation (`cambrian/tool_creation.py`)

Agents invent new CLI tools during evolution:

```
ToolSpec(name, description, command_template, input_schema)

ToolInventor(backend, max_tools_per_agent)
 └── invent_tool(agent, task) → ToolInventionResult

ToolPopulationRegistry
 ├── register(tool_spec)
 ├── top_tools(n) → list[ToolSpec]
 └── deduplicate()
```

`ToolSpec` objects are stored in `Genome.tool_specs` and survive mutation/crossover.

### Self-Play (`cambrian/self_play.py`)

Head-to-head agent competition as an additional selection pressure:

```
SelfPlayResult(agent_a_id, agent_b_id, score_a, score_b, winner_id, margin, task)
 ├── is_draw → bool
 └── loser_id() → str | None

SelfPlayEvaluator(base_evaluator, win_bonus, loss_penalty, draw_bonus)
 ├── evaluate(agent, task) → float
 └── compete(a, b, task) → SelfPlayResult   # applies win/loss fitness delta

TournamentRecord
 ├── wins / losses / draws / total_score per agent
 ├── ranking() → list[(agent_id, win_rate)]
 └── win_rate(agent_id) → float

run_tournament(population, evaluator, task) → TournamentRecord
```

### Meta-Evolution (`cambrian/meta_evolution.py`)

MAML-inspired outer loop that evolves hyperparameters alongside genomes:

```
HyperParams(mutation_rate, crossover_rate, temperature, tournament_k, elite_ratio)
 ├── clamp() → HyperParams          # clip all values to valid ranges
 ├── perturb(scale, rng) → HyperParams   # Gaussian perturbation
 ├── to_dict() / from_dict()
 └── fitness_history: list[float]

MetaEvolutionEngine
 ├── evolve(seed_genomes, task, n_generations, meta_interval, on_generation) → Agent
 ├── hp: HyperParams                # current hyperparameters (updated each meta-step)
 └── hp_history: list[HyperParams]  # all configurations tried
```

At every `meta_interval` generations, `n_candidates` perturbed HP configs are
tried on one quick evaluation step.  The best-performing config is kept.

### World Model (`cambrian/world_model.py`)

Per-agent predictive model inspired by Dyna-Q / Ha & Schmidhuber 2018:

```
WorldModelPrediction(predicted_score, confidence, n_similar)
 └── is_uncertain → bool    # confidence < 0.5

WorldModel(buffer_size, default_score, decay)
 ├── update(task, score)             # add experience
 ├── predict(task) → WorldModelPrediction
 └── experience_count() → int

WorldModelEvaluator(base_evaluator, accuracy_weight, buffer_size, decay, min_confidence_for_blend)
 ├── evaluate(agent, task) → float   # blend raw score + prediction accuracy
 ├── get_model(agent_id) → WorldModel | None
 └── prediction_errors() → dict[str, float]

world_model_fitness(raw_score, prediction_error, accuracy_weight) → float
```

Agents that **predict their own performance accurately** receive a fitness
bonus — creating selective pressure for self-aware agents.

### Dream Phase (`cambrian/dream.py`)

Mimics memory consolidation during sleep — agents are re-evaluated on LLM-generated
synthetic scenarios that blend past experiences:

```
Experience(task, response, score, genome_id)

DreamScenario(task, source_scores, blend_description)
 └── expected_difficulty → float   # mean of source scores

DreamPhase(backend, n_experiences, n_dreams, blend_weight, min_score_threshold)
 ├── generate_scenario(experiences) → DreamScenario
 └── run(population, task, evaluator, experiences)
       For each agent:
         • Evaluate on n_dreams synthetic scenarios → dream_score
         • Blend: new_fitness = (1-w)*real + w*dream   (clips to [0,1])

extract_experiences_from_memory(memory, task, limit) → list[Experience]
```

Useful as a **diversity maintenance tool**: agents that generalise well across dream
scenarios receive higher blended fitness than specialists.

### Quorum Sensing (`cambrian/quorum.py`)

Monitors Shannon entropy of the population fitness distribution and auto-adjusts
`mutation_rate` and `elite_n` toward a target entropy:

```
QuorumState(generation, entropy, mutation_rate, elite_n, population_size, diversity_ratio)

QuorumSensor(n_bins, target_entropy, min/max_mutation_rate, min/max_elite_ratio, lr)
 ├── compute_entropy(fitnesses) → float     # normalised Shannon entropy [0,1]
 ├── update(population) → QuorumState       # adjust rate: rate += lr*(target-H)
 ├── stagnation_detected(window, threshold) → bool
 ├── should_inject_diversity(low_threshold, window) → bool
 └── summary() → dict

Feedback loop:
  low entropy (converged) → raise mutation_rate, raise elite_n
  high entropy (chaotic)  → lower mutation_rate, lower elite_n
```

### Mixture of Agents (`cambrian/moa.py`)

N independent agents answer in parallel; an aggregator LLM synthesises a final answer:

```
MoAResult(final_answer, individual_answers, n_agents)

MixtureOfAgents(agents, aggregator_backend, aggregator_temperature, n_agents)
 ├── run(task) → MoAResult
 ├── _aggregate(task, answers) → str    # LLM synthesises; fallback = longest answer
 └── from_population(population, aggregator_backend) → MixtureOfAgents   # factory
```

Robust to single-agent failures — the aggregator falls back to the longest answer
if its LLM call fails.

### Quantum Tunneling (`cambrian/moa.py`)

Prevents population convergence by probabilistically replacing non-elite agents with
fresh random genomes — analogous to quantum tunneling out of local optima:

```
TunnelEvent(generation, agent_id, old_fitness, genome_id)

QuantumTunneler(tunnel_prob, protect_elites, n_elites, seed)
 ├── apply(population, model) → list[Agent]
 │     For each non-elite: replace at tunnel_prob with random genome
 ├── _random_genome(model) → Genome      # sampled from predefined diversity pool
 ├── tunnel_count: int                   # total replacements so far
 └── summary() → dict
```

### Reflexion (`cambrian/reflexion.py`)

Generate → critique → revise loop (Shinn et al. 2023):

```
ReflexionRound(round_number, response, critique, improved)

ReflexionResult(final_response, rounds, task, n_rounds_used)
 ├── initial_response → str    # round 0 response
 └── improved → bool           # True if any round was non-trivial

ReflexionAgent(agent, n_rounds, critique_temperature, revision_temperature,
               stop_if_excellent, reflection_backend)
 └── run(task) → ReflexionResult

ReflexionEvaluator(base_evaluator, n_rounds, reflection_backend)
 └── evaluate(agent, task) → float
       1. Run Reflexion to produce improved_response
       2. Create _ProxyAgent (wraps agent, overrides run() → improved_response)
       3. Score proxy via base_evaluator
```

Early-exit: if critique starts with `"EXCELLENT:"` and `stop_if_excellent=True`,
the loop stops and the current response is returned.

### Constitutional AI (`cambrian/constitutional.py`)

```
ConstitutionalWrapper(base_evaluator, constitution, n_revisions)
 └── __call__(agent, task) → float

Cycle per evaluation:
  1. Gather critiques against each constitutional principle
  2. If any critique is non-OK: revise the system prompt
  3. Score revised prompt with base_evaluator
  4. Always restore original genome prompt (finally block)
```

### Knowledge Distillation (CLI `distill-agent`)

Compresses an evolved large-model genome for deployment on a smaller model:

```
1. caveman_compress()  — remove stopwords/filler
2. procut_prune()      — paragraph pruning to token budget
3. LLM adaptation      — model-specific prompt rewrite
```

---

## Forge Mode

Forge mode evolves **executable artifacts** (code, pipelines) rather than prompt text.
It uses separate genome and engine classes that follow the same generational loop pattern.

### Code Genome (`cambrian/code_genome.py`)

```
TestCase(input_data, expected_output, weight, label)

CodeGenome(code, description, version, genome_id, parent_id, metadata)
 └── loc() → int                 # non-blank, non-comment lines

CodeEvaluationResult(pass_rate, runtime_s, loc, passed, total, error, timed_out, fitness)
 └── fitness = 0.95*pass_rate + 0.02*loc_term + 0.03*runtime_term

CodeEvaluator(test_cases, timeout, loc_bonus, runtime_bonus_cap)
 └── evaluate(genome) → CodeEvaluationResult

CodeMutator(backend, mutation_temperature, fallback_on_error)
 ├── mutate(genome, task) → CodeGenome    # LLM rewrites code
 └── _seed(task) → CodeGenome             # generate from scratch if empty

CodeEvolutionEngine(backend, evaluator, population_size, elite_ratio, crossover_rate)
 └── evolve(seed, task, n_generations, on_generation) → CodeGenome
       Early exit when fitness >= 1.0
```

Each test case is run in a **subprocess sandbox** (`utils/sandbox.py`) with API keys
stripped.  Fitness blends pass rate (95%), LOC efficiency (2%), and runtime (3%).

### Pipeline Genome (`cambrian/pipeline.py`)

```
PipelineStep(role, system_prompt, temperature, step_id)

Pipeline(steps, description, version, pipeline_id, parent_id, metadata)
 └── step_count() → int

PipelineRunner(backend)
 └── run(pipeline, task) → str     # chain: output[N] → input[N+1]; "" if empty

PipelineEvaluator(backend, score_fn, empty_pipeline_penalty)
 └── evaluate(pipeline) → float    # score_fn(output, task); clips to [0,1]

PipelineMutator(backend, mutation_temperature, fallback_on_error)
 ├── mutate(pipeline, task) → Pipeline      # LLM adds/removes/reorders steps
 └── crossover(parent_a, parent_b) → Pipeline  # dedup roles, max 6 steps

PipelineEvolutionEngine(backend, evaluator, population_size, elite_ratio, crossover_rate)
 └── evolve(seed, task, n_generations, on_generation) → Pipeline
```

---

## Memory & Analytics

### EvolutionaryMemory (`cambrian/memory.py`)

Directed graph (NetworkX `DiGraph`) of agent lineage.

```
Nodes: agent_id → {generation, fitness, genome_snapshot}
Edges: parent_id → child_id

Methods:
  add_agent(id, gen, fitness, genome, parents)
  update_fitness(id, fitness)
  get_top_ancestors(n, min_fitness)
  get_lineage(agent_id) → list[str]    # ancestor chain
  add_trace(agent_id, content, score)  # stigmergy
  get_traces(task, limit)              # stigmergy retrieval
  to_json() / from_json()
```

### Population Statistics (`cambrian/stats.py`)

| Class              | What it computes                                     |
|--------------------|------------------------------------------------------|
| `ParetoFront`      | Non-dominated agents (fitness × brevity + custom)    |
| `DiversityTracker` | Per-gen snapshots: entropy, temp_std, prompt_std     |
| `FitnessLandscape` | 2D grid mean fitness (temp × token bins)             |

---

## CLI Commands

```
cambrian evolve TASK              Run prompt evolution (pop, gens, mutation, crossover flags)
cambrian forge TASK               Forge mode: evolve code or pipelines (--mode code|pipeline)
cambrian run --agent FILE         Load evolved genome, run on a task
cambrian analyze MEMORY           Deep analysis: trajectory, diversity, lineage
cambrian snapshot --memory FILE   Show population state at a specific generation
cambrian compare RUN1 RUN2        Compare two NDJSON evolution log files
cambrian dashboard                Streamlit live dashboard (--port, --log-file)
cambrian distill GENOME           Pretty-print a saved genome JSON
cambrian distill-agent            Compress evolved genome for a smaller model
cambrian version                  Print version
```

---

## File Layout

```
cambrian/
├── agent.py             Genome, Agent
├── evolution.py         EvolutionEngine (main loop)
├── mutator.py           LLMMutator (LLM-guided mutation + stigmergy)
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
├── stats.py             ParetoAnalyzer, DiversityTracker, FitnessLandscape
├── diffcot.py           DiffCoTConfig, DiffCoTReasoner, DiffCoTEvaluator
├── causal.py            CausalEdge, CausalGraph, CausalMutator, inject_causal_context
├── tool_creation.py     ToolSpec, ToolInventor, ToolPopulationRegistry
├── self_play.py         SelfPlayEvaluator, SelfPlayResult, TournamentRecord, run_tournament
├── meta_evolution.py    HyperParams, MetaEvolutionEngine
├── world_model.py       WorldModel, WorldModelEvaluator, WorldModelPrediction
│
│  ─── Round 9: Forge Mode ───
├── code_genome.py       CodeGenome, CodeEvaluator, CodeMutator, CodeEvolutionEngine, TestCase
├── pipeline.py          Pipeline, PipelineStep, PipelineRunner, PipelineEvaluator,
│                        PipelineMutator, PipelineEvolutionEngine
│
│  ─── Round 9: Adaptive Pressures ───
├── dream.py             DreamPhase, DreamScenario, Experience, extract_experiences_from_memory
├── quorum.py            QuorumSensor, QuorumState
│
│  ─── Round 9: Ensemble & Reflexion ───
├── moa.py               MixtureOfAgents, MoAResult, QuantumTunneler, TunnelEvent
├── reflexion.py         ReflexionAgent, ReflexionResult, ReflexionRound, ReflexionEvaluator
│
├── dashboard.py         Streamlit dashboard (Evolve + Forge + Logs tabs)
├── cli.py               Click CLI entry point (10 commands incl. forge)
├── __main__.py          python -m cambrian entry point
├── backends/
│   ├── base.py          LLMBackend ABC
│   ├── openai_compat.py OpenAICompatBackend (httpx)
│   ├── anthropic.py     AnthropicBackend (anthropic SDK)
│   └── gemini.py        GeminiBackend (google-genai SDK)
├── evaluators/
│   ├── code.py          CodeEvaluator (prompt-based; see code_genome.py for Forge)
│   ├── llm_judge.py     LLMJudgeEvaluator
│   ├── composite.py     CompositeEvaluator
│   ├── variance_aware.py VarianceAwareEvaluator + build_diversified_evaluator
│   └── baldwin.py       BaldwinEvaluator
└── utils/
    ├── logging.py       get_logger, JSONLogger, load_json_log
    └── sandbox.py       run_in_sandbox (API-key-safe), extract_python_code
```

---

## Extension Points

| Goal                        | Where to hook in                              |
|-----------------------------|-----------------------------------------------|
| Custom fitness metric       | Implement `Evaluator.evaluate()` or pass any callable |
| Custom mutation strategy    | Subclass `LLMMutator` or pass custom prompts  |
| New LLM provider            | Subclass `LLMBackend`                         |
| Custom epigenetic rule      | `EpigeneticLayer.add_rule(fn)`                |
| Custom MCTS expansion       | Override `MCTSSelector.expand()`              |
| Post-generation hook        | Pass `on_generation` to `EvolutionEngine.evolve()` |
| Immune suppression policy   | Subclass `ImmuneMemory`                       |
