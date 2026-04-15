# Changelog

All notable changes to Cambrian are documented here.

---

## [1.0.3] — 112 new tests, DEPLOYMENT.md, copyright headers, asyncio fix

**conftest.py, test_export, test_a2a, test_cli_tools, DEPLOYMENT.md.**  
**1741 tests passing.** Zero mypy errors. Zero ruff warnings.

### Added
- **`tests/conftest.py`** — reusable fixtures: `mock_backend`, `mock_backend_text`,
  `sample_genome`, `expert_genome`, `sample_agent`, `scored_agent`, `sample_population`.
- **`tests/test_export.py`** — 35 tests covering all four export formats (`export_genome_json`,
  `load_genome_json`, `export_standalone`, `export_mcp`, `export_api`), including Python syntax
  validation, round-trip JSON integrity, and parent-dir auto-creation.
- **`tests/test_a2a.py`** — 45 tests covering `AgentCard`, `A2AMessage`, `AgentNetwork`
  (register, route, delegate, broadcast, chain, majority_vote, summary).
- **`tests/test_cli_tools.py`** — 32 tests covering `CLITool`, `CLIToolResult`, `CLIToolkit`,
  and the `make_python_tool` / `make_shell_tool` factories.
- **`docs/DEPLOYMENT.md`** — deployment guide for Render (Web Service), Docker
  (Dockerfile + docker-compose), and Kubernetes (Deployment + Service + Ingress + HPA).

### Fixed
- **`tests/test_round6.py`**: replaced `WindowsSelectorEventLoopPolicy` with
  `WindowsProactorEventLoopPolicy` in `_run_async` — eliminates `WinError 10055` socket
  buffer exhaustion when the full 1741-test suite runs on Windows (Proactor uses IOCP,
  not `socketpair()`). Two previously-failing `TestSpeculate` tests now pass.
- **`scripts/benchmark.py`**: removed extraneous `f` prefix from 3 plain strings (ruff F541).
- **Copyright headers**: added `# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT`
  to 15 files that were missing it (`cambrian/backends/`, `cambrian/evaluators/`,
  `cambrian/utils/`, `scripts/benchmark.py`).

---

## [1.0.2] — CLI Round 8, TUTORIAL, Benchmark, Comparison

**New CLI commands, docs, benchmark, VISION, COMPARISON, CLI tests, ENV_VARS.**  
**1629 tests passing.** Zero mypy errors. Zero ruff warnings.

### Added
- **`cambrian meta-evolve` CLI command** — wraps `MetaEvolutionEngine`; co-evolves
  agents and hyperparameters (mutation rate, temperature, tournament_k) simultaneously.
  Prints live hp stats per generation.
- **`cambrian tournament` CLI command** — runs a round-robin `SelfPlayEvaluator`
  tournament across a population; prints a ranked leaderboard (W/L/D); optionally
  saves JSON results.
- **`docs/TUTORIAL.md`** — complete step-by-step tutorial: install → Evolve mode →
  Forge mode → analyse results → export agent → mock backend (no API key needed).
- **`examples/benchmark.py`** — standalone performance benchmark: 4 scenarios
  (10×10 through 50×100), wall-clock time, `tracemalloc` peak memory, simulated
  token count, throughput (evals/s). No API key required.
- **`py.typed` marker** — PEP 561 compliance; downstream typed packages can now
  use Cambrian without `ignore_missing_imports`.
- **`docs/API_REFERENCE.md`** — added DPO & Safeguards section with `DPOPair`,
  `DPOSelector`, `DPOTrainer`, `GoalDriftDetector`, `DriftEvent`,
  `FitnessAnomalyDetector`, `SafeguardController`. Bumped to v1.0.1 → v1.0.2.
- **`docs/COMPARISON.md`** — detailed feature-by-feature comparison with DSPy,
  DGM, AVO, EvoAgent, MiroFish; includes architectural comparison tables.
- **`VISION.md`** — project vision: Era 1 (prompt evolution) → Era 2 (ecosystem
  intelligence) → Era 3 (autonomous self-improvement). The "Cambrian Explosion"
  analogy. Guiding principles.
- **`docs/ENV_VARS.md`** — complete reference for all supported environment
  variables (`OPENAI_API_KEY`, `CAMBRIAN_BASE_URL`, `OPENAI_API_BASE`,
  `CAMBRIAN_LOG_LEVEL`) with provider examples.
- **`tests/test_cli_round8.py`** — 36 tests for `meta-evolve`, `tournament`,
  and `forge` CLI commands; covers --help output, defaults, file I/O, error cases.

### Changed
- **`README.md`** — "Cambrian vs the Field" comparison table (DSPy, DGM, AVO,
  TextGrad feature checklist); CLI section documents `meta-evolve` and `tournament`;
  test badge updated to 1619.
- **`CHANGELOG.md`** — corrected test badge from 1494 → 1583 → 1619.

### Fixed
- `cambrian/cli.py`: `_make_evaluator` call signature in `tournament` command
  (wrong number of arguments — mypy caught this).
- `examples/demo_end_to_end.py`: E402 module-level import not at top; F541
  f-string without placeholder (both ruff fixes).

---

## [1.0.1] — Tier 5: Lifecycle, Ecology & Fractal Evolution

**New modules:** Metamorphosis, Ecosystem, Fractal, DPO selection, SAHOO Safeguards.
**1583 tests passing.** Zero mypy errors. Zero ruff warnings.

### Added
- **`cambrian/metamorphosis.py`** — Holometabolous agent lifecycle: LARVA (broad
  exploration, 1.5x mutation) → CHRYSALIS (LLM-driven genome reorganisation,
  frozen mutation) → IMAGO (exploit, 0.5x mutation). `MetamorphosisController`
  tracks per-agent phase, enforces min-generations + fitness-threshold criteria,
  and calls `metamorphose()` for chrysalis reorganisation. `MetamorphicPopulation`
  orchestrates population-wide ticks.
- **`cambrian/ecosystem.py`** — 4-role ecological fitness dynamics: HERBIVORE
  (diversity bonus per unique strategy), PREDATOR (hunts weak agents), DECOMPOSER
  (recycles low-fitness agents), PARASITE (drains strongest host). Deltas are
  collected atomically before application. `EcosystemEvaluator` blends base score
  with ecological signal. `auto_assign()` distributes roles by fitness rank.
- **`cambrian/fractal.py`** — Recursive multi-scale evolution: MACRO → MESO → MICRO.
  Each scale seeds the next from its elite, results bubble back up. `FractalMutator`
  applies broad/medium/localised genome rewrites per scale. `FractalEvolution`
  orchestrates the full recursive loop.
- **`cambrian/dpo.py`** — Direct Preference Optimization as alternative selection:
  `DPOSelector` builds preferred/rejected pairs and applies DPO fitness bonuses.
  `DPOTrainer` uses an LLM backend to refine bottom-50% agents toward chosen patterns.
- **`cambrian/safeguards.py`** — SAHOO-inspired safety monitors: `GoalDriftDetector`
  uses Jaccard word-overlap to flag agents diverging from their original intent.
  `FitnessAnomalyDetector` catches reward-hacking spikes via z-score. `SafeguardController`
  orchestrates both + optional LLM-guided remediation.
- **`docs/API_REFERENCE.md`** — Comprehensive reference: 119 public symbols across
  15 sections, full parameter tables and return types.
- **`examples/demo_full_evolution.py`** — Offline demo combining 8 techniques
  (QuorumSensor, DreamPhase, ApoptosisController, MetamorphosisController,
  EcosystemInteraction, NeuromodulatorBank) — runs instantly with mocked backends.

### Tests
- 61 tests for metamorphosis (phase transitions, chrysalis reorg, phase pressure)
- 67 tests for ecosystem (auto-assign, interact, apply_events, EcosystemEvaluator)
- 60 tests for fractal (scale ordering, mutations, FractalEvolution end-to-end)
- 43 integration tests combining 3+ techniques (QuorumSensor+Apoptosis+Neuro, etc.)
- Edge-case tests: IMAGO terminal lock, extinction scenarios, singleton populations

### Fixed
- Flaky test `test_speculate_returns_best_candidate` — WinError 10055 (socket buffer
  exhaustion) fixed with `asyncio.run()` + `WindowsSelectorEventLoopPolicy` + retry

---

## [1.0.0] — First Stable Release

**Cambrian is feature-complete and production-stable.**

Sixty-six evolutionary techniques across four tiers, 1264 tests, zero mypy
errors, zero ruff warnings.  This release consolidates the entire development
arc from the initial genome/agent skeleton through all Tier 3 and Tier 4
bio-inspired extensions.

### Summary of the full implementation arc

| Version | Round | Highlights |
|---------|-------|------------|
| 0.1.0 | 1 | Genome, Agent, Evaluator ABC, LLMBackend ABC |
| 0.3.0 | 3 | EvolutionEngine, LLMMutator, OpenAICompatBackend, CodeEvaluator, CLI |
| 0.4.0 | 4 | Epigenetics, ImmuneMemory (fingerprint), LamarckianAdapter, EvolutionaryMemory, Stigmergy, DiversityTracker, FitnessLandscape, LLMJudgeEvaluator, CompositeEvaluator, Anthropic/Gemini backends |
| 0.5.0 | 5 | A2A protocol, CLITools, advanced CLI (analyze/dashboard/distill), VarianceAwareEvaluator, BaldwinEvaluator, CoEvolution, Curriculum, ConstitutionalAI, MCTSSelector |
| 0.6.0 | 6 | SpeculativeMutator, Archipelago/Island model, NSGA-II, RewardShaping, Export (JSON/standalone/MCP/FastAPI), `cambrian run` CLI |
| 0.7.0 | 7 | DiffCoT, CausalReasoning, ToolInventor, `cambrian snapshot` CLI, SECURITY.md |
| 0.8.0 | 8 | SelfPlay, MetaEvolution, WorldModel, JSONLogger, `cambrian compare` CLI, sandbox env-leak fix |
| 0.9.0 | 9 | PRD (57-technique inventory), Forge mode (CodeEvolution + PipelineEvolution), Dashboard 2-tab UI, `cambrian forge` CLI |
| 0.10.0 | 10 | DreamPhase, QuorumSensor, MixtureOfAgents, QuantumTunneler, ReflexionEvaluator |
| 0.11.0 | Tier 3 | SymbioticFuser, HormesisAdapter, ApoptosisController, CatalysisEngine, LLMCascade, AgentEnsemble/BoostingEnsemble, GlossaloliaReasoner, BestOfN/BeamSearch |
| 0.12.0 | Tier 4a | TransferAdapter/Bank, TabuList/Mutator, AnnealingSchedule/Selector, RedTeamAgent/RobustnessEvaluator, ZeitgeberClock/Scheduler, HGTransfer/HGTPool, TransgenerationalRegistry |
| 0.13.0 | Tier 4b | BCellMemory, TCellMemory, ImmuneCortex, NeuromodulatorBank (Dopamine/Serotonin/Acetylcholine/Noradrenaline) |

### This release
- Bumped `pyproject.toml` version to `1.0.0`; `Development Status :: 5 - Production/Stable`
- Bumped `cambrian.__version__` to `"1.0.0"`
- Updated `docs/ARCHITECTURE.md` header to version 0.19.0
- Full test suite: **1264 passed**, 0 mypy errors, 0 ruff warnings

---

## [0.13.0] — Tier 4 Part 2: Immune Memory & Neuromodulation

### Added

#### B/T-cell Immune Memory (Technique 65)
- **`cambrian/immune_memory.py`** — `BCellMemory`: stores high-fitness
  (genome, task) pairs; `recall(task)` returns the best Jaccard match above
  a configurable similarity threshold (fast path — analogous to antibody
  recognition).  `TCellMemory`: adaptive lookup returning the most similar
  stored cell even below the B-cell threshold, useful for seeding evolution
  from a related starting point.  `ImmuneCortex`: coordinator that gates
  storage on fitness thresholds and checks B-cell first, T-cell second,
  returning a typed `RecallResult`.

#### Neuromodulation (Technique 66)
- **`cambrian/neuromodulation.py`** — Four biologically-inspired modulators:
  `DopamineModulator` (rising fitness → exploit), `SerotoninModulator`
  (low diversity → explore), `AcetylcholineModulator` (high variance →
  lower selection pressure), `NoradrenalineModulator` (stagnation →
  exploration spike).  `NeuromodulatorBank` aggregates all four and
  produces clamped `mutation_rate` and `selection_pressure` via
  `modulate(population, generation)`.

#### Exports
- `cambrian.__init__`: exported `ImmuneCortex`, `BCellMemory`, `TCellMemory`,
  `MemoryCell`, `RecallResult`, `NeuromodulatorBank`, `NeuroState`,
  `DopamineModulator`, `SerotoninModulator`, `AcetylcholineModulator`,
  `NoradrenalineModulator` (11 new symbols).

### Tests
- 83 new tests across `test_immune_memory.py` (44) and
  `test_neuromodulation.py` (39).
- Full suite: **1194 passed**.

---

## [0.12.0] — Tier 4 Part 1: Transfer, Tabu, Annealing, Red Teaming, Zeitgeber, HGT, Transgenerational

### Added

#### Transfer Learning (Technique 57)
- **`cambrian/transfer.py`** — `TransferAdapter(backend, intensity)`: adapts a
  source genome to a target task via LLM at light/medium/heavy intensity.
  `TransferBank(max_per_domain)`: registers agents by domain; `best_for(domain)`
  retrieves the highest-fitness source genome.

#### Tabu Search (Technique 58)
- **`cambrian/tabu.py`** — `TabuList(max_size)`: FIFO bi-gram fingerprint list
  (SHA-256 hex[:16]) preventing revisitation of recent genome regions.
  `TabuMutator(base_mutator, tabu_list, max_retries)`: retries mutation up to
  `max_retries` times; `tabu_hit_rate` property for monitoring.

#### Simulated Annealing (Technique 59)
- **`cambrian/annealing.py`** — `AnnealingSchedule(T_max, T_min, n_steps,
  schedule_type)`: linear, exponential, and cosine cooling curves.
  `AnnealingSelector.step(current_fitness, candidate_fitness)`: Metropolis
  acceptance criterion; tracks `acceptance_rate` and full history.

#### Red Teaming (Technique 60)
- **`cambrian/red_team.py`** — `RedTeamAgent(backend, n_attacks)`: LLM-generated
  adversarial attacks with JSON parsing and fallback perturbations.
  `RobustnessEvaluator(judge_backend)`: scores agent robustness 0–1 via regex.
  `RedTeamSession.run(agent, task)` → `RobustnessReport` combining normal and
  adversarial performance.

#### Zeitgeber (Technique 61)
- **`cambrian/zeitgeber.py`** — `ZeitgeberClock(period, amplitude, phase_offset)`:
  sinusoidal circadian oscillator; `exploration_factor()` ∈ [0.5-amp/2, 0.5+amp/2].
  `ZeitgeberScheduler`: maps oscillator to `mutation_rate` and `threshold`
  via configurable base values and ranges.

#### Horizontal Gene Transfer (Technique 62)
- **`cambrian/hgt.py`** — `HGTransfer(n_sentences, mode, fitness_threshold)`:
  extracts sentence-level genome fragments (plasmids) from high-fitness donors;
  injects via prefix/suffix/replace modes.  `HGTPool(max_plasmids)`: domain-tagged
  plasmid pool with `contribute()`, `draw()`, `best_for()`.

#### Transgenerational Epigenetics (Technique 64)
- **`cambrian/transgenerational.py`** — `EpigeneMark`: named annotation with
  strength and decay.  `TransgenerationalRegistry`: records marks, decays them
  per generation, inherits top-N to offspring with additional decay,
  injects context into genome system prompt.

#### Exports
- All Tier 4 Part 1 symbols added to `cambrian.__init__`.

### Tests
- 141 new tests across 7 test files.
- Full suite: **1111 passed** before Tier 4 Part 2.

---

## [0.11.0] — Tier 3: Symbiosis, Hormesis, Apoptosis, Catalysis, LLM Cascade, Ensemble, Glossolalia, Inference Scaling

### Added

#### Symbiotic Fusion (Technique 51)
- **`cambrian/symbiosis.py`** — `SymbioticFuser(backend, fitness_threshold,
  min_distance)`: LLM-driven endosymbiosis — merges genomes of compatible
  host/donor pairs (high fitness AND low word overlap). Falls back to
  naive concatenation on LLM failure.  `fuse_best_pair(population, task)`.

#### Hormesis (Technique 52)
- **`cambrian/hormesis.py`** — `HormesisAdapter(backend, stress_threshold)`:
  graduated stress response — mild (temperature boost), moderate (hint
  injection), severe (LLM re-prompt).  `stimulate_population(population, task)`.

#### Apoptosis (Technique 53)
- **`cambrian/apoptosis.py`** — `ApoptosisController(stagnation_window,
  min_fitness, grace_period)`: programmed removal of chronically poor agents;
  tracks fitness history per agent; optionally replaces removed agents with
  clones of the best survivor.

#### Catalysis (Technique 54)
- **`cambrian/catalysis.py`** — `CatalystSelector.select(population)`: picks
  catalyst by composite fitness+vocab+strategy score.  `CatalysisEngine.
  catalyse(target, catalyst, task)`: temporarily augments target prompt with
  catalyst context; always restores via `finally`.

#### LLM Cascade (Technique 55)
- **`cambrian/llm_cascade.py`** — `LLMCascade(levels)`: routes queries through
  a tiered list of `CascadeLevel` (backend + confidence_fn + threshold);
  escalates when confidence is too low.  Built-in scorers: `hedging_confidence`,
  `length_confidence`.

#### Ensemble / Boosting (Technique 56)
- **`cambrian/ensemble.py`** — `AgentEnsemble`: weighted majority vote across
  agents.  `BoostingEnsemble(AgentEnsemble)`: AdaBoost-style weight updates.
  Scorers: `exact_match_scorer`, `substring_scorer`.

#### Glossolalia (Technique 49)
- **`cambrian/glossolalia.py`** — `GlossaloliaReasoner`: two-phase latent
  monologue → structured synthesis; configurable temperature differential.
  `GlossaloliaEvaluator(Evaluator)`: wraps an inner evaluator for use in
  evolution loops.

#### Inference-time Scaling (Technique 50)
- **`cambrian/inference_scaling.py`** — `BestOfN`: generate N candidates,
  return highest-scoring.  `BeamSearch`: beam_width × branching_factor tree
  search with configurable n_steps.  Scorers: `length_scorer`,
  `KeywordScorer`, `SelfConsistencyScorer`.

#### Exports
- All Tier 3 symbols added to `cambrian.__init__`.

### Tests
- 152 new tests across 8 test files.
- Full suite grows to **970 passed** (before Tier 4).

---

## [0.10.0] — Round 10

### Added

#### Dream Phase (Technique 16)
- **`cambrian/dream.py`** — `DreamPhase`: GraphRAG-style offline recombination of
  high-fitness ancestor genomes from the lineage graph.  `should_dream(gen)` fires
  every configurable interval of generations.  `dream(task, n_offspring)` queries
  the lineage, formats an experience context, and asks the LLM to synthesise novel
  genome variants.  `dream_count` tracks total dream events.

#### Quorum Sensing (Technique 15)
- **`cambrian/quorum.py`** — `QuorumSensor`: Shannon entropy of the fitness
  distribution auto-regulates mutation rate.  Low diversity → boost;
  high diversity → decay.  Configurable thresholds, boost/decay factors, and
  rate clamps.  Per-call history tracking and `reset()`.

#### Mixture of Agents (Technique 36)
- **`cambrian/moa.py`** — `MixtureOfAgents`: Runs N agents on the same task,
  then an aggregator LLM synthesises the best answer from all responses.
  Handles individual failures gracefully; falls back to longest response on
  aggregator failure.

#### Quantum Tunneling (Technique 17)
- **`cambrian/moa.py`** — `QuantumTunneler`: Stochastic large-jump mutation.
  With probability `tunnel_prob`, replaces an agent's genome with a fully
  randomised variant (temperature, strategy, optional LLM-generated prompt).
  `tunnel_all()` applies to an entire population.

#### Reflexion (Technique 35)
- **`cambrian/reflexion.py`** — `ReflexionEvaluator`: generate → reflect →
  revise loop for `n_reflections` cycles.  Critique LLM returns CRITIQUE/SCORE
  format; revision LLM improves the response.  Early exit on perfect score.

#### Exports
- `cambrian.__init__`: exposed `DreamPhase`, `QuorumSensor`, `MixtureOfAgents`,
  `QuantumTunneler`, `ReflexionEvaluator` (40 total exports).

### Tests
- 86 new tests across `test_dream.py`, `test_quorum.py`, `test_moa_reflexion.py`.
- Full suite: **781 passed**.

---

## [0.9.0] — Round 9

### Added

#### PRD
- **`docs/PRD.md`** — Full Product Requirements Document: Evolve + Forge modes,
  57-technique inventory, UI spec (CLI/Dashboard/Python API), data models,
  timeline, cost model, non-goals.

#### Forge Mode — Code Evolution (Techniques 39–43)
- **`cambrian/code_genome.py`** — `CodeGenome` (executable Python as evolvable
  artifact), `CodeAgent` (fitness tracking), `CodeMutator` (LLM rewrite +
  crossover), `CodeEvaluator` (sandbox test-case scoring: empty→0.0,
  error→0.1, partial→0.1+0.9×k/N, perfect→1.0), `CodeEvolutionEngine`
  (full generational loop with elitism, tournament, crossover).

#### Forge Mode — Pipeline Evolution (Techniques 44–48)
- **`cambrian/pipeline.py`** — `PipelineStep` (named step with system prompt,
  role, temperature), `Pipeline` (ordered step list + version counter),
  `PipelineMutator` (LLM adds/removes/reorders steps; JSON parsing with fence
  stripping; max_steps enforcement), `PipelineEvaluator` (sequential execution,
  exact-match or LLM-judge scoring), `PipelineEvolutionEngine`.

#### Forge CLI (Technique 48)
- **`cambrian forge TASK`** — new CLI command with `--mode code|pipeline`,
  `--test-case INPUT:EXPECTED`, `--seed-code`, `--seed-pipeline`, `--output`,
  `--temperature`, `--timeout` options.

#### Dashboard — Two Tabs (Phase 7)
- **`cambrian/dashboard.py`** — Rebuilt with `st.tabs(["Evolve", "Forge"])`.
  Evolve tab: fitness trajectory, generation slider, sortable population table,
  best genome viewer, landscape heatmap.  Forge tab: code viewer (test-case
  grid, Python export) and pipeline viewer (step expanders, JSON export).

#### Exports
- `cambrian.__init__`: exposed all Forge mode classes (`CodeGenome`, `CodeAgent`,
  `CodeMutator`, `CodeGenomeEvaluator`, `CodeEvolutionEngine`, `PipelineStep`,
  `Pipeline`, `PipelineMutator`, `PipelineEvaluator`, `PipelineEvolutionEngine`).

### Tests
- 93 new tests across `test_code_genome.py` (47), `test_pipeline.py` (46).
- Dashboard tests updated for 2-tab `st.tabs()` mock.

---

## [0.8.0] — Round 8

### Added

#### Competition & Meta-Learning
- **Self-play** (`cambrian/self_play.py`): Head-to-head agent competition as
  an additional selection pressure.  `SelfPlayEvaluator` runs two agents on the
  same task; the winner receives a configurable fitness bonus, the loser a penalty.
  `TournamentRecord` tracks win/loss/draw across a full round-robin.
  `run_tournament(population, evaluator, task)` runs all agent pairs and returns
  a ranked record.

- **Meta-evolution** (`cambrian/meta_evolution.py`): MAML-inspired outer loop
  that evolves hyperparameters alongside genomes.  `HyperParams` bundles
  `mutation_rate`, `crossover_rate`, `temperature`, `tournament_k`, `elite_ratio`
  with `perturb()`, `clamp()`, `to_dict()`/`from_dict()`.
  `MetaEvolutionEngine` tries `n_candidates` perturbed HP configs every
  `meta_interval` generations and keeps the best-performing configuration.

- **World model** (`cambrian/world_model.py`): Per-agent predictive model inspired
  by Dyna-Q / Ha & Schmidhuber 2018.  Each agent accumulates experience in a
  fixed-capacity buffer; `WorldModel.predict(task)` uses weighted nearest-neighbour
  (word-level Jaccard) to predict performance.
  `WorldModelEvaluator` blends raw fitness with prediction accuracy — rewarding
  agents that understand their own capabilities.

#### Observability
- **Structured NDJSON logging** (`cambrian/utils/logging.py`): `JSONLogger`
  writes one JSON object per generation (timestamp, run_id, fitness stats,
  best agent ID, prompt length, arbitrary extras).  Context-manager-safe,
  flush-on-write, append mode.  `load_json_log(path)` reads NDJSON files
  skipping malformed lines.

#### CLI
- **`cambrian compare RUN1 RUN2`**: Compare two NDJSON evolution run logs on
  any metric (default `best_fitness`).  Supports `--format text|json`.
  Reports per-run stats, winner, and fitness delta.

#### Documentation
- **`docs/QUICKSTART.md`**: End-to-end tutorial from install to custom evaluator,
  Python API examples, CLI cheat-sheet, troubleshooting.
- **`docs/FAQ.md`**: Common questions on architecture, LLM support, cost,
  custom evaluators, Lamarck/stigmergy/NSGA-II/meta-evolution/world-model,
  deployment, and contributing.
- **`docs/BENCHMARKS.md`**: HumanEval, SWE-bench Lite, and MMLU placeholder
  tables with setup descriptions and contributing guidelines.

#### Tests
- **`tests/test_integration_real.py`** (9 tests): Full 3-generation, 5-agent
  evolution cycle with a deterministic keyword-based mock backend.  Asserts
  elitism guarantee (fitness never decreases), population size stability,
  callback invocation, memory population, and seed diversity.
- **`tests/test_round8.py`** (70 tests): Unit tests for all Round 8 features —
  `SelfPlayResult`, `SelfPlayEvaluator`, `run_tournament`, `HyperParams`,
  `MetaEvolutionEngine`, `WorldModel`, `world_model_fitness`,
  `WorldModelEvaluator`, `JSONLogger`, `load_json_log`, `cambrian compare` CLI.

### Fixed
- **Security — sandbox env leak** (`cambrian/utils/sandbox.py`): The subprocess
  sandbox previously forwarded `os.environ.copy()` to untrusted code, exposing
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and other secrets.  Fixed by stripping
  all non-whitelisted environment variables before spawning the subprocess.
  Only `PATH`, `PYTHONPATH`, `SYSTEMROOT`, `TEMP`, `TMP`, `TMPDIR`, `HOME`,
  `LANG`, `LC_ALL`, `LC_CTYPE` are forwarded.

### Internal
- Added `cambrian/__main__.py` — `python -m cambrian` now works.
- Renamed `stats.py::ParetoFront` → `ParetoAnalyzer` to eliminate naming
  collision with `pareto.py::ParetoFront` (NSGA-II).  Backwards-compatible
  alias `ParetoFront = ParetoAnalyzer` retained.
- Updated `__init__.py` to export all 14 primary public symbols.
- Added `build>=1.0` to `[project.optional-dependencies.dev]` in `pyproject.toml`.

---

## [0.7.0] — Round 7

### Added

#### Reasoning
- **DiffCoT** (`cambrian/diffcot.py`): Diffusion-inspired iterative chain-of-thought denoising.
  `DiffCoTConfig` controls `n_steps`, `noise_level`, and `temperature_schedule` (cosine/linear/constant).
  `DiffCoTReasoner` runs the denoising loop; `DiffCoTEvaluator` wraps any base evaluator.
  `make_diffcot_evaluator(base, backend, n_steps)` factory for quick setup.

- **Causal reasoning** (`cambrian/causal.py`): Explicit cause-effect representation in agent strategies.
  `CausalEdge` stores cause, effect, strength, and confidence.
  `CausalGraph` parses IF/THEN, arrow notation, and natural language causal relations.
  `CausalStrategyExtractor` uses an LLM to extract causal graphs from strategy text.
  `CausalMutator` evolves causal graphs alongside genomes.
  `inject_causal_context(genome, graph)` appends a causal context block to the system prompt.

- **Tool creation** (`cambrian/tool_creation.py`): Agents invent new CLI tools during evolution.
  `ToolSpec` dataclass added to `Genome.tool_specs` in `agent.py`.
  `ToolInventor` prompts an LLM to invent tool specs, validates names, dry-runs commands.
  `ToolPopulationRegistry` — shared cross-population tool registry with deduplication and top-N ranking.

#### CLI
- **`cambrian snapshot`**: Show population state at a specific generation from a lineage file.
  Supports `--format text|json` and `--top N` to limit output.

#### Documentation & Examples
- **`SECURITY.md`**: Comprehensive security guidelines covering subprocess sandboxing, API key
  management, prompt injection mitigations, ToolInventor safe deployment, and a production checklist.
- **`examples/evolve_researcher.py`**: Advanced example demonstrating LLM-judge evaluator with a
  domain-specific rubric, Lamarckian adapter, stigmergy, epigenetics, and multi-objective Pareto
  analysis.

#### README
- Complete rewrite with all 8 feature categories, all 9 CLI commands, and Python API examples
  for every major subsystem.

### Internal
- ruff lint: removed unused `Any`, `re`, `field` imports across 8 files.

---

## [0.6.0] — Round 6

### Added

#### Core Evolution
- **`SpeculativeMutator`** (`cambrian/speculative.py`): Generate K mutations concurrently via
  `asyncio.gather`, keep the best. `speculate(agent, task, mutator, evaluator, k)` async helper.
  `SpeculativeResult` reports `best_fitness`, `mean_fitness`, and `improvement_over_mean`.

- **Archipelago / Island model** (`cambrian/archipelago.py`): N isolated populations evolving
  independently with periodic migration. Supports `ring`, `all-to-all`, and `random` topologies.
  `Archipelago.evolve()` orchestrates full island evolution with configurable migration rate.

#### Multi-Objective
- **NSGA-II** (`cambrian/pareto.py`): Full non-dominated sorting + crowding-distance selection.
  `ObjectiveVector`, `ParetoFront`, `fast_non_dominated_sort`, `crowding_distance`, `nsga2_select`.
  Built-in objectives: `fitness_objective`, `brevity_objective`, `attach_diversity_scores`.

#### Reward Shaping
- **`cambrian/reward_shaping.py`**: Composable reward shaping pipeline.
  `ClipShaper`, `NormalisationShaper` (z-score + min-max, sliding window),
  `PotentialShaper` (Ng 1999 potential-based), `RankShaper`, `CuriosityShaper` (trigram novelty).
  `build_shaped_evaluator(base, "clip+normalise+curiosity")` factory.

#### Export & Deployment
- **`cambrian/export.py`**: Four export formats for evolved agents.
  `export_genome_json` / `load_genome_json`, `export_standalone` (self-contained Python script),
  `export_mcp` (MCP server stub with manifest + handler), `export_api` (FastAPI REST application).

#### CLI
- **`cambrian run`**: Load an evolved agent and run it on a task.
  `--format text|json` for machine-readable output.

#### Docs
- **`docs/METHODOLOGY.md`**: Academic references for all 17 techniques, from tournament selection
  to potential-based reward shaping and stigmergic pheromone traces.

### Tests
- 97 tests added in `tests/test_round6.py` covering all Round 6 features.

---

## [0.5.0] — Round 5

### Added

#### Agent-to-Agent (A2A)
- **`cambrian/a2a.py`**: Full A2A protocol.
  `AgentCard` capability descriptor with domain confidence scores.
  `A2AMessage` structured request/response envelope.
  `AgentNetwork` — register agents, route by domain, delegate, broadcast, chain, majority-vote.

#### Tools
- **`cambrian/cli_tools.py`**: Wrap shell commands as LLM-callable tools.
  `CLITool` — `{input}` template substitution, configurable `timeout`, `shell=False` safe default.
  `CLIToolkit` — named collection with text-markup protocol `[TOOL: name | input]`.

#### CLI
- **`cambrian analyze`**: Deep trajectory, diversity, and lineage analysis with `--top N`.
- **`cambrian dashboard`**: Streamlit live evolution dashboard with `--port` and `--log-file`.
- **`cambrian distill`**: Pretty-print a saved genome.
- **`cambrian distill-agent`**: Compress genome for a smaller model with `--max-tokens`.

#### Evaluators
- **`VarianceAwareEvaluator`** (`cambrian/evaluators/variance_aware.py`): Multi-trial evaluation
  with variance penalty to suppress reward hacking.
- **`BaldwinEvaluator`** (`cambrian/evaluators/baldwin.py`): Multi-trial evaluation with
  in-context learning bonus for agents that improve across trials.

#### Bio-Inspired
- **`CoEvolutionEngine`** (`cambrian/coevolution.py`): Adversarial generator/adversary arms race.
- **`CurriculumScheduler`** (`cambrian/curriculum.py`): Task difficulty progression with thresholds.
- **`ConstitutionalWrapper`** (`cambrian/constitutional.py`): Critique-revise safety cycles.
- **`MCTSSelector`** (`cambrian/mcts.py`): UCB1-guided tree search over the mutation tree.

### Tests
- Tests added for all Round 5 features in `tests/test_round5.py`.

---

## [0.4.0] — Round 4

### Added
- **`EpigeneticLayer`** and `EpigenomicContext`: Context-dependent system prompt annotations.
- **`ImmuneMemory`**: SHA-256 fingerprinting to suppress re-evaluation of barren regions.
- **`LamarckianAdapter`**: Capture successful examples into genome's few-shot examples.
- **`EvolutionaryMemory`**: NetworkX lineage graph — ancestry tracing, JSON export/import.
- **`StigmergyTrace`**: Pheromone traces that bias LLM mutation toward high-scoring regions.
- **`DiversityTracker`**: Per-generation entropy, temperature, and prompt std stats.
- **`FitnessLandscape`**: 2D fitness grid over temperature x token-length.
- **`LLMJudgeEvaluator`**: Judge LLM scores agent responses against a custom rubric.
- **`CompositeEvaluator`**: Weighted average of multiple evaluators.
- **`AnthropicBackend`**: Native Anthropic Claude backend via the `anthropic` SDK.
- **`GeminiBackend`**: Google Gemini backend via `google-genai`.

---

## [0.3.0] — Round 3

### Added
- **`EvolutionEngine`**: Full generational loop with tournament selection, elitism, crossover,
  mutation.
- **`LLMMutator`**: LLM reads genome + fitness and writes an improved version.
- **`OpenAICompatBackend`**: OpenAI-compatible backend (OpenAI, Ollama, Groq, vLLM).
- **`CodeEvaluator`**: Subprocess sandbox for code evaluation with partial-credit scoring.
- **CLI**: `cambrian evolve`, `cambrian stats`, `cambrian version`.

---

## [0.1.0] — Initial Release

### Added
- `Genome` and `Agent` dataclasses.
- `Evaluator` ABC.
- `LLMBackend` ABC.
- Minimal `EvolutionEngine` skeleton.
