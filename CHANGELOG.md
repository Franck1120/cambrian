# Changelog

All notable changes to Cambrian are documented here.

---

## [0.9.0] — Round 9

### Added

#### Forge Mode — Code Evolution
- **CodeGenome** (`cambrian/code_genome.py`): Executable Python code as an
  evolvable genome.  Tracks `version`, `parent_id`, `loc()` (non-blank lines).
  `CodeEvaluator` runs each test case in a subprocess sandbox and scores by
  weighted pass rate + LOC efficiency + runtime bonuses.
  `CodeMutator` asks the LLM to rewrite code (seed or improve), strips markdown
  fences, falls back gracefully on errors.
  `CodeEvolutionEngine` runs the generational loop with elitism, early-exit on
  perfect fitness, and optional `on_generation` callback.

#### Forge Mode — Pipeline Composition
- **Pipeline** (`cambrian/pipeline.py`): Ordered list of `PipelineStep` objects
  as an evolvable genome.  `PipelineRunner` chains steps sequentially (output of
  step N is input to step N+1).  `PipelineEvaluator` scores by user-provided
  `score_fn(output, task) -> float`.  `PipelineMutator` adds/removes/reorders
  steps via LLM, supports crossover.  `PipelineEvolutionEngine` runs the loop.
- **CLI**: `cambrian forge TASK --mode code|pipeline` entry point.

#### Adaptive Pressures
- **Dream Phase** (`cambrian/dream.py`): Recombines past experiences via LLM
  into hybrid synthetic scenarios.  Agents evaluated on dreams get a blended
  fitness signal (configurable `blend_weight`).  Mimics memory consolidation
  during sleep.
- **Quorum Sensing** (`cambrian/quorum.py`): Monitors Shannon entropy of
  population fitness distribution.  Auto-adjusts `mutation_rate` and `elite_n`
  toward a target entropy.  `stagnation_detected()` and `should_inject_diversity()`
  helpers enable reactive control loops.

#### Ensemble & Reflexion
- **Mixture of Agents** (`cambrian/moa.py`): N agents answer independently; an
  aggregator LLM synthesises a final answer.  `MoAResult` carries
  `individual_answers` + `final_answer`.  Robust to single-agent failures.
- **Quantum Tunneling** (`cambrian/moa.py`): `QuantumTunneler` replaces agents
  with fresh random genomes at a configurable `tunnel_prob`.  Elite protection,
  event history, `summary()`.
- **Reflexion** (`cambrian/reflexion.py`): Generate → critique → revise cycle
  (Shinn et al. 2023).  `ReflexionAgent` supports N rounds, early-exit on
  "EXCELLENT" critique, custom reflection backend.  `ReflexionEvaluator` wraps
  any evaluator and applies Reflexion before scoring via a proxy agent.

#### Dashboard
- **Streamlit dashboard** (`cambrian/dashboard.py`): Fully rewritten with three
  tabs — Evolve (fitness chart, population table, best genome viewer), Forge
  (code viewer + test-case pass/fail grid, pipeline step viewer), Logs (live
  NDJSON stream).  Supports both NDJSON and legacy JSON array log formats.

### Tests
- 776 tests total (was 480): +296 new tests across 4 new test files.
- `test_code_genome.py` (38 tests), `test_pipeline.py` (35 tests),
  `test_dream_quorum.py` (38 tests), `test_moa_reflexion.py` (39 tests),
  `test_round9_integration.py` (24 tests).

### Quality
- ruff clean, mypy clean on all 54 source files.

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
