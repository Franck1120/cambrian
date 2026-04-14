# Changelog

All notable changes to Cambrian are documented here.

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
