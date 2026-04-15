# Cambrian API Reference

> Version 1.0.1 · Auto-generated from source · 130 public symbols

This document covers every class and function exported from the `cambrian` package.
Import any symbol directly from the top-level package:

```python
from cambrian import EvolutionEngine, Genome, Agent, LLMMutator
```

---

## Table of Contents

1. [Core](#core)
2. [Evaluators](#evaluators)
3. [Backends](#backends)
4. [Mutators](#mutators)
5. [Bio-Inspired](#bio-inspired)
6. [Reasoning Modules](#reasoning-modules)
7. [Competition & Meta-Learning](#competition--meta-learning)
8. [World Model](#world-model)
9. [Forge Mode — Code Evolution](#forge-mode--code-evolution)
10. [Forge Mode — Pipeline Evolution](#forge-mode--pipeline-evolution)
11. [Dream, Quorum, MoA, Reflexion, Quantum Tunneling](#dream-quorum-moa-reflexion-quantum-tunneling)
12. [Tier 3 — Advanced Bio-Inspired](#tier-3--advanced-bio-inspired)
13. [Tier 4 — Cutting-Edge Bio-Inspired](#tier-4--cutting-edge-bio-inspired)
14. [Tier 5 — Metamorphosis, Ecosystem, Fractal](#tier-5--metamorphosis-ecosystem-fractal)
15. [Tier 5 — DPO & Safeguards](#tier-5--dpo--safeguards)
16. [Utilities](#utilities)

---

## Core

### `Genome`

`cambrian.agent.Genome`

The evolvable unit. Carries everything the LLM needs to behave as an agent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str` | `""` | The agent's system prompt — the primary evolving artifact |
| `tools` | `list[str]` | `[]` | Tool names available to the agent |
| `strategy` | `str` | `"step-by-step"` | Reasoning strategy hint injected at runtime |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `model` | `str` | `"gpt-4o-mini"` | LLM model identifier |
| `few_shot_examples` | `list[dict]` | `[]` | In-context demonstrations (Lamarckian inheritance) |
| `tool_specs` | `list[ToolSpec]` | `[]` | Invented tool specifications |

**Key methods:**
- `to_dict() -> dict` — serialise to JSON-compatible dict
- `from_dict(d: dict) -> Genome` — deserialise (classmethod)

---

### `Agent`

`cambrian.agent.Agent`

Wraps a `Genome` with runtime state: fitness, generation, unique ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genome` | `Genome` | required | The agent's evolvable genome |

**Key attributes:**
- `id: str` — UUID-based unique identifier
- `fitness: Optional[float]` — current fitness score (None until evaluated)
- `generation: int` — generation in which this agent was born
- `run(task: str, backend: LLMBackend) -> str` — generate a response for `task`

---

### `EvolutionEngine`

`cambrian.evolution.EvolutionEngine`

Full generational evolutionary loop with tournament selection and elitism.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator` | `Evaluator` | required | Fitness function |
| `mutator` | `LLMMutator` | required | Mutation operator |
| `backend` | `LLMBackend` | required | LLM backend for agent runs |
| `population_size` | `int` | `10` | Agents per generation |
| `mutation_rate` | `float` | `0.8` | Probability of mutation per slot |
| `crossover_rate` | `float` | `0.2` | Probability of crossover per slot |
| `elite_ratio` | `float` | `0.1` | Fraction of top agents preserved unchanged |
| `tournament_k` | `int` | `3` | Tournament selection pool size |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

**Key methods:**
- `evolve(seed_genomes, task, n_generations, on_generation=None) -> Agent` — run evolution; returns best agent
- `population: list[Agent]` — current population (after last generation)

---

### `Evaluator` (ABC)

`cambrian.evaluator.Evaluator`

Abstract base class for all evaluators.

```python
class MyEvaluator(Evaluator):
    def evaluate(self, agent: Agent, task: str) -> float:
        ...  # return score in [0.0, 1.0]
```

---

### `LLMMutator`

`cambrian.mutator.LLMMutator`

Uses an LLM to rewrite agent genomes toward better performance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | LLM backend for mutation |
| `mutation_temperature` | `float` | `0.9` | Sampling temperature for creative mutation |
| `max_prompt_tokens` | `int` | `2000` | Truncate long genomes before sending |

**Key methods:**
- `mutate(agent: Agent, task: str) -> Agent` — returns a new Agent with mutated genome
- `crossover(parent_a: Agent, parent_b: Agent, task: str) -> Agent` — combine two genomes

---

### `EvolutionaryMemory`

`cambrian.memory.EvolutionaryMemory`

NetworkX lineage graph — trace ancestry, store pheromone traces.

**Key methods:**
- `add_agent(agent: Agent, parent_id: str | None = None) -> None`
- `add_trace(agent_id: str, content: str, score: float) -> None`
- `get_top_traces(n: int = 5) -> list[StigmergyTrace]`
- `ancestors(agent_id: str) -> list[str]`
- `export_json(path: str) -> None` / `load_json(path: str) -> None`

---

## Evaluators

### `CodeEvaluator`

`cambrian.evaluators.code.CodeEvaluator`

Runs agent-generated code in a subprocess sandbox with partial-credit scoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_output` | `str` | required | Expected stdout |
| `timeout` | `float` | `10.0` | Subprocess timeout in seconds |

Scoring: empty→0.0, error→0.1, partial→0.1+0.9×(matches/total), perfect→1.0

---

### `LLMJudgeEvaluator`

`cambrian.evaluators.llm_judge.LLMJudgeEvaluator`

Uses a judge LLM to score agent responses against a rubric.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Judge LLM |
| `rubric` | `str` | required | Scoring rubric |
| `scale` | `int` | `10` | Max score (normalised to [0,1]) |

---

### `CompositeEvaluator`

`cambrian.evaluators.composite.CompositeEvaluator`

Weighted average across multiple evaluators.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluators` | `list[Evaluator]` | required | Sub-evaluators |
| `weights` | `list[float] \| None` | `None` | Per-evaluator weights (uniform if None) |
| `aggregate` | `str` | `"mean"` | `"mean"` or `"min"` or `"max"` |

---

### `VarianceAwareEvaluator`

`cambrian.evaluators.variance_aware.VarianceAwareEvaluator`

Multi-trial evaluation with variance penalty to suppress reward hacking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_evaluator` | `Evaluator` | required | Inner evaluator |
| `n_trials` | `int` | `3` | Evaluation trials |
| `penalty` | `float` | `0.5` | Variance penalty weight |

---

### `BaldwinEvaluator`

`cambrian.evaluators.baldwin.BaldwinEvaluator`

Multi-trial evaluation with in-context learning bonus.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_evaluator` | `Evaluator` | required | Inner evaluator |
| `n_trials` | `int` | `3` | Trials (feedback passed forward) |
| `baldwin_bonus` | `float` | `0.2` | Bonus for improvement across trials |

---

### `DiffCoTEvaluator`

`cambrian.diffcot.DiffCoTEvaluator`

Wraps any evaluator with diffusion-inspired iterative chain-of-thought denoising before scoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_evaluator` | `Evaluator` | required | Inner evaluator |
| `backend` | `LLMBackend` | required | LLM for denoising steps |
| `n_steps` | `int` | `3` | Denoising iterations |

---

## Backends

### `OpenAICompatBackend`

`cambrian.backends.openai_compat.OpenAICompatBackend`

OpenAI-compatible HTTP backend (OpenAI, Ollama, Groq, vLLM, LM Studio).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | Model identifier |
| `base_url` | `str` | OpenAI API | API base URL |
| `api_key` | `str` | required | API key |
| `timeout` | `int` | `60` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Retry attempts on transient errors |

---

### `AnthropicBackend`

`cambrian.backends.anthropic.AnthropicBackend`

Native Anthropic Claude backend via the `anthropic` SDK.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-3-5-sonnet-20241022"` | Claude model ID |
| `api_key` | `str \| None` | env `ANTHROPIC_API_KEY` | API key |

---

### `GeminiBackend`

`cambrian.backends.gemini.GeminiBackend`

Google Gemini backend via `google-genai`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gemini-1.5-flash"` | Gemini model ID |
| `api_key` | `str \| None` | env `GOOGLE_API_KEY` | API key |

---

## Mutators

### `SpeculativeMutator`

`cambrian.speculative.SpeculativeMutator`

Generates K mutations in parallel via `asyncio`, keeps the best.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_mutator` | `LLMMutator` | required | Inner mutator |
| `evaluator` | `Evaluator` | required | Evaluator for ranking candidates |
| `k` | `int` | `4` | Number of speculative candidates |

**Key methods:**
- `mutate(agent: Agent, task: str) -> Agent`
- `speculate(agent, task) -> SpeculativeResult` (async)

---

### `TabuMutator`

`cambrian.tabu.TabuMutator`

Wraps any mutator with a tabu list preventing genome revisitation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_mutator` | `LLMMutator` | required | Inner mutator |
| `tabu_list` | `TabuList` | required | Tabu list instance |
| `max_retries` | `int` | `5` | Retry attempts before accepting tabu move |

**Key properties:** `tabu_hit_rate: float`

---

### `TabuList`

`cambrian.tabu.TabuList`

FIFO bi-gram fingerprint list (SHA-256 hex[:16]) preventing revisitation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size` | `int` | `20` | Maximum tabu entries |

**Key methods:** `add(agent)`, `is_tabu(agent) -> bool`, `clear()`

---

### `CausalMutator`

`cambrian.causal.CausalMutator`

Evolves causal graphs alongside genomes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | LLM backend |
| `base_mutator` | `LLMMutator` | required | Inner genome mutator |

---

## Bio-Inspired

### `LamarckianAdapter`

`cambrian.lamarck.LamarckianAdapter`

Captures successful examples into genome few-shot examples.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capture_threshold` | `float` | `0.7` | Fitness threshold to capture example |
| `max_examples` | `int` | `5` | Maximum examples per genome |

**Key methods:** `adapt(agent: Agent, task: str, response: str, score: float) -> Agent`

---

### `Archipelago`

`cambrian.archipelago.Archipelago`

Multi-island evolution with configurable migration topology.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_islands` | `int` | `4` | Number of independent islands |
| `topology` | `str` | `"ring"` | `"ring"`, `"all_to_all"`, or `"random"` |
| `migration_rate` | `float` | `0.1` | Fraction of population migrated per interval |
| `migration_interval` | `int` | `5` | Generations between migrations |

---

### `MCTSSelector`

`cambrian.mcts.MCTSSelector`

UCB1-guided tree search over the mutation tree.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration_weight` | `float` | `1.414` | UCB1 exploration constant C |

**Key methods:** `select(population: list[Agent]) -> Agent`, `backpropagate(agent_id, reward)`

---

### `CoEvolutionEngine`

`cambrian.coevolution.CoEvolutionEngine`

Adversarial generator/adversary arms race.

**Key methods:** `evolve(seed_genome, task, n_generations) -> tuple[Agent, Agent]`

---

### `CurriculumScheduler`

`cambrian.curriculum.CurriculumScheduler`

Task difficulty progression with fitness-based stage promotion.

**Key methods:** `advance(fitness_values: list[float]) -> bool`, `current_task() -> str`

---

### `ConstitutionalWrapper`

`cambrian.constitutional.ConstitutionalWrapper`

Critique-revise cycles enforcing behavioural constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Critique + revision LLM |
| `principles` | `list[str]` | required | Constitutional principles |

---

### `EpigeneticLayer`

`cambrian.epigenetics.EpigeneticLayer`

Context-dependent system prompt annotations at runtime (non-heritable).

**Key methods:** `apply(agent: Agent, generation: int, population_fitness: list[float]) -> Agent`

---

### `ImmuneMemory`

`cambrian.immune.ImmuneMemory`

SHA-256 fingerprinting — suppresses re-evaluation of barren genome regions.

**Key methods:** `is_suppressed(agent: Agent) -> bool`, `record(agent: Agent, score: float)`

---

### `StigmergyTrace`

`cambrian.memory.StigmergyTrace`

Pheromone-like trace encoding high-scoring prompt patterns.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Source agent |
| `content` | `str` | Genome fragment |
| `score` | `float` | Fitness at time of deposit |
| `timestamp` | `float` | Unix timestamp |

---

## Reasoning Modules

### `DiffCoTReasoner`

`cambrian.diffcot.DiffCoTReasoner`

Diffusion-inspired iterative chain-of-thought denoising.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | LLM backend |
| `config` | `DiffCoTConfig` | default | n_steps, noise_level, temperature_schedule |

**Key methods:** `reason(system_prompt: str, task: str) -> str`

---

### `CausalGraph`

`cambrian.causal.CausalGraph`

Explicit cause-effect relationship graph for agent strategies.

**Key methods:**
- `parse(text: str) -> None` — extract IF/THEN and arrow-notation edges
- `edges: list[CausalEdge]`
- `to_context() -> str` — formatted context block for injection

### `inject_causal_context(genome: Genome, graph: CausalGraph) -> Genome`

Appends causal context to a genome's system prompt.

---

### `ToolInventor`

`cambrian.tool_creation.ToolInventor`

LLM invents new CLI tool specifications during evolution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Invention LLM |
| `dry_run_timeout` | `float` | `5.0` | Timeout for tool dry-run validation |

**Key methods:** `invent(task: str) -> ToolSpec | None`

---

### `ToolPopulationRegistry`

`cambrian.tool_creation.ToolPopulationRegistry`

Shared cross-population registry with deduplication and fitness-weighted ranking.

**Key methods:** `register(spec: ToolSpec, fitness: float)`, `top_n(k: int) -> list[ToolSpec]`

---

## Competition & Meta-Learning

### `SelfPlayEvaluator`

`cambrian.self_play.SelfPlayEvaluator`

Head-to-head agent competition with win/loss fitness adjustments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_evaluator` | `Evaluator` | required | Scoring evaluator |
| `win_bonus` | `float` | `0.1` | Fitness bonus for winner |
| `loss_penalty` | `float` | `0.05` | Fitness penalty for loser |

**Key methods:**
- `compete(agent_a, agent_b, task, score_a=None, score_b=None) -> SelfPlayResult`
- `evaluate(agent, task) -> float`

### `run_tournament(population, evaluator, task) -> TournamentRecord`

Full round-robin tournament across all agent pairs.

---

### `MetaEvolutionEngine`

`cambrian.meta_evolution.MetaEvolutionEngine`

MAML-inspired outer loop: evolves hyperparameters alongside genomes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inner_engine` | `EvolutionEngine` | required | Inner evolutionary engine |
| `n_candidates` | `int` | `3` | HP candidate configs to try per meta-step |
| `meta_interval` | `int` | `5` | Generations between meta-updates |

### `HyperParams`

Mutable hyperparameter bundle.

| Field | Type | Default |
|-------|------|---------|
| `mutation_rate` | `float` | `0.8` |
| `crossover_rate` | `float` | `0.2` |
| `temperature` | `float` | `0.7` |
| `tournament_k` | `int` | `3` |
| `elite_ratio` | `float` | `0.1` |

**Key methods:** `perturb(scale=0.05, rng=None) -> HyperParams`, `to_dict()`, `from_dict()`

---

## World Model

### `WorldModel`

`cambrian.world_model.WorldModel`

Per-agent experience buffer with weighted nearest-neighbour prediction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_size` | `int` | `20` | Maximum experience entries |
| `default_score` | `float` | `0.5` | Prediction before any experience |
| `decay` | `float` | `0.95` | Weight decay per experience entry |

**Key methods:**
- `update(task: str, score: float) -> None`
- `predict(task: str) -> WorldModelPrediction`

### `WorldModelPrediction`

| Field | Type | Description |
|-------|------|-------------|
| `predicted_score` | `float` | Estimated performance |
| `confidence` | `float` | Prediction confidence (0–1) |
| `n_similar` | `int` | Number of similar experiences used |

### `WorldModelEvaluator`

Wraps any evaluator; rewards agents for accurate self-prediction.

---

## Forge Mode — Code Evolution

### `CodeGenome`

`cambrian.code_genome.CodeGenome`

Executable Python code as the evolvable artifact.

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Python source code |
| `description` | `str` | Natural language description |
| `version` | `int` | Generation counter |

### `CodeEvolutionEngine`

Full code-evolution loop: evaluate → mutate → crossover.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | LLM for code mutation |
| `population_size` | `int` | `6` | Code agents per generation |
| `timeout` | `float` | `10.0` | Sandbox execution timeout |

**Key methods:**
- `evolve(seed, task, test_cases, n_generations) -> CodeAgent`

### `CodeEvaluator` (as `CodeGenomeEvaluator`)

Scores code agents against test cases. Partial credit: 0.1+0.9×(k/N) for k/N passing.

---

## Forge Mode — Pipeline Evolution

### `Pipeline`

`cambrian.pipeline.Pipeline`

Ordered sequence of `PipelineStep` objects with version tracking.

### `PipelineStep`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Step identifier |
| `system_prompt` | `str` | Step-specific prompt |
| `role` | `str` | Agent role for this step |
| `temperature` | `float` | Sampling temperature |

### `PipelineEvolutionEngine`

Evolves full pipeline architectures. LLM adds/removes/reorders steps.

---

## Dream, Quorum, MoA, Reflexion, Quantum Tunneling

### `DreamPhase`

`cambrian.dream.DreamPhase`

GraphRAG-style offline recombination of high-fitness ancestor genomes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Synthesis LLM |
| `memory` | `EvolutionaryMemory` | required | Lineage graph |
| `interval` | `int` | `5` | Generations between dream events |
| `top_n` | `int` | `5` | Ancestors to recombine |

**Key methods:** `should_dream(generation: int) -> bool`, `dream(task, n_offspring) -> list[Agent]`

---

### `QuorumSensor`

`cambrian.quorum.QuorumSensor`

Shannon entropy of fitness distribution auto-regulates mutation rate.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `low_entropy_threshold` | `float` | `1.0` | Below → boost mutation |
| `high_entropy_threshold` | `float` | `2.5` | Above → decay mutation |
| `boost_factor` | `float` | `1.3` | Rate multiplier when low diversity |
| `decay_factor` | `float` | `0.85` | Rate multiplier when high diversity |

**Key methods:** `update(scores, current_rate) -> float`, `compute_entropy(scores) -> float`

---

### `MixtureOfAgents`

`cambrian.moa.MixtureOfAgents`

Runs N agent specialisations on the same task, aggregates via meta-LLM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[Agent]` | required | Base agent pool |
| `aggregator_backend` | `LLMBackend` | required | Synthesis LLM |

---

### `QuantumTunneler`

`cambrian.moa.QuantumTunneler`

Stochastic large-jump mutation that escapes local optima.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tunnel_prob` | `float` | `0.1` | Probability of tunneling per agent |

**Key methods:** `tunnel(agent, backend) -> Agent`, `tunnel_all(population, backend) -> list[Agent]`

---

### `ReflexionEvaluator`

`cambrian.reflexion.ReflexionEvaluator`

Generate → reflect → revise loop for n_reflections cycles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Critique + revision LLM |
| `n_reflections` | `int` | `2` | Reflection cycles |
| `reflect_temperature` | `float` | `0.2` | Temperature for critique |
| `revise_temperature` | `float` | `0.5` | Temperature for revision |

**Key methods:** `evaluate(agent, task) -> tuple[str, float]`

---

## Tier 3 — Advanced Bio-Inspired

### `SymbioticFuser`

`cambrian.symbiosis.SymbioticFuser`

LLM-guided endosymbiosis: merges genomes of compatible agents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Fusion LLM |
| `fitness_threshold` | `float` | `0.6` | Both agents must exceed this |
| `min_distance` | `float` | `0.2` | Minimum word-overlap distance |

**Key methods:**
- `fuse(host, donor, task) -> Agent | None`
- `fuse_best_pair(population, task) -> Agent | None`

---

### `HormesisAdapter`

`cambrian.hormesis.HormesisAdapter`

Graduated stress response: mild/moderate/severe stimulation by fitness gap.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Re-prompt LLM |
| `stress_threshold` | `float` | `0.5` | Fitness below which stress is applied |

**Key methods:**
- `stress_level(agent) -> str` — `"none"`, `"mild"`, `"moderate"`, `"severe"`
- `stimulate(agent, task) -> Agent | None`
- `stimulate_population(population, task) -> list[Agent]`

---

### `ApoptosisController`

`cambrian.apoptosis.ApoptosisController`

Programmed removal of chronically poor agents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stagnation_window` | `int` | `5` | Generations of no improvement before removal |
| `min_fitness` | `float` | `0.2` | Fitness floor; must be below AND stagnant |
| `grace_period` | `int` | `2` | Generations before any agent is eligible |

**Key methods:**
- `record(agent: Agent) -> None`
- `apply(population, best_agent) -> list[Agent]`

---

### `CatalysisEngine`

`cambrian.catalysis.CatalysisEngine`

Catalyst agent injects strategy context into target mutation; always restores.

**Key methods:** `catalyse(target, catalyst, task) -> float`

### `CatalystSelector`

Picks the best catalyst from a population by composite score.

**Key methods:** `select(population) -> Agent | None`

---

### `LLMCascade`

`cambrian.llm_cascade.LLMCascade`

Tiered routing across model sizes; escalates when confidence is below threshold.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `levels` | `list[CascadeLevel]` | required | Ordered backend levels |

**Key methods:** `query(system_prompt, task) -> tuple[str, int]`

### `CascadeLevel`

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `LLMBackend` | LLM at this level |
| `confidence_fn` | `Callable` | Confidence scorer (`hedging_confidence`, `length_confidence`) |
| `confidence_threshold` | `float` | Minimum confidence to stop escalating |

---

### `AgentEnsemble`

`cambrian.ensemble.AgentEnsemble`

Weighted majority vote across agent population.

**Key methods:** `query(task, correct_answer=None) -> str`

### `BoostingEnsemble(AgentEnsemble)`

AdaBoost-style weight updates based on per-query accuracy.

**Key methods:** `query(task, correct_answer) -> str`

---

### `GlossaloliaReasoner`

`cambrian.glossolalia.GlossaloliaReasoner`

Two-phase latent monologue → structured synthesis.

**Key methods:** `reason(system_prompt, task) -> GlossaloliaResult`

### `GlossaloliaResult`

| Field | Type | Description |
|-------|------|-------------|
| `monologue` | `str` | Free-form latent stream |
| `synthesis` | `str` | Structured final answer |

---

### `BestOfN`

`cambrian.inference_scaling.BestOfN`

Generate N candidates, return highest-scoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Generation LLM |
| `n` | `int` | `5` | Candidates to generate |
| `scorer` | `ScoringFn` | `length_scorer` | Candidate scoring function |

**Key methods:** `run(system, user, temperature=None) -> tuple[str, float]`

### `BeamSearch`

Tree search with beam_width × branching_factor candidates per step.

---

## Tier 4 — Cutting-Edge Bio-Inspired

### `TransferAdapter`

`cambrian.transfer.TransferAdapter`

Adapts a source genome to a new task domain at light/medium/heavy intensity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Adaptation LLM |
| `intensity` | `str` | `"medium"` | `"light"`, `"medium"`, or `"heavy"` |

**Key methods:** `adapt(source_agent, target_task) -> Agent`

### `TransferBank`

Domain-indexed registry of source agents.

**Key methods:** `register(agent, domain)`, `best_for(domain) -> Agent | None`

---

### `AnnealingSchedule`

`cambrian.annealing.AnnealingSchedule`

Cooling curve for simulated annealing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T_max` | `float` | `1.0` | Initial temperature |
| `T_min` | `float` | `0.01` | Final temperature |
| `n_steps` | `int` | `100` | Total cooling steps |
| `schedule_type` | `str` | `"cosine"` | `"linear"`, `"exponential"`, `"cosine"` |

### `AnnealingSelector`

Metropolis acceptance criterion.

**Key methods:**
- `step(current_fitness, candidate_fitness) -> bool`
- `acceptance_rate() -> float`

---

### `RedTeamSession`

`cambrian.red_team.RedTeamSession`

Adversarial attack generation + robustness scoring.

**Key methods:** `run(agent, task) -> RobustnessReport`

### `RobustnessReport`

| Field | Type | Description |
|-------|------|-------------|
| `normal_score` | `float` | Performance on standard task |
| `adversarial_score` | `float` | Performance under attack |
| `robustness_ratio` | `float` | `adversarial / normal` |

---

### `ZeitgeberClock`

`cambrian.zeitgeber.ZeitgeberClock`

Sinusoidal circadian oscillator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | `int` | `10` | Oscillation period in ticks |
| `amplitude` | `float` | `0.5` | Sinusoidal amplitude |
| `phase_offset` | `float` | `0.0` | Initial phase |

**Key methods:** `advance() -> None`, `exploration_factor() -> float`, `reset()`

### `ZeitgeberScheduler`

Maps clock to `mutation_rate` and `acceptance_threshold`.

**Key methods:** `tick() -> tuple[float, float]`

---

### `HGTransfer`

`cambrian.hgt.HGTransfer`

Horizontal Gene Transfer: extracts genome fragments (plasmids) from donors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_sentences` | `int` | `2` | Sentences to extract |
| `mode` | `str` | `"suffix"` | `"prefix"`, `"suffix"`, `"replace"` |
| `fitness_threshold` | `float` | `0.6` | Donor minimum fitness |

**Key methods:** `transfer(donor, recipient) -> Agent | None`

### `HGTPool`

Domain-tagged plasmid pool.

**Key methods:** `contribute(agent, domain)`, `draw(domain) -> HGTPlasmid | None`, `best_for(domain) -> HGTPlasmid | None`

---

### `TransgenerationalRegistry`

`cambrian.transgenerational.TransgenerationalRegistry`

Heritable epigenetic marks that decay per generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_generations` | `int` | `5` | Maximum mark lifetime |
| `strength_threshold` | `float` | `0.1` | Minimum strength to retain |
| `inherit_top_n` | `int` | `5` | Maximum marks inherited |

**Key methods:**
- `record_mark(agent, name, strength)`
- `inherit(parent, child) -> int`
- `apply_to_genome(agent) -> Genome`
- `get_marks(agent) -> list[EpigeneMark]`

---

### `ImmuneCortex`

`cambrian.immune_memory.ImmuneCortex`

B-cell (fast exact recall) + T-cell (adaptive seed) memory coordinator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `b_threshold` | `float` | `0.8` | Fitness threshold for B-cell storage |
| `t_threshold` | `float` | `0.5` | Fitness threshold for T-cell storage |
| `b_similarity` | `float` | `0.75` | Minimum Jaccard similarity for B-cell recall |
| `t_min_similarity` | `float` | `0.2` | Minimum similarity for T-cell recall |

**Key methods:**
- `record(agent, task) -> None`
- `recall(task) -> RecallResult`

### `RecallResult`

| Field | Type | Description |
|-------|------|-------------|
| `recalled` | `bool` | Whether a match was found |
| `agent` | `Agent \| None` | Recalled agent |
| `cell_type` | `str` | `"b_cell"` or `"t_cell"` |
| `similarity` | `float` | Jaccard similarity score |
| `source_task` | `str` | Task associated with recalled genome |

---

### `NeuromodulatorBank`

`cambrian.neuromodulation.NeuromodulatorBank`

Four bio-inspired modulators dynamically adjusting evolutionary hyperparameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_mutation_rate` | `float` | `0.3` | Starting mutation rate |
| `base_selection_pressure` | `float` | `0.5` | Starting selection pressure |
| `mr_range` | `float` | `0.2` | Max mutation rate deviation |
| `sp_range` | `float` | `0.2` | Max selection pressure deviation |

**Key methods:** `modulate(population, generation) -> NeuroState`, `reset()`

### `NeuroState`

| Field | Type | Description |
|-------|------|-------------|
| `mutation_rate` | `float` | Adjusted mutation rate |
| `selection_pressure` | `float` | Adjusted selection pressure |
| `dopamine` | `float` | Reward signal (fitness trend) |
| `serotonin` | `float` | Stability signal (diversity) |
| `acetylcholine` | `float` | Surprise signal (variance) |
| `noradrenaline` | `float` | Alertness signal (stagnation) |
| `generation` | `int` | Generation index |

---

## Tier 5 — Metamorphosis, Ecosystem, Fractal

### `MetamorphosisController`

`cambrian.metamorphosis.MetamorphosisController`

Discrete lifecycle management: LARVA → CHRYSALIS → IMAGO.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Restructuring LLM (chrysalis phase) |
| `larva_config` | `PhaseConfig \| None` | defaults | Larva phase config |
| `chrysalis_config` | `PhaseConfig \| None` | defaults | Chrysalis config |
| `imago_config` | `PhaseConfig \| None` | defaults | Imago config |

Default configs: LARVA(min_gen=3, thresh=0.4, mult=1.5), CHRYSALIS(min_gen=1, thresh=0.6, mult=0.0), IMAGO(mult=0.5)

**Key methods:**
- `current_phase(agent) -> MetamorphicPhase`
- `mutation_rate_multiplier(agent) -> float`
- `advance(agent, generation, fitness) -> MorphEvent | None`
- `metamorphose(agent, task) -> Agent` — LLM restructures genome (chrysalis)
- `apply_phase_pressure(genome, phase) -> Genome`
- `phase_distribution() -> dict[str, int]`
- `events: list[MorphEvent]`

### `MetamorphicPhase`

`Enum`: `LARVA = "larva"`, `CHRYSALIS = "chrysalis"`, `IMAGO = "imago"`

### `MetamorphicPopulation`

Manages a full population through metamorphic lifecycle.

**Key methods:**
- `register(agent) -> None`
- `tick(population, generation, task) -> list[MorphEvent]`

---

### `EcosystemInteraction`

`cambrian.ecosystem.EcosystemInteraction`

4-role ecological fitness dynamics (herbivore, predator, decomposer, parasite).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `EcosystemConfig \| None` | defaults | Interaction parameters |

**Key methods:**
- `assign_role(agent, role) -> None`
- `get_role(agent) -> EcologicalRole | None`
- `auto_assign(population) -> None` — fitness-rank-based role assignment
- `interact(population, task) -> list[EcosystemEvent]`
- `apply_events(events, population) -> None`
- `role_counts() -> dict[str, int]`
- `events: list[EcosystemEvent]`

### `EcologicalRole`

`Enum`: `HERBIVORE`, `PREDATOR`, `DECOMPOSER`, `PARASITE`

### `EcosystemEvaluator`

Wraps any evaluator with ecological fitness weighting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_evaluator` | `Evaluator` | required | Inner evaluator |
| `interaction` | `EcosystemInteraction` | required | Ecosystem context |
| `interaction_weight` | `float` | `0.2` | Ecological influence weight |

---

### `FractalEvolution`

`cambrian.fractal.FractalEvolution`

Recursive multi-scale evolutionary search across MACRO, MESO, MICRO granularities.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `LLMBackend` | required | Mutation LLM |
| `evaluator` | `Evaluator` | required | Fitness function |
| `macro_config` | `ScaleConfig \| None` | defaults | MACRO scale config |
| `meso_config` | `ScaleConfig \| None` | defaults | MESO scale config |
| `micro_config` | `ScaleConfig \| None` | defaults | MICRO scale config |

**Key methods:**
- `evolve(seed_genome, task, n_cycles=2) -> FractalResult`
- `results: list[FractalResult]`

### `FractalScale`

`IntEnum`: `MACRO = 0`, `MESO = 1`, `MICRO = 2`

### `ScaleConfig`

| Field | Type | Default (MACRO/MESO/MICRO) |
|-------|------|---------------------------|
| `scale` | `FractalScale` | required |
| `population_size` | `int` | `4` |
| `n_generations` | `int` | `3` |
| `mutation_temperature` | `float` | `0.7 / 0.5 / 0.3` |
| `fragment_size` | `int` | `500 / 100 / 20` |

### `FractalMutator`

| Method | Description |
|--------|-------------|
| `mutate_macro(genome, task)` | Rewrites overall strategy |
| `mutate_meso(genome, task)` | Rewrites one paragraph block |
| `mutate_micro(genome, task)` | Rewrites a short phrase |
| `mutate(genome, task, scale)` | Dispatches to correct scale |

---

## Tier 5 — DPO & Safeguards

### `DPOPair`

```python
from cambrian import DPOPair
```

Dataclass representing a **Direct Preference Optimisation** pair.

| Field | Type | Description |
|-------|------|-------------|
| `chosen` | `Agent` | The higher-fitness (preferred) agent |
| `rejected` | `Agent` | The lower-fitness (rejected) agent |
| `task` | `str` | Task description for context |
| `margin` | `float` | Fitness gap: `chosen.fitness - rejected.fitness` |

---

### `DPOSelector`

```python
from cambrian import DPOSelector

selector = DPOSelector(beta=0.1, pair_strategy="adjacent")  # or "random"
pairs   = selector.build_pairs(population, task)
reward  = selector.compute_dpo_reward(pair)    # → float ∈ [0, 1]
updated = selector.apply(population, task)     # mutates fitness in-place
```

Builds preference pairs from a ranked population and applies a margin-based
DPO reward. `pair_strategy="adjacent"` pairs consecutive ranks;
`"random"` pairs random agents.

**Key parameters:**
- `beta` — scale factor for preference reward (default `0.1`)
- `pair_strategy` — `"adjacent"` or `"random"`

---

### `DPOTrainer`

```python
from cambrian import DPOTrainer

trainer = DPOTrainer(backend=backend, beta=0.1, n_refinements=3)
pairs   = trainer.collect_pairs(population, task)
refined = trainer.refine(agent, pairs, task)   # → Agent
updated = trainer.train(population, task)      # → list[Agent]
```

Wraps `DPOSelector` with LLM-driven genome refinement. For each top
preference pair, asks the backend to produce an improved genome that
captures the `chosen` agent's strengths while avoiding the `rejected`
agent's weaknesses.

---

### `GoalDriftDetector`

```python
from cambrian import GoalDriftDetector

det = GoalDriftDetector(drift_threshold=0.4, window=5)
det.register(agent, intent="expert step-by-step reasoning")
event = det.measure(agent, generation=3)    # → DriftEvent
flagged = det.scan_population(population, generation=3)
```

Detects **semantic drift** from the original optimisation intent using
Jaccard word-overlap (no external dependencies). A `drift_score` of `1.0`
means the prompt shares no words with the original intent.

**Key parameters:**
- `drift_threshold` — Jaccard distance above which an agent is flagged (default `0.4`)
- `window` — rolling window for drift history

---

### `DriftEvent`

Dataclass returned by `GoalDriftDetector.measure()`.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | UUID of the agent |
| `generation` | `int` | Generation number |
| `drift_score` | `float` | Jaccard distance ∈ [0, 1] |
| `original_intent` | `str` | The intent string at registration |
| `current_prompt` | `str` | Current system prompt |
| `flagged` | `bool` | `True` if drift_score > threshold |

---

### `FitnessAnomalyDetector`

```python
from cambrian import FitnessAnomalyDetector

det = FitnessAnomalyDetector(z_threshold=2.5, min_history=5)
det.record(agent, generation=10)
is_bad = det.is_anomalous(agent)
ids = det.scan(population, generation=10)   # → list[str] of anomalous IDs
```

Flags agents whose fitness jumps more than `z_threshold` standard deviations
above the population mean — a signal of potential **reward hacking**.
Uses `statistics.mean` / `statistics.stdev` (stdlib only).

---

### `SafeguardController`

```python
from cambrian import SafeguardController, GoalDriftDetector, FitnessAnomalyDetector

ctrl = SafeguardController(
    drift_detector=GoalDriftDetector(drift_threshold=0.4),
    anomaly_detector=FitnessAnomalyDetector(z_threshold=2.5),
    backend=backend,   # optional — enables remediation
)
report = ctrl.check(population, generation=5)
# report = {"drift_violations": [...], "anomaly_detections": [...]}

remediated_agent = ctrl.remediate(agent, task)
```

Combines drift detection and anomaly detection into a single controller.
If `backend` is provided, `remediate()` calls the LLM to re-ground the
agent's prompt toward its original intent.

---

## Utilities

### `JSONLogger`

`cambrian.utils.logging.JSONLogger`

NDJSON per-generation structured logging.

```python
with JSONLogger("run.ndjson") as log:
    log.log_generation(gen=1, population=[...], extras={"task": "..."})
```

**Key methods:** `log_generation(gen, population, extras={})`, `log_run_summary(best_agent, extras={})`

### `load_json_log(path: str) -> list[dict]`

Read NDJSON evolution log; skips malformed lines.

---

### `export_genome_json(genome, path)` / `load_genome_json(path) -> Genome`

Save and load evolved genomes as JSON.

### `export_standalone(agent, path)`

Export as self-contained Python script with embedded genome.

### `export_mcp(agent, path)`

Export as MCP server stub (manifest + handler).

### `export_api(agent, path)`

Export as FastAPI REST application.

---

### Reward Shapers (`cambrian.reward_shaping`)

| Class | Effect |
|-------|--------|
| `ClipShaper` | Clamp to [min, max] |
| `NormalisationShaper` | z-score or min-max normalisation |
| `PotentialShaper` | Potential-based shaping (Ng 1999) |
| `RankShaper` | Convert to fractional rank |
| `CuriosityShaper` | Intrinsic motivation for novel genomes |

`build_shaped_evaluator(base, "clip+normalise+curiosity")` — factory string.

---

*This document is generated from Cambrian v1.0.1 source. For the latest API, run `help(cambrian.<ClassName>)` or read the inline docstrings.*
