# Cambrian — Methodology & Theoretical Background

> Version: 1.0.0 · Last updated: 2026-04-15

This document maps every evolutionary technique implemented in Cambrian to its
primary academic reference and explains how the theory translates into code.

---

## Table of Contents

1. [Evolutionary Framework](#1-evolutionary-framework)
2. [Lamarckian Evolution](#2-lamarckian-evolution)
3. [Stigmergy](#3-stigmergy)
4. [Epigenetic Layer](#4-epigenetic-layer)
5. [Artificial Immune System](#5-artificial-immune-system)
6. [Baldwin Effect](#6-baldwin-effect)
7. [Monte Carlo Tree Search (MCTS)](#7-monte-carlo-tree-search-mcts)
8. [Adversarial Co-Evolution](#8-adversarial-co-evolution)
9. [Curriculum Learning](#9-curriculum-learning)
10. [Constitutional AI](#10-constitutional-ai)
11. [Multi-Objective Optimisation — NSGA-II](#11-multi-objective-optimisation--nsga-ii)
12. [Archipelago / Island Model](#12-archipelago--island-model)
13. [Speculative Execution](#13-speculative-execution)
14. [Reward Shaping](#14-reward-shaping)
15. [Agent-to-Agent (A2A) Protocol](#15-agent-to-agent-a2a-protocol)
16. [Anti-Reward-Hacking](#16-anti-reward-hacking)
17. [Population Statistics](#17-population-statistics)
18. [Diffusion Chain-of-Thought (DiffCoT)](#18-diffusion-chain-of-thought-diffcot)
19. [Causal Reasoning](#19-causal-reasoning)
20. [Tool Invention](#20-tool-invention)
21. [Self-play](#21-self-play)
22. [Meta-Evolution (MAML-inspired)](#22-meta-evolution-maml-inspired)
23. [World Model](#23-world-model)
24. [Dream Phase (GraphRAG Recombination)](#24-dream-phase-graphrag-recombination)
25. [Quorum Sensing](#25-quorum-sensing)
26. [Mixture of Agents (MoA)](#26-mixture-of-agents-moa)
27. [Quantum Tunneling](#27-quantum-tunneling)
28. [Reflexion](#28-reflexion)
29. [Inference-time Scaling](#29-inference-time-scaling)
30. [Glossolalia (Latent Monologue)](#30-glossolalia-latent-monologue)
31. [Symbiotic Fusion](#31-symbiotic-fusion)
32. [Hormesis (Graduated Stress Response)](#32-hormesis-graduated-stress-response)
33. [Apoptosis (Programmed Agent Removal)](#33-apoptosis-programmed-agent-removal)
34. [Catalysis](#34-catalysis)
35. [LLM Cascade (Tiered Routing)](#35-llm-cascade-tiered-routing)
36. [Ensemble & Boosting](#36-ensemble--boosting)
37. [Transfer Learning](#37-transfer-learning)
38. [Tabu Search](#38-tabu-search)
39. [Simulated Annealing](#39-simulated-annealing)
40. [Red Teaming & Robustness Evaluation](#40-red-teaming--robustness-evaluation)
41. [Zeitgeber (Circadian Oscillator)](#41-zeitgeber-circadian-oscillator)
42. [Horizontal Gene Transfer (HGT)](#42-horizontal-gene-transfer-hgt)
43. [Transgenerational Epigenetics](#43-transgenerational-epigenetics)
44. [B/T-cell Immune Memory](#44-bt-cell-immune-memory)
45. [Neuromodulation](#45-neuromodulation)

---

## 1. Evolutionary Framework

**Cambrian implements a (μ + λ) evolutionary strategy with tournament selection.**

At each generation:
1. The top-`elite_n` agents survive unchanged (elitism).
2. Remaining slots are filled via tournament selection: draw `tournament_k`
   agents at random, keep the winner.
3. Each selected agent is either mutated (probability `mutation_rate`) or
   crossed over with a second parent (probability `crossover_rate`), using an
   LLM as the genetic operator.

**Core reference:**

> Rechenberg, I. (1973). *Evolutionsstrategie: Optimierung technischer Systeme
> nach Prinzipien der biologischen Evolution*. Stuttgart: Frommann-Holzboog.

> Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*.
> University of Michigan Press.

**LLM as mutation operator:**

> Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., … & Lu, Y. (2023).
> Connecting Large Language Models with Evolutionary Algorithms Yields Powerful
> Prompt Optimizers. *arXiv:2309.08532*.

> Fernando, C., Banarse, D., Michalewski, H., Osindero, S., & Rocktäschel, T.
> (2023). Promptbreeder: Self-referential Self-Improvement via Prompt Evolution.
> *arXiv:2309.16797*.

---

## 2. Lamarckian Evolution

**Core idea:** Acquired characteristics (successful solutions observed at
runtime) are incorporated into the genome and inherited by offspring.

In Cambrian (`cambrian/lamarck.py`):
- When an agent scores above `capture_threshold`, the `(task, response, score)`
  triple is stored in `genome.few_shot_examples`.
- Offspring inherit these examples as in-context demonstrations, giving them a
  head start on similar tasks.

**References:**

> Whitley, D., Gordon, V.S., & Mathias, K. (1994). Lamarckian Evolution, the
> Baldwin Effect and Function Optimization. *PPSN III*.

> Turney, P. (1996). Myths and Legends of the Baldwin Effect.
> *ICML Workshop on Evolutionary Computing and Machine Learning*.

---

## 3. Stigmergy

**Core idea:** Agents deposit pheromone-like traces encoding high-scoring
prompt patterns into shared memory.  Future mutations are biased toward proven
patterns.

In Cambrian (`cambrian/memory.py`, `cambrian/mutator.py`):
- `EvolutionaryMemory.add_trace(agent_id, content, score)` stores a
  `StigmergyTrace`.
- `LLMMutator.mutate()` retrieves the top-N traces and injects them into the
  LLM mutation prompt.

**References:**

> Dorigo, M., & Gambardella, L.M. (1997). Ant Colony System: A Cooperative
> Learning Approach to the Traveling Salesman Problem. *IEEE Transactions on
> Evolutionary Computation*, 1(1), 53–66.

> Dorigo, M., Birattari, M., & Stutzle, T. (2006). Ant Colony Optimization.
> *IEEE Computational Intelligence Magazine*.

---

## 4. Epigenetic Layer

**Core idea:** Gene expression is context-dependent.  The same genome can
produce different phenotypes depending on environmental signals (generation
number, population fitness, task type).

In Cambrian (`cambrian/epigenetics.py`):
- `EpigeneticLayer` applies `EpigeneticRule` functions that append annotations
  to the agent's system prompt at runtime.
- The genome is never modified; only the runtime context changes.
- Standard rules: generation phase (explore/exploit), fitness signal, task mode,
  population diversity pressure.

**References:**

> Jablonka, E., & Lamb, M.J. (2005). *Evolution in Four Dimensions: Genetic,
> Epigenetic, Behavioral, and Symbolic Variation in the History of Life*.
> MIT Press.

> Jaenisch, R., & Bird, A. (2003). Epigenetic regulation of gene expression:
> how the genome integrates intrinsic and environmental signals.
> *Nature Genetics*, 33, 245–254.

---

## 5. Artificial Immune System

**Core idea:** Biological immune systems maintain memory of past pathogens and
suppress re-stimulation.  Applied to evolution: agents with genome
configurations that previously scored poorly are *suppressed* (skipped) to
avoid wasting evaluations.

In Cambrian (`cambrian/immune.py`):
- `fingerprint(agent)` hashes the normalised prompt + strategy + temperature
  bucket + model into a 16-character ID.
- `ImmuneMemory.is_suppressed(agent)` returns `True` if this fingerprint's
  best recorded fitness is below `suppression_threshold`.

**References:**

> de Castro, L.N., & Timmis, J. (2002). *Artificial Immune Systems: A New
> Computational Intelligence Approach*. Springer.

> Forrest, S., Perelson, A.S., Allen, L., & Cherukuri, R. (1994).
> Self-nonself discrimination in a computer. *IEEE Symposium on Security and
> Privacy*.

---

## 6. Baldwin Effect

**Core idea:** Individual learning during a lifetime (in-context adaptation)
can guide genetic evolution without Lamarckian inheritance.  Agents that
benefit most from within-lifetime adaptation are selected.

In Cambrian (`cambrian/evaluators/baldwin.py`):
- `BaldwinEvaluator` runs the agent `n_trials` times on the same task,
  prepending feedback from the previous trial.
- Fitness = `base_score + baldwin_bonus × max(0, last_score − first_score)`.

**References:**

> Baldwin, J.M. (1896). A new factor in evolution. *The American Naturalist*,
> 30(354), 441–451.

> Hinton, G.E., & Nowlan, S.J. (1987). How Learning Can Guide Evolution.
> *Complex Systems*, 1, 495–502.

---

## 7. Monte Carlo Tree Search (MCTS)

**Core idea:** UCB1-guided tree search balances exploration of novel mutation
paths with exploitation of proven lineages.

In Cambrian (`cambrian/mcts.py`):
- Each agent is a node in a tree rooted at the seed genome.
- `MCTSSelector.select()` picks the agent with the highest UCB1 score for
  mutation.
- After evaluation, `backpropagate(agent_id, reward)` updates all ancestors.

UCB1 formula:
```
UCB1(v) = v.total_reward / v.visits + C * sqrt(ln(parent.visits) / v.visits)
```

**References:**

> Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning.
> *ECML 2006*, LNCS 4212, 282–293.

> Browne, C., Powley, E., Whitehouse, D., et al. (2012). A Survey of Monte
> Carlo Tree Search Methods. *IEEE TCIAIG*, 4(1), 1–43.

---

## 8. Adversarial Co-Evolution

**Core idea:** Two populations evolve antagonistically: generators produce
solutions, adversaries produce test cases that break them.  This creates an
arms race that prevents fitness stagnation.

In Cambrian (`cambrian/coevolution.py`):
- Generator fitness is penalised proportionally to the fraction of adversarial
  test cases it fails.
- Adversary fitness is the fraction of generators it breaks.

**References:**

> Rosin, C.D., & Belew, R.K. (1997). New Methods for Competitive Coevolution.
> *Evolutionary Computation*, 5(1), 1–29.

> Ficici, S.G., & Pollack, J.B. (1998). Challenges in Coevolutionary Learning.
> *ALIFE VI*.

---

## 9. Curriculum Learning

**Core idea:** Present tasks in order of increasing difficulty.  The
population advances to the next stage only when the current stage is mastered.

In Cambrian (`cambrian/curriculum.py`):
- `CurriculumScheduler` manages a sequence of `CurriculumStage` objects.
- `advance(fitness_values)` triggers stage promotion when the population's
  best fitness exceeds `stage.threshold`.

**References:**

> Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum
> Learning. *ICML 2009*.

> Graves, A., Bellemare, M.G., Menick, J., Munos, R., & Kavukcuoglu, K.
> (2017). Automated Curriculum Learning for Neural Networks. *ICML 2017*.

---

## 10. Constitutional AI

**Core idea:** Critique-revise cycles enforce behavioural constraints (a
"constitution") before fitness is evaluated.

In Cambrian (`cambrian/constitutional.py`):
- `ConstitutionalWrapper` gathers LLM-generated critiques for each
  constitutional principle.
- If any critique is non-trivial, the system prompt is revised to address it.
- The base evaluator scores the revised prompt; the genome is always restored.

**References:**

> Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI:
> Harmlessness from AI Feedback. *arXiv:2212.08073*.

---

## 11. Multi-Objective Optimisation — NSGA-II

**Core idea:** Instead of a single scalar fitness, agents are evaluated on
multiple objectives simultaneously (e.g. task accuracy and prompt brevity).
NSGA-II selects the non-dominated Pareto-optimal set while maintaining
diversity via crowding distance.

In Cambrian (`cambrian/pareto.py`):
- `fast_non_dominated_sort()` assigns front ranks in O(M·N²).
- `crowding_distance()` assigns diversity scores within each front.
- `nsga2_select()` combines both into a selection operator.

**References:**

> Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
> elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on
> Evolutionary Computation*, 6(2), 182–197.

> Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the
> Strength Pareto Evolutionary Algorithm. *TIK Report 103, ETH Zürich*.

---

## 12. Archipelago / Island Model

**Core idea:** Divide the population into isolated sub-populations (islands)
that evolve independently, with periodic migration of top individuals.  This
maintains global diversity while allowing local convergence.

In Cambrian (`cambrian/archipelago.py`):
- `Archipelago` manages N `Island` objects, each with its own
  `EvolutionEngine`.
- Migration topologies: `ring`, `all_to_all`, `random`.
- Migration interval and rate are configurable.

**References:**

> Whitley, D., Rana, S., & Heckendorn, R.B. (1998). The Island Model Genetic
> Algorithm: On Separability, Population Size and Convergence.
> *Journal of Computing and Information Technology*, 6(3), 243–260.

> Cantu-Paz, E. (1998). A Survey of Parallel Genetic Algorithms.
> *Calculateurs Paralleles*, 10(2), 141–171.

---

## 13. Speculative Execution

**Core idea:** Generate K candidate mutations in parallel and keep the best.
Named after CPU speculative execution: issue multiple branches simultaneously
and discard the losing paths.

In Cambrian (`cambrian/speculative.py`):
- `speculate()` uses `asyncio` to generate K mutations concurrently via thread
  pool.
- `SpeculativeMutator` wraps `LLMMutator` for drop-in use in
  `EvolutionEngine`.

**References:**

> This is an engineering technique adapted from computer architecture
> (speculative branch execution) for stochastic LLM sampling:

> Smith, J.E., & Pleszkun, A.R. (1988). Implementing Precise Interrupts in
> Pipelined Processors. *IEEE Transactions on Computers*, 37(5), 562–573.

> Chen, T., et al. (2023). Accelerating Large Language Model Decoding with
> Speculative Sampling. *arXiv:2302.01318*.

---

## 14. Reward Shaping

**Core idea:** Post-process raw fitness signals to improve the learning signal
without changing the optimal solution (potential-based shaping is policy
invariant).

In Cambrian (`cambrian/reward_shaping.py`):

| Shaper | Effect |
|--------|--------|
| `ClipShaper` | Bound fitness to [0, 1] |
| `NormalisationShaper` | Online z-score or min-max normalisation |
| `PotentialShaper` | Brevity bonus via potential-based shaping |
| `RankShaper` | Convert scores to fractional rank |
| `CuriosityShaper` | Intrinsic motivation for novel genomes |

**References:**

> Ng, A.Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward
> Transformations: Theory and Application to Reward Shaping. *ICML 1999*.

> Burda, Y., et al. (2018). Large-Scale Study of Curiosity-Driven Learning.
> *arXiv:1808.04355*.

---

## 15. Agent-to-Agent (A2A) Protocol

**Core idea:** Capable agents decompose complex tasks and delegate sub-tasks to
specialist agents.  Selection pressure rewards agents that effectively leverage
the network.

In Cambrian (`cambrian/a2a.py`):
- `AgentCard` declares domain capabilities.
- `AgentNetwork.route()` finds the best agent for a given task using a
  relevance score × fitness weighted sum.
- `delegate()`, `broadcast()`, `chain()`, and `majority_vote()` implement
  different inter-agent communication patterns.

**References:**

> Google (2025). *Agent2Agent (A2A) Protocol Specification*.
> github.com/google-deepmind/agent2agent

> Wooldridge, M., & Jennings, N.R. (1995). Intelligent Agents: Theory and
> Practice. *Knowledge Engineering Review*, 10(2), 115–152.

---

## 16. Anti-Reward-Hacking

**Core idea:** Agents that maximise a single evaluator can exploit its blind
spots.  Variance-aware evaluation penalises inconsistent agents.

In Cambrian (`cambrian/evaluators/variance_aware.py`):
- `VarianceAwareEvaluator` runs `n_trials` evaluations and computes
  `mean − penalty × variance`.
- `build_diversified_evaluator()` combines multiple evaluators with different
  rubrics to reduce exploitability.

**References:**

> Gao, L., et al. (2022). Scaling Laws for Reward Model Overoptimization.
> *arXiv:2210.10760*.

> Krakovna, V., et al. (2020). Specification gaming: the flip side of AI
> ingenuity. *DeepMind blog*.

---

## 17. Population Statistics

**Core idea:** Track Pareto fronts, diversity, and fitness landscapes to
diagnose evolution health and stagnation.

In Cambrian (`cambrian/stats.py`):

| Class | What it measures |
|-------|-----------------|
| `ParetoFront` | Non-dominated agents by (fitness, brevity) |
| `DiversityTracker` | Per-generation entropy, temperature std, prompt std |
| `FitnessLandscape` | 2D grid: mean fitness by temperature × token-length bins |

**References:**

> Knowles, J., & Corne, D. (2000). Approximating the Nondominated Front
> Using the Pareto Archived Evolution Strategy.
> *Evolutionary Computation*, 8(2), 149–172.

> Squillero, G., & Tonda, A. (2016). Divergence of Character and Premature
> Convergence: A Survey of Methodologies for Promoting Diversity in
> Evolutionary Optimisation. *Information Sciences*, 329, 782–799.

---

## 18. Diffusion Chain-of-Thought (DiffCoT)

**Core idea:** Apply the iterative denoising philosophy of diffusion models to
chain-of-thought reasoning.  Start with a noisy (high-temperature) draft
response and progressively refine it over N steps with decreasing temperature,
similar to score-based denoising.

In Cambrian (`cambrian/diffcot.py`):
- `DiffCoTConfig` controls `n_steps`, `noise_level`, and `temperature_schedule`
  (cosine / linear / constant).
- `DiffCoTReasoner.reason(system_prompt, task)` runs the denoising loop: each
  step appends the previous draft as context and asks the LLM to refine it.
- `DiffCoTEvaluator` wraps any base evaluator; `make_diffcot_evaluator()`
  factory for quick setup.

**References:**

> Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
> Models. *NeurIPS 2020*.

> Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of
> the Data Distribution. *NeurIPS 2019*.

---

## 19. Causal Reasoning

**Core idea:** Represent agent strategy as an explicit causal graph of
cause-effect relationships.  Evolving the graph alongside the genome allows the
agent to reason about interventions, not just correlations.

In Cambrian (`cambrian/causal.py`):
- `CausalEdge` stores cause, effect, strength, and confidence.
- `CausalGraph` parses IF/THEN, arrow notation, and natural-language causal
  statements extracted from strategy text.
- `CausalStrategyExtractor` uses an LLM to extract causal graphs.
- `CausalMutator` evolves causal graphs alongside genomes.
- `inject_causal_context(genome, graph)` appends a structured causal context
  block to the system prompt at evaluation time.

**References:**

> Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.).
> Cambridge University Press.

> Scholkopf, B., et al. (2021). Toward Causal Representation Learning.
> *Proceedings of the IEEE*, 109(5), 612–634.

---

## 20. Tool Invention

**Core idea:** Agents propose and dry-run new CLI tools during evolution.  The
population shares a tool registry; tools that survive usage across agents
become part of the shared toolkit — analogous to niche construction.

In Cambrian (`cambrian/tool_creation.py`):
- `ToolSpec` is stored in `Genome.tool_specs` alongside the system prompt.
- `ToolInventor.invent(task, backend)` prompts an LLM to propose a tool,
  validates its name format, and dry-runs the command with a sentinel input.
- `ToolPopulationRegistry` deduplicates by name, ranks by fitness-weighted
  usage, and exposes `top_n(k)`.

**References:**

> Creswell, A., et al. (2022). Selection-Inference: Exploiting Large Language
> Models for Interpretable Logical Reasoning. *arXiv:2205.09712*.

> Schick, T., et al. (2024). Toolformer: Language Models Can Teach Themselves
> to Use Tools. *NeurIPS 2024*.

---

## 21. Self-play

**Core idea:** Head-to-head competition between agents acts as an additional
selection pressure that is independent of (and complementary to) the primary
evaluator, preventing agents from overfitting a single evaluator's blind spots.

In Cambrian (`cambrian/self_play.py`):
- `SelfPlayEvaluator` runs two agents on the same task and awards a fitness
  bonus to the winner and a penalty to the loser.
- `run_tournament(population, evaluator, task)` runs a full round-robin and
  returns a ranked `TournamentRecord`.

**References:**

> Samuel, A.L. (1959). Some Studies in Machine Learning Using the Game of
> Checkers. *IBM Journal of Research and Development*, 3(3), 210–229.

> Silver, D., et al. (2017). Mastering the Game of Go without Human Knowledge.
> *Nature*, 550, 354–359.

---

## 22. Meta-Evolution (MAML-inspired)

**Core idea:** Evolve the hyperparameters of the evolutionary process itself
alongside the genomes.  This second-order optimisation finds configurations
that generalise well across tasks — the evolutionary equivalent of
Model-Agnostic Meta-Learning (MAML).

In Cambrian (`cambrian/meta_evolution.py`):
- `HyperParams` bundles `mutation_rate`, `crossover_rate`, `temperature`,
  `tournament_k`, `elite_ratio` with `perturb()` and `clamp()`.
- `MetaEvolutionEngine` tries `n_candidates` perturbed HP configs every
  `meta_interval` generations and keeps the best-performing configuration.

**References:**

> Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for
> Fast Adaptation of Deep Networks. *ICML 2017*.

> Schmidhuber, J. (1987). *Evolutionary Principles in Self-Referential
> Learning*. Institut f. Informatik, TU Munich.

---

## 23. World Model

**Core idea:** Each agent maintains a predictive model of its own performance,
analogous to Dyna-Q / Ha & Schmidhuber's world model.  Agents that accurately
predict their own fitness receive a bonus — selecting for self-aware agents.

In Cambrian (`cambrian/world_model.py`):
- Each agent accumulates `(task, fitness)` experience in a fixed-capacity
  rolling buffer.
- `WorldModel.predict(task)` uses weighted nearest-neighbour (word-level
  Jaccard similarity) to estimate expected performance.
- `WorldModelEvaluator` blends raw fitness with prediction accuracy:
  `blend * raw + (1 - blend) * accuracy`.

**References:**

> Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122*.

> Sutton, R.S. (1991). Dyna, an Integrated Architecture for Learning,
> Planning, and Reacting. *SIGART Bulletin*, 2(4), 160–163.

---

## 24. Dream Phase (GraphRAG Recombination)

**Core idea:** Between active generations, the system enters a "dream" phase —
a GraphRAG-style offline recombination step.  High-fitness ancestor genomes
are retrieved from the lineage graph and an LLM synthesises novel variants
from their combined experience, without any new evaluations.

In Cambrian (`cambrian/dream.py`):
- `DreamPhase.should_dream(generation)` fires every `dream_interval` generations.
- `dream(task, n_offspring)` queries the lineage graph for top-fitness
  ancestors, formats an experience context, and asks the LLM to synthesise
  novel offspring.
- `dream_count` tracks total dream events; offspring enter the next generation
  as pre-seeded candidates.

**References:**

> O'Neill, J., & Bhatt, D.L. (2022). Memory Consolidation During Sleep.
> *Science*, 374(6567).

> Edge, D., et al. (2024). From Local to Global: A Graph RAG Approach to
> Query-Focused Summarization. *arXiv:2404.16130*.

---

## 25. Quorum Sensing

**Core idea:** Biological bacteria auto-regulate collective behaviour based on
population density.  Applied to evolution: the Shannon entropy of the fitness
distribution auto-regulates the mutation rate.  Low diversity triggers
exploration boosts; high diversity triggers exploitation damping.

In Cambrian (`cambrian/quorum.py`):
- `QuorumSensor.sense(population)` computes Shannon entropy of the fitness
  histogram.
- Below `low_threshold`: `mutation_rate *= boost_factor`.
- Above `high_threshold`: `mutation_rate *= decay_factor`.
- Rate is clamped to `[min_rate, max_rate]`.

**References:**

> Miller, M.B., & Bassler, B.L. (2001). Quorum Sensing in Bacteria.
> *Annual Review of Microbiology*, 55, 165–199.

> Whiteson, S., & Stone, P. (2006). Evolutionary Function Approximation for
> Reinforcement Learning. *JMLR*, 7, 877–917.

---

## 26. Mixture of Agents (MoA)

**Core idea:** Run N agent specialisations on the same task and aggregate their
responses with a meta-LLM.  Diversity among base agents (different system
prompts, temperatures, strategies) improves aggregate performance beyond any
single agent.

In Cambrian (`cambrian/moa.py`):
- `MixtureOfAgents.query(task)` runs all base agents, collects their
  responses, and calls an aggregator LLM to synthesise the final answer.
- Handles individual agent failures gracefully; falls back to longest response
  on aggregator failure.

**References:**

> Wang, J., et al. (2024). Mixture-of-Agents Enhances Large Language Model
> Capabilities. *arXiv:2406.04692*.

> Ensemble methods: Dietterich, T.G. (2000). Ensemble Methods in Machine
> Learning. *MCS 2000*, LNCS 1857, 1–15.

---

## 27. Quantum Tunneling

**Core idea:** In quantum mechanics, a particle can tunnel through an energy
barrier it classically cannot surmount.  Applied to evolutionary search:
occasional large-jump mutations escape local optima by replacing a genome with
a radically different variant — stochastic teleportation across the fitness
landscape.

In Cambrian (`cambrian/moa.py` — `QuantumTunneler`):
- With probability `tunnel_prob`, `tunnel(agent, backend)` replaces the
  agent's genome with a fully randomised variant (temperature, strategy, and
  optionally an LLM-generated novel prompt).
- `tunnel_all(population, backend)` applies tunneling to the entire
  population.

**References:**

> Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and
> Machine Learning*. Addison-Wesley.

> Quantum-inspired optimisation: Narayanan, A., & Moore, M. (1996).
> Quantum-Inspired Genetic Algorithms. *CEC 1996*.

---

## 28. Reflexion

**Core idea:** Agents self-reflect on their own outputs via a critique loop
before finalising a response.  This "chain of hindsight" accelerates
improvement without gradient updates — the agent uses verbal reinforcement
signals.

In Cambrian (`cambrian/reflexion.py`):
- `ReflexionEvaluator.evaluate(agent, task)` runs a generate → critique →
  revise loop for `n_reflections` cycles.
- The critique LLM returns a `CRITIQUE: ...` and `SCORE: X.XX` format.
- Revision LLM receives the original task + current response + critique and
  returns an improved response.
- Early exit when score exceeds `early_exit_threshold`.

**References:**

> Shinn, N., Cassano, F., Berman, E., Gopalan, A., Narasimhan, K., & Yao, S.
> (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.
> *NeurIPS 2023*.

---

## 29. Inference-time Scaling

**Core idea:** Allocate more compute at inference time (not training time) to
improve output quality.  Best-of-N and beam search trade LLM calls for higher
accuracy — complementary to evolutionary search which trades generations for
genome quality.

In Cambrian (`cambrian/inference_scaling.py`):

| Class | Strategy |
|-------|---------|
| `BestOfN` | Generate N candidates, return highest-scoring |
| `BeamSearch` | Beam-width × branching-factor tree; prune after each step |
| `SelfConsistencyScorer` | Majority-vote across candidates |
| `KeywordScorer` | Score by keyword coverage |

**References:**

> Wang, X., et al. (2022). Self-Consistency Improves Chain of Thought
> Reasoning in Language Models. *arXiv:2203.11171*.

> Snell, C., et al. (2024). Scaling LLM Test-Time Compute Optimally Can Be
> More Effective than Scaling Model Parameters. *arXiv:2408.03314*.

---

## 30. Glossolalia (Latent Monologue)

**Core idea:** Before producing a final structured answer, the agent generates
a free-form, high-temperature "latent monologue" (stream of consciousness).
This acts as a scratchpad that surfaces implicit knowledge.  The monologue is
then fed to a second lower-temperature call that synthesises a structured
answer — a two-phase decode.

In Cambrian (`cambrian/glossolalia.py`):
- `GlossaloliaReasoner.reason(system_prompt, task)` runs the two phases.
- Phase 1 (monologue): `temperature = config.monologue_temperature` (high).
- Phase 2 (synthesis): `temperature = config.synthesis_temperature` (low).
- `GlossaloliaEvaluator(Evaluator)` wraps an inner evaluator for use in
  evolution loops.

**References:**

> Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in
> Large Language Models. *NeurIPS 2022*.

> Zeiler, M.D., & Fergus, R. (2014). Visualizing and Understanding
> Convolutional Networks. *ECCV 2014*. (inspiration for latent representation
> analysis)

---

## 31. Symbiotic Fusion

**Core idea:** Endosymbiosis — two previously independent organisms merge into
a single more capable entity (e.g. the mitochondrial merger).  Applied to
evolution: high-fitness agents with complementary strategies are fused by an
LLM into a hybrid genome that inherits strengths from both.

In Cambrian (`cambrian/symbiosis.py`):
- `SymbioticFuser.fuse(host, donor, task)` requires both agents to exceed
  `fitness_threshold` AND for their prompts to differ by at least
  `min_distance` (word-overlap below threshold).
- The LLM is asked to synthesise a new system prompt that combines the best
  of both strategies.
- Falls back to naive concatenation if the LLM call fails.
- `fuse_best_pair(population, task)` finds the optimal host/donor pair.

**References:**

> Margulis, L. (1967). On the Origin of Mitosing Cells. *Journal of
> Theoretical Biology*, 14(3), 225–274.

> Sagan, L. (1967). On the origin of mitosing cells (same paper, alternative
> attribution form used in evolutionary computing literature).

---

## 32. Hormesis (Graduated Stress Response)

**Core idea:** Low doses of stress can stimulate biological systems, leading
to increased robustness (hormesis).  Applied to agent evolution: poor-fitness
agents receive graduated interventions proportional to their stress level,
rather than immediate replacement.

In Cambrian (`cambrian/hormesis.py`):
- `HormesisAdapter.stress_level(agent)` maps fitness to `"none"`, `"mild"`,
  `"moderate"`, or `"severe"`.
- Mild: temperature boost (increased exploration).
- Moderate: inject a hint into the system prompt.
- Severe: full LLM re-prompt with an explicit stress signal.
- `stimulate_population(population, task)` applies graduated stress to the
  entire population.

**References:**

> Calabrese, E.J. (2008). Hormesis: Why it is Important to Toxicology and
> Toxicologists. *Environmental Toxicology and Chemistry*, 27(7), 1451–1474.

> Mattson, M.P. (2008). Hormesis Defined. *Ageing Research Reviews*, 7(1),
> 1–7.

---

## 33. Apoptosis (Programmed Agent Removal)

**Core idea:** In multicellular organisms, cells that are chronically damaged
or fail to contribute are destroyed (apoptosis) to protect the organism.
Applied to evolution: agents that stagnate below a fitness floor for
`stagnation_window` consecutive generations are removed and optionally
replaced with clones of the best survivor.

In Cambrian (`cambrian/apoptosis.py`):
- `ApoptosisController.record(agent)` appends the current fitness to the
  agent's history.
- `apply(population, best_agent)` prunes agents that are both below
  `min_fitness` AND have shown no improvement (`improvement_epsilon`) over
  the last `stagnation_window` generations.  A `grace_period` protects
  new agents from immediate pruning.

**References:**

> Kerr, J.F.R., Wyllie, A.H., & Currie, A.R. (1972). Apoptosis: A Basic
> Biological Phenomenon with Wide-Ranging Implications in Tissue Kinetics.
> *British Journal of Cancer*, 26(4), 239–257.

> Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*.
> Wiley. (discusses fitness-based pruning strategies)

---

## 34. Catalysis

**Core idea:** A catalyst accelerates a reaction without being consumed.
Applied to evolution: a high-fitness "catalyst agent" temporarily augments a
target agent's prompt context, guiding it toward better solutions without
permanently modifying either genome.

In Cambrian (`cambrian/catalysis.py`):
- `CatalystSelector.select(population)` picks the catalyst by a composite
  score of fitness, vocabulary richness, and strategy type.
- `CatalysisEngine.catalyse(target, catalyst, task)` prepends a catalyst
  context block to the target's system prompt, evaluates, then always
  restores via `finally`.

**References:**

> Biological catalysis analogy applied to: Whitley, D. (2001). An Overview of
> Evolutionary Algorithms: Practical Issues and Common Pitfalls.
> *Information and Software Technology*, 43(14), 817–831.

---

## 35. LLM Cascade (Tiered Routing)

**Core idea:** Route queries through a hierarchy of LLMs of increasing
capability and cost.  A fast/cheap model handles confident cases; expensive
models are invoked only when confidence is insufficient.  This is the LLM
equivalent of early-exit neural networks.

In Cambrian (`cambrian/llm_cascade.py`):
- `LLMCascade(levels)` holds a list of `CascadeLevel(backend, confidence_fn,
  confidence_threshold)`.
- `query(system_prompt, task)` calls levels in order, escalating when
  `confidence_fn(response) < threshold`.
- Returns `(response, level_index)` — enabling monitoring of escalation rates.
- Built-in scorers: `hedging_confidence` (penalises hedge words),
  `length_confidence` (longer = more confident proxy).

**References:**

> Teerapittayanon, S., McDanel, B., & Kung, H.T. (2016). BranchyNet: Fast
> Inference via Early Exiting from Deep Neural Networks. *ICPR 2016*.

> Chen, L., et al. (2023). FrugalGPT: How to Use Large Language Models While
> Reducing Cost and Improving Performance. *arXiv:2305.05176*.

---

## 36. Ensemble & Boosting

**Core idea:** Combine weak learners (agents) into a stronger aggregate
prediction.  Standard ensembles use weighted majority vote; AdaBoost-style
boosting iteratively re-weights agents based on per-query accuracy.

In Cambrian (`cambrian/ensemble.py`):
- `AgentEnsemble` maintains normalised weights; `query(task)` returns the
  response supported by the highest total weight.
- `BoostingEnsemble` extends with `update_weights(task, correct_answer)`:
  correct agents get weights multiplied by `boost_factor`; wrong agents by
  `decay_factor`; weights are then re-normalised.
- Built-in scorers: `exact_match_scorer`, `substring_scorer`.

**References:**

> Freund, Y., & Schapire, R.E. (1997). A Decision-Theoretic Generalization of
> On-Line Learning and an Application to Boosting. *Journal of Computer and
> System Sciences*, 55(1), 119–139.

> Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.

---

## 37. Transfer Learning

**Core idea:** Reuse knowledge acquired in one domain to accelerate learning
in a related domain.  A source genome adapted to a new task should converge
faster than starting from scratch — especially when the source domain is
well-represented in the agent's existing few-shot examples.

In Cambrian (`cambrian/transfer.py`):
- `TransferAdapter(backend, intensity)`: an LLM rewrites a source genome for
  the target task at `"light"` (hint only), `"medium"` (moderate rewrite), or
  `"heavy"` (full specialisation) intensity.
- `TransferBank(max_per_domain)` registers agents by domain label;
  `best_for(domain)` retrieves the highest-fitness source genome.

**References:**

> Pan, S.J., & Yang, Q. (2010). A Survey on Transfer Learning.
> *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359.

> Weiss, K., Khoshgoftaar, T.M., & Wang, D. (2016). A Survey of Transfer
> Learning. *Journal of Big Data*, 3(9).

---

## 38. Tabu Search

**Core idea:** Maintain a short-term memory (tabu list) of recently explored
regions to prevent the search from cycling back to local optima.  Candidate
mutations that fall in the tabu region are retried.

In Cambrian (`cambrian/tabu.py`):
- `TabuList(max_size)` stores SHA-256 fingerprints (first 16 hex chars) of
  bi-gram tokenisations of recent genome prompts.  FIFO eviction when full.
- `TabuMutator(base_mutator, tabu_list, max_retries)` wraps any mutator;
  retries up to `max_retries` times before accepting a tabu move as a last
  resort.
- `tabu_hit_rate` property enables monitoring.

**References:**

> Glover, F. (1989). Tabu Search — Part I. *ORSA Journal on Computing*,
> 1(3), 190–206.

> Glover, F. (1990). Tabu Search — Part II. *ORSA Journal on Computing*,
> 2(1), 4–32.

---

## 39. Simulated Annealing

**Core idea:** Accept worse solutions with probability `exp(-ΔE / T)` where T
decreases over time (cooling schedule).  This escapes local optima early when
T is high, and exploits near-optimally when T is low — the Metropolis
criterion applied to agent selection.

In Cambrian (`cambrian/annealing.py`):
- `AnnealingSchedule(T_max, T_min, n_steps, schedule_type)` supports
  `"linear"`, `"exponential"`, and `"cosine"` cooling curves.
- `AnnealingSelector.step(current_fitness, candidate_fitness)` implements the
  Metropolis criterion: always accepts improvements; accepts regressions with
  probability proportional to temperature.
- `acceptance_rate()` and full history enable monitoring.

**References:**

> Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983). Optimization by
> Simulated Annealing. *Science*, 220(4598), 671–680.

> Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H., &
> Teller, E. (1953). Equation of State Calculations by Fast Computing
> Machines. *Journal of Chemical Physics*, 21(6), 1087–1092.

---

## 40. Red Teaming & Robustness Evaluation

**Core idea:** Systematically probe agents with adversarial inputs to measure
robustness beyond normal task performance.  Agents that maintain high fitness
under adversarial attack are more reliable in deployment.

In Cambrian (`cambrian/red_team.py`):
- `RedTeamAgent(backend, n_attacks)` generates adversarial task variants via
  LLM with JSON parsing and regex fallback.
- `RobustnessEvaluator(judge_backend)` scores agent robustness 0–1 by
  comparing performance on normal vs. adversarial inputs.
- `RedTeamSession.run(agent, task)` → `RobustnessReport` combining
  `normal_score`, `adversarial_score`, and `robustness_ratio`.

**References:**

> Perez, E., et al. (2022). Red Teaming Language Models with Language Models.
> *arXiv:2202.03286*.

> Ganguli, D., et al. (2022). Red Teaming Language Models to Reduce Harms:
> Methods, Scaling Behaviors, and Lessons Learned. *arXiv:2209.07858*.

---

## 41. Zeitgeber (Circadian Oscillator)

**Core idea:** Biological circadian rhythms (driven by environmental
"Zeitgeber" — time-givers such as light cycles) modulate metabolic and
behavioural rates.  Applied to evolution: a sinusoidal oscillator modulates
mutation rate and selection threshold across generations, creating natural
exploration/exploitation cycles.

In Cambrian (`cambrian/zeitgeber.py`):
- `ZeitgeberClock(period, amplitude, phase_offset)` computes
  `phase = 2π × tick / period`; `exploration_factor = 0.5 + amplitude × sin(phase)`.
- `ZeitgeberScheduler` maps the clock to `mutation_rate` and
  `acceptance_threshold` via configurable base values and ranges.
- History is tracked per-tick as `ZeitgeberState` objects.

**References:**

> Pittendrigh, C.S. (1960). Circadian Rhythms and the Circadian Organization
> of Living Systems. *Cold Spring Harbor Symposia on Quantitative Biology*,
> 25, 159–184.

> Aschoff, J. (1960). Exogenous and Endogenous Components in Circadian
> Rhythms. *Cold Spring Harbor Symposia on Quantitative Biology*, 25, 11–28.

---

## 42. Horizontal Gene Transfer (HGT)

**Core idea:** In prokaryotes, genetic material is transferred between
non-parent organisms (plasmids, transduction, conjugation).  Applied to
evolution: sentence-level genome fragments (plasmids) are extracted from
high-fitness donors and injected into recipient agents, allowing beneficial
strategies to spread across the population outside the normal parent-offspring
channel.

In Cambrian (`cambrian/hgt.py`):
- `HGTransfer(n_sentences, mode, fitness_threshold)` extracts the top-scoring
  sentences from a donor genome and injects them into a recipient via
  `"prefix"`, `"suffix"`, or `"replace"` mode.
- `HGTPool(max_plasmids)` is a domain-tagged plasmid pool: `contribute()`,
  `draw()`, `best_for(domain)`.

**References:**

> Ochman, H., Lawrence, J.G., & Groisman, E.A. (2000). Lateral Gene Transfer
> and the Nature of Bacterial Innovation. *Nature*, 405, 299–304.

> Koonin, E.V., Makarova, K.S., & Aravind, L. (2001). Horizontal Gene
> Transfer in Prokaryotes: Quantification and Classification.
> *Annual Review of Microbiology*, 55, 709–742.

---

## 43. Transgenerational Epigenetics

**Core idea:** Epigenetic marks — heritable but non-genetic annotations —
are passed from parents to offspring across multiple generations, decaying
gradually.  Applied to evolution: successful behavioural patterns (strategy
annotations) are recorded, inherited with decay, and injected into offspring
genomes as contextual priors.

In Cambrian (`cambrian/transgenerational.py`):
- `EpigeneMark` stores a name, strength ∈ [0, 1], and per-generation decay.
- `TransgenerationalRegistry.record_mark(agent, name, strength)` records marks
  per agent.
- `inherit(parent, child)` transfers top-`max_marks` marks with additional
  decay applied.
- `apply_to_genome(agent)` injects surviving marks as a context block in the
  system prompt.

**References:**

> Jablonka, E., & Raz, G. (2009). Transgenerational Epigenetic Inheritance:
> Prevalence, Mechanisms, and Implications for the Study of Heredity and
> Evolution. *The Quarterly Review of Biology*, 84(2), 131–176.

> Bird, A. (2007). Perceptions of Epigenetics. *Nature*, 447, 396–398.

---

## 44. B/T-cell Immune Memory

**Core idea:** The adaptive immune system maintains two complementary memory
populations: B-cells (high-affinity antibody memory for well-matched antigens)
and T-cells (adaptive recognition of novel or partially-matched antigens).
Applied to evolution: B-cell memory caches high-fitness genomes for fast recall
on near-identical tasks; T-cell memory provides a seeded starting point for
novel tasks.

In Cambrian (`cambrian/immune_memory.py`):
- `BCellMemory(similarity_threshold, max_cells)` stores `(genome, task, fitness)`
  triples.  `recall(task)` returns the best Jaccard match above threshold (O(N)
  scan; FIFO eviction when full).
- `TCellMemory(min_similarity, max_cells)` stores all cells above a lower
  threshold and returns the best match even below the B-cell threshold.
- `ImmuneCortex(b_threshold, t_threshold, b_similarity, t_min_similarity)` is
  the coordinator: `record()` gates storage on fitness; `recall()` checks B-cell
  first, T-cell second, returning a typed `RecallResult`.

Similarity metric: word-level Jaccard (`|A ∩ B| / |A ∪ B|` on whitespace
tokenisation) — O(1) per word set, no embedding required.

**References:**

> Burnet, F.M. (1959). *The Clonal Selection Theory of Acquired Immunity*.
> Vanderbilt University Press.

> de Castro, L.N., & Von Zuben, F.J. (2002). Learning and Optimization Using
> the Clonal Selection Principle. *IEEE Transactions on Evolutionary
> Computation*, 6(3), 239–251.

---

## 45. Neuromodulation

**Core idea:** Biological neuromodulators (dopamine, serotonin, acetylcholine,
noradrenaline) do not carry information directly but modulate the *gain* of
neural circuits — tuning the balance between exploration and exploitation.
Applied to evolution: four modulators compute scalar signals from population
state and combine them to dynamically adjust `mutation_rate` and
`selection_pressure` each generation.

In Cambrian (`cambrian/neuromodulation.py`):

| Modulator | Signal | Effect |
|-----------|--------|--------|
| `DopamineModulator` | Rising fitness trend | Reduce mutation (exploit reward) |
| `SerotoninModulator` | Low prompt diversity | Increase mutation (escape sameness) |
| `AcetylcholineModulator` | High fitness variance | Reduce selection pressure |
| `NoradrenalineModulator` | Stagnation detection | Spike mutation (alertness) |

`NeuromodulatorBank.modulate(population, generation)` aggregates the four
signals and returns a clamped `NeuroState` with new `mutation_rate` and
`selection_pressure`.

Aggregation formulas:
```
mr_delta = mr_range × (0.5×serotonin + 0.5×noradrenaline − 0.5×dopamine)
sp_delta = sp_range × (0.5×dopamine − 0.5×acetylcholine)
```

**References:**

> Doya, K. (2002). Metalearning and Neuromodulation.
> *Neural Networks*, 15(4–6), 495–506.

> Friston, K., et al. (2012). Dopamine, Affordance and Active Inference.
> *PLOS Computational Biology*, 8(1), e1002327.

---

## Citing Cambrian

If you use Cambrian in research, please cite the primary evolutionary prompt
optimisation papers listed above, particularly:

```bibtex
@article{guo2023connecting,
  title   = {Connecting Large Language Models with Evolutionary Algorithms
             Yields Powerful Prompt Optimizers},
  author  = {Guo, Qingyan and Wang, Rui and Guo, Junliang and others},
  journal = {arXiv preprint arXiv:2309.08532},
  year    = {2023}
}

@inproceedings{deb2002fast,
  title     = {A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II},
  author    = {Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and
               Meyarivan, T.},
  booktitle = {IEEE Transactions on Evolutionary Computation},
  volume    = {6},
  number    = {2},
  pages     = {182--197},
  year      = {2002}
}
```
