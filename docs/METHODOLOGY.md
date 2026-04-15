# Cambrian — Methodology & Theoretical Background

> Version: 0.9.0 · Last updated: 2026-04-15

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
18. [Dream Phase — Memory Consolidation](#18-dream-phase--memory-consolidation)
19. [Quorum Sensing](#19-quorum-sensing)
20. [Mixture of Agents](#20-mixture-of-agents)
21. [Quantum Tunneling](#21-quantum-tunneling)
22. [Reflexion](#22-reflexion)

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

---

## 18. Dream Phase — Memory Consolidation

**Core idea:** During biological sleep, the hippocampus replays recent
experiences to consolidate memories and improve generalisation.  Cambrian's
`DreamPhase` mimics this by recombining past `Experience` objects via LLM into
synthetic "dream scenarios" that are novel but semantically related to the
original tasks.

In Cambrian (`cambrian/dream.py`):
- Past experiences (task, response, score) are collected across generations.
- `DreamPhase.generate_scenario(experiences)` prompts the LLM to create a
  hybrid scenario by blending N experiences.
- Agents are evaluated on the dream, and fitness is blended:
  `new_fitness = (1 - w) * real + w * dream_score`.

**References:**

> Wilson, M.A., & McNaughton, B.L. (1994). Reactivation of Hippocampal Ensemble
> Memories During Sleep. *Science*, 265(5172), 676–679.

> Schmidhuber, J. (1991). A Possibility for Implementing Curiosity and Boredom
> in Model-Building Neural Controllers. *Proceedings of the First International
> Conference on Simulation of Adaptive Behavior*.

> Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122*.

---

## 19. Quorum Sensing

**Core idea:** In bacterial colonies, cells measure population density by
detecting secreted chemical signals and adjust behaviour collectively.  Cambrian
uses Shannon entropy of the fitness distribution as an analogous signal: a
converged population (low entropy) triggers increased mutation rate; a chaotic
one (high entropy) triggers reduced rate.

In Cambrian (`cambrian/quorum.py`):
- `QuorumSensor.compute_entropy(fitnesses)` bins fitness into `n_bins` and
  computes normalised Shannon entropy H(X) / log₂(n_bins) ∈ [0, 1].
- `update()` adjusts mutation rate via a proportional controller:
  `new_rate = old_rate + lr × (target_entropy − current_entropy)`.
- `stagnation_detected()` triggers when entropy variance falls below a
  threshold over a rolling window.

**References:**

> Miller, M.B., & Bassler, B.L. (2001). Quorum Sensing in Bacteria.
> *Annual Review of Microbiology*, 55, 165–199.

> Shannon, C.E. (1948). A Mathematical Theory of Communication.
> *Bell System Technical Journal*, 27(3), 379–423.

---

## 20. Mixture of Agents

**Core idea:** Ensemble methods reduce variance and improve expected performance
by aggregating multiple independent predictions.  `MixtureOfAgents` runs N
agents independently and synthesises a final answer via an LLM aggregator.

In Cambrian (`cambrian/moa.py`):
- N agents run in sequence; each produces an independent answer.
- An aggregator LLM receives all N answers and synthesises the final response.
- If the aggregator fails, the longest individual answer is used as a fallback.

**References:**

> Wang, J., Wang, F., Xiong, W., et al. (2024). Mixture-of-Agents Enhances
> Large Language Model Capabilities. *arXiv:2406.04692*.

> Dietterich, T.G. (2000). Ensemble Methods in Machine Learning.
> *Lecture Notes in Computer Science*, 1857, 1–15.

---

## 21. Quantum Tunneling

**Core idea:** Quantum tunneling allows particles to pass through potential
barriers even without sufficient energy.  Analogously, `QuantumTunneler`
allows the evolutionary search to escape local optima by probabilistically
replacing non-elite agents with random genomes, independent of fitness.

In Cambrian (`cambrian/moa.py`):
- With probability `tunnel_prob`, each non-elite agent is replaced with a
  fresh random genome.
- Elite agents (top `n_elites`) are always protected.
- This is more aggressive than standard diversity injection (which is
  fitness-guided) and prevents permanent convergence to suboptimal basins.

**References:**

> Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and
> Machine Learning*. Addison-Wesley.  (Sec. 10.5: Stochastic population
> replacement as escape from local minima.)

> Szu, H., & Hartley, R. (1987). Fast Simulated Annealing. *Physics Letters A*,
> 122(3–4), 157–162.  (Tunneling as a metaphor for global optimisation.)

---

## 22. Reflexion

**Core idea:** Verbal reinforcement learning: an agent generates an output,
a critic (the same or a different LLM) provides verbal feedback, and the agent
revises its output based on the feedback.  This is applied iteratively.

In Cambrian (`cambrian/reflexion.py`):
- `ReflexionAgent.run(task)` performs:
  1. **Generate**: agent produces initial response.
  2. **Critique**: LLM critiques the response against the task.
  3. **Revise**: LLM rewrites the response to address the critique.
  4. Repeat for `n_rounds` rounds or until critique says "EXCELLENT".
- `ReflexionEvaluator` applies Reflexion before scoring, using a proxy agent
  that serves the improved response to the base evaluator.

**References:**

> Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., &
> Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement
> Learning. *arXiv:2303.11366*.

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
