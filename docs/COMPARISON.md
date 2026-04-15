# Cambrian vs the Field — Detailed Comparison

> Last updated: Cambrian v1.0.2

This document compares Cambrian to the five closest alternatives across
dimensions that matter most for evolutionary agent optimisation.

---

## Quick Matrix

| Feature | **Cambrian** | DSPy | DGM | AVO | EvoAgent | MiroFish |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Evolutionary genetic loop | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| LLM-guided mutation | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| No gradient required | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tournament / self-play | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Island / Archipelago | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Meta-evolution (auto hyperparams) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Code evolution (Forge) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Pipeline evolution | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Causal reasoning graphs | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| DPO preference selection | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Goal-drift safeguards | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Bio-inspired operators (Lamarck, HGT, …) | ✅ 50+ | ❌ | ❌ | ❌ | ✅ few | ❌ |
| Agent lifecycle (Metamorphosis) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ecosystem / ecological roles | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fractal multi-scale evolution | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| OpenAI-compatible (any provider) | ✅ | ✅ | ✅ | partial | ✅ | ❌ |
| CLI interface | ✅ rich | ❌ | ❌ | ❌ | ❌ | ❌ |
| Live dashboard | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Export (API / MCP / script) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Strict mypy typing | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| MIT licence | ✅ | ✅ | ✅ | varies | ✅ | ✅ |

---

## System Descriptions

### DSPy (Stanford NLP)

**What it is:** A declarative programming framework for LLMs. You compose
"modules" (Predict, ChainOfThought, ReAct) into pipelines, and DSPy's
optimiser (MIPROv2, BootstrapFewShot) searches for the best prompts and
few-shot examples given a training set.

**Strengths:**
- Best-in-class for structured pipelines with labelled training data
- Strong teleprompter / few-shot bootstrapping
- Composable, Pythonic API

**Limitations:**
- No evolutionary pressure between agents
- Optimises prompts, not agent behaviour over time
- No tournament, no multi-agent competition
- No bio-inspired operators

**When to prefer DSPy:** You have a well-defined pipeline (e.g., RAG,
structured extraction) with labelled examples. DSPy's compiler approach
is better than evolution for supervised tasks.

**When to prefer Cambrian:** You want open-ended optimisation without
labels, multi-agent competition, island models, or bio-inspired search.

---

### DGM — Darwin Gödel Machine (DeepMind, 2024)

**What it is:** A self-improving coding agent that rewrites its own source
code using LLM-generated diffs. Each version is tested; only improvements
survive.

**Strengths:**
- True self-modification (code rewrites itself)
- Strong on coding tasks
- Evolutionary code improvement loop

**Limitations:**
- Narrowly scoped to code generation
- No prompt evolution, no multi-objective support
- No bio-inspired operators beyond code mutation
- Not a general-purpose agent framework
- Not open-source (research paper only as of 2024)

**When to prefer DGM:** You need a self-modifying coding agent that
bootstraps its own capabilities.

**When to prefer Cambrian:** You want a general framework for evolving
any kind of agent behaviour (prompts, strategies, tools, pipelines),
with multi-agent competition and full CLI support.

---

### AVO — Automated Verbal Optimisation

**What it is:** A prompt optimisation technique that uses an LLM to
critique and rewrite prompts based on task feedback ("verbal gradients").

**Strengths:**
- Simple, interpretable: LLM explains why a prompt failed
- No gradient computation
- Works with any task that has text feedback

**Limitations:**
- Single-agent: no population, no selection pressure
- No crossover, no tournament
- Greedy / one-shot improvement, not global search
- No bio-inspired diversity mechanisms

**When to prefer AVO:** You want rapid single-prompt improvement with
explanations, and your task already provides good textual feedback.

**When to prefer Cambrian:** You want global search across a population
of agents, cross-pollination of strategies, and selection pressure that
prevents premature convergence.

---

### EvoAgent

**What it is:** A multi-agent framework that uses evolutionary algorithms
(mutation, crossover, selection) to refine agent teams for collaborative
tasks.

**Strengths:**
- Multi-agent evolution, not single-agent
- Supports role specialisation across agents
- Tournament-style evaluation

**Limitations:**
- Fewer bio-inspired operators
- No meta-evolution, no island model
- No pipeline/code evolution
- No CLI, no live dashboard
- Smaller community

**When to prefer EvoAgent:** You specifically want to evolve a *team* of
specialised agents that collaborate on a shared task.

**When to prefer Cambrian:** You want a broader feature set, richer
evolutionary operators, CLI support, meta-evolution, and export capabilities.

---

### MiroFish (Evolutionary Prompting)

**What it is:** A genetic algorithm approach to prompt optimisation that
treats prompts as DNA sequences, applying crossover and mutation operators
derived from evolutionary biology.

**Strengths:**
- Strict evolutionary metaphor (codons, gene splicing)
- Interesting crossover primitives

**Limitations:**
- No LLM-guided mutation (mutation is random string operations)
- No evaluator abstraction
- No meta-evolution, no island model
- Research prototype, not production-ready
- No typed API, no CLI

**When to prefer MiroFish:** Academic exploration of pure evolutionary
methods on text; useful baseline.

**When to prefer Cambrian:** Any production or research use requiring
LLM-guided mutation, typed API, diverse evaluators, or operational tooling.

---

## Architectural Comparison

### Mutation Strategy

| Framework | Mutation operator |
|-----------|------------------|
| **Cambrian** | LLM reads genome + fitness → writes improved genome (LLMMutator) |
| DSPy | Teleprompter bootstraps few-shot examples from training data |
| DGM | LLM generates code diffs to self-modify |
| AVO | LLM writes "verbal gradient" critique → rewrites prompt |
| EvoAgent | LLM + random template splicing |
| MiroFish | Random string splice / crossover (no LLM) |

### Selection Mechanism

| Framework | Selection |
|-----------|----------|
| **Cambrian** | Tournament (k=3) + elitism + NSGA-II (Pareto) + MCTS |
| DSPy | Greedy beam search over compiled programs |
| DGM | Survival of improvements (no population) |
| AVO | Greedy (take best revision) |
| EvoAgent | Tournament |
| MiroFish | Roulette-wheel selection |

### Population Management

| Framework | Population |
|-----------|-----------|
| **Cambrian** | N agents, island model, archipelago, speculative parallel |
| DSPy | Single compiled program (no population) |
| DGM | Single agent (no population) |
| AVO | Single prompt (no population) |
| EvoAgent | Multi-agent teams |
| MiroFish | Fixed-size population |

---

## Benchmark Comparison (Mock Backend)

Using Cambrian's own benchmark (50 agents × 100 generations, keyword evaluator):

| Metric | Value |
|--------|-------|
| Throughput | ~3,200 evals/s |
| 100 gen × 20 agents | < 1s |
| Peak memory (50×100) | ~21 MB |

Note: direct comparison with other frameworks requires identical tasks
and evaluators — the figures above reflect Cambrian's internal overhead
only (no LLM API calls).

---

## Summary Recommendation

```
Has labelled training data + structured pipeline?
    → DSPy

Need a self-improving coding agent?
    → DGM

Want global search + multi-agent competition + bio-inspired diversity?
    → Cambrian ← You are here

Need academic baseline / pure string GA?
    → MiroFish
```

---

*Cambrian v1.0.2 · [GitHub](https://github.com/Franck1120/cambrian)*
