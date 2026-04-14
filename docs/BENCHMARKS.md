# Benchmarks

> **Status: placeholder** — Results will be populated as evaluation runs complete.
> All numbers below are provisional targets, not measured results.

---

## Overview

Cambrian is evaluated on three standard benchmarks:

| Benchmark | Task type | Metric | Status |
|-----------|-----------|--------|--------|
| [HumanEval](https://github.com/openai/human-eval) | Python code generation | pass@1 | Pending |
| [SWE-bench](https://swe-bench.github.io) | GitHub issue resolution | Resolved% | Pending |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | Multi-domain Q&A | Accuracy | Pending |

---

## HumanEval

**Task**: Generate a Python function from a docstring.  164 problems.

**Setup**:
- Evaluator: `CodeEvaluator` with `pass@1` sampling
- Seed genome: `"You are an expert Python programmer. Output only valid Python code."`
- Population: 10 agents
- Generations: 20

| Model | Baseline (0 gen) | Gen 5 | Gen 10 | Gen 20 |
|-------|-----------------|-------|--------|--------|
| gpt-4o-mini | — | — | — | — |
| gpt-4o | — | — | — | — |
| claude-3-5-haiku | — | — | — | — |

_Cells will be filled once experiments complete._

---

## SWE-bench (Lite)

**Task**: Resolve a GitHub issue by modifying repository files.  300 instances.

**Setup**:
- Evaluator: `LLMJudgeEvaluator` with patch correctness rubric
- Additional features: `LamarckianAdapter`, `EpigeneticLayer`
- Population: 8 agents
- Generations: 15

| Model | Baseline | Cambrian-evolved | Improvement |
|-------|----------|-----------------|-------------|
| gpt-4o | — | — | — |
| claude-opus-4 | — | — | — |

---

## MMLU (5-shot)

**Task**: Multiple-choice questions across 57 academic subjects.

**Setup**:
- Evaluator: `LLMJudgeEvaluator` with answer-matching rubric
- Seed genomes: chain-of-thought, direct answer, step-by-step
- Population: 6 agents
- Generations: 10

| Subject | Baseline | Cambrian-evolved |
|---------|----------|-----------------|
| STEM | — | — |
| Humanities | — | — |
| Social Sciences | — | — |
| Overall | — | — |

---

## Running Benchmarks Yourself

### HumanEval

```bash
# Install human-eval
pip install human-eval

# Run Cambrian evolution against HumanEval
python benchmarks/run_humaneval.py \
    --model gpt-4o-mini \
    --generations 10 \
    --population 8 \
    --output results/humaneval_run1.json
```

### Custom benchmark

```python
from cambrian.evaluators.code import CodeEvaluator
from cambrian.evolution import EvolutionEngine

# Replace expected_output with your benchmark oracle
evaluator = CodeEvaluator(expected_output="<expected>")
engine = EvolutionEngine(evaluator=evaluator, ...)
best = engine.evolve(seed_genomes=[...], task="<your task>", n_generations=10)
```

---

## Comparison Methodology

All runs use:
- Random seed `42` for reproducibility
- `tournament_k=3`, `elite_ratio=0.2`
- No external retrieval (closed-book)
- Wall-clock time measured on a single machine with the API as the bottleneck

Results are averaged over 3 independent runs.  Standard deviation is reported
where multiple runs are available.

---

## Contributing Results

If you run Cambrian on a benchmark and want to contribute results, open a PR
that adds your numbers to the tables above, including:
- Model name and version
- Cambrian version (`cambrian version`)
- Full hyperparameter configuration
- Raw result file (attach to the PR)
