# Changelog

All notable changes to Cambrian are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes not yet released._

---

## [0.1.0] — 2026-04-14

### Added

**Core framework**
- `Genome` dataclass: evolvable agent specification (system prompt, tools, strategy, temperature, model)
- `Agent` class: wraps `Genome` with an optional LLM backend; exposes `run()`, `clone()`, `fitness`, `to_dict()`
- `EvolutionEngine`: full generational loop with tournament selection, elitism, crossover, and mutation; `backend` propagated to all agents
- `LLMMutator`: LLM-guided genome mutation and crossover; deterministic fallbacks (`_random_tweak`, `_deterministic_crossover`) on error
- `MAPElites`: 3×3 quality-diversity archive keyed on (prompt-length bucket, temperature bucket)
- `EvolutionaryMemory`: NetworkX directed-graph lineage tracking; `get_top_ancestors`, `get_lineage`, JSON serialisation
- `SemanticCache`: SHA-256 keyed LRU cache with TTL and hit-rate introspection
- `ModelRouter`: complexity-based tier routing (cheap / medium / premium) using token count + regex signals
- `caveman_compress`: aggressive stopword removal for prompt length reduction
- `procut_prune`: paragraph-dropping pruner to fit a token budget

**Backends**
- `LLMBackend`: abstract base class with typed `generate(**kwargs: Any)` signature
- `OpenAICompatBackend`: httpx-based client with exponential back-off; supports OpenAI, Ollama, Groq, LM Studio, vLLM

**Evaluators**
- `CodeEvaluator`: sandboxed subprocess execution; partial-credit line-level scoring
- `LLMJudgeEvaluator`: 0–10 rubric via judge LLM; JSON parse + integer regex fallback; markdown-fence aware
- `CompositeEvaluator`: weighted mean or min-aggregate; exception-safe sub-evaluator calls

**CLI** (`cambrian`)
- `cambrian evolve TASK` — run evolutionary search
- `cambrian dashboard MEMORY_FILE` — per-generation stats from lineage JSON
- `cambrian distill GENOME_FILE` — pretty-print a saved genome
- `cambrian version` — print version

**Utilities**
- `run_in_sandbox`: subprocess execution with hard timeout, tempfile-based injection safety
- `extract_python_code`: fenced code block extraction from LLM output
- `get_logger`: Rich/plain logging auto-detected by TTY
- `log_generation_summary`: structured generation log line

**Examples**
- `examples/evolve_fizzbuzz.py` — FizzBuzz evolution with custom partial-credit evaluator
- `examples/evolve_coding.py` — five-challenge coding benchmark with composite evaluator
- `examples/evolve_prompt.py` — open-ended Socratic tutor prompt evolution with LLM judge

**CI / tooling**
- GitHub Actions: pytest matrix (Python 3.11 + 3.12), ruff lint + format, mypy
- 76 pytest tests across all modules; zero warnings

### Fixed
- `Agent.backend` made optional (default `None`) — allows genome/fitness tests without a real LLM
- `Agent.id` property added as alias for `agent_id`
- `Agent.to_dict()` method added
- `EvolutionEngine` now accepts and propagates `backend` to all created agents
- `evolve_fizzbuzz.py`: `timeout_seconds` corrected to `timeout`
- Tournament selection statistical threshold lowered to match true expected value (k/n × trials)

---

[Unreleased]: https://github.com/Franck1120/cambrian/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Franck1120/cambrian/releases/tag/v0.1.0
