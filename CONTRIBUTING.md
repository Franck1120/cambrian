# Contributing to Cambrian

Thank you for your interest in contributing! This document explains the development workflow, coding standards, and how to submit changes.

---

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- Git

### Fork and clone

```bash
git clone https://github.com/Franck1120/cambrian.git
cd cambrian
```

### Set up the development environment

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
cambrian/           Core library
  agent.py          Genome + Agent
  evolution.py      EvolutionEngine (main loop)
  mutator.py        LLMMutator
  diversity.py      MAPElites archive
  memory.py         EvolutionaryMemory (lineage graph)
  cache.py          SemanticCache
  router.py         ModelRouter
  compress.py       Prompt compression utilities
  cli.py            Click CLI entry point
  evaluator.py      Abstract Evaluator base
  backends/         LLM backend implementations
  evaluators/       Concrete evaluator implementations
  utils/            Logging + sandbox helpers

examples/           Runnable example scripts
scripts/            Development and benchmark utilities
tests/              pytest test suite
```

---

## Coding standards

### Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Type-annotate every public function signature
- Docstrings on all public classes and methods (Google style)
- No `# type: ignore` without a comment explaining why

### Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(module): short description
fix(module): short description
test(module): short description
docs(module): short description
refactor(module): short description
chore: short description
```

### Tests

- Every new public function/class needs at least one test
- Tests live in `tests/test_<module>.py`
- Use `pytest` fixtures and `monkeypatch` for isolation
- Avoid calling real LLM APIs in tests — use fake backends

### Backends in tests

```python
class _FakeBackend:
    model_name = "fake"
    def generate(self, prompt: str, **kwargs) -> str:
        return '{"score": 8}'
```

---

## Pull request process

1. Fork the repo and create a branch: `feature/my-feature` or `fix/bug-description`
2. Write your code and tests
3. Run the full test suite: `pytest tests/ -v`
4. Ensure no new warnings are introduced
5. Open a PR against `master` with a clear description

### PR checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] No lint warnings
- [ ] Docstrings on new public API
- [ ] Type annotations on new functions
- [ ] `CHANGELOG.md` updated (add entry under `## Unreleased`)

---

## Adding a new backend

1. Create `cambrian/backends/my_backend.py`
2. Subclass `LLMBackend` from `cambrian.backends.base`
3. Implement `generate(self, prompt, **kwargs) -> str` and the `model_name` property
4. Add tests in `tests/test_modules.py` or a new `tests/test_backend_my.py`
5. Export from `cambrian/backends/__init__.py`

---

## Adding a new evaluator

1. Create `cambrian/evaluators/my_evaluator.py`
2. Subclass `Evaluator` from `cambrian.evaluator`
3. Implement `evaluate(self, agent, task) -> float` (return value in `[0.0, 1.0]`)
4. Add tests

---

## Reporting issues

Please open a GitHub issue with:
- Cambrian version (`cambrian version`)
- Python version (`python --version`)
- Minimal reproducible example
- Full error traceback

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
