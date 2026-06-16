# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Real convergence benchmark: Cambrian GA vs random search vs hill climbing.

The honest question this answers
--------------------------------
Does Cambrian's evolutionary loop find a better agent **prompt** for solving
code tasks than dumb baselines, *under an equal evaluation budget*? If the GA
does not beat random search, the GA adds no value and we should say so.

What it measures
----------------
For each HumanEval-style problem, each method searches over agent genomes
(system prompt + temperature + strategy). Fitness = **pass@1**: the generated
code is executed in a real subprocess sandbox against the canonical unit tests
(1.0 = passes, 0.0 = fails). All three methods get the **same** per-evaluation
budget and the **same** backend, so the comparison is fair.

Methods
-------
* ``random``  — sample N random genomes, keep the best.
* ``hill``    — start from one genome, perturb one attribute per step, keep if better.
* ``cambrian``— EvolutionEngine: population P over G generations (LLM-guided mutation).

Backend
-------
Set ``CAMBRIAN_BACKEND``:
* ``gemini`` (default) — needs ``GEMINI_API_KEY`` (free tier ≈ 1500 req/day).
* ``mock`` — offline, deterministic. **Validates the harness only — the mock does
  no real inference, so its numbers are NOT evidence of real-world value.**

Usage
-----
    # 1. Get a free key: https://aistudio.google.com/apikey
    export GEMINI_API_KEY=AIza...
    export CAMBRIAN_BACKEND=gemini

    # 2. Run (prints the LLM-call budget first, then asks nothing — just runs):
    python benchmarks/humaneval_real.py --problems 5 --budget 24

    # Quick offline plumbing check (no key, no evidence):
    CAMBRIAN_BACKEND=mock python benchmarks/humaneval_real.py --problems 3 --budget 12

Output (in benchmarks/results/)
-------------------------------
* ``humaneval_<date>.json`` — full results + config + per-evaluation convergence.
* ``humaneval_<date>.csv``  — per-method/per-problem pass + call counts.
* ``humaneval_<date>.png``  — convergence curves (best-so-far pass rate vs eval).

Verdict thresholds (vs random search, pass@1)
---------------------------------------------
* Cambrian beats random by **>15%**  → 🟢 worth it.
* **5–15%**                           → 🟡 marginal, polish.
* **<5%** or loses                    → 🔴 archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.evolution import EvolutionEngine
from cambrian.mutator import LLMMutator
from cambrian.utils.sandbox import extract_python_code, run_in_sandbox

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── HumanEval-style problem set (embedded, no download) ────────────────────────
# A small public subset. Each problem: a function stub to implement and a
# canonical test that raises on failure. Source: OpenAI HumanEval (MIT).

PROBLEMS: list[dict[str, str]] = [
    {
        "task_id": "HumanEval/0",
        "entry_point": "has_close_elements",
        "prompt": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\"Return True if any two numbers in the list are closer to each "
            "other than the given threshold.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/2",
        "entry_point": "truncate_number",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            "    \"\"\"Return the decimal (fractional) part of a positive float.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3.5) == 0.5\n"
            "    assert abs(candidate(1.33) - 0.33) < 1e-6\n"
            "    assert abs(candidate(123.456) - 0.456) < 1e-6\n"
        ),
    },
    {
        "task_id": "HumanEval/7",
        "entry_point": "filter_by_substring",
        "prompt": (
            "from typing import List\n\n"
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
            "    \"\"\"Filter an input list of strings to those that contain the "
            "given substring.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([], 'john') == []\n"
            "    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') "
            "== ['xxx', 'xxxAAA', 'xxx']\n"
            "    assert candidate(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') "
            "== ['grunt', 'prune']\n"
        ),
    },
    {
        "task_id": "HumanEval/12",
        "entry_point": "longest",
        "prompt": (
            "from typing import List, Optional\n\n"
            "def longest(strings: List[str]) -> Optional[str]:\n"
            "    \"\"\"Return the longest string in the list. If several have the same "
            "max length, return the first. Return None for an empty list.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) is None\n"
            "    assert candidate(['x', 'y', 'z']) == 'x'\n"
            "    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'\n"
        ),
    },
    {
        "task_id": "HumanEval/13",
        "entry_point": "greatest_common_divisor",
        "prompt": (
            "def greatest_common_divisor(a: int, b: int) -> int:\n"
            "    \"\"\"Return the greatest common divisor of two integers a and b.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3, 7) == 1\n"
            "    assert candidate(10, 15) == 5\n"
            "    assert candidate(49, 14) == 7\n"
            "    assert candidate(144, 60) == 12\n"
        ),
    },
    {
        "task_id": "HumanEval/23",
        "entry_point": "strlen",
        "prompt": (
            "def strlen(string: str) -> int:\n"
            "    \"\"\"Return the length of the given string.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('x') == 1\n"
            "    assert candidate('asdasnakj') == 9\n"
        ),
    },
    {
        "task_id": "HumanEval/53",
        "entry_point": "add",
        "prompt": (
            "def add(x: int, y: int) -> int:\n"
            "    \"\"\"Add two numbers x and y and return the sum.\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(0, 1) == 1\n"
            "    assert candidate(1, 0) == 1\n"
            "    assert candidate(2, 3) == 5\n"
            "    assert candidate(5, 7) == 12\n"
        ),
    },
    {
        "task_id": "HumanEval/55",
        "entry_point": "fib",
        "prompt": (
            "def fib(n: int) -> int:\n"
            "    \"\"\"Return the n-th Fibonacci number (fib(1)=1, fib(2)=1).\"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(2) == 1\n"
            "    assert candidate(5) == 5\n"
            "    assert candidate(8) == 21\n"
            "    assert candidate(10) == 55\n"
        ),
    },
]


# ── Genome search space ────────────────────────────────────────────────────────

_PROMPT_POOL = [
    "You are a helpful AI assistant.",
    "You are an expert Python programmer. Write correct, complete functions.",
    "You are a meticulous software engineer. Think step by step, handle edge cases, "
    "then return only the implementation inside a Python code block.",
    "You write production-quality Python. Be precise. Return only code.",
    "You are a coding assistant. Implement the requested function exactly as specified.",
]
_STRATEGIES = ["step-by-step", "concise", "chain-of-thought", "direct"]


def _random_genome(rng: random.Random) -> Genome:
    return Genome(
        system_prompt=rng.choice(_PROMPT_POOL),
        strategy=rng.choice(_STRATEGIES),
        temperature=round(rng.uniform(0.1, 1.0), 2),
    )


# ── Real pass@1 evaluation ─────────────────────────────────────────────────────

def make_evaluator(problem: dict[str, str]) -> Callable[[Agent, str], float]:
    """Return ``(agent, task) -> {0.0, 1.0}`` scoring pass@1 in a real sandbox."""

    def _evaluate(agent: Agent, task: str) -> float:
        try:
            response = agent.run(task)
        except BackendExhausted:
            raise  # never score a quota-blocked call as a 0.0 failure
        except Exception:
            return 0.0
        code = extract_python_code(response)
        if not code.strip():
            return 0.0
        program = (
            f"{code}\n\n{problem['test']}\n"
            f"check({problem['entry_point']})\n"
            "print('PASS')\n"
        )
        result = run_in_sandbox(program, timeout=10.0)
        return 1.0 if (result.success and "PASS" in result.stdout) else 0.0

    return _evaluate


def _problem_task(problem: dict[str, str]) -> str:
    return (
        "Complete the following Python function. Return only the full "
        "implementation inside a Python code block.\n\n"
        f"{problem['prompt']}"
    )


# ── Method results ─────────────────────────────────────────────────────────────

class BackendExhausted(RuntimeError):
    """Raised when the LLM backend fails (quota/rate-limit/auth) so the run
    can abort loudly instead of silently scoring failed calls as 0.0 — which
    would corrupt the benchmark into a fake 'everything fails' result."""


_RATE_LIMIT_SIGNS = ("429", "resource_exhausted", "quota", "rate limit",
                     "permission", "401", "403", "exceeded")


def _is_backend_exhaustion(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(s in msg for s in _RATE_LIMIT_SIGNS)


@dataclass
class MethodRun:
    method: str
    best_fitness: float
    n_evals: int
    n_llm_calls: int
    wall_s: float
    convergence: list[float] = field(default_factory=list)  # best-so-far per eval
    n_failures: int = 0  # backend calls that errored (quota/rate-limit)


class _CountingBackend(LLMBackend):
    """Wraps a backend to count generate() calls (≈ API cost proxy) and to
    detect backend exhaustion (quota/rate-limit) so the harness can abort."""

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.calls = 0
        self.failures = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        try:
            return self._inner.generate(prompt, **kwargs)
        except Exception as exc:
            self.failures += 1
            if _is_backend_exhaustion(exc):
                raise BackendExhausted(str(exc)) from exc
            raise


# ── Baselines ──────────────────────────────────────────────────────────────────

def run_random(problem: dict, backend: LLMBackend, budget: int, seed: int) -> MethodRun:
    rng = random.Random(seed)
    counter = _CountingBackend(backend)
    evaluate = make_evaluator(problem)
    task = _problem_task(problem)
    t0 = time.monotonic()
    best = 0.0
    curve: list[float] = []
    for _ in range(budget):
        agent = Agent(genome=_random_genome(rng), backend=counter)
        score = evaluate(agent, task)
        best = max(best, score)
        curve.append(best)
    return MethodRun("random", best, budget, counter.calls, time.monotonic() - t0,
                     curve, counter.failures)


def run_hill(problem: dict, backend: LLMBackend, budget: int, seed: int) -> MethodRun:
    rng = random.Random(seed)
    counter = _CountingBackend(backend)
    evaluate = make_evaluator(problem)
    task = _problem_task(problem)
    t0 = time.monotonic()

    current = _random_genome(rng)
    current_fit = evaluate(Agent(genome=current, backend=counter), task)
    best = current_fit
    curve = [best]

    for _ in range(budget - 1):
        cand = Genome.from_dict(current.to_dict())
        # Perturb exactly one attribute.
        choice = rng.randint(0, 2)
        if choice == 0:
            cand.system_prompt = rng.choice(_PROMPT_POOL)
        elif choice == 1:
            cand.strategy = rng.choice(_STRATEGIES)
        else:
            cand.temperature = round(
                max(0.1, min(1.0, cand.temperature + rng.uniform(-0.3, 0.3))), 2
            )
        cand_fit = evaluate(Agent(genome=cand, backend=counter), task)
        if cand_fit >= current_fit:  # keep if not worse (plateau-friendly)
            current, current_fit = cand, cand_fit
        best = max(best, cand_fit)
        curve.append(best)
    return MethodRun("hill", best, budget, counter.calls, time.monotonic() - t0,
                     curve, counter.failures)


def run_cambrian(
    problem: dict, backend: LLMBackend, population: int, generations: int, seed: int
) -> MethodRun:
    counter = _CountingBackend(backend)
    evaluate = make_evaluator(problem)
    task = _problem_task(problem)
    t0 = time.monotonic()

    curve: list[float] = []
    best_so_far = [0.0]

    def tracking_eval(agent: Agent, t: str) -> float:
        score = evaluate(agent, t)
        best_so_far[0] = max(best_so_far[0], score)
        curve.append(best_so_far[0])
        return score

    engine = EvolutionEngine(
        evaluator=tracking_eval,
        mutator=LLMMutator(backend=counter, mutation_temperature=0.7),
        backend=counter,
        population_size=population,
        seed=seed,
    )
    best = engine.evolve(
        seed_genomes=[_random_genome(random.Random(seed))],
        task=task,
        n_generations=generations,
    )
    n_evals = len(curve)
    return MethodRun(
        "cambrian",
        float(best.fitness or 0.0),
        n_evals,
        counter.calls,
        time.monotonic() - t0,
        curve,
        counter.failures,
    )


# ── Orchestration ──────────────────────────────────────────────────────────────

def _resolve_backend() -> tuple[LLMBackend, str, bool]:
    choice = os.getenv("CAMBRIAN_BACKEND", "gemini").strip().lower()
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if choice == "gemini" and has_key:
        from cambrian.backends.gemini import GeminiBackend

        model = os.getenv("CAMBRIAN_GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiBackend(model=model), model, False
    from cambrian.backends.mock import MockBackend

    if choice == "gemini" and not has_key:
        print(
            "WARNING: CAMBRIAN_BACKEND=gemini but no GEMINI_API_KEY set.\n"
            "         Falling back to the MOCK backend — results validate the\n"
            "         harness ONLY and are NOT evidence of real-world value.\n"
            "         Get a free key: https://aistudio.google.com/apikey\n",
            file=sys.stderr,
        )
    return MockBackend(), "mock", True


def _preflight(backend: LLMBackend) -> None:
    """One probe call. If the backend is unauthorised / quota-blocked from the
    start, abort with a clear message instead of producing a fake all-zero run."""
    try:
        backend.generate("Reply with the single word OK.", temperature=0, max_tokens=5)
    except Exception as exc:  # noqa: BLE001
        if _is_backend_exhaustion(RuntimeError(str(exc))):
            print(
                "\nABORT — backend unusable on the very first call:\n"
                f"  {str(exc)[:300]}\n\n"
                "  The key authenticates but has no usable quota for this model.\n"
                "  Fix: generate a standard AI Studio key (https://aistudio.google.com/apikey)\n"
                "  on a project with the normal free tier, or enable billing. Then re-run.\n",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        # Other errors: surface but let the run proceed (could be transient).
        print(f"(preflight warning: {str(exc)[:200]})", file=sys.stderr)


def _abort_quota(
    date: str, label: str, reason: str, problems_done: int, rows: list[dict]
) -> None:
    """Write an explicit ABORTED artifact (never a misleading verdict) and report."""
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"ABORTED — backend exhausted after {problems_done} complete problem(s).",
          file=sys.stderr)
    print(f"Reason: {reason}", file=sys.stderr)
    print("This is NOT a verdict. The quota ran out mid-run, so every blocked\n"
          "call would have scored a fake 0.0. Re-run with a key that has enough\n"
          "daily quota (some projects cap the free tier at 20 requests/day/model).",
          file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"humaneval_{date}_ABORTED.json"
    path.write_text(json.dumps({
        "status": "ABORTED_BACKEND_EXHAUSTED",
        "backend": label,
        "reason": reason,
        "problems_completed": problems_done,
        "partial_rows": rows,
        "note": "Not a verdict. Quota ran out mid-run; results are invalid.",
    }, indent=2))
    print(f"Wrote {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cambrian convergence benchmark")
    parser.add_argument("--problems", type=int, default=5, help="number of problems (max %d)" % len(PROBLEMS))
    parser.add_argument("--budget", type=int, default=24, help="evaluation budget per method per problem")
    parser.add_argument("--population", type=int, default=6, help="GA population size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", type=str, default="unset", help="date tag for output files (YYYY_MM_DD)")
    args = parser.parse_args(argv)

    problems = PROBLEMS[: max(1, min(args.problems, len(PROBLEMS)))]
    # GA budget = population * (generations+1 initial). Match the baseline budget.
    generations = max(1, args.budget // args.population - 1)
    ga_evals = args.population * (generations + 1)

    backend, label, is_mock = _resolve_backend()

    # Up-front budget disclosure (honesty: no silent cost).
    per_problem = args.budget * 2 + ga_evals  # random + hill + GA evals
    # GA also spends mutation/crossover LLM calls (~ one per non-elite child).
    est_calls = len(problems) * (per_problem + ga_evals)
    print(f"Backend: {label}{'  (MOCK - not real evidence)' if is_mock else ''}")
    print(f"Problems: {len(problems)} | budget/method: {args.budget} evals | "
          f"GA: pop={args.population} gens={generations} ({ga_evals} evals)")
    print(f"Estimated LLM calls: ~{est_calls} "
          f"(needs a key whose free-tier/daily quota >= this; some projects cap at 20/day)\n")

    if not is_mock:
        _preflight(backend)

    results: dict[str, list[MethodRun]] = {"random": [], "hill": [], "cambrian": []}
    per_problem_rows: list[dict[str, Any]] = []

    for i, problem in enumerate(problems):
        seed = args.seed + i
        print(f"[{i+1}/{len(problems)}] {problem['task_id']} ...", flush=True)
        try:
            r_rand = run_random(problem, backend, args.budget, seed)
            r_hill = run_hill(problem, backend, args.budget, seed)
            r_camb = run_cambrian(problem, backend, args.population, generations, seed)
        except BackendExhausted as exc:
            _abort_quota(args.date, label, str(exc), i, per_problem_rows)
            return 2
        # A method that silently logged backend failures (e.g. GA, whose engine
        # swallows evaluator exceptions) also invalidates the comparison.
        total_fail = r_rand.n_failures + r_hill.n_failures + r_camb.n_failures
        if total_fail > 0:
            _abort_quota(
                args.date, label,
                f"{total_fail} backend calls failed (quota/rate-limit) on "
                f"{problem['task_id']} — comparison invalid",
                i, per_problem_rows,
            )
            return 2
        for r in (r_rand, r_hill, r_camb):
            results[r.method].append(r)
        per_problem_rows.append({
            "task_id": problem["task_id"],
            "random_pass": r_rand.best_fitness,
            "hill_pass": r_hill.best_fitness,
            "cambrian_pass": r_camb.best_fitness,
            "random_calls": r_rand.n_llm_calls,
            "hill_calls": r_hill.n_llm_calls,
            "cambrian_calls": r_camb.n_llm_calls,
        })
        print(f"    random={r_rand.best_fitness:.0f} hill={r_hill.best_fitness:.0f} "
              f"cambrian={r_camb.best_fitness:.0f}")

    def pass_at_1(method: str) -> float:
        runs = results[method]
        return sum(r.best_fitness for r in runs) / len(runs) if runs else 0.0

    summary = {
        "backend": label,
        "is_mock": is_mock,
        "n_problems": len(problems),
        "budget_per_method": args.budget,
        "ga_population": args.population,
        "ga_generations": generations,
        "pass_at_1": {m: pass_at_1(m) for m in results},
        "total_llm_calls": {m: sum(r.n_llm_calls for r in results[m]) for m in results},
        "mean_wall_s": {m: sum(r.wall_s for r in results[m]) / len(results[m]) for m in results},
    }

    rand_p, camb_p = summary["pass_at_1"]["random"], summary["pass_at_1"]["cambrian"]
    delta = camb_p - rand_p
    if is_mock:
        verdict = "MOCK RUN — harness validated, NOT evidence. Re-run with GEMINI_API_KEY."
    elif delta > 0.15:
        verdict = f"GREEN — Cambrian beats random by {delta*100:.0f}pp pass@1. Worth it."
    elif delta >= 0.05:
        verdict = f"YELLOW — Cambrian beats random by {delta*100:.0f}pp. Marginal; polish."
    else:
        verdict = f"RED — Cambrian beats random by only {delta*100:.0f}pp. Archive."
    summary["verdict"] = verdict

    print("\n" + "=" * 60)
    print(f"pass@1  random={rand_p:.2f}  hill={summary['pass_at_1']['hill']:.2f}  "
          f"cambrian={camb_p:.2f}")
    print(verdict)
    print("=" * 60)

    _write_outputs(args.date, summary, per_problem_rows, results)
    return 0


def _write_outputs(
    date: str,
    summary: dict,
    rows: list[dict],
    results: dict[str, list[MethodRun]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"humaneval_{date}"

    json_path = RESULTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "per_problem": rows,
        "convergence": {
            m: [r.convergence for r in runs] for m, runs in results.items()
        },
    }, indent=2))
    print(f"Wrote {json_path}")

    csv_path = RESULTS_DIR / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    _plot_convergence(RESULTS_DIR / f"{stem}.png", results)


def _plot_convergence(path: Path, results: dict[str, list[MethodRun]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(skipping plot — matplotlib unavailable: {exc})")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, runs in results.items():
        if not runs:
            continue
        max_len = max(len(r.convergence) for r in runs)
        # Average best-so-far across problems, padding short curves with last value.
        avg = []
        for t in range(max_len):
            vals = [r.convergence[min(t, len(r.convergence) - 1)] for r in runs if r.convergence]
            avg.append(sum(vals) / len(vals))
        ax.plot(range(1, len(avg) + 1), avg, marker="o", markersize=3, label=method)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("best-so-far pass@1 (avg over problems)")
    ax.set_title("Convergence: Cambrian GA vs baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
