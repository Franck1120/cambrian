from __future__ import annotations
import io, json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
from cambrian.agent import Agent, Genome
from cambrian.evaluator import Evaluator
from cambrian.self_play import SelfPlayEvaluator, SelfPlayResult, TournamentRecord, run_tournament
from cambrian.world_model import WorldModel, WorldModelEvaluator, WorldModelPrediction, world_model_fitness
from cambrian.utils.logging import JSONLogger, load_json_log


def _agent(system_prompt: str = "Test", fitness: float | None = None) -> Agent:
    genome = Genome(system_prompt=system_prompt, temperature=0.5)
    agent = Agent(genome=genome)
    if fitness is not None:
        agent.fitness = fitness
    return agent


class _ConstEvaluator(Evaluator):
    def __init__(self, score: float = 0.5) -> None:
        self._score = score
    def evaluate(self, agent: Agent, task: str) -> float:
        return self._score


class _PromptLenEvaluator(Evaluator):
    def evaluate(self, agent: Agent, task: str) -> float:
        return min(1.0, len(agent.genome.system_prompt) / 200.0)


# ── SelfPlayResult ─────────────────────────────────────────────────────────────


class TestSelfPlayResult:
    def test_winner_fields(self) -> None:
        r = SelfPlayResult("a", "b", 0.8, 0.6, "a", 0.2, "task")
        assert r.winner_id == "a" and not r.is_draw and r.loser_id() == "b"

    def test_draw(self) -> None:
        r = SelfPlayResult("a", "b", 0.7, 0.7, None, 0.0, "task")
        assert r.is_draw and r.loser_id() is None

    def test_loser_when_a_wins(self) -> None:
        r = SelfPlayResult("a", "b", 0.9, 0.3, "a", 0.6, "task")
        assert r.loser_id() == "b"

    def test_loser_when_b_wins(self) -> None:
        r = SelfPlayResult("a", "b", 0.3, 0.9, "b", 0.6, "task")
        assert r.loser_id() == "a"


# ── SelfPlayEvaluator ──────────────────────────────────────────────────────────


class TestSelfPlayEvaluator:
    def test_evaluate_delegates(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.7))
        assert sp.evaluate(_agent(), "task") == pytest.approx(0.7)

    def test_winner_gets_bonus(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_PromptLenEvaluator(), win_bonus=0.1, loss_penalty=0.05)
        a, b = _agent("A" * 100), _agent("B" * 40)
        result = sp.compete(a, b, "task")
        assert result.winner_id == a.id and (a.fitness or 0.0) > result.score_a

    def test_loser_gets_penalty(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_PromptLenEvaluator(), win_bonus=0.1, loss_penalty=0.05)
        a, b = _agent("A" * 100), _agent("B" * 40)
        result = sp.compete(a, b, "task")
        assert (b.fitness or 0.0) < result.score_b

    def test_draw(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.5), draw_threshold=1.0)
        assert sp.compete(_agent(), _agent(), "task").is_draw

    def test_pre_scores(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.0))
        result = sp.compete(_agent(), _agent(), "task", score_a=0.9, score_b=0.3)
        assert result.score_a == pytest.approx(0.9) and result.score_b == pytest.approx(0.3)

    def test_fitness_clipped_max(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.99), win_bonus=0.5, max_score=1.0)
        a, b = _agent(), _agent()
        sp.compete(a, b, "task", score_a=0.99, score_b=0.5)
        assert (a.fitness or 0.0) <= 1.0

    def test_fitness_clipped_min(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.01), loss_penalty=0.5, min_score=0.0)
        a, b = _agent(), _agent()
        sp.compete(a, b, "task", score_a=0.01, score_b=0.8)
        assert (a.fitness or 0.0) >= 0.0

    def test_returns_selfplayresult(self) -> None:
        sp = SelfPlayEvaluator(base_evaluator=_ConstEvaluator(0.6))
        assert isinstance(sp.compete(_agent(), _agent(), "task"), SelfPlayResult)


# ── run_tournament ─────────────────────────────────────────────────────────────


class TestRunTournament:
    def _pop(self, n: int) -> list[Agent]:
        return [_agent("A" * (10 * (i + 1))) for i in range(n)]

    def test_returns_record(self) -> None:
        assert isinstance(run_tournament(self._pop(3), SelfPlayEvaluator(_PromptLenEvaluator()), "t"), TournamentRecord)

    def test_match_count(self) -> None:
        n = 4
        record = run_tournament(self._pop(n), SelfPlayEvaluator(_PromptLenEvaluator()), "t")
        assert record.match_count() == n * (n - 1) // 2

    def test_all_agents_have_stats(self) -> None:
        pop = self._pop(4)
        record = run_tournament(pop, SelfPlayEvaluator(_PromptLenEvaluator()), "t")
        for a in pop:
            assert a.id in record.wins and a.id in record.losses

    def test_wins_losses_sum(self) -> None:
        n = 4
        pop = self._pop(n)
        record = run_tournament(pop, SelfPlayEvaluator(_PromptLenEvaluator()), "t")
        for a in pop:
            total = record.wins[a.id] + record.losses[a.id] + record.draws.get(a.id, 0)
            assert total == n - 1

    def test_ranking(self) -> None:
        pop = self._pop(3)
        record = run_tournament(pop, SelfPlayEvaluator(_PromptLenEvaluator()), "t")
        assert len(record.ranking()) == 3

    def test_win_rate_in_range(self) -> None:
        pop = self._pop(4)
        record = run_tournament(pop, SelfPlayEvaluator(_PromptLenEvaluator()), "t")
        for a in pop:
            assert 0.0 <= record.win_rate(a.id) <= 1.0

    def test_pre_scores(self) -> None:
        pop = self._pop(3)
        sp = SelfPlayEvaluator(_ConstEvaluator(0.0))
        pre = {a.id: float(i) / 10 for i, a in enumerate(pop)}
        assert run_tournament(pop, sp, "t", pre_scores=pre).match_count() == 3

    def test_single_agent(self) -> None:
        assert run_tournament(self._pop(1), SelfPlayEvaluator(_ConstEvaluator(0.5)), "t").match_count() == 0

    def test_empty(self) -> None:
        assert run_tournament([], SelfPlayEvaluator(_ConstEvaluator(0.5)), "t").match_count() == 0


# ── HyperParams ────────────────────────────────────────────────────────────────


class TestHyperParams:
    def test_defaults(self) -> None:
        from cambrian.meta_evolution import HyperParams
        hp = HyperParams()
        assert 0.0 <= hp.mutation_rate <= 1.0 and hp.temperature > 0.0 and hp.tournament_k >= 1

    def test_clamp(self) -> None:
        from cambrian.meta_evolution import HyperParams
        hp = HyperParams(mutation_rate=2.0, crossover_rate=-0.5, temperature=-1.0).clamp()
        assert 0.0 <= hp.mutation_rate <= 1.0 and hp.temperature >= 0.05

    def test_perturb_in_bounds(self) -> None:
        from cambrian.meta_evolution import HyperParams
        import random as _rnd
        rng = _rnd.Random(42)
        hp = HyperParams()
        for _ in range(20):
            p = hp.perturb(scale=0.1, rng=rng)
            assert 0.0 <= p.mutation_rate <= 1.0 and p.temperature >= 0.05 and p.tournament_k >= 1

    def test_roundtrip(self) -> None:
        from cambrian.meta_evolution import HyperParams
        hp = HyperParams(mutation_rate=0.7, crossover_rate=0.2, temperature=0.55)
        r = HyperParams.from_dict(hp.to_dict())
        assert r.mutation_rate == pytest.approx(0.7) and r.temperature == pytest.approx(0.55)

    def test_repr(self) -> None:
        from cambrian.meta_evolution import HyperParams
        assert "HyperParams" in repr(HyperParams())

    def test_empty_history(self) -> None:
        from cambrian.meta_evolution import HyperParams
        assert HyperParams().fitness_history == []


# ── MetaEvolutionEngine ────────────────────────────────────────────────────────


class TestMetaEvolutionEngine:
    def _make(self) -> Any:
        from cambrian.meta_evolution import HyperParams, MetaEvolutionEngine
        from cambrian.mutator import LLMMutator
        backend = MagicMock()
        backend.model_name = "mock"
        backend.generate.return_value = (
            '{"system_prompt": "improved", "strategy": "s", '
            '"temperature": 0.5, "max_tokens": 512, "few_shot_examples": [], "tool_specs": []}'
        )
        return MetaEvolutionEngine(
            evaluator=_ConstEvaluator(0.6),
            mutator=LLMMutator(backend=backend, fallback_on_error=True),
            backend=backend,
            initial_hp=HyperParams(),
            meta_lr=0.05,
            n_candidates=2,
            population_size=4,
            seed=42,
        )

    def test_evolve_returns_agent(self) -> None:
        engine = self._make()
        best = engine.evolve([Genome(system_prompt=".")], "t", n_generations=2)
        assert isinstance(best, Agent) and best.fitness is not None

    def test_hp_history_grows(self) -> None:
        engine = self._make()
        engine.evolve([Genome(system_prompt=".")], "t", n_generations=4, meta_interval=2)
        assert len(engine.hp_history) >= 1

    def test_fitness_history(self) -> None:
        engine = self._make()
        engine.evolve([Genome(system_prompt=".")], "t", n_generations=3)
        assert len(engine.hp.fitness_history) > 0

    def test_callback(self) -> None:
        from cambrian.meta_evolution import HyperParams
        engine = self._make()
        calls: list[int] = []
        engine.evolve([Genome(system_prompt=".")], "t", n_generations=2,
                      on_generation=lambda g, p, hp: calls.append(g))
        assert len(calls) == 3


# ── WorldModel ─────────────────────────────────────────────────────────────────


class TestWorldModel:
    def test_predict_empty(self) -> None:
        pred = WorldModel(default_score=0.5).predict("task")
        assert pred.predicted_score == pytest.approx(0.5) and pred.confidence == pytest.approx(0.0)

    def test_update_predict(self) -> None:
        wm = WorldModel()
        wm.update("python code", 0.9)
        wm.update("python code", 0.85)
        pred = wm.predict("python code")
        assert pred.predicted_score > 0.0 and pred.confidence > 0.0

    def test_buffer_limit(self) -> None:
        wm = WorldModel(buffer_size=5)
        for i in range(10):
            wm.update(f"t{i}", float(i) / 10)
        assert wm.experience_count() == 5

    def test_experience_count(self) -> None:
        wm = WorldModel()
        assert wm.experience_count() == 0
        wm.update("t", 0.5)
        assert wm.experience_count() == 1

    def test_repr(self) -> None:
        assert "WorldModel" in repr(WorldModel())

    def test_is_uncertain(self) -> None:
        assert WorldModelPrediction(predicted_score=0.5, confidence=0.3).is_uncertain
        assert not WorldModelPrediction(predicted_score=0.5, confidence=0.8).is_uncertain

    def test_similarity_exact(self) -> None:
        assert WorldModel()._similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_similarity_none(self) -> None:
        assert WorldModel()._similarity("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_predict_in_range(self) -> None:
        wm = WorldModel()
        for i in range(5):
            wm.update("task", float(i) / 5)
        pred = wm.predict("task")
        assert 0.0 <= pred.predicted_score <= 1.0


class TestWorldModelFitness:
    def test_zero_weight(self) -> None:
        assert world_model_fitness(0.8, 0.5, 0.0) == pytest.approx(0.8)

    def test_full_weight_zero_error(self) -> None:
        assert world_model_fitness(0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_high_error_reduces(self) -> None:
        assert world_model_fitness(0.7, 0.0) > world_model_fitness(0.7, 0.5)

    def test_default_in_range(self) -> None:
        assert 0.0 <= world_model_fitness(0.8, 0.2) <= 1.0


class TestWorldModelEvaluator:
    def test_first_eval_raw(self) -> None:
        wm = WorldModelEvaluator(_ConstEvaluator(0.75), min_confidence_for_blend=0.1)
        assert wm.evaluate(_agent(), "task") == pytest.approx(0.75)

    def test_model_per_agent(self) -> None:
        wm = WorldModelEvaluator(_ConstEvaluator(0.6))
        wm.evaluate(_agent("A"), "t")
        wm.evaluate(_agent("B"), "t")
        assert wm.model_count() == 2

    def test_model_updates(self) -> None:
        wm = WorldModelEvaluator(_ConstEvaluator(0.7))
        a = _agent()
        for _ in range(3):
            wm.evaluate(a, "task")
        assert wm.get_model(a.id) is not None
        assert wm.get_model(a.id).experience_count() == 3  # type: ignore[union-attr]

    def test_get_unknown(self) -> None:
        assert WorldModelEvaluator(_ConstEvaluator(0.5)).get_model("x") is None

    def test_count_zero(self) -> None:
        assert WorldModelEvaluator(_ConstEvaluator(0.5)).model_count() == 0

    def test_score_in_range(self) -> None:
        wm = WorldModelEvaluator(_ConstEvaluator(0.8))
        a = _agent()
        for _ in range(5):
            assert 0.0 <= wm.evaluate(a, "task") <= 1.0

    def test_errors_empty(self) -> None:
        assert WorldModelEvaluator(_ConstEvaluator(0.5)).prediction_errors() == {}

    def test_errors_after_evals(self) -> None:
        wm = WorldModelEvaluator(_ConstEvaluator(0.6))
        a = _agent()
        for _ in range(3):
            wm.evaluate(a, "task")
        errors = wm.prediction_errors()
        assert a.id in errors and isinstance(errors[a.id], float)


# ── JSONLogger ─────────────────────────────────────────────────────────────────


class TestJSONLogger:
    def test_log_to_stream(self) -> None:
        buf = io.StringIO()
        JSONLogger(output=buf, run_id="r").log_generation(0, [0.5, 0.6])
        entry = json.loads(buf.getvalue().strip())
        assert entry["run_id"] == "r" and entry["generation"] == 0

    def test_fields(self) -> None:
        buf = io.StringIO()
        entry = JSONLogger(output=buf).log_generation(1, [0.4, 0.6, 0.8], best_agent_id="abc", best_prompt_len=100)
        assert entry["best_fitness"] == pytest.approx(0.8)
        assert entry["min_fitness"] == pytest.approx(0.4)
        assert entry["best_agent_id"] == "abc"
        assert entry["population_size"] == 3

    def test_extra_fields(self) -> None:
        buf = io.StringIO()
        entry = JSONLogger(output=buf).log_generation(0, [0.5], diversity=0.42)
        assert entry["extra"]["diversity"] == pytest.approx(0.42)

    def test_summary(self) -> None:
        buf = io.StringIO()
        entry = JSONLogger(output=buf, run_id="r").log_run_summary(10, 0.95, best_agent_id="xyz")
        assert entry["event"] == "run_complete" and entry["n_generations"] == 10

    def test_ndjson(self) -> None:
        buf = io.StringIO()
        lg = JSONLogger(output=buf)
        lg.log_generation(0, [0.4])
        lg.log_generation(1, [0.6])
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 2
        for line in lines:
            json.loads(line)

    def test_context_manager(self, tmp_path: Path) -> None:
        p = tmp_path / "run.json"
        with JSONLogger(output=p, run_id="ctx") as lg:
            lg.log_generation(0, [0.5])
        assert p.exists() and json.loads(p.read_text().strip())["run_id"] == "ctx"

    def test_append_mode(self, tmp_path: Path) -> None:
        p = tmp_path / "run.json"
        with JSONLogger(output=p, run_id="r") as lg:
            lg.log_generation(0, [0.5])
        with JSONLogger(output=p, run_id="r") as lg:
            lg.log_generation(1, [0.6])
        assert len([ln for ln in p.read_text().splitlines() if ln.strip()]) == 2

    def test_repr(self) -> None:
        assert "myrun" in repr(JSONLogger(output=io.StringIO(), run_id="myrun"))

    def test_empty_fitnesses(self) -> None:
        buf = io.StringIO()
        entry = JSONLogger(output=buf).log_generation(0, [])
        assert entry["best_fitness"] == pytest.approx(0.0) and entry["population_size"] == 0

    def test_std_single(self) -> None:
        buf = io.StringIO()
        entry = JSONLogger(output=buf).log_generation(0, [0.7])
        assert entry["std_fitness"] == pytest.approx(0.0)


class TestLoadJsonLog:
    def test_load(self, tmp_path: Path) -> None:
        p = tmp_path / "log.json"
        with JSONLogger(output=p, run_id="r") as lg:
            lg.log_generation(0, [0.5])
            lg.log_generation(1, [0.7])
        assert len(load_json_log(p)) == 2

    def test_skips_malformed(self, tmp_path: Path) -> None:
        p = tmp_path / "log.json"
        p.write_text('{"ok": true}\nnot valid\n{"ok": true}\n')
        assert len(load_json_log(p)) == 2

    def test_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "log.json"
        p.write_text("")
        assert load_json_log(p) == []


# ── compare CLI ────────────────────────────────────────────────────────────────


def _write_run(path: Path, run_id: str, fitnesses: list[float]) -> None:
    with JSONLogger(output=path, run_id=run_id) as lg:
        for i, f in enumerate(fitnesses):
            lg.log_generation(i, [f])
    with JSONLogger(output=path, run_id=run_id) as lg:
        lg.log_run_summary(len(fitnesses), max(fitnesses))


class TestCompareCLI:
    def test_text_output(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        _write_run(a, "run_a", [0.3, 0.5, 0.7])
        _write_run(b, "run_b", [0.2, 0.4, 0.6])
        result = CliRunner().invoke(cli, ["compare", str(a), str(b)])
        assert result.exit_code == 0, result.output

    def test_json_output(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        _write_run(a, "run_a", [0.5, 0.8])
        _write_run(b, "run_b", [0.4, 0.7])
        result = CliRunner().invoke(cli, ["compare", str(a), str(b), "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "winner" in parsed and "run_a" in parsed

    def test_identifies_winner(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        _write_run(a, "run_a", [0.9, 0.95])
        _write_run(b, "run_b", [0.1, 0.2])
        result = CliRunner().invoke(cli, ["compare", str(a), str(b), "--format", "json"])
        assert result.exit_code == 0 and json.loads(result.output)["winner"] == "run_a"

    def test_missing_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        result = CliRunner().invoke(cli, ["compare", str(tmp_path / "x.json"), str(tmp_path / "y.json")])
        assert result.exit_code != 0

    def test_custom_metric(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from cambrian.cli import main as cli
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        _write_run(a, "run_a", [0.5, 0.6])
        _write_run(b, "run_b", [0.4, 0.5])
        result = CliRunner().invoke(cli, ["compare", str(a), str(b), "--metric", "mean_fitness", "--format", "json"])
        assert result.exit_code == 0 and json.loads(result.output)["metric"] == "mean_fitness"
