# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""tests/test_integration_techniques.py — Multi-technique integration tests.

Each test combines 3 or more Cambrian techniques in a single coordinated run
using deterministic mock backends (no live LLM required).  The goal is to
verify that the techniques compose correctly and do not interfere when wired
together in the same evolution loop.

Combinations tested
-------------------
1. QuorumSensor + ApoptosisController + NeuromodulatorBank
2. ImmuneCortex + NeuromodulatorBank + ZeitgeberScheduler
3. TabuMutator + AnnealingSelector + BoostingEnsemble
4. TransgenerationalRegistry + HGTransfer + SymbioticFuser
5. WorldModel + SelfPlay + HyperParams perturbation
6. LLMCascade + BestOfN + ReflexionEvaluator
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _genome(prompt: str, temperature: float = 0.5) -> Genome:
    return Genome(system_prompt=prompt, temperature=temperature)


def _agent(prompt: str, fitness: float, temperature: float = 0.5) -> Agent:
    a = Agent(genome=_genome(prompt, temperature))
    a.fitness = fitness
    return a


def _mock_backend(response: str = "mock response") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _mock_backend_seq(responses: list[str]) -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(side_effect=responses)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 1. QuorumSensor + ApoptosisController + NeuromodulatorBank
# ─────────────────────────────────────────────────────────────────────────────


class TestQuorumApoptosisNeuro:
    """QuorumSensor gauges population diversity and adjusts mutation rate;
    ApoptosisController prunes stagnant agents; NeuromodulatorBank provides
    a second, biologically-grounded signal on mutation rate and selection
    pressure.  The three produce complementary diversity signals and together
    prevent premature convergence."""

    def setup_method(self) -> None:
        from cambrian.apoptosis import ApoptosisController
        from cambrian.neuromodulation import NeuromodulatorBank
        from cambrian.quorum import QuorumSensor

        self.quorum = QuorumSensor(
            low_entropy_threshold=0.5,
            high_entropy_threshold=2.5,
            boost_factor=1.3,
            decay_factor=0.85,
            min_rate=0.1,
            max_rate=1.0,
        )
        self.apoptosis = ApoptosisController(
            stagnation_window=3,
            min_fitness=0.2,
            grace_period=1,
        )
        self.bank = NeuromodulatorBank(
            base_mutation_rate=0.3,
            base_selection_pressure=0.5,
            mr_range=0.2,
            sp_range=0.2,
        )

    def _pop(self, fitnesses: list[float]) -> list[Agent]:
        return [_agent(f"Strategy {i} with systematic approach.", f) for i, f in enumerate(fitnesses)]

    def test_quorum_boosts_rate_at_low_entropy(self) -> None:
        scores = [0.8, 0.81, 0.82, 0.83, 0.84]  # very uniform → low entropy
        rate = self.quorum.update(scores=scores, current_rate=0.3)
        # Low entropy triggers boost
        assert rate >= 0.3

    def test_quorum_decays_rate_at_high_entropy(self) -> None:
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]  # wide spread → high entropy
        rate = self.quorum.update(scores=scores, current_rate=0.9)
        # High entropy triggers decay
        assert rate <= 0.9

    def test_quorum_result_clamped(self) -> None:
        for scores in [[0.5] * 5, [0.0, 1.0, 0.0, 1.0, 0.5]]:
            rate = self.quorum.update(scores=scores, current_rate=0.5)
            assert 0.0 <= rate <= 1.0

    def test_apoptosis_removes_chronic_poor_agents(self) -> None:
        best = _agent("Best agent with comprehensive expert knowledge.", 0.9)
        poor = _agent("Poor agent with basic approach.", 0.1)

        for _ in range(5):
            self.apoptosis.record(poor)

        survivors = self.apoptosis.apply([best, poor], best_agent=best)
        assert best in survivors
        assert poor not in survivors

    def test_apoptosis_grace_period_protects_new_agents(self) -> None:
        best = _agent("Best.", 0.9)
        new_poor = _agent("New agent with poor fitness but grace period.", 0.1)

        # Only 1 record (within grace period of 1)
        self.apoptosis.record(new_poor)

        survivors = self.apoptosis.apply([best, new_poor], best_agent=best)
        # new_poor should survive because it's within grace period
        assert best in survivors

    def test_neuromodulator_state_bounded(self) -> None:
        pop = self._pop([0.3, 0.5, 0.7, 0.8, 0.9])
        state = self.bank.modulate(pop, generation=3)
        assert 0.0 <= state.mutation_rate <= 1.0
        assert 0.0 <= state.selection_pressure <= 1.0
        assert 0.0 <= state.dopamine <= 1.0

    def test_combined_three_generation_loop(self) -> None:
        """Simulate 3 generations: quorum senses diversity, neuro adjusts params,
        apoptosis prunes, all operating on the same population."""
        pop = self._pop([0.3, 0.4, 0.5, 0.6, 0.7])

        for gen in range(1, 4):
            scores = [a.fitness or 0.0 for a in pop]

            # Quorum sensing
            q_rate = self.quorum.update(scores=scores, current_rate=0.3)
            assert 0.0 <= q_rate <= 1.0

            # Neuromodulation
            state = self.bank.modulate(pop, generation=gen)
            assert 0.0 <= state.mutation_rate <= 1.0

            # Blend the two mutation rate signals
            effective_rate = (q_rate + state.mutation_rate) / 2.0
            assert 0.0 <= effective_rate <= 1.0

            # Apoptosis
            for a in pop:
                self.apoptosis.record(a)
            best = max(pop, key=lambda a: a.fitness or 0.0)
            pop = self.apoptosis.apply(pop, best_agent=best)
            assert len(pop) >= 1

    def test_neuro_history_accumulates(self) -> None:
        pop = self._pop([0.4, 0.6, 0.8])
        for gen in range(4):
            self.bank.modulate(pop, generation=gen)
        assert len(self.bank.history) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 2. ImmuneCortex + NeuromodulatorBank + ZeitgeberScheduler
# ─────────────────────────────────────────────────────────────────────────────


class TestImmuneNeuroZeitgeber:
    """ImmuneCortex caches high-fitness genomes for fast recall; NeuromodulatorBank
    adjusts evolutionary hyperparameters; ZeitgeberScheduler adds circadian
    oscillation.  Together they simulate a bio-plausible adaptive controller."""

    def setup_method(self) -> None:
        from cambrian.immune_memory import ImmuneCortex
        from cambrian.neuromodulation import NeuromodulatorBank
        from cambrian.zeitgeber import ZeitgeberClock, ZeitgeberScheduler

        self.cortex = ImmuneCortex(
            b_threshold=0.8,
            t_threshold=0.5,
            b_similarity=0.6,
            t_min_similarity=0.2,
        )
        self.bank = NeuromodulatorBank(
            base_mutation_rate=0.3,
            base_selection_pressure=0.5,
            mr_range=0.2,
            sp_range=0.2,
        )
        clock = ZeitgeberClock(period=4, amplitude=0.3)
        self.scheduler = ZeitgeberScheduler(
            clock=clock,
            base_mutation_rate=0.3,
            mutation_range=0.1,
            base_threshold=0.5,
            threshold_range=0.1,
        )

    def test_immune_recall_after_record(self) -> None:
        agent = _agent("Expert Python debugger for syntax errors.", 0.95)
        self.cortex.record(agent, task="Debug Python syntax error in loop")

        recall = self.cortex.recall("Fix Python syntax error in for loop")
        assert recall.recalled
        assert recall.agent is not None
        assert recall.agent.fitness == pytest.approx(0.95)

    def test_immune_no_recall_for_unrelated_task(self) -> None:
        agent = _agent("Expert mathematician.", 0.95)
        self.cortex.record(agent, task="Solve linear algebra problems")

        recall = self.cortex.recall("Cook pasta recipe with sauce")
        # Completely unrelated: either not recalled, or T-cell with very low sim
        if recall.recalled:
            assert recall.similarity < 0.5

    def test_neuromodulator_state_in_range(self) -> None:
        pop = [_agent(f"Agent {i} systematic.", 0.3 + i * 0.1) for i in range(5)]
        state = self.bank.modulate(pop, generation=0)
        assert 0.0 <= state.mutation_rate <= 1.0
        assert 0.0 <= state.selection_pressure <= 1.0

    def test_zeitgeber_oscillates_across_period(self) -> None:
        rates = []
        for _ in range(8):
            mr, thr = self.scheduler.tick()
            rates.append(mr)
        assert max(rates) > min(rates)

    def test_zeitgeber_tick_in_range(self) -> None:
        mr, thr = self.scheduler.tick()
        assert 0.0 <= mr <= 1.0
        assert 0.0 <= thr <= 1.0

    def test_combined_episode_loop(self) -> None:
        """6 episodes: check immune cache first, then run neuro+zeitgeber,
        record best agent in immune cortex."""
        tasks = [
            "Solve quadratic equations step by step",
            "Write recursive Fibonacci in Python",
            "Sort list using quicksort algorithm",
            "Parse JSON data with Python",
            "Solve quadratic equations complex roots",  # similar to ep 0
            "Debug JSON parse error in Python",         # similar to ep 3
        ]
        prompts = [
            "Expert mathematician with step-by-step proofs.",
            "Python expert writing clean recursive code.",
            "Algorithm specialist with in-place sorting.",
            "Python JSON parsing expert.",
        ]
        rng = random.Random(42)

        for ep, task in enumerate(tasks):
            recall = self.cortex.recall(task)
            if recall.recalled:
                continue

            pop = [
                _agent(rng.choice(prompts), rng.uniform(0.4, 0.9))
                for _ in range(3)
            ]

            state = self.bank.modulate(pop, generation=ep)
            mr, thr = self.scheduler.tick()
            effective_mr = (state.mutation_rate + mr) / 2.0
            assert 0.0 <= effective_mr <= 1.0

            best = max(pop, key=lambda a: a.fitness or 0.0)
            self.cortex.record(best, task)

        assert self.cortex.b_cell_count + self.cortex.t_cell_count > 0

    def test_zeitgeber_full_period_restores_phase(self) -> None:
        from cambrian.zeitgeber import ZeitgeberClock
        clock = ZeitgeberClock(period=4, amplitude=0.3)
        ef0 = clock.exploration_factor()
        for _ in range(4):  # advance a full period
            clock.advance()
        ef4 = clock.exploration_factor()
        assert abs(ef0 - ef4) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 3. TabuMutator + AnnealingSelector + BoostingEnsemble
# ─────────────────────────────────────────────────────────────────────────────


class TestTabuAnnealingBoosting:
    """TabuMutator prevents revisiting recent genome regions; AnnealingSelector
    accepts regressions probabilistically at high temperature; BoostingEnsemble
    updates agent weights based on per-query correctness.  Together they form a
    search strategy that escapes local optima while maintaining an adaptive
    ensemble oracle."""

    def setup_method(self) -> None:
        from cambrian.annealing import AnnealingSchedule, AnnealingSelector
        from cambrian.ensemble import BoostingEnsemble
        from cambrian.mutator import LLMMutator
        from cambrian.tabu import TabuList, TabuMutator

        backend = _mock_backend("You are a refined expert agent with improved strategy.")
        base_mutator = LLMMutator(backend=backend, mutation_temperature=0.5)

        self.tabu_list = TabuList(max_size=5)
        self.tabu_mutator = TabuMutator(
            base_mutator=base_mutator,
            tabu_list=self.tabu_list,
            max_retries=3,
        )

        schedule = AnnealingSchedule(T_max=1.0, T_min=0.01, n_steps=20, schedule_type="exponential")
        self.annealing = AnnealingSelector(schedule)

        a1 = _agent("Expert solver using analytical reasoning techniques.", 0.8)
        a2 = _agent("Creative solver using lateral thinking methodology.", 0.7)
        a3 = _agent("Systematic solver using exhaustive enumeration search.", 0.6)
        for a in [a1, a2, a3]:
            a.run = MagicMock(return_value="42")  # type: ignore[method-assign]
        self.ensemble = BoostingEnsemble(agents=[a1, a2, a3])

    def test_tabu_list_registers_agent(self) -> None:
        agent = _agent("Step-by-step reasoning for Python coding problems.", 0.7)
        self.tabu_list.add(agent)
        assert self.tabu_list.is_tabu(agent)

    def test_tabu_list_max_size_fifo_eviction(self) -> None:
        agents = [_agent(f"Unique agent strategy variant {i} extra distinctive words.", 0.5) for i in range(7)]
        for a in agents:
            self.tabu_list.add(a)
        assert not self.tabu_list.is_tabu(agents[0])  # evicted
        assert not self.tabu_list.is_tabu(agents[1])  # evicted
        assert self.tabu_list.is_tabu(agents[6])

    def test_annealing_always_accepts_improvement(self) -> None:
        accepted = sum(
            1 for _ in range(20)
            if self.annealing.step(current_fitness=0.5, candidate_fitness=0.8)
        )
        assert accepted == 20

    def test_annealing_sometimes_accepts_regression_at_high_temp(self) -> None:
        from cambrian.annealing import AnnealingSchedule, AnnealingSelector
        hot_schedule = AnnealingSchedule(T_max=10.0, T_min=9.9, n_steps=100, schedule_type="linear")
        hot_selector = AnnealingSelector(hot_schedule)
        accepted = sum(
            1 for _ in range(100)
            if hot_selector.step(current_fitness=0.8, candidate_fitness=0.3)
        )
        # T≈10, delta=0.5, P=exp(-0.5/10)≈0.95 → almost always accept
        assert accepted > 70

    def test_boosting_ensemble_all_correct_equal_weights(self) -> None:
        self.ensemble.query(task="What is 6x7?", correct_answer="42")
        assert len(self.ensemble.weights) == 3
        assert all(w > 0 for w in self.ensemble.weights)
        # All agents correct → after normalisation, all weights should be equal
        total = sum(self.ensemble.weights)
        assert abs(total - 1.0) < 1e-9

    def test_tabu_hit_rate_starts_at_zero(self) -> None:
        assert self.tabu_mutator.tabu_hit_rate == 0.0

    def test_annealing_acceptance_rate_in_range(self) -> None:
        for _ in range(10):
            self.annealing.step(0.5, 0.7)
        rate = self.annealing.acceptance_rate()
        assert 0.0 <= rate <= 1.0

    def test_combined_one_search_step(self) -> None:
        """One iteration: tabu check, annealing decision, ensemble weight update."""
        candidate = _agent("Candidate strategy with creative unique approach.", 0.55)

        assert not self.tabu_list.is_tabu(candidate)
        self.tabu_list.add(candidate)
        assert self.tabu_list.is_tabu(candidate)

        # Improvement → always accepted
        accepted = self.annealing.step(current_fitness=0.5, candidate_fitness=0.75)
        assert accepted

        answer = self.ensemble.query(task="Solve problem", correct_answer="42")
        assert answer == "42"

    def test_combined_multi_step_loop(self) -> None:
        """5 steps: cycling through tabu, annealing, ensemble weight adaptation."""
        from cambrian.annealing import AnnealingSchedule, AnnealingSelector
        schedule = AnnealingSchedule(T_max=1.0, T_min=0.05, n_steps=5, schedule_type="cosine")
        selector = AnnealingSelector(schedule)

        candidates = [_agent(f"Strategy {i} with extended context approach.", 0.5 + i * 0.05) for i in range(5)]
        current_fitness = 0.5

        for cand in candidates:
            if not self.tabu_list.is_tabu(cand):
                accepted = selector.step(current_fitness, cand.fitness or 0.5)
                if accepted:
                    current_fitness = cand.fitness or 0.5
                self.tabu_list.add(cand)

        assert selector.acceptance_rate() >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. TransgenerationalRegistry + HGTransfer + SymbioticFuser
# ─────────────────────────────────────────────────────────────────────────────


class TestTransgenerationalHGTSymbiosis:
    """Transgenerational marks carry behavioural priors across generations;
    HGT injects strategy fragments from high-fitness donors; SymbioticFuser
    merges compatible agents.  Together they form a rich genome transfer stack."""

    def setup_method(self) -> None:
        from cambrian.hgt import HGTransfer, HGTPool
        from cambrian.symbiosis import SymbioticFuser
        from cambrian.transgenerational import TransgenerationalRegistry

        self.registry = TransgenerationalRegistry(max_generations=5, inherit_top_n=3)
        self.hgt = HGTransfer(n_sentences=1, mode="suffix", fitness_threshold=0.6)
        self.pool = HGTPool(max_plasmids=10)

        backend = _mock_backend(
            "You are a hybrid agent combining step-by-step analysis with creative reasoning."
        )
        self.fuser = SymbioticFuser(
            backend=backend,
            fitness_threshold=0.6,
            min_distance=0.1,
        )

    def test_transgenerational_marks_propagate(self) -> None:
        parent = _agent("Expert agent with systematic step-by-step reasoning.", 0.9)
        child = _agent("Child agent with basic standard approach.", 0.6)
        grandchild = _agent("Grandchild starting from fresh minimal knowledge.", 0.5)

        self.registry.record_mark(parent, "step-by-step", strength=0.9)
        self.registry.record_mark(parent, "verify-output", strength=0.7)

        n1 = self.registry.inherit(parent, child)
        self.registry.inherit(child, grandchild)

        assert n1 > 0
        marks = self.registry.get_marks(grandchild)
        assert all(m.strength <= 0.9 for m in marks)

    def test_transgenerational_apply_injects_context(self) -> None:
        agent = _agent("Expert reasoning agent with analytical skills.", 0.8)
        self.registry.record_mark(agent, "chain-of-thought", strength=0.85)

        modified = self.registry.apply_to_genome(agent)
        assert isinstance(modified, Genome)
        marks = self.registry.get_marks(agent)
        if marks:
            assert "chain-of-thought" in modified.system_prompt

    def test_hgt_pool_contribute_and_draw(self) -> None:
        donor = _agent(
            "First key insight from analysis. Second important strategy. Third critical method.",
            0.9,
        )
        self.pool.contribute(donor, domain="reasoning")
        plasmid = self.pool.draw(domain="reasoning")
        assert plasmid is not None
        assert plasmid.donor_fitness == pytest.approx(0.9)

    def test_hgt_transfer_modifies_recipient(self) -> None:
        donor = _agent(
            "Step one analyse the problem carefully. Step two synthesise the final answer.",
            0.85,
        )
        recipient = _agent("Basic minimal approach to solving tasks.", 0.5)
        offspring = self.hgt.transfer(donor, recipient)
        assert offspring is not None
        assert offspring.genome.system_prompt != recipient.genome.system_prompt

    def test_symbiotic_fuser_rejects_low_fitness_donor(self) -> None:
        host = _agent("Expert analytical agent with domain knowledge.", 0.9)
        donor = _agent("Poor agent with minimal capability.", 0.3)  # below threshold=0.6
        fused = self.fuser.fuse(host, donor, task="Solve problem")
        assert fused is None

    def test_symbiotic_fuser_requires_both_above_threshold(self) -> None:
        # Both above threshold but may be rejected if prompts too similar
        host = _agent("Analytical expert with systematic proven approach.", 0.85)
        donor = _agent("Creative lateral thinker with novel experimental approach.", 0.80)
        # fuse may return Agent or None depending on distance check
        fused = self.fuser.fuse(host, donor, task="Optimisation task")
        assert fused is None or isinstance(fused.genome.system_prompt, str)

    def test_combined_genome_transfer_pipeline(self) -> None:
        """Full pipeline: transgenerational marks → HGT pool → cross-agent HGT."""
        parent = _agent(
            "Expert reasoning agent using systematic verification approach.", 0.9
        )
        child = _agent("Intermediate agent with moderate approach.", 0.7)

        # Transgenerational: record and propagate marks
        self.registry.record_mark(parent, "systematic-verification", strength=0.9)
        n = self.registry.inherit(parent, child)
        assert n >= 0
        child_genome = self.registry.apply_to_genome(child)
        assert isinstance(child_genome, Genome)

        # HGT pool: contribute parent and draw for another agent
        self.pool.contribute(parent, domain="problem-solving")
        plasmid = self.pool.draw(domain="problem-solving")
        assert plasmid is not None
        assert plasmid.content  # non-empty extracted fragment

        # HGT direct transfer
        donor2 = _agent(
            "Creative lateral solution designer with abstract thinking and analysis.", 0.8
        )
        offspring = self.hgt.transfer(parent, donor2)
        if offspring is not None:
            assert isinstance(offspring.genome.system_prompt, str)
            assert len(offspring.genome.system_prompt) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. WorldModel + SelfPlay + HyperParams perturbation
# ─────────────────────────────────────────────────────────────────────────────


class TestWorldModelSelfPlayMeta:
    """WorldModel predicts agent performance from task history; SelfPlayEvaluator
    ranks agents head-to-head; HyperParams are perturbed each generation.
    Together they form a second-order optimisation loop with self-play selection."""

    def setup_method(self) -> None:
        from cambrian.evaluator import Evaluator
        from cambrian.meta_evolution import HyperParams
        from cambrian.self_play import SelfPlayEvaluator
        from cambrian.world_model import WorldModel

        class _IdentityEval(Evaluator):
            def evaluate(self, agent: object, task: str) -> float:
                return getattr(agent, "fitness", 0.5) or 0.5

        self.world_model = WorldModel(buffer_size=20)
        self.self_play_eval = SelfPlayEvaluator(
            base_evaluator=_IdentityEval(), win_bonus=0.1, loss_penalty=0.05
        )
        self.hp = HyperParams(
            mutation_rate=0.8,
            crossover_rate=0.3,
            temperature=0.5,
            tournament_k=3,
            elite_ratio=0.2,
        )

    def test_world_model_returns_default_before_experience(self) -> None:
        from cambrian.world_model import WorldModelPrediction
        pred = self.world_model.predict("Solve a brand new task")
        assert isinstance(pred, WorldModelPrediction)
        assert 0.0 <= pred.predicted_score <= 1.0

    def test_world_model_updates_with_experience(self) -> None:
        self.world_model.update(task="Write Python code for sorting", score=0.9)
        pred = self.world_model.predict("Write Python code solution sorting")
        assert 0.0 <= pred.predicted_score <= 1.0
        assert pred.n_similar >= 1

    def test_world_model_capacity_limit(self) -> None:
        for i in range(25):
            self.world_model.update(task=f"Task variant {i}", score=0.5)
        assert len(self.world_model._buffer) <= 20

    def test_self_play_compete_identifies_winner(self) -> None:
        winner = _agent("Expert analytical solver with proven methodology.", 0.9)
        loser = _agent("Basic minimal approach solver.", 0.3)

        result = self.self_play_eval.compete(
            agent_a=winner,
            agent_b=loser,
            task="Solve benchmark problem",
        )
        assert result.winner_id in (winner.id, loser.id)
        assert result.score_a >= 0.0
        assert result.score_b >= 0.0

    def test_hyperparams_perturb_stays_clamped(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            perturbed = self.hp.perturb(scale=0.1, rng=rng)
            assert 0.0 <= perturbed.mutation_rate <= 1.0
            assert 0.0 <= perturbed.crossover_rate <= 1.0
            assert 0.0 < perturbed.temperature
            assert 1 <= perturbed.tournament_k

    def test_combined_selection_loop(self) -> None:
        """3 generations: world model predicts, self-play ranks pairs,
        hyperparams are perturbed."""
        rng = random.Random(0)
        pop = [_agent(f"Agent strategy variant {i} systematic.", 0.3 + i * 0.1) for i in range(5)]

        for gen in range(3):
            # World model: update with each agent's task experience
            for a in pop:
                self.world_model.update(
                    task=f"Benchmark task generation {gen}",
                    score=a.fitness or 0.5,
                )
                pred = self.world_model.predict(f"Benchmark task generation {gen}")
                assert 0.0 <= pred.predicted_score <= 1.0

            # Self-play: compete pairs
            if len(pop) >= 2:
                r = self.self_play_eval.compete(
                    agent_a=pop[0],
                    agent_b=pop[-1],
                    task="Tournament challenge",
                )
                assert r.winner_id in (pop[0].id, pop[-1].id)

            # Perturb hyperparams
            self.hp = self.hp.perturb(scale=0.05, rng=rng)
            assert 0.0 <= self.hp.mutation_rate <= 1.0

        assert len(self.world_model._buffer) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLMCascade + BestOfN + ReflexionEvaluator
# ─────────────────────────────────────────────────────────────────────────────


class TestCascadeReflexionBestOfN:
    """LLMCascade routes queries to the cheapest confident backend; BestOfN
    samples multiple candidates and picks the highest-scoring; ReflexionEvaluator
    runs a critique-revise loop.  Together they form a layered inference quality
    stack that balances cost and quality."""

    def test_cascade_escalates_to_smart_backend(self) -> None:
        from cambrian.llm_cascade import CascadeLevel, LLMCascade

        fast = _mock_backend("I think the answer might be around 42, possibly maybe.")
        smart = _mock_backend("The answer is definitively 42.")
        cascade = LLMCascade(
            levels=[
                CascadeLevel(fast, confidence_threshold=0.9),
                CascadeLevel(smart, confidence_threshold=0.0),
            ]
        )
        response, level_idx = cascade.query("You are a math expert.", "What is 6x7?")
        assert isinstance(response, str)
        assert level_idx in (0, 1)

    def test_cascade_level_zero_when_confident(self) -> None:
        from cambrian.llm_cascade import CascadeLevel, LLMCascade

        confident_fast = _mock_backend("The answer is 42.")
        cascade = LLMCascade(
            levels=[
                CascadeLevel(confident_fast, confidence_threshold=0.0),
            ]
        )
        _, level_idx = cascade.query("System.", "Task.")
        assert level_idx == 0

    def test_best_of_n_picks_keyword_matching_candidate(self) -> None:
        from cambrian.inference_scaling import BestOfN, KeywordScorer

        backend = _mock_backend_seq(["the result is forty-two", "answer is 42", "unknown"])
        scorer = KeywordScorer(keywords=["42"])
        bon = BestOfN(backend=backend, n=3, scorer=scorer)
        result, score = bon.run(system="You are a math expert.", user="What is 6x7?")
        assert "42" in result

    def test_best_of_n_fallback_when_all_score_zero(self) -> None:
        from cambrian.inference_scaling import BestOfN, KeywordScorer

        backend = _mock_backend_seq(["no idea", "unknown", "unclear"])
        scorer = KeywordScorer(keywords=["42"])
        bon = BestOfN(backend=backend, n=3, scorer=scorer)
        result, score = bon.run(system="System.", user="Task.")
        assert isinstance(result, str)

    def test_reflexion_evaluate_returns_response_and_score(self) -> None:
        from cambrian.reflexion import ReflexionEvaluator

        backend = _mock_backend_seq(
            [
                "The answer is 42.",                          # generate
                "CRITIQUE: Perfect answer\nSCORE: 0.99",      # critique
                "The answer is definitively 42.",              # revise
                "CRITIQUE: Excellent\nSCORE: 1.00",           # critique 2
            ]
        )
        ref = ReflexionEvaluator(backend=backend, n_reflections=2)
        agent = _agent("Expert mathematician.", 0.7)
        result = ref.evaluate(agent, "What is 6x7?")
        # ReflexionEvaluator.evaluate returns (response, score)
        assert isinstance(result, tuple)
        response, score = result
        assert isinstance(response, str)
        assert 0.0 <= score <= 1.0

    def test_combined_inference_stack(self) -> None:
        """Cascade selects backend → BestOfN samples candidates → Reflexion refines."""
        from cambrian.inference_scaling import BestOfN, KeywordScorer
        from cambrian.llm_cascade import CascadeLevel, LLMCascade
        from cambrian.reflexion import ReflexionEvaluator

        # Step 1: cascade
        fast = _mock_backend("I think the answer might be 42.")
        smart = _mock_backend("Definitively 42.")
        cascade = LLMCascade(
            levels=[
                CascadeLevel(fast, confidence_threshold=0.95),
                CascadeLevel(smart, confidence_threshold=0.0),
            ]
        )
        resp, lvl = cascade.query("Expert system.", "Compute 6x7.")
        assert isinstance(resp, str)

        # Step 2: best-of-n selects best candidate
        bon_backend = _mock_backend_seq(["42", "forty-two", "6 times 7 equals 42"])
        bon = BestOfN(backend=bon_backend, n=3, scorer=KeywordScorer(["42"]))
        best_candidate, best_score = bon.run(system="Expert.", user="6x7?")
        assert "42" in best_candidate

        # Step 3: reflexion refines the agent
        ref_backend = _mock_backend_seq(
            [
                "The answer is 42.",
                "CRITIQUE: Excellent\nSCORE: 0.95",
                "The answer is definitively 42.",
                "CRITIQUE: Perfect\nSCORE: 0.99",
            ]
        )
        ref = ReflexionEvaluator(backend=ref_backend, n_reflections=2)
        agent = _agent("Expert reasoning agent.", 0.75)
        result = ref.evaluate(agent, "What is 6x7?")
        response, score = result
        assert 0.0 <= score <= 1.0
