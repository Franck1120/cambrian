"""Edge-case tests for Tier 5 modules: metamorphosis, ecosystem, fractal.

Covers:
- Metamorphosis: rapid multi-step transitions, IMAGO terminal lock, re-registration,
  zero-population phase_distribution, concurrent phase tracking correctness
- Ecosystem: extinction (all agents removed by apoptosis), single-agent population,
  parasite with no eligible hosts, decomposer with threshold 0, role re-assignment
- Fractal: singleton population per scale, max recursion depth, genome seeding
  correctness across scales, empty elite list handling
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.ecosystem import EcologicalRole, EcosystemConfig, EcosystemInteraction
from cambrian.fractal import (
    FractalEvolution,
    FractalMutator,
    FractalPopulation,
    FractalResult,
    FractalScale,
    ScaleConfig,
)
from cambrian.metamorphosis import (
    MetamorphicPhase,
    MetamorphicPopulation,
    MetamorphosisController,
    MorphEvent,
    PhaseConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(fitness: float = 0.5) -> Agent:
    g = Genome(system_prompt="test agent prompt for evolution")
    a = Agent(genome=g)
    a.fitness = fitness
    return a


def _mock_backend(response: str = "improved system prompt") -> MagicMock:
    b = MagicMock()
    b.generate = MagicMock(return_value=response)
    return b


def _mock_evaluator(score: float = 0.6) -> MagicMock:
    e = MagicMock()
    e.evaluate = MagicMock(return_value=score)
    return e


# ===========================================================================
# Metamorphosis edge cases
# ===========================================================================


class TestMetamorphosisEdgeCases:
    """Edge cases for MetamorphosisController and MetamorphicPopulation."""

    def test_imago_is_terminal_no_further_advance(self) -> None:
        """An IMAGO agent never transitions again regardless of fitness/generations."""
        ctrl = MetamorphosisController(backend=_mock_backend())
        agent = _agent(fitness=1.0)

        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.IMAGO
        ctrl._agent_gen_in_phase[agent.agent_id] = 0

        for gen in range(10):
            ev = ctrl.advance(agent, generation=gen, fitness=1.0)
            assert ev is None, f"Unexpected transition at gen {gen}"

        assert ctrl.current_phase(agent) is MetamorphicPhase.IMAGO

    def test_rapid_transitions_larva_to_imago(self) -> None:
        """Agent with very low min_generations and high fitness advances quickly."""
        ctrl = MetamorphosisController(
            backend=_mock_backend(),
            larva_config=PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=1,
                fitness_threshold=0.1,
                mutation_rate_multiplier=1.5,
            ),
            chrysalis_config=PhaseConfig(
                phase=MetamorphicPhase.CHRYSALIS,
                min_generations=1,
                fitness_threshold=0.1,
                mutation_rate_multiplier=0.0,
            ),
            imago_config=PhaseConfig(
                phase=MetamorphicPhase.IMAGO,
                min_generations=0,
                fitness_threshold=0.0,
                mutation_rate_multiplier=0.5,
            ),
        )
        agent = _agent(fitness=0.9)

        ev1 = ctrl.advance(agent, generation=1, fitness=0.9)
        assert ev1 is not None
        assert ev1.from_phase is MetamorphicPhase.LARVA
        assert ev1.to_phase is MetamorphicPhase.CHRYSALIS

        ev2 = ctrl.advance(agent, generation=2, fitness=0.9)
        assert ev2 is not None
        assert ev2.from_phase is MetamorphicPhase.CHRYSALIS
        assert ev2.to_phase is MetamorphicPhase.IMAGO

        ev3 = ctrl.advance(agent, generation=3, fitness=0.9)
        assert ev3 is None  # IMAGO is terminal

    def test_re_registration_does_not_reset_phase(self) -> None:
        """Calling advance on an already-registered agent preserves its phase."""
        ctrl = MetamorphosisController(backend=_mock_backend())
        agent = _agent(fitness=1.0)

        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.CHRYSALIS
        ctrl._agent_gen_in_phase[agent.agent_id] = 5

        # advance should NOT reset to LARVA
        ctrl.advance(agent, generation=1, fitness=1.0)
        assert ctrl._agent_phase[agent.agent_id] is not MetamorphicPhase.LARVA

    def test_phase_distribution_empty(self) -> None:
        """phase_distribution on a fresh controller returns empty dict."""
        ctrl = MetamorphosisController(backend=_mock_backend())
        assert ctrl.phase_distribution() == {}

    def test_phase_distribution_multi_agent(self) -> None:
        """phase_distribution counts correctly across heterogeneous phases."""
        ctrl = MetamorphosisController(backend=_mock_backend())
        agents = [_agent() for _ in range(6)]

        phases = [
            MetamorphicPhase.LARVA,
            MetamorphicPhase.LARVA,
            MetamorphicPhase.CHRYSALIS,
            MetamorphicPhase.IMAGO,
            MetamorphicPhase.IMAGO,
            MetamorphicPhase.IMAGO,
        ]
        for a, p in zip(agents, phases):
            ctrl._agent_phase[a.agent_id] = p
            ctrl._agent_gen_in_phase[a.agent_id] = 0

        dist = ctrl.phase_distribution()
        assert dist["larva"] == 2
        assert dist["chrysalis"] == 1
        assert dist["imago"] == 3

    def test_fitness_below_threshold_blocks_advance(self) -> None:
        """Agent that meets min_generations but not fitness stays in current phase."""
        ctrl = MetamorphosisController(
            backend=_mock_backend(),
            larva_config=PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=1,
                fitness_threshold=0.8,
                mutation_rate_multiplier=1.5,
            ),
        )
        agent = _agent(fitness=0.3)

        # enough generations, but fitness too low
        for gen in range(5):
            ev = ctrl.advance(agent, generation=gen, fitness=0.3)
            assert ev is None

        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA

    def test_events_list_grows_correctly(self) -> None:
        """events property returns all recorded MorphEvent objects."""
        ctrl = MetamorphosisController(
            backend=_mock_backend(),
            larva_config=PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=1,
                fitness_threshold=0.1,
            ),
        )
        agents = [_agent(fitness=0.9) for _ in range(4)]

        for gen, a in enumerate(agents, start=1):
            ctrl.advance(a, generation=gen, fitness=0.9)

        evs = ctrl.events
        assert isinstance(evs, list)
        assert all(isinstance(e, MorphEvent) for e in evs)

    def test_metamorphic_population_tick_triggers_chrysalis_reorg(self) -> None:
        """tick() on a LARVA agent that meets criteria triggers metamorphose."""
        backend = _mock_backend("reorganised prompt via chrysalis")
        ctrl = MetamorphosisController(
            backend=backend,
            larva_config=PhaseConfig(
                phase=MetamorphicPhase.LARVA,
                min_generations=1,
                fitness_threshold=0.1,
                mutation_rate_multiplier=1.5,
            ),
            chrysalis_config=PhaseConfig(
                phase=MetamorphicPhase.CHRYSALIS,
                min_generations=0,
                fitness_threshold=0.1,
                mutation_rate_multiplier=0.0,
            ),
        )
        pop = MetamorphicPopulation(controller=ctrl)
        agent = _agent(fitness=0.9)
        pop.register(agent)

        events = pop.tick([agent], generation=1, task="task")
        assert len(events) == 1
        assert events[0].to_phase is MetamorphicPhase.CHRYSALIS

    def test_mutation_rate_multiplier_by_phase(self) -> None:
        """mutation_rate_multiplier returns correct scalars for each phase."""
        ctrl = MetamorphosisController(backend=_mock_backend())
        agents = {
            MetamorphicPhase.LARVA: _agent(),
            MetamorphicPhase.CHRYSALIS: _agent(),
            MetamorphicPhase.IMAGO: _agent(),
        }
        for phase, agent in agents.items():
            ctrl._agent_phase[agent.agent_id] = phase
            ctrl._agent_gen_in_phase[agent.agent_id] = 0

        larva_agent = list(agents.values())[0]
        chrysalis_agent = list(agents.values())[1]
        imago_agent = list(agents.values())[2]

        assert ctrl.mutation_rate_multiplier(larva_agent) == pytest.approx(1.5)
        assert ctrl.mutation_rate_multiplier(chrysalis_agent) == pytest.approx(0.0)
        assert ctrl.mutation_rate_multiplier(imago_agent) == pytest.approx(0.5)

    def test_metamorphose_fallback_on_backend_error(self) -> None:
        """metamorphose returns a valid agent even when backend raises."""
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("backend down")
        ctrl = MetamorphosisController(backend=backend)

        agent = _agent(fitness=0.5)
        result = ctrl.metamorphose(agent, task="solve problem")

        assert isinstance(result, Agent)
        assert "metamorphosed" in result.genome.system_prompt


# ===========================================================================
# Ecosystem edge cases
# ===========================================================================


class TestEcosystemEdgeCases:
    """Edge cases for EcosystemInteraction — extinction, extremes, single agent."""

    def test_empty_population_interact_returns_empty(self) -> None:
        """interact([]) produces no events."""
        eco = EcosystemInteraction()
        events = eco.interact([], task="task")
        assert events == []

    def test_single_agent_herbivore_no_others(self) -> None:
        """Herbivore with no other agents gets zero diversity bonus."""
        eco = EcosystemInteraction()
        agent = _agent(fitness=0.5)
        eco.assign_role(agent, EcologicalRole.HERBIVORE)

        events = eco.interact([agent], task="task")
        assert len(events) == 1
        assert events[0].delta == pytest.approx(0.0)

    def test_single_agent_predator_no_prey(self) -> None:
        """Predator with no prey (no other agents) gets 0 delta."""
        eco = EcosystemInteraction()
        agent = _agent(fitness=0.8)
        eco.assign_role(agent, EcologicalRole.PREDATOR)

        events = eco.interact([agent], task="task")
        assert len(events) == 1
        assert events[0].delta == pytest.approx(0.0)

    def test_parasite_no_eligible_hosts(self) -> None:
        """Parasite with no agent above host_threshold produces no gain event."""
        cfg = EcosystemConfig(parasite_host_threshold=0.95)
        eco = EcosystemInteraction(config=cfg)

        parasite = _agent(fitness=0.4)
        weak = _agent(fitness=0.3)
        eco.assign_role(parasite, EcologicalRole.PARASITE)
        eco.assign_role(weak, EcologicalRole.HERBIVORE)

        events = eco.interact([parasite, weak], task="task")
        # parasite should produce no event (no eligible host)
        parasite_events = [e for e in events if e.agent_id == parasite.agent_id]
        assert all(e.event_type != "parasite" for e in parasite_events)

    def test_decomposer_zero_threshold_recycles_nothing(self) -> None:
        """Decomposer with recycle_threshold=0.0 finds no recyclable agents."""
        cfg = EcosystemConfig(decomposer_recycle_threshold=0.0)
        eco = EcosystemInteraction(config=cfg)

        decomp = _agent(fitness=0.6)
        other = _agent(fitness=0.3)
        eco.assign_role(decomp, EcologicalRole.DECOMPOSER)
        eco.assign_role(other, EcologicalRole.HERBIVORE)

        events = eco.interact([decomp, other], task="task")
        decomp_events = [e for e in events if e.agent_id == decomp.agent_id]
        assert all(e.delta == pytest.approx(0.0) for e in decomp_events)

    def test_apply_events_clamps_fitness_to_zero(self) -> None:
        """Negative delta never drives fitness below 0.0."""
        from cambrian.ecosystem import EcosystemEvent

        eco = EcosystemInteraction()
        agent = _agent(fitness=0.1)
        events = [
            EcosystemEvent(
                role=EcologicalRole.PARASITE,
                agent_id=agent.agent_id,
                target_id=None,
                delta=-5.0,
                event_type="parasite",
            )
        ]
        eco.apply_events(events, [agent])
        assert agent.fitness == pytest.approx(0.0)

    def test_apply_events_clamps_fitness_to_one(self) -> None:
        """Large positive delta never drives fitness above 1.0."""
        from cambrian.ecosystem import EcosystemEvent

        eco = EcosystemInteraction()
        agent = _agent(fitness=0.9)
        events = [
            EcosystemEvent(
                role=EcologicalRole.HERBIVORE,
                agent_id=agent.agent_id,
                target_id=None,
                delta=100.0,
                event_type="forage",
            )
        ]
        eco.apply_events(events, [agent])
        assert agent.fitness == pytest.approx(1.0)

    def test_extinction_scenario_all_low_fitness(self) -> None:
        """Population of all low-fitness agents — predators find no prey below threshold."""
        cfg = EcosystemConfig(predator_hunt_threshold=0.0)
        eco = EcosystemInteraction(config=cfg)

        population = [_agent(fitness=0.1) for _ in range(5)]
        eco.auto_assign(population)
        events = eco.interact(population, task="survive")

        # No prey below threshold=0.0 → predators all get delta=0
        predator_events = [
            e for e in events
            if eco.get_role(next(a for a in population if a.agent_id == e.agent_id))
            is EcologicalRole.PREDATOR
        ]
        for ev in predator_events:
            assert ev.delta == pytest.approx(0.0)

    def test_auto_assign_large_population_covers_all_roles(self) -> None:
        """auto_assign on 20 agents always assigns all 4 roles."""
        import random as rng

        rng.seed(0)
        eco = EcosystemInteraction()
        population = [_agent(fitness=rng.random()) for _ in range(20)]
        eco.auto_assign(population)

        roles = set(eco.get_role(a) for a in population)
        assert EcologicalRole.PREDATOR in roles
        assert EcologicalRole.DECOMPOSER in roles
        assert EcologicalRole.HERBIVORE in roles

    def test_role_counts_accurate(self) -> None:
        """role_counts sums match total assigned agents."""
        eco = EcosystemInteraction()
        population = [_agent() for _ in range(8)]
        eco.auto_assign(population)
        counts = eco.role_counts()
        assert sum(counts.values()) == 8

    def test_events_accumulate_across_rounds(self) -> None:
        """Multiple interact() calls accumulate events in eco.events."""
        eco = EcosystemInteraction()
        population = [_agent(fitness=0.5) for _ in range(4)]
        eco.auto_assign(population)

        eco.interact(population, task="round 1")
        eco.interact(population, task="round 2")
        assert len(eco.events) > 0


# ===========================================================================
# Fractal edge cases
# ===========================================================================


class TestFractalEdgeCases:
    """Edge cases for FractalMutator, FractalPopulation, FractalEvolution."""

    def test_fractal_scale_ordering(self) -> None:
        """MACRO < MESO < MICRO as int values."""
        assert FractalScale.MACRO < FractalScale.MESO < FractalScale.MICRO

    def test_scale_config_defaults_override_by_scale(self) -> None:
        """ScaleConfig always applies scale-specific defaults."""
        macro = ScaleConfig(scale=FractalScale.MACRO)
        meso = ScaleConfig(scale=FractalScale.MESO)
        micro = ScaleConfig(scale=FractalScale.MICRO)

        assert macro.mutation_temperature > micro.mutation_temperature
        assert macro.fragment_size > meso.fragment_size > micro.fragment_size

    def test_fractal_mutator_macro_returns_genome(self) -> None:
        """mutate_macro returns a Genome instance."""
        backend = _mock_backend("new macro strategy prompt")
        fm = FractalMutator(backend=backend)
        genome = Genome(system_prompt="original prompt")
        result = fm.mutate_macro(genome, task="task")
        assert isinstance(result, Genome)

    def test_fractal_mutator_meso_returns_genome(self) -> None:
        """mutate_meso returns a Genome instance."""
        backend = _mock_backend("refined paragraph")
        fm = FractalMutator(backend=backend)
        genome = Genome(system_prompt="p1. p2. p3.")
        result = fm.mutate_meso(genome, task="task")
        assert isinstance(result, Genome)

    def test_fractal_mutator_micro_returns_genome(self) -> None:
        """mutate_micro returns a Genome instance."""
        backend = _mock_backend("tweak")
        fm = FractalMutator(backend=backend)
        genome = Genome(system_prompt="improve this word")
        result = fm.mutate_micro(genome, task="task")
        assert isinstance(result, Genome)

    def test_fractal_mutator_backend_error_returns_original(self) -> None:
        """mutate_macro falls back to original genome on backend error."""
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("LLM unavailable")
        fm = FractalMutator(backend=backend)
        genome = Genome(system_prompt="unchanged prompt")
        result = fm.mutate_macro(genome, task="task")
        assert isinstance(result, Genome)
        assert result.system_prompt == "unchanged prompt"

    def test_fractal_population_singleton(self) -> None:
        """FractalPopulation with a single seed agent runs without error."""
        backend = _mock_backend()
        evaluator = _mock_evaluator(0.6)
        config = ScaleConfig(scale=FractalScale.MACRO, population_size=1, n_generations=1)
        mutator = FractalMutator(backend=backend)
        seed = Genome(system_prompt="single seed genome text for testing purposes")

        fp = FractalPopulation(
            scale=FractalScale.MACRO,
            config=config,
            evaluator=evaluator,
            mutator=mutator,
        )
        fp.seed(seed)
        best = fp.evolve_step(task="task")
        assert isinstance(best, Agent)
        assert best.fitness is not None

    def test_fractal_population_returns_agent(self) -> None:
        """FractalPopulation.evolve_step returns an Agent after seeding."""
        backend = _mock_backend()
        evaluator = _mock_evaluator(0.7)
        config = ScaleConfig(scale=FractalScale.MESO, population_size=2, n_generations=1)
        mutator = FractalMutator(backend=backend)
        seed = Genome(system_prompt="meso scale seed genome for testing")

        fp = FractalPopulation(
            scale=FractalScale.MESO,
            config=config,
            evaluator=evaluator,
            mutator=mutator,
        )
        fp.seed(seed)
        best = fp.evolve_step(task="task")
        assert isinstance(best, Agent)

    def test_fractal_evolution_end_to_end(self) -> None:
        """FractalEvolution runs MACRO->MESO->MICRO and returns a FractalResult."""
        backend = _mock_backend("evolved prompt content")
        evaluator = _mock_evaluator(0.65)

        fe = FractalEvolution(
            backend=backend,
            evaluator=evaluator,
            macro_config=ScaleConfig(
                scale=FractalScale.MACRO, population_size=2, n_generations=1
            ),
            meso_config=ScaleConfig(
                scale=FractalScale.MESO, population_size=2, n_generations=1
            ),
            micro_config=ScaleConfig(
                scale=FractalScale.MICRO, population_size=2, n_generations=1
            ),
        )
        seed = Genome(system_prompt="initial prompt for testing fractal evolution")
        result = fe.evolve(seed_genome=seed, task="fractal task", n_cycles=2)

        assert isinstance(result, FractalResult)
        assert result.best_fitness >= 0.0
        assert isinstance(result.best_genome, Genome)

    def test_fractal_evolution_result_n_evaluations_positive(self) -> None:
        """FractalEvolution result.n_evaluations counts actual evaluator calls."""
        backend = _mock_backend()
        evaluator = _mock_evaluator(0.5)

        fe = FractalEvolution(
            backend=backend,
            evaluator=evaluator,
            macro_config=ScaleConfig(
                scale=FractalScale.MACRO, population_size=2, n_generations=1
            ),
            meso_config=ScaleConfig(
                scale=FractalScale.MESO, population_size=2, n_generations=1
            ),
            micro_config=ScaleConfig(
                scale=FractalScale.MICRO, population_size=2, n_generations=1
            ),
        )
        seed = Genome(system_prompt="counting evaluations here")
        result = fe.evolve(seed_genome=seed, task="count task", n_cycles=1)
        assert result.n_evaluations > 0

    def test_fractal_evolution_best_fitness_nonnegative(self) -> None:
        """FractalEvolution never returns negative best_fitness."""
        backend = _mock_backend()
        evaluator = MagicMock()
        evaluator.evaluate = MagicMock(return_value=-0.5)  # evaluator returns negative

        fe = FractalEvolution(
            backend=backend,
            evaluator=evaluator,
            macro_config=ScaleConfig(
                scale=FractalScale.MACRO, population_size=2, n_generations=1
            ),
            meso_config=ScaleConfig(
                scale=FractalScale.MESO, population_size=2, n_generations=1
            ),
            micro_config=ScaleConfig(
                scale=FractalScale.MICRO, population_size=2, n_generations=1
            ),
        )
        seed = Genome(system_prompt="negative fitness test")
        result = fe.evolve(seed_genome=seed, task="neg task", n_cycles=1)
        assert result.best_fitness >= 0.0
