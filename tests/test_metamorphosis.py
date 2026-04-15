"""Tests for cambrian.metamorphosis — MetamorphicPhase, PhaseConfig, MorphEvent,
MetamorphosisController, and MetamorphicPopulation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cambrian.agent import Agent, Genome
from cambrian.backends.base import LLMBackend
from cambrian.metamorphosis import (
    MetamorphicPhase,
    MetamorphicPopulation,
    MetamorphosisController,
    MorphEvent,
    PhaseConfig,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_backend(response: str = "Mature specialized prompt.") -> LLMBackend:
    """Return a MagicMock that satisfies the LLMBackend interface."""
    backend: LLMBackend = MagicMock(spec=LLMBackend)
    backend.generate.return_value = response  # type: ignore[attr-defined]
    return backend


def _make_agent(
    system_prompt: str = "You are a test agent.",
    fitness_val: float | None = None,
) -> Agent:
    genome = Genome(system_prompt=system_prompt)
    agent = Agent(genome=genome)
    if fitness_val is not None:
        agent.fitness = fitness_val
    return agent


def _make_controller(
    backend: LLMBackend | None = None,
) -> MetamorphosisController:
    b = backend or _mock_backend()
    return MetamorphosisController(backend=b)


# ── MetamorphicPhase ──────────────────────────────────────────────────────────


class TestMetamorphicPhase:
    def test_larva_value(self) -> None:
        assert MetamorphicPhase.LARVA.value == "larva"

    def test_chrysalis_value(self) -> None:
        assert MetamorphicPhase.CHRYSALIS.value == "chrysalis"

    def test_imago_value(self) -> None:
        assert MetamorphicPhase.IMAGO.value == "imago"

    def test_phase_is_str_subclass(self) -> None:
        assert isinstance(MetamorphicPhase.LARVA, str)

    def test_three_phases_exist(self) -> None:
        phases = list(MetamorphicPhase)
        assert len(phases) == 3

    def test_phases_are_unique(self) -> None:
        values = [p.value for p in MetamorphicPhase]
        assert len(set(values)) == len(values)


# ── PhaseConfig ───────────────────────────────────────────────────────────────


class TestPhaseConfig:
    def test_required_fields(self) -> None:
        cfg = PhaseConfig(phase=MetamorphicPhase.LARVA, min_generations=5)
        assert cfg.phase is MetamorphicPhase.LARVA
        assert cfg.min_generations == 5

    def test_default_fitness_threshold(self) -> None:
        cfg = PhaseConfig(phase=MetamorphicPhase.IMAGO, min_generations=0)
        assert cfg.fitness_threshold == 0.0

    def test_default_mutation_rate_multiplier(self) -> None:
        cfg = PhaseConfig(phase=MetamorphicPhase.LARVA, min_generations=3)
        assert cfg.mutation_rate_multiplier == 1.0

    def test_default_description(self) -> None:
        cfg = PhaseConfig(phase=MetamorphicPhase.CHRYSALIS, min_generations=1)
        assert cfg.description == ""

    def test_custom_values(self) -> None:
        cfg = PhaseConfig(
            phase=MetamorphicPhase.LARVA,
            min_generations=7,
            fitness_threshold=0.8,
            mutation_rate_multiplier=2.0,
            description="Custom larva.",
        )
        assert cfg.min_generations == 7
        assert cfg.fitness_threshold == 0.8
        assert cfg.mutation_rate_multiplier == 2.0
        assert cfg.description == "Custom larva."


# ── MorphEvent ────────────────────────────────────────────────────────────────


class TestMorphEvent:
    def test_fields_stored_correctly(self) -> None:
        ev = MorphEvent(
            agent_id="abc123",
            from_phase=MetamorphicPhase.LARVA,
            to_phase=MetamorphicPhase.CHRYSALIS,
            generation=5,
            fitness_at_transition=0.45,
        )
        assert ev.agent_id == "abc123"
        assert ev.from_phase is MetamorphicPhase.LARVA
        assert ev.to_phase is MetamorphicPhase.CHRYSALIS
        assert ev.generation == 5
        assert ev.fitness_at_transition == pytest.approx(0.45)

    def test_morph_event_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(MorphEvent)


# ── MetamorphosisController — init and defaults ───────────────────────────────


class TestMetamorphosisControllerInit:
    def test_default_larva_config(self) -> None:
        ctrl = _make_controller()
        cfg = ctrl._configs[MetamorphicPhase.LARVA]
        assert cfg.min_generations == 3
        assert cfg.fitness_threshold == pytest.approx(0.4)
        assert cfg.mutation_rate_multiplier == pytest.approx(1.5)

    def test_default_chrysalis_config(self) -> None:
        ctrl = _make_controller()
        cfg = ctrl._configs[MetamorphicPhase.CHRYSALIS]
        assert cfg.min_generations == 1
        assert cfg.fitness_threshold == pytest.approx(0.6)
        assert cfg.mutation_rate_multiplier == pytest.approx(0.0)

    def test_default_imago_config(self) -> None:
        ctrl = _make_controller()
        cfg = ctrl._configs[MetamorphicPhase.IMAGO]
        assert cfg.min_generations == 0
        assert cfg.fitness_threshold == pytest.approx(0.0)
        assert cfg.mutation_rate_multiplier == pytest.approx(0.5)

    def test_custom_larva_config_overrides_default(self) -> None:
        custom = PhaseConfig(
            phase=MetamorphicPhase.LARVA,
            min_generations=10,
            fitness_threshold=0.9,
            mutation_rate_multiplier=3.0,
        )
        ctrl = MetamorphosisController(backend=_mock_backend(), larva_config=custom)
        cfg = ctrl._configs[MetamorphicPhase.LARVA]
        assert cfg.min_generations == 10

    def test_events_initially_empty(self) -> None:
        ctrl = _make_controller()
        assert ctrl.events == []

    def test_phase_distribution_initially_empty(self) -> None:
        ctrl = _make_controller()
        assert ctrl.phase_distribution() == {}


# ── current_phase ─────────────────────────────────────────────────────────────


class TestCurrentPhase:
    def test_unregistered_agent_defaults_to_larva(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA

    def test_registered_agent_starts_as_larva(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.LARVA
        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA


# ── mutation_rate_multiplier ──────────────────────────────────────────────────


class TestMutationRateMultiplier:
    def test_larva_multiplier_is_gt_imago(self) -> None:
        ctrl = _make_controller()
        larva_agent = _make_agent()
        imago_agent = _make_agent()
        ctrl._agent_phase[larva_agent.agent_id] = MetamorphicPhase.LARVA
        ctrl._agent_phase[imago_agent.agent_id] = MetamorphicPhase.IMAGO
        assert ctrl.mutation_rate_multiplier(larva_agent) > ctrl.mutation_rate_multiplier(imago_agent)

    def test_chrysalis_multiplier_is_zero(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.CHRYSALIS
        assert ctrl.mutation_rate_multiplier(agent) == pytest.approx(0.0)

    def test_larva_multiplier_default(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # Default (unregistered) is LARVA
        assert ctrl.mutation_rate_multiplier(agent) == pytest.approx(1.5)

    def test_imago_multiplier_default(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.IMAGO
        assert ctrl.mutation_rate_multiplier(agent) == pytest.approx(0.5)


# ── advance ───────────────────────────────────────────────────────────────────


class TestAdvance:
    def test_returns_none_when_below_min_generations(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # Default larva min_generations=3; call only once with high fitness
        result = ctrl.advance(agent, generation=1, fitness=0.9)
        assert result is None

    def test_returns_none_when_below_fitness_threshold(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # Satisfy min_gen by calling 3 times with low fitness
        for i in range(3):
            result = ctrl.advance(agent, generation=i, fitness=0.1)
        assert result is None

    def test_stays_in_larva_when_threshold_not_met(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        for i in range(5):
            ctrl.advance(agent, generation=i, fitness=0.1)
        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA

    def test_stays_in_larva_when_min_gen_not_met(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # Only 2 calls but min is 3
        ctrl.advance(agent, generation=0, fitness=0.9)
        ctrl.advance(agent, generation=1, fitness=0.9)
        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA

    def test_transitions_larva_to_chrysalis(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        event: MorphEvent | None = None
        for i in range(3):
            event = ctrl.advance(agent, generation=i, fitness=0.5)
        assert event is not None
        assert event.from_phase is MetamorphicPhase.LARVA
        assert event.to_phase is MetamorphicPhase.CHRYSALIS

    def test_current_phase_is_chrysalis_after_transition(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        for i in range(3):
            ctrl.advance(agent, generation=i, fitness=0.5)
        assert ctrl.current_phase(agent) is MetamorphicPhase.CHRYSALIS

    def test_transitions_chrysalis_to_imago(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # Reach chrysalis
        for i in range(3):
            ctrl.advance(agent, generation=i, fitness=0.5)
        # Now in chrysalis — min_gen=1, fitness_thresh=0.6
        event = ctrl.advance(agent, generation=3, fitness=0.7)
        assert event is not None
        assert event.from_phase is MetamorphicPhase.CHRYSALIS
        assert event.to_phase is MetamorphicPhase.IMAGO

    def test_imago_advance_returns_none(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.IMAGO
        ctrl._agent_gen_in_phase[agent.agent_id] = 0
        result = ctrl.advance(agent, generation=99, fitness=1.0)
        assert result is None

    def test_imago_stays_in_imago(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        ctrl._agent_phase[agent.agent_id] = MetamorphicPhase.IMAGO
        ctrl._agent_gen_in_phase[agent.agent_id] = 0
        for i in range(5):
            ctrl.advance(agent, generation=i, fitness=1.0)
        assert ctrl.current_phase(agent) is MetamorphicPhase.IMAGO

    def test_morph_event_fields_on_transition(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        event: MorphEvent | None = None
        for i in range(3):
            event = ctrl.advance(agent, generation=i, fitness=0.5)
        assert event is not None
        assert event.agent_id == agent.agent_id
        assert event.generation == 2
        assert event.fitness_at_transition == pytest.approx(0.5)

    def test_gen_counter_resets_after_transition(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        for i in range(3):
            ctrl.advance(agent, generation=i, fitness=0.5)
        # After transition gen_in_phase should reset
        assert ctrl._agent_gen_in_phase[agent.agent_id] == 0

    def test_advance_registers_unknown_agent(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        assert agent.agent_id not in ctrl._agent_phase
        ctrl.advance(agent, generation=0, fitness=0.1)
        assert agent.agent_id in ctrl._agent_phase


# ── metamorphose ──────────────────────────────────────────────────────────────


class TestMetamorphose:
    def test_calls_backend_and_returns_new_agent(self) -> None:
        backend = _mock_backend(response="Specialized expert prompt.")
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Original prompt.")
        new_agent = ctrl.metamorphose(agent, task="Solve equations.")
        backend.generate.assert_called_once()  # type: ignore[attr-defined]
        assert new_agent.genome.system_prompt == "Specialized expert prompt."

    def test_returns_new_agent_with_different_genome(self) -> None:
        backend = _mock_backend(response="New specialized prompt.")
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Original prompt.")
        new_agent = ctrl.metamorphose(agent, task="Some task.")
        assert new_agent is not agent
        assert new_agent.genome.system_prompt != "Original prompt."

    def test_original_agent_genome_unchanged(self) -> None:
        backend = _mock_backend(response="New specialized prompt.")
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Original prompt.")
        original_prompt = agent.genome.system_prompt
        ctrl.metamorphose(agent, task="Some task.")
        assert agent.genome.system_prompt == original_prompt

    def test_metamorphose_preserves_agent_id(self) -> None:
        backend = _mock_backend(response="New specialized prompt.")
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Original prompt.")
        new_agent = ctrl.metamorphose(agent, task="Some task.")
        assert new_agent.agent_id == agent.agent_id

    def test_fallback_on_backend_failure(self) -> None:
        backend: LLMBackend = MagicMock(spec=LLMBackend)
        backend.generate.side_effect = RuntimeError("LLM unavailable")  # type: ignore[attr-defined]
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Original prompt.")
        new_agent = ctrl.metamorphose(agent, task="Some task.")
        # Should not raise and should return an agent with a fallback prompt
        assert new_agent is not None
        assert "Original prompt." in new_agent.genome.system_prompt
        assert new_agent.genome.system_prompt != "Original prompt."

    def test_fallback_includes_task(self) -> None:
        backend: LLMBackend = MagicMock(spec=LLMBackend)
        backend.generate.side_effect = Exception("error")  # type: ignore[attr-defined]
        ctrl = MetamorphosisController(backend=backend)
        agent = _make_agent("Base prompt.")
        new_agent = ctrl.metamorphose(agent, task="physics homework")
        assert "physics homework" in new_agent.genome.system_prompt


# ── apply_phase_pressure ──────────────────────────────────────────────────────


class TestApplyPhasePressure:
    def test_larva_raises_temperature(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", temperature=0.7)
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.LARVA)
        assert new_genome.temperature > genome.temperature

    def test_imago_sets_strategy_to_chain_of_thought(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", strategy="step-by-step")
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.IMAGO)
        assert new_genome.strategy == "chain-of-thought"

    def test_imago_lowers_temperature(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", temperature=0.7)
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.IMAGO)
        assert new_genome.temperature < genome.temperature

    def test_chrysalis_does_not_change_genome(self) -> None:
        ctrl = _make_controller()
        genome = Genome(
            system_prompt="Test.",
            temperature=0.7,
            strategy="step-by-step",
        )
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.CHRYSALIS)
        assert new_genome.temperature == pytest.approx(genome.temperature)
        assert new_genome.strategy == genome.strategy

    def test_original_genome_not_mutated(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", temperature=0.7)
        orig_temp = genome.temperature
        ctrl.apply_phase_pressure(genome, MetamorphicPhase.LARVA)
        assert genome.temperature == pytest.approx(orig_temp)

    def test_larva_temperature_clamped_to_two(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", temperature=2.0)
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.LARVA)
        assert new_genome.temperature <= 2.0

    def test_imago_temperature_clamped_to_zero(self) -> None:
        ctrl = _make_controller()
        genome = Genome(system_prompt="Test.", temperature=0.0)
        new_genome = ctrl.apply_phase_pressure(genome, MetamorphicPhase.IMAGO)
        assert new_genome.temperature >= 0.0


# ── events property and phase_distribution ────────────────────────────────────


class TestEventsAndDistribution:
    def test_events_returns_all_morph_events(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        for i in range(3):
            ctrl.advance(agent, generation=i, fitness=0.5)
        assert len(ctrl.events) == 1

    def test_events_accumulate_across_transitions(self) -> None:
        ctrl = _make_controller()
        agent = _make_agent()
        # LARVA → CHRYSALIS
        for i in range(3):
            ctrl.advance(agent, generation=i, fitness=0.5)
        # CHRYSALIS → IMAGO
        ctrl.advance(agent, generation=3, fitness=0.7)
        assert len(ctrl.events) == 2

    def test_events_returns_copy(self) -> None:
        ctrl = _make_controller()
        events_ref = ctrl.events
        events_ref.append(  # type: ignore[arg-type]
            MorphEvent("x", MetamorphicPhase.LARVA, MetamorphicPhase.CHRYSALIS, 0, 0.5)
        )
        assert len(ctrl.events) == 0

    def test_phase_distribution_counts_correctly(self) -> None:
        ctrl = _make_controller()
        a1 = _make_agent()
        a2 = _make_agent()
        ctrl._agent_phase[a1.agent_id] = MetamorphicPhase.LARVA
        ctrl._agent_phase[a2.agent_id] = MetamorphicPhase.IMAGO
        dist = ctrl.phase_distribution()
        assert dist.get("larva") == 1
        assert dist.get("imago") == 1

    def test_phase_distribution_multiple_per_phase(self) -> None:
        ctrl = _make_controller()
        for _ in range(3):
            a = _make_agent()
            ctrl._agent_phase[a.agent_id] = MetamorphicPhase.LARVA
        dist = ctrl.phase_distribution()
        assert dist.get("larva") == 3


# ── MetamorphicPopulation ─────────────────────────────────────────────────────


class TestMetamorphicPopulation:
    def test_register_adds_agent_as_larva(self) -> None:
        ctrl = _make_controller()
        pop = MetamorphicPopulation(controller=ctrl)
        agent = _make_agent()
        pop.register(agent)
        assert ctrl.current_phase(agent) is MetamorphicPhase.LARVA

    def test_register_agent_appears_in_distribution(self) -> None:
        ctrl = _make_controller()
        pop = MetamorphicPopulation(controller=ctrl)
        agent = _make_agent()
        pop.register(agent)
        dist = ctrl.phase_distribution()
        assert dist.get("larva", 0) == 1

    def test_tick_returns_empty_when_no_transitions(self) -> None:
        ctrl = _make_controller()
        pop = MetamorphicPopulation(controller=ctrl)
        agent = _make_agent(fitness_val=0.1)
        pop.register(agent)
        events = pop.tick([agent], generation=0, task="test task")
        assert events == []

    def test_tick_returns_events_on_transition(self) -> None:
        ctrl = _make_controller()
        pop = MetamorphicPopulation(controller=ctrl)
        agent = _make_agent(fitness_val=0.5)
        pop.register(agent)
        all_events: list[MorphEvent] = []
        for i in range(3):
            all_events.extend(pop.tick([agent], generation=i, task="test"))
        assert len(all_events) == 1
        assert all_events[0].to_phase is MetamorphicPhase.CHRYSALIS

    def test_multiple_agents_different_phases(self) -> None:
        ctrl = _make_controller()
        pop = MetamorphicPopulation(controller=ctrl)
        # One will advance, one won't
        fast_agent = _make_agent(fitness_val=0.8)
        slow_agent = _make_agent(fitness_val=0.1)
        pop.register(fast_agent)
        pop.register(slow_agent)
        all_events: list[MorphEvent] = []
        for i in range(3):
            all_events.extend(pop.tick([fast_agent, slow_agent], generation=i, task="task"))
        # fast_agent should have transitioned; slow_agent should not
        assert any(e.agent_id == fast_agent.agent_id for e in all_events)
        assert not any(e.agent_id == slow_agent.agent_id for e in all_events)

    def test_tick_calls_metamorphose_on_chrysalis_entry(self) -> None:
        backend = _mock_backend(response="New specialized system prompt.")
        ctrl = MetamorphosisController(backend=backend)
        pop = MetamorphicPopulation(controller=ctrl)
        # Use high fitness to also pass chrysalis fitness_threshold check in tick
        agent = _make_agent(fitness_val=0.8)
        pop.register(agent)
        for i in range(3):
            pop.tick([agent], generation=i, task="metamorphosis task")
        # backend.generate should have been called during chrysalis metamorphose
        backend.generate.assert_called()  # type: ignore[attr-defined]
