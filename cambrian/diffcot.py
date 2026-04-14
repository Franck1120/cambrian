"""DiffCoT — Diffusion Chain-of-Thought iterative denoising for agent reasoning.

Inspired by diffusion models, DiffCoT improves an agent's answer through
multiple "denoising" steps instead of a single forward pass.  Each step
refines the previous response by injecting the prior answer back into the
prompt and adjusting sampling temperature according to a schedule:

- Early steps use a higher temperature (exploration / noisy sampling).
- Later steps use a lower temperature (exploitation / convergence).

Three schedules are supported:
- ``"cosine"``: smooth cosine annealing from high to low temperature.
- ``"linear"``: linear decay from high to low temperature.
- ``"constant"``: temperature fixed to the genome's base temperature.

The :class:`DiffCoTReasoner` drives the iterative denoising loop and returns a
:class:`DiffCoTResult` containing all intermediate steps and a convergence
score.  The :class:`DiffCoTEvaluator` wraps any existing :class:`Evaluator`
so that the final DiffCoT answer is used in place of the normal single-step
response when scoring an agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cambrian.backends.base import LLMBackend
from cambrian.evaluator import Evaluator

if TYPE_CHECKING:
    from cambrian.agent import Agent, Genome


# ---------------------------------------------------------------------------
# Default step prompt template
# ---------------------------------------------------------------------------

_DEFAULT_STEP_TEMPLATE: str = (
    "Step {step}/{total_steps}: Refine the following reasoning.\n\n"
    "Task: {task}\n\n"
    "Previous answer:\n{previous}\n\n"
    "Produce an improved, more coherent answer."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DiffCoTConfig:
    """Configuration for a DiffCoT denoising run.

    Attributes:
        n_steps: Number of denoising iterations.
        noise_level: Controls how much additional randomness is injected in
            early steps (added on top of the base temperature).
        temperature_schedule: How temperature evolves across steps.
            One of ``"cosine"``, ``"linear"``, or ``"constant"``.
        inject_previous: Whether each step receives the previous step's answer
            as context.  If ``False``, each step sees only the bare task.
        step_prompt_template: Jinja-free string template with ``{step}``,
            ``{total_steps}``, ``{previous}``, and ``{task}`` placeholders.
    """

    n_steps: int = 3
    noise_level: float = 0.3
    temperature_schedule: str = "cosine"
    inject_previous: bool = True
    step_prompt_template: str = _DEFAULT_STEP_TEMPLATE


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DiffCoTStep:
    """A single denoising step recorded during a DiffCoT run.

    Attributes:
        step: Zero-based step index.
        temperature: Effective sampling temperature used at this step.
        prompt: The full prompt sent to the backend at this step.
        response: The backend's response for this step.
    """

    step: int
    temperature: float
    prompt: str
    response: str


@dataclass
class DiffCoTResult:
    """Result of a full DiffCoT denoising run.

    Attributes:
        final_answer: The response produced by the last denoising step.
        steps: All :class:`DiffCoTStep` records in order.
        convergence_score: Character-level Jaccard similarity between the last
            two steps' responses (``0.0`` = fully diverged, ``1.0`` = identical).
            Computed as ``|set(a) & set(b)| / max(|set(a) | set(b)|, 1)``.
        n_steps: Number of steps that were actually executed.
    """

    final_answer: str
    steps: list[DiffCoTStep] = field(default_factory=list)
    convergence_score: float = 0.0
    n_steps: int = 0


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------


class DiffCoTReasoner:
    """Iterative chain-of-thought denoiser backed by any :class:`LLMBackend`.

    Args:
        backend: LLM backend used to generate each denoising step.
        config: DiffCoT configuration; defaults to :class:`DiffCoTConfig`
            with its default field values if ``None``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: DiffCoTConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config: DiffCoTConfig = config if config is not None else DiffCoTConfig()

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------

    def _temperature_at_step(self, step: int, base_temp: float) -> float:
        """Return the effective sampling temperature for *step*.

        Args:
            step: Zero-based step index.
            base_temp: The agent genome's base temperature.

        Returns:
            Effective temperature as a non-negative float.
        """
        cfg = self._config
        total: int = max(cfg.n_steps - 1, 1)  # avoid /0 when n_steps==1
        progress: float = step / total

        schedule = cfg.temperature_schedule
        if schedule == "cosine":
            # High at step 0, low at final step
            temp = base_temp * 0.5 * (1.0 + math.cos(math.pi * progress))
            noise = cfg.noise_level * base_temp * (1.0 - progress)
            return temp + noise
        elif schedule == "linear":
            temp = base_temp * (1.0 - progress)
            noise = cfg.noise_level * base_temp * progress
            return temp + noise
        else:
            # "constant" (and any unrecognised schedule falls back to constant)
            return base_temp

    # ------------------------------------------------------------------
    # Core denoising loop
    # ------------------------------------------------------------------

    def reason(self, agent_genome: "Genome", task: str) -> DiffCoTResult:
        """Run the DiffCoT denoising loop and return a :class:`DiffCoTResult`.

        Args:
            agent_genome: The agent's :class:`~cambrian.agent.Genome`.
                Used for the system prompt, base temperature, and model name.
            task: Natural-language task to reason about.

        Returns:
            A :class:`DiffCoTResult` with all intermediate steps, the final
            answer, and a convergence score.
        """
        cfg = self._config
        steps: list[DiffCoTStep] = []
        previous_response: str = ""

        for step_idx in range(cfg.n_steps):
            temp = self._temperature_at_step(step_idx, agent_genome.temperature)

            if step_idx == 0 or not cfg.inject_previous:
                # First step (or no injection): plain task
                prompt = task
            else:
                prompt = cfg.step_prompt_template.format(
                    step=step_idx + 1,
                    total_steps=cfg.n_steps,
                    previous=previous_response,
                    task=task,
                )

            response: str = self._backend.generate(
                prompt,
                system=agent_genome.system_prompt,
                temperature=temp,
            )

            steps.append(
                DiffCoTStep(
                    step=step_idx,
                    temperature=temp,
                    prompt=prompt,
                    response=response,
                )
            )
            previous_response = response

        # Compute convergence score between last two steps
        convergence: float = 0.0
        if len(steps) >= 2:
            a: str = steps[-2].response
            b: str = steps[-1].response
            set_a: set[str] = set(a)
            set_b: set[str] = set(b)
            union_size: int = len(set_a | set_b)
            convergence = len(set_a & set_b) / max(union_size, 1)

        return DiffCoTResult(
            final_answer=previous_response,
            steps=steps,
            convergence_score=convergence,
            n_steps=len(steps),
        )


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------


class DiffCoTEvaluator(Evaluator):
    """Wraps any :class:`Evaluator` to use DiffCoT reasoning before scoring.

    Instead of calling ``agent.run(task)`` directly, this evaluator runs the
    DiffCoT denoising loop and feeds the *final* refined answer to the
    base evaluator for scoring.

    Args:
        base_evaluator: The underlying evaluator that scores responses.
        backend: LLM backend used by the :class:`DiffCoTReasoner`.
        config: DiffCoT configuration; defaults to :class:`DiffCoTConfig`
            with its default field values if ``None``.
    """

    def __init__(
        self,
        base_evaluator: Evaluator,
        backend: LLMBackend,
        config: DiffCoTConfig | None = None,
    ) -> None:
        self._base_evaluator = base_evaluator
        self._reasoner = DiffCoTReasoner(backend=backend, config=config)

    def evaluate(self, agent: "Agent", task: str) -> float:
        """Evaluate *agent* on *task* using DiffCoT-refined reasoning.

        Runs the full DiffCoT denoising loop, then asks the base evaluator
        to score the final answer.  A lightweight proxy agent whose
        ``run()`` method returns the DiffCoT final answer is passed to
        the base evaluator so that the base evaluator's interface is
        satisfied without mutation of the original agent.

        Args:
            agent: The agent whose genome and backend are used for reasoning.
            task: Natural-language task description.

        Returns:
            Fitness score in ``[0.0, 1.0]`` as returned by the base evaluator.
        """
        # Run the denoising loop using the agent's genome
        result: DiffCoTResult = self._reasoner.reason(agent.genome, task)

        # Build a thin proxy agent that returns the DiffCoT answer
        proxy = _ProxyAgent(agent, result.final_answer)
        return self._base_evaluator.evaluate(proxy, task)  # type: ignore[arg-type]


class _ProxyAgent:
    """Minimal duck-typed proxy that makes the DiffCoT answer available via run().

    This avoids importing Agent at runtime (preventing circular imports) while
    still satisfying the ``agent.run(task) -> str`` contract that evaluators rely
    on when they call the agent directly.
    """

    def __init__(self, original: "Agent", fixed_response: str) -> None:
        self._original = original
        self._fixed_response = fixed_response
        # Expose attributes that evaluators commonly read
        self.genome = original.genome
        self.agent_id = original.agent_id
        self.id = original.id
        self._fitness = original.fitness
        self._generation = original.generation

    def run(self, task: str) -> str:  # noqa: ARG002
        """Return the pre-computed DiffCoT final answer."""
        return self._fixed_response

    @property
    def fitness(self) -> float | None:
        """Delegate to original agent's fitness."""
        return self._original.fitness


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_diffcot_evaluator(
    base_evaluator: Evaluator,
    backend: LLMBackend,
    n_steps: int = 3,
    noise_level: float = 0.3,
) -> DiffCoTEvaluator:
    """Convenience factory for :class:`DiffCoTEvaluator`.

    Args:
        base_evaluator: Evaluator to wrap.
        backend: LLM backend for denoising steps.
        n_steps: Number of denoising iterations.
        noise_level: Noise magnitude injected in early steps.

    Returns:
        A configured :class:`DiffCoTEvaluator`.
    """
    config = DiffCoTConfig(n_steps=n_steps, noise_level=noise_level)
    return DiffCoTEvaluator(
        base_evaluator=base_evaluator,
        backend=backend,
        config=config,
    )
