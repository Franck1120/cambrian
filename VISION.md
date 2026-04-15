# Vision — Cambrian

> *"Life finds a way." — Ian Malcolm, Jurassic Park*

---

## The Problem We're Solving

Building capable AI agents today is manual, brittle, and slow.

You write a system prompt. You test it. It fails on edge cases. You tweak
it. You repeat this loop for hours, days, weeks — guided by intuition,
luck, and the sheer will to keep going.

There is no principled search. No way to know if you've found the optimum.
No mechanism to let agents improve themselves. No force that pushes them
toward competence the way evolution pushes biology.

We think this is wrong. AI agents should evolve.

---

## The Core Insight

Darwin's core insight was this: **given variation, selection, and
inheritance, adaptation is inevitable.**

The same logic applies to AI agents:

- **Variation** — LLMs can generate diverse mutations of any system prompt
- **Selection** — any evaluator (code tests, human judgment, LLM judges)
  can rank agents by fitness
- **Inheritance** — genomes pass from parents to offspring, preserving
  what works

Chain these three together and you get an engine that **finds capable
agents automatically**, without gradient descent, without labelled datasets,
without hand-engineering.

Cambrian is that engine.

---

## The Roadmap: From Tools to Autonomy

We think about this in three eras:

### Era 1 — Prompt Evolution (Today, v1.x)

Cambrian already does this well. You give it a task. It evolves system
prompts until agents are demonstrably capable. You export the winner.

This is already useful. Prompt engineering is the biggest bottleneck in
deploying LLMs. Automating it cuts development time from days to minutes.

**What exists:**
- 50+ bio-inspired evolutionary operators
- DPO preference selection
- Goal-drift safeguards
- Tournament / self-play competition
- Island models (Archipelago)
- Code evolution (Forge)
- Pipeline evolution
- CLI, dashboard, export

### Era 2 — Ecosystem Intelligence (Near, v2.x)

The next step is to let agents not just compete but **specialise and
cooperate**. Nature solved the combinatorial explosion of niches by
evolving ecosystems, not individuals. We're doing the same.

**What comes next:**
- Persistent agent ecosystems — herbivores, predators, decomposers
  co-existing and co-evolving across thousands of tasks
- Memory transfer — agents that recall past experiences and build
  on each other's learnings (horizontal gene transfer at semantic scale)
- Emergent roles — agents that discover their own niches through
  interaction, not top-down assignment
- Multi-task co-evolution — populations that simultaneously solve
  multiple tasks, trading off specialisation and generality

### Era 3 — Autonomous Self-Improvement (Future, v3.x)

The long-term vision is an AI system that improves itself without human
intervention — a continuous evolutionary loop that runs in the background,
discovering more capable agents over time.

**The long game:**
- Agents that invent new tools and pass them to offspring
- Meta-cognition — agents that understand their own failure modes
- World-model evolution — agents that build and test predictive models
  of their environment before acting
- Constitutional self-correction — populations that develop their own
  norms and correct drift autonomously

This is not science fiction. Each component exists in Cambrian today as
a module. The question is orchestration.

---

## Why It Matters

The bottleneck for AI is no longer raw capability — it's **alignment to
specific tasks and contexts**. GPT-4, Claude, Gemini are powerful. But a
generic model is not the same as a model tuned for your task.

Today, that tuning is done by humans, manually. Tomorrow, it should be
done by evolution, automatically.

An organisation that can evolve capable agents faster than its competitors
has a permanent advantage. Not because of the model it uses — but because
of the *selection pressure* it applies.

Cambrian is that selection pressure.

---

## The Name

The Cambrian Explosion (~541 million years ago) was the most dramatic
diversification event in the history of life. In less than 25 million
years, almost every major animal body plan appeared for the first time.

The trigger was likely a combination of environmental change, increased
oxygen, and — crucially — **the evolution of eyes**. Once organisms could
see each other, predation and arms races accelerated evolution dramatically.

We think AI is at its own Cambrian moment. Models can now evaluate each
other, judge each other, and compete with each other. That's our version
of eyes. The explosion follows.

---

## Guiding Principles

1. **Evolution over engineering.** Don't hand-tune what can be evolved.
2. **Pressure creates capability.** Competition, not instruction, drives
   improvement.
3. **Diversity is infrastructure.** Monocultures collapse. Ecosystems adapt.
4. **Transparency first.** Safeguards, drift detection, and lineage
   tracking are not optional — they're core.
5. **Open by default.** MIT licence. No telemetry. No lock-in.

---

## Contributing

If this vision resonates, we'd love your help. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) or open an issue.

The Cambrian explosion took 25 million years. We're aiming for faster.

---

*Cambrian v1.0.2 · [GitHub](https://github.com/Franck1120/cambrian)*
