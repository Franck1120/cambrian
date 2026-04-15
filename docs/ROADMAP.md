# Cambrian Roadmap

> Living document — updated with each release.  
> Items marked ✅ are shipped. Items marked 🔜 are planned.

---

## Shipped

### v1.0.x — Foundation + Tier 5

✅ Core evolution engine (EvolutionEngine, LLMMutator, Evaluator)  
✅ Multi-backend support (OpenAI, Anthropic, Gemini, Ollama)  
✅ Forge mode — code genome + pipeline evolution  
✅ DPO selection (DPOPair, DPOSelector, DPOTrainer)  
✅ Diff-CoT reasoning (DiffCoTReasoner, DiffCoTEvaluator)  
✅ Causal mutation (CausalGraph, CausalMutator)  
✅ Tool creation (ToolInventor, ToolPopulationRegistry)  
✅ Self-play & tournament evaluation  
✅ Meta-evolution of hyperparameters  
✅ World model evaluation  
✅ Dream phase (speculative rollout)  
✅ Quorum sensing (diversity collapse detection)  
✅ Mixture of Agents + Quantum Tunneling  
✅ Reflexion evaluator  
✅ Symbiotic fusion  
✅ Hormesis adapter (stress-induced improvement)  
✅ Apoptosis controller (population pruning)  
✅ Catalysis engine (cross-domain pollination)  
✅ LLM Cascade (tiered inference cost routing)  
✅ Ensemble (Boosting, voting)  
✅ Glossolalia reasoning  
✅ Inference scaling (BestOfN, BeamSearch)  
✅ Transfer learning adapter  
✅ Tabu search mutator  
✅ Simulated annealing selector  
✅ Red-team robustness evaluator  
✅ Zeitgeber clock (circadian scheduling)  
✅ Horizontal gene transfer  
✅ Transgenerational epigenetics  
✅ Immune memory (B/T-cell)  
✅ Neuromodulation bank  
✅ Metamorphosis controller  
✅ Ecosystem evaluator  
✅ Fractal evolution  
✅ Safeguards (drift detection, anomaly detection)  
✅ Lamarckian few-shot capture  
✅ Archipelago (island model evolution)  
✅ Semantic cache + model router  
✅ py.typed — fully typed package (PEP 561)  
✅ 2 000+ tests, mypy strict, ruff clean  

---

## Planned

### v1.1 — Observability & Persistence

🔜 **Evolution dashboard** — real-time Streamlit fitness/diversity charts  
🔜 **Checkpoint / resume** — save and reload evolution state mid-run  
🔜 **Structured logging** — JSONL run logs for post-hoc analysis  
🔜 **Wandb integration** — first-class Weights & Biases experiment tracking  
🔜 **MLflow integration** — log genomes, metrics, artifacts  

### v1.2 — Deployment & Serving

🔜 **REST API server** — serve best agent via FastAPI endpoint  
🔜 **Docker image** — `ghcr.io/cambrian-ai/cambrian:latest` with all backends  
🔜 **Kubernetes operator** — CRD for declarative evolution jobs  
🔜 **One-click Render/Railway deploy** — from CLI to live endpoint in one command  

### v1.3 — Multi-Agent & Collaboration

🔜 **A2A mesh** — full agent-to-agent protocol with discovery and delegation  
🔜 **Shared memory pool** — cross-population knowledge sharing  
🔜 **Asynchronous evolution** — non-blocking fitness evaluation at scale  
🔜 **Distributed archipelago** — island model across multiple machines  

### v1.4 — Advanced Learning

🔜 **RLHF loop** — human preference signal integrated into fitness  
🔜 **Constitutional AI filter** — harm/alignment guardrails on mutated genomes  
🔜 **Curriculum learning** — task difficulty scheduling over generations  
🔜 **Reward shaping** — shaped fitness signals for sparse-reward tasks  
🔜 **Co-evolution** — adversarial population pairs (agent vs. evaluator)  

### v2.0 — Open Platform

🔜 **Plugin registry** — community-contributed mutators, evaluators, backends  
🔜 **Genome marketplace** — share and download evolved agent genomes  
🔜 **Cloud execution** — managed evolution runs (no local GPU/API key needed)  
🔜 **GUI** — visual genome editor and evolution explorer  

---

## Contributing

Have a feature idea? [Open a feature request](https://github.com/Franck1120/cambrian/issues/new?template=feature_request.md).  
Want to implement something on this roadmap? Check the issue tracker and open a PR.
