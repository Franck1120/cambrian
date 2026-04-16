import type { Plugin } from '../types'

export const PLUGINS: Plugin[] = [
  // MUTATION
  { id: 'llm-mutator', name: 'LLM Mutator', icon: 'Dna', description: 'Uses an LLM to generate diverse prompt mutations while preserving semantic intent.', category: 'mutation', enabled: true, impact: 0.87, order: 1, tags: ['core', 'llm'] },
  { id: 'template-inject', name: 'Template Injection', icon: 'FileCode', description: 'Injects strategy templates (CoT, ReAct, Reflexion) into agent genomes during mutation.', category: 'mutation', enabled: true, impact: 0.72, order: 2, tags: ['templates'] },
  { id: 'semantic-mutator', name: 'Semantic Mutator', icon: 'Sparkles', description: 'Mutates prompts while maintaining semantic distance within a configurable range.', category: 'mutation', enabled: false, impact: 0.61, order: 3, tags: ['semantic'] },
  { id: 'persona-mutator', name: 'Persona Mutator', icon: 'UserCog', description: 'Injects expert persona descriptions to shape agent behavior and domain expertise.', category: 'mutation', enabled: true, impact: 0.65, order: 4, tags: ['persona', 'llm'] },
  { id: 'chain-builder', name: 'Chain Builder', icon: 'Link', description: 'Assembles chain-of-thought scaffolding around existing genome content.', category: 'mutation', enabled: false, impact: null, order: 5, tags: ['cot'] },
  { id: 'diff-mutator', name: 'Diff Mutator', icon: 'GitDiff', description: 'Applies minimal targeted diffs to genomes, preserving most structure while exploring neighbors.', category: 'mutation', enabled: false, impact: 0.58, order: 6, tags: ['minimal'] },
  { id: 'exemplar-injector', name: 'Exemplar Injector', icon: 'BookOpen', description: 'Injects few-shot examples from high-fitness ancestors into mutated prompts.', category: 'mutation', enabled: true, impact: 0.79, order: 7, tags: ['few-shot'] },
  { id: 'temperature-sweep', name: 'Temperature Sweep', icon: 'Thermometer', description: 'Generates variants at multiple temperature values and selects the most distinct.', category: 'mutation', enabled: false, impact: null, order: 8, tags: ['sampling'] },
  { id: 'constraint-mutator', name: 'Constraint Mutator', icon: 'Shield', description: 'Adds explicit constraints and guardrails to prompts during mutation.', category: 'mutation', enabled: false, impact: 0.44, order: 9, tags: ['safety'] },
  { id: 'style-mutator', name: 'Style Mutator', icon: 'Brush', description: 'Varies communication style (concise, verbose, formal, casual) while preserving instructions.', category: 'mutation', enabled: false, impact: 0.38, order: 10, tags: ['style'] },

  // SELECTION
  { id: 'tournament-sel', name: 'Tournament Selection', icon: 'Trophy', description: 'Selects parents by running random tournaments. Controls selection pressure via k parameter.', category: 'selection', enabled: true, impact: 0.91, order: 11, tags: ['core'] },
  { id: 'roulette-sel', name: 'Roulette Wheel', icon: 'CircleDot', description: 'Fitness-proportional selection. Higher fitness = higher probability of being selected.', category: 'selection', enabled: false, impact: 0.74, order: 12, tags: ['classic'] },
  { id: 'rank-sel', name: 'Rank Selection', icon: 'ListOrdered', description: 'Ranks agents by fitness and selects based on rank. More robust to fitness scaling issues.', category: 'selection', enabled: false, impact: 0.68, order: 13, tags: ['classic'] },
  { id: 'nsga-ii', name: 'NSGA-II', icon: 'Network', description: 'Multi-objective selection using non-dominated sorting and crowding distance (Pareto fronts).', category: 'selection', enabled: false, impact: 0.83, order: 14, tags: ['multi-objective', 'pareto'] },
  { id: 'elitism', name: 'Elitism', icon: 'Star', description: 'Guarantees top-k agents survive to the next generation unchanged.', category: 'selection', enabled: true, impact: 0.88, order: 15, tags: ['core'] },
  { id: 'stochastic-sel', name: 'SUS', icon: 'Shuffle', description: 'Stochastic Universal Sampling: evenly spaced selection for better coverage of fitness landscape.', category: 'selection', enabled: false, impact: 0.71, order: 16, tags: ['advanced'] },
  { id: 'boltzmann-sel', name: 'Boltzmann Selection', icon: 'Zap', description: 'Temperature-annealed selection: exploratory early, exploitative late in evolution.', category: 'selection', enabled: false, impact: null, order: 17, tags: ['annealing'] },
  { id: 'lexicase-sel', name: 'Lexicase Selection', icon: 'Filter', description: 'Selects agents that perform best on a random shuffle of test cases. Great for diverse niches.', category: 'selection', enabled: false, impact: 0.76, order: 18, tags: ['test-based'] },

  // MEMORY
  { id: 'episodic-mem', name: 'Episodic Memory', icon: 'Brain', description: 'Injects the agent\'s best past runs as context during evaluation, enabling learning from experience.', category: 'memory', enabled: false, impact: 0.69, order: 19, tags: ['context'] },
  { id: 'ancestor-mem', name: 'Ancestor Memory', icon: 'GitFork', description: 'Provides agents with high-fitness ancestral genomes as reference during generation.', category: 'memory', enabled: false, impact: 0.62, order: 20, tags: ['lineage'] },
  { id: 'experience-replay', name: 'Experience Replay', icon: 'RotateCcw', description: 'Replays a random sample of past high-scoring evaluations to stabilize fitness estimates.', category: 'memory', enabled: false, impact: 0.55, order: 21, tags: ['replay'] },
  { id: 'failure-mem', name: 'Failure Memory', icon: 'AlertCircle', description: 'Records failed patterns and injects them as negative examples to avoid.', category: 'memory', enabled: false, impact: null, order: 22, tags: ['negative-examples'] },
  { id: 'working-mem', name: 'Working Memory', icon: 'Cpu', description: 'Maintains a scratchpad of intermediate results that persists across evaluation steps.', category: 'memory', enabled: false, impact: 0.73, order: 23, tags: ['scratchpad'] },

  // DIVERSITY
  { id: 'diversity-niching', name: 'Fitness Sharing', icon: 'Share2', description: 'Penalizes agents that are too similar to neighbors, maintaining population diversity.', category: 'diversity', enabled: true, impact: 0.81, order: 24, tags: ['core'] },
  { id: 'island-model', name: 'Island Model', icon: 'Globe', description: 'Partitions population into isolated subpopulations with periodic migration between islands.', category: 'diversity', enabled: false, impact: 0.77, order: 25, tags: ['topology'] },
  { id: 'crowding', name: 'Deterministic Crowding', icon: 'Users', description: 'Offspring compete with most-similar parents, preserving multiple fitness peaks.', category: 'diversity', enabled: false, impact: 0.66, order: 26, tags: ['niching'] },
  { id: 'novelty-search', name: 'Novelty Search', icon: 'Compass', description: 'Rewards novelty of behavior rather than fitness, enabling exploration of deceptive landscapes.', category: 'diversity', enabled: false, impact: null, order: 27, tags: ['exploration'] },
  { id: 'quality-diversity', name: 'Quality-Diversity', icon: 'BarChart2', description: 'MAP-Elites variant: maintains a diverse archive of high-quality solutions across behavioral dimensions.', category: 'diversity', enabled: false, impact: 0.84, order: 28, tags: ['map-elites', 'advanced'] },
  { id: 'entropy-reg', name: 'Entropy Regularization', icon: 'Activity', description: 'Adds an entropy bonus to fitness scores to discourage premature convergence.', category: 'diversity', enabled: false, impact: 0.59, order: 29, tags: ['regularization'] },

  // EVALUATION
  { id: 'fitness-eval', name: 'LLM Judge', icon: 'Scale', description: 'Uses a separate LLM to evaluate response quality on configurable criteria and scoring rubric.', category: 'evaluation', enabled: true, impact: 0.93, order: 30, tags: ['core', 'llm'] },
  { id: 'test-executor', name: 'Test Executor', icon: 'PlayCircle', description: 'Runs automated test cases against generated code and scores by pass rate.', category: 'evaluation', enabled: true, impact: 0.96, order: 31, tags: ['code', 'testing'] },
  { id: 'semantic-sim', name: 'Semantic Similarity', icon: 'GitCompare', description: 'Computes cosine similarity between output embeddings and target embeddings.', category: 'evaluation', enabled: false, impact: 0.67, order: 32, tags: ['embeddings'] },
  { id: 'bleu-eval', name: 'BLEU Score', icon: 'Calculator', description: 'n-gram overlap metric for evaluating text generation quality against reference outputs.', category: 'evaluation', enabled: false, impact: 0.52, order: 33, tags: ['nlp', 'classic'] },
  { id: 'human-feedback', name: 'Human Feedback', icon: 'MessageSquare', description: 'Pauses evolution to collect human preference ratings on candidate outputs.', category: 'evaluation', enabled: false, impact: null, order: 34, tags: ['rlhf', 'interactive'] },
  { id: 'multi-eval', name: 'Multi-Evaluator', icon: 'Layers', description: 'Runs multiple evaluators and aggregates scores via weighted voting.', category: 'evaluation', enabled: false, impact: 0.88, order: 35, tags: ['ensemble'] },
  { id: 'adversarial-eval', name: 'Adversarial Evaluation', icon: 'Sword', description: 'Evaluates agents against adversarially constructed edge cases designed to reveal failures.', category: 'evaluation', enabled: false, impact: null, order: 36, tags: ['robustness'] },
  { id: 'consistency-eval', name: 'Consistency Check', icon: 'CheckSquare', description: 'Evaluates semantic consistency of the agent across multiple paraphrases of the same input.', category: 'evaluation', enabled: false, impact: 0.74, order: 37, tags: ['robustness'] },

  // CROSSOVER
  { id: 'genome-crossover', name: 'Genome Crossover', icon: 'Scissors', description: 'Splices two parent genomes at a random crossover point to create a child genome.', category: 'crossover', enabled: false, impact: 0.70, order: 38, tags: ['classic'] },
  { id: 'semantic-cross', name: 'Semantic Crossover', icon: 'Merge', description: 'LLM merges two parent genomes while preserving the best semantic content from each.', category: 'crossover', enabled: false, impact: 0.78, order: 39, tags: ['llm', 'semantic'] },
  { id: 'prompt-blend', name: 'Prompt Blend', icon: 'Blend', description: 'Creates child prompts by interpolating between parent prompt representations.', category: 'crossover', enabled: false, impact: 0.65, order: 40, tags: ['interpolation'] },
  { id: 'uniform-cross', name: 'Uniform Crossover', icon: 'ToggleLeft', description: 'Each segment of the child genome is independently drawn from either parent with equal probability.', category: 'crossover', enabled: false, impact: 0.63, order: 41, tags: ['classic'] },
  { id: 'archipelago-cross', name: 'Archipelago Exchange', icon: 'ArrowLeftRight', description: 'Migrates elite genomes between island subpopulations for inter-island genetic exchange.', category: 'crossover', enabled: false, impact: 0.76, order: 42, tags: ['islands'] },
  { id: 'block-cross', name: 'Block Crossover', icon: 'LayoutGrid', description: 'Identifies functional blocks in prompts and swaps entire blocks between parents.', category: 'crossover', enabled: false, impact: null, order: 43, tags: ['structural'] },
]
