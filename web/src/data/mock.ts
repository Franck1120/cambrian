import type { Agent, GenerationStat, EvolutionRun, PipelineStep } from '../types'

const GENOME_REFLEXION = `You are an expert software engineer specializing in elegant, correct code. When presented with a coding task:

1. ANALYZE the requirements carefully — identify edge cases and constraints
2. DRAFT an initial solution with clean code
3. CRITIQUE your own solution: Is it correct? Efficient? Handling edge cases?
4. REVISE based on your critique
5. FINALIZE with well-documented, production-ready code

Always prefer clarity over cleverness. Show your reasoning at each step.`

const GENOME_COT = `You are a methodical programmer who solves problems step by step. For every coding problem:

STEP 1: Understand — restate the problem in your own words
STEP 2: Plan — outline the algorithm before writing code
STEP 3: Implement — write clean, readable code following your plan
STEP 4: Verify — trace through with an example to confirm correctness

Be explicit about your reasoning. Write code that a junior developer can understand.`

const GENOME_TOT = `You are an expert programmer who explores multiple solution paths before committing. When solving a coding problem:

- Branch A: Consider the naive/direct approach
- Branch B: Consider an optimized approach
- Branch C: Consider edge cases and defensive coding

Evaluate each branch on correctness, performance, and maintainability. Choose the best path and explain why. Implement thoroughly.`

const GENOME_REACT = `You are a pragmatic software engineer who uses structured reasoning and action. When given a coding task:

Thought: What is being asked? What do I know?
Action: Write a solution
Observation: Does it handle edge cases? Is it efficient?
Thought: What can be improved?
Action: Refine the solution
Final Answer: Clean, complete implementation

Be concise in your reasoning, thorough in your implementation.`

const GENOME_DIRECT = `Write correct, efficient code. No preamble. No explanation unless asked. Just the implementation.`

const GENOME_MOA = `You are a mixture of expert agents. Approach each problem with multiple perspectives:

ENGINEER: Focus on correctness and edge cases
OPTIMIZER: Focus on time/space complexity
REVIEWER: Check for bugs and code quality

Synthesize the best solution from all perspectives. Output clean, production-ready code with brief inline comments where logic is non-obvious.`

export const MOCK_AGENTS: Agent[] = [
  {
    id: 'a-001', generation: 8, fitness: 0.94, temperature: 0.7,
    strategy: 'reflexion', prompt_tokens: 312, status: 'elite',
    parent_id: 'a-017',
    genome: GENOME_REFLEXION,
    fitness_history: [0.45, 0.58, 0.67, 0.74, 0.80, 0.86, 0.91, 0.94],
    plugins_active: ['llm-mutator', 'tournament-sel', 'fitness-eval'],
    run_logs: [
      { id: 'l1', input: 'reverse("hello")', output: '"olleh"', fitness: 0.95, timestamp: '14:22:01', tokens_used: 89 },
      { id: 'l2', input: 'reverse("")', output: '""', fitness: 0.98, timestamp: '14:22:14', tokens_used: 72 },
      { id: 'l3', input: 'reverse("a")', output: '"a"', fitness: 0.91, timestamp: '14:22:28', tokens_used: 65 },
    ],
  },
  {
    id: 'a-002', generation: 8, fitness: 0.91, temperature: 0.5,
    strategy: 'chain-of-thought', prompt_tokens: 287, status: 'elite',
    parent_id: 'a-018',
    genome: GENOME_COT,
    fitness_history: [0.38, 0.52, 0.62, 0.70, 0.77, 0.83, 0.87, 0.91],
    plugins_active: ['llm-mutator', 'tournament-sel'],
    run_logs: [
      { id: 'l4', input: 'reverse("world")', output: '"dlrow"', fitness: 0.92, timestamp: '14:22:02', tokens_used: 94 },
      { id: 'l5', input: 'reverse("12345")', output: '"54321"', fitness: 0.90, timestamp: '14:22:16', tokens_used: 88 },
    ],
  },
  {
    id: 'a-003', generation: 7, fitness: 0.88, temperature: 0.8,
    strategy: 'tree-of-thought', prompt_tokens: 401, status: 'active',
    parent_id: 'a-015',
    genome: GENOME_TOT,
    fitness_history: [0.40, 0.55, 0.63, 0.71, 0.78, 0.83, 0.88],
    plugins_active: ['llm-mutator', 'fitness-eval', 'diversity-niching'],
    run_logs: [
      { id: 'l6', input: 'reverse("Python")', output: '"nohtyP"', fitness: 0.89, timestamp: '14:21:55', tokens_used: 128 },
    ],
  },
  {
    id: 'a-004', generation: 8, fitness: 0.85, temperature: 0.6,
    strategy: 'react', prompt_tokens: 256, status: 'active',
    parent_id: 'a-019',
    genome: GENOME_REACT,
    fitness_history: [0.35, 0.48, 0.58, 0.66, 0.73, 0.78, 0.82, 0.85],
    plugins_active: ['llm-mutator', 'tournament-sel'],
    run_logs: [
      { id: 'l7', input: 'reverse("abc")', output: '"cba"', fitness: 0.86, timestamp: '14:22:04', tokens_used: 76 },
      { id: 'l8', input: 'reverse("  ")', output: '"  "', fitness: 0.84, timestamp: '14:22:18', tokens_used: 71 },
    ],
  },
  {
    id: 'a-005', generation: 6, fitness: 0.81, temperature: 0.9,
    strategy: 'reflexion', prompt_tokens: 345, status: 'active',
    parent_id: 'a-012',
    genome: GENOME_REFLEXION,
    fitness_history: [0.42, 0.55, 0.64, 0.72, 0.78, 0.81],
    plugins_active: ['llm-mutator', 'genome-crossover'],
    run_logs: [
      { id: 'l9', input: 'reverse("test")', output: '"tset"', fitness: 0.82, timestamp: '14:21:40', tokens_used: 102 },
    ],
  },
  {
    id: 'a-006', generation: 5, fitness: 0.73, temperature: 1.0,
    strategy: 'direct', prompt_tokens: 198, status: 'dead',
    parent_id: 'a-010',
    genome: GENOME_DIRECT,
    fitness_history: [0.33, 0.44, 0.56, 0.65, 0.73],
    plugins_active: ['llm-mutator'],
    run_logs: [
      { id: 'l10', input: 'reverse("x")', output: '"x"', fitness: 0.74, timestamp: '14:21:20', tokens_used: 45 },
    ],
  },
  {
    id: 'a-007', generation: 7, fitness: 0.79, temperature: 0.4,
    strategy: 'chain-of-thought', prompt_tokens: 412, status: 'active',
    parent_id: 'a-014',
    genome: GENOME_COT,
    fitness_history: [0.36, 0.50, 0.60, 0.68, 0.74, 0.77, 0.79],
    plugins_active: ['llm-mutator', 'tournament-sel', 'fitness-eval'],
    run_logs: [
      { id: 'l11', input: 'reverse("hello world")', output: '"dlrow olleh"', fitness: 0.80, timestamp: '14:21:58', tokens_used: 134 },
    ],
  },
  {
    id: 'a-008', generation: 8, fitness: 0.82, temperature: 0.7,
    strategy: 'moa', prompt_tokens: 534, status: 'active',
    parent_id: 'a-016',
    genome: GENOME_MOA,
    fitness_history: [0.41, 0.54, 0.63, 0.70, 0.75, 0.78, 0.80, 0.82],
    plugins_active: ['llm-mutator', 'tournament-sel', 'diversity-niching'],
    run_logs: [
      { id: 'l12', input: 'reverse("123abc")', output: '"cba321"', fitness: 0.83, timestamp: '14:22:06', tokens_used: 167 },
      { id: 'l13', input: 'reverse("")', output: '""', fitness: 0.97, timestamp: '14:22:20', tokens_used: 142 },
    ],
  },
]

export const MOCK_GENERATIONS: GenerationStat[] = [
  { generation: 1, best_fitness: 0.42, avg_fitness: 0.31, diversity: 0.92 },
  { generation: 2, best_fitness: 0.55, avg_fitness: 0.43, diversity: 0.86 },
  { generation: 3, best_fitness: 0.64, avg_fitness: 0.52, diversity: 0.79 },
  { generation: 4, best_fitness: 0.72, avg_fitness: 0.60, diversity: 0.71 },
  { generation: 5, best_fitness: 0.78, avg_fitness: 0.66, diversity: 0.65 },
  { generation: 6, best_fitness: 0.83, avg_fitness: 0.71, diversity: 0.58 },
  { generation: 7, best_fitness: 0.88, avg_fitness: 0.77, diversity: 0.51 },
  { generation: 8, best_fitness: 0.94, avg_fitness: 0.83, diversity: 0.44 },
]

export const MOCK_HISTORY: EvolutionRun[] = [
  { id: 'run-001', task: 'Write a Python function that reverses a string', generations: 10, population: 8, best_fitness: 0.94, created_at: '2026-04-14 14:22', duration_s: 183 },
  { id: 'run-002', task: 'Generate a SQL query to find duplicate emails', generations: 8, population: 6, best_fitness: 0.89, created_at: '2026-04-14 11:05', duration_s: 127 },
  { id: 'run-003', task: 'Summarize a research paper in 3 bullet points', generations: 12, population: 10, best_fitness: 0.91, created_at: '2026-04-13 17:44', duration_s: 241 },
  { id: 'run-004', task: 'Write a regex to validate email addresses', generations: 6, population: 8, best_fitness: 0.86, created_at: '2026-04-13 09:18', duration_s: 98 },
  { id: 'run-005', task: 'Explain recursion to a 10-year-old', generations: 8, population: 6, best_fitness: 0.92, created_at: '2026-04-12 20:30', duration_s: 155 },
]

export const MOCK_PIPELINE: PipelineStep[] = [
  { id: 's-1', type: 'mutate', label: 'LLM Mutator', params: { model: 'llama-3.3-70b', temperature: 0.9, strategies: 'reflexion,cot,react' }, enabled: true },
  { id: 's-2', type: 'evaluate', label: 'Fitness Evaluator', params: { evaluator: 'llm_judge', retries: 2 }, enabled: true },
  { id: 's-3', type: 'select', label: 'Tournament Selection', params: { k: 3, elitism: 2 }, enabled: true },
  { id: 's-4', type: 'crossover', label: 'Genome Crossover', params: { rate: 0.3, blend_prompts: true }, enabled: false },
  { id: 's-5', type: 'filter', label: 'Apoptosis Filter', params: { threshold: 0.3, keep_min: 2 }, enabled: true },
]

// Extended agents for the generation timeline (all gens)
export const MOCK_ALL_AGENTS: Agent[] = [
  // Gen 1
  { id: 'g1-a1', generation: 1, fitness: 0.42, temperature: 1.0, strategy: 'direct', prompt_tokens: 120, status: 'dead', parent_id: null, genome: GENOME_DIRECT, fitness_history: [0.42], plugins_active: [], run_logs: [] },
  { id: 'g1-a2', generation: 1, fitness: 0.38, temperature: 0.9, strategy: 'direct', prompt_tokens: 115, status: 'dead', parent_id: null, genome: GENOME_DIRECT, fitness_history: [0.38], plugins_active: [], run_logs: [] },
  { id: 'g1-a3', generation: 1, fitness: 0.31, temperature: 0.8, strategy: 'cot', prompt_tokens: 210, status: 'dead', parent_id: null, genome: GENOME_COT, fitness_history: [0.31], plugins_active: [], run_logs: [] },
  { id: 'g1-a4', generation: 1, fitness: 0.28, temperature: 1.1, strategy: 'direct', prompt_tokens: 98, status: 'dead', parent_id: null, genome: GENOME_DIRECT, fitness_history: [0.28], plugins_active: [], run_logs: [] },
  // Gen 2
  { id: 'g2-a1', generation: 2, fitness: 0.55, temperature: 0.9, strategy: 'reflexion', prompt_tokens: 280, status: 'dead', parent_id: 'g1-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55], plugins_active: [], run_logs: [] },
  { id: 'g2-a2', generation: 2, fitness: 0.48, temperature: 0.8, strategy: 'cot', prompt_tokens: 240, status: 'dead', parent_id: 'g1-a1', genome: GENOME_COT, fitness_history: [0.42, 0.48], plugins_active: [], run_logs: [] },
  { id: 'g2-a3', generation: 2, fitness: 0.41, temperature: 0.7, strategy: 'direct', prompt_tokens: 130, status: 'dead', parent_id: 'g1-a2', genome: GENOME_DIRECT, fitness_history: [0.38, 0.41], plugins_active: [], run_logs: [] },
  { id: 'g2-a4', generation: 2, fitness: 0.36, temperature: 1.0, strategy: 'react', prompt_tokens: 210, status: 'dead', parent_id: 'g1-a3', genome: GENOME_REACT, fitness_history: [0.31, 0.36], plugins_active: [], run_logs: [] },
  // Gen 3
  { id: 'g3-a1', generation: 3, fitness: 0.64, temperature: 0.8, strategy: 'reflexion', prompt_tokens: 300, status: 'dead', parent_id: 'g2-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55, 0.64], plugins_active: [], run_logs: [] },
  { id: 'g3-a2', generation: 3, fitness: 0.58, temperature: 0.7, strategy: 'cot', prompt_tokens: 260, status: 'dead', parent_id: 'g2-a1', genome: GENOME_COT, fitness_history: [0.42, 0.48, 0.58], plugins_active: [], run_logs: [] },
  { id: 'g3-a3', generation: 3, fitness: 0.49, temperature: 0.9, strategy: 'tot', prompt_tokens: 380, status: 'dead', parent_id: 'g2-a2', genome: GENOME_TOT, fitness_history: [0.42, 0.48, 0.49], plugins_active: [], run_logs: [] },
  { id: 'g3-a4', generation: 3, fitness: 0.44, temperature: 0.6, strategy: 'react', prompt_tokens: 230, status: 'dead', parent_id: 'g2-a4', genome: GENOME_REACT, fitness_history: [0.31, 0.36, 0.44], plugins_active: [], run_logs: [] },
  // Gen 4
  { id: 'g4-a1', generation: 4, fitness: 0.72, temperature: 0.75, strategy: 'reflexion', prompt_tokens: 310, status: 'dead', parent_id: 'g3-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55, 0.64, 0.72], plugins_active: [], run_logs: [] },
  { id: 'g4-a2', generation: 4, fitness: 0.66, temperature: 0.65, strategy: 'cot', prompt_tokens: 275, status: 'dead', parent_id: 'g3-a1', genome: GENOME_COT, fitness_history: [0.42, 0.55, 0.58, 0.66], plugins_active: [], run_logs: [] },
  { id: 'g4-a3', generation: 4, fitness: 0.60, temperature: 0.8, strategy: 'tot', prompt_tokens: 395, status: 'dead', parent_id: 'g3-a2', genome: GENOME_TOT, fitness_history: [0.42, 0.48, 0.49, 0.60], plugins_active: [], run_logs: [] },
  { id: 'g4-a4', generation: 4, fitness: 0.53, temperature: 0.7, strategy: 'moa', prompt_tokens: 490, status: 'dead', parent_id: 'g3-a3', genome: GENOME_MOA, fitness_history: [0.42, 0.48, 0.49, 0.53], plugins_active: [], run_logs: [] },
  // Gen 5
  { id: 'g5-a1', generation: 5, fitness: 0.78, temperature: 0.7, strategy: 'reflexion', prompt_tokens: 320, status: 'dead', parent_id: 'g4-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55, 0.64, 0.72, 0.78], plugins_active: [], run_logs: [] },
  { id: 'g5-a2', generation: 5, fitness: 0.73, temperature: 0.6, strategy: 'cot', prompt_tokens: 280, status: 'dead', parent_id: 'g4-a1', genome: GENOME_COT, fitness_history: [0.42, 0.55, 0.58, 0.66, 0.73], plugins_active: [], run_logs: [] },
  { id: 'g5-a3', generation: 5, fitness: 0.66, temperature: 0.75, strategy: 'tot', prompt_tokens: 400, status: 'dead', parent_id: 'g4-a2', genome: GENOME_TOT, fitness_history: [0.42, 0.55, 0.64, 0.60, 0.66], plugins_active: [], run_logs: [] },
  { id: 'g5-a4', generation: 5, fitness: 0.58, temperature: 0.9, strategy: 'react', prompt_tokens: 255, status: 'dead', parent_id: 'g4-a3', genome: GENOME_REACT, fitness_history: [0.42, 0.48, 0.49, 0.60, 0.58], plugins_active: [], run_logs: [] },
  // Gen 6
  { id: 'g6-a1', generation: 6, fitness: 0.83, temperature: 0.7, strategy: 'reflexion', prompt_tokens: 330, status: 'dead', parent_id: 'g5-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55, 0.64, 0.72, 0.78, 0.83], plugins_active: [], run_logs: [] },
  { id: 'g6-a2', generation: 6, fitness: 0.78, temperature: 0.65, strategy: 'cot', prompt_tokens: 285, status: 'dead', parent_id: 'g5-a1', genome: GENOME_COT, fitness_history: [0.42, 0.55, 0.62, 0.68, 0.73, 0.78], plugins_active: [], run_logs: [] },
  { id: 'g6-a3', generation: 6, fitness: 0.71, temperature: 0.8, strategy: 'tot', prompt_tokens: 410, status: 'active', parent_id: 'g5-a2', genome: GENOME_TOT, fitness_history: [0.42, 0.55, 0.58, 0.60, 0.66, 0.71], plugins_active: [], run_logs: [] },
  { id: 'g6-a4', generation: 6, fitness: 0.64, temperature: 0.7, strategy: 'moa', prompt_tokens: 510, status: 'active', parent_id: 'g5-a3', genome: GENOME_MOA, fitness_history: [0.42, 0.54, 0.55, 0.60, 0.58, 0.64], plugins_active: [], run_logs: [] },
  // Gen 7
  { id: 'g7-a1', generation: 7, fitness: 0.88, temperature: 0.7, strategy: 'reflexion', prompt_tokens: 340, status: 'active', parent_id: 'g6-a1', genome: GENOME_REFLEXION, fitness_history: [0.42, 0.55, 0.64, 0.72, 0.78, 0.83, 0.88], plugins_active: [], run_logs: [] },
  { id: 'g7-a2', generation: 7, fitness: 0.83, temperature: 0.6, strategy: 'cot', prompt_tokens: 290, status: 'active', parent_id: 'g6-a1', genome: GENOME_COT, fitness_history: [0.42, 0.55, 0.62, 0.68, 0.73, 0.78, 0.83], plugins_active: [], run_logs: [] },
  { id: 'g7-a3', generation: 7, fitness: 0.77, temperature: 0.75, strategy: 'tot', prompt_tokens: 415, status: 'active', parent_id: 'g6-a2', genome: GENOME_TOT, fitness_history: [0.42, 0.55, 0.58, 0.60, 0.66, 0.71, 0.77], plugins_active: [], run_logs: [] },
  { id: 'g7-a4', generation: 7, fitness: 0.70, temperature: 0.8, strategy: 'react', prompt_tokens: 260, status: 'active', parent_id: 'g6-a3', genome: GENOME_REACT, fitness_history: [0.42, 0.55, 0.58, 0.60, 0.66, 0.64, 0.70], plugins_active: [], run_logs: [] },
  // Gen 8 (current)
  ...MOCK_AGENTS,
]
