export interface Agent {
  id: string
  generation: number
  fitness: number
  temperature: number
  strategy: string
  prompt_tokens: number
  status: 'elite' | 'active' | 'dead'
}

export interface GenerationStat {
  generation: number
  best_fitness: number
  avg_fitness: number
  diversity: number
}

export interface HistoricalRun {
  id: string
  task: string
  generations: number
  population: number
  best_fitness: number
  created_at: string
  duration_s: number
}

export interface PipelineStep {
  id: string
  type: 'mutate' | 'evaluate' | 'select' | 'crossover' | 'filter'
  label: string
  params: Record<string, string | number | boolean>
  enabled: boolean
}

export const MOCK_AGENTS: Agent[] = [
  { id: 'a-001', generation: 8, fitness: 0.94, temperature: 0.7, strategy: 'reflexion', prompt_tokens: 312, status: 'elite' },
  { id: 'a-002', generation: 8, fitness: 0.91, temperature: 0.5, strategy: 'chain-of-thought', prompt_tokens: 287, status: 'elite' },
  { id: 'a-003', generation: 7, fitness: 0.88, temperature: 0.8, strategy: 'tree-of-thought', prompt_tokens: 401, status: 'active' },
  { id: 'a-004', generation: 8, fitness: 0.85, temperature: 0.6, strategy: 'react', prompt_tokens: 256, status: 'active' },
  { id: 'a-005', generation: 6, fitness: 0.81, temperature: 0.9, strategy: 'reflexion', prompt_tokens: 345, status: 'active' },
  { id: 'a-006', generation: 5, fitness: 0.73, temperature: 1.0, strategy: 'direct', prompt_tokens: 198, status: 'dead' },
  { id: 'a-007', generation: 7, fitness: 0.79, temperature: 0.4, strategy: 'chain-of-thought', prompt_tokens: 412, status: 'active' },
  { id: 'a-008', generation: 8, fitness: 0.82, temperature: 0.7, strategy: 'moa', prompt_tokens: 534, status: 'active' },
]

export const MOCK_GENERATIONS: GenerationStat[] = [
  { generation: 1, best_fitness: 0.42, avg_fitness: 0.31, diversity: 0.9 },
  { generation: 2, best_fitness: 0.55, avg_fitness: 0.41, diversity: 0.85 },
  { generation: 3, best_fitness: 0.63, avg_fitness: 0.50, diversity: 0.78 },
  { generation: 4, best_fitness: 0.70, avg_fitness: 0.58, diversity: 0.71 },
  { generation: 5, best_fitness: 0.75, avg_fitness: 0.64, diversity: 0.66 },
  { generation: 6, best_fitness: 0.81, avg_fitness: 0.70, diversity: 0.60 },
  { generation: 7, best_fitness: 0.88, avg_fitness: 0.76, diversity: 0.55 },
  { generation: 8, best_fitness: 0.94, avg_fitness: 0.82, diversity: 0.48 },
]

export const MOCK_HISTORY: HistoricalRun[] = [
  { id: 'run-001', task: 'Write a Python function that reverses a string', generations: 10, population: 8, best_fitness: 0.94, created_at: '2026-04-14 14:22', duration_s: 183 },
  { id: 'run-002', task: 'Generate a SQL query to find duplicate emails', generations: 8, population: 6, best_fitness: 0.89, created_at: '2026-04-14 11:05', duration_s: 127 },
  { id: 'run-003', task: 'Summarize a research paper in 3 bullet points', generations: 12, population: 10, best_fitness: 0.91, created_at: '2026-04-13 17:44', duration_s: 241 },
  { id: 'run-004', task: 'Write a regex to validate email addresses', generations: 6, population: 8, best_fitness: 0.86, created_at: '2026-04-13 09:18', duration_s: 98 },
  { id: 'run-005', task: 'Explain recursion to a 10-year-old', generations: 8, population: 6, best_fitness: 0.92, created_at: '2026-04-12 20:30', duration_s: 155 },
]

export const MOCK_PIPELINE: PipelineStep[] = [
  { id: 's-1', type: 'mutate', label: 'LLM Mutator', params: { model: 'gpt-4o-mini', temperature: 0.9, strategies: 'reflexion,cot,react' }, enabled: true },
  { id: 's-2', type: 'evaluate', label: 'Fitness Evaluator', params: { evaluator: 'llm_judge', retries: 2 }, enabled: true },
  { id: 's-3', type: 'select', label: 'Tournament Selection', params: { k: 3, elitism: 2 }, enabled: true },
  { id: 's-4', type: 'crossover', label: 'Genome Crossover', params: { rate: 0.3, blend_prompts: true }, enabled: false },
  { id: 's-5', type: 'filter', label: 'Apoptosis Filter', params: { threshold: 0.3, keep_min: 2 }, enabled: true },
]
