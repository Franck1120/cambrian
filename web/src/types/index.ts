export type AgentStatus = 'elite' | 'active' | 'dead'
export type PluginCategory = 'mutation' | 'selection' | 'memory' | 'diversity' | 'evaluation' | 'crossover'
export type ModelProvider = 'groq' | 'openai' | 'anthropic' | 'local' | 'proxy'
export type ModelStatus = 'online' | 'offline' | 'rate-limited'
export type ModelRole = 'mutation' | 'evaluation' | 'crossover'
export type PipelineStepType = 'mutate' | 'evaluate' | 'select' | 'crossover' | 'filter'

export interface RunLog {
  id: string
  input: string
  output: string
  fitness: number
  timestamp: string
  tokens_used: number
}

export interface Agent {
  id: string
  generation: number
  fitness: number
  temperature: number
  strategy: string
  prompt_tokens: number
  status: AgentStatus
  parent_id: string | null
  genome: string
  fitness_history: number[]
  plugins_active: string[]
  run_logs: RunLog[]
}

export interface GenerationStat {
  generation: number
  best_fitness: number
  avg_fitness: number
  diversity: number
}

export interface EvolutionRun {
  id: string
  task: string
  generations: number
  population: number
  best_fitness: number
  created_at: string
  duration_s: number
}

export interface Plugin {
  id: string
  name: string
  icon: string
  description: string
  category: PluginCategory
  enabled: boolean
  impact: number | null
  order: number
  tags: string[]
}

export interface Model {
  id: string
  name: string
  provider: ModelProvider
  context_window: number
  speed: 'fast' | 'medium' | 'slow'
  cost_per_1k: number
  status: ModelStatus
}

export interface ModelAssignment {
  mutation: string
  evaluation: string
  crossover: string
}

export interface PipelineStep {
  id: string
  type: PipelineStepType
  label: string
  params: Record<string, string | number | boolean>
  enabled: boolean
}

export interface EvolutionConfig {
  task: string
  generations: number
  population: number
  model_id: string
  plugins: string[]
}

export interface WSMessage {
  type: 'agent_update' | 'generation_complete' | 'run_complete' | 'error'
  payload: unknown
}
