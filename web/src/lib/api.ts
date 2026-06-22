import type { Agent, EvolutionConfig, EvolutionRun, Model, Plugin } from '../types'

const BASE = 'http://localhost:8000'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export const getHealth = (): Promise<{ status: string }> =>
  apiFetch('/api/health')

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

export interface ApiPlugin {
  id: string
  name: string
  description: string
  category: string
  impact: number | null
  enabled: boolean
  order: number
  tags: string[]
  icon: string
}

export const getPlugins = (): Promise<ApiPlugin[]> =>
  apiFetch('/api/plugins')

export const getPlugin = (name: string): Promise<ApiPlugin> =>
  apiFetch(`/api/plugins/${name}`)

export const togglePlugin = (name: string, enabled: boolean): Promise<ApiPlugin> =>
  apiFetch(`/api/plugins/${name}/${enabled ? 'enable' : 'disable'}`, { method: 'POST' })

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export const getModels = (): Promise<Model[]> =>
  apiFetch('/api/models')

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export const getRuns = (): Promise<EvolutionRun[]> =>
  apiFetch('/api/runs')

export const getRun = (id: string): Promise<EvolutionRun> =>
  apiFetch(`/api/runs/${id}`)

export const deleteRun = (id: string): Promise<{ status: string }> =>
  apiFetch(`/api/runs/${id}`, { method: 'DELETE' })

export const getRunAgents = (id: string): Promise<Agent[]> =>
  apiFetch(`/api/runs/${id}/agents`)

export const getAgent = (id: string): Promise<Agent> =>
  apiFetch(`/api/agents/${id}`)

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

export interface EvolveResponse {
  run_id: string
}

export const startEvolution = (config: EvolutionConfig): Promise<EvolveResponse> =>
  apiFetch('/api/evolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })

export const WS_BASE = 'ws://localhost:8000'

export const evolveWsUrl = (run_id: string): string =>
  `${WS_BASE}/ws/evolve/${run_id}`

// ---------------------------------------------------------------------------
// Merge helpers — backend plugin data overlaid onto static UI metadata
// ---------------------------------------------------------------------------

/**
 * Merges API-returned enabled state onto a static Plugin array.
 * Fields from staticPlugins (icon, tags, order, description) are preserved;
 * only `enabled` is overridden by the API response.
 */
export const mergePluginState = (
  staticPlugins: Plugin[],
  apiPlugins: ApiPlugin[],
): Plugin[] => {
  const apiMap = new Map(apiPlugins.map((p) => [p.name, p]))
  return staticPlugins.map((p) => {
    const api = apiMap.get(p.name) ?? apiMap.get(p.id)
    return api ? { ...p, enabled: api.enabled } : p
  })
}
