import { useCallback, useRef, useState } from 'react'
import { evolveWsUrl, startEvolution } from '../lib/api'
import type { Agent, EvolutionConfig, GenerationStat, WSMessage } from '../types'

export type EvolutionStatus = 'idle' | 'starting' | 'running' | 'complete' | 'error' | 'aborted'

export interface EvolutionState {
  status: EvolutionStatus
  runId: string | null
  agents: Agent[]
  allAgents: Agent[]
  generations: GenerationStat[]
  currentGeneration: number
  error: string | null
}

interface UseEvolutionReturn {
  state: EvolutionState
  start: (config: EvolutionConfig) => Promise<void>
  stop: () => void
}

const INITIAL: EvolutionState = {
  status: 'idle',
  runId: null,
  agents: [],
  allAgents: [],
  generations: [],
  currentGeneration: 0,
  error: null,
}

export const useEvolution = (): UseEvolutionReturn => {
  const [state, setState] = useState<EvolutionState>(INITIAL)
  const wsRef = useRef<WebSocket | null>(null)
  const mountedRef = useRef(true)

  const stop = useCallback(() => {
    wsRef.current?.close()
    if (mountedRef.current) {
      setState((prev) => ({ ...prev, status: 'aborted' }))
    }
  }, [])

  const start = useCallback(async (config: EvolutionConfig) => {
    wsRef.current?.close()

    setState({ ...INITIAL, status: 'starting' })

    let run_id: string
    try {
      const res = await startEvolution(config)
      run_id = res.run_id
    } catch (err) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
      }))
      return
    }

    setState((prev) => ({ ...prev, runId: run_id, status: 'running' }))

    const ws = new WebSocket(evolveWsUrl(run_id))
    wsRef.current = ws

    ws.onmessage = (event: MessageEvent<string>) => {
      if (!mountedRef.current) return
      let msg: WSMessage
      try {
        msg = JSON.parse(event.data) as WSMessage
      } catch {
        return
      }

      setState((prev) => {
        switch (msg.type) {
          case 'agent_update': {
            const agent = msg.payload as Agent
            // Keep agents for the current (latest) generation; keep all in allAgents
            const isCurrentGen = prev.generations.length === 0
              || agent.generation > (prev.generations[prev.generations.length - 1]?.generation ?? 0)
            const newAgents = isCurrentGen
              ? [...prev.agents.filter((a) => a.generation === agent.generation), agent]
              : prev.agents
            return {
              ...prev,
              agents: newAgents,
              allAgents: [...prev.allAgents, agent],
            }
          }
          case 'generation_complete': {
            const stat = msg.payload as GenerationStat
            return {
              ...prev,
              generations: [...prev.generations, stat],
              currentGeneration: stat.generation,
              // Reset agents to show current generation only
              agents: prev.allAgents.filter((a) => a.generation === stat.generation),
            }
          }
          case 'run_complete': {
            return { ...prev, status: 'complete' }
          }
          case 'error': {
            const errPayload = msg.payload as { message?: string }
            return {
              ...prev,
              status: 'error',
              error: errPayload.message ?? 'Unknown error',
            }
          }
          default:
            return prev
        }
      })
    }

    ws.onerror = () => {
      if (!mountedRef.current) return
      setState((prev) => ({ ...prev, status: 'error', error: 'WebSocket connection failed' }))
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setState((prev) =>
        prev.status === 'running' ? { ...prev, status: 'aborted' } : prev
      )
    }
  }, [])

  return { state, start, stop }
}
