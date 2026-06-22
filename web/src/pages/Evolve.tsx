import { useState } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, Legend,
} from 'recharts'
import { MOCK_AGENTS, MOCK_GENERATIONS } from '../data/mock'
import type { Agent } from '../types'

const ACCENT = '#10b981'
const CARD = { background: '#111111', border: '1px solid #1a1a1a', borderRadius: 10 }

const statusColor: Record<Agent['status'], string> = {
  elite: '#10b981',
  active: '#3b82f6',
  dead: '#4b5563',
}

interface TooltipPayload { name: string; value: number; color: string }
interface TooltipProps { active?: boolean; payload?: TooltipPayload[]; label?: string | number }

const CustomTooltip = ({ active, payload, label }: TooltipProps) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#161616', border: '1px solid #222', borderRadius: 8,
      padding: '10px 14px', fontSize: 13 }}>
      <div style={{ color: '#9ca3af', marginBottom: 6 }}>Gen {label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{p.value.toFixed(3)}</strong>
        </div>
      ))}
    </div>
  )
}

const Evolve = () => {
  const [task, setTask] = useState('Write a Python function that reverses a string')
  const [generations, setGenerations] = useState(10)
  const [population, setPopulation] = useState(8)
  const [running, setRunning] = useState(false)
  const [hasRun, setHasRun] = useState(true)

  const handleRun = () => {
    setRunning(true)
    setTimeout(() => { setRunning(false); setHasRun(true) }, 2000)
  }

  return (
    <div style={{ padding: '32px 40px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>
          Evolve
        </h1>
        <p style={{ fontSize: 14, color: '#6b7280' }}>
          Run a genetic algorithm over LLM agent genomes
        </p>
      </div>

      {/* Config panel */}
      <div style={{ ...CARD, padding: '24px 28px', marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto auto', gap: 20, alignItems: 'end' }}>
          {/* Task input */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 500,
              color: '#9ca3af', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Task
            </label>
            <textarea
              value={task}
              onChange={e => setTask(e.target.value)}
              rows={2}
              style={{ width: '100%', background: '#0d0d0d', border: '1px solid #222',
                borderRadius: 8, padding: '10px 14px', color: '#f0f0f0', fontSize: 14,
                fontFamily: 'Inter, sans-serif', resize: 'none', outline: 'none' }}
              onFocus={e => e.target.style.borderColor = ACCENT}
              onBlur={e => e.target.style.borderColor = '#222'}
            />
          </div>

          {/* Generations slider */}
          <div style={{ minWidth: 140 }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 500,
              color: '#9ca3af', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Generations
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <input type="range" min={2} max={20} value={generations}
                onChange={e => setGenerations(+e.target.value)}
                style={{ flex: 1, accentColor: ACCENT }} />
              <span style={{ fontSize: 18, fontWeight: 600, color: ACCENT, minWidth: 24 }}>
                {generations}
              </span>
            </div>
          </div>

          {/* Population slider */}
          <div style={{ minWidth: 140 }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 500,
              color: '#9ca3af', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Population
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <input type="range" min={2} max={20} value={population}
                onChange={e => setPopulation(+e.target.value)}
                style={{ flex: 1, accentColor: ACCENT }} />
              <span style={{ fontSize: 18, fontWeight: 600, color: ACCENT, minWidth: 24 }}>
                {population}
              </span>
            </div>
          </div>

          {/* Run button */}
          <button onClick={handleRun} disabled={running}
            style={{ height: 42, padding: '0 24px', background: running ? '#0d2b1f' : ACCENT,
              border: 'none', borderRadius: 8, color: running ? ACCENT : '#0a0a0a',
              fontWeight: 600, fontSize: 14, cursor: running ? 'not-allowed' : 'pointer',
              fontFamily: 'Inter, sans-serif', transition: 'all 0.15s',
              display: 'flex', alignItems: 'center', gap: 8 }}>
            {running ? (
              <>
                <span style={{ display: 'inline-block', width: 14, height: 14,
                  border: `2px solid ${ACCENT}`, borderTopColor: 'transparent',
                  borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                Evolving…
              </>
            ) : 'Run Evolution'}
          </button>
        </div>
      </div>

      {/* Charts + table */}
      {hasRun && (
        <>
          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Best Fitness', value: '0.940', color: ACCENT },
              { label: 'Avg Fitness', value: '0.820', color: '#f0f0f0' },
              { label: 'Generations', value: '8 / 10', color: '#f0f0f0' },
              { label: 'Diversity', value: '0.48', color: '#f0f0f0' },
            ].map(s => (
              <div key={s.label} style={{ ...CARD, padding: '18px 20px' }}>
                <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6,
                  textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
                  {s.label}
                </div>
                <div style={{ fontSize: 26, fontWeight: 700, color: s.color, letterSpacing: '-0.5px' }}>
                  {s.value}
                </div>
              </div>
            ))}
          </div>

          {/* Fitness chart */}
          <div style={{ ...CARD, padding: '24px 20px', marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#9ca3af', marginBottom: 16,
              textTransform: 'uppercase', letterSpacing: '0.05em', paddingLeft: 8 }}>
              Fitness Over Generations
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={MOCK_GENERATIONS} margin={{ top: 4, right: 16, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="bestGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={ACCENT} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={ACCENT} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="avgGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                <XAxis dataKey="generation" stroke="#333" tick={{ fill: '#6b7280', fontSize: 12 }} />
                <YAxis domain={[0, 1]} stroke="#333" tick={{ fill: '#6b7280', fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af', paddingTop: 8 }} />
                <Area type="monotone" dataKey="best_fitness" name="Best"
                  stroke={ACCENT} fill="url(#bestGrad)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="avg_fitness" name="Avg"
                  stroke="#3b82f6" fill="url(#avgGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Agents table */}
          <div style={{ ...CARD, overflow: 'hidden' }}>
            <div style={{ padding: '18px 20px', borderBottom: '1px solid #1a1a1a',
              fontSize: 13, fontWeight: 600, color: '#9ca3af',
              textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Agent Population
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #1a1a1a' }}>
                  {['ID', 'Gen', 'Fitness', 'Temp', 'Strategy', 'Tokens', 'Status'].map(h => (
                    <th key={h} style={{ padding: '10px 16px', textAlign: 'left',
                      fontSize: 11, fontWeight: 600, color: '#4b5563',
                      textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MOCK_AGENTS.map((agent, i) => (
                  <tr key={agent.id}
                    style={{ borderBottom: i < MOCK_AGENTS.length - 1 ? '1px solid #141414' : 'none',
                      transition: 'background 0.1s' }}
                    onMouseEnter={e => (e.currentTarget.style.background = '#161616')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                    <td style={{ padding: '11px 16px', fontSize: 12, color: '#6b7280', fontFamily: 'monospace' }}>
                      {agent.id}
                    </td>
                    <td style={{ padding: '11px 16px', fontSize: 13, color: '#9ca3af' }}>
                      {agent.generation}
                    </td>
                    <td style={{ padding: '11px 16px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ flex: 1, maxWidth: 80, height: 4, background: '#1a1a1a', borderRadius: 2 }}>
                          <div style={{ height: '100%', width: `${agent.fitness * 100}%`,
                            background: ACCENT, borderRadius: 2 }} />
                        </div>
                        <span style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f0', minWidth: 36 }}>
                          {agent.fitness.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '11px 16px', fontSize: 13, color: '#9ca3af' }}>
                      {agent.temperature}
                    </td>
                    <td style={{ padding: '11px 16px' }}>
                      <span style={{ background: '#1a1a1a', borderRadius: 5, padding: '3px 8px',
                        fontSize: 12, color: '#9ca3af' }}>
                        {agent.strategy}
                      </span>
                    </td>
                    <td style={{ padding: '11px 16px', fontSize: 13, color: '#6b7280' }}>
                      {agent.prompt_tokens}
                    </td>
                    <td style={{ padding: '11px 16px' }}>
                      <span style={{ fontSize: 12, fontWeight: 500,
                        color: statusColor[agent.status] }}>
                        ● {agent.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

export default Evolve
