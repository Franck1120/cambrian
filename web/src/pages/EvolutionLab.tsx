import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Square, Zap, Users, Layers, DollarSign } from 'lucide-react'
import GlassPanel from '../components/ui/GlassPanel'
import GlowButton from '../components/ui/GlowButton'
import StatCard from '../components/ui/StatCard'
import FitnessChart from '../components/ui/FitnessChart'
import GenerationTimeline from '../components/ui/GenerationTimeline'
import AgentProfileModal from '../components/AgentProfileModal'
import { COLORS, STATUS_COLOR } from '../lib/theme'
import { useEvolution } from '../hooks/useEvolution'
import type { Agent, EvolutionConfig } from '../types'

const EvolutionLab = () => {
  const [task, setTask] = useState('Write a Python function that reverses a string')
  const [generations, setGenerations] = useState(10)
  const [population, setPopulation] = useState(8)
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)

  const { state, start, stop } = useEvolution()

  const running = state.status === 'starting' || state.status === 'running'
  const hasRun = state.allAgents.length > 0 || state.generations.length > 0

  const handleRun = () => {
    if (running) {
      stop()
      return
    }
    const config: EvolutionConfig = {
      task,
      generations,
      population,
      model_id: 'llama-3.3-70b',
      plugins: [],
    }
    void start(config)
  }

  const displayAgents = state.agents.length > 0 ? state.agents : []
  const displayGenerations = state.generations

  const bestFitness = displayAgents.length > 0
    ? Math.max(...displayAgents.map((a) => a.fitness))
    : 0
  const avgFitness = displayAgents.length > 0
    ? displayAgents.reduce((s, a) => s + a.fitness, 0) / displayAgents.length
    : 0
  const totalTokens = displayAgents.reduce((s, a) => s + a.prompt_tokens, 0)
  const estimatedCost = ((totalTokens / 1000) * 0.27).toFixed(3)

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1280, margin: '0 auto' }}>
      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ marginBottom: 24 }}
      >
        <h1
          style={{
            fontSize: 28,
            fontWeight: 700,
            fontFamily: "'Space Grotesk', sans-serif",
            margin: '0 0 6px',
            letterSpacing: '-0.5px',
          }}
          className="text-gradient-bio"
        >
          Evolution Lab
        </h1>
        <p style={{ fontSize: 14, color: COLORS.textMuted, margin: 0 }}>
          Genetic algorithm over LLM agent genomes — watch populations evolve in real time
        </p>
      </motion.div>

      {/* Config panel */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <GlassPanel style={{ padding: '20px 24px', marginBottom: 24 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr auto auto auto',
              gap: 20,
              alignItems: 'end',
            }}
          >
            {/* Task input */}
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  fontWeight: 700,
                  color: COLORS.textMuted,
                  marginBottom: 8,
                  textTransform: 'uppercase',
                  letterSpacing: '0.09em',
                }}
              >
                Task
              </label>
              <textarea
                value={task}
                onChange={(e) => setTask(e.target.value)}
                rows={2}
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(0,240,255,0.15)',
                  borderRadius: 10,
                  padding: '10px 14px',
                  color: COLORS.textPrimary,
                  fontSize: 14,
                  lineHeight: 1.5,
                  resize: 'none',
                  transition: 'border-color 0.15s',
                  boxSizing: 'border-box',
                }}
                onFocus={(e) => (e.target.style.borderColor = COLORS.cyan + '55')}
                onBlur={(e) => (e.target.style.borderColor = 'rgba(0,240,255,0.15)')}
              />
            </div>

            {/* Generations slider */}
            <div style={{ minWidth: 148 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.textMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.09em' }}>
                Generations
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <input type="range" min={2} max={20} value={generations} onChange={(e) => setGenerations(+e.target.value)} style={{ flex: 1 }} />
                <span style={{ fontSize: 20, fontWeight: 700, color: COLORS.cyan, fontFamily: "'Space Grotesk', sans-serif", minWidth: 24 }}>
                  {generations}
                </span>
              </div>
            </div>

            {/* Population slider */}
            <div style={{ minWidth: 148 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: COLORS.textMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.09em' }}>
                Population
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <input type="range" min={2} max={20} value={population} onChange={(e) => setPopulation(+e.target.value)} style={{ flex: 1 }} />
                <span style={{ fontSize: 20, fontWeight: 700, color: COLORS.magenta, fontFamily: "'Space Grotesk', sans-serif", minWidth: 24 }}>
                  {population}
                </span>
              </div>
            </div>

            {/* Run button */}
            <GlowButton
              color={running ? 'amber' : 'cyan'}
              size="lg"
              loading={state.status === 'starting'}
              icon={running ? <Square size={15} /> : <Play size={15} />}
              onClick={handleRun}
            >
              {running ? 'Evolving…' : 'Run Evolution'}
            </GlowButton>
          </div>

          {/* Error banner */}
          {state.status === 'error' && state.error && (
            <div style={{ marginTop: 12, padding: '8px 14px', borderRadius: 8, background: 'rgba(255,60,60,0.1)', border: '1px solid rgba(255,60,60,0.3)', fontSize: 12, color: '#ff6b6b' }}>
              {state.error}
            </div>
          )}
        </GlassPanel>
      </motion.div>

      {/* Stats + charts */}
      {hasRun && (
        <>
          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Best Fitness', value: bestFitness.toFixed(3), trend: 0.06, sparkline: displayGenerations.map((g) => g.best_fitness), color: 'cyan' as const, icon: <Zap size={14} /> },
              { label: 'Avg Fitness', value: avgFitness.toFixed(3), trend: 0.01, sparkline: displayGenerations.map((g) => g.avg_fitness), color: 'magenta' as const, icon: <Layers size={14} /> },
              { label: 'Generations', value: `${state.currentGeneration} / ${generations}`, sub: 'completed', color: 'green' as const, icon: <Users size={14} /> },
              { label: 'Est. Cost', value: `$${estimatedCost}`, sub: `${totalTokens.toLocaleString()} tokens`, color: 'amber' as const, icon: <DollarSign size={14} /> },
            ].map((s, i) => (
              <motion.div key={s.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 + i * 0.06 }}>
                <StatCard {...s} animateIn={false} />
              </motion.div>
            ))}
          </div>

          {/* Charts */}
          {displayGenerations.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
                <FitnessChart data={displayGenerations} height={220} title="Fitness Over Generations" />
              </motion.div>
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <FitnessChart data={displayGenerations} showDiversity height={220} title="Diversity Decay" />
              </motion.div>
            </div>
          )}

          {/* Generation Timeline */}
          {displayGenerations.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45 }}>
              <GlassPanel style={{ padding: '20px 28px', marginBottom: 24 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 20 }}>
                  Genealogy Timeline
                </div>
                <GenerationTimeline
                  generations={displayGenerations}
                  agents={state.allAgents}
                  onAgentClick={setSelectedAgent}
                  selectedAgentId={selectedAgent?.id}
                />
              </GlassPanel>
            </motion.div>
          )}

          {/* Agent Population Table */}
          {displayAgents.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              <GlassPanel style={{ overflow: 'hidden' }}>
                <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)', fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>Agent Population — Gen {state.currentGeneration}</span>
                  <span style={{ fontWeight: 400 }}>
                    {displayAgents.filter((a) => a.status !== 'dead').length} active · {displayAgents.filter((a) => a.status === 'elite').length} elite
                  </span>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        {['Agent', 'Gen', 'Fitness', 'Temp', 'Strategy', 'Tokens', 'Status'].map((h) => (
                          <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 10, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {displayAgents.map((agent, i) => (
                        <AgentRow key={agent.id} agent={agent} index={i} total={displayAgents.length} onClick={setSelectedAgent} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassPanel>
            </motion.div>
          )}
        </>
      )}

      <AgentProfileModal agent={selectedAgent} onClose={() => setSelectedAgent(null)} />
      <div style={{ height: 32 }} />
    </div>
  )
}

const AgentRow = ({
  agent,
  index,
  total,
  onClick,
}: {
  agent: Agent
  index: number
  total: number
  onClick: (a: Agent) => void
}) => {
  const [hovered, setHovered] = useState(false)
  const color = STATUS_COLOR[agent.status] ?? COLORS.textMuted

  return (
    <tr
      onClick={() => onClick(agent)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        borderBottom: index < total - 1 ? '1px solid rgba(255,255,255,0.03)' : 'none',
        cursor: 'pointer',
        background: hovered ? 'rgba(0,240,255,0.04)' : 'transparent',
        transition: 'background 0.12s',
      }}
    >
      <td style={{ padding: '11px 16px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: COLORS.cyan, whiteSpace: 'nowrap' }}>
        {agent.id}
      </td>
      <td style={{ padding: '11px 16px', fontSize: 13, color: COLORS.textSecondary }}>
        {agent.generation}
      </td>
      <td style={{ padding: '11px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 64, height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${agent.fitness * 100}%`, background: `linear-gradient(90deg, ${COLORS.cyan}, ${COLORS.magenta})`, borderRadius: 2 }} />
          </div>
          <span style={{ fontSize: 13, fontWeight: 700, color, minWidth: 36 }}>
            {agent.fitness.toFixed(2)}
          </span>
        </div>
      </td>
      <td style={{ padding: '11px 16px', fontSize: 12, color: COLORS.textSecondary, fontFamily: "'JetBrains Mono', monospace" }}>
        {agent.temperature}
      </td>
      <td style={{ padding: '11px 16px' }}>
        <span style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, padding: '3px 9px', fontSize: 11, color: COLORS.textSecondary, whiteSpace: 'nowrap' }}>
          {agent.strategy}
        </span>
      </td>
      <td style={{ padding: '11px 16px', fontSize: 12, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
        {agent.prompt_tokens.toLocaleString()}
      </td>
      <td style={{ padding: '11px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: color, boxShadow: agent.status === 'elite' ? `0 0 6px ${COLORS.cyan}` : 'none' }} />
          <span style={{ fontSize: 12, fontWeight: 500, color }}>{agent.status}</span>
        </div>
      </td>
    </tr>
  )
}

export default EvolutionLab
