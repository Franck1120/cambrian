import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from 'recharts'
import { TrendingUp, Clock, Target, Layers, Download } from 'lucide-react'
import GlassPanel from '../components/ui/GlassPanel'
import GlowButton from '../components/ui/GlowButton'
import StatCard from '../components/ui/StatCard'
import { COLORS } from '../lib/theme'
import { MOCK_HISTORY, MOCK_AGENTS } from '../data/mock'

interface TooltipPayloadItem {
  name: string
  value: number
  fill?: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string | number
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'rgba(5,8,25,0.95)', border: '1px solid rgba(0,240,255,0.2)', borderRadius: 10, padding: '10px 14px', fontSize: 12, backdropFilter: 'blur(12px)' }}>
      <div style={{ color: COLORS.textMuted, marginBottom: 6 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.fill ?? COLORS.cyan }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

const topAgents = [...MOCK_AGENTS].sort((a, b) => b.fitness - a.fitness).slice(0, 5)

const radarData = [
  { subject: 'Fitness', value: 94 },
  { subject: 'Diversity', value: 48 },
  { subject: 'Speed', value: 72 },
  { subject: 'Efficiency', value: 85 },
  { subject: 'Stability', value: 78 },
]

const Dashboard = () => {
  const [selectedRun, setSelectedRun] = useState<string | null>(null)

  const totalRuns = MOCK_HISTORY.length
  const bestFitness = Math.max(...MOCK_HISTORY.map((r) => r.best_fitness))
  const avgGen = (MOCK_HISTORY.reduce((s, r) => s + r.generations, 0) / totalRuns).toFixed(1)
  const totalAgents = MOCK_HISTORY.reduce((s, r) => s + r.generations * r.population, 0)

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1280, margin: '0 auto' }}>
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1
            style={{ fontSize: 28, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", margin: '0 0 6px', letterSpacing: '-0.5px' }}
            className="text-gradient-bio"
          >
            Dashboard
          </h1>
          <p style={{ fontSize: 14, color: COLORS.textMuted, margin: 0 }}>
            Historical runs · performance analytics · top agents
          </p>
        </div>
        <GlowButton color="cyan" variant="outline" size="sm" icon={<Download size={13} />}>
          Export CSV
        </GlowButton>
      </motion.div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Runs', value: totalRuns, sub: 'all time', color: 'cyan' as const, icon: <Target size={14} /> },
          { label: 'Best Fitness', value: bestFitness.toFixed(3), sub: 'run-001', color: 'green' as const, icon: <TrendingUp size={14} /> },
          { label: 'Avg Generations', value: avgGen, sub: 'per run', color: 'magenta' as const, icon: <Layers size={14} /> },
          { label: 'Total Agents', value: totalAgents.toLocaleString(), sub: 'evolved', color: 'amber' as const, icon: <Clock size={14} /> },
        ].map((s, i) => (
          <motion.div key={s.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.06 }}>
            <StatCard {...s} animateIn={false} />
          </motion.div>
        ))}
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Fitness by run */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <GlassPanel style={{ padding: '20px 16px 12px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16, paddingLeft: 8 }}>
              Best Fitness per Run
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={MOCK_HISTORY.map((r) => ({ name: r.id.replace('run-', '#'), fitness: r.best_fitness }))}
                margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.1)" tick={{ fill: COLORS.textMuted, fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis domain={[0.7, 1]} stroke="rgba(255,255,255,0.1)" tick={{ fill: COLORS.textMuted, fontSize: 11 }} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar
                  dataKey="fitness"
                  name="Fitness"
                  fill={`url(#barGrad)`}
                  radius={[6, 6, 0, 0]}
                  maxBarSize={48}
                />
                <defs>
                  <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.cyan} />
                    <stop offset="100%" stopColor={COLORS.magenta} stopOpacity={0.7} />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </GlassPanel>
        </motion.div>

        {/* Radar */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
          <GlassPanel style={{ padding: '20px 16px 12px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16, paddingLeft: 8 }}>
              Best Agent Profile
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarData} margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: COLORS.textMuted, fontSize: 11 }} />
                <Radar
                  name="Agent"
                  dataKey="value"
                  stroke={COLORS.cyan}
                  fill={COLORS.cyan}
                  fillOpacity={0.12}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </GlassPanel>
        </motion.div>
      </div>

      {/* Run history table */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <GlassPanel style={{ overflow: 'hidden', marginBottom: 24 }}>
          <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)', fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Run History
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  {['Run', 'Task', 'Gen', 'Pop', 'Best Fitness', 'Duration', 'Date'].map((h) => (
                    <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 10, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MOCK_HISTORY.map((run, i) => (
                  <RunRow
                    key={run.id}
                    run={run}
                    index={i}
                    isSelected={selectedRun === run.id}
                    onClick={() => setSelectedRun(selectedRun === run.id ? null : run.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </GlassPanel>
      </motion.div>

      {/* Top agents */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <GlassPanel style={{ overflow: 'hidden' }}>
          <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)', fontSize: 11, fontWeight: 700, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Top Agents — All Time
          </div>
          <div style={{ padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            {topAgents.map((agent, i) => (
              <motion.div
                key={agent.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.55 + i * 0.05 }}
                style={{ display: 'flex', alignItems: 'center', gap: 14 }}
              >
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: i === 0 ? COLORS.cyan : i === 1 ? COLORS.textSecondary : COLORS.textMuted,
                    minWidth: 22,
                    fontFamily: "'Space Grotesk', sans-serif",
                  }}
                >
                  #{i + 1}
                </span>
                <span style={{ fontSize: 12, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace", minWidth: 56 }}>
                  {agent.id}
                </span>
                <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${agent.fitness * 100}%` }}
                    transition={{ delay: 0.6 + i * 0.05, duration: 0.5 }}
                    style={{
                      height: '100%',
                      background: i === 0 ? `linear-gradient(90deg, ${COLORS.cyan}, ${COLORS.magenta})` : 'rgba(255,255,255,0.2)',
                      borderRadius: 2,
                    }}
                  />
                </div>
                <span style={{ fontSize: 13, fontWeight: 700, color: i === 0 ? COLORS.cyan : COLORS.textSecondary, minWidth: 36 }}>
                  {agent.fitness.toFixed(3)}
                </span>
                <span style={{ fontSize: 11, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 5, padding: '2px 8px', color: COLORS.textMuted }}>
                  {agent.strategy}
                </span>
              </motion.div>
            ))}
          </div>
        </GlassPanel>
      </motion.div>

      <div style={{ height: 32 }} />
    </div>
  )
}

const RunRow = ({
  run,
  index,
  isSelected,
  onClick,
}: {
  run: typeof MOCK_HISTORY[0]
  index: number
  isSelected: boolean
  onClick: () => void
}) => {
  const [hovered, setHovered] = useState(false)

  return (
    <tr
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        borderBottom: index < MOCK_HISTORY.length - 1 ? '1px solid rgba(255,255,255,0.03)' : 'none',
        cursor: 'pointer',
        background: isSelected
          ? 'rgba(0,240,255,0.06)'
          : hovered
            ? 'rgba(0,240,255,0.03)'
            : 'transparent',
        transition: 'background 0.12s',
      }}
    >
      <td style={{ padding: '11px 16px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: COLORS.cyan, whiteSpace: 'nowrap' }}>
        {run.id}
      </td>
      <td style={{ padding: '11px 16px', fontSize: 13, color: COLORS.textSecondary, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {run.task}
      </td>
      <td style={{ padding: '11px 16px', fontSize: 13, color: COLORS.textMuted }}>
        {run.generations}
      </td>
      <td style={{ padding: '11px 16px', fontSize: 13, color: COLORS.textMuted }}>
        {run.population}
      </td>
      <td style={{ padding: '11px 16px' }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: run.best_fitness >= 0.9 ? COLORS.green : run.best_fitness >= 0.8 ? COLORS.cyan : COLORS.amber }}>
          {run.best_fitness.toFixed(3)}
        </span>
      </td>
      <td style={{ padding: '11px 16px', fontSize: 12, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
        {run.duration_s}s
      </td>
      <td style={{ padding: '11px 16px', fontSize: 11, color: COLORS.textMuted }}>
        {run.created_at}
      </td>
    </tr>
  )
}

export default Dashboard
