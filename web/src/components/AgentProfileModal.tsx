import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Copy,
  Download,
  GitBranch,
  Dna,
  Activity,
  Terminal,
  Network,
  CheckCircle,
} from 'lucide-react'
import GlassPanel from './ui/GlassPanel'
import GlowButton from './ui/GlowButton'
import { COLORS, STATUS_COLOR, GLOW } from '../lib/theme'
import { MOCK_AGENTS } from '../data/mock'
import type { Agent } from '../types'

interface AgentProfileModalProps {
  agent: Agent | null
  onClose: () => void
}

type Tab = 'genome' | 'history' | 'logs' | 'lineage'

const TAB_CONFIG: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'genome', label: 'Genome', icon: <Dna size={13} /> },
  { id: 'history', label: 'History', icon: <Activity size={13} /> },
  { id: 'logs', label: 'Run Logs', icon: <Terminal size={13} /> },
  { id: 'lineage', label: 'Lineage', icon: <Network size={13} /> },
]

const FitnessSparkline = ({ data, color }: { data: number[]; color: string }) => {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const w = 200
  const h = 48
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(' ')

  return (
    <svg width={w} height={h}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      {data.map((v, i) => (
        <circle
          key={i}
          cx={(i / (data.length - 1)) * w}
          cy={h - ((v - min) / range) * h}
          r={i === data.length - 1 ? 4 : 2.5}
          fill={i === data.length - 1 ? color : `${color}88`}
        />
      ))}
    </svg>
  )
}

const GenomeTab = ({ agent }: { agent: Agent }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    void navigator.clipboard.writeText(agent.genome)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const parent = agent.parent_id ? MOCK_AGENTS.find((a) => a.id === agent.parent_id) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12, color: COLORS.textMuted }}>System prompt genome</span>
        <GlowButton
          size="sm"
          color="cyan"
          variant="outline"
          icon={copied ? <CheckCircle size={12} /> : <Copy size={12} />}
          onClick={handleCopy}
        >
          {copied ? 'Copied' : 'Copy'}
        </GlowButton>
      </div>

      <div
        style={{
          background: 'rgba(0,0,0,0.4)',
          border: '1px solid rgba(0,240,255,0.1)',
          borderRadius: 10,
          padding: '16px 18px',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          lineHeight: 1.7,
          color: COLORS.textPrimary,
          whiteSpace: 'pre-wrap',
          maxHeight: 280,
          overflowY: 'auto',
        }}
      >
        {agent.genome}
      </div>

      {parent && (
        <div>
          <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Diff from parent ({parent.id})
          </div>
          <div
            style={{
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 8,
              padding: '12px 14px',
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              lineHeight: 1.6,
              color: COLORS.textSecondary,
            }}
          >
            <div style={{ color: `${COLORS.green}cc` }}>+ Revised: self-critique and revision step added</div>
            <div style={{ color: `${COLORS.magenta}cc` }}>- Removed: vague "do your best" instruction</div>
            <div style={{ color: `${COLORS.cyan}cc`, marginTop: 4 }}>~ Strategy: {parent.strategy} → {agent.strategy}</div>
          </div>
        </div>
      )}
    </div>
  )
}

const HistoryTab = ({ agent }: { agent: Agent }) => {
  const color = STATUS_COLOR[agent.status] ?? COLORS.textMuted
  const current = agent.fitness_history[agent.fitness_history.length - 1]
  const first = agent.fitness_history[0]
  const improvement = current - first

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-end' }}>
        <FitnessSparkline data={agent.fitness_history} color={color} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div>
            <div style={{ fontSize: 10, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 2 }}>Current</div>
            <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: "'Space Grotesk', sans-serif" }}>{current.toFixed(3)}</div>
          </div>
          <div style={{ fontSize: 12, color: improvement >= 0 ? COLORS.green : COLORS.magenta }}>
            {improvement >= 0 ? '+' : ''}{improvement.toFixed(3)} since gen 1
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {agent.fitness_history.map((f, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 11, color: COLORS.textMuted, minWidth: 40, fontFamily: "'JetBrains Mono', monospace" }}>
              Gen {i + 1}
            </span>
            <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${f * 100}%` }}
                transition={{ delay: i * 0.04 }}
                style={{
                  height: '100%',
                  background: `linear-gradient(90deg, ${COLORS.cyan}, ${COLORS.magenta})`,
                  borderRadius: 2,
                }}
              />
            </div>
            <span style={{ fontSize: 11, fontWeight: 600, color, minWidth: 36, fontFamily: "'JetBrains Mono', monospace" }}>
              {f.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

const LogsTab = ({ agent }: { agent: Agent }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
    {agent.run_logs.length === 0 && (
      <div style={{ textAlign: 'center', color: COLORS.textMuted, padding: 32 }}>No run logs yet</div>
    )}
    {agent.run_logs.map((log) => (
      <GlassPanel key={log.id} glow="none" style={{ padding: '12px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={{ fontSize: 11, color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
            {log.timestamp}
          </span>
          <div style={{ display: 'flex', gap: 12 }}>
            <span style={{ fontSize: 11, color: COLORS.textMuted }}>{log.tokens_used} tokens</span>
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: log.fitness > 0.8 ? COLORS.green : log.fitness > 0.6 ? COLORS.amber : COLORS.magenta,
              }}
            >
              {log.fitness.toFixed(2)}
            </span>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Input</div>
            <code style={{ fontSize: 11, color: COLORS.cyan, fontFamily: "'JetBrains Mono', monospace" }}>{log.input}</code>
          </div>
          <div>
            <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Output</div>
            <code style={{ fontSize: 11, color: COLORS.green, fontFamily: "'JetBrains Mono', monospace" }}>{log.output}</code>
          </div>
        </div>
      </GlassPanel>
    ))}
  </div>
)

const LineageTab = ({ agent }: { agent: Agent }) => {
  const parent = agent.parent_id ? MOCK_AGENTS.find((a) => a.id === agent.parent_id) : null
  const children = MOCK_AGENTS.filter((a) => a.parent_id === agent.id)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {parent && (
        <div>
          <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
            Parent
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: STATUS_COLOR[parent.status] ?? COLORS.textMuted, flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: COLORS.textPrimary, fontFamily: "'JetBrains Mono', monospace" }}>{parent.id}</span>
            <span style={{ fontSize: 12, color: COLORS.textMuted }}>Gen {parent.generation}</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: STATUS_COLOR[parent.status] ?? COLORS.textMuted }}>{parent.fitness.toFixed(3)}</span>
            <span style={{ fontSize: 11, color: COLORS.textMuted }}>{parent.strategy}</span>
          </div>
        </div>
      )}

      <div>
        <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
          This Agent
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '10px 14px',
            background: `${STATUS_COLOR[agent.status] ?? COLORS.textMuted}12`,
            borderRadius: 10,
            border: `1px solid ${STATUS_COLOR[agent.status] ?? COLORS.textMuted}33`,
          }}
        >
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: STATUS_COLOR[agent.status] ?? COLORS.textMuted, boxShadow: GLOW[agent.status === 'elite' ? 'cyan' : agent.status === 'active' ? 'green' : 'none'] ?? 'none' }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.textPrimary, fontFamily: "'JetBrains Mono', monospace" }}>{agent.id}</span>
          <span style={{ fontSize: 12, color: COLORS.textMuted }}>Gen {agent.generation}</span>
          <span style={{ fontSize: 14, fontWeight: 700, color: STATUS_COLOR[agent.status] ?? COLORS.textMuted }}>{agent.fitness.toFixed(3)}</span>
        </div>
      </div>

      {children.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
            Children ({children.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {children.map((child) => (
              <div key={child.id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: STATUS_COLOR[child.status] ?? COLORS.textMuted }} />
                <span style={{ fontSize: 12, color: COLORS.textSecondary, fontFamily: "'JetBrains Mono', monospace" }}>{child.id}</span>
                <span style={{ fontSize: 11, color: COLORS.textMuted }}>Gen {child.generation}</span>
                <span style={{ fontSize: 12, fontWeight: 600, color: STATUS_COLOR[child.status] ?? COLORS.textMuted }}>{child.fitness.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {children.length === 0 && !parent && (
        <div style={{ textAlign: 'center', color: COLORS.textMuted, padding: 32 }}>
          Progenitor agent — no recorded ancestors
        </div>
      )}
    </div>
  )
}

const AgentProfileModal = ({ agent, onClose }: AgentProfileModalProps) => {
  const [tab, setTab] = useState<Tab>('genome')

  const handleExport = () => {
    if (!agent) return
    const data = JSON.stringify({ id: agent.id, genome: agent.genome, strategy: agent.strategy, fitness: agent.fitness }, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `agent-${agent.id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <AnimatePresence>
      {agent && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(5, 8, 25, 0.7)',
              backdropFilter: 'blur(4px)',
              zIndex: 100,
            }}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 60 }}
            transition={{ type: 'spring', stiffness: 400, damping: 35 }}
            style={{
              position: 'fixed',
              right: 0,
              top: 0,
              bottom: 0,
              width: '100%',
              maxWidth: 520,
              zIndex: 101,
              background: 'rgba(8, 11, 32, 0.98)',
              borderLeft: '1px solid rgba(0,240,255,0.15)',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '-8px 0 40px rgba(0,0,0,0.5)',
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: '20px 24px',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: STATUS_COLOR[agent.status] ?? COLORS.textMuted,
                      boxShadow: agent.status === 'elite' ? GLOW.cyan : agent.status === 'active' ? GLOW.green : 'none',
                    }}
                  />
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 16,
                      fontWeight: 600,
                      color: COLORS.textPrimary,
                    }}
                  >
                    {agent.id}
                  </span>
                  <span
                    style={{
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 6,
                      background: `${STATUS_COLOR[agent.status] ?? COLORS.textMuted}20`,
                      border: `1px solid ${STATUS_COLOR[agent.status] ?? COLORS.textMuted}40`,
                      color: STATUS_COLOR[agent.status] ?? COLORS.textMuted,
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      letterSpacing: '0.07em',
                    }}
                  >
                    {agent.status}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 16, fontSize: 12, color: COLORS.textMuted }}>
                  <span>Gen <strong style={{ color: COLORS.textSecondary }}>{agent.generation}</strong></span>
                  <span>
                    Fitness{' '}
                    <strong style={{ color: STATUS_COLOR[agent.status] ?? COLORS.textMuted, fontSize: 14 }}>
                      {agent.fitness.toFixed(3)}
                    </strong>
                  </span>
                  <span>
                    Strategy{' '}
                    <strong style={{ color: COLORS.textSecondary }}>{agent.strategy}</strong>
                  </span>
                </div>
              </div>

              <button
                onClick={onClose}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: COLORS.textMuted,
                  padding: 4,
                  borderRadius: 8,
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Action buttons */}
            <div style={{ padding: '12px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', gap: 10 }}>
              <GlowButton color="cyan" size="sm" icon={<GitBranch size={13} />}>
                Clone & Evolve
              </GlowButton>
              <GlowButton color="green" variant="outline" size="sm" icon={<Download size={13} />} onClick={handleExport}>
                Export
              </GlowButton>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '0 24px' }}>
              {TAB_CONFIG.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '12px 14px',
                    background: 'transparent',
                    border: 'none',
                    borderBottom: `2px solid ${tab === t.id ? COLORS.cyan : 'transparent'}`,
                    cursor: 'pointer',
                    color: tab === t.id ? COLORS.cyan : COLORS.textMuted,
                    fontSize: 13,
                    fontWeight: tab === t.id ? 600 : 400,
                    transition: 'all 0.15s',
                    fontFamily: "'Inter', sans-serif",
                  }}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.15 }}
                >
                  {tab === 'genome' && <GenomeTab agent={agent} />}
                  {tab === 'history' && <HistoryTab agent={agent} />}
                  {tab === 'logs' && <LogsTab agent={agent} />}
                  {tab === 'lineage' && <LineageTab agent={agent} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export default AgentProfileModal
