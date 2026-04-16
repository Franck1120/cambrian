import { useState } from 'react'
import { motion } from 'framer-motion'
import { Clock, DollarSign, Wifi, WifiOff, AlertTriangle, Check } from 'lucide-react'
import GlassPanel from '../components/ui/GlassPanel'
import GlowButton from '../components/ui/GlowButton'
import { COLORS } from '../lib/theme'
import { MODELS, MODEL_PROVIDER_LABELS, MODEL_PROVIDER_COLORS } from '../data/models'
import type { ModelRole, ModelStatus, ModelProvider } from '../types'

const ROLE_CONFIG: { id: ModelRole; label: string; description: string; color: string }[] = [
  { id: 'mutation', label: 'Mutation', description: 'Generates new agent genome variants', color: COLORS.cyan },
  { id: 'evaluation', label: 'Evaluation', description: 'Judges agent output quality', color: COLORS.magenta },
  { id: 'crossover', label: 'Crossover', description: 'Blends parent genomes', color: COLORS.green },
]

const STATUS_CONFIG: Record<ModelStatus, { icon: React.ReactNode; color: string; label: string }> = {
  online: { icon: <Wifi size={12} />, color: COLORS.green, label: 'Online' },
  offline: { icon: <WifiOff size={12} />, color: COLORS.textMuted, label: 'Offline' },
  'rate-limited': { icon: <AlertTriangle size={12} />, color: COLORS.amber, label: 'Rate limited' },
}

const SPEED_LABEL: Record<string, string> = { fast: '< 1s', medium: '1–3s', slow: '> 3s' }

const ModelSelector = () => {
  const [assignment, setAssignment] = useState<Record<ModelRole, string>>({
    mutation: 'llama-3.3-70b',
    evaluation: 'llama-3.3-70b',
    crossover: 'llama-3.1-8b',
  })
  const [selectedRole, setSelectedRole] = useState<ModelRole>('mutation')
  const [filterProvider, setFilterProvider] = useState<ModelProvider | 'all'>('all')

  const filteredModels = MODELS.filter(
    (m) => filterProvider === 'all' || m.provider === filterProvider
  )

  const allProviders = [...new Set(MODELS.map((m) => m.provider))]

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1280, margin: '0 auto' }}>
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: 24 }}>
        <h1
          style={{ fontSize: 28, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", margin: '0 0 6px', letterSpacing: '-0.5px' }}
          className="text-gradient-bio"
        >
          Model Selector
        </h1>
        <p style={{ fontSize: 14, color: COLORS.textMuted, margin: 0 }}>
          Assign different models to each evolution role
        </p>
      </motion.div>

      {/* Role assignment summary */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 28 }}>
          {ROLE_CONFIG.map((role) => {
            const assigned = MODELS.find((m) => m.id === assignment[role.id])
            const isActive = selectedRole === role.id
            return (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.id)}
                style={{
                  padding: 0,
                  border: 'none',
                  cursor: 'pointer',
                  background: 'transparent',
                  textAlign: 'left',
                }}
              >
                <GlassPanel
                  style={{
                    padding: '18px 20px',
                    borderColor: isActive ? `${role.color}44` : undefined,
                    boxShadow: isActive ? `0 0 20px ${role.color}18` : undefined,
                    transition: 'all 0.2s',
                  }}
                >
                  <div style={{ fontSize: 10, fontWeight: 700, color: role.color, textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 8 }}>
                    {role.label}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textPrimary, marginBottom: 4 }}>
                    {assigned?.name ?? 'Not assigned'}
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.textMuted }}>{role.description}</div>
                  {assigned && (
                    <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
                      <span style={{ fontSize: 10, color: MODEL_PROVIDER_COLORS[assigned.provider], background: `${MODEL_PROVIDER_COLORS[assigned.provider]}18`, padding: '2px 7px', borderRadius: 5 }}>
                        {MODEL_PROVIDER_LABELS[assigned.provider]}
                      </span>
                      <span style={{ fontSize: 10, color: COLORS.textMuted }}>
                        {(assigned.context_window / 1000).toFixed(0)}k ctx
                      </span>
                      <span style={{ fontSize: 10, color: COLORS.textMuted }}>
                        ${assigned.cost_per_1k}/1k
                      </span>
                    </div>
                  )}
                </GlassPanel>
              </button>
            )
          })}
        </div>
      </motion.div>

      {/* Model list */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <GlassPanel style={{ padding: '16px 20px', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 12, color: COLORS.textMuted, fontWeight: 600 }}>Provider:</span>
            {(['all', ...allProviders] as (ModelProvider | 'all')[]).map((p) => {
              const active = filterProvider === p
              const col = p === 'all' ? COLORS.textSecondary : MODEL_PROVIDER_COLORS[p]
              return (
                <button
                  key={p}
                  onClick={() => setFilterProvider(p)}
                  style={{
                    padding: '4px 12px',
                    borderRadius: 7,
                    fontSize: 12,
                    cursor: 'pointer',
                    border: `1px solid ${active ? col + '55' : 'rgba(255,255,255,0.1)'}`,
                    background: active ? `${col}18` : 'transparent',
                    color: active ? col : COLORS.textMuted,
                    transition: 'all 0.15s',
                    fontWeight: active ? 600 : 400,
                  }}
                >
                  {p === 'all' ? 'All' : MODEL_PROVIDER_LABELS[p]}
                </button>
              )
            })}

            <span style={{ marginLeft: 'auto', fontSize: 12, color: COLORS.textMuted }}>
              Assigning to: <strong style={{ color: ROLE_CONFIG.find((r) => r.id === selectedRole)?.color }}>{selectedRole}</strong>
            </span>
          </div>
        </GlassPanel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {filteredModels.map((model, i) => {
            const isAssigned = Object.values(assignment).includes(model.id)
            const assignedRoles = ROLE_CONFIG.filter((r) => assignment[r.id] === model.id)
            const statusCfg = STATUS_CONFIG[model.status]
            const provColor = MODEL_PROVIDER_COLORS[model.provider]

            return (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                <GlassPanel
                  style={{
                    padding: '14px 18px',
                    opacity: model.status === 'offline' ? 0.5 : 1,
                    borderColor: assignedRoles.some((r) => r.id === selectedRole) ? `${ROLE_CONFIG.find((r) => r.id === selectedRole)?.color}44` : undefined,
                    cursor: model.status !== 'offline' ? 'pointer' : 'not-allowed',
                    transition: 'all 0.15s',
                  }}
                  onClick={() => {
                    if (model.status === 'offline') return
                    setAssignment((prev) => ({ ...prev, [selectedRole]: model.id }))
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    {/* Provider badge */}
                    <div
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: provColor,
                        flexShrink: 0,
                        boxShadow: `0 0 6px ${provColor}`,
                      }}
                    />

                    {/* Name + provider */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 14, fontWeight: 600, color: COLORS.textPrimary }}>{model.name}</span>
                        <span style={{ fontSize: 10, color: provColor, background: `${provColor}18`, padding: '2px 7px', borderRadius: 5, fontWeight: 600 }}>
                          {MODEL_PROVIDER_LABELS[model.provider]}
                        </span>
                      </div>
                    </div>

                    {/* Context */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 60 }}>
                      <span style={{ fontSize: 11, color: COLORS.textMuted }}>{(model.context_window / 1000).toFixed(0)}k ctx</span>
                    </div>

                    {/* Speed */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 64 }}>
                      <Clock size={11} color={COLORS.textMuted} />
                      <span style={{ fontSize: 11, color: model.speed === 'fast' ? COLORS.green : model.speed === 'slow' ? COLORS.magenta : COLORS.amber }}>
                        {SPEED_LABEL[model.speed]}
                      </span>
                    </div>

                    {/* Cost */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 72 }}>
                      <DollarSign size={11} color={COLORS.textMuted} />
                      <span style={{ fontSize: 11, color: model.cost_per_1k === 0 ? COLORS.green : COLORS.textSecondary }}>
                        {model.cost_per_1k === 0 ? 'free' : `$${model.cost_per_1k}/1k`}
                      </span>
                    </div>

                    {/* Status */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 88 }}>
                      <span style={{ color: statusCfg.color }}>{statusCfg.icon}</span>
                      <span style={{ fontSize: 11, color: statusCfg.color }}>{statusCfg.label}</span>
                    </div>

                    {/* Assignment indicators */}
                    <div style={{ display: 'flex', gap: 4, minWidth: 80 }}>
                      {assignedRoles.map((r) => (
                        <span
                          key={r.id}
                          style={{
                            fontSize: 9,
                            fontWeight: 700,
                            color: r.color,
                            background: `${r.color}18`,
                            border: `1px solid ${r.color}33`,
                            borderRadius: 4,
                            padding: '2px 6px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.06em',
                          }}
                        >
                          {r.label}
                        </span>
                      ))}
                      {!isAssigned && model.status !== 'offline' && (
                        <span style={{ fontSize: 11, color: COLORS.textMuted }}>select</span>
                      )}
                      {assignedRoles.some((r) => r.id === selectedRole) && (
                        <Check size={14} color={ROLE_CONFIG.find((r) => r.id === selectedRole)?.color} />
                      )}
                    </div>

                    {/* Quick select for active role */}
                    {model.status !== 'offline' && (
                      <GlowButton
                        color={ROLE_CONFIG.find((r) => r.id === selectedRole)?.color === COLORS.cyan ? 'cyan' : ROLE_CONFIG.find((r) => r.id === selectedRole)?.color === COLORS.magenta ? 'magenta' : 'green'}
                        size="sm"
                        variant={assignedRoles.some((r) => r.id === selectedRole) ? 'solid' : 'outline'}
                        onClick={(e) => {
                          e.stopPropagation()
                          setAssignment((prev) => ({ ...prev, [selectedRole]: model.id }))
                        }}
                      >
                        {assignedRoles.some((r) => r.id === selectedRole) ? 'Selected' : 'Select'}
                      </GlowButton>
                    )}
                  </div>
                </GlassPanel>
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      <div style={{ height: 32 }} />
    </div>
  )
}

export default ModelSelector
