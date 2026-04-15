import { useState } from 'react'
import { MOCK_PIPELINE, type PipelineStep } from '../data/mock'

const ACCENT = '#10b981'
const CARD = { background: '#111111', border: '1px solid #1a1a1a', borderRadius: 10 }

const stepColors: Record<PipelineStep['type'], { bg: string; text: string; border: string }> = {
  mutate:    { bg: '#0d2b1f', text: '#10b981', border: '#10b98133' },
  evaluate:  { bg: '#1a1a2e', text: '#818cf8', border: '#818cf833' },
  select:    { bg: '#2a1a0e', text: '#f59e0b', border: '#f59e0b33' },
  crossover: { bg: '#1f1a2e', text: '#a78bfa', border: '#a78bfa33' },
  filter:    { bg: '#2a1414', text: '#ef4444', border: '#ef444433' },
}

const ParamBadge = ({ k, v }: { k: string; v: string | number | boolean }) => (
  <span style={{ background: '#1a1a1a', border: '1px solid #222', borderRadius: 5,
    padding: '2px 8px', fontSize: 11, color: '#9ca3af', display: 'inline-flex',
    alignItems: 'center', gap: 4 }}>
    <span style={{ color: '#6b7280' }}>{k}=</span>
    <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{String(v)}</span>
  </span>
)

const StepCard = ({
  step, index, total,
  onToggle,
}: {
  step: PipelineStep
  index: number
  total: number
  onToggle: (id: string) => void
}) => {
  const colors = stepColors[step.type]
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 0 }}>
      {/* Connector line */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
        width: 40, flexShrink: 0 }}>
        <div style={{ width: 2, height: 20, background: index === 0 ? 'transparent' : '#1a1a1a' }} />
        <div style={{ width: 28, height: 28, borderRadius: '50%',
          background: step.enabled ? colors.bg : '#111',
          border: `2px solid ${step.enabled ? colors.border : '#1a1a1a'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700, color: step.enabled ? colors.text : '#333',
          flexShrink: 0, zIndex: 1 }}>
          {index + 1}
        </div>
        <div style={{ width: 2, flex: 1,
          background: index === total - 1 ? 'transparent' : '#1a1a1a' }} />
      </div>

      {/* Card */}
      <div style={{ flex: 1, marginBottom: 8,
        ...CARD,
        opacity: step.enabled ? 1 : 0.45,
        border: step.enabled ? `1px solid ${colors.border}` : '1px solid #1a1a1a',
        transition: 'all 0.15s' }}>
        <div style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 14 }}>
          {/* Type badge */}
          <span style={{ background: colors.bg, color: colors.text, fontSize: 11,
            fontWeight: 600, padding: '3px 9px', borderRadius: 5,
            textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
            {step.type}
          </span>

          {/* Label */}
          <span style={{ fontSize: 14, fontWeight: 600, color: '#f0f0f0', flex: 1 }}>
            {step.label}
          </span>

          {/* Params */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, flex: 2 }}>
            {Object.entries(step.params).map(([k, v]) => (
              <ParamBadge key={k} k={k} v={v} />
            ))}
          </div>

          {/* Toggle */}
          <button onClick={() => onToggle(step.id)}
            style={{ background: step.enabled ? ACCENT : '#1a1a1a',
              border: 'none', borderRadius: 12, cursor: 'pointer',
              width: 40, height: 22, position: 'relative', flexShrink: 0,
              transition: 'background 0.15s' }}>
            <div style={{ position: 'absolute', top: 3,
              left: step.enabled ? 20 : 3, width: 16, height: 16,
              borderRadius: '50%', background: step.enabled ? '#0a0a0a' : '#444',
              transition: 'left 0.15s' }} />
          </button>
        </div>
      </div>
    </div>
  )
}

const Forge = () => {
  const [steps, setSteps] = useState<PipelineStep[]>(MOCK_PIPELINE)
  const [taskSpec, setTaskSpec] = useState('reverse(s: str) -> str')
  const [testCase, setTestCase] = useState('hello:olleh')

  const toggleStep = (id: string) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, enabled: !s.enabled } : s))
  }

  const enabledCount = steps.filter(s => s.enabled).length

  return (
    <div style={{ padding: '32px 40px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>
          Forge
        </h1>
        <p style={{ fontSize: 14, color: '#6b7280' }}>
          Build and configure the evolution pipeline step by step
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24 }}>
        {/* Pipeline */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: 16 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#6b7280',
              textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Pipeline Steps
            </span>
            <span style={{ fontSize: 12, color: '#6b7280' }}>
              {enabledCount} / {steps.length} active
            </span>
          </div>

          {steps.map((step, i) => (
            <StepCard key={step.id} step={step} index={i}
              total={steps.length} onToggle={toggleStep} />
          ))}

          {/* Add step button */}
          <div style={{ display: 'flex', alignItems: 'stretch', gap: 0, marginTop: 4 }}>
            <div style={{ width: 40, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ width: 2, height: 20, background: '#1a1a1a' }} />
            </div>
            <button style={{ flex: 1, ...CARD, border: '1px dashed #222',
              background: 'transparent', padding: '12px 18px', cursor: 'pointer',
              color: '#4b5563', fontSize: 13, fontFamily: 'Inter, sans-serif',
              transition: 'all 0.15s', marginBottom: 8, textAlign: 'left' }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = ACCENT + '55'
                e.currentTarget.style.color = ACCENT
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = '#222'
                e.currentTarget.style.color = '#4b5563'
              }}>
              + Add Step
            </button>
          </div>
        </div>

        {/* Config panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Task spec */}
          <div style={{ ...CARD, padding: '20px' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280',
              textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>
              Task Specification
            </div>
            <label style={{ display: 'block', fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
              Function Signature
            </label>
            <input value={taskSpec} onChange={e => setTaskSpec(e.target.value)}
              style={{ width: '100%', background: '#0d0d0d', border: '1px solid #222',
                borderRadius: 7, padding: '9px 12px', color: '#f0f0f0', fontSize: 13,
                fontFamily: 'monospace', outline: 'none', marginBottom: 12 }}
              onFocus={e => e.target.style.borderColor = ACCENT}
              onBlur={e => e.target.style.borderColor = '#222'}
            />
            <label style={{ display: 'block', fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
              Test Case (input:expected)
            </label>
            <input value={testCase} onChange={e => setTestCase(e.target.value)}
              style={{ width: '100%', background: '#0d0d0d', border: '1px solid #222',
                borderRadius: 7, padding: '9px 12px', color: '#f0f0f0', fontSize: 13,
                fontFamily: 'monospace', outline: 'none' }}
              onFocus={e => e.target.style.borderColor = ACCENT}
              onBlur={e => e.target.style.borderColor = '#222'}
            />
          </div>

          {/* Pipeline summary */}
          <div style={{ ...CARD, padding: '20px' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280',
              textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>
              Summary
            </div>
            {steps.filter(s => s.enabled).map(s => {
              const c = stepColors[s.type]
              return (
                <div key={s.id} style={{ display: 'flex', alignItems: 'center',
                  gap: 8, marginBottom: 8 }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%',
                    background: c.text, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: '#d1d5db' }}>{s.label}</span>
                </div>
              )
            })}
            <button style={{ width: '100%', marginTop: 16, background: ACCENT,
              border: 'none', borderRadius: 8, padding: '11px', color: '#0a0a0a',
              fontWeight: 600, fontSize: 14, cursor: 'pointer',
              fontFamily: 'Inter, sans-serif' }}>
              Run Forge
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Forge
