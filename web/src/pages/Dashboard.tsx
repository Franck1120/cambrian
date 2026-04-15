import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from 'recharts'
import { MOCK_HISTORY, MOCK_AGENTS } from '../data/mock'

const ACCENT = '#10b981'
const CARD = { background: '#111111', border: '1px solid #1a1a1a', borderRadius: 10 }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#161616', border: '1px solid #222', borderRadius: 8,
      padding: '10px 14px', fontSize: 13 }}>
      <div style={{ color: '#9ca3af', marginBottom: 4 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.fill || ACCENT }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</strong>
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

const Dashboard = () => (
  <div style={{ padding: '32px 40px', maxWidth: 1200, margin: '0 auto' }}>
    {/* Header */}
    <div style={{ marginBottom: 28 }}>
      <h1 style={{ fontSize: 22, fontWeight: 600, color: '#f0f0f0', marginBottom: 4 }}>
        Dashboard
      </h1>
      <p style={{ fontSize: 14, color: '#6b7280' }}>
        Historical runs and top performing agents
      </p>
    </div>

    {/* Stats row */}
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
      {[
        { label: 'Total Runs', value: '5', sub: 'all time' },
        { label: 'Best Fitness', value: '0.94', sub: 'run-001' },
        { label: 'Avg Generations', value: '8.8', sub: 'per run' },
        { label: 'Total Agents', value: '284', sub: 'evolved' },
      ].map(s => (
        <div key={s.label} style={{ ...CARD, padding: '18px 20px' }}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6,
            textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            {s.label}
          </div>
          <div style={{ fontSize: 26, fontWeight: 700, color: '#f0f0f0', letterSpacing: '-0.5px' }}>
            {s.value}
          </div>
          <div style={{ fontSize: 12, color: '#4b5563', marginTop: 2 }}>{s.sub}</div>
        </div>
      ))}
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
      {/* Fitness by run */}
      <div style={{ ...CARD, padding: '20px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 16,
          textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Best Fitness per Run
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={MOCK_HISTORY.map(r => ({ name: r.id.split('-')[1], fitness: r.best_fitness }))}
            margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="name" stroke="#333" tick={{ fill: '#6b7280', fontSize: 12 }} />
            <YAxis domain={[0.7, 1]} stroke="#333" tick={{ fill: '#6b7280', fontSize: 12 }} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="fitness" name="Fitness" fill={ACCENT} radius={[4, 4, 0, 0]}
              maxBarSize={40} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Performance radar */}
      <div style={{ ...CARD, padding: '20px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 16,
          textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Best Agent Profile
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <RadarChart data={radarData} margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
            <PolarGrid stroke="#1a1a1a" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 11 }} />
            <Radar name="Agent" dataKey="value" stroke={ACCENT} fill={ACCENT}
              fillOpacity={0.15} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>

    {/* Run history table */}
    <div style={{ ...CARD, marginBottom: 24, overflow: 'hidden' }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #1a1a1a',
        fontSize: 12, fontWeight: 600, color: '#6b7280',
        textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        Run History
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #1a1a1a' }}>
            {['Run', 'Task', 'Gen', 'Pop', 'Best Fitness', 'Duration', 'Date'].map(h => (
              <th key={h} style={{ padding: '10px 16px', textAlign: 'left',
                fontSize: 11, fontWeight: 600, color: '#4b5563',
                textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MOCK_HISTORY.map((run, i) => (
            <tr key={run.id}
              style={{ borderBottom: i < MOCK_HISTORY.length - 1 ? '1px solid #141414' : 'none',
                transition: 'background 0.1s', cursor: 'pointer' }}
              onMouseEnter={e => (e.currentTarget.style.background = '#161616')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
              <td style={{ padding: '11px 16px', fontSize: 12,
                color: '#6b7280', fontFamily: 'monospace' }}>
                {run.id}
              </td>
              <td style={{ padding: '11px 16px', fontSize: 13, color: '#d1d5db',
                maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {run.task}
              </td>
              <td style={{ padding: '11px 16px', fontSize: 13, color: '#9ca3af' }}>
                {run.generations}
              </td>
              <td style={{ padding: '11px 16px', fontSize: 13, color: '#9ca3af' }}>
                {run.population}
              </td>
              <td style={{ padding: '11px 16px' }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: ACCENT }}>
                  {run.best_fitness.toFixed(2)}
                </span>
              </td>
              <td style={{ padding: '11px 16px', fontSize: 13, color: '#6b7280' }}>
                {run.duration_s}s
              </td>
              <td style={{ padding: '11px 16px', fontSize: 12, color: '#4b5563' }}>
                {run.created_at}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    {/* Top agents */}
    <div style={{ ...CARD, overflow: 'hidden' }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #1a1a1a',
        fontSize: 12, fontWeight: 600, color: '#6b7280',
        textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        Top Agents (All Time)
      </div>
      <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {topAgents.map((agent, i) => (
          <div key={agent.id} style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: i === 0 ? ACCENT : '#4b5563',
              minWidth: 18 }}>
              #{i + 1}
            </span>
            <span style={{ fontSize: 12, color: '#4b5563', fontFamily: 'monospace',
              minWidth: 60 }}>
              {agent.id}
            </span>
            <div style={{ flex: 1, height: 4, background: '#1a1a1a', borderRadius: 2 }}>
              <div style={{ height: '100%', width: `${agent.fitness * 100}%`,
                background: i === 0 ? ACCENT : '#2a2a2a', borderRadius: 2,
                transition: 'width 0.3s ease' }} />
            </div>
            <span style={{ fontSize: 13, fontWeight: 600, color: i === 0 ? ACCENT : '#9ca3af',
              minWidth: 36 }}>
              {agent.fitness.toFixed(2)}
            </span>
            <span style={{ fontSize: 12, background: '#1a1a1a', borderRadius: 5,
              padding: '2px 8px', color: '#6b7280' }}>
              {agent.strategy}
            </span>
          </div>
        ))}
      </div>
    </div>
  </div>
)

export default Dashboard
