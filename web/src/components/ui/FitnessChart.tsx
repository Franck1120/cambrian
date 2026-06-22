import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import GlassPanel from './GlassPanel'
import { COLORS } from '../../lib/theme'
import type { GenerationStat } from '../../types'

interface TooltipPayloadItem {
  name: string
  value: number
  color: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string | number
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (!active || !payload?.length) return null
  return (
    <div
      style={{
        background: 'rgba(5, 8, 25, 0.92)',
        border: '1px solid rgba(0,240,255,0.2)',
        borderRadius: 10,
        padding: '10px 14px',
        fontSize: 12,
        backdropFilter: 'blur(12px)',
      }}
    >
      <div style={{ color: COLORS.textMuted, marginBottom: 8, fontWeight: 600 }}>
        Generation {label}
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, marginBottom: 4 }}>
          {p.name}: <strong>{p.value.toFixed(3)}</strong>
        </div>
      ))}
    </div>
  )
}

interface FitnessChartProps {
  data: GenerationStat[]
  showDiversity?: boolean
  height?: number
  title?: string
}

const FitnessChart = ({
  data,
  showDiversity = false,
  height = 220,
  title = 'Fitness Over Generations',
}: FitnessChartProps) => (
  <GlassPanel style={{ padding: '20px 16px 12px' }}>
    <div
      style={{
        fontSize: 11,
        fontWeight: 700,
        color: COLORS.textMuted,
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: 16,
        paddingLeft: 8,
      }}
    >
      {title}
    </div>
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="gradBest" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.cyan} stopOpacity={0.25} />
            <stop offset="95%" stopColor={COLORS.cyan} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradAvg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.magenta} stopOpacity={0.2} />
            <stop offset="95%" stopColor={COLORS.magenta} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradDiv" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLORS.green} stopOpacity={0.2} />
            <stop offset="95%" stopColor={COLORS.green} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis
          dataKey="generation"
          stroke="rgba(255,255,255,0.1)"
          tick={{ fill: COLORS.textMuted, fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          domain={[0, 1]}
          stroke="rgba(255,255,255,0.1)"
          tick={{ fill: COLORS.textMuted, fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, color: COLORS.textMuted, paddingTop: 8 }}
        />
        <Area
          type="monotone"
          dataKey="best_fitness"
          name="Best"
          stroke={COLORS.cyan}
          fill="url(#gradBest)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: COLORS.cyan, strokeWidth: 0 }}
        />
        <Area
          type="monotone"
          dataKey="avg_fitness"
          name="Avg"
          stroke={COLORS.magenta}
          fill="url(#gradAvg)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: COLORS.magenta, strokeWidth: 0 }}
        />
        {showDiversity && (
          <Area
            type="monotone"
            dataKey="diversity"
            name="Diversity"
            stroke={COLORS.green}
            fill="url(#gradDiv)"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            activeDot={{ r: 3, fill: COLORS.green, strokeWidth: 0 }}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  </GlassPanel>
)

export default FitnessChart
