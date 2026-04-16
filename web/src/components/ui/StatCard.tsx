import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import GlassPanel from './GlassPanel'
import { COLORS } from '../../lib/theme'

type StatColor = 'cyan' | 'magenta' | 'green' | 'amber' | 'default'

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  trend?: number
  sparkline?: number[]
  color?: StatColor
  icon?: React.ReactNode
  animateIn?: boolean
}

const COLOR_MAP: Record<StatColor, string> = {
  cyan: COLORS.cyan,
  magenta: COLORS.magenta,
  green: COLORS.green,
  amber: COLORS.amber,
  default: COLORS.textPrimary,
}

const Sparkline = ({ data, color }: { data: number[]; color: string }) => {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const w = 80
  const h = 28
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w
      const y = h - ((v - min) / range) * h
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.8}
      />
      <circle
        cx={(data.length - 1) / (data.length - 1) * w}
        cy={h - ((data[data.length - 1] - min) / range) * h}
        r={2.5}
        fill={color}
      />
    </svg>
  )
}

const StatCard = ({
  label,
  value,
  sub,
  trend,
  sparkline,
  color = 'default',
  icon,
  animateIn = true,
}: StatCardProps) => {
  const c = COLOR_MAP[color]

  const TrendIcon =
    trend === undefined || trend === 0 ? Minus : trend > 0 ? TrendingUp : TrendingDown
  const trendColor =
    trend === undefined || trend === 0 ? COLORS.textMuted : trend > 0 ? COLORS.green : COLORS.magenta

  return (
    <motion.div
      initial={animateIn ? { opacity: 0, y: 16 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      <GlassPanel style={{ padding: '20px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: COLORS.textMuted,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            {label}
          </span>
          {icon && <span style={{ color: COLORS.textMuted, opacity: 0.7 }}>{icon}</span>}
        </div>

        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            color: c,
            letterSpacing: '-1px',
            fontFamily: "'Space Grotesk', sans-serif",
            lineHeight: 1,
            marginBottom: 10,
          }}
        >
          {value}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {trend !== undefined && (
              <span style={{ display: 'flex', alignItems: 'center', gap: 3, color: trendColor, fontSize: 12 }}>
                <TrendIcon size={12} />
                {Math.abs(trend).toFixed(2)}
              </span>
            )}
            {sub && (
              <span style={{ fontSize: 12, color: COLORS.textMuted }}>
                {sub}
              </span>
            )}
          </div>
          {sparkline && sparkline.length > 1 && (
            <Sparkline data={sparkline} color={c} />
          )}
        </div>
      </GlassPanel>
    </motion.div>
  )
}

export default StatCard
