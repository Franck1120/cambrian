import { motion } from 'framer-motion'
import * as Icons from 'lucide-react'
import { BarChart2 } from 'lucide-react'
import type { LucideProps } from 'lucide-react'
import { COLORS, CATEGORY_COLOR, GLASS } from '../../lib/theme'
import type { Plugin } from '../../types'

interface PluginCardProps {
  plugin: Plugin
  onToggle: (id: string) => void
}

type IconName = keyof typeof Icons

const DynamicIcon = ({ name, size = 16, color }: { name: string; size?: number; color?: string }) => {
  const LucideIcon = (Icons[name as IconName] as React.ComponentType<LucideProps>) ?? BarChart2
  return <LucideIcon size={size} color={color} />
}

const CATEGORY_LABELS: Record<string, string> = {
  mutation: 'Mutation',
  selection: 'Selection',
  memory: 'Memory',
  diversity: 'Diversity',
  evaluation: 'Evaluation',
  crossover: 'Crossover',
}

const PluginCard = ({ plugin, onToggle }: PluginCardProps) => {
  const catColor = CATEGORY_COLOR[plugin.category] ?? COLORS.textMuted

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      style={{
        ...GLASS,
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        opacity: plugin.enabled ? 1 : 0.6,
        transition: 'opacity 0.2s',
        borderColor: plugin.enabled ? `${catColor}22` : 'rgba(255,255,255,0.06)',
        cursor: 'default',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Icon */}
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              background: `${catColor}18`,
              border: `1px solid ${catColor}33`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <DynamicIcon name={plugin.icon} size={16} color={catColor} />
          </div>

          {/* Name + category */}
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: COLORS.textPrimary,
                lineHeight: 1.3,
              }}
            >
              {plugin.name}
            </div>
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: catColor,
                textTransform: 'uppercase',
                letterSpacing: '0.07em',
              }}
            >
              {CATEGORY_LABELS[plugin.category]}
            </span>
          </div>
        </div>

        {/* Toggle */}
        <button
          onClick={() => onToggle(plugin.id)}
          style={{
            width: 38,
            height: 22,
            borderRadius: 11,
            background: plugin.enabled ? `${catColor}33` : 'rgba(255,255,255,0.08)',
            border: `1px solid ${plugin.enabled ? catColor + '55' : 'rgba(255,255,255,0.1)'}`,
            cursor: 'pointer',
            position: 'relative',
            flexShrink: 0,
            transition: 'background 0.2s, border-color 0.2s',
          }}
        >
          <motion.div
            animate={{ left: plugin.enabled ? 18 : 3 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            style={{
              position: 'absolute',
              top: 3,
              width: 14,
              height: 14,
              borderRadius: '50%',
              background: plugin.enabled ? catColor : 'rgba(255,255,255,0.35)',
              boxShadow: plugin.enabled ? `0 0 6px ${catColor}` : 'none',
            }}
          />
        </button>
      </div>

      {/* Description */}
      <p
        style={{
          fontSize: 12,
          color: COLORS.textSecondary,
          lineHeight: 1.5,
          margin: 0,
        }}
      >
        {plugin.description}
      </p>

      {/* Footer: impact badge + tags */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {plugin.impact !== null ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: plugin.impact > 0.7 ? COLORS.green : plugin.impact > 0.5 ? COLORS.amber : COLORS.magenta,
              }}
            />
            <span style={{ fontSize: 11, color: COLORS.textMuted }}>
              +{(plugin.impact * 100).toFixed(0)}% impact
            </span>
          </div>
        ) : (
          <span style={{ fontSize: 11, color: COLORS.textMuted, fontStyle: 'italic' }}>no A/B data</span>
        )}

        <div style={{ display: 'flex', gap: 4 }}>
          {plugin.tags.slice(0, 2).map((tag) => (
            <span
              key={tag}
              style={{
                fontSize: 10,
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 5,
                padding: '1px 6px',
                color: COLORS.textMuted,
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default PluginCard
