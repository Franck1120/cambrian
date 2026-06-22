import { motion } from 'framer-motion'
import { COLORS, STATUS_COLOR, STATUS_GLOW } from '../../lib/theme'
import type { Agent } from '../../types'

interface AgentNodeProps {
  agent: Agent
  isSelected?: boolean
  onClick?: (agent: Agent) => void
  size?: 'sm' | 'md'
}

const AgentNode = ({ agent, isSelected = false, onClick, size = 'md' }: AgentNodeProps) => {
  const color = STATUS_COLOR[agent.status] ?? COLORS.textMuted
  const glow = STATUS_GLOW[agent.status] ?? 'none'
  const nodeSize = size === 'sm' ? 32 : 40

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.6 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => onClick?.(agent)}
      style={{
        position: 'relative',
        width: nodeSize,
        height: nodeSize,
        cursor: onClick ? 'pointer' : 'default',
        flexShrink: 0,
      }}
      title={`${agent.id} · fitness ${agent.fitness.toFixed(2)} · ${agent.strategy}`}
    >
      {/* Outer glow ring when selected */}
      {isSelected && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            position: 'absolute',
            inset: -3,
            borderRadius: '50%',
            border: `2px solid ${color}`,
            boxShadow: glow,
          }}
        />
      )}

      {/* Node circle */}
      <div
        style={{
          width: nodeSize,
          height: nodeSize,
          borderRadius: '50%',
          background: `radial-gradient(circle at 35% 35%, ${color}44 0%, ${color}11 60%, transparent 100%)`,
          border: `1.5px solid ${color}${isSelected ? 'cc' : '66'}`,
          boxShadow: isSelected ? glow : agent.status === 'elite' ? `0 0 8px ${color}44` : 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 1,
          transition: 'box-shadow 0.2s',
        }}
      >
        {/* Fitness number */}
        <span
          style={{
            fontSize: 9,
            fontWeight: 700,
            color,
            fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1,
          }}
        >
          {agent.fitness.toFixed(2)}
        </span>
      </div>

      {/* Elite star */}
      {agent.status === 'elite' && (
        <div
          style={{
            position: 'absolute',
            top: -2,
            right: -2,
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: COLORS.amber,
            boxShadow: `0 0 6px ${COLORS.amber}`,
          }}
        />
      )}
    </motion.div>
  )
}

export default AgentNode
