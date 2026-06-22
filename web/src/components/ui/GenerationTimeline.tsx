import { useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import AgentNode from './AgentNode'
import { COLORS, GLASS } from '../../lib/theme'
import type { Agent, GenerationStat } from '../../types'

interface GenerationTimelineProps {
  generations: GenerationStat[]
  agents: Agent[]
  onAgentClick?: (agent: Agent) => void
  selectedAgentId?: string
}

const GenerationTimeline = ({
  generations,
  agents,
  onAgentClick,
  selectedAgentId,
}: GenerationTimelineProps) => {
  const scrollRef = useRef<HTMLDivElement>(null)

  const scroll = (dir: 'left' | 'right') => {
    if (!scrollRef.current) return
    scrollRef.current.scrollBy({ left: dir === 'left' ? -240 : 240, behavior: 'smooth' })
  }

  const agentsByGen = (gen: number) =>
    agents
      .filter((a) => a.generation === gen)
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, 5)

  return (
    <div style={{ position: 'relative' }}>
      {/* Scroll controls */}
      <button
        onClick={() => scroll('left')}
        style={{
          position: 'absolute',
          left: -16,
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 10,
          width: 32,
          height: 32,
          borderRadius: '50%',
          background: 'rgba(10,14,40,0.9)',
          border: '1px solid rgba(0,240,255,0.2)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: COLORS.cyan,
        }}
      >
        <ChevronLeft size={16} />
      </button>
      <button
        onClick={() => scroll('right')}
        style={{
          position: 'absolute',
          right: -16,
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 10,
          width: 32,
          height: 32,
          borderRadius: '50%',
          background: 'rgba(10,14,40,0.9)',
          border: '1px solid rgba(0,240,255,0.2)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: COLORS.cyan,
        }}
      >
        <ChevronRight size={16} />
      </button>

      {/* Scrollable container */}
      <div
        ref={scrollRef}
        style={{
          overflowX: 'auto',
          overflowY: 'visible',
          paddingBottom: 8,
          scrollbarWidth: 'thin',
        }}
      >
        <div style={{ display: 'flex', gap: 16, padding: '4px 0', minWidth: 'max-content' }}>
          {generations.map((gen, i) => {
            const genAgents = agentsByGen(gen.generation)
            const bestFitness = gen.best_fitness

            return (
              <motion.div
                key={gen.generation}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05, duration: 0.3 }}
                style={{
                  ...GLASS,
                  minWidth: 100,
                  padding: '12px 10px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                  alignItems: 'center',
                  borderRadius: 12,
                }}
              >
                {/* Gen header */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      color: COLORS.textMuted,
                      textTransform: 'uppercase',
                      letterSpacing: '0.08em',
                    }}
                  >
                    Gen {gen.generation}
                  </span>
                  <span
                    style={{
                      fontSize: 13,
                      fontWeight: 700,
                      color: COLORS.cyan,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    {bestFitness.toFixed(2)}
                  </span>
                </div>

                {/* Fitness bar */}
                <div
                  style={{
                    width: '100%',
                    height: 3,
                    background: 'rgba(255,255,255,0.08)',
                    borderRadius: 2,
                    overflow: 'hidden',
                  }}
                >
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${bestFitness * 100}%` }}
                    transition={{ delay: i * 0.05 + 0.2, duration: 0.5 }}
                    style={{
                      height: '100%',
                      background: `linear-gradient(90deg, ${COLORS.cyan}, ${COLORS.magenta})`,
                      borderRadius: 2,
                    }}
                  />
                </div>

                {/* Agent nodes */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
                  {genAgents.map((agent) => (
                    <AgentNode
                      key={agent.id}
                      agent={agent}
                      size="sm"
                      isSelected={agent.id === selectedAgentId}
                      onClick={onAgentClick}
                    />
                  ))}
                  {genAgents.length === 0 && (
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        border: '1px dashed rgba(255,255,255,0.1)',
                      }}
                    />
                  )}
                </div>

                {/* Diversity indicator */}
                <span
                  style={{
                    fontSize: 10,
                    color: COLORS.textMuted,
                    opacity: 0.7,
                  }}
                >
                  div {gen.diversity.toFixed(2)}
                </span>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default GenerationTimeline
