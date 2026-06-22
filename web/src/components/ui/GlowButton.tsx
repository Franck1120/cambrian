import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { motion } from 'framer-motion'
import { COLORS, GLOW } from '../../lib/theme'

type GlowColor = 'cyan' | 'magenta' | 'green' | 'amber'
type GlowVariant = 'solid' | 'outline' | 'ghost'
type GlowSize = 'sm' | 'md' | 'lg'

interface GlowButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  color?: GlowColor
  variant?: GlowVariant
  size?: GlowSize
  loading?: boolean
  icon?: ReactNode
}

const COLOR_MAP: Record<GlowColor, string> = {
  cyan: COLORS.cyan,
  magenta: COLORS.magenta,
  green: COLORS.green,
  amber: COLORS.amber,
}

const SIZE_MAP: Record<GlowSize, { height: number; px: number; fontSize: number }> = {
  sm: { height: 32, px: 14, fontSize: 12 },
  md: { height: 40, px: 20, fontSize: 14 },
  lg: { height: 48, px: 28, fontSize: 15 },
}

const GlowButton = ({
  children,
  color = 'cyan',
  variant = 'solid',
  size = 'md',
  loading = false,
  icon,
  disabled,
  style,
  ...props
}: GlowButtonProps) => {
  const c = COLOR_MAP[color]
  const s = SIZE_MAP[size]
  const isDisabled = disabled || loading

  const baseStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: s.height,
    paddingLeft: s.px,
    paddingRight: s.px,
    borderRadius: 10,
    fontSize: s.fontSize,
    fontWeight: 600,
    fontFamily: "'Inter', sans-serif",
    cursor: isDisabled ? 'not-allowed' : 'pointer',
    border: 'none',
    whiteSpace: 'nowrap' as const,
    opacity: isDisabled && !loading ? 0.5 : 1,
    width: '100%',
    ...style,
  }

  const variantStyle: React.CSSProperties =
    variant === 'solid'
      ? {
          background: `linear-gradient(135deg, ${c}28 0%, ${c}14 100%)`,
          border: `1px solid ${c}55`,
          color: c,
          boxShadow: isDisabled ? 'none' : `0 0 12px ${c}22`,
        }
      : variant === 'outline'
        ? {
            background: 'transparent',
            border: `1px solid ${c}44`,
            color: c,
          }
        : {
            background: 'transparent',
            border: '1px solid transparent',
            color: c,
          }

  return (
    <motion.div
      style={{ display: 'inline-flex' }}
      whileHover={isDisabled ? {} : { boxShadow: GLOW[color], scale: 1.02 }}
      whileTap={isDisabled ? {} : { scale: 0.97 }}
    >
      <button style={{ ...baseStyle, ...variantStyle }} disabled={isDisabled} {...props}>
        {loading ? (
          <span
            style={{
              display: 'inline-block',
              width: 14,
              height: 14,
              border: `2px solid ${c}44`,
              borderTopColor: c,
              borderRadius: '50%',
            }}
            className="animate-spin"
          />
        ) : (
          icon
        )}
        {children}
      </button>
    </motion.div>
  )
}

export default GlowButton
