import type { HTMLAttributes, ReactNode } from 'react'
import { GLASS, GLASS_MAGENTA, GLASS_DARK } from '../../lib/theme'

type GlowVariant = 'cyan' | 'magenta' | 'dark' | 'none'

interface GlassPanelProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  glow?: GlowVariant
}

const VARIANT_STYLE: Record<GlowVariant, React.CSSProperties> = {
  cyan: GLASS,
  magenta: GLASS_MAGENTA,
  dark: GLASS_DARK,
  none: {
    background: 'rgba(10, 14, 40, 0.5)',
    border: '1px solid rgba(255,255,255,0.05)',
    borderRadius: 16,
  },
}

const GlassPanel = ({ children, glow = 'cyan', style, className, ...props }: GlassPanelProps) => (
  <div
    style={{ ...VARIANT_STYLE[glow], ...style }}
    className={className}
    {...props}
  >
    {children}
  </div>
)

export default GlassPanel
