export const COLORS = {
  cyan: '#00f0ff',
  magenta: '#ff00e5',
  green: '#00ff88',
  amber: '#ffaa00',
  bgDeep: '#0a0e27',
  textPrimary: '#e8eaf6',
  textSecondary: '#8892b0',
  textMuted: '#4a5080',
  elite: '#00f0ff',
  active: '#00ff88',
  dead: '#4a5080',
} as const

export const GLASS = {
  background: 'rgba(10, 14, 40, 0.65)',
  backdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(0, 240, 255, 0.10)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.04)',
  borderRadius: 16,
} as const

export const GLASS_MAGENTA = {
  ...GLASS,
  border: '1px solid rgba(255, 0, 229, 0.12)',
} as const

export const GLASS_DARK = {
  ...GLASS,
  background: 'rgba(5, 8, 25, 0.75)',
  border: '1px solid rgba(255,255,255,0.06)',
} as const

export const GLOW = {
  cyan: '0 0 16px rgba(0,240,255,0.5), 0 0 40px rgba(0,240,255,0.15)',
  magenta: '0 0 16px rgba(255,0,229,0.5), 0 0 40px rgba(255,0,229,0.15)',
  green: '0 0 16px rgba(0,255,136,0.5), 0 0 40px rgba(0,255,136,0.15)',
  amber: '0 0 16px rgba(255,170,0,0.5), 0 0 40px rgba(255,170,0,0.15)',
  subtle: '0 4px 24px rgba(0,0,0,0.6)',
  none: 'none',
} as const

export const STATUS_COLOR: Record<string, string> = {
  elite: COLORS.cyan,
  active: COLORS.green,
  dead: COLORS.textMuted,
}

export const STATUS_GLOW: Record<string, string> = {
  elite: GLOW.cyan,
  active: GLOW.green,
  dead: GLOW.none,
}

export const CATEGORY_COLOR: Record<string, string> = {
  mutation: COLORS.cyan,
  selection: COLORS.magenta,
  memory: COLORS.amber,
  diversity: COLORS.green,
  evaluation: '#a78bfa',
  crossover: '#fb923c',
}
