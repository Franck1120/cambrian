import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Dna, FlaskConical, LayoutDashboard, Cpu, Menu, X, Wifi, WifiOff } from 'lucide-react'
import { COLORS, GLASS } from '../lib/theme'

interface NavItem {
  to: string
  label: string
  icon: React.ReactNode
  end?: boolean
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Evolution Lab', icon: <Dna size={15} />, end: true },
  { to: '/plugins', label: 'Plugin Hub', icon: <FlaskConical size={15} /> },
  { to: '/models', label: 'Models', icon: <Cpu size={15} /> },
  { to: '/dashboard', label: 'Dashboard', icon: <LayoutDashboard size={15} /> },
]

interface NavbarProps {
  wsConnected?: boolean
}

const Navbar = ({ wsConnected = false }: NavbarProps) => {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <>
      <nav
        style={{
          ...GLASS,
          borderRadius: 0,
          borderLeft: 'none',
          borderRight: 'none',
          borderTop: 'none',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
          height: 56,
          gap: 8,
        }}
      >
        {/* Logo */}
        <NavLink to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 10, marginRight: 24 }}>
          <motion.div
            animate={{ boxShadow: ['0 0 8px rgba(0,240,255,0.4)', '0 0 16px rgba(0,240,255,0.8)', '0 0 8px rgba(0,240,255,0.4)'] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: COLORS.cyan,
            }}
          />
          <span
            style={{
              fontFamily: "'Space Grotesk', sans-serif",
              fontWeight: 700,
              fontSize: 16,
              letterSpacing: '-0.4px',
              background: `linear-gradient(135deg, ${COLORS.textPrimary}, ${COLORS.cyan})`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            cambrian
          </span>
        </NavLink>

        {/* Desktop nav links */}
        <div className="hidden md:flex" style={{ display: 'flex', alignItems: 'center', gap: 4, flex: 1 }}>
          {NAV_ITEMS.map(({ to, label, icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: 7,
                padding: '6px 14px',
                borderRadius: 8,
                textDecoration: 'none',
                fontSize: 13,
                fontWeight: 500,
                color: isActive ? COLORS.cyan : COLORS.textSecondary,
                background: isActive ? `${COLORS.cyan}14` : 'transparent',
                border: `1px solid ${isActive ? COLORS.cyan + '33' : 'transparent'}`,
                transition: 'all 0.15s',
              })}
            >
              {icon}
              {label}
            </NavLink>
          ))}
        </div>

        {/* Right side: status + version */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* WS status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 11,
              color: wsConnected ? COLORS.green : COLORS.textMuted,
              padding: '4px 10px',
              borderRadius: 6,
              background: wsConnected ? `${COLORS.green}14` : 'rgba(255,255,255,0.04)',
              border: `1px solid ${wsConnected ? COLORS.green + '33' : 'rgba(255,255,255,0.08)'}`,
            }}
          >
            {wsConnected ? <Wifi size={11} /> : <WifiOff size={11} />}
            {wsConnected ? 'live' : 'offline'}
          </div>

          {/* Version */}
          <div
            style={{
              fontSize: 11,
              color: COLORS.cyan,
              fontFamily: "'JetBrains Mono', monospace",
              padding: '3px 8px',
              borderRadius: 6,
              background: `${COLORS.cyan}10`,
              border: `1px solid ${COLORS.cyan}25`,
            }}
          >
            v1.0.4
          </div>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen((v) => !v)}
            style={{
              display: 'none',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: COLORS.textSecondary,
              padding: 4,
            }}
            className="mobile-menu-btn"
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            style={{
              ...GLASS,
              borderRadius: 0,
              position: 'fixed',
              top: 56,
              left: 0,
              right: 0,
              zIndex: 49,
              padding: '12px 16px',
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
            }}
          >
            {NAV_ITEMS.map(({ to, label, icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                onClick={() => setMobileOpen(false)}
                style={({ isActive }) => ({
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 14px',
                  borderRadius: 10,
                  textDecoration: 'none',
                  fontSize: 14,
                  fontWeight: 500,
                  color: isActive ? COLORS.cyan : COLORS.textSecondary,
                  background: isActive ? `${COLORS.cyan}14` : 'transparent',
                })}
              >
                {icon}
                {label}
              </NavLink>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Navbar
