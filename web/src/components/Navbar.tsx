import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Evolve' },
  { to: '/forge', label: 'Forge' },
  { to: '/dashboard', label: 'Dashboard' },
]

const Navbar = () => (
  <nav style={{ borderBottom: '1px solid #1a1a1a', background: '#0a0a0a' }}
    className="flex items-center gap-8 px-8 py-4 sticky top-0 z-50">
    <div className="flex items-center gap-2 mr-4">
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981',
        boxShadow: '0 0 8px #10b981' }} />
      <span style={{ color: '#f0f0f0', fontWeight: 600, fontSize: 15, letterSpacing: '-0.3px' }}>
        cambrian
      </span>
    </div>
    {links.map(({ to, label }) => (
      <NavLink key={to} to={to} end={to === '/'}
        style={({ isActive }) => ({
          color: isActive ? '#10b981' : '#6b7280',
          fontSize: 14,
          fontWeight: 500,
          textDecoration: 'none',
          transition: 'color 0.15s',
        })}>
        {label}
      </NavLink>
    ))}
    <div className="ml-auto flex items-center gap-3">
      <div style={{ background: '#0d2b1f', border: '1px solid #10b98133',
        borderRadius: 6, padding: '4px 10px', fontSize: 12, color: '#10b981', fontWeight: 500 }}>
        v1.0.4
      </div>
    </div>
  </nav>
)

export default Navbar
