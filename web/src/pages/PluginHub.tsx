import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Search, LayoutGrid, List } from 'lucide-react'
import GlassPanel from '../components/ui/GlassPanel'
import GlowButton from '../components/ui/GlowButton'
import PluginCard from '../components/ui/PluginCard'
import { COLORS, CATEGORY_COLOR } from '../lib/theme'
import { getPlugins, mergePluginState, togglePlugin } from '../lib/api'
import { PLUGINS } from '../data/plugins'
import type { Plugin, PluginCategory } from '../types'

type ViewMode = 'grid' | 'list'

const ALL_CATEGORIES: { id: PluginCategory | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'mutation', label: 'Mutation' },
  { id: 'selection', label: 'Selection' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'memory', label: 'Memory' },
  { id: 'diversity', label: 'Diversity' },
  { id: 'crossover', label: 'Crossover' },
]

const PluginHub = () => {
  const [plugins, setPlugins] = useState<Plugin[]>(PLUGINS)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState<PluginCategory | 'all'>('all')
  const [view, setView] = useState<ViewMode>('grid')

  // Fetch enabled state from API and merge with static data
  useEffect(() => {
    let cancelled = false
    getPlugins()
      .then((apiPlugins) => {
        if (!cancelled) setPlugins(mergePluginState(PLUGINS, apiPlugins))
      })
      .catch(() => {
        // API unavailable — keep static defaults
      })
    return () => { cancelled = true }
  }, [])

  const handleToggle = async (id: string) => {
    const plugin = plugins.find((p) => p.id === id)
    if (!plugin) return
    const nextEnabled = !plugin.enabled
    // Optimistic update
    setPlugins((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: nextEnabled } : p)))
    try {
      await togglePlugin(id, nextEnabled)
    } catch {
      // Revert on failure
      setPlugins((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: plugin.enabled } : p)))
    }
  }

  const handleEnableAll = async () => {
    setPlugins((prev) => prev.map((p) => ({ ...p, enabled: true })))
    await Promise.allSettled(plugins.map((p) => togglePlugin(p.id, true)))
  }

  const handleDisableAll = async () => {
    setPlugins((prev) => prev.map((p) => ({ ...p, enabled: false })))
    await Promise.allSettled(plugins.map((p) => togglePlugin(p.id, false)))
  }

  const filtered = useMemo(() => {
    return plugins.filter((p) => {
      const matchesCat = category === 'all' || p.category === category
      const q = search.toLowerCase()
      const matchesSearch = !q || p.name.toLowerCase().includes(q) || p.description.toLowerCase().includes(q) || p.tags.some((t) => t.includes(q))
      return matchesCat && matchesSearch
    })
  }, [plugins, search, category])

  const enabledCount = plugins.filter((p) => p.enabled).length
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = { all: plugins.length }
    for (const p of plugins) counts[p.category] = (counts[p.category] ?? 0) + 1
    return counts
  }, [plugins])

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1280, margin: '0 auto' }}>
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
          <div>
            <h1
              style={{
                fontSize: 28,
                fontWeight: 700,
                fontFamily: "'Space Grotesk', sans-serif",
                margin: '0 0 6px',
                letterSpacing: '-0.5px',
              }}
              className="text-gradient-bio"
            >
              Plugin Hub
            </h1>
            <p style={{ fontSize: 14, color: COLORS.textMuted, margin: 0 }}>
              {enabledCount} of {plugins.length} plugins active
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <GlowButton
              color="cyan"
              variant="outline"
              size="sm"
              onClick={() => void handleEnableAll()}
            >
              Enable All
            </GlowButton>
            <GlowButton
              color="magenta"
              variant="outline"
              size="sm"
              onClick={() => void handleDisableAll()}
            >
              Disable All
            </GlowButton>
          </div>
        </div>
      </motion.div>

      {/* Filters bar */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <GlassPanel style={{ padding: '14px 20px', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            {/* Search */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 200 }}>
              <Search size={14} color={COLORS.textMuted} />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search plugins…"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: COLORS.textPrimary,
                  fontSize: 13,
                  flex: 1,
                }}
              />
            </div>

            {/* Category pills */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {ALL_CATEGORIES.map(({ id, label }) => {
                const active = category === id
                const col = id === 'all' ? COLORS.textSecondary : CATEGORY_COLOR[id as PluginCategory] ?? COLORS.textMuted
                return (
                  <button
                    key={id}
                    onClick={() => setCategory(id as PluginCategory | 'all')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: 8,
                      fontSize: 12,
                      fontWeight: active ? 600 : 400,
                      cursor: 'pointer',
                      border: `1px solid ${active ? col + '55' : 'rgba(255,255,255,0.1)'}`,
                      background: active ? `${col}18` : 'transparent',
                      color: active ? col : COLORS.textMuted,
                      transition: 'all 0.15s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 5,
                    }}
                  >
                    {label}
                    <span
                      style={{
                        fontSize: 10,
                        background: active ? `${col}33` : 'rgba(255,255,255,0.08)',
                        borderRadius: 4,
                        padding: '1px 5px',
                        color: active ? col : COLORS.textMuted,
                      }}
                    >
                      {categoryCounts[id] ?? 0}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* View toggle */}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 3, gap: 2 }}>
              {(['grid', 'list'] as ViewMode[]).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  style={{
                    padding: '5px 8px',
                    borderRadius: 6,
                    border: 'none',
                    cursor: 'pointer',
                    background: view === v ? 'rgba(0,240,255,0.15)' : 'transparent',
                    color: view === v ? COLORS.cyan : COLORS.textMuted,
                    transition: 'all 0.15s',
                  }}
                >
                  {v === 'grid' ? <LayoutGrid size={14} /> : <List size={14} />}
                </button>
              ))}
            </div>
          </div>
        </GlassPanel>
      </motion.div>

      {/* Plugin grid / list */}
      {filtered.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 64, color: COLORS.textMuted }}>
          No plugins match your search
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
          style={{
            display: view === 'grid' ? 'grid' : 'flex',
            gridTemplateColumns: view === 'grid' ? 'repeat(auto-fill, minmax(280px, 1fr))' : undefined,
            flexDirection: view === 'list' ? 'column' : undefined,
            gap: 12,
          }}
        >
          {filtered.map((plugin, i) => (
            <motion.div
              key={plugin.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03, duration: 0.25 }}
            >
              {view === 'grid' ? (
                <PluginCard plugin={plugin} onToggle={(id) => void handleToggle(id)} />
              ) : (
                <ListPluginRow plugin={plugin} onToggle={(id) => void handleToggle(id)} />
              )}
            </motion.div>
          ))}
        </motion.div>
      )}

      <div style={{ height: 32 }} />
    </div>
  )
}

const ListPluginRow = ({ plugin, onToggle }: { plugin: Plugin; onToggle: (id: string) => void }) => {
  const catColor = CATEGORY_COLOR[plugin.category] ?? COLORS.textMuted

  return (
    <GlassPanel
      style={{
        padding: '12px 18px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        opacity: plugin.enabled ? 1 : 0.6,
        transition: 'opacity 0.2s',
      }}
    >
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: catColor, flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.textPrimary }}>{plugin.name}</span>
          <span style={{ fontSize: 10, color: catColor, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            {plugin.category}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 12, color: COLORS.textMuted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {plugin.description}
        </p>
      </div>
      {plugin.impact !== null && (
        <span style={{ fontSize: 12, color: COLORS.green, minWidth: 64, textAlign: 'right' }}>
          +{(plugin.impact * 100).toFixed(0)}%
        </span>
      )}
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
          transition: 'background 0.2s',
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
          }}
        />
      </button>
    </GlassPanel>
  )
}

export default PluginHub
