import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import EvolutionLab from './pages/EvolutionLab'
import PluginHub from './pages/PluginHub'
import ModelSelector from './pages/ModelSelector'
import Dashboard from './pages/Dashboard'

const App = () => (
  <BrowserRouter>
    <Navbar />
    <main style={{ flex: 1 }}>
      <Routes>
        <Route path="/" element={<EvolutionLab />} />
        <Route path="/plugins" element={<PluginHub />} />
        <Route path="/models" element={<ModelSelector />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </main>
  </BrowserRouter>
)

export default App
