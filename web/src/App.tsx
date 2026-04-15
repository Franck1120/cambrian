import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Evolve from './pages/Evolve'
import Forge from './pages/Forge'
import Dashboard from './pages/Dashboard'

const App = () => (
  <BrowserRouter>
    <Navbar />
    <main style={{ flex: 1 }}>
      <Routes>
        <Route path="/" element={<Evolve />} />
        <Route path="/forge" element={<Forge />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </main>
  </BrowserRouter>
)

export default App
