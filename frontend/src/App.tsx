import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { EngineeringGate } from './components/EngineeringGate'
import { ConsolePage } from './pages/ConsolePage'
import { DatasetsPage } from './pages/DatasetsPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { MechanismsPage } from './pages/MechanismsPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { OrdersPage } from './pages/OrdersPage'
import { PreferencesPage } from './pages/PreferencesPage'
import { ProductsPage } from './pages/ProductsPage'
import { RunsPage } from './pages/RunsPage'
import { SandboxPage } from './pages/SandboxPage'

export default function App() {
  return <Routes>
    <Route element={<Layout />}>
      <Route element={<EngineeringGate />}>
        <Route index element={<ConsolePage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:runId" element={<RunsPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="mechanisms" element={<MechanismsPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="sandbox/tasks" element={<RunsPage businessMode />} />
        <Route path="sandbox/tasks/:runId" element={<RunsPage businessMode />} />
      </Route>
      <Route path="sandbox" element={<SandboxPage />} />
      <Route path="sandbox/products" element={<ProductsPage />} />
      <Route path="sandbox/orders" element={<OrdersPage />} />
      <Route path="sandbox/preferences" element={<PreferencesPage />} />
      <Route path="console" element={<Navigate to="/" replace />} />
      <Route path="*" element={<NotFoundPage />} />
    </Route>
  </Routes>
}
