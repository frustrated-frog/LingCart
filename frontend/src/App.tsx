import { Navigate, Route, Routes } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { Layout } from './components/Layout'
import { EngineeringGate } from './components/EngineeringGate'
import { ConsolePage } from './pages/ConsolePage'
import { DatasetsPage } from './pages/DatasetsPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { LoadingState } from './components/SystemUI'
import { MechanismsPage } from './pages/MechanismsPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { OrdersPage } from './pages/OrdersPage'
import { PreferencesPage } from './pages/PreferencesPage'
import { ProductsPage } from './pages/ProductsPage'
import { RunsPage } from './pages/RunsPage'
import { SandboxPage } from './pages/SandboxPage'

const RSILabPage = lazy(() => import('./pages/RSILabPage').then(module => ({ default: module.RSILabPage })))
const RecoveryLabPage = lazy(() => import('./pages/RecoveryLabPage').then(module => ({ default: module.RecoveryLabPage })))

const PGLabPage = lazy(() => import('./pages/PGLabPage').then(module => ({ default: module.PGLabPage })))

export default function App() {
  return <Routes>
    <Route element={<Layout />}>
      <Route element={<EngineeringGate />}>
        <Route index element={<ConsolePage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:runId" element={<RunsPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/rsi" element={<Suspense fallback={<LoadingState label="正在打开自进化对比" />}><RSILabPage /></Suspense>} />
        <Route path="experiments/recovery" element={<Suspense fallback={<LoadingState label="正在打开恢复搜索对比" />}><RecoveryLabPage /></Suspense>} />
        <Route path="experiments/pg" element={<Suspense fallback={<LoadingState label="正在打开 PG 实验台" />}><PGLabPage /></Suspense>} />
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
