import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { RunDetail, RunSummary } from '../types'
import { RunsPage } from './RunsPage'

const summary: RunSummary = {
  run_id: 'run-liveplan', conversation_id: 'conversation-1', request: '推荐预算 500 元的耳机',
  created_at: '2026-10-20T10:00:00Z', status: 'WAITING_CLARIFICATION', agent: 'adaptive',
  terminal_reason: 'MODEL_STOPPED', profile: { profile_id: 'adaptive-liveplan-v1' }, event_count: 5,
  tool_calls: 2, model_calls: 3, latency_ms: 100, tokens: 80,
}

const detail: RunDetail = {
  ...summary,
  result: {
    reply: '需要补充佩戴偏好。',
    events: [
      { type: 'trajectory_monitor_evaluated', sequence: 1 },
      { type: 'trajectory_drift_detected', sequence: 2, signals: [{
        code: 'ACTION_OSCILLATION', evidence_sequences: [1, 2], detail: '相同查询没有增加权威证据。',
      }] },
      { type: 'corrective_advice_generated', sequence: 3, source: 'predefined', advice: {
        reason_code: 'ACTION_OSCILLATION', guidance: '改用不同检索条件。',
      } },
      { type: 'intervention_applied', sequence: 4, outcome: 'ACTIVATED', mode: 'CONSTRAIN_NEXT_ACTION' },
      { type: 'intervention_applied', sequence: 5, outcome: 'ACTION_REJECTED', mode: 'CONSTRAIN_NEXT_ACTION' },
    ],
    trace: [],
  },
  checkpoint: { adaptive: { liveplan: {
    advisor_calls: 0,
    last_advice: {
      reason_code: 'ACTION_OSCILLATION', guidance: '改用不同检索条件。', source: 'predefined',
      recommended_action: 'CHANGE_APPROACH', confidence: 1,
    },
  } } },
  tool_ledger: [], commands: [],
  trace_projection: { integrity: {}, summary: {}, timeline: [] },
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/runs/run-liveplan']}>
        <Routes><Route path="/runs/:runId" element={<RunsPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => vi.restoreAllMocks())

describe('RunsPage LivePlan evidence', () => {
  it('shows persisted monitor evidence, advice and precommit rejection', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(detail)

    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '在线监控与纠偏' }))

    expect(screen.getByText('相同查询没有增加权威证据。')).toBeInTheDocument()
    expect(screen.getAllByText('ACTION_OSCILLATION').length).toBeGreaterThan(0)
    expect(screen.getAllByText('改用不同检索条件。').length).toBeGreaterThan(0)
    expect(screen.getByText('候选动作与已确认的无进展动作完全相同，已在执行前阻止。')).toBeInTheDocument()
  })
})

describe('RunsPage PG evidence', () => {
  it('distinguishes generated guidance, exposure, budget skip and committed action', async () => {
    const pgDetail: RunDetail = { ...detail, checkpoint: { context: { procedural_graph: {
      graph_id: 'lingcart-pg', version: 1, sha256: 'frozen-graph',
    } } }, result: { trace: [], events: [
      { type: 'model_started', sequence: 1, purpose: 'pg_guidance' },
      { type: 'pg_guidance', sequence: 2, decision_key: 'first', graph_sha256: 'frozen-graph',
        anchor: 'search_products', status: 'READY', guidance: '空结果后扩大检索范围', edge_ids: ['search-retry'], fallback: false },
      { type: 'evidence_context', sequence: 3, pg_decision_key: 'first', pg_status: 'EXPOSED' },
      { type: 'adaptive_decision', sequence: 4, decision: { action: 'tool', tool: 'search_products' } },
      { type: 'pg_guidance', sequence: 5, decision_key: 'second', anchor: 'unknown_action', fallback: true,
        status: 'SKIPPED', guidance: '', edge_ids: [], reason: 'RESERVED_DECISION_AND_AUDIT' },
    ] } }
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(pgDetail)
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('已送入决策')).toBeInTheDocument()
    expect(screen.getByText('空结果后扩大检索范围')).toBeInTheDocument()
    expect(screen.getByText('为决策和独立核验保留预算')).toBeInTheDocument()
    expect(screen.getByText(/引用关系：search-retry · 后续行动：search_products/)).toBeInTheDocument()
    expect(screen.getByText(/全图回退/)).toBeInTheDocument()
  })

  it('shows an empty state when no graph has been bound', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(detail)
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('该运行尚无程序图指导记录。')).toBeInTheDocument()
  })

  it('does not imply graph exposure when guidance was omitted by context budget', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue({ ...detail, result: { trace: [], events: [
      { type: 'pg_guidance', sequence: 1, decision_key: 'omitted', status: 'READY', guidance: '建议内容', edge_ids: ['edge-a'] },
      { type: 'evidence_context', sequence: 2, pg_decision_key: 'omitted', pg_status: 'CONTEXT_OMITTED' },
    ] } })
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('上下文不足，未送入决策')).toBeInTheDocument()
    expect(screen.queryByText('已送入决策')).not.toBeInTheDocument()
  })
})
