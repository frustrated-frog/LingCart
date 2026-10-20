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
