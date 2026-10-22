import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { AfterSalesCaseDetail, AfterSalesCaseSummary, Order } from '../types'
import { OrdersPage } from './OrdersPage'

const order: Order = {
  order_id: 1001,
  item: 'Wireless Headphones',
  customer_name: '演示客户',
  status: 'delivered',
  order_date: '2026-10-01',
  amount: 699,
  payment_status: 'paid',
  refund_status: 'requested',
  refund_reason: '间歇性异响',
}

const caseSummary: AfterSalesCaseSummary = {
  case_id: 'case-audit-001',
  case_type: 'REFUND',
  order_id: 1001,
  order_item_id: 'OI-1001-1',
  requested_resolution: 'REFUND',
  reason_text: '耳机出现间歇性异响',
  status: 'SUBMITTED',
  revision: 5,
  origin_session_id: 'session-audit',
  opened_at: '2026-10-22T09:00:00Z',
  updated_at: '2026-10-22T09:05:00Z',
  policy_outcome: 'ELIGIBLE',
  policy_reason_codes: ['POLICY_REQUIREMENTS_SATISFIED'],
  policy_version_id: 'standard-after-sales:2026-01',
  refund_id: 71,
}

const caseDetail: AfterSalesCaseDetail = {
  case: {
    case_id: caseSummary.case_id,
    request_key: 'after-sales:session-audit:req-1:refund:1001',
    case_type: 'REFUND',
    order_id: 1001,
    order_item_id: 'OI-1001-1',
    requested_resolution: 'REFUND',
    reason_code: null,
    reason_text: '耳机出现间歇性异响',
    status: 'SUBMITTED',
    revision: 5,
    origin_session_id: 'session-audit',
    context: { source: 'AGENT_RUNTIME' },
    opened_at: '2026-10-22T09:00:00Z',
    updated_at: '2026-10-22T09:05:00Z',
    closed_at: null,
    latest_decision: {
      decision_id: 'decision-001',
      policy_version_id: 'standard-after-sales:2026-01',
      case_revision: 4,
      input_hash: 'input-hash-001',
      outcome: 'ELIGIBLE',
      reason_codes: ['POLICY_REQUIREMENTS_SATISFIED'],
      allowed_actions: ['REQUEST_CUSTOMER_CONFIRMATION', 'ESCALATE_TO_HUMAN'],
      facts: { order_status: 'delivered', payment_status: 'paid' },
      evidence_refs: ['orders:1001', 'payments:11', 'shipment_events:21'],
      evaluator_version: 'after-sales-policy-v1',
      decided_at: '2026-10-22T09:03:00Z',
      supersedes_decision_id: null,
    },
    evidence: [{
      evidence_id: 'evidence-001', evidence_type: 'ISSUE_DESCRIPTION', source_type: 'CUSTOMER_STATEMENT',
      source_ref: 'agent_runs:run-intake:request', payload: { statement: '耳机出现间歇性异响' },
      checksum: 'evidence-checksum-001', status: 'ACTIVE', observed_at: '2026-10-22T09:01:00Z',
      created_at: '2026-10-22T09:01:00Z',
    }],
    events: [
      { event_id: 'event-1', sequence: 1, case_revision: 1, event_type: 'CASE_OPENED', actor_type: 'CUSTOMER', actor_id: 'session-audit', idempotency_key: 'open:req-1', payload: { requested_resolution: 'REFUND' }, occurred_at: '2026-10-22T09:00:00Z' },
      { event_id: 'event-2', sequence: 2, case_revision: 2, event_type: 'RUN_LINKED', actor_type: 'SYSTEM', actor_id: null, idempotency_key: 'run:run-intake', payload: { role: 'INTAKE' }, occurred_at: '2026-10-22T09:00:30Z' },
      { event_id: 'event-3', sequence: 3, case_revision: 4, event_type: 'POLICY_DECIDED', actor_type: 'SYSTEM', actor_id: 'after-sales-policy-v1', idempotency_key: 'policy:hash', payload: { outcome: 'ELIGIBLE' }, occurred_at: '2026-10-22T09:03:00Z' },
      { event_id: 'event-4', sequence: 4, case_revision: 5, event_type: 'CASE_SUBMITTED', actor_type: 'CUSTOMER', actor_id: 'session-audit', idempotency_key: 'submit:decision-001', payload: { refund_id: 71 }, occurred_at: '2026-10-22T09:05:00Z' },
    ],
  },
  runs: [
    { run_id: 'run-intake', session_id: 'session-audit', role: 'INTAKE', linked_at: '2026-10-22T09:00:30Z' },
    { run_id: 'run-submit', session_id: 'session-audit', role: 'SUBMISSION', linked_at: '2026-10-22T09:04:30Z' },
  ],
  refund: {
    refund_id: 71, status: 'requested', reason: '耳机出现间歇性异响', policy_decision_id: 'decision-001',
    submitted_run_id: 'run-submit', linked_at: '2026-10-22T09:05:00Z',
  },
  policy: {
    policy_version_id: 'standard-after-sales:2026-01', policy_id: 'standard-after-sales', version: '2026-01',
    status: 'ACTIVE', effective_from: '2026-01-01T00:00:00Z', effective_until: null,
    rules: { return_window_days: 30 }, source_ref: 'LOCAL_SEED_POLICY', checksum: 'policy-checksum-001',
  },
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={queryClient}><MemoryRouter><OrdersPage /></MemoryRouter></QueryClientProvider>)
}

beforeEach(() => vi.restoreAllMocks())

describe('OrdersPage after-sales case ledger', () => {
  it('renders persisted policy, evidence, run and side-effect records', async () => {
    vi.spyOn(api, 'orders').mockResolvedValue({ orders: [order] })
    vi.spyOn(api, 'afterSalesCases').mockResolvedValue({ cases: [caseSummary] })
    vi.spyOn(api, 'afterSalesCase').mockResolvedValue(caseDetail)

    renderPage()

    expect(await screen.findByText('订单事实满足当前政策')).toBeInTheDocument()
    expect(screen.getByText('退款申请已提交')).toBeInTheDocument()
    expect(screen.getByText('#71 · 退款申请中')).toBeInTheDocument()
    expect(screen.getByText('问题描述')).toBeInTheDocument()
    expect(screen.getByText('案件受理')).toBeInTheDocument()
    expect(screen.getByText('提交执行')).toBeInTheDocument()
    expect(screen.getByText('LOCAL_SEED_POLICY')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /建立售后案件/ })).toBeInTheDocument()
  })

  it('keeps order operations available when the case projection fails independently', async () => {
    vi.spyOn(api, 'orders').mockResolvedValue({ orders: [order] })
    vi.spyOn(api, 'afterSalesCases').mockRejectedValue(new Error('案件接口暂时不可用'))

    renderPage()

    expect(await screen.findByText('案件账本读取失败')).toBeInTheDocument()
    expect(await screen.findByText('无线耳机')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /状态与物流/ })).toBeInTheDocument()
  })
})
