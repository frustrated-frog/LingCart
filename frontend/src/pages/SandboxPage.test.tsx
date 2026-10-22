import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as clientModule from '../api/client'
import { api, ApiError } from '../api/client'
import { saveRunOperation } from '../state/runOperation'
import type { RunEvent } from '../types'
import { SandboxPage } from './SandboxPage'

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, streamRunEvents: vi.fn(() => new AbortController()) }
})

const sessionId = 'browser-session'

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={['/sandbox']}><SandboxPage /></MemoryRouter></QueryClientProvider>)
}

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  localStorage.setItem('lingcart.session_id', sessionId)
  vi.clearAllMocks()
})

describe('SandboxPage durable session behavior', () => {
  it('uses the selected execution mode when Enter submits a task', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: null, messages: [], state: {} })
    const start = vi.spyOn(api, 'startRun').mockImplementation(() => new Promise(() => undefined))
    renderPage()

    const mode = await screen.findByRole('switch', { name: '自适应规划' })
    expect(mode).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(mode)
    expect(mode).toHaveAttribute('aria-checked', 'true')

    const input = screen.getByLabelText('消费需求')
    fireEvent.change(input, { target: { value: '400 元以内选耳机和键盘' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', shiftKey: true })
    expect(start).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    await waitFor(() => expect(start).toHaveBeenCalledTimes(1))
    expect(start.mock.calls[0][0]).toMatchObject({
      message: '400 元以内选耳机和键盘', harness_profile: 'adaptive-v1', execution_mode: 'adaptive',
    })
  })

  it('offers a retry when persisted conversation recovery fails', async () => {
    vi.spyOn(api, 'conversation').mockRejectedValue(new Error('本地 API 暂时不可用'))
    renderPage()

    expect(await screen.findByRole('alert')).toHaveTextContent('暂时无法恢复会话')
    expect(screen.getByText('本地 API 暂时不可用')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重新连接' })).toBeInTheDocument()
  })

  it('shows an execution acknowledgement before the server accepts the run', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: null, messages: [], state: {} })
    vi.spyOn(api, 'startRun').mockImplementation(() => new Promise(() => undefined))
    renderPage()

    fireEvent.change(await screen.findByLabelText('消费需求'), { target: { value: '帮我查询订单' } })
    fireEvent.click(screen.getByLabelText('发送需求'))

    expect(await screen.findByText('正在提交可恢复任务')).toBeInTheDocument()
    expect(screen.getByText(/执行进度会持续同步/)).toBeInTheDocument()
  })

  it('restores and renders a complete grouped catalog artifact', async () => {
    const products = [
      { product_id: 'P1', name: '基础耳机', category: '耳机', price: 99, currency: 'CNY', summary: '有线 · 轻量', specs: {}, stock: 3, data_source: 'DEMO_CATALOG', values: {} },
      { product_id: 'P2', name: '静音键盘', category: '键盘', price: 199, currency: 'CNY', summary: '静音轴 · 三模', specs: {}, stock: 5, data_source: 'DEMO_CATALOG', values: {} },
    ]
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      state: {},
      messages: [
        { role: 'user', content: '把商品按类目展示', run_id: 'catalog-run' },
        { role: 'assistant', content: '已完成结构化目录查询。', run_id: 'catalog-run', result: {
          run_id: 'catalog-run', reply: '已完成结构化目录查询。', agent: 'shopping', route: 'DIRECT', status: 'SUCCEEDED',
          terminal_reason: 'COMPLETED', events: [], trace: [{ tool: 'query_catalog' }], pending_action: null,
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          artifacts: [{
            artifact_type: 'catalog_query_result', version: 'catalog-query/v1', result_set_id: 'crs_test',
            source_result_set_id: null, catalog_snapshot: { version: 'snapshot-v1' }, catalog_data_fingerprint: 'catalog-data-v1', result_data_fingerprint: 'result-data-v1',
            query_spec: { partition: { field: { core: 'category' } }, ordering: [{ field: { core: 'price' }, direction: 'ASC' }], selector: { mode: 'ALL' }, projection: { fields: [{ core: 'name' }] } },
            matched_count: 2, delivered_count: 2, population_ids: ['P1', 'P2'], member_ids: ['P1', 'P2'], products,
            groups: [{ key: '耳机', product_ids: ['P1'], count: 1 }, { key: '键盘', product_ids: ['P2'], count: 1 }],
            metrics: {}, delivery: { mode: 'ALL', requested_complete: true, complete: true, selection_complete: true, population_complete: true, has_more: false, page: null, page_size: null, next_page: null, matched_count: 2, delivered_count: 2 },
            verification: { status: 'VERIFIED', goal: { verdict: 'SATISFIED' }, delivery: { verdict: 'SATISFIED', complete: true } }, data_source: 'DEMO_CATALOG',
          }],
        } },
      ],
    })

    renderPage()

    expect(await screen.findByLabelText('结构化商品目录结果')).toBeInTheDocument()
    expect(screen.getByText('目标与交付已核验')).toBeInTheDocument()
    expect(screen.getByText('耳机')).toBeInTheDocument()
    expect(screen.getByText('键盘')).toBeInTheDocument()
    expect(screen.getByText('基础耳机')).toBeInTheDocument()
    expect(screen.getByText('静音键盘')).toBeInTheDocument()
    expect(screen.getByText('价格升序')).toBeInTheDocument()
  })

  it('does not present a rejected catalog artifact as verified product output', async () => {
    const product = { product_id: 'P1', name: '未核验耳机', category: '耳机', price: 99, currency: 'CNY', summary: '诊断行', specs: {}, stock: 3, data_source: 'DEMO_CATALOG', values: {} }
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      state: {},
      messages: [
        { role: 'user', content: '给我目录结果', run_id: 'rejected-catalog-run' },
        { role: 'assistant', content: '目录核验失败。', run_id: 'rejected-catalog-run', result: {
          run_id: 'rejected-catalog-run', reply: '目录核验失败。', agent: 'shopping', route: 'DIRECT', status: 'FAILED',
          terminal_reason: 'CATALOG_GOAL_NOT_SATISFIED', events: [], trace: [], pending_action: null,
          usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
          artifacts: [{
            artifact_type: 'catalog_query_result', version: 'catalog-query/v1', result_set_id: 'crs_rejected',
            catalog_snapshot: { version: 'snapshot-v1' }, catalog_data_fingerprint: 'source', result_data_fingerprint: 'result',
            query_spec: { partition: null, ordering: [], selector: { mode: 'ALL' }, projection: { fields: [{ core: 'name' }] } },
            matched_count: 1, delivered_count: 1, population_ids: ['P1'], member_ids: ['P1'], products: [product],
            groups: [], metrics: {}, delivery: { mode: 'ALL', requested_complete: true, complete: false, matched_count: 1, delivered_count: 1 },
            verification: { status: 'REJECTED', error: { code: 'CATALOG_GOAL_NOT_SATISFIED', message: '目录结果没有满足全部语义目标' } },
            data_source: 'DEMO_CATALOG',
          }],
        } },
      ],
    })

    renderPage()

    expect(await screen.findByLabelText('目录结果未通过核验')).toBeInTheDocument()
    expect(screen.getByText('REJECTED')).toBeInTheDocument()
    expect(screen.getByText('CATALOG_GOAL_NOT_SATISFIED')).toBeInTheDocument()
    expect(screen.queryByText('目标与交付已核验')).not.toBeInTheDocument()
    expect(screen.queryByText('未核验耳机')).not.toBeInTheDocument()
  })

  it('renders verified global constraint evidence without implying preference knowledge', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      state: {},
      messages: [
        { role: 'user', content: '400 元买一个耳机和一个键盘', run_id: 'constraint-run' },
        { role: 'assistant', content: '已找到可行方案。', run_id: 'constraint-run', result: {
          run_id: 'constraint-run', reply: '已找到可行方案。', agent: 'planning', route: 'DIRECT', status: 'SUCCEEDED',
          terminal_reason: 'COMPLETED', events: [], trace: [{ tool: 'solve_constraint_plan' }], pending_action: null,
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          artifacts: [{
            artifact_type: 'constraint_plan_result', version: 'constraint-plan/v1', solution_set_id: 'cpr_verified',
            plan_spec: {
              goals: [
                { goal_id: 'headphones', source_quote: '一个耳机', quantity: 1 },
                { goal_id: 'keyboard', source_quote: '一个键盘', quantity: 1 },
              ],
              constraints: [{ requirement_id: 'shared_budget', source_quote: '400 元', kind: 'AGGREGATE' }],
              optimization: null,
            },
            compilation_coverage: [{ requirement_id: 'constraint:shared_budget', source_quote: '400 元', status: 'COMPILED' }],
            provider_registry_version: 'providers-v1',
            domains: [
              { goal_id: 'headphones', provider: 'CATALOG', provider_fingerprint: 'headphones-fp', candidate_count: 20, evidence: {} },
              { goal_id: 'keyboard', provider: 'CATALOG', provider_fingerprint: 'keyboard-fp', candidate_count: 20, evidence: {} },
            ],
            feasibility: 'FEASIBLE',
            solutions: [{
              solution_id: 'cps_one', optimization_value: null,
              selections: [
                { goal_id: 'headphones', source_quote: '一个耳机', quantity: 1, candidates: [{
                  candidate_id: 'P1', label: '基础耳机', attributes: { price: 99 },
                  evidence: { product_id: 'P1', name: '基础耳机', price: 99, currency: 'CNY', summary: '轻量有线耳机' },
                  resource_claims: [{ resource_id: 'catalog.product:P1', capacity: 8, units: 1 }],
                }] },
                { goal_id: 'keyboard', source_quote: '一个键盘', quantity: 1, candidates: [{
                  candidate_id: 'P2', label: '静音键盘', attributes: { price: 199 },
                  evidence: { product_id: 'P2', name: '静音键盘', price: 199, currency: 'CNY', summary: '静音轴键盘' },
                  resource_claims: [{ resource_id: 'catalog.product:P2', capacity: 4, units: 1 }],
                }] },
              ],
              constraint_checks: [{ requirement_id: 'shared_budget', kind: 'AGGREGATE', observed: 298,
                operator: 'LTE', expected: 400, satisfied: true }],
              resource_checks: [
                { resource_id: 'catalog.product:P1', capacity: 8, used: 1, remaining: 7, satisfied: true,
                  consumers: [{ goal_id: 'headphones', candidate_id: 'P1', units: 1 }] },
                { resource_id: 'catalog.product:P2', capacity: 4, used: 1, remaining: 3, satisfied: true,
                  consumers: [{ goal_id: 'keyboard', candidate_id: 'P2', units: 1 }] },
              ],
            }],
            search: { complete: true, nodes_examined: 400, feasible_count: 22, feasible_count_exact: true,
              retained_feasible_count: 22, stored_solution_limit_reached: false, max_search_nodes: 100000 },
            optimality: { status: 'NOT_REQUESTED', objective_value: null, proof: null },
            selection_policy: 'STABLE_FARTHEST_FIRST',
            verification: { status: 'VERIFIED', verdict: 'SATISFIED', global_goal: true,
              checked_requirements: ['constraint:shared_budget'] },
          }],
        } },
      ],
    })

    renderPage()

    expect(await screen.findByLabelText('已核验约束方案')).toBeInTheDocument()
    expect(screen.getByText('全部硬约束已核验')).toBeInTheDocument()
    expect(screen.getByText('基础耳机')).toBeInTheDocument()
    expect(screen.getByText('静音键盘')).toBeInTheDocument()
    expect(screen.getByText('298 ≤ 400')).toBeInTheDocument()
    expect(screen.getByText('占用 1 / 容量 8 · 剩余 7')).toBeInTheDocument()
    expect(screen.getByText('未请求')).toBeInTheDocument()
    expect(screen.getByText('确定性差异覆盖 · 非个人偏好排序')).toBeInTheDocument()
  })

  it('restores a pending approval from persisted conversation state', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      messages: [
        { role: 'user', content: '订单 1001 退款', run_id: 'run-1' },
        { role: 'assistant', content: '请确认退款', run_id: 'run-1' },
      ],
      state: { pending_action: { action_id: 'approval-1', type: 'INITIATE_REFUND', order_id: 1001 } },
    })

    renderPage()

    expect(await screen.findByText('确认订单 #1001 的退款操作')).toBeInTheDocument()
    expect(screen.getByText(/审批状态已持久化/)).toBeInTheDocument()
  })

  it('distinguishes a policy-approved case submission from a generic refund', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      messages: [
        { role: 'user', content: '订单 1001 的耳机有异响', run_id: 'run-case' },
        { role: 'assistant', content: '案件已通过政策核验，请确认是否提交。', run_id: 'run-case' },
      ],
      state: { pending_action: {
        action_id: 'approval-case-1', type: 'SUBMIT_AFTER_SALES_CASE', order_id: 1001,
        case_id: 'case-audit-001', policy_version_id: 'standard-after-sales:2026-01',
        policy_decision_id: 'decision-001',
      } },
    })

    renderPage()

    expect(await screen.findByText('确认订单 #1001 的售后退款提交操作')).toBeInTheDocument()
    expect(screen.getByText(/案件 case-audit-001 已通过政策核验/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '确认提交退款申请' })).toBeInTheDocument()
  })

  it('shows persisted machine-readable failure evidence', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({
      active_run_id: null,
      state: {},
      messages: [
        { role: 'user', content: '耳机里面最便宜的是哪款', run_id: 'failed-run' },
        { role: 'assistant', content: '任务执行失败，系统已保留明确的错误原因，请稍后重试。', run_id: 'failed-run', result: {
          run_id: 'failed-run', reply: '任务执行失败，系统已保留明确的错误原因，请稍后重试。', agent: 'runtime',
          route: 'CATALOG_AUDIT_FORMAT_ERROR', status: 'FAILED', terminal_reason: 'CATALOG_AUDIT_FORMAT_ERROR',
          events: [], trace: [], pending_action: null, usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          error: { code: 'CATALOG_AUDIT_FORMAT_ERROR', message: '目录需求独立审查响应无效：历史上下文未标记为不适用\n内部详情', retryable: false },
        } },
      ],
    })

    renderPage()

    expect(await screen.findByLabelText('运行失败详情')).toBeInTheDocument()
    expect(screen.getByText('目录需求审查协议无效')).toBeInTheDocument()
    expect(screen.getByText('CATALOG_AUDIT_FORMAT_ERROR')).toBeInTheDocument()
    expect(screen.getByText('目录需求独立审查响应无效：历史上下文未标记为不适用')).toBeInTheDocument()
    expect(screen.queryByText('内部详情')).not.toBeInTheDocument()
  })

  it('reuses one request id when the user retries an unacknowledged start', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: null, messages: [], state: {} })
    const start = vi.spyOn(api, 'startRun').mockRejectedValue(new ApiError(503, '暂时无法连接'))
    renderPage()
    const input = await screen.findByLabelText('消费需求')
    fireEvent.change(input, { target: { value: '查询订单 1002' } })
    fireEvent.click(screen.getByLabelText('发送需求'))
    expect(await screen.findByText('使用同一请求编号重试')).toBeInTheDocument()
    const first = start.mock.calls[0][0].request_id

    fireEvent.click(screen.getByText('使用同一请求编号重试'))
    await waitFor(() => expect(start).toHaveBeenCalledTimes(2))
    expect(start.mock.calls[1][0].request_id).toBe(first)
  })

  it('sends live steering with the persisted control revision', async () => {
    saveRunOperation(sessionId, {
      requestId: 'request-1', message: '推荐耳机', harnessProfile: 'adaptive-v1',
      executionMode: 'adaptive', runId: 'active-run',
    })
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: 'active-run', messages: [], state: {} })
    vi.spyOn(api, 'productRun').mockResolvedValue({
      run_id: 'active-run', session_id: sessionId, request: '推荐耳机', status: 'RUNNING',
      created_at: new Date().toISOString(), result: null, checkpoint: {},
      events: [{ type: 'intake', sequence: 1 }],
    })
    vi.spyOn(api, 'runCommands').mockResolvedValue({ revision: 3, commands: [] })
    const control = vi.spyOn(api, 'controlRun').mockResolvedValue({
      command_id: 'command-1', run_id: 'active-run', kind: 'STEER', payload: {}, status: 'QUEUED', outcome: null,
    })
    renderPage()
    const input = await screen.findByLabelText('运行时指令')
    await waitFor(() => expect(api.runCommands).toHaveBeenCalled())
    fireEvent.change(input, { target: { value: '预算改为 300 元' } })
    fireEvent.click(screen.getByText('调整当前目标'))

    await waitFor(() => expect(control).toHaveBeenCalled())
    expect(control.mock.calls[0][0]).toBe('active-run')
    expect(control.mock.calls[0][1]).toBe('steer')
    expect(control.mock.calls[0][2]).toMatchObject({ message: '预算改为 300 元', expected_revision: 3 })
    expect(clientModule.streamRunEvents).toHaveBeenCalled()
  })

  it('renders persisted response deltas and the latest public agent status', async () => {
    saveRunOperation(sessionId, {
      requestId: 'request-stream', message: '推荐耳机', harnessProfile: 'adaptive-v1',
      executionMode: 'adaptive', runId: 'stream-run',
    })
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: 'stream-run', messages: [], state: {} })
    vi.spyOn(api, 'productRun').mockResolvedValue({
      run_id: 'stream-run', session_id: sessionId, request: '推荐耳机', status: 'RUNNING',
      created_at: new Date().toISOString(), result: null, checkpoint: {}, events: [],
    })
    vi.spyOn(api, 'runCommands').mockResolvedValue({ revision: 0, commands: [] })
    let emit: ((event: RunEvent) => void) | undefined
    vi.mocked(clientModule.streamRunEvents).mockImplementation((_runId, _sessionId, onEvent) => {
      emit = onEvent
      return new AbortController()
    })
    renderPage()
    await waitFor(() => expect(emit).toBeDefined())

    act(() => {
      emit?.({ type: 'agent_status_changed', sequence: 1, status: 'RESPONDING', label: '正在组织回答' })
      emit?.({ type: 'response_started', sequence: 2, message_id: 'answer-1' })
      emit?.({ type: 'response_delta', sequence: 3, message_id: 'answer-1', chunk_index: 0, delta: '正在为你比较候选' })
    })

    expect((await screen.findAllByText('正在组织回答')).length).toBeGreaterThan(0)
    expect(screen.getByText('正在为你比较候选')).toBeInTheDocument()
  })
})
