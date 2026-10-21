import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as clientModule from '../api/client'
import { api, ApiError } from '../api/client'
import { saveRunOperation } from '../state/runOperation'
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
})
