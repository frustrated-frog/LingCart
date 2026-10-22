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
  it('shows an execution acknowledgement before the server accepts the run', async () => {
    vi.spyOn(api, 'conversation').mockResolvedValue({ active_run_id: null, messages: [], state: {} })
    vi.spyOn(api, 'startRun').mockImplementation(() => new Promise(() => undefined))
    renderPage()

    fireEvent.change(await screen.findByLabelText('消费需求'), { target: { value: '帮我查询订单' } })
    fireEvent.click(screen.getByLabelText('发送需求'))

    expect(await screen.findByText('正在提交可恢复任务')).toBeInTheDocument()
    expect(screen.getByText(/执行进度会持续同步/)).toBeInTheDocument()
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
