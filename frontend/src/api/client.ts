import type {
  AfterSalesCaseDetail,
  AfterSalesCaseStatus,
  AfterSalesCaseSummary,
  ChatResponse,
  Conversation,
  DatasetDetail,
  DatasetSummary,
  EngineeringOverview,
  Experiment,
  Order,
  Preference,
  Product,
  ProductRun,
  PGCatalog,
  PGJob,
  PGJobSummary,
  PGPreset,
  PGSample,
  RunCommand,
  RunDetail,
  RunEvent,
  RunHandle,
  RunSummary,
} from '../types'
import type { StreamConnectionState } from '../state/runStream'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '')
  ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')
const ENGINEERING_KEY = 'lingcart:engineering-key'

export function getEngineeringKey(): string {
  return sessionStorage.getItem(ENGINEERING_KEY) ?? ''
}

export function setEngineeringKey(value: string): void {
  const key = value.trim()
  if (key) sessionStorage.setItem(ENGINEERING_KEY, key)
  else sessionStorage.removeItem(ENGINEERING_KEY)
}

export class ApiError extends Error {
  readonly status: number
  readonly code?: string

  constructor(status: number, message: string, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const engineeringKey = path.startsWith('/engineering/') || path === '/engineering'
    ? getEngineeringKey() : ''
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(engineeringKey ? { 'X-Engineering-Key': engineeringKey } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as {
      detail?: string | { message?: string; code?: string }
    }
    const detail = payload.detail
    const message = typeof detail === 'string' ? detail : detail?.message ?? `请求失败（${response.status}）`
    throw new ApiError(response.status, message, typeof detail === 'object' ? detail?.code : undefined)
  }
  return response.json() as Promise<T>
}

export function streamRunEvents(runId: string, sessionId: string, onEvent: (event: RunEvent) => void,
  after = 0, onState?: (state: StreamConnectionState) => void): AbortController {
  const controller = new AbortController()
  const pause = (milliseconds: number) => new Promise<void>((resolve) => {
    const timer = window.setTimeout(resolve, milliseconds)
    controller.signal.addEventListener('abort', () => {
      window.clearTimeout(timer)
      resolve()
    }, { once: true })
  })
  const aborted = (error: unknown) => error instanceof Error && error.name === 'AbortError'
  void (async () => {
    let cursor = after
    let retries = 0
    let terminal = false
    while (!controller.signal.aborted && !terminal) {
      onState?.(retries ? 'reconnecting' : 'connecting')
      const path = `/runs/${encodeURIComponent(runId)}/events${query({ session_id: sessionId, after: cursor })}`
      try {
        const response = await fetch(`${API_BASE}${path}`, {
          signal: controller.signal,
          headers: { Accept: 'text/event-stream', ...(cursor ? { 'Last-Event-ID': String(cursor) } : {}) },
        })
        if (!response.ok || !response.body) throw new ApiError(response.status, `事件流连接失败（${response.status}）`)
        onState?.('connected')
        retries = 0
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (!controller.signal.aborted) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n')
          const frames = buffer.split('\n\n')
          buffer = frames.pop() ?? ''
          for (const frame of frames) {
            const data = frame.split('\n').filter((line) => line.startsWith('data:'))
              .map((line) => line.slice(5).trimStart()).join('\n')
            if (!data) continue
            const event = JSON.parse(data) as RunEvent
            if (typeof event.sequence === 'number' && event.sequence <= cursor) continue
            if (typeof event.sequence === 'number') cursor = event.sequence
            onEvent(event)
            if (event.type === 'terminal') terminal = true
          }
        }
      } catch (error) {
        if (!controller.signal.aborted && !aborted(error)) console.warn(error)
      }
      if (!terminal && !controller.signal.aborted) {
        retries += 1
        onState?.('reconnecting')
        await pause(Math.min(2_000, 250 * 2 ** Math.min(retries - 1, 3)))
      }
    }
    onState?.('closed')
  })()
  return controller
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  })
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export const api = {
  pgCatalog: () => request<PGCatalog>('/engineering/pg/catalog'),
  pgJobs: () => request<{ jobs: PGJobSummary[] }>('/engineering/pg/jobs'),
  pgJob: (id: string) => request<PGJob>(`/engineering/pg/jobs/${encodeURIComponent(id)}`),
  pgStart: (payload: { request_id: string; preset: PGPreset; confirm_full_cost?: boolean; evolution_policy?: 'basic-v1' | 'credit-memory-v1' }) => request<PGJob>('/engineering/pg/jobs', {
    method: 'POST', body: JSON.stringify(payload),
  }),
  pgCancel: (id: string) => request<PGJob>(`/engineering/pg/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  pgSamples: (id: string) => request<{ samples: PGSample[] }>(`/engineering/pg/jobs/${encodeURIComponent(id)}/samples`),
  pgSample: (id: string, sample: PGSample) => request<import('../types').PGSampleDetail>(
    `/engineering/pg/jobs/${encodeURIComponent(id)}/samples/${encodeURIComponent(sample.stage)}/${encodeURIComponent(sample.case_id)}/${sample.repetition}`),
  overview: () => request<EngineeringOverview>('/engineering/overview'),
  runs: (filters: { status?: string; profile_id?: string; limit?: number } = {}) =>
    request<{ runs: RunSummary[] }>(`/engineering/runs${query(filters)}`),
  run: (runId: string) => request<RunDetail>(`/engineering/runs/${encodeURIComponent(runId)}`),
  datasets: () => request<{ category_labels: Record<string, string>; datasets: DatasetSummary[] }>('/engineering/datasets'),
  dataset: (datasetId: string) => request<{ category_labels: Record<string, string>; dataset: DatasetDetail }>(
    `/engineering/datasets/${encodeURIComponent(datasetId)}`,
  ),
  experiments: () => request<{ experiments: Experiment[] }>('/engineering/experiments'),
  products: (filters: { query?: string; category?: string; max_price?: number; available_only?: boolean }) =>
    request<{ products: Product[]; total: number }>(`/business/products${query(filters)}`),
  orders: (filters: { query?: string; status?: string }) =>
    request<{ orders: Order[] }>(`/business/orders${query(filters)}`),
  afterSalesCases: (filters: { order_id?: number; session_id?: string; status?: AfterSalesCaseStatus; limit?: number } = {}) =>
    request<{ cases: AfterSalesCaseSummary[] }>(`/business/after-sales/cases${query(filters)}`),
  afterSalesCase: (caseId: string) => request<AfterSalesCaseDetail>(
    `/business/after-sales/cases/${encodeURIComponent(caseId)}`,
  ),
  conversation: (sessionId: string) => request<Conversation>(`/conversations/${encodeURIComponent(sessionId)}`),
  chat: (payload: { message: string; session_id?: string; request_id?: string; action_id?: string; harness_profile?: string }) =>
    request<ChatResponse>('/chat', { method: 'POST', body: JSON.stringify(payload) }),
  startRun: (payload: { message: string; session_id: string; request_id: string; action_id?: string;
    harness_profile: string; execution_mode: 'auto' | 'adaptive' }) =>
    request<RunHandle>('/runs', { method: 'POST', body: JSON.stringify(payload) }),
  productRun: (runId: string, sessionId: string) => request<ProductRun>(
    `/runs/${encodeURIComponent(runId)}${query({ session_id: sessionId })}`,
  ),
  runCommands: (runId: string, sessionId: string) => request<{ revision: number; commands: RunCommand[] }>(
    `/runs/${encodeURIComponent(runId)}/commands${query({ session_id: sessionId })}`,
  ),
  controlRun: (runId: string, kind: 'steer' | 'cancel' | 'follow-up', payload: {
    session_id: string; message: string; command_id: string; expected_revision: number
  }) => request<RunCommand>(`/runs/${encodeURIComponent(runId)}/${kind}`, {
    method: 'POST', body: JSON.stringify(payload),
  }),
  resumeRun: (runId: string, sessionId: string) => request<RunHandle>(
    `/runs/${encodeURIComponent(runId)}/resume-async`,
    { method: 'POST', body: JSON.stringify({ session_id: sessionId }) },
  ),
  createPreferenceProfile: () => request<{ profile_id: string }>('/profiles', { method: 'POST' }),
  bindPreferenceProfile: (profileId: string, sessionId: string) => request<{ bound: boolean }>(
    `/profiles/${encodeURIComponent(profileId)}/bindings`,
    { method: 'POST', body: JSON.stringify({ session_id: sessionId }) },
  ),
  preferences: (profileId: string) => request<{ preferences: Preference[] }>(
    `/profiles/${encodeURIComponent(profileId)}/preferences`,
  ),
  savePreference: (profileId: string, payload: { key: string; value: unknown; confirmed: boolean }) =>
    request<{ preferences: Preference[] }>(`/profiles/${encodeURIComponent(profileId)}/preferences`, {
      method: 'PUT', body: JSON.stringify(payload),
    }),
  forgetPreference: (profileId: string, key: string) => request<{ forgotten: boolean }>(
    `/profiles/${encodeURIComponent(profileId)}/preferences/${encodeURIComponent(key)}`,
    { method: 'DELETE' },
  ),
}
