import type {
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
  RunCommand,
  RunDetail,
  RunEvent,
  RunHandle,
  RunSummary,
} from '../types'

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
  after = 0): AbortController {
  const controller = new AbortController()
  const path = `/runs/${encodeURIComponent(runId)}/events${query({ session_id: sessionId, after })}`
  void fetch(`${API_BASE}${path}`, { signal: controller.signal, headers: { Accept: 'text/event-stream' } })
    .then(async (response) => {
      if (!response.ok || !response.body) throw new ApiError(response.status, `事件流连接失败（${response.status}）`)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n')
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''
        for (const frame of frames) {
          const data = frame.split('\n').filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trimStart()).join('\n')
          if (data) onEvent(JSON.parse(data) as RunEvent)
        }
      }
    })
    .catch((error: unknown) => {
      if (!(error instanceof DOMException && error.name === 'AbortError')) console.warn(error)
    })
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
