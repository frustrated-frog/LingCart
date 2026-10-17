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
  RunDetail,
  RunSummary,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '')
  ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')

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
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
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
