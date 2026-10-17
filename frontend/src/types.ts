export type ResearchStatus = 'baseline' | 'candidate' | 'promoted'

export interface MechanismRef {
  slot: string
  id: string
  version: string
}

export interface ResolvedMechanism extends MechanismRef {
  implementation_sha256?: string
  source_files?: string[]
  [key: string]: unknown
}

export interface HarnessProfile {
  profile_id: string
  version: string
  display_name: string
  description: string
  engine: { id: string; version: string }
  mechanisms: Record<string, MechanismRef>
  capabilities: string[]
  research: {
    status: ResearchStatus
    source_type: 'native' | 'paper' | 'hybrid'
    paper_id: string | null
    hypothesis: string | null
  }
  fingerprint: string
  execution_sha256: string
  source_revision: string | null
  resolved_mechanisms: Record<string, ResolvedMechanism>
  product_eligible: boolean
}

export interface DatasetSummary {
  dataset_id: string
  display_name: string
  path: string
  suite: string
  execution_source: string
  description: string
  case_count: number
  turn_count: number
  categories: Record<string, number>
  splits: Record<string, number>
}

export interface DatasetCase {
  id: string
  category?: string
  language?: string
  description?: string
  split?: string
  turns: Array<{
    message: string
    expect?: {
      statuses?: string[]
      tools?: Array<{ name: string; args?: Record<string, unknown>; result?: unknown }>
      forbidden_tools?: string[]
      [key: string]: unknown
    }
  }>
  adaptive_expect?: Record<string, unknown>
  [key: string]: unknown
}

export interface DatasetDetail extends DatasetSummary {
  cases: DatasetCase[]
}

export interface MetricRule {
  name: string
  objective: 'maximize' | 'minimize'
  max_regression: number
  min_improvement: number
}

export interface Experiment {
  experiment_id: string
  baseline_profile: string
  candidate_profile: string
  dataset: string
  suite: string
  repeats: number
  tuning_split: string
  evaluation_split: string
  changes: Array<{
    kind: string
    slot?: string
    implementation_id?: string
    paths: string[]
  }>
  primary_metrics: MetricRule[]
  guard_metrics: MetricRule[]
  compatibility_review: string | null
  path: string
  status: 'COMPLETED' | 'AWAITING_REPORT'
  report_path: string | null
  report?: { metrics?: Record<string, unknown> } | null
}

export interface ProviderStatus {
  provider: string
  model: string
  slot: number
  custom_base_url: boolean
  health: string
  failure_count: number
  retry_after_seconds: number
}

export interface ModelStatus {
  ready: boolean
  configured: boolean
  live_verified: boolean
  message: string
  providers: ProviderStatus[]
}

export interface RunProfile {
  profile_id: string
  legacy_profile_reference?: boolean
  [key: string]: unknown
}

export interface RunSummary {
  run_id: string
  conversation_id: string
  request: string
  created_at: string
  status: string
  agent: string
  terminal_reason: string | null
  profile: RunProfile
  event_count: number
  tool_calls: number
  model_calls: number
  latency_ms: number | null
  tokens: number | null
}

export interface RunEvent {
  type: string
  sequence?: number
  timestamp?: string
  status?: string
  tool?: string
  purpose?: string
  [key: string]: unknown
}

export interface RunResult {
  run_id?: string
  reply?: string
  status?: string
  agent?: string
  route?: string
  terminal_reason?: string
  events?: RunEvent[]
  trace?: Array<{ agent?: string; tool?: string; args?: Record<string, unknown>; result?: unknown }>
  plan?: { steps?: Array<Record<string, unknown>>; verification?: Record<string, unknown> }
  budget?: Record<string, unknown>
  usage?: Record<string, number | null>
  pending_action?: PendingAction | null
  [key: string]: unknown
}

export interface RunDetail extends RunSummary {
  result: RunResult
  checkpoint: Record<string, unknown>
  tool_ledger: Array<Record<string, unknown>>
  commands: Array<Record<string, unknown>>
  trace_projection: {
    integrity: Record<string, unknown>
    summary: Record<string, unknown>
    timeline: Array<Record<string, unknown>>
  }
}

export interface EngineeringOverview {
  generated_at: string
  profiles: HarnessProfile[]
  datasets: DatasetSummary[]
  experiments: Experiment[]
  runs: RunSummary[]
  kernel: Array<{ boundary_id: string; label: string; paths: string[] }>
  model: ModelStatus
  counts: {
    profiles: number
    candidate_profiles: number
    dataset_cases: number
    dataset_turns: number
    runs: number
    profiled_runs: number
    events: number
    waiting_runs: number
  }
  status_counts: Record<string, number>
}

export interface Product {
  product_id: string
  name: string
  category: string
  price: number
  currency: string
  summary: string
  specs: Record<string, string | number>
  stock: number
  data_source: string
}

export interface Order {
  order_id: number
  item: string
  customer_name: string
  status: string
  order_date: string
  amount: number
  payment_status: string | null
  refund_status: string | null
  refund_reason: string | null
}

export interface PendingAction {
  action_id: string
  type: string
  order_id: number
  amount?: number
  [key: string]: unknown
}

export interface ChatResponse extends RunResult {
  session_id: string
  run_id: string
  reply: string
  agent: string
  route: string
  status: string
  terminal_reason: string
  events: RunEvent[]
  trace: Array<{ agent?: string; tool?: string; args?: Record<string, unknown>; result?: unknown }>
  pending_action?: PendingAction | null
  usage: { input_tokens: number | null; output_tokens: number | null; total_tokens: number | null }
  cached: boolean
}

export interface Conversation {
  state: Record<string, unknown>
  messages: Array<{ role: 'user' | 'assistant'; content: string; run_id: string }>
  active_run_id: string | null
}

export interface Preference {
  key: string
  value: { value: unknown }
  version: number
  updated_at: string
}
