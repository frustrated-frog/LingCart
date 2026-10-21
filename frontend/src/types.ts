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
  status: 'CURRENT_REPRODUCIBLE' | 'CURRENT_REJECTED' | 'HISTORICAL_PASSED' | 'HISTORICAL_REJECTED' | 'AWAITING_REPORT'
  report_path: string | null
  report_mismatches: string[]
  report?: { metrics?: Record<string, unknown>, gate?: { passed: boolean, failures: string[] } } | null
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

export interface RunFailure {
  code: string
  message: string
  retryable: boolean
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
  artifacts?: ResultArtifact[]
  error?: RunFailure | null
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

export interface CatalogFieldRef {
  core?: 'product_id' | 'name' | 'category' | 'price' | 'stock' | 'summary' | null
  concept_id?: string | null
}

export interface CatalogResultProduct extends Product {
  values: Record<string, unknown>
}

export interface CatalogResultArtifact {
  artifact_type: 'catalog_query_result'
  version: string
  result_set_id: string
  source_result_set_id?: string | null
  source_result_set_view?: 'DELIVERED' | 'POPULATION' | null
  catalog_snapshot: { version?: string; fingerprint?: string; source?: string; [key: string]: unknown }
  catalog_data_fingerprint: string
  population_data_fingerprint?: string
  result_data_fingerprint: string
  query_spec: {
    partition?: { field: CatalogFieldRef } | null
    ordering: Array<{ field: CatalogFieldRef; direction: 'ASC' | 'DESC' }>
    selector: { mode: 'ALL' | 'TOP_K' | 'PAGE' | 'ARGMIN' | 'ARGMAX' }
    metrics?: Array<{ alias: string; function: 'COUNT' | 'MIN' | 'MAX' | 'AVG' | 'SUM'; field?: CatalogFieldRef | null }>
    projection: { fields: CatalogFieldRef[] }
    [key: string]: unknown
  }
  matched_count: number
  delivered_count: number
  population_ids: string[]
  population_products?: Product[]
  selected_ids?: string[]
  delivered_ids?: string[]
  member_ids: string[]
  products: CatalogResultProduct[]
  groups: Array<{ key: string; product_ids: string[]; count: number }>
  metrics: Record<string, unknown>
  delivery: {
    mode: string
    requested_complete: boolean
    complete: boolean
    selection_complete?: boolean
    population_complete?: boolean
    has_more?: boolean
    page?: number | null
    page_size?: number | null
    next_page?: number | null
    matched_count: number
    delivered_count: number
  }
  verification?: {
    status?: 'PENDING' | 'VERIFIED' | 'REJECTED' | 'PARTIAL'
    goal?: { verdict?: string; checked_requirements?: string[] }
    delivery?: { verdict?: string; complete?: boolean }
    error?: { code?: string; message?: string; retryable?: boolean }
  }
  data_source: string
}

export interface ConstraintCandidate {
  candidate_id: string
  label: string
  attributes: Record<string, unknown>
  evidence: Partial<Product> & Record<string, unknown>
}

export interface ConstraintSelection {
  goal_id: string
  source_quote: string
  quantity: number
  candidates: ConstraintCandidate[]
}

export interface ConstraintCheck {
  requirement_id: string
  kind: 'AGGREGATE' | 'COMPARISON' | 'ALL_DIFFERENT'
  observed?: unknown
  operator?: string
  expected?: unknown
  left?: unknown
  right?: unknown
  attribute?: string
  satisfied: boolean
}

export interface ConstraintSolution {
  solution_id: string
  selections: ConstraintSelection[]
  constraint_checks: ConstraintCheck[]
  optimization_value?: unknown
}

export interface ConstraintPlanResultArtifact {
  artifact_type: 'constraint_plan_result'
  version: 'constraint-plan/v1' | string
  solution_set_id: string
  plan_spec: {
    goals: Array<{ goal_id: string; source_quote: string; quantity: number }>
    constraints: Array<{ requirement_id: string; source_quote: string; kind: string; [key: string]: unknown }>
    optimization?: { requirement_id: string; direction: 'MIN' | 'MAX'; [key: string]: unknown } | null
    [key: string]: unknown
  }
  compilation_coverage: Array<{ requirement_id: string; source_quote: string; status: string }>
  provider_registry_version: string
  domains: Array<{
    goal_id: string
    provider: string
    provider_fingerprint: string
    candidate_count: number
    evidence: Record<string, unknown>
  }>
  feasibility: 'FEASIBLE' | 'INFEASIBLE'
  solutions: ConstraintSolution[]
  search: {
    complete: boolean
    nodes_examined: number
    feasible_count: number
    feasible_count_exact: boolean
    stored_solution_limit_reached: boolean
    max_search_nodes: number
  }
  selection_policy: 'EXPLICIT_OPTIMIZATION' | 'STABLE_FARTHEST_FIRST' | string
  verification?: {
    status?: 'PENDING' | 'VERIFIED' | 'REJECTED'
    verdict?: 'SATISFIED' | 'PROVEN_INFEASIBLE' | string
    checked_requirements?: string[]
    global_goal?: boolean
    error?: { code?: string; message?: string; retryable?: boolean }
  }
}

export type ResultArtifact = CatalogResultArtifact | ConstraintPlanResultArtifact | { artifact_type: string; [key: string]: unknown }

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

export interface ExecutionResult extends RunResult {
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
}

export interface ChatResponse extends ExecutionResult {
  session_id: string
  cached: boolean
}

export interface Conversation {
  state: { pending_action?: PendingAction | null; [key: string]: unknown }
  messages: Array<{ role: 'user' | 'assistant'; content: string; run_id: string; result?: ExecutionResult | null }>
  active_run_id: string | null
}

export interface ProductRun {
  run_id: string
  session_id: string
  request: string
  status: string
  created_at: string
  result: ExecutionResult | null
  checkpoint: Record<string, unknown>
  events: RunEvent[]
}

export interface RunHandle {
  run_id: string
  session_id: string
  status: string
  cached: boolean
  result?: ExecutionResult
}

export interface RunCommand {
  command_id: string
  run_id: string
  kind: 'STEER' | 'CANCEL' | 'FOLLOW_UP'
  payload: Record<string, unknown>
  status: string
  outcome: Record<string, unknown> | null
}

export interface Preference {
  key: string
  value: { value: unknown }
  version: number
  updated_at: string
}
