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

export interface RequestSourceSpan {
  source: 'CURRENT_TURN' | 'HISTORY'
  history_index: number | null
  start: number
  end: number
  text: string
  source_sha256: string
}

export interface RequestSemanticRequirement {
  requirement_id: string
  kind: 'BUSINESS_OUTCOME' | 'USER_FACT' | 'PRODUCT_PREDICATE' | 'CROSS_GOAL_CONSTRAINT'
    | 'SOFT_PREFERENCE' | 'PROCESS_INSTRUCTION' | 'EXECUTION_BOUNDARY' | 'PROHIBITION' | 'UNKNOWN'
  description: string
  source: RequestSourceSpan
  objective_ids: string[]
  constraint_refs: string[]
  quantity?: number | null
}

export interface UserRequestIR {
  version: 'request-ir/v1' | 'request-ir/v2'
  requirements: RequestSemanticRequirement[]
  adaptive_goal_sha256?: string | null
  sha256: string
}

export type ExecutionRestriction = 'READ_ONLY' | 'NO_PAYMENT' | 'NO_REFUND' | 'NO_SUBMISSION'

export interface ExecutionBoundary {
  restriction: ExecutionRestriction
  source: RequestSourceSpan
  origin: 'adaptive' | 'dialogue'
  requirement_id?: string
}

export interface RequestSemantics {
  version: 'request-semantics/v1'
  sha256: string
  previous_sha256: string | null
  requirements: Array<{
    source: RequestSourceSpan
    requirement: {
      requirement_id: string
      kind: 'OUTCOME' | 'CONDITION' | 'PROHIBITION' | 'EXECUTION_BOUNDARY' | 'USER_FACT' | 'SOFT_PREFERENCE' | 'UNKNOWN'
      description: string
      source_quote: string
      targets: string[]
      outcome: { kind: string; reason_code?: string | null; quantity: number; order_id: number | null; question: string | null; product_ids: string[] } | null
      predicate: { field: string | null; op: string; value: unknown } | null
      restriction: ExecutionRestriction | null
    }
  }>
  revisions: Array<{
    operation: 'PRESERVE' | 'REPLACE' | 'RETRACT' | 'ADD'
    previous_id: string | null
    requirement_id: string | null
    source_quote: string | null
  }>
}

export interface SemanticRevision {
  preserved: string[]
  changed: string[]
  added: string[]
  retracted: string[]
  previous_sha256: string | null
  current_sha256: string
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
  request_ir?: UserRequestIR | null
  request_semantics?: RequestSemantics | null
  semantic_revision?: SemanticRevision | null
  execution_boundaries?: ExecutionBoundary[]
  budget?: Record<string, unknown>
  usage?: Record<string, number | null>
  pending_action?: PendingAction | null
  pending_clarification?: PendingClarification | null
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
  resource_claims?: Array<{ resource_id: string; capacity: number; units: number }>
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
  resource_checks: Array<{
    resource_id: string
    capacity: number
    used: number
    remaining: number
    satisfied: boolean
    consumers: Array<{ goal_id: string; candidate_id: string; units: number }>
  }>
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
    retained_feasible_count: number
    stored_solution_limit_reached: boolean
    max_search_nodes: number
  }
  optimality: {
    status: 'NOT_REQUESTED' | 'NO_FEASIBLE_SOLUTION' | 'BEST_KNOWN' | 'PROVEN_OPTIMAL'
    objective_value: unknown
    proof: 'EXHAUSTIVE_SEARCH' | 'BOUNDED_SEARCH' | null
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

export type AfterSalesCaseStatus = 'OPEN' | 'AWAITING_EVIDENCE' | 'AWAITING_CONFIRMATION' | 'SUBMITTED'
  | 'MANUAL_REVIEW' | 'RESOLVED' | 'REJECTED' | 'CANCELLED'

export type PolicyOutcome = 'ELIGIBLE' | 'NEEDS_EVIDENCE' | 'INELIGIBLE' | 'MANUAL_REVIEW'

export interface EvidenceRequirement {
  requirement_id: string
  policy_version_id: string
  policy_checksum: string
  target_case_id: string
  evidence_type: string
  description: string
  accepted_sources: string[]
  validation_condition: string
  max_age_days: number | null
  status: 'SATISFIED' | 'UNSATISFIED'
  evidence_refs: string[]
  rejected_evidence: Record<string, string>
  semantic_assessment?: {
    assessment_id: string
    verifier_version: string
    status: 'SUFFICIENT' | 'INCOMPLETE' | 'IRRELEVANT' | 'CONFLICT' | 'UNKNOWN'
    explanation: string
    question: string
    missing_information: string[]
    citations: { evidence_id: string; quote: string }[]
    error_code?: string | null
  } | null
}

export interface PolicyDecision {
  decision_id: string
  policy_version_id: string
  case_revision: number
  input_hash: string
  outcome: PolicyOutcome
  reason_codes: string[]
  allowed_actions: string[]
  facts: Record<string, unknown> & { evidence_requirements?: EvidenceRequirement[] }
  valid?: boolean
  stale_reason?: string
  evidence_refs: string[]
  evaluator_version: string
  decided_at: string
  supersedes_decision_id: string | null
}

export interface AfterSalesEvidence {
  evidence_id: string
  evidence_type: string
  source_type: string
  source_ref: string
  payload: Record<string, unknown>
  checksum: string
  status: string
  observed_at: string
  created_at: string
  validation?: {
    status: 'VERIFIED' | 'UNVERIFIED' | 'REJECTED'
    reason_code: string
    validator_version: string
    checksum: string
    case_revision: number
    validated_at: string
  } | null
}

export interface AfterSalesCaseEvent {
  event_id: string
  sequence: number
  case_revision: number
  event_type: string
  actor_type: string
  actor_id: string | null
  idempotency_key: string
  payload: Record<string, unknown>
  occurred_at: string
}

export interface AfterSalesCaseProjection {
  case_id: string
  request_key: string
  case_type: string
  order_id: number
  order_item_id: string | null
  requested_resolution: string
  reason_code: string | null
  reason_text: string
  status: AfterSalesCaseStatus
  revision: number
  origin_session_id: string | null
  context: Record<string, unknown>
  opened_at: string
  updated_at: string
  closed_at: string | null
  latest_decision: PolicyDecision | null
  evidence: AfterSalesEvidence[]
  events: AfterSalesCaseEvent[]
  plans?: CasePlan[]
}

export interface CasePlan {
  schema_version: number
  planner_version: string
  plan_id: string
  case_id: string
  revision: number
  case_revision: number
  decision_id: string
  policy_version_id: string
  status: string
  supersedes_plan_id: string | null
  goals: { goal_id: string; description: string; status: string; depends_on: string[]; evidence_refs: string[]; basis_hash: string; requirement_id?: string }[]
  next_actions: { kind: string; goal_id: string; operation?: string; requirement_id?: string; question?: string; requires_approval?: boolean }[]
  retained_goal_ids: string[]
  changed_goal_ids: string[]
  removed_goal_ids: string[]
  invalidated_decision_id: string | null
  created_at: string
  valid: boolean
}

export interface AfterSalesCaseSummary {
  case_id: string
  case_type: string
  order_id: number
  order_item_id: string | null
  requested_resolution: string
  reason_text: string
  status: AfterSalesCaseStatus
  revision: number
  origin_session_id: string | null
  opened_at: string
  updated_at: string
  policy_outcome: PolicyOutcome | null
  policy_reason_codes: string[]
  policy_version_id: string | null
  refund_id: number | null
}

export interface AfterSalesCaseDetail {
  case: AfterSalesCaseProjection
  runs: Array<{ run_id: string; session_id: string; role: string; linked_at: string }>
  refund: {
    refund_id: number
    status: string
    reason: string | null
    policy_decision_id: string
    submitted_run_id: string | null
    linked_at: string
  } | null
  policy: {
    policy_version_id: string
    policy_id: string
    version: string
    status: string
    effective_from: string
    effective_until: string | null
    rules: Record<string, unknown>
    source_ref: string
    checksum: string
  } | null
}

export interface PendingAction {
  action_id: string
  type: string
  order_id: number
  amount?: number
  case_id?: string
  policy_version_id?: string
  policy_decision_id?: string
  [key: string]: unknown
}

export interface PendingClarification {
  clarification_id: string
  status: 'OPEN' | 'ANSWERED' | 'SUPERSEDED' | 'CLOSED'
  case_id: string
  requirement_id: string
  question_id: string
  question: string
  case_revision: number
  policy_version_id: string
  policy_checksum: string
  policy_decision_id: string
  question_checksum: string
  binding_checksum: string
  authorization_granted?: false
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
  pending_clarification?: PendingClarification | null
  usage: { input_tokens: number | null; output_tokens: number | null; total_tokens: number | null }
}

export interface ChatResponse extends ExecutionResult {
  session_id: string
  cached: boolean
}

export interface Conversation {
  state: {
    pending_action?: PendingAction | null
    pending_clarification?: PendingClarification | null
    [key: string]: unknown
  }
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

export type PGPreset = 'smoke' | 'full'
export type PGArm = 'baseline' | 'seed' | 'evolved'
export interface PGGraph {
  graph_id: string; version: number; parent_sha256: string | null
  nodes: { id: string; kind: string; description: string }[]
  edges: PGEdge[]
}
export interface PGEdge {
  id: string; source: string; target: string; relation: string
  condition: string; guidance: string; pitfalls: string
  applicability?: { all_of?: PGConditionClause[]; any_of?: PGConditionClause[] } | null
}
export interface PGConditionClause { fact: string; operator: string; value?: string | boolean | number | string[] | null }
export interface PGTransitionPrediction {
  edge_id: string; source: string; target: string; action: string | null
  state_basis: 'CURRENT_VERIFIED' | 'PREDICTED_ACTION_ONLY'; condition_status: 'TRUE' | 'FALSE' | 'UNKNOWN'
  condition_reason: string; effect_hypotheses: string[]; reliability_score: number
  support_count: number; counter_count: number; utility_components: Record<string, number>; projected_state_sha256: string
}
export interface PGPreplayPath {
  path_id: string; edge_ids: string[]; node_ids: string[]; first_action: string | null; score: number; executable_prefix: 1
  future_edges_require_revalidation: string[]; predictions?: PGTransitionPrediction[]
}
export interface PGPreplay {
  plan_id: string; method: string; status: 'READY' | 'ABSTAINED' | 'TERMINAL'; reason: string
  causal_claim: false; probabilistic_claim?: false; replanned_from?: string | null
  selected_path: PGPreplayPath | null; paths?: PGPreplayPath[]
}
export interface PGPredictionOutcome {
  prediction_outcome_id?: string; preplay_plan_id?: string; status?: 'SUPPORT' | 'COUNTER' | 'DIVERGED' | 'UNKNOWN'
  action_match?: boolean; expected_action?: string | null; actual_action?: string | null; causal_claim?: boolean
  effects?: { effect?: string; status?: string; reason?: string }[]
}
export interface PGPlan {
  evolution_policy?: 'basic-v1' | 'credit-memory-v1'; evaluation_scope?: 'dev_only' | 'three_arm_test'
  preset: PGPreset; spec: { repeats: number; iterations: number; max_model_calls_per_run: number
    max_prompt_tokens_per_run?: number; max_completion_tokens_per_run?: number; max_seconds_per_run?: number }
  groups: Record<'train' | 'validation' | 'test', string[]>
  max_samples: number; max_model_attempts: number; max_token_units: number | null; dataset_sha256: string; seed_sha256: string
}
export interface PGCatalog {
  learning_plans?: Record<PGPreset, PGPlan>; learning_seed_graph?: PGGraph
  dataset: { cases: { id: string; learning_partition: 'train' | 'validation' | 'test'; scenario_family: string; category: string; description: string }[] }
  plans: Record<PGPreset, PGPlan>; seed_graph: PGGraph; model_configured: boolean; current_identity: Record<string, unknown>
}
export interface PGSample {
  stage: string; case_id: string; repetition: number; passed: boolean; failures: string[]
  interventions: number; failure_stage: string | null
}
export interface PGIntervention {
  intervention_id: string; graph: { graph_id: string; graph_version: number; graph_sha256: string }
  localization: { anchor: string | null; fallback: boolean; abstained?: boolean; algorithm?: string; status?: string
    reason?: string; confidence?: number | null; candidates?: Record<string, unknown>[]
    shadow?: { algorithm?: string; anchor?: string | null; fallback?: boolean; node_ids?: string[]; edge_ids?: string[] } | null
    shadow_changed?: boolean; node_ids: string[]; state_snapshot: Record<string, unknown>; sequence: number }
  retrieval: { retrieved_edge_ids: string[]; selected_edge_ids: string[]; excluded_edge_ids?: string[]; subgraph: { edges?: PGEdge[] } | null }
  applicability?: { strategy?: string | null; status?: string | null; reason?: string | null
    evaluations?: { edge_id?: string; source?: string; target?: string; status?: string; reason?: string; score?: number | null
      evaluation?: { evidence_bindings?: { fact?: string; actual?: unknown; source?: string }[] } }[]
    selected_edge_ids?: string[]; false_edge_ids?: string[]; unknown_edge_ids?: string[]
    projected_paths?: { path_id?: string; edge_ids?: string[]; currently_applicable_prefix?: number; future_edges_require_reevaluation?: string[] }[] }
  preplay?: PGPreplay | null
  guidance: { status: string; reason: string | null; content: string; exposed: boolean; sequence: number
    generation_mode?: string; generation_model_calls?: number | null
    contract?: { contract_id?: string; policy?: string; recommended_actions?: string[]; forbidden_repeats?: Record<string, unknown>[] } | null }
  decision: { sequence: number; source: string; payload: Record<string, unknown> } | null
  action_alignment: { status: string; matched_edge_ids: string[]; expected_actions?: string[]; actual_action?: string; reason?: string
    blocked?: boolean; review_id?: string }
  action_progress?: { status?: string; reason?: string; signals?: string[]; cost_delta?: { model_calls?: number; tool_calls?: number }
    schema_version?: number | null; execution_progress?: boolean | null; learning_eligible?: boolean | null
    information_gain?: { status: string; changed_observation_scopes?: string[]; new_evidence_count?: number; newly_verified_objective_ids?: string[] } | null
    outcome_id?: string; review_id?: string; matched_edge_ids?: string[]; before_sha256?: string; after_sha256?: string } | null
  prediction_outcome?: PGPredictionOutcome | null
  tool: { name: string; args: Record<string, unknown>; status: string; sequence: number; result: unknown; invalid_result: boolean } | null
  verification: Record<string, unknown> | null
}
export interface PGSampleDetail {
  turns?: { message?: string; result?: { events?: RunEvent[] } }[]
  case_id: string; repetition: number; passed: boolean; [key: string]: unknown
  pg_evidence?: { schema_version: number; summary: { interventions: number; exposed: number; aligned: number; diverged: number
    unknown_alignment: number; blocked_repeats?: number; deterministic_contracts?: number; guidance_model_calls?: number
    localization_fallbacks: number; localization_abstentions?: number
    shadow_differences?: number; shadow_fallbacks?: number
    progressed_actions?: number; no_progress_actions?: number; failed_actions?: number; deferred_actions?: number
    unknown_progress_actions?: number; typed_applicability_interventions?: number; preplay_interventions?: number
    prediction_supported?: number; prediction_countered?: number; prediction_diverged?: number; prediction_unknown?: number
    environment_triggered: boolean; recovered: boolean; failure_stage: string | null }
    turns: { turn_index: number; request: string; passed: boolean; failures: string[]; interventions: PGIntervention[] }[] }
}
export interface PGEdgeCredit {
  edge_id: string; edge_sha256: string; classification: string; support: number; counter: number; unknown: number; case_count: number
  evidence: { evidence_id: string; sample_ref: string; case_id: string; repetition: number; turn_index: number
    classification: string; reason: string; condition: string; exposed: boolean; adopted: boolean; information_gain: string
    task_passed: boolean; later_verified_objective_ids: string[]; guidance_sequence: number }[]
  contrasts: { case_id: string; successful: { samples: string[]; counts: Record<string, number> }; failed: { samples: string[]; counts: Record<string, number> } }[]
}
export interface PGHypothesis {
  problem: string; failure_mode: string; expected_change: string; target_edges: string[]; source_samples: string[]; evidence_ids: string[]
}
export interface PGHypothesisMemory {
  hypothesis_id: string; hypothesis: PGHypothesis; iteration: number; status: string; reason: string; blocks_retry: boolean
  validation_summary: { regressed_samples: number; improved_samples: number; compared_samples: number }
}
export interface PGSelection {
  evolution_policy?: string; training_graph?: PGGraph
  edge_credit?: { graph_sha256: string; graph_id: string; graph_version: number; edges: PGEdgeCredit[]; causal_claim: false; sample_refs: string[] }
  hypothesis?: PGHypothesis; hypothesis_id?: string; hypothesis_memory?: PGHypothesisMemory[]; memory_entry?: PGHypothesisMemory
  validation_changes?: { case_id: string; repetition: number; before_passed: boolean; after_passed: boolean; model_attempt_delta: number }[]
  iteration: number; status: string; reason: string; parent_sha256: string; candidate_sha256?: string; retained_sha256: string
  incumbent_metrics?: { score: number; model_attempts: number }
  candidate_metrics?: { score: number; model_attempts: number }
  patch?: { reason: string; source_cases: string[]; add_nodes: PGGraph['nodes']; delete_nodes: string[]; add_edges: PGEdge[]; delete_edges: string[] }
}
export interface PGArmResult {
  profile: { profile_id: string }; execution_source: string; live_verified: boolean
  metrics: { samples: number; mean_success_rate: number | null; recovery_rate: number | null
    triggered_samples: number; pg_exposure_rate: number | null; repeated_ineffective_calls: number; model_attempts: number
    pg_localization_fallbacks?: number; pg_localization_abstentions?: number
    pg_shadow_disagreements?: number; pg_shadow_full_graph_fallbacks?: number
    pg_no_progress_repeats_blocked?: number; pg_deterministic_contracts?: number; pg_guidance_model_calls?: number
    pg_typed_applicability_interventions?: number; pg_actions_progressed?: number
    pg_actions_without_progress?: number; pg_actions_failed?: number
    pg_preplay_interventions?: number; pg_prediction_supported?: number; pg_prediction_countered?: number
    pg_prediction_diverged?: number; pg_prediction_unknown?: number
    tool_calls: number; tokens: { total_tokens: number | null }; latency_ms: { p95: number | null } }
}
export interface PGJobSummary {
  job_id: string; created_at: string; preset: PGPreset; status: string; stage: string; completed_samples: number
}
export interface PGJob extends PGJobSummary {
  cases: PGCatalog['dataset']['cases']
  learning_cost: { rollout_model_attempts: number; refiner_model_attempts: number; rollout_tokens: number | null; refiner_token_units: number }
  budget_usage: { model_attempts: number | null; token_units: number | null }
  frozen: PGPlan; stage_completed: number; stage_total: number; cancel_requested: boolean; stale: boolean
  identity: Record<string, unknown>; execution_source: string; history: PGSelection[]; arms: Partial<Record<PGArm, PGArmResult>>
  error: string | null; error_type?: string | null; seed_graph: PGGraph; retained_graph: PGGraph | null
  comparison: { conclusion: string; evidence_valid: boolean; failures: string[]; gates: Record<string, { passed: boolean; failures: string[] }> } | null
}
