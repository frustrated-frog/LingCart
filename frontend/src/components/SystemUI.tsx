import type { HarnessProfile, RunEvent } from '../types'

export const SLOT_ORDER = ['planning', 'context', 'memory', 'verification', 'steering', 'tool_protocol'] as const
export const SLOT_LABELS: Record<string, string> = {
  planning: '规划', context: '上下文', memory: '记忆', verification: '核验', steering: '在线纠偏', tool_protocol: '工具协议',
}

export const STATUS_LABELS: Record<string, string> = {
  SUCCEEDED: '已完成', FAILED: '执行失败', CANCELLED: '已取消', PARTIAL: '部分完成',
  INTERRUPTED: '等待恢复', RUNNING: '执行中', WAITING_APPROVAL: '等待确认',
  WAITING_CLARIFICATION: '需要补充信息', WAITING_HUMAN: '等待人工', ESCALATED: '已转人工',
  PENDING: '待执行', SKIPPED: '已跳过', UNKNOWN: '状态未知',
  placed: '已下单', shipped: '运输中', delivered: '已送达', cancelled: '已取消',
  pending: '待支付', paid: '已支付', refunded: '已退款', none: '暂无退款', requested: '退款申请中',
}

export const TERMINAL_REASON_LABELS: Record<string, string> = {
  COMPLETED: '正常完成', FAILED: '执行失败', CANCELLED: '用户取消',
  EVIDENCE_UNAVAILABLE: '缺少可验证证据', NO_CANDIDATE_EVIDENCE: '没有可验证候选',
  PARTIAL_UNSUPPORTED_INTENT: '部分目标超出能力范围', WAITING_APPROVAL: '等待用户确认',
  WAITING_CLARIFICATION: '等待补充信息', WAITING_HUMAN: '等待人工处理',
  MISSING_PAYMENT_METHOD: '缺少支付方式', BUSINESS_DATA_NOT_FOUND: '未找到业务数据',
  BUSINESS_RULE_REJECTED: '业务规则拒绝执行', INVALID_PENDING_ACTION: '待处理操作无效',
  APPROVAL_REQUIRED: '需要用户确认', NO_PENDING_ACTION: '没有待确认操作',
  MODEL_CLARIFICATION: '模型请求补充信息', MISSING_ORDER_ID: '缺少订单号',
  MISSING_REQUIRED_INFORMATION: '缺少必要信息', UNSUPPORTED_INTENT: '请求超出能力范围',
  USER_CANCELLED_ACTION: '用户取消操作', USER_CANCELLED_RUN: '用户取消运行',
  AMBIGUOUS_ORDER_GOALS: '订单目标存在歧义',
  MULTIPLE_ACTIONS_REQUIRE_SEPARATE_APPROVAL: '多个操作需要分别确认',
  MODEL_ERROR: '模型调用失败', MODEL_TIMEOUT: '模型调用超时', TOOL_ERROR: '工具执行失败',
  BUDGET_EXHAUSTED: '运行预算已用尽', LOOP_GUARD: '循环保护终止',
}

export const RESEARCH_STATUS_LABELS: Record<string, string> = {
  baseline: '基线', candidate: '候选', promoted: '已晋升',
}

export const SOURCE_TYPE_LABELS: Record<string, string> = {
  native: '原生实现', paper: '论文实现', hybrid: '混合实现',
}

export const HEALTH_LABELS: Record<string, string> = {
  READY: '就绪', NOT_READY: '未就绪', OPEN: '熔断中', RECOVERING: '恢复中',
  DEGRADED: '降级运行', UNKNOWN: '状态未知',
}

export const TOOL_LABELS: Record<string, string> = {
  get_order_status: '查询订单状态', get_order_details: '读取订单详情', track_shipment: '查询物流进度',
  check_refund_eligibility: '核验退款资格', prepare_refund: '准备退款方案', initiate_refund: '发起退款申请',
  get_payment_status: '查询支付状态', prepare_payment: '准备支付方案', process_payment: '处理支付',
  answer_faq: '查询服务规则', escalate_to_human: '转交人工客服', get_catalog_facets: '读取目录分类',
  search_products: '搜索商品目录', discover_products: '发现商品候选', compare_products: '对比商品参数',
  select_product: '核验并选择商品', read_clock: '读取当前时间', calculate: '执行计算',
  __model_answer__: '生成知识回答',
}

export const CAPABILITY_LABELS: Record<string, string> = {
  deterministic_intake: '确定性请求接入', open_set_dialogue_understanding: '开放集对话理解',
  multi_objective_planning: '多目标规划', durable_runtime: '持久化运行时', approval_guard: '审批保护',
  tool_ledger: '工具调用账本', model_directed_planning: '模型驱动规划', bounded_replanning: '有界重规划',
  evidence_retrieval: '证据检索', objective_first_context: '目标证据优先上下文',
  ace_frozen_playbook: 'ACE 冻结策略手册', offline_experience_learning: '离线经验学习',
  goal_verification: '目标核验', durable_resume: '持久化恢复', runtime_steering: '运行时调整',
  deterministic_trajectory_monitoring: '确定性轨迹监控', conditional_corrective_advice: '条件式纠偏建议',
  precommit_action_intervention: '执行前动作干预',
}

export const MECHANISM_LABELS: Record<string, string> = {
  'structured-dialogue-understanding': '结构化对话理解', 'runtime-tool-ledger': '运行时工具账本',
  'deterministic-business-gates': '确定性业务门禁', 'bounded-model-loop': '有界模型循环',
  'evidence-reference-context': '证据引用上下文', 'objective-first-context': '目标证据优先上下文',
  'ace-playbook-context': 'ACE 策略手册上下文', 'objective-ledger': '目标账本',
  'independent-request-audit': '独立请求审查', 'native-strict-tools': '原生严格工具协议',
  'liveplan-online-steering': 'LivePlan 在线监控与纠偏',
}

export const ENGINE_LABELS: Record<string, string> = {
  adaptive: '自主执行引擎', orchestrator: '编排执行引擎', runtime: '运行时引擎',
}

export const SPECIALIST_LABELS: Record<string, string> = {
  order: '订单专员', refund: '退款专员', payment: '支付专员', faq: '规则问答',
  shopping: '商品选购', knowledge: '知识回答', utility: '通用工具', runtime: '运行时',
  orchestrator: '任务编排', intake: '请求接入', human: '人工客服',
}

export const SPLIT_LABELS: Record<string, string> = { fixed: '固定集', dev: '开发集', test: '测试集' }
export const SUITE_LABELS: Record<string, string> = { support: '客服业务', adaptive: '自主智能体' }
export const EXECUTION_SOURCE_LABELS: Record<string, string> = { scripted: '脚本模型', live: '真实模型' }
export const LANGUAGE_LABELS: Record<string, string> = { zh: '中文', en: '英语', 'zh-CN': '中文' }
export const OBJECTIVE_LABELS: Record<string, string> = { maximize: '越高越好', minimize: '越低越好' }

export const METRIC_LABELS: Record<string, string> = {
  mean_success_rate: '平均任务成功率', pass_at_1: '首次执行通过率',
  all_repeats_success_rate: '全部重复执行成功率', safety_violations: '安全违规次数',
  protocol_violations: '协议违规次数', 'tokens.total_tokens': '总令牌数',
  'latency_ms.p95': '延迟 P95', model_attempts: '模型尝试次数', tool_calls: '工具调用次数',
  'trajectory_control.drift_detections': '轨迹漂移次数',
  'trajectory_control.advisor_calls': '纠偏模型调用次数',
  'trajectory_control.action_rejections': '动作阻断次数',
  'trajectory_control.recovery_rate': '漂移后恢复率',
}

const EVENT_LABELS: Record<string, [string, string]> = {
  agent_status_changed: ['状态', 'Agent 执行状态'], response_started: ['回答', '开始生成回答'],
  response_delta: ['回答', '接收回答分片'], response_completed: ['回答', '回答生成完成'],
  response_failed: ['回答', '回答生成中断'],
  harness_profile_selected: ['配置', '装配配置'], mechanism_applied: ['机制', '应用机制'],
  intake: ['接入', '解析需求'], route_decision: ['路由', '选择路径'], plan_created: ['计划', '生成计划'],
  step_started: ['步骤', '开始步骤'], step_completed: ['步骤', '完成步骤'], step_skipped: ['步骤', '跳过步骤'],
  model_started: ['模型', '开始模型调用'], model_call: ['模型', '模型调用'], model_retry: ['模型', '重试模型调用'],
  model_reused: ['模型', '复用模型结果'], tool_requested: ['工具', '请求调用工具'],
  tool_call: ['工具', '调用工具'], tool_result: ['结果', '工具结果'], tool_retry: ['工具', '重试工具调用'],
  tool_protocol: ['协议', '工具协议'], native_submission: ['提交', '提交原生工具调用'],
  native_batch_cancelled: ['取消', '取消原生工具批次'], verification: ['核验', '完成核验'],
  verification_failed: ['核验', '核验未通过'], request_audited: ['审查', '独立审查'],
  audit_protocol_retry: ['审查', '重试审查协议'], audit_invalidated: ['审查', '审查结果失效'],
  control_applied: ['控制', '运行时调整'], plan_replanned: ['重规划', '完成重规划'],
  replan_requested: ['重规划', '请求重规划'], adaptive_decision: ['决策', '自主决策'],
  decision_rejected: ['决策', '拒绝无效决策'], approval_required: ['审批', '需要用户确认'],
  context_compacted: ['上下文', '压缩上下文'], evidence_context: ['证据', '装配证据上下文'],
  evidence_read: ['证据', '读取证据'], constraint_binding_proposed: ['约束', '生成映射提案'],
  constraint_binding_unresolved: ['约束', '等待兼容证据'], constraint_binding_rejected: ['审查', '拒绝映射提案'],
  constraint_bindings_refined: ['约束', '提交约束映射'], decision_action_surface: ['预算', '收敛决策动作面'],
  trajectory_monitor_evaluated: ['监控', '评估执行轨迹'], trajectory_drift_detected: ['漂移', '检测轨迹漂移'],
  corrective_advice_generated: ['纠偏', '生成纠偏建议'], corrective_advice_failed: ['降级', '纠偏建议安全降级'],
  intervention_applied: ['干预', '应用受限干预'],
  dialogue_protocol_retry: ['理解', '重试对话协议'], unknown_capability_downgraded: ['能力', '降级未知能力'],
  hook_applied: ['扩展', '应用扩展'], hook_failed: ['扩展', '扩展执行失败'],
  knowledge_result: ['知识', '生成知识结果'], error: ['错误', '执行错误'], terminal: ['结束', '运行结束'],
}

const MODEL_PURPOSE_LABELS: Record<string, string> = {
  dialogue_understanding: '对话理解', goal: '目标提取', planner: '任务规划',
  agent: '智能体执行', final_response: '最终回答', audit: '独立审查', liveplan_advisor: 'LivePlan 纠偏建议',
}

export function localizeEngineeringText(value: string): string {
  const replacements: Array<[RegExp, string]> = [
    [/\s*\bHarness Profile\b\s*/g, '运行配置档案'], [/\s*\bProfile\b\s*/g, '配置档案'], [/\s*\bRegistry\b\s*/g, '注册表'],
    [/\s*\bPlaybook\b\s*/g, '策略手册'], [/\s*\bGenerator\b\s*/g, '生成器'], [/\s*\bReflector\b\s*/g, '反思器'], [/\s*\bCurator\b\s*/g, '策展器'],
    [/\s*\bProvider\b\s*/g, '模型服务商'], [/\s*\bGate\b\s*/g, '评测门禁'], [/\s*\bAgent\b\s*/g, '智能体'],
    [/\s*\bdev\b\s*/g, '开发集'], [/\s*\btest\b\s*/g, '测试集'], [/\s*\bHarness\b\s*/g, '运行框架'],
    [/\s*\bhistory\b\s*/gi, '历史记录'], [/\s*\bContext\b\s*/g, '上下文'], [/\s+\bID\b\s*/g, '编号'], [/\s*\bnone\b\s*/g, '无退款'],
  ]
  return replacements.reduce((text, [pattern, replacement]) => text.replace(pattern, replacement), value)
}

export function PageHeader({ code, title, subtitle, marker = '实时系统' }: {
  code: string; title: string; subtitle: string; marker?: string
}) {
  return (
    <header className="page-header">
      <div className="eyebrow"><span>{code}</span><span className="live-marker"><i />{marker}</span></div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </header>
  )
}

export function SectionHeading({ index, title, note }: { index: string; title: string; note?: string }) {
  return <div className="section-heading"><div><b>{index}</b><span>{title}</span></div>{note && <small>{note}</small>}</div>
}

export function TelemetryStrip({ items }: { items: Array<{ label: string; value: string | number; note: string; tone?: string }> }) {
  return <div className="telemetry-strip">{items.map((item) => (
    <div className={`telemetry-cell ${item.tone ?? 'ink'}`} key={item.label}>
      <span>{item.label}</span><strong>{item.value}</strong><small>{item.note}</small>
    </div>
  ))}</div>
}

export function StatusBadge({ status }: { status: string }) {
  const tone = status === 'SUCCEEDED' || status === 'delivered' || status === 'paid' ? 'ok'
    : status === 'FAILED' ? 'bad'
      : status.startsWith('WAITING') || status === 'INTERRUPTED' || status === 'pending' ? 'warn' : ''
  return <span className={`status-badge ${tone}`}>{STATUS_LABELS[status] ?? status}</span>
}

export function MechanismRail({ profile, reference }: { profile: HarnessProfile; reference?: HarnessProfile }) {
  return (
    <div>
      <div className="fingerprint-head"><span>{profile.profile_id}</span><span>{ENGINE_LABELS[profile.engine.id] ?? profile.engine.id} · {RESEARCH_STATUS_LABELS[profile.research.status] ?? profile.research.status}</span></div>
      <div className="mechanism-rail">
        {SLOT_ORDER.map((slot, index) => {
          const mechanism = profile.mechanisms[slot]
          const changed = reference !== undefined && JSON.stringify(mechanism) !== JSON.stringify(reference.mechanisms[slot])
          const mechanismLabel = mechanism ? MECHANISM_LABELS[mechanism.id] : undefined
          return (
            <div className={`mechanism-cell ${!mechanism ? 'empty' : changed ? 'changed' : 'active'}`} key={slot}>
              <span className="slot-index">{String(index + 1).padStart(2, '0')}</span>
              <span className="slot-label">{SLOT_LABELS[slot]}</span>
              <strong>{mechanismLabel ?? mechanism?.id ?? '未绑定'}</strong>
              <small>{mechanism ? `${mechanismLabel ? `${mechanism.id} · ` : ''}v${mechanism.version}` : '—'}</small>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function eventLabel(event: RunEvent): [string, string] {
  const [code, defaultLabel] = EVENT_LABELS[event.type] ?? ['事件', event.type]
  if (event.type === 'agent_status_changed' && typeof event.label === 'string') return [code, event.label]
  if (event.type.startsWith('tool_') && event.tool) return [code, TOOL_LABELS[event.tool] ?? event.tool]
  if (event.type.startsWith('model_') && event.purpose) return [code, `模型 · ${MODEL_PURPOSE_LABELS[event.purpose] ?? event.purpose}`]
  if (event.type === 'terminal' && event.status) return [code, STATUS_LABELS[event.status] ?? event.status]
  return [code, defaultLabel]
}

export function ExecutionSpine({ events, selected, onSelect }: {
  events: RunEvent[]; selected?: number; onSelect?: (index: number) => void
}) {
  if (events.length === 0) return <div className="empty-inline">暂未记录执行事件</div>
  return <div className="execution-spine">{events.map((event, index) => {
    const [code, label] = eventLabel(event)
    const tone = event.type === 'terminal' ? 'terminal' : ['error', 'verification_failed', 'hook_failed'].includes(event.type) ? 'error' : ''
    return (
      <button type="button" className={`spine-node ${tone}${selected === index ? ' selected' : ''}`} key={`${event.sequence ?? index}-${event.type}`} onClick={() => onSelect?.(index)}>
        <i /><span>{String(event.sequence ?? index + 1).padStart(2, '0')} / {code}</span><strong>{label}</strong>
      </button>
    )
  })}</div>
}

export function DefinitionGrid({ items }: { items: Array<[string, string | number]> }) {
  return <dl className="definition-grid">{items.map(([label, value]) => (
    <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
  ))}</dl>
}

export function Chips({ values }: { values: string[] }) {
  return <div className="chip-row">{values.map((value) => <span key={value} title={value}>{CAPABILITY_LABELS[value] ?? value}</span>)}</div>
}

export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

export function LoadingState({ label = '正在读取系统状态' }: { label?: string }) {
  return <div className="loading-state"><i /><span>{label}</span></div>
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  return <div className="error-state"><b>数据未加载</b><p>{error instanceof Error ? error.message : '发生未知错误'}</p>{retry && <button onClick={retry}>重新读取</button>}</div>
}

export function formatTime(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value))
}
