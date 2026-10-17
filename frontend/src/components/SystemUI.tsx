import type { HarnessProfile, RunEvent } from '../types'

export const SLOT_ORDER = ['planning', 'context', 'memory', 'verification', 'tool_protocol'] as const
export const SLOT_LABELS: Record<string, string> = {
  planning: '规划', context: '上下文', memory: '记忆', verification: '核验', tool_protocol: '工具协议',
}

export const STATUS_LABELS: Record<string, string> = {
  SUCCEEDED: '已完成', FAILED: '执行失败', CANCELLED: '已取消', PARTIAL: '部分完成',
  INTERRUPTED: '等待恢复', RUNNING: '执行中', WAITING_APPROVAL: '等待确认',
  WAITING_CLARIFICATION: '需要补充信息', WAITING_HUMAN: '等待人工', ESCALATED: '已转人工',
  placed: '已下单', shipped: '运输中', delivered: '已送达', cancelled: '已取消',
  pending: '待支付', paid: '已支付', refunded: '已退款', none: '暂无退款', requested: '退款申请中',
}

const EVENT_LABELS: Record<string, [string, string]> = {
  harness_profile_selected: ['PROFILE', '装配配置'], mechanism_applied: ['MECHANISM', '应用机制'],
  intake: ['INTAKE', '解析需求'], route_decision: ['ROUTE', '选择路径'], plan_created: ['PLAN', '生成计划'],
  step_started: ['STEP', '开始步骤'], step_completed: ['STEP', '完成步骤'], model_call: ['MODEL', '模型调用'],
  tool_call: ['TOOL', '调用工具'], tool_result: ['RESULT', '工具结果'], verification: ['VERIFY', '完成核验'],
  verification_failed: ['VERIFY', '核验未通过'], request_audited: ['AUDIT', '独立审查'],
  control_applied: ['CONTROL', '运行时调整'], plan_replanned: ['REPLAN', '重规划'],
  error: ['ERROR', '执行错误'], terminal: ['END', '运行结束'],
}

const TOOL_LABELS: Record<string, string> = {
  get_order_status: '查询订单状态', get_order_details: '读取订单详情', track_shipment: '查询物流进度',
  check_refund_eligibility: '核验退款资格', initiate_refund: '发起退款申请', get_payment_status: '查询支付状态',
  process_payment: '处理支付', answer_faq: '查询服务规则', escalate_to_human: '转交人工客服',
  search_products: '搜索商品目录', compare_products: '对比商品参数', select_product: '核验并选择商品',
}

export function PageHeader({ code, title, subtitle, marker = 'LIVE SYSTEM' }: {
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
      <div className="fingerprint-head"><span>{profile.profile_id}</span><span>{profile.engine.id} · {profile.research.status}</span></div>
      <div className="mechanism-rail">
        {SLOT_ORDER.map((slot, index) => {
          const mechanism = profile.mechanisms[slot]
          const changed = reference !== undefined && JSON.stringify(mechanism) !== JSON.stringify(reference.mechanisms[slot])
          return (
            <div className={`mechanism-cell ${!mechanism ? 'empty' : changed ? 'changed' : 'active'}`} key={slot}>
              <span className="slot-index">{String(index + 1).padStart(2, '0')}</span>
              <span className="slot-label">{SLOT_LABELS[slot]}</span>
              <strong>{mechanism?.id ?? 'unbound'}</strong>
              <small>{mechanism ? `v${mechanism.version}` : '—'}</small>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function eventLabel(event: RunEvent): [string, string] {
  const [code, defaultLabel] = EVENT_LABELS[event.type] ?? [event.type.toUpperCase(), event.type.replaceAll('_', ' ')]
  if ((event.type === 'tool_call' || event.type === 'tool_result') && event.tool) return [code, TOOL_LABELS[event.tool] ?? event.tool]
  if (event.type === 'model_call' && event.purpose) return [code, `模型 · ${event.purpose}`]
  if (event.type === 'terminal' && event.status) return [code, STATUS_LABELS[event.status] ?? event.status]
  return [code, defaultLabel]
}

export function ExecutionSpine({ events, selected, onSelect }: {
  events: RunEvent[]; selected?: number; onSelect?: (index: number) => void
}) {
  if (events.length === 0) return <div className="empty-inline">NO EVENTS RECORDED · 暂无事件</div>
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
  return <div className="chip-row">{values.map((value) => <span key={value}>{value}</span>)}</div>
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
