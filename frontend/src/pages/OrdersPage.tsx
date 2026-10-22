import { useQuery } from '@tanstack/react-query'
import { CreditCard, FilePlus2, Fingerprint, RotateCcw, Scale, Search, ShieldCheck, Truck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, SectionHeading, STATUS_LABELS, StatusBadge } from '../components/SystemUI'
import type { AfterSalesCaseDetail, AfterSalesCaseEvent, AfterSalesCaseSummary } from '../types'

const ITEM_LABELS: Record<string, string> = {
  'Wireless Headphones': '无线耳机', 'Mechanical Keyboard': '机械键盘', 'USB-C Hub': 'USB-C 扩展坞',
  'Standing Desk': '升降桌', 'Desk Lamp': '台灯', 'Bluetooth Speaker': '蓝牙音箱',
  'Laptop Stand': '笔记本支架', 'Webcam 1080p': '1080p 摄像头', 'Ergonomic Mouse': '人体工学鼠标',
  'Noise-Cancelling Earbuds': '降噪耳塞',
}

const CASE_TYPE_LABELS: Record<string, string> = {
  REFUND: '退款案件', RETURN: '退货案件', EXCHANGE: '换货案件', REPAIR: '维修案件', DELIVERY_ISSUE: '配送异常',
}

const RESOLUTION_LABELS: Record<string, string> = {
  REFUND: '仅退款', RETURN_AND_REFUND: '退货退款', REPLACEMENT: '换货', REPAIR: '维修', INFORMATION: '信息核实',
}

const CASE_EVENT_LABELS: Record<string, string> = {
  CASE_OPENED: '案件建立', RUN_LINKED: '运行关联', EVIDENCE_ADDED: '证据入账', POLICY_DECIDED: '政策裁决',
  CASE_SUBMITTED: '退款申请已提交', CASE_CANCELLED: '案件取消', CASE_RESOLVED: '案件解决', CASE_REJECTED: '案件拒绝',
}

const ACTOR_LABELS: Record<string, string> = {
  CUSTOMER: '用户', AGENT: 'Agent', SYSTEM: '系统', HUMAN: '人工专员',
}

const POLICY_REASON_LABELS: Record<string, string> = {
  POLICY_REQUIREMENTS_SATISFIED: '订单事实满足当前政策',
  AUTHORITATIVE_BUSINESS_DATA_MISSING: '缺少权威业务事实',
  ORDER_STATUS_NOT_ELIGIBLE: '订单状态不符合政策',
  PAYMENT_STATUS_NOT_ELIGIBLE: '支付状态不符合政策',
  REFUND_ALREADY_EXISTS: '订单已有退款记录',
  DELIVERY_EVIDENCE_MISSING: '缺少签收证据',
  DELIVERY_TIMESTAMP_IN_FUTURE: '签收时间异常',
  RETURN_WINDOW_EXPIRED: '已超过售后时限',
  AMOUNT_REQUIRES_MANUAL_REVIEW: '订单金额需要人工审核',
}

const ACTION_LABELS: Record<string, string> = {
  REQUEST_CUSTOMER_CONFIRMATION: '请求用户确认', ESCALATE_TO_HUMAN: '转交人工', ADD_EVIDENCE: '补充证据',
  CLOSE_CASE: '关闭案件', CANCEL_CASE: '取消案件', VIEW_POLICY: '查看政策', VIEW_EXISTING_REFUND: '查看已有退款',
  WAIT_FOR_HUMAN: '等待人工审核',
}

const EVIDENCE_LABELS: Record<string, string> = {
  ISSUE_DESCRIPTION: '问题描述', ORDER_FACT: '订单事实', PAYMENT_FACT: '支付事实', DELIVERY_FACT: '配送事实',
}

const RUN_ROLE_LABELS: Record<string, string> = {
  INTAKE: '案件受理', REASSESSMENT: '重新核验', SUBMISSION: '提交执行', READ: '案件读取',
}

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
})

function formatTimestamp(value: string): string {
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : dateFormatter.format(parsed)
}

function compactId(value: string | null | undefined, length = 12): string {
  if (!value) return '—'
  return value.length <= length ? value : `${value.slice(0, length)}…`
}

function policyReasonLabel(reason: string): string {
  if (reason.startsWith('MISSING_EVIDENCE:')) {
    const evidenceType = reason.slice('MISSING_EVIDENCE:'.length)
    return `缺少${EVIDENCE_LABELS[evidenceType] ?? evidenceType}证据`
  }
  return POLICY_REASON_LABELS[reason] ?? reason
}

function eventSummary(event: AfterSalesCaseEvent): string | null {
  const payload = event.payload
  if (event.event_type === 'CASE_OPENED') {
    const resolution = typeof payload.requested_resolution === 'string' ? payload.requested_resolution : ''
    return resolution ? `处理目标：${RESOLUTION_LABELS[resolution] ?? resolution}` : null
  }
  if (event.event_type === 'RUN_LINKED') {
    const role = typeof payload.role === 'string' ? payload.role : ''
    return role ? `运行职责：${RUN_ROLE_LABELS[role] ?? role}` : null
  }
  if (event.event_type === 'EVIDENCE_ADDED') {
    const evidenceType = typeof payload.evidence_type === 'string' ? payload.evidence_type : ''
    return evidenceType ? `证据类型：${EVIDENCE_LABELS[evidenceType] ?? evidenceType}` : null
  }
  if (event.event_type === 'POLICY_DECIDED') {
    const outcome = typeof payload.outcome === 'string' ? payload.outcome : ''
    return outcome ? `裁决结果：${STATUS_LABELS[outcome] ?? outcome}` : null
  }
  if (event.event_type === 'CASE_SUBMITTED') {
    return payload.refund_id ? `已关联退款 #${String(payload.refund_id)}` : null
  }
  const from = typeof payload.from_status === 'string' ? payload.from_status : ''
  const to = typeof payload.to_status === 'string' ? payload.to_status : ''
  return from && to ? `${STATUS_LABELS[from] ?? from} → ${STATUS_LABELS[to] ?? to}` : null
}

function CaseIndexRow({ item, selected, onSelect }: {
  item: AfterSalesCaseSummary; selected: boolean; onSelect: () => void
}) {
  return <button className={`after-sales-case-row ${selected ? 'selected' : ''}`} type="button" onClick={onSelect} aria-pressed={selected}>
    <span className="case-row-sequence">CASE</span>
    <span className="case-row-content">
      <strong>{CASE_TYPE_LABELS[item.case_type] ?? item.case_type} · 订单 #{item.order_id}</strong>
      <span>{item.reason_text}</span>
      <small>{compactId(item.case_id, 16)} · REV {item.revision} · {formatTimestamp(item.updated_at)}</small>
    </span>
    <span className="case-row-state">
      <StatusBadge status={item.status} />
      {item.policy_outcome && <span>{STATUS_LABELS[item.policy_outcome] ?? item.policy_outcome}</span>}
    </span>
  </button>
}

function CaseDossier({ detail }: { detail: AfterSalesCaseDetail }) {
  const { case: caseData, policy, refund, runs } = detail
  const decision = caseData.latest_decision
  const reasons = decision?.reason_codes ?? []
  const allowedActions = decision?.allowed_actions ?? []
  return <div className="case-dossier">
    <header className="case-dossier-header">
      <div>
        <span>CASE DOSSIER / {CASE_TYPE_LABELS[caseData.case_type] ?? caseData.case_type}</span>
        <h2>订单 #{caseData.order_id} · {RESOLUTION_LABELS[caseData.requested_resolution] ?? caseData.requested_resolution}</h2>
        <p>{caseData.reason_text}</p>
      </div>
      <div className="case-identity"><StatusBadge status={caseData.status} /><code title={caseData.case_id}>{caseData.case_id}</code></div>
    </header>

    <section className={`policy-verdict ${decision?.outcome.toLowerCase() ?? 'pending'}`} aria-label="政策裁决">
      <div className="policy-verdict-mark"><Scale size={19} /><span>POLICY<br />DECISION</span></div>
      <div className="policy-verdict-copy">
        <small>确定性政策引擎裁决</small>
        <h3>{decision ? STATUS_LABELS[decision.outcome] ?? decision.outcome : '尚未生成裁决'}</h3>
        {reasons.length > 0
          ? <div className="policy-reason-list">{reasons.map((reason) => <span key={reason} title={reason}>{policyReasonLabel(reason)}</span>)}</div>
          : <p>当前案件尚无可展示的裁决依据。</p>}
      </div>
      <div className="policy-stamp">
        <ShieldCheck size={16} />
        <span>{policy ? `POLICY ${policy.version}` : 'NO POLICY SNAPSHOT'}</span>
        <code title={policy?.checksum}>{compactId(policy?.checksum, 14)}</code>
      </div>
    </section>

    <dl className="case-fact-grid">
      <div><dt>案件版本</dt><dd>REV {caseData.revision}</dd></div>
      <div><dt>订单商品</dt><dd>{caseData.order_item_id ?? '未绑定行项目'}</dd></div>
      <div><dt>活动证据</dt><dd>{caseData.evidence.length} 条</dd></div>
      <div><dt>关联运行</dt><dd>{runs.length} 个</dd></div>
      <div><dt>退款副作用</dt><dd>{refund ? `#${refund.refund_id} · ${STATUS_LABELS[refund.status] ?? refund.status}` : '尚未创建'}</dd></div>
      <div><dt>最后更新</dt><dd>{formatTimestamp(caseData.updated_at)}</dd></div>
    </dl>

    <div className="case-evidence-layout">
      <section className="case-ledger-panel">
        <div className="case-panel-title"><span>不可变事件账本</span><small>{caseData.events.length} EVENTS</small></div>
        {caseData.events.length > 0 ? <ol className="case-timeline">{caseData.events.map((event) => {
          const summary = eventSummary(event)
          return <li key={event.event_id}>
            <span className="case-event-sequence">{String(event.sequence).padStart(2, '0')}</span>
            <i />
            <div>
              <div><strong>{CASE_EVENT_LABELS[event.event_type] ?? event.event_type}</strong><code>REV {event.case_revision}</code></div>
              {summary && <p>{summary}</p>}
              <small>{ACTOR_LABELS[event.actor_type] ?? event.actor_type} · {formatTimestamp(event.occurred_at)}</small>
            </div>
          </li>
        })}</ol> : <div className="case-inline-state">尚无案件事件。</div>}
      </section>

      <div className="case-side-ledgers">
        <section className="case-ledger-panel">
          <div className="case-panel-title"><span>证据引用</span><small>{decision?.evidence_refs.length ?? 0} REFS</small></div>
          {caseData.evidence.length > 0 && <div className="case-record-list">{caseData.evidence.map((evidence) => <article key={evidence.evidence_id}>
            <div><strong>{EVIDENCE_LABELS[evidence.evidence_type] ?? evidence.evidence_type}</strong><StatusBadge status={evidence.status} /></div>
            <p>{evidence.source_type} · {evidence.source_ref}</p>
            <code title={evidence.checksum}>{compactId(evidence.checksum, 18)}</code>
          </article>)}</div>}
          {decision?.evidence_refs.length ? <div className="evidence-reference-list">{decision.evidence_refs.map((reference) => <code key={reference}>{reference}</code>)}</div> : <div className="case-inline-state">裁决尚未引用业务证据。</div>}
        </section>

        <section className="case-ledger-panel">
          <div className="case-panel-title"><span>执行与动作边界</span><small>{runs.length} RUNS</small></div>
          {allowedActions.length > 0 && <div className="case-action-list">{allowedActions.map((action) => <span key={action}>{ACTION_LABELS[action] ?? action}</span>)}</div>}
          {runs.length > 0 ? <div className="case-record-list">{runs.map((run) => <article key={run.run_id}>
            <div><strong>{RUN_ROLE_LABELS[run.role] ?? run.role}</strong><span>{formatTimestamp(run.linked_at)}</span></div>
            <p>会话 {compactId(run.session_id, 18)}</p>
            <code title={run.run_id}>{run.run_id}</code>
          </article>)}</div> : <div className="case-inline-state">尚无关联运行。</div>}
          {policy && <footer className="case-policy-source"><Fingerprint size={13} /><span>{policy.source_ref}</span><code>{policy.policy_version_id}</code></footer>}
        </section>
      </div>
    </div>
  </div>
}

export function OrdersPage() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('')
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const orders = useQuery({ queryKey: ['orders', search, status], queryFn: () => api.orders({ query: search, status: status || undefined }) })
  const cases = useQuery({ queryKey: ['after-sales-cases'], queryFn: () => api.afterSalesCases({ limit: 100 }), staleTime: 15_000 })
  const caseItems = cases.data?.cases ?? []

  useEffect(() => {
    if (caseItems.length === 0) {
      if (selectedCaseId !== null) setSelectedCaseId(null)
      return
    }
    if (!selectedCaseId || !caseItems.some((item) => item.case_id === selectedCaseId)) {
      setSelectedCaseId(caseItems[0].case_id)
    }
  }, [caseItems, selectedCaseId])

  const caseDetail = useQuery({
    queryKey: ['after-sales-case', selectedCaseId],
    queryFn: () => api.afterSalesCase(selectedCaseId!),
    enabled: Boolean(selectedCaseId),
    staleTime: 10_000,
  })
  const ask = (prompt: string) => navigate(`/sandbox?prompt=${encodeURIComponent(prompt)}`)

  return <>
    <PageHeader code="AFTER-SALES / CASE LEDGER" title="订单与售后案件" subtitle="从真实业务 API 读取订单、案件、政策裁决与执行证据；退款等副作用仍由 Agent 在用户明确确认后执行。" marker="真实业务投影" />

    <SectionHeading index="01" title="售后案件账本" note="CASE ≠ RUN ≠ CONVERSATION" />
    <div className="after-sales-workbench" aria-label="售后案件账本">
      <aside className="after-sales-case-index">
        <div className="case-index-header"><div><span>CASE INDEX</span><strong>{caseItems.length} 个案件</strong></div><small>按最后更新时间排序</small></div>
        {cases.isLoading && <div className="case-inline-state">正在读取案件账本…</div>}
        {cases.isError && <div className="case-inline-state error"><strong>案件账本读取失败</strong><button type="button" onClick={() => void cases.refetch()}>重新读取</button></div>}
        {!cases.isLoading && !cases.isError && caseItems.length === 0 && <div className="case-index-empty">
          <FilePlus2 size={20} /><strong>尚无售后案件</strong><p>从下方订单发起售后诉求后，Agent 会把案件、政策裁决和执行证据写入这里。</p>
        </div>}
        {caseItems.length > 0 && <div className="after-sales-case-list">{caseItems.map((item) => <CaseIndexRow
          key={item.case_id} item={item} selected={selectedCaseId === item.case_id} onSelect={() => setSelectedCaseId(item.case_id)}
        />)}</div>}
      </aside>
      <section className="case-dossier-shell" aria-label="案件档案">
        {!selectedCaseId && !cases.isLoading && <div className="case-dossier-empty"><Scale size={24} /><strong>选择案件查看政策与执行证据</strong><p>这里不会根据聊天文本猜测案件状态，只展示持久化业务投影。</p></div>}
        {selectedCaseId && caseDetail.isLoading && <div className="case-dossier-empty"><i className="case-loading-indicator" /><strong>正在装配案件档案</strong><p>读取政策版本、证据、运行关联和事件账本。</p></div>}
        {selectedCaseId && caseDetail.isError && <div className="case-dossier-empty error"><strong>案件档案读取失败</strong><p>{caseDetail.error instanceof Error ? caseDetail.error.message : '未知错误'}</p><button type="button" onClick={() => void caseDetail.refetch()}>重新读取</button></div>}
        {caseDetail.data && <CaseDossier detail={caseDetail.data} />}
      </section>
    </div>

    <SectionHeading index="02" title="订单业务数据" note="LOCAL SQLITE / API PROJECTION" />
    <div className="filter-row two"><label>搜索订单<div className="input-with-icon"><Search size={14} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="订单号、商品名称或演示客户" /></div></label><label>订单状态<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="">全部状态</option><option value="placed">已下单</option><option value="shipped">运输中</option><option value="delivered">已送达</option><option value="cancelled">已取消</option></select></label></div>
    {orders.isLoading && <LoadingState label="正在读取订单" />}
    {orders.isError && <ErrorState error={orders.error} retry={() => void orders.refetch()} />}
    {orders.data && <>
      <div className="catalog-note">{orders.data.orders.length} 条订单 · 数据来自共享本地演示数据库，不代表真实客户账户</div>
      {orders.data.orders.length === 0 ? <div className="empty-inline">没有符合当前筛选条件的订单。</div> : <div className="order-list">{orders.data.orders.map((order) => <article className="order-card" key={order.order_id}>
        <div className="order-main"><div className="order-symbol">↗</div><div><small>ORDER #{order.order_id}</small><h3>{ITEM_LABELS[order.item] ?? order.item}</h3><p>{order.order_date} · ¥{order.amount.toFixed(2)} · 支付 {STATUS_LABELS[order.payment_status ?? 'none'] ?? order.payment_status}</p></div></div><StatusBadge status={order.status} />
        <div className="order-actions"><button onClick={() => ask(`查询订单 ${order.order_id} 的状态和物流`)}><Truck size={14} />状态与物流</button><button onClick={() => ask(`查询订单 ${order.order_id} 是否可以退款`)}><RotateCcw size={14} />检查退款资格</button><button className="primary" disabled={order.status === 'cancelled'} onClick={() => ask(order.payment_status === 'pending' ? `帮我支付订单 ${order.order_id}` : `订单 ${order.order_id} 的商品出现问题，请建立一个可追踪的售后案件，核验适用政策，并在任何退款提交前要求我明确确认`)}><CreditCard size={14} />{order.payment_status === 'pending' ? '准备支付' : '建立售后案件'}</button></div>
      </article>)}</div>}
    </>}
  </>
}
