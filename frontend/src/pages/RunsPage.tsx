import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { ResultArtifactViews } from '../components/CatalogResultView'
import type { RunDetail } from '../types'
import {
  DefinitionGrid,
  ErrorState,
  eventLabel,
  ExecutionSpine,
  formatTime,
  JsonBlock,
  LoadingState,
  PageHeader,
  SegmentedTabs,
  SectionHeading,
  SPECIALIST_LABELS,
  STATUS_LABELS,
  TERMINAL_REASON_LABELS,
  TelemetryStrip,
  TOOL_LABELS,
  ViewTransition,
} from '../components/SystemUI'

type Tab = 'result' | 'evidence' | 'steering' | 'pg' | 'tools' | 'raw'

export function RunsPage({ businessMode = false }: { businessMode?: boolean }) {
  const { runId } = useParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState('ALL')
  const [profile, setProfile] = useState('ALL')
  const [tab, setTab] = useState<Tab>('result')
  const [selectedEvent, setSelectedEvent] = useState(0)
  const runsQuery = useQuery({ queryKey: ['runs'], queryFn: () => api.runs({ limit: 100 }), refetchInterval: 5_000 })
  const filtered = useMemo(() => (runsQuery.data?.runs ?? []).filter((run) =>
    (status === 'ALL' || run.status === status) && (profile === 'ALL' || run.profile.profile_id === profile),
  ), [profile, runsQuery.data, status])

  useEffect(() => {
    if (filtered[0] && (!runId || !filtered.some((run) => run.run_id === runId))) {
      navigate(`${businessMode ? '/sandbox/tasks' : '/runs'}/${filtered[0].run_id}`, { replace: true })
    }
  }, [businessMode, filtered, navigate, runId])
  useEffect(() => setSelectedEvent(0), [runId])

  const detailQuery = useQuery({
    queryKey: ['run-detail', runId],
    queryFn: () => api.run(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query) => query.state.data?.status === 'RUNNING' ? 1_000 : false,
  })

  if (runsQuery.isLoading) return <LoadingState label="正在读取运行档案" />
  if (runsQuery.isError) return <ErrorState error={runsQuery.error} retry={() => void runsQuery.refetch()} />
  const runs = runsQuery.data?.runs ?? []
  const statuses = [...new Set(runs.map((run) => run.status))].sort()
  const profiles = [...new Set(runs.map((run) => run.profile.profile_id))].sort()
  const detail = detailQuery.data
  const events = detail?.result.events ?? []
  const event = events[selectedEvent]

  return (
    <>
      <PageHeader code={businessMode ? '任务档案 / 01' : '实时运行 / 01'}
        title={businessMode ? '任务记录' : '运行检查'}
        subtitle={businessMode ? '查看每次业务处理的真实结果，并可进入工程视图检查执行证据。' : '沿执行主脊检查一次任务：配置身份、事件顺序、证据状态、工具调用和终止原因。'} />
      {runs.length === 0 ? <div className="empty-state"><b>尚无运行记录</b><p>请先在业务沙盒执行一个消费任务。</p></div> : <>
        <div className="filter-row run-filters">
          <label>状态<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="ALL">全部状态</option>{statuses.map((value) => <option key={value} value={value}>{STATUS_LABELS[value] ?? value}</option>)}</select></label>
          <label>配置档案<select value={profile} onChange={(e) => setProfile(e.target.value)}><option value="ALL">全部配置档案</option>{profiles.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>运行<select value={runId ?? ''} onChange={(e) => navigate(`${businessMode ? '/sandbox/tasks' : '/runs'}/${e.target.value}`)}>
            {filtered.map((run) => <option value={run.run_id} key={run.run_id}>{formatTime(run.created_at)} · {run.request}</option>)}
          </select></label>
        </div>
        {detailQuery.isLoading && <LoadingState label="正在装载运行证据" />}
        {detailQuery.isError && <ErrorState error={detailQuery.error} retry={() => void detailQuery.refetch()} />}
        {detail && <ViewTransition viewKey={detail.run_id}>
          <TelemetryStrip items={[
            { label: '运行状态', value: STATUS_LABELS[detail.status] ?? detail.status, note: detail.terminal_reason ? (TERMINAL_REASON_LABELS[detail.terminal_reason] ?? detail.terminal_reason) : '尚未终止', tone: 'teal' },
            { label: '运行配置档案', value: detail.profile.profile_id, note: detail.profile.legacy_profile_reference ? '历史运行未记录' : '已记录', tone: 'violet' },
            { label: '工具 / 模型', value: `${detail.tool_calls} / ${detail.model_calls}`, note: `共 ${detail.event_count} 个事件`, tone: 'blue' },
            { label: '令牌用量', value: detail.tokens ?? '未报告', note: '模型服务商未报告时不伪造', tone: 'ink' },
          ]} />
          <SectionHeading index="01" title="执行主脊" note={`${detail.run_id.slice(0, 12)} / ${formatTime(detail.created_at)}`} />
          <div className="run-prompt">{detail.request}</div>
          <ExecutionSpine events={events} selected={selectedEvent} onSelect={setSelectedEvent} />
          {event && <ViewTransition viewKey={String(event.sequence ?? selectedEvent)}>
            <section className="event-probe">
              <div className="probe-title"><span>事件探针</span><b>{String(event.sequence ?? selectedEvent + 1).padStart(2, '0')} · {eventLabel(event).join(' / ')}</b></div>
              <DefinitionGrid items={[
                ['事件类型', eventLabel(event)[1]], ['事件语义', eventLabel(event).join(' / ')],
                ['发生时间', event.timestamp ?? '—'], ['事件序号', event.sequence ?? selectedEvent + 1],
              ]} />
              <details><summary>查看原始事件数据</summary><JsonBlock value={event} /></details>
            </section>
          </ViewTransition>}
          <SegmentedTabs value={tab} onChange={(value) => setTab(value as Tab)} layoutId="run-detail-tab"
            items={
              [['result', '结果与计划'], ['evidence', '上下文与证据'], ['steering', '在线监控与纠偏'], ['pg', '程序图指导'], ['tools', '工具账本'], ['raw', '原始记录']]
            } />
          <ViewTransition viewKey={tab} className="tab-panel">
            {tab === 'result' && <ResultTab detail={detail} />}
            {tab === 'evidence' && <EvidenceTab detail={detail} />}
            {tab === 'steering' && <SteeringTab detail={detail} />}
            {tab === 'pg' && <ProceduralGraphTab detail={detail} />}
            {tab === 'tools' && <ToolsTab detail={detail} />}
            {tab === 'raw' && <JsonBlock value={{ run_id: detail.run_id, harness: detail.profile, events, budget: detail.result.budget, usage: detail.result.usage, commands: detail.commands }} />}
          </ViewTransition>
        </ViewTransition>}
      </>}
    </>
  )
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function ProceduralGraphTab({ detail }: { detail: RunDetail }) {
  const events = detail.result.events ?? []
  const guidance = events.filter((item) => item.type === 'pg_guidance')
  const graph = record(record(detail.checkpoint.context).procedural_graph)
  const labels: Record<string, string> = { READY: '建议已生成', EXPOSED: '已送入决策', ABSTAINED: '图不适用，已弃权', SKIPPED: '已跳过', FAILED: '指导失败', CONTEXT_OMITTED: '上下文不足，未送入决策' }
  const reasons: Record<string, string> = { CONTEXT_BUDGET: '图或轨迹超出上下文预算', LOCALIZATION_ABSTAINED: '轨迹证据不足，定位器主动弃权', NO_OUTGOING_EDGES: '当前节点没有后继关系', RESERVED_DECISION_AND_AUDIT: '为决策和独立核验保留预算', GUIDANCE_SCHEMA_INVALID: '指导格式或引用无效' }
  if (!guidance.length && !Object.keys(graph).length) return <div className="empty-inline">该运行尚无程序图指导记录。</div>
  return <>
    <DefinitionGrid items={[
      ['图标识', String(graph.graph_id ?? guidance[0]?.graph_id ?? '未记录')],
      ['冻结图版本', String(graph.version ?? guidance[0]?.graph_version ?? '未记录')],
      ['指导记录', guidance.length],
      ['指导模型调用', events.filter((item) => ['model_started', 'model_retry'].includes(item.type) && item.purpose === 'pg_guidance').length],
    ]} />
    <p className="muted">图在本次运行中保持冻结。候选后继保持建议性质；只有已有无进展证据的完全相同重放可以在提交前阻断，任何单次行动都不代表已经证明效果改善。</p>
    <details><summary>查看冻结图指纹</summary><JsonBlock value={graph.sha256 ?? guidance[0]?.graph_sha256 ?? null} /></details>
    {!guidance.length && <div className="empty-inline">已绑定图，尚未生成指导；无法判断是否实际参与决策。</div>}
    <div className="steering-ledger pg-guidance-ledger">{guidance.map((item, index) => {
      const exposure = events.filter((event) => event.type === 'evidence_context' && event.pg_decision_key === item.decision_key).at(-1)
      const review = events.filter((event) => event.type === 'pg_decision_reviewed' && event.decision_key === item.decision_key).at(-1)
      const next = events.find((event) => event.type === 'adaptive_decision' && Number(event.sequence) > Number(item.sequence)
        && (index + 1 === guidance.length || Number(event.sequence) < Number(guidance[index + 1].sequence)))
      const decision = record(next?.decision)
      const status = String(exposure?.pg_status ?? item.status)
      const edgeIds = Array.isArray(item.edge_ids) ? item.edge_ids.map(String) : []
      const subgraph = record(item.subgraph)
      const edges = Array.isArray(subgraph.edges) ? subgraph.edges.map(record) : []
      const localization = record(item.localization)
      const shadow = record(localization.shadow)
      const confidence = typeof localization.confidence === 'number' ? `${Math.round(localization.confidence * 100)}%` : '未记录'
      const scope = item.abstained ? '主动弃权' : item.fallback ? '全图回退' : '有界局部图'
      const generation = record(item.generation)
      const contract = record(item.guidance_contract)
      return <article key={String(item.sequence ?? index)}>
        <span>事件 {String(item.sequence ?? '—')} · 定位 {String(item.anchor ?? '主动弃权')} · {scope}</span>
        <b>{labels[status] ?? status}</b>
        <p>{String(item.guidance || reasons[String(item.reason)] || item.reason || '没有生成可用建议')}</p>
        <small>生成方式：{generation.mode === 'deterministic-contract-v1' ? '确定性指导契约' : '模型情境化'} · 额外模型调用：{String(generation.model_calls ?? '未记录')}{Object.keys(contract).length ? ` · 契约 ${String(contract.contract_id ?? '').slice(0, 8)}` : ''}</small>
        <small>定位器：{String(localization.algorithm ?? 'action-anchor-v1')} · 置信度：{confidence}{Object.keys(shadow).length ? ` · v1 影子：${String(shadow.anchor ?? '未知')}${shadow.fallback ? '（全图回退）' : ''}` : ''}</small>
        <small>引用关系：{edgeIds.join(' / ') || '无'} · 后续行动：{String(decision.tool ?? decision.action ?? review?.actual_action ?? '尚无已提交行动')}{review ? ` · 采用审查：${String(review.status)}${review.blocked ? '（提交前阻断）' : ''}` : ''}</small>
        {edges.length > 0 && <details><summary>查看本次检索的行动关系与条件</summary><div className="table-wrap"><table><thead><tr><th>关系</th><th>条件</th><th>指导</th><th>常见错误</th></tr></thead><tbody>
          {edges.map((edge) => <tr key={String(edge.id)}><td>{String(edge.source)} → {String(edge.target)}<br /><code>{String(edge.relation)}</code></td><td>{String(edge.condition || '无条件')}</td><td>{String(edge.guidance)}</td><td>{String(edge.pitfalls || '未记录')}</td></tr>)}
        </tbody></table></div></details>}
        <details><summary>查看检索与指导原始证据</summary><JsonBlock value={item} /></details>
      </article>
    })}</div>
  </>
}

function SteeringTab({ detail }: { detail: RunDetail }) {
  const events = detail.result.events ?? []
  const monitorEvents = events.filter((item) => item.type === 'trajectory_monitor_evaluated')
  const driftEvents = events.filter((item) => item.type === 'trajectory_drift_detected')
  const adviceEvents = events.filter((item) => item.type === 'corrective_advice_generated')
  const interventionEvents = events.filter((item) => item.type === 'intervention_applied')
  const rejected = interventionEvents.filter((item) => item.outcome === 'ACTION_REJECTED')
  const adaptive = record(detail.checkpoint.adaptive)
  const liveplan = record(adaptive.liveplan)
  const lastAdvice = record(liveplan.last_advice)
  const active = record(liveplan.active_intervention)
  const isEnabled = detail.profile.profile_id.includes('liveplan') || monitorEvents.length > 0 || Object.keys(liveplan).length > 0

  if (!isEnabled) return <div className="empty-inline">该运行未启用在线轨迹监控；不会伪造漂移、建议或恢复数据。</div>
  return <>
    <DefinitionGrid items={[
      ['轨迹评估次数', monitorEvents.length],
      ['漂移触发次数', driftEvents.length],
      ['纠偏模型调用', Number(liveplan.advisor_calls ?? 0)],
      ['执行前阻断', rejected.length],
    ]} />
    {Object.keys(lastAdvice).length > 0 ? <section className="steering-advice">
      <div><span>最近纠偏建议</span><b>{String(lastAdvice.reason_code ?? '未分类')}</b></div>
      <p>{String(lastAdvice.guidance ?? '建议内容未记录')}</p>
      <small>{String(lastAdvice.source ?? 'predefined')} · {String(lastAdvice.recommended_action ?? '—')} · 置信度 {String(lastAdvice.confidence ?? '—')}</small>
    </section> : <div className="empty-inline">Monitor 尚未检测到需要纠偏的轨迹漂移。</div>}
    <SectionHeading index="M" title="Monitor 漂移证据" note="确定性规则，不调用模型" />
    {driftEvents.length > 0 ? <div className="table-wrap"><table><thead><tr><th>事件</th><th>漂移信号</th><th>证据序号</th><th>判断依据</th></tr></thead><tbody>
      {driftEvents.map((item, index) => {
        const signals = Array.isArray(item.signals) ? item.signals.map(record) : []
        return <tr key={`${String(item.sequence)}-${index}`}><td>{String(item.sequence ?? '—')}</td><td>{signals.map((signal) => String(signal.code ?? 'UNKNOWN')).join(' / ')}</td><td>{signals.flatMap((signal) => Array.isArray(signal.evidence_sequences) ? signal.evidence_sequences : []).join(', ') || '—'}</td><td>{signals.map((signal) => String(signal.detail ?? '')).join('；') || '—'}</td></tr>
      })}
    </tbody></table></div> : <div className="empty-inline">当前轨迹没有确定性漂移证据。</div>}
    <SectionHeading index="I" title="Advisor 与干预结果" note="建议无执行权限，阻断仅发生在动作提交前" />
    {adviceEvents.length + interventionEvents.length > 0 ? <div className="steering-ledger">
      {[...adviceEvents, ...interventionEvents].sort((left, right) => Number(left.sequence ?? 0) - Number(right.sequence ?? 0)).map((item, index) => {
        const advice = record(item.advice)
        return <article key={`${String(item.sequence)}-${index}`}><span>{String(item.sequence ?? '—')} / {item.type === 'corrective_advice_generated' ? '建议' : '干预'}</span><b>{String(item.outcome ?? advice.reason_code ?? '已记录')}</b><p>{String(advice.guidance ?? (item.outcome === 'ACTION_REJECTED' ? '候选动作与已确认的无进展动作完全相同，已在执行前阻止。' : '干预状态已持久化。'))}</p><small>{String(item.source ?? item.mode ?? 'deterministic')}</small></article>
      })}
    </div> : <div className="empty-inline">尚无 Advisor 或控制器事件。</div>}
    {Object.keys(active).length > 0 && <details><summary>查看当前未消费干预</summary><JsonBlock value={active} /></details>}
  </>
}

function ResultTab({ detail }: { detail: RunDetail }) {
  const steps = detail.result.plan?.steps ?? []
  return <>
    <SectionHeading index="A" title="最终结果" />
    <div className="result-copy">{detail.result.reply ?? '该运行尚未生成最终回复。'}</div>
    <ResultArtifactViews artifacts={detail.result.artifacts} />
    {steps.length > 0 ? <div className="table-wrap"><table><thead><tr><th>步骤</th><th>执行角色</th><th>操作</th><th>状态</th></tr></thead><tbody>
      {steps.map((step, index) => <tr key={String(step.step_id ?? index)}><td>{String(step.step_id ?? index + 1)}</td><td>{SPECIALIST_LABELS[String(step.specialist ?? '')] ?? String(step.specialist ?? '—')}</td><td>{TOOL_LABELS[String(step.tool ?? '')] ?? String(step.tool ?? '—')}</td><td>{STATUS_LABELS[String(step.status ?? '')] ?? String(step.status ?? '—')}</td></tr>)}
    </tbody></table></div> : <div className="empty-inline">该路径没有记录结构化计划</div>}
  </>
}

function EvidenceTab({ detail }: { detail: RunDetail }) {
  const checkpoint = detail.checkpoint
  const adaptive = (checkpoint.adaptive ?? {}) as Record<string, unknown>
  const bindingTransaction = (adaptive.binding_tx ?? {}) as Record<string, unknown>
  return <>
    <DefinitionGrid items={[
      ['目标修订版本', String(adaptive.goal_revision ?? 0)],
      ['约束映射版本', String(adaptive.binding_revision ?? 0)],
      ['映射事务', bindingTransaction.status ? `${String(bindingTransaction.status)} / ${String(bindingTransaction.source ?? '—')}` : '尚无事务'],
      ['模型已读证据', Array.isArray(adaptive.read_evidence_refs) ? adaptive.read_evidence_refs.length : 0],
      ['目标数量', Array.isArray(adaptive.objective_ledger) ? adaptive.objective_ledger.length : 0],
      ['上下文', checkpoint.context ? '已记录' : '未记录'],
    ]} />
    <div className="split-json"><div><h3>结构化上下文</h3><JsonBlock value={checkpoint.context ?? {}} /></div><div><h3>目标、映射事务与证据</h3><JsonBlock value={{ goal: adaptive.goal, objective_ledger: adaptive.objective_ledger ?? [], constraint_bindings: adaptive.constraint_bindings ?? {}, binding_transaction: adaptive.binding_tx ?? null, binding_resolution: adaptive.binding_resolution ?? null, binding_evidence_access: adaptive.binding_evidence_access ?? {}, working_memory: adaptive.working_memory, read_evidence_refs: adaptive.read_evidence_refs ?? [] }} /></div></div>
  </>
}

function ToolsTab({ detail }: { detail: RunDetail }) {
  const trace = detail.result.trace ?? []
  return <>
    <DefinitionGrid items={[["轨迹条目", trace.length], ["持久化账本条目", detail.tool_ledger.length]]} />
    {trace.length === 0 && detail.tool_ledger.length === 0 && <div className="empty-inline">该运行没有工具调用</div>}
    <div className="tool-grid">{trace.map((entry, index) => <article key={`${entry.tool}-${index}`}><span>{String(index + 1).padStart(2, '0')} / 工具</span><h3>{TOOL_LABELS[entry.tool ?? ''] ?? entry.tool}</h3><code>{entry.tool}</code><div><small>调用参数</small><JsonBlock value={entry.args ?? {}} /></div><div><small>执行结果</small><JsonBlock value={entry.result} /></div></article>)}</div>
    {detail.tool_ledger.length > 0 && <><SectionHeading index="L" title="持久化工具账本" /><JsonBlock value={detail.tool_ledger} /></>}
  </>
}
