import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { CatalogResultViews } from '../components/CatalogResultView'
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
  SectionHeading,
  SPECIALIST_LABELS,
  STATUS_LABELS,
  TERMINAL_REASON_LABELS,
  TelemetryStrip,
  TOOL_LABELS,
} from '../components/SystemUI'

type Tab = 'result' | 'evidence' | 'tools' | 'raw'

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
        {detail && <>
          <TelemetryStrip items={[
            { label: '运行状态', value: STATUS_LABELS[detail.status] ?? detail.status, note: detail.terminal_reason ? (TERMINAL_REASON_LABELS[detail.terminal_reason] ?? detail.terminal_reason) : '尚未终止', tone: 'teal' },
            { label: '运行配置档案', value: detail.profile.profile_id, note: detail.profile.legacy_profile_reference ? '历史运行未记录' : '已记录', tone: 'violet' },
            { label: '工具 / 模型', value: `${detail.tool_calls} / ${detail.model_calls}`, note: `共 ${detail.event_count} 个事件`, tone: 'blue' },
            { label: '令牌用量', value: detail.tokens ?? '未报告', note: '模型服务商未报告时不伪造', tone: 'ink' },
          ]} />
          <SectionHeading index="01" title="执行主脊" note={`${detail.run_id.slice(0, 12)} / ${formatTime(detail.created_at)}`} />
          <div className="run-prompt">{detail.request}</div>
          <ExecutionSpine events={events} selected={selectedEvent} onSelect={setSelectedEvent} />
          {event && <section className="event-probe">
            <div className="probe-title"><span>事件探针</span><b>{String(event.sequence ?? selectedEvent + 1).padStart(2, '0')} · {eventLabel(event).join(' / ')}</b></div>
            <DefinitionGrid items={[
              ['事件类型', eventLabel(event)[1]], ['事件语义', eventLabel(event).join(' / ')],
              ['发生时间', event.timestamp ?? '—'], ['事件序号', event.sequence ?? selectedEvent + 1],
            ]} />
            <details><summary>查看原始事件数据</summary><JsonBlock value={event} /></details>
          </section>}
          <div className="tab-list" role="tablist">
            {([['result', '结果与计划'], ['evidence', '上下文与证据'], ['tools', '工具账本'], ['raw', '原始记录']] as Array<[Tab, string]>).map(([id, label]) => (
              <button role="tab" aria-selected={tab === id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)} key={id}>{label}</button>
            ))}
          </div>
          <section className="tab-panel">
            {tab === 'result' && <ResultTab detail={detail} />}
            {tab === 'evidence' && <EvidenceTab detail={detail} />}
            {tab === 'tools' && <ToolsTab detail={detail} />}
            {tab === 'raw' && <JsonBlock value={{ run_id: detail.run_id, harness: detail.profile, events, budget: detail.result.budget, usage: detail.result.usage, commands: detail.commands }} />}
          </section>
        </>}
      </>}
    </>
  )
}

function ResultTab({ detail }: { detail: RunDetail }) {
  const steps = detail.result.plan?.steps ?? []
  return <>
    <SectionHeading index="A" title="最终结果" />
    <div className="result-copy">{detail.result.reply ?? '该运行尚未生成最终回复。'}</div>
    <CatalogResultViews artifacts={detail.result.artifacts} />
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
