import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
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
  STATUS_LABELS,
  TelemetryStrip,
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
      <PageHeader code={businessMode ? 'TASK ARCHIVE / 01' : 'LIVE RUNS / 01'}
        title={businessMode ? '任务记录' : '运行检查'}
        subtitle={businessMode ? '查看每次业务处理的真实结果，并可进入工程视图检查执行证据。' : '沿执行主脊检查一次任务：配置身份、事件顺序、证据状态、工具调用和终止原因。'} />
      {runs.length === 0 ? <div className="empty-state"><b>尚无运行记录</b><p>请先在业务沙盒执行一个消费任务。</p></div> : <>
        <div className="filter-row run-filters">
          <label>状态<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="ALL">全部状态</option>{statuses.map((value) => <option key={value} value={value}>{STATUS_LABELS[value] ?? value}</option>)}</select></label>
          <label>Profile<select value={profile} onChange={(e) => setProfile(e.target.value)}><option value="ALL">全部 Profile</option>{profiles.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>运行<select value={runId ?? ''} onChange={(e) => navigate(`${businessMode ? '/sandbox/tasks' : '/runs'}/${e.target.value}`)}>
            {filtered.map((run) => <option value={run.run_id} key={run.run_id}>{formatTime(run.created_at)} · {run.request}</option>)}
          </select></label>
        </div>
        {detailQuery.isLoading && <LoadingState label="正在装载运行证据" />}
        {detailQuery.isError && <ErrorState error={detailQuery.error} retry={() => void detailQuery.refetch()} />}
        {detail && <>
          <TelemetryStrip items={[
            { label: 'RUN STATUS', value: STATUS_LABELS[detail.status] ?? detail.status, note: detail.terminal_reason ?? '尚未终止', tone: 'teal' },
            { label: 'HARNESS PROFILE', value: detail.profile.profile_id, note: detail.profile.legacy_profile_reference ? '历史运行未记录' : '已记录', tone: 'violet' },
            { label: 'TOOL / MODEL', value: `${detail.tool_calls} / ${detail.model_calls}`, note: `${detail.event_count} total events`, tone: 'blue' },
            { label: 'TOKEN USAGE', value: detail.tokens ?? 'N/A', note: 'Provider 未报告时不伪造', tone: 'ink' },
          ]} />
          <SectionHeading index="01" title="Execution Spine" note={`${detail.run_id.slice(0, 12)} / ${formatTime(detail.created_at)}`} />
          <div className="run-prompt">{detail.request}</div>
          <ExecutionSpine events={events} selected={selectedEvent} onSelect={setSelectedEvent} />
          {event && <section className="event-probe">
            <div className="probe-title"><span>事件探针</span><b>{String(event.sequence ?? selectedEvent + 1).padStart(2, '0')} · {eventLabel(event).join(' / ')}</b></div>
            <DefinitionGrid items={[
              ['EVENT TYPE', event.type], ['SEMANTIC', eventLabel(event).join(' / ')],
              ['TIMESTAMP', event.timestamp ?? '—'], ['SEQUENCE', event.sequence ?? selectedEvent + 1],
            ]} />
            <JsonBlock value={event} />
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
    {steps.length > 0 ? <div className="table-wrap"><table><thead><tr><th>STEP</th><th>SPECIALIST</th><th>OPERATION</th><th>STATUS</th></tr></thead><tbody>
      {steps.map((step, index) => <tr key={String(step.step_id ?? index)}><td>{String(step.step_id ?? index + 1)}</td><td>{String(step.specialist ?? '—')}</td><td>{String(step.tool ?? '—')}</td><td>{String(step.status ?? '—')}</td></tr>)}
    </tbody></table></div> : <div className="empty-inline">该路径没有记录结构化计划</div>}
  </>
}

function EvidenceTab({ detail }: { detail: RunDetail }) {
  const checkpoint = detail.checkpoint
  const adaptive = (checkpoint.adaptive ?? {}) as Record<string, unknown>
  return <>
    <DefinitionGrid items={[
      ['GOAL REVISION', String(adaptive.binding_revision ?? 0)],
      ['EVIDENCE READ', Array.isArray(adaptive.read_evidence_refs) ? adaptive.read_evidence_refs.length : 0],
      ['OBJECTIVES', Array.isArray(adaptive.objective_ledger) ? adaptive.objective_ledger.length : 0],
      ['CONTEXT', checkpoint.context ? 'RECORDED' : 'NOT RECORDED'],
    ]} />
    <div className="split-json"><div><h3>结构化上下文</h3><JsonBlock value={checkpoint.context ?? {}} /></div><div><h3>目标、约束与证据</h3><JsonBlock value={{ goal: adaptive.goal, objective_ledger: adaptive.objective_ledger ?? [], constraint_bindings: adaptive.constraint_bindings ?? {}, working_memory: adaptive.working_memory, read_evidence_refs: adaptive.read_evidence_refs ?? [] }} /></div></div>
  </>
}

function ToolsTab({ detail }: { detail: RunDetail }) {
  const trace = detail.result.trace ?? []
  return <>
    <DefinitionGrid items={[["TRACE ENTRIES", trace.length], ["DURABLE LEDGER", detail.tool_ledger.length]]} />
    {trace.length === 0 && detail.tool_ledger.length === 0 && <div className="empty-inline">该运行没有工具调用</div>}
    <div className="tool-grid">{trace.map((entry, index) => <article key={`${entry.tool}-${index}`}><span>{String(index + 1).padStart(2, '0')} / TOOL</span><h3>{entry.tool}</h3><div><small>ARGS</small><JsonBlock value={entry.args ?? {}} /></div><div><small>RESULT</small><JsonBlock value={entry.result} /></div></article>)}</div>
    {detail.tool_ledger.length > 0 && <><SectionHeading index="L" title="持久化 Tool Ledger" /><JsonBlock value={detail.tool_ledger} /></>}
  </>
}
