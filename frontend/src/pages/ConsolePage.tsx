import { useQuery } from '@tanstack/react-query'
import { ArrowRight, FlaskConical } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import {
  Chips,
  ErrorState,
  formatTime,
  LoadingState,
  MechanismRail,
  PageHeader,
  SectionHeading,
  SLOT_LABELS,
  SLOT_ORDER,
  StatusBadge,
  TelemetryStrip,
} from '../components/SystemUI'

export function ConsolePage() {
  const navigate = useNavigate()
  const overview = useQuery({ queryKey: ['engineering-overview'], queryFn: api.overview, refetchInterval: 10_000 })
  if (overview.isLoading) return <LoadingState />
  if (overview.isError || !overview.data) return <ErrorState error={overview.error} retry={() => void overview.refetch()} />

  const data = overview.data
  const production = data.profiles.find((profile) => profile.profile_id === 'adaptive-v1') ?? data.profiles[0]
  const candidate = data.profiles.find((profile) => profile.research.status === 'candidate')
  if (!production) return <ErrorState error={new Error('Registry 中没有可执行 Profile')} />
  const modelState = data.model.ready ? 'READY' : 'NOT READY'
  const modelNote = data.model.live_verified ? '已实时验证' : data.model.configured ? '配置已加载' : '等待配置'
  const changed = candidate ? SLOT_ORDER.filter((slot) => JSON.stringify(candidate.mechanisms[slot]) !== JSON.stringify(production.mechanisms[slot])) : []

  return (
    <>
      <PageHeader code="AGENT CONSOLE / 00" title="Agent 工程控制台"
        subtitle="观察运行、比较机制、验证假设。这里展示的是当前代码和本地数据的真实状态。" marker="RUNTIME CONNECTED" />
      <TelemetryStrip items={[
        { label: 'HARNESS PROFILES', value: data.counts.profiles, note: `${data.counts.candidate_profiles} 个候选`, tone: 'blue' },
        { label: 'EVALUATION CASES', value: data.counts.dataset_cases, note: `${data.counts.dataset_turns} turns`, tone: 'teal' },
        { label: 'RECORDED RUNS', value: data.counts.runs, note: `${data.counts.events} events`, tone: 'ink' },
        { label: 'MODEL GATEWAY', value: modelState, note: modelNote, tone: data.model.ready ? 'violet' : 'amber' },
      ]} />

      <section className="console-thesis">
        <div><span>THESIS / 01</span><h2>让 Agent 的每次进步<br />都能被隔离、复现与证伪。</h2></div>
        <div className="thesis-copy">
          <p>稳定内核负责状态、安全边界与恢复；Harness Profile 负责装配可替换机制；实验契约只允许一个清晰变量进入比较。</p>
          <dl><dt>PRODUCT</dt><dd>稳定交付</dd><dt>RESEARCH</dt><dd>受控演化</dd></dl>
        </div>
      </section>

      <div className="console-columns">
        <div>
          <SectionHeading index="01" title="当前自主链路" note="PROFILE FINGERPRINT" />
          <section className="panel profile-panel">
            <div className="panel-lead"><div><span>PRODUCTION BASELINE</span><h3>{production.display_name}</h3></div><code>SHA {production.execution_sha256.slice(0, 12)}</code></div>
            <p className="muted">{production.description}</p>
            <MechanismRail profile={production} />
            <Chips values={production.capabilities} />
          </section>

          <SectionHeading index="02" title="最近执行" note="LATEST RUN ARCHIVE" />
          <section className="run-list">
            {data.runs.length === 0 && <div className="empty-inline">还没有可检查的运行，请从业务沙盒执行一个任务。</div>}
            {data.runs.slice(0, 5).map((run) => (
              <button type="button" className="run-row" key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <span className="run-id">{run.run_id.slice(0, 10)}</span><StatusBadge status={run.status} />
                <strong>{run.request}</strong>
                <small>{formatTime(run.created_at)} · {run.profile.profile_id} · {run.event_count} events · {run.tool_calls} tools</small>
                <ArrowRight size={15} className="row-arrow" />
              </button>
            ))}
          </section>
          <Link className="text-link" to="/runs">打开完整 Run Inspector <ArrowRight size={14} /></Link>
        </div>

        <div>
          <SectionHeading index="03" title="实验队列" note="ONE VARIABLE AT A TIME" />
          <section className="panel candidate-panel">
            {candidate ? <>
              <span className="candidate-kicker">CANDIDATE · {candidate.profile_id}</span>
              <h3>{candidate.display_name}</h3>
              <p>{candidate.research.hypothesis}</p>
              <div className="change-callout"><span>CHANGED SLOT</span><b>{changed.map((slot) => SLOT_LABELS[slot]).join(' / ') || '跨 Profile 对照'}</b></div>
              <Link className="text-link" to="/experiments"><FlaskConical size={14} />查看实验契约</Link>
            </> : <div className="empty-inline">Registry 中没有候选 Profile</div>}
          </section>

          <SectionHeading index="04" title="模型接入" note="SANITIZED STATUS" />
          <section className="panel gateway-panel">
            <div className={`gateway-state ${data.model.ready ? 'ok' : 'warn'}`}><i /><span>MODEL GATEWAY</span><b>{modelState}</b></div>
            <p className="muted">{data.model.message}</p>
            {data.model.providers.length > 0 ? data.model.providers.map((provider) => (
              <div className="provider-row" key={`${provider.provider}-${provider.model}-${provider.slot}`}>
                <code>{provider.provider} / {provider.model}</code><b>{provider.health}</b>
              </div>
            )) : <div className="empty-inline">NO PROVIDER CONFIGURED</div>}
          </section>
        </div>
      </div>

      <SectionHeading index="05" title="稳定内核边界" note="CHANGES REQUIRE COMPATIBILITY REVIEW" />
      <div className="kernel-grid">{data.kernel.map((boundary) => (
        <article key={boundary.boundary_id}><code>{boundary.boundary_id}</code><h3>{boundary.label}</h3><p>{boundary.paths.join('\n')}</p></article>
      ))}</div>
    </>
  )
}
