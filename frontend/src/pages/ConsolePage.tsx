import { useQuery } from '@tanstack/react-query'
import { ArrowRight, FlaskConical } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import {
  Chips,
  ErrorState,
  formatTime,
  LoadingState,
  localizeEngineeringText,
  MechanismRail,
  PageHeader,
  HEALTH_LABELS,
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
  if (!production) return <ErrorState error={new Error('注册表中没有可执行配置档案')} />
  const modelState = data.model.ready ? '就绪' : '未就绪'
  const modelNote = data.model.live_verified ? '已实时验证' : data.model.configured ? '配置已加载' : '等待配置'
  const changed = candidate ? SLOT_ORDER.filter((slot) => JSON.stringify(candidate.mechanisms[slot]) !== JSON.stringify(production.mechanisms[slot])) : []

  return (
    <>
      <PageHeader code="智能体控制台 / 00" title="智能体工程控制台"
        subtitle="观察运行、比较机制、验证假设。这里展示的是当前代码和本地数据的真实状态。" marker="运行时已连接" />
      <TelemetryStrip items={[
        { label: '运行配置档案', value: data.counts.profiles, note: `${data.counts.candidate_profiles} 个候选`, tone: 'blue' },
        { label: '评测案例', value: data.counts.dataset_cases, note: `${data.counts.dataset_turns} 轮对话`, tone: 'teal' },
        { label: '运行记录', value: data.counts.runs, note: `${data.counts.events} 个事件`, tone: 'ink' },
        { label: '模型网关', value: modelState, note: modelNote, tone: data.model.ready ? 'violet' : 'amber' },
      ]} />

      <section className="console-thesis">
        <div><span>工程主张 / 01</span><h2>让智能体的每次进步<br />都能被隔离、复现与证伪。</h2></div>
        <div className="thesis-copy">
          <p>稳定内核负责状态、安全边界与恢复；运行配置档案负责装配可替换机制；实验契约只允许一个清晰变量进入比较。</p>
          <dl><dt>产品</dt><dd>稳定交付</dd><dt>研究</dt><dd>受控演化</dd></dl>
        </div>
      </section>

      <div className="console-columns">
        <div>
          <SectionHeading index="01" title="当前自主链路" note="配置档案指纹" />
          <section className="panel profile-panel">
            <div className="panel-lead"><div><span>生产基线</span><h3>{localizeEngineeringText(production.display_name)}</h3></div><code>摘要 {production.execution_sha256.slice(0, 12)}</code></div>
            <p className="muted">{localizeEngineeringText(production.description)}</p>
            <MechanismRail profile={production} />
            <Chips values={production.capabilities} />
          </section>

          <SectionHeading index="02" title="最近执行" note="最新运行档案" />
          <section className="run-list">
            {data.runs.length === 0 && <div className="empty-inline">还没有可检查的运行，请从业务沙盒执行一个任务。</div>}
            {data.runs.slice(0, 5).map((run) => (
              <button type="button" className="run-row" key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <span className="run-id">{run.run_id.slice(0, 10)}</span><StatusBadge status={run.status} />
                <strong>{run.request}</strong>
                <small>{formatTime(run.created_at)} · {run.profile.profile_id} · {run.event_count} 个事件 · {run.tool_calls} 次工具调用</small>
                <ArrowRight size={15} className="row-arrow" />
              </button>
            ))}
          </section>
          <Link className="text-link" to="/runs">打开完整运行检查器 <ArrowRight size={14} /></Link>
        </div>

        <div>
          <SectionHeading index="03" title="实验队列" note="每次只比较一个变量" />
          <section className="panel candidate-panel">
            {candidate ? <>
              <span className="candidate-kicker">候选配置 · {candidate.profile_id}</span>
              <h3>{localizeEngineeringText(candidate.display_name)}</h3>
              <p>{candidate.research.hypothesis ? localizeEngineeringText(candidate.research.hypothesis) : '未声明实验假设'}</p>
              <div className="change-callout"><span>已变更槽位</span><b>{changed.map((slot) => SLOT_LABELS[slot]).join(' / ') || '跨配置档案对照'}</b></div>
              <Link className="text-link" to="/experiments"><FlaskConical size={14} />查看实验契约</Link>
            </> : <div className="empty-inline">注册表中没有候选配置档案</div>}
          </section>

          <SectionHeading index="04" title="模型接入" note="脱敏状态" />
          <section className="panel gateway-panel">
            <div className={`gateway-state ${data.model.ready ? 'ok' : 'warn'}`}><i /><span>模型网关</span><b>{modelState}</b></div>
            <p className="muted">{localizeEngineeringText(data.model.message)}</p>
            {data.model.providers.length > 0 ? data.model.providers.map((provider) => (
              <div className="provider-row" key={`${provider.provider}-${provider.model}-${provider.slot}`}>
                <code>{provider.provider} / {provider.model}</code><b>{HEALTH_LABELS[provider.health] ?? provider.health}</b>
              </div>
            )) : <div className="empty-inline">尚未配置模型服务商</div>}
          </section>
        </div>
      </div>

      <SectionHeading index="05" title="稳定内核边界" note="变更前必须完成兼容性审查" />
      <div className="kernel-grid">{data.kernel.map((boundary) => (
        <article key={boundary.boundary_id}><code>{boundary.boundary_id}</code><h3>{localizeEngineeringText(boundary.label)}</h3><p>{boundary.paths.join('\n')}</p></article>
      ))}</div>
    </>
  )
}
