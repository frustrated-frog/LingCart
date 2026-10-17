import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { ErrorState, JsonBlock, LoadingState, MechanismRail, PageHeader, SectionHeading, TelemetryStrip } from '../components/SystemUI'

export function ExperimentsPage() {
  const overview = useQuery({ queryKey: ['engineering-overview'], queryFn: api.overview })
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: api.experiments })
  const [experimentId, setExperimentId] = useState('')
  const list = experiments.data?.experiments ?? []
  useEffect(() => { if (list.length && !list.some((item) => item.experiment_id === experimentId)) setExperimentId(list[0].experiment_id) }, [experimentId, list])
  if (overview.isLoading || experiments.isLoading) return <LoadingState label="正在读取实验契约" />
  if (overview.isError || experiments.isError) return <ErrorState error={overview.error ?? experiments.error} retry={() => { void overview.refetch(); void experiments.refetch() }} />
  const experiment = list.find((item) => item.experiment_id === experimentId)
  if (!experiment) return <ErrorState error={new Error('尚无结构化实验契约')} />
  const baseline = overview.data?.profiles.find((profile) => profile.profile_id === experiment.baseline_profile)
  const candidate = overview.data?.profiles.find((profile) => profile.profile_id === experiment.candidate_profile)
  if (!baseline || !candidate) return <ErrorState error={new Error('实验引用了 Registry 中不存在的 Profile')} />
  const ready = experiment.status === 'COMPLETED'

  return <>
    <PageHeader code="EXPERIMENTS / 04" title="实验契约"
      subtitle="先冻结基线、变量和指标，再执行候选。没有可比报告时，界面明确保持“不能下结论”。" marker="RESEARCH MODE" />
    <label className="standalone-select">实验<select value={experimentId} onChange={(e) => setExperimentId(e.target.value)}>{list.map((item) => <option key={item.experiment_id}>{item.experiment_id}</option>)}</select></label>
    <TelemetryStrip items={[
      { label: 'CONTRACT', value: 'VALID', note: experiment.path, tone: 'teal' },
      { label: 'VARIABLES', value: experiment.changes.length, note: '契约声明的变化', tone: 'violet' },
      { label: 'TEST REPEATS', value: experiment.repeats, note: `${experiment.evaluation_split} split`, tone: 'blue' },
      { label: 'CONCLUSION', value: ready ? 'REPORT READY' : 'NO COMPARABLE REPORT', note: ready ? '可以下结论' : '禁止声称增益', tone: ready ? 'teal' : 'amber' },
    ]} />
    <div className="experiment-pipeline">{[
      ['01', 'FREEZE', '冻结 Profile 与数据集', 'done'], ['02', 'DECLARE', '声明唯一变化与指标', 'done'],
      ['03', 'RUN', 'dev 调试 / test 结论', 'current'], ['04', 'GATE', '正确率、安全、成本', ''],
    ].map(([index, title, note, state], i) => <div className="pipeline-fragment" key={title}><div className={state}><span>{index}</span><b>{title}</b><small>{note}</small></div>{i < 3 && <i />}</div>)}</div>
    <SectionHeading index="01" title="实验变量" note="BASELINE → CANDIDATE" />
    <div className="comparison-grid">
      <section className="panel"><span className="comparison-label">BASELINE</span><h3>{baseline.display_name}</h3><MechanismRail profile={baseline} /></section>
      <section className="panel"><span className="comparison-label candidate">CANDIDATE</span><h3>{candidate.display_name}</h3><MechanismRail profile={candidate} reference={baseline} /></section>
    </div>
    <div className="hypothesis"><span>HYPOTHESIS</span><p>{candidate.research.hypothesis ?? '未声明'}</p></div>
    <div className="experiment-detail-grid">
      <div><SectionHeading index="02" title="变更白名单" />{experiment.changes.map((change) => <article className="panel change-card" key={`${change.kind}-${change.slot}`}><dl><dt>KIND</dt><dd>{change.kind}</dd><dt>SLOT</dt><dd>{change.slot ?? '—'}</dd><dt>IMPLEMENTATION</dt><dd>{change.implementation_id ?? '—'}</dd></dl><pre>{change.paths.join('\n')}</pre></article>)}</div>
      <div><SectionHeading index="03" title="Gate 指标" /><div className="table-wrap"><table><thead><tr><th>KIND</th><th>METRIC</th><th>OBJECTIVE</th><th>MAX REGRESSION</th><th>MIN IMPROVEMENT</th></tr></thead><tbody>
        {[...experiment.primary_metrics.map((rule) => ({ kind: 'PRIMARY', ...rule })), ...experiment.guard_metrics.map((rule) => ({ kind: 'GUARD', ...rule }))].map((rule) => <tr key={rule.name}><td>{rule.kind}</td><td><code>{rule.name}</code></td><td>{rule.objective}</td><td>{rule.max_regression}</td><td>{rule.min_improvement}</td></tr>)}
      </tbody></table></div></div>
    </div>
    {ready ? <><SectionHeading index="04" title="实验报告" note={experiment.report_path ?? ''} /><JsonBlock value={experiment.report?.metrics ?? {}} /></>
      : <div className="warning-state"><b>NO COMPARABLE REPORT</b><p>当前仓库没有同时绑定该实验契约且通过 Gate 的候选报告，因此不能声称候选机制更好。</p><small>下一步应分别在 test split 上运行 baseline 与 candidate，保持模型配置、依赖锁、重复次数和业务环境一致。</small></div>}
  </>
}
