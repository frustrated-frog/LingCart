import { useQuery } from '@tanstack/react-query'
import * as m from 'motion/react-m'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import {
  ErrorState,
  JsonBlock,
  localizeEngineeringText,
  LoadingState,
  MechanismRail,
  METRIC_LABELS,
  OBJECTIVE_LABELS,
  PageHeader,
  SectionHeading,
  SLOT_LABELS,
  SPLIT_LABELS,
  TelemetryStrip,
  ViewTransition,
} from '../components/SystemUI'

const REPORT_MISMATCH_LABELS: Record<string, string> = {
  experiment_spec: '实验契约', dataset_path: '数据集路径', dataset_sha256: '数据集指纹',
  suite: '评测套件', split: '数据分组', repeats: '重复次数', source_tree_sha256: '源码指纹',
  evaluation_protocol_sha256: '评测协议指纹', business_protocol_sha256: '业务协议指纹',
  dependency_lock_sha256: '依赖锁指纹', purpose_models: '用途模型配置',
  'harness_profile.profile_id': '配置档案编号', 'harness_profile.profile_sha256': '配置档案指纹',
  'harness_profile.execution_sha256': '执行指纹',
}

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
  if (!baseline || !candidate) return <ErrorState error={new Error('实验引用了注册表中不存在的配置档案')} />
  const ready = experiment.status === 'CURRENT_REPRODUCIBLE'
  const historical = experiment.status === 'HISTORICAL_PASSED'
  const rejected = experiment.status === 'CURRENT_REJECTED' || experiment.status === 'HISTORICAL_REJECTED'
  const historicalRejected = experiment.status === 'HISTORICAL_REJECTED'

  return <>
    <PageHeader code="实验契约 / 04" title="实验契约"
      subtitle="先冻结基线、变量和指标，再执行候选。没有可比报告时，界面明确保持“不能下结论”。" marker="研究模式" />
    <label className="standalone-select">实验<select value={experimentId} onChange={(e) => setExperimentId(e.target.value)}>{list.map((item) => <option key={item.experiment_id}>{item.experiment_id}</option>)}</select></label>
    <ViewTransition viewKey={experiment.experiment_id}>
    <TelemetryStrip items={[
      { label: '契约状态', value: '有效', note: experiment.path, tone: 'teal' },
      { label: '实验变量', value: experiment.changes.length, note: '契约声明的变化', tone: 'violet' },
      { label: '测试重复次数', value: experiment.repeats, note: SPLIT_LABELS[experiment.evaluation_split] ?? experiment.evaluation_split, tone: 'blue' },
      { label: '实验结论', value: ready ? '当前版本可复现' : rejected ? '评测门禁未通过' : historical ? '仅供历史参考' : '暂无可比报告', note: ready ? '当前版本可复现' : rejected ? (historicalRejected ? '历史版本未通过' : '当前版本未通过') : historical ? '实现或环境已变化' : '禁止声称增益', tone: ready ? 'teal' : 'amber' },
    ]} />
    <m.div className="experiment-pipeline" initial="hidden" animate="visible"
      variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.065 } } }}>{[
      ['01', '冻结', '冻结配置档案与数据集', 'done'], ['02', '声明', '声明唯一变化与指标', 'done'],
      ['03', '运行', '开发集调试 / 测试集结论', 'current'], ['04', '门禁', '正确率、安全、成本', ''],
    ].map(([index, title, note, state], i) => <m.div className="pipeline-fragment" key={title}
      variants={{ hidden: { opacity: 0, x: -9 }, visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 430, damping: 38 } } }}>
      <div className={state}><span>{index}</span><b>{title}</b><small>{note}</small></div>{i < 3 && <i />}
    </m.div>)}</m.div>
    <SectionHeading index="01" title="实验变量" note="基线 → 候选" />
    <div className="comparison-grid">
      <section className="panel"><span className="comparison-label">基线配置</span><h3>{localizeEngineeringText(baseline.display_name)}</h3><MechanismRail profile={baseline} /></section>
      <section className="panel"><span className="comparison-label candidate">候选配置</span><h3>{localizeEngineeringText(candidate.display_name)}</h3><MechanismRail profile={candidate} reference={baseline} /></section>
    </div>
    <div className="hypothesis"><span>实验假设</span><p>{candidate.research.hypothesis ? localizeEngineeringText(candidate.research.hypothesis) : '未声明'}</p></div>
    <div className="experiment-detail-grid">
      <div><SectionHeading index="02" title="变更白名单" />{experiment.changes.map((change) => <article className="panel change-card" key={`${change.kind}-${change.slot}`}><dl><dt>变更类型</dt><dd>{change.kind === 'mechanism' ? '机制变更' : change.kind}</dd><dt>变更槽位</dt><dd>{change.slot ? (SLOT_LABELS[change.slot] ?? change.slot) : '—'}</dd><dt>实现标识</dt><dd>{change.implementation_id ?? '—'}</dd></dl><pre>{change.paths.join('\n')}</pre></article>)}</div>
      <div><SectionHeading index="03" title="评测门禁指标" /><div className="table-wrap"><table><thead><tr><th>指标类型</th><th>指标</th><th>优化方向</th><th>最大允许回退</th><th>最低改善要求</th></tr></thead><tbody>
        {[...experiment.primary_metrics.map((rule) => ({ kind: '核心指标', ...rule })), ...experiment.guard_metrics.map((rule) => ({ kind: '保护指标', ...rule }))].map((rule) => <tr key={rule.name}><td>{rule.kind}</td><td><span>{METRIC_LABELS[rule.name] ?? rule.name}</span><br /><code>{rule.name}</code></td><td>{OBJECTIVE_LABELS[rule.objective] ?? rule.objective}</td><td>{rule.max_regression}</td><td>{rule.min_improvement}</td></tr>)}
      </tbody></table></div></div>
    </div>
    {ready ? <><SectionHeading index="04" title="当前可复现实验报告" note={experiment.report_path ?? ''} /><JsonBlock value={experiment.report?.metrics ?? {}} /></>
      : rejected ? <><div className="warning-state"><b>{historicalRejected ? '历史报告 · 评测门禁未通过' : '当前报告 · 评测门禁未通过'}</b><p>候选报告已完成，但没有通过实验契约。失败报告仍保留并展示，不能声称候选机制更好。</p>{historicalRejected && <code>{experiment.report_mismatches.map((item) => REPORT_MISMATCH_LABELS[item] ?? item).join(' · ')}</code>}<small>{experiment.report?.gate?.failures.join('；') || '评测门禁未提供失败原因'}</small></div><SectionHeading index="04" title="被拒绝实验的指标" note={experiment.report_path ?? ''} /><JsonBlock value={experiment.report?.metrics ?? {}} /></>
      : <div className="warning-state"><b>{historical ? '历史报告 · 当前版本尚未验证' : '暂无可比报告'}</b><p>{historical ? '该实验曾有通过评测门禁的报告，但源码、配置档案、模型配置、数据集或评测协议已变化，不能把历史结论当作当前版本结论。' : '当前仓库没有同时绑定该实验契约且完成评测门禁的候选报告，因此不能声称候选机制更好。'}</p>{historical && <code>{experiment.report_mismatches.map((item) => REPORT_MISMATCH_LABELS[item] ?? item).join(' · ')}</code>}<small>下一步应分别在测试集上运行基线与候选配置，保持模型配置、依赖锁、重复次数和业务环境一致。</small></div>}
    </ViewTransition>
  </>
}
