import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import {
  Chips,
  DefinitionGrid,
  ErrorState,
  JsonBlock,
  LoadingState,
  localizeEngineeringText,
  MECHANISM_LABELS,
  MechanismRail,
  PageHeader,
  RESEARCH_STATUS_LABELS,
  SectionHeading,
  SLOT_LABELS,
  SLOT_ORDER,
  SOURCE_TYPE_LABELS,
} from '../components/SystemUI'

type View = 'research' | 'capabilities' | 'source' | 'all'

export function MechanismsPage() {
  const query = useQuery({ queryKey: ['engineering-overview'], queryFn: api.overview })
  const [referenceId, setReferenceId] = useState('adaptive-v1')
  const [comparisonId, setComparisonId] = useState('adaptive-objective-context-v1')
  const [view, setView] = useState<View>('research')
  const profiles = query.data?.profiles ?? []
  useEffect(() => {
    if (profiles.length && !profiles.some((item) => item.profile_id === referenceId)) setReferenceId(profiles[0].profile_id)
    if (profiles.length && !profiles.some((item) => item.profile_id === comparisonId)) setComparisonId(profiles[Math.min(1, profiles.length - 1)].profile_id)
  }, [comparisonId, profiles, referenceId])
  if (query.isLoading) return <LoadingState label="正在读取运行机制注册表" />
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />
  const reference = profiles.find((item) => item.profile_id === referenceId)
  const comparison = profiles.find((item) => item.profile_id === comparisonId)
  if (!reference || !comparison) return <ErrorState error={new Error('注册表中没有可比较的配置档案')} />

  return <>
    <PageHeader code="机制注册表 / 02" title="机制注册表"
      subtitle="配置档案是可执行配置，不是标签。比较槽位、源码指纹和研究状态，确认究竟改变了什么。" />
    <div className="filter-row two">
      <label>参考配置档案<select value={referenceId} onChange={(e) => setReferenceId(e.target.value)}>{profiles.map((item) => <option key={item.profile_id}>{item.profile_id}</option>)}</select></label>
      <label>对照配置档案<select value={comparisonId} onChange={(e) => setComparisonId(e.target.value)}>{profiles.map((item) => <option key={item.profile_id}>{item.profile_id}</option>)}</select></label>
    </div>
    <SectionHeading index="01" title="机制指纹对照" note="已变更槽位会突出显示" />
    <div className="comparison-grid">
      <section className="panel"><span className="comparison-label">参考配置</span><h3>{localizeEngineeringText(reference.display_name)}</h3><MechanismRail profile={reference} /></section>
      <section className="panel"><span className="comparison-label candidate">对照配置</span><h3>{localizeEngineeringText(comparison.display_name)}</h3><MechanismRail profile={comparison} reference={reference} /></section>
    </div>
    <div className="table-wrap"><table><thead><tr><th>槽位标识</th><th>槽位</th><th>参考配置</th><th>对照配置</th><th>差异</th></tr></thead><tbody>
      {SLOT_ORDER.map((slot) => {
        const before = reference.mechanisms[slot]
        const after = comparison.mechanisms[slot]
        const changed = JSON.stringify(before) !== JSON.stringify(after)
        return <tr className={changed ? 'changed-row' : ''} key={slot}><td><code>{slot}</code></td><td>{SLOT_LABELS[slot]}</td><td>{before ? `${MECHANISM_LABELS[before.id] ?? before.id}（${before.id}@${before.version}）` : '未绑定'}</td><td>{after ? `${MECHANISM_LABELS[after.id] ?? after.id}（${after.id}@${after.version}）` : '未绑定'}</td><td>{changed ? '已变更' : '未变更'}</td></tr>
      })}
    </tbody></table></div>
    <div className="tab-list" role="tablist">{([
      ['research', '研究声明'], ['capabilities', '能力边界'], ['source', '可复现来源'], ['all', '全部配置档案'],
    ] as Array<[View, string]>).map(([id, label]) => <button role="tab" aria-selected={view === id} className={view === id ? 'active' : ''} onClick={() => setView(id)} key={id}>{label}</button>)}</div>
    <section className="tab-panel">
      {view === 'research' && <div className="comparison-grid">
        {([['参考配置', reference], ['对照配置', comparison]] as const).map(([label, profile]) => <article className="panel" key={label}>
          <DefinitionGrid items={[[String(label), profile.profile_id], ['研究状态', RESEARCH_STATUS_LABELS[profile.research.status] ?? profile.research.status], ['实现来源', SOURCE_TYPE_LABELS[profile.research.source_type] ?? profile.research.source_type], ['论文编号', profile.research.paper_id ?? '原生实现']]} />
          <p>{profile.research.hypothesis ? localizeEngineeringText(profile.research.hypothesis) : '基线配置不主张实验增益。'}</p>
        </article>)}</div>}
      {view === 'capabilities' && <div className="comparison-grid"><div><h3>参考配置能力</h3><Chips values={reference.capabilities} /></div><div><h3>对照配置能力</h3><Chips values={comparison.capabilities} /></div></div>}
      {view === 'source' && <div className="comparison-grid">{[reference, comparison].map((profile) => <div key={profile.profile_id}><h3>{profile.profile_id}</h3><JsonBlock value={{ profile_sha256: profile.fingerprint, execution_sha256: profile.execution_sha256, source_revision: profile.source_revision, mechanisms: profile.resolved_mechanisms }} /></div>)}</div>}
      {view === 'all' && <div className="profile-stack">{profiles.map((profile) => <article className="panel" key={profile.profile_id}><div className="panel-lead"><div><code>{profile.profile_id}</code><h3>{localizeEngineeringText(profile.display_name)}</h3></div><span className="comparison-label">{RESEARCH_STATUS_LABELS[profile.research.status] ?? profile.research.status}</span></div><p className="muted">{localizeEngineeringText(profile.description)}</p><MechanismRail profile={profile} /></article>)}</div>}
    </section>
  </>
}
