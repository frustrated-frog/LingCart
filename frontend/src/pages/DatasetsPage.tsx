import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import {
  CoverageBar,
  DefinitionGrid,
  ErrorState,
  EXECUTION_SOURCE_LABELS,
  JsonBlock,
  LANGUAGE_LABELS,
  LoadingState,
  localizeEngineeringText,
  PageHeader,
  SectionHeading,
  SPLIT_LABELS,
  STATUS_LABELS,
  SUITE_LABELS,
  TelemetryStrip,
  TOOL_LABELS,
  ViewTransition,
} from '../components/SystemUI'

export function DatasetsPage() {
  const summaries = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
  const [datasetId, setDatasetId] = useState('offline-core-v1')
  const [caseId, setCaseId] = useState('')
  const detail = useQuery({ queryKey: ['dataset', datasetId], queryFn: () => api.dataset(datasetId), enabled: Boolean(datasetId) })
  const datasets = summaries.data?.datasets ?? []
  useEffect(() => { if (datasets.length && !datasets.some((item) => item.dataset_id === datasetId)) setDatasetId(datasets[0].dataset_id) }, [datasetId, datasets])
  const cases = useMemo(() => detail.data?.dataset.cases ?? [], [detail.data])
  useEffect(() => { if (cases.length && !cases.some((item) => item.id === caseId)) setCaseId(cases[0].id) }, [caseId, cases])
  if (summaries.isLoading) return <LoadingState label="正在读取评测任务集" />
  if (summaries.isError) return <ErrorState error={summaries.error} retry={() => void summaries.refetch()} />
  const dataset = detail.data?.dataset
  const labels = detail.data?.category_labels ?? summaries.data?.category_labels ?? {}
  const selectedCase = cases.find((item) => item.id === caseId)
  const maximum = Math.max(1, ...Object.values(dataset?.categories ?? {}))

  return <>
    <PageHeader code="数据集覆盖 / 03" title="数据集覆盖"
      subtitle="从任务类别、数据分组和断言深度观察评测底座，避免用少量顺利案例代替能力证据。" />
    <TelemetryStrip items={[
      { label: '数据集', value: datasets.length, note: '固定 / 真实模型 / 自主专项', tone: 'blue' },
      { label: '案例总数', value: datasets.reduce((sum, item) => sum + item.case_count, 0), note: '来自版本化 JSON', tone: 'teal' },
      { label: '对话轮次', value: datasets.reduce((sum, item) => sum + item.turn_count, 0), note: '含多轮任务', tone: 'ink' },
      { label: '测试集案例', value: datasets.reduce((sum, item) => sum + (item.splits.test ?? 0), 0), note: '未参与调参', tone: 'violet' },
    ]} />
    <label className="standalone-select">任务集<select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>{datasets.map((item) => <option value={item.dataset_id} key={item.dataset_id}>{localizeEngineeringText(item.display_name)} · {item.case_count} 个案例</option>)}</select></label>
    {detail.isLoading && <LoadingState label="正在读取任务断言" />}
    {detail.isError && <ErrorState error={detail.error} retry={() => void detail.refetch()} />}
    {dataset && <ViewTransition viewKey={dataset.dataset_id}>
      <SectionHeading index="01" title={localizeEngineeringText(dataset.display_name)} note={dataset.path} />
      <DefinitionGrid items={[
        ['数据集编号', dataset.dataset_id], ['评测套件', SUITE_LABELS[dataset.suite] ?? dataset.suite], ['执行来源', EXECUTION_SOURCE_LABELS[dataset.execution_source] ?? dataset.execution_source],
        ['数据分组', Object.entries(dataset.splits).map(([key, value]) => `${SPLIT_LABELS[key] ?? key}：${value}`).join(' · ')],
      ]} />
      <p className="muted">{localizeEngineeringText(dataset.description)}</p>
      <div className="dataset-grid">
        <div><SectionHeading index="02" title="类别覆盖" note="案例分布" /><div className="coverage-list">
          {Object.entries(dataset.categories).sort((a, b) => b[1] - a[1]).map(([category, count]) => <div className="coverage-row" key={category}>
            <div><span>{labels[category] ?? category}</span><b>{String(count).padStart(2, '0')}</b></div>
            <CoverageBar percent={Math.max(8, Math.round(count / maximum * 100))} /><code>{category}</code>
          </div>)}
        </div></div>
        <div><SectionHeading index="03" title="任务清单" note="选择案例查看断言" /><div className="case-list">
          {cases.map((item) => <button className={caseId === item.id ? 'active' : ''} onClick={() => setCaseId(item.id)} key={item.id}><code>{item.id}</code><span>{localizeEngineeringText(item.description ?? '未命名案例')}</span><small>{SPLIT_LABELS[item.split ?? 'fixed'] ?? item.split ?? '固定集'} · {item.turns.length} 轮</small></button>)}
        </div></div>
      </div>
      {selectedCase && <ViewTransition viewKey={selectedCase.id}>
        <section className="panel case-inspector">
          <DefinitionGrid items={[["案例编号", selectedCase.id], ["任务类别", labels[selectedCase.category ?? ''] ?? selectedCase.category ?? '—'], ["数据分组", SPLIT_LABELS[selectedCase.split ?? 'fixed'] ?? selectedCase.split ?? '固定集'], ["语言", LANGUAGE_LABELS[selectedCase.language ?? ''] ?? selectedCase.language ?? '—']]} />
          {selectedCase.turns.map((turn, index) => <div className="case-turn" key={`${selectedCase.id}-${index}`}><b>第 {String(index + 1).padStart(2, '0')} 轮</b><p>{turn.message}</p><small>状态：{turn.expect?.statuses?.map((value) => STATUS_LABELS[value] ?? value).join('、') || '未限定'} · 工具：{turn.expect?.tools?.map((tool) => TOOL_LABELS[tool.name] ?? tool.name).join('、') || '无强制工具'}</small></div>)}
          <details><summary>完整机器断言</summary><JsonBlock value={selectedCase} /></details>
        </section>
      </ViewTransition>}
    </ViewTransition>}
  </>
}
