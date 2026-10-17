import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { DefinitionGrid, ErrorState, JsonBlock, LoadingState, PageHeader, SectionHeading, TelemetryStrip } from '../components/SystemUI'

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
    <PageHeader code="DATASET COVERAGE / 03" title="数据集覆盖"
      subtitle="从任务类别、数据分组和断言深度观察评测底座，避免用少量顺利案例代替能力证据。" />
    <TelemetryStrip items={[
      { label: 'DATASETS', value: datasets.length, note: '固定 / 真实模型 / 自主专项', tone: 'blue' },
      { label: 'TOTAL CASES', value: datasets.reduce((sum, item) => sum + item.case_count, 0), note: '来自版本化 JSON', tone: 'teal' },
      { label: 'TOTAL TURNS', value: datasets.reduce((sum, item) => sum + item.turn_count, 0), note: '含多轮任务', tone: 'ink' },
      { label: 'TEST SPLIT', value: datasets.reduce((sum, item) => sum + (item.splits.test ?? 0), 0), note: '未参与调参', tone: 'violet' },
    ]} />
    <label className="standalone-select">任务集<select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>{datasets.map((item) => <option value={item.dataset_id} key={item.dataset_id}>{item.display_name} · {item.case_count} cases</option>)}</select></label>
    {detail.isLoading && <LoadingState label="正在读取任务断言" />}
    {detail.isError && <ErrorState error={detail.error} retry={() => void detail.refetch()} />}
    {dataset && <>
      <SectionHeading index="01" title={dataset.display_name} note={dataset.path} />
      <DefinitionGrid items={[
        ['DATASET ID', dataset.dataset_id], ['SUITE', dataset.suite], ['EXECUTION', dataset.execution_source],
        ['SPLITS', Object.entries(dataset.splits).map(([key, value]) => `${key}:${value}`).join(' · ')],
      ]} />
      <p className="muted">{dataset.description}</p>
      <div className="dataset-grid">
        <div><SectionHeading index="02" title="类别覆盖" note="CASE DISTRIBUTION" /><div className="coverage-list">
          {Object.entries(dataset.categories).sort((a, b) => b[1] - a[1]).map(([category, count]) => <div className="coverage-row" key={category}>
            <div><span>{labels[category] ?? category}</span><b>{String(count).padStart(2, '0')}</b></div>
            <div className="coverage-track"><i style={{ width: `${Math.max(8, Math.round(count / maximum * 100))}%` }} /></div><code>{category}</code>
          </div>)}
        </div></div>
        <div><SectionHeading index="03" title="任务清单" note="SELECT A CASE TO INSPECT" /><div className="case-list">
          {cases.map((item) => <button className={caseId === item.id ? 'active' : ''} onClick={() => setCaseId(item.id)} key={item.id}><code>{item.id}</code><span>{item.description}</span><small>{item.split ?? 'fixed'} · {item.turns.length} turn</small></button>)}
        </div></div>
      </div>
      {selectedCase && <section className="panel case-inspector">
        <DefinitionGrid items={[["CASE ID", selectedCase.id], ["CATEGORY", selectedCase.category ?? '—'], ["SPLIT", selectedCase.split ?? 'fixed'], ["LANGUAGE", selectedCase.language ?? '—']]} />
        {selectedCase.turns.map((turn, index) => <div className="case-turn" key={`${selectedCase.id}-${index}`}><b>TURN {String(index + 1).padStart(2, '0')}</b><p>{turn.message}</p><small>状态：{turn.expect?.statuses?.join(', ') || '未限定'} · 工具：{turn.expect?.tools?.map((tool) => tool.name).join(', ') || '无强制工具'}</small></div>)}
        <details><summary>完整机器断言</summary><JsonBlock value={selectedCase} /></details>
      </section>}
    </>}
  </>
}
