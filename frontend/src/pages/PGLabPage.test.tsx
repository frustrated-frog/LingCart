import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { PGCatalog, PGJob, PGPlan } from '../types'
import { PGLabPage } from './PGLabPage'

const plan: PGPlan = { preset: 'smoke', spec: { repeats: 1, iterations: 1, max_model_calls_per_run: 8 },
  groups: { train: ['PG01'], validation: ['PG07'], test: ['PG13'] }, max_samples: 6, max_model_attempts: 50, max_token_units: 361500,
  dataset_sha256: 'dataset-frozen-sha', seed_sha256: 'seed-frozen-sha' }
const graph: PGCatalog['seed_graph'] = { graph_id: 'test-graph', version: 0, parent_sha256: null,
  nodes: [{ id: 'Start', kind: 'STATE', description: '开始' }, { id: 'End', kind: 'STATE', description: '结束' }],
  edges: [{ id: 'start-end', source: 'Start', target: 'End', relation: 'LEADS_TO', condition: '核验通过', guidance: '只有已核验才能结束', pitfalls: '不能提前结束' }] }
const catalog: PGCatalog = { model_configured: true, plans: { smoke: plan, full: { ...plan, preset: 'full', max_samples: 180, max_model_attempts: 2886, max_token_units: 49_000_000 } },
  seed_graph: graph, current_identity: {}, dataset: { cases: [
    { id: 'PG13', learning_partition: 'test', scenario_family: 'empty-policy', category: '空结果恢复', description: '查找键盘并说明政策' },
    { id: 'PG14', learning_partition: 'test', scenario_family: 'hidden-full-only', category: '工具选择', description: '完整实验才展示的案例' },
  ] } }
const job: PGJob = { job_id: 'pg-test-job', created_at: '2026-10-23T01:00:00Z', preset: 'smoke', status: 'COMPLETED', stage: 'completed', completed_samples: 30,
  frozen: plan, stage_completed: 6, stage_total: 6, cancel_requested: false, stale: false, identity: {}, execution_source: 'live_provider',
  history: [{ iteration: 1, status: 'REJECTED', reason: 'VALIDATION_REGRESSION', parent_sha256: 'seed', retained_sha256: 'seed',
    incumbent_metrics: { score: .8, model_attempts: 6 }, candidate_metrics: { score: .6, model_attempts: 8 },
    patch: { reason: '尝试空结果恢复', source_cases: ['PG01'], add_nodes: [], delete_nodes: [], add_edges: [{ ...graph.edges[0], id: 'retry', guidance: '空结果改换条件' }], delete_edges: [] } }],
  arms: {}, error: null, seed_graph: graph, retained_graph: graph, cases: catalog.dataset.cases,
  learning_cost: { rollout_model_attempts: 30, refiner_model_attempts: 1, rollout_tokens: null, refiner_token_units: 1200 },
  budget_usage: { model_attempts: 31, token_units: 48_000 },
  comparison: { conclusion: 'SMOKE_ONLY', evidence_valid: false, failures: [], gates: { evolved: { passed: false, failures: ['没有保留新的图结构'] } } } }

function page() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}><MemoryRouter><PGLabPage /></MemoryRouter></QueryClientProvider>)
}
beforeEach(() => {
  vi.restoreAllMocks()
  vi.spyOn(api, 'pgCatalog').mockResolvedValue(catalog)
  vi.spyOn(api, 'pgJobs').mockResolvedValue({ jobs: [] })
  vi.spyOn(api, 'pgJob').mockResolvedValue(job)
  vi.spyOn(api, 'pgSamples').mockResolvedValue({ samples: [] })
})
it('shows real empty state and bounded paid launch without invented metrics', async () => {
  page()
  expect(await screen.findByText('尚无实验结果')).toBeInTheDocument()
  expect(screen.getByText('50')).toBeInTheDocument()
  expect(screen.queryByText('0.0%')).not.toBeInTheDocument()
  expect(screen.getByText('只有已核验才能结束')).toBeInTheDocument()
  expect(screen.getByText('查找键盘并说明政策')).toBeInTheDocument()
  expect(screen.queryByText('完整实验才展示的案例')).not.toBeInTheDocument()
})
it('keeps idempotency key when retrying a failed launch', async () => {
  vi.mocked(api.pgJob).mockResolvedValue({ ...job, status: 'QUEUED' })
  const start = vi.spyOn(api, 'pgStart').mockRejectedValueOnce(new Error('网络中断')).mockResolvedValue({ ...job, status: 'QUEUED' })
  page()
  fireEvent.click(await screen.findByRole('button', { name: '启动真实实验' }))
  await screen.findByText('网络中断')
  fireEvent.click(screen.getByRole('button', { name: '启动真实实验' }))
  await waitFor(() => expect(start).toHaveBeenCalledTimes(2))
  expect(start.mock.calls[0][0].request_id).toBe(start.mock.calls[1][0].request_id)
  expect(await screen.findByRole('button', { name: '实验执行中' })).toBeDisabled()
})
it('preserves rejected modifications and shows no gain for smoke or stale evidence', async () => {
  vi.mocked(api.pgJobs).mockResolvedValue({ jobs: [job] })
  vi.mocked(api.pgJob).mockResolvedValue({ ...job, stale: true })
  page()
  expect(await screen.findByText('冒烟已完成，尚不能判断收益')).toBeInTheDocument()
  expect(screen.getByText('验证成功率退化')).toBeInTheDocument()
  expect(screen.getByText(/空结果改换条件/, { selector: 'p' })).toBeInTheDocument()
  expect(screen.getByText(/当前源码、数据或模型配置已变化/)).toBeInTheDocument()
  fireEvent.click(screen.getByText('检查对照门禁与失败原因'))
  expect(screen.getByText('没有保留新的图结构')).toBeInTheDocument()
})
it('shows cooperative stop and preserves partial sample evidence', async () => {
  const running = { ...job, status: 'RUNNING', comparison: null, stage: 'iteration-01-training' }
  vi.mocked(api.pgJobs).mockResolvedValue({ jobs: [running] })
  vi.mocked(api.pgJob).mockResolvedValue(running)
  const stop = vi.spyOn(api, 'pgCancel').mockResolvedValue({ ...running, cancel_requested: true })
  vi.mocked(api.pgSamples).mockResolvedValue({ samples: [{ stage: 'seed-validation', case_id: 'PG07', repetition: 1, passed: false, failures: ['目标未完成'], interventions: 0, failure_stage: 'BASELINE_TASK_FAILURE' }] })
  vi.spyOn(api, 'pgSample').mockResolvedValue({ case_id: 'PG07', repetition: 1, passed: false,
    events: [{ type: 'model_call', purpose: 'pg_guidance' }] })
  page()
  fireEvent.click(await screen.findByRole('button', { name: '停止实验' }))
  expect(await screen.findByRole('button', { name: '等待当前样本结束' })).toBeDisabled()
  expect(stop).toHaveBeenCalledWith(job.job_id)
  fireEvent.click(await screen.findByRole('button', { name: '查看 PG07 证据' }))
  expect(await screen.findByText(/pg_guidance/)).toBeInTheDocument()
})
it('shows unconfigured gateway and read errors without enabling calls', async () => {
  vi.mocked(api.pgCatalog).mockResolvedValue({ ...catalog, model_configured: false })
  vi.mocked(api.pgJobs).mockRejectedValue(new Error('读取失败'))
  page()
  expect(await screen.findByRole('button', { name: '启动真实实验' })).toBeDisabled()
  expect(await screen.findByRole('alert')).toHaveTextContent('部分状态读取失败')
  expect(screen.getByText('Model Gateway 尚未配置')).toBeInTheDocument()
})

it('requires explicit cost confirmation before starting the full experiment', async () => {
  const start = vi.spyOn(api, 'pgStart').mockResolvedValue({ ...job, preset: 'full', status: 'QUEUED' })
  page()
  fireEvent.change(await screen.findByLabelText('实验规模'), { target: { value: 'full' } })
  expect(screen.getByRole('button', { name: '启动真实实验' })).toBeDisabled()
  fireEvent.click(screen.getByLabelText('我确认完整实验的高调用预算'))
  fireEvent.click(screen.getByRole('button', { name: '启动真实实验' }))
  await waitFor(() => expect(start).toHaveBeenCalled())
  expect(start.mock.calls[0][0].confirm_full_cost).toBe(true)
})

it('renders the persisted intervention chain instead of only raw json', async () => {
  vi.mocked(api.pgJobs).mockResolvedValue({ jobs: [job] })
  vi.mocked(api.pgSamples).mockResolvedValue({ samples: [{ stage: 'test-seed', case_id: 'PG13', repetition: 1,
    passed: true, failures: [], interventions: 1, failure_stage: null }] })
  vi.spyOn(api, 'pgSample').mockResolvedValue({ case_id: 'PG13', repetition: 1, passed: true, pg_evidence: {
    schema_version: 1, summary: { interventions: 1, exposed: 1, aligned: 1, diverged: 0,
      unknown_alignment: 0, localization_fallbacks: 0, environment_triggered: true, recovered: true, failure_stage: null },
    turns: [{ turn_index: 0, request: '推荐耳机', passed: true, failures: [], interventions: [{
      intervention_id: 'decision-1', graph: { graph_id: 'pg', graph_version: 1, graph_sha256: 'a' },
      localization: { anchor: 'Start', fallback: false, node_ids: ['Start', 'search_products'], state_snapshot: {}, sequence: 1 },
      retrieval: { retrieved_edge_ids: ['start-search'], selected_edge_ids: ['start-search'], subgraph: null },
      guidance: { status: 'READY', reason: null, content: '先查询真实目录', exposed: true, sequence: 1 },
      decision: { sequence: 4, source: 'model', payload: { action: 'tool', tool: 'search_products' } },
      action_alignment: { status: 'ALIGNED', matched_edge_ids: ['start-search'] },
      tool: { name: 'search_products', args: {}, status: 'SUCCEEDED', sequence: 6, result: { products: [] }, invalid_result: false },
      verification: { all_goals_satisfied: true },
    }] }],
  } })
  page()
  fireEvent.click(await screen.findByRole('button', { name: '查看 PG13 证据' }))
  expect(await screen.findByText('先查询真实目录')).toBeInTheDocument()
  expect(screen.getByText('01 · 状态定位')).toBeInTheDocument()
  expect(screen.getByText('对齐判定：ALIGNED')).toBeInTheDocument()
})
