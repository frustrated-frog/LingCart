import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ArrowRight, GitBranch, Play, Square, RefreshCw, Check, X } from 'lucide-react'
import { api } from '../api/client'
import { ErrorState, JsonBlock, LoadingState, PageHeader, SectionHeading, TOOL_LABELS } from '../components/SystemUI'
import type { PGArm, PGGraph, PGJob, PGPreset, PGSample, PGSelection } from '../types'
import '../styles/pg-lab.css'

const arms: { id: PGArm; label: string; note: string }[] = [
  { id: 'baseline', label: '无 PG', note: '原始上下文 · 控制组' },
  { id: 'seed', label: '种子图', note: '固定程序知识 · 指导效果' },
  { id: 'evolved', label: '演化保留图', note: '验证选择后的图 · 学习效果' },
]
const statuses: Record<string, string> = { QUEUED: '排队中', RUNNING: '执行中', COMPLETED: '已完成', FAILED: '执行失败', CANCELLED: '已停止', INTERRUPTED: '进程中断' }
const conclusions: Record<string, string> = {
  SMOKE_ONLY: '冒烟已完成，尚不能判断收益', TEST_HARNESS: '测试替身结果，不代表模型效果',
  INCONCLUSIVE: '对照未通过，保留负结果', OBSERVED_IMPROVEMENT: '本批测试观察到改善', NO_REGRESSION: '本批测试无退化，未观察到成功率提升',
}
const reasons: Record<string, string> = {
  VALIDATION_NON_REGRESSION: '独立验证未退化，保留修改', VALIDATION_REGRESSION: '验证成功率退化',
  COST_REGRESSION: '调用成本超出允许值', PG_NOT_EXPOSED: '指导未覆盖全部样本',
  SAFETY_OR_PROTOCOL_VIOLATION: '安全或协议检查未通过', REPEATED_REJECTED_STRUCTURE: '重复提出已拒绝结构',
  INVALID_PROPOSAL_OR_REPORT: '提案或验证报告无效', INCOMPARABLE_VALIDATION: '验证条件不一致',
  REFINER_FAILED: '图修改模型调用失败', REFINER_CONTEXT_TOO_LARGE: '训练轨迹超过上下文上限',
}
const errors: Record<string, string> = {
  PG_WORKER_START_FAILED: '实验进程启动失败。', PG_CANCELLED: '实验已停止，已完成样本仍可查看。',
  PG_IDENTITY_CHANGED: '源码、数据或模型配置发生变化，实验已停止。',
  PG_BUDGET_EXCEEDED: '实验触及冻结的总预算上限，已停止并保留已有证据。',
  PG_FULL_COST_CONFIRMATION_REQUIRED: '完整实验需要明确确认高调用预算。',
  PG_EXPERIMENT_FAILED: '实验未完整完成，请检查下方阶段、失败类型和已保存样本。',
}
const phase = (value: string) => value === 'seed-validation' ? '种子图验证' : value === 'refinement' ? '生成图修改'
  : value === 'queued' ? '等待进程启动' : value === 'completed' ? '三组对照完成'
    : value.startsWith('test-') ? `${arms.find(a => value === `test-${a.id}`)?.label ?? value} · 最终测试`
      : value.replace(/iteration-(\d+)-training/, '第 $1 轮 · 训练轨迹').replace(/iteration-(\d+)-validation/, '第 $1 轮 · 候选验证')
const number = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('zh-CN', { maximumFractionDigits: 1 })
const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const node = (value: string) => TOOL_LABELS[value] ?? ({ Start: '开始', End: '结束', set_goal: '建立目标', recommend: '推荐候选', complete: '完成核验', inspect: '检查证据', clarify: '澄清需求', revise_plan: '修订计划', refine_constraints: '细化约束', finish: '结束执行', respond: '回应用户', handoff: '转交处理', unsupported: '说明能力范围' }[value] ?? value)
const partitions = [['train', '训练', '只从这里学习'], ['validation', '验证', '决定是否保留'], ['test', '测试', '冻结后统一对照']] as const
const active = (status?: string) => status === 'RUNNING' || status === 'QUEUED'

function GraphInspector({ graph }: { graph: PGGraph }) {
  const [selectedId, setSelectedId] = useState('')
  const edge = graph.edges.find(item => item.id === selectedId) ?? graph.edges[0]
  return <div className="pg-graph">
    <div className="pg-edge-list" role="list" aria-label="程序图关系">
      {graph.edges.map(item => <button key={item.id} type="button" aria-pressed={item.id === edge?.id} onClick={() => setSelectedId(item.id)}>
        <span>{node(item.source)} <ArrowRight size={12} /> {node(item.target)}</span><small>{item.relation} · {item.id}</small>
      </button>)}
    </div>
    {edge ? <div className="pg-edge-detail">
      <div className="pg-node-path"><div><GitBranch size={18} /><strong>{node(edge.source)}</strong><code>{edge.source}</code></div>
        <span><small>{edge.relation}</small><ArrowRight size={30} /></span>
        <div><GitBranch size={18} /><strong>{node(edge.target)}</strong><code>{edge.target}</code></div></div>
      <dl><dt>适用条件</dt><dd>{edge.condition || '未限定额外条件'}</dd><dt>执行指导</dt><dd>{edge.guidance}</dd><dt>常见错误</dt><dd>{edge.pitfalls || '尚未记录'}</dd></dl>
      <small className="pg-muted">局部关系来自冻结图；是否实际指导过行动，请查看样本中的暴露事件。</small>
    </div> : <p>这张图尚无关系。</p>}
  </div>
}

function SelectionDetail({ selection }: { selection: PGSelection }) {
  const patch = selection.patch
  return <div className="pg-selection-detail">
    <div className="pg-selection-outcome"><strong>{reasons[selection.reason] ?? selection.reason}</strong>
      <span>验证成功率 {percent(selection.incumbent_metrics?.score)} <ArrowRight size={14} /> {percent(selection.candidate_metrics?.score)}</span></div>
    {patch ? <><p>{patch.reason}</p><small className="pg-muted">依据训练案例 {patch.source_cases.join('、')}</small>
      <div className="pg-patch">
        {patch.add_nodes.map(n => <p className="pg-added" key={`n${n.id}`}>＋节点 <code>{n.id}</code> · {n.description}</p>)}
        {patch.delete_nodes.map(n => <p className="pg-deleted" key={`d${n}`}>−节点 <code>{n}</code></p>)}
        {patch.add_edges.map(e => <div className="pg-added" key={e.id}><b>＋关系 {node(e.source)} → {node(e.target)}</b><p>{e.condition || '通用条件'} · {e.guidance}</p>{e.pitfalls && <small>常见错误：{e.pitfalls}</small>}</div>)}
        {patch.delete_edges.map(e => <p className="pg-deleted" key={e}>−关系 <code>{e}</code></p>)}
      </div></> : <p className="pg-muted">没有可应用的结构修改。原始拒绝记录已保留。</p>}
    <details><summary>查看本轮原始记录与指纹</summary><JsonBlock value={selection} /></details>
  </div>
}

function SampleEvidence({ jobId, sample }: { jobId: string; sample: PGSample }) {
  const query = useQuery({ queryKey: ['pg-sample', jobId, sample.stage, sample.case_id, sample.repetition],
    queryFn: () => api.pgSample(jobId, sample) })
  if (query.isPending) return <LoadingState label="正在读取原始执行证据" />
  if (query.isError) return <ErrorState error={query.error} retry={() => { void query.refetch() }} />
  const evidence = query.data?.pg_evidence
  if (!evidence) return <div className="pg-sample-evidence"><b>{sample.case_id} · 第 {sample.repetition} 次 · {phase(sample.stage)}</b>
    <p className="pg-muted">该历史样本尚未生成结构化 PG 干预投影，以下保留原始证据。</p><JsonBlock value={query.data} /></div>
  return <div className="pg-sample-evidence"><div className="pg-evidence-heading"><div><b>{sample.case_id} · 第 {sample.repetition} 次 · {phase(sample.stage)}</b>
    <p>{evidence.summary.failure_stage ? `失败阶段：${evidence.summary.failure_stage}` : evidence.summary.recovered ? '扰动后已恢复' : '样本未记录失败阶段'}</p></div>
    <dl><div><dt>干预</dt><dd>{evidence.summary.interventions}</dd></div><div><dt>已暴露</dt><dd>{evidence.summary.exposed}</dd></div>
      <div><dt>行动对齐</dt><dd>{evidence.summary.aligned}</dd></div><div><dt>行动分歧</dt><dd>{evidence.summary.diverged}</dd></div></dl></div>
    {evidence.turns.flatMap(turn => turn.interventions.map((item, index) => {
      const decision = item.decision?.payload ?? {}
      const action = String(decision.tool ?? decision.action ?? '没有后续决策')
      return <article className="pg-intervention-chain" key={`${turn.turn_index}-${item.intervention_id}-${index}`}>
        <header><span>干预 {index + 1} · 事件 {item.guidance.sequence}</span><b>{item.guidance.exposed ? '已进入模型决策上下文' : '未暴露给决策'}</b></header>
        <div className="pg-chain-grid">
          <section><small>01 · 状态定位</small><strong>{node(item.localization.anchor)}</strong><p>{item.localization.fallback ? '未知行动，回退全图' : '命中局部锚点'}</p><em>{item.localization.node_ids.length} 个局部节点</em></section>
          <section><small>02 · 子图检索</small><strong>{item.retrieval.selected_edge_ids.join(' / ') || '未选择关系'}</strong><p>候选 {item.retrieval.retrieved_edge_ids.length} 条 · 采用 {item.retrieval.selected_edge_ids.length} 条</p></section>
          <section><small>03 · 情境指导</small><strong>{item.guidance.status}</strong><p>{item.guidance.content || item.guidance.reason || '指导器选择弃权'}</p></section>
          <section><small>04 · 实际行动</small><strong>{node(action)}</strong><p>对齐判定：{item.action_alignment.status}</p><em>{item.decision ? `决策事件 ${item.decision.sequence}` : '没有已提交行动'}</em></section>
          <section><small>05 · 工具与核验</small><strong>{item.tool ? node(item.tool.name) : '未调用工具'}</strong><p>{item.tool ? `${item.tool.status ?? '已返回'}${item.tool.invalid_result ? ' · 无效结果' : ''}` : item.verification ? '已进入目标核验' : '尚无后续结果'}</p></section>
        </div>
        <details><summary>检查状态、局部子图、工具结果和核验原始投影</summary><JsonBlock value={item} /></details>
      </article>
    }))}
    {!evidence.summary.interventions && <div className="pg-empty"><div><strong>该样本没有 PG 干预</strong><p>无 PG 对照组属于预期；若这是种子图或演化图样本，应结合失败阶段检查原因。</p></div></div>}
    <details><summary>查看完整原始样本</summary><JsonBlock value={query.data} /></details></div>
}

function Comparison({ job }: { job?: PGJob }) {
  const metrics: { label: string; value: (id: PGArm) => string }[] = [
    { label: '测试成功率', value: id => percent(job?.arms[id]?.metrics.mean_success_rate) },
    { label: '扰动后恢复率', value: id => percent(job?.arms[id]?.metrics.recovery_rate) },
    { label: '重复无效调用', value: id => number(job?.arms[id]?.metrics.repeated_ineffective_calls) },
    { label: 'PG 指导暴露率', value: id => id === 'baseline' ? '不适用' : percent(job?.arms[id]?.metrics.pg_exposure_rate) },
    { label: '模型调用次数', value: id => number(job?.arms[id]?.metrics.model_attempts) },
    { label: '总 Token', value: id => number(job?.arms[id]?.metrics.tokens.total_tokens) },
    { label: 'P95 延迟 / 秒', value: id => { const ms = job?.arms[id]?.metrics.latency_ms.p95; return number(ms == null ? null : ms / 1000) } },
  ]
  return <div className="pg-comparison"><table><thead><tr><th><span className="pg-muted">相同任务 / 模型 / 预算</span></th>{arms.map((arm, i) => <th key={arm.id}>
    <span className="pg-arm-index">0{i + 1}</span><strong>{arm.label}</strong><small>{arm.note}</small>
    <span className="pg-arm-status">{job?.arms[arm.id] ? `${job.arms[arm.id]!.metrics.samples} 个样本已完成` : '尚无完整报告'}</span>
  </th>)}</tr></thead><tbody>{metrics.map((metric, i) => <tr key={metric.label} className={i === 0 ? 'pg-primary-metric' : ''}><th>{metric.label}</th>{arms.map(arm => <td key={arm.id}>{metric.value(arm.id)}</td>)}</tr>)}</tbody></table>
    <p className="pg-metric-note">“—”表示尚无数据或提供方未报告。恢复率仅统计实际触发扰动的样本；重复无效调用仅统计同参数、同空结果或错误。Token 包含 PG 指导调用。</p>
  </div>
}

export function PGLabPage() {
  const queryClient = useQueryClient()
  const [params, setParams] = useSearchParams()
  const catalog = useQuery({ queryKey: ['pg-catalog'], queryFn: api.pgCatalog, refetchInterval: 30_000 })
  const jobs = useQuery({ queryKey: ['pg-jobs'], queryFn: api.pgJobs, refetchInterval: 5_000 })
  const jobId = params.get('job') ?? jobs.data?.jobs[0]?.job_id ?? ''
  const jobQuery = useQuery({ queryKey: ['pg-job', jobId], queryFn: () => api.pgJob(jobId), enabled: !!jobId,
    refetchInterval: query => active(query.state.data?.status) ? 3_000 : 30_000 })
  const samples = useQuery({ queryKey: ['pg-samples', jobId], queryFn: () => api.pgSamples(jobId), enabled: !!jobId,
    refetchInterval: active(jobQuery.data?.status) ? 5_000 : false })
  const [preset, setPreset] = useState<PGPreset>('smoke')
  const [confirmFullCost, setConfirmFullCost] = useState(false)
  const [round, setRound] = useState(0)
  const [graphView, setGraphView] = useState<'seed' | 'retained'>('seed')
  const [partition, setPartition] = useState<'train' | 'validation' | 'test'>('test')
  const [sampleKey, setSampleKey] = useState('')
  const request = useRef({ preset, id: crypto.randomUUID() })
  const start = useMutation({ mutationFn: () => {
    if (request.current.preset !== preset) request.current = { preset, id: crypto.randomUUID() }
    return api.pgStart({ preset, request_id: request.current.id, confirm_full_cost: preset === 'full' && confirmFullCost })
  }, onSuccess: job => {
    setParams({ job: job.job_id }); queryClient.setQueryData(['pg-job', job.job_id], job)
    request.current = { preset, id: crypto.randomUUID() }
    void queryClient.invalidateQueries({ queryKey: ['pg-jobs'] })
  } })
  const cancel = useMutation({ mutationFn: () => api.pgCancel(jobId), onSuccess: job => {
    queryClient.setQueryData(['pg-job', jobId], job); void queryClient.invalidateQueries({ queryKey: ['pg-jobs'] })
  } })
  useEffect(() => {
    if (jobId) void queryClient.invalidateQueries({ queryKey: ['pg-samples', jobId] })
  }, [jobId, jobQuery.data?.completed_samples, queryClient])
  if (catalog.isPending) return <LoadingState label="正在读取 PG 冻结任务与程序图" />
  if (!catalog.data) return <ErrorState error={catalog.error} retry={() => { void catalog.refetch() }} />
  const data = catalog.data
  const job = jobQuery.data
  const busy = jobs.data?.jobs.some(j => active(j.status)) || active(job?.status)
  const frozen = job?.frozen ?? data.plans[preset]
  const launchPlan = data.plans[preset]
  const graph = graphView === 'retained' && job?.retained_graph ? job.retained_graph : job?.seed_graph ?? data.seed_graph
  const selection = job?.history.find(r => r.iteration === round) ?? job?.history.at(-1)
  const key = (s: PGSample) => `${s.stage}/${s.case_id}/${s.repetition}`
  const selectedSample = samples.data?.samples.find(s => key(s) === sampleKey)
  const refresh = () => { void catalog.refetch(); void jobs.refetch(); if (jobId) { void jobQuery.refetch(); void samples.refetch() } }
  return <div className="pg-lab">
    <nav className="pg-navigation"><Link to="/experiments">论文实验室 / 实验契约</Link><span>/ PG 自演化</span><button type="button" onClick={refresh}><RefreshCw size={14} />刷新状态</button></nav>
    <PageHeader code="PROCEDURAL GRAPHS / 实验台" title="让执行经验，改变下一次行动" subtitle="从训练轨迹修改程序图，用独立验证选择，再在同一测试集上比较三种上下文。" marker="候选机制" />
    {(catalog.isError || jobs.isError || jobQuery.isError || samples.isError) && <div className="pg-alert" role="alert">部分状态读取失败。已显示数据可能过期。<button onClick={refresh}>重试读取</button></div>}
    <section className="pg-launch" aria-label="启动实验">
      <div><span className="pg-kicker">冻结配置</span><h2>从一轮小实验开始</h2><p>每项任务使用独立种子数据库。实验将调用真实模型，可能产生费用。</p>
        <label>实验规模<select value={preset} disabled={start.isPending} onChange={e => { setPreset(e.target.value as PGPreset); setConfirmFullCost(false) }}>
          <option value="smoke">冒烟 · 1 轮演化 × 1 次重复</option><option value="full">完整 · 3 轮演化 × 3 次重复</option></select></label></div>
      <div className="pg-launch-budget"><div><strong>{launchPlan.max_samples}</strong><span>最多任务样本</span></div><div><strong>{launchPlan.max_model_attempts.toLocaleString()}</strong><span>最多模型尝试（含重试）</span></div>
        <small>每 Run {launchPlan.spec.max_model_calls_per_run} 次 · Refiner 每轮最多 2 次 · 总 token unit 上限 {number(launchPlan.max_token_units)}</small></div>
      <div className="pg-launch-action">{preset === 'full' && <label className="pg-cost-confirm"><input type="checkbox" checked={confirmFullCost} onChange={e => setConfirmFullCost(e.target.checked)} />我确认完整实验的高调用预算</label>}
        <button className="pg-primary-button" type="button" disabled={busy || start.isPending || !data.model_configured || jobs.isPending || jobs.isError || (preset === 'full' && !confirmFullCost)}
        onClick={() => start.mutate()}><Play size={16} />{start.isPending ? '正在提交…' : busy ? '实验执行中' : '启动真实实验'}</button>
        <small>{!data.model_configured ? 'Model Gateway 尚未配置' : '图与测试条件在执行前冻结'}</small></div>
    </section>
    {start.isError && <ErrorState error={start.error} retry={() => start.mutate()} />}
    <div className="pg-run-selector"><label>实验记录<select value={jobId} onChange={e => { setParams({ job: e.target.value }); setSampleKey(''); setRound(0) }}>
      {!jobs.data?.jobs.length && <option value="">尚未启动实验</option>}
      {jobs.data?.jobs.map(j => <option key={j.job_id} value={j.job_id}>{new Date(j.created_at).toLocaleString('zh-CN')} · {j.preset === 'smoke' ? '冒烟' : '完整'} · {statuses[j.status]}</option>)}
    </select></label>{job && <code>{job.job_id}</code>}</div>
    {jobQuery.isPending && jobId && <LoadingState label="正在读取实验进度" />}
    {job && <section className="pg-progress" aria-label="实验进度" aria-live="polite"><div><span className={`pg-status pg-status-${job.status.toLowerCase()}`}>{statuses[job.status]}</span><strong>{phase(job.stage)}</strong><span>已落盘 {job.completed_samples} / 最多 {frozen.max_samples} 个样本</span></div>
      {active(job.status) && <><progress value={job.stage_completed} max={job.stage_total || 1} aria-label="当前阶段样本进度" /><small>当前阶段 {job.stage_completed} / {job.stage_total} · 拒绝的候选可能跳过后续验证</small>
        <button className="pg-stop" disabled={job.cancel_requested || cancel.isPending} onClick={() => cancel.mutate()}><Square size={12} />{job.cancel_requested ? '等待当前样本结束' : '停止实验'}</button></>}
      {job.error && <p>{errors[job.error] ?? job.error} {job.error_type && <code>{job.error_type}</code>}</p>}
      {job.status === 'INTERRUPTED' && <p>工作进程心跳已中断。保留已有证据，不自动重跑计费请求。</p>}
    </section>}
    {cancel.isError && <ErrorState error={cancel.error} />}
    {job?.stale && <div className="pg-alert">历史实验：当前源码、数据或模型配置已变化，以下结果不能代表当前版本。</div>}
    {job?.execution_source === 'test_harness' && <div className="pg-alert">测试替身数据 · 仅验证工程链路，不代表真实模型效果。</div>}
    <SectionHeading index="01" title="三组对照" note={`${frozen.groups.test.length} 个测试情境 × ${frozen.spec.repeats} 次重复 / 每组`} />
    <Comparison job={job} />
    {job && <p className="pg-training-cost">累计硬预算：模型尝试 {number(job.budget_usage.model_attempts)} / {number(frozen.max_model_attempts)} · token unit {number(job.budget_usage.token_units)} / {number(frozen.max_token_units)}。学习阶段：轨迹调用 {number(job.learning_cost.rollout_model_attempts)} 次 · Refiner {number(job.learning_cost.refiner_model_attempts)} 次。停止时未结束调用可能尚未汇总。</p>}
    <div className="pg-verdict"><GitBranch size={20} /><div><strong>{job?.comparison ? conclusions[job.comparison.conclusion] : job ? '实验尚未完成，保留已获得的证据' : '尚无实验结果'}</strong>
      <p>{job?.comparison ? '结论仅适用于本次冻结挑战集，未进行统计显著性检验；机制仍为 candidate。' : '完成训练、验证和三组测试后，这里会展示门禁结论。'}</p>
      {job?.comparison && <details><summary>检查对照门禁与失败原因</summary>{job.comparison.failures.map(f => <p key={f}>{f}</p>)}
        {Object.entries(job.comparison.gates).map(([name, gate]) => <div key={name}><b>{arms.find(a => a.id === name)?.label}：{gate.passed ? '门禁通过' : '门禁未通过'}</b>{gate.failures.map(f => <p key={f}>{f}</p>)}</div>)}</details>}</div></div>
    <SectionHeading index="02" title="程序图与演化记录" note="查看学习实际修改了什么" />
    <div className="pg-graph-toolbar"><div className="pg-tabs" role="tablist" aria-label="图版本"><button role="tab" aria-selected={graphView === 'seed'} onClick={() => setGraphView('seed')}>种子图</button>
      <button role="tab" aria-selected={graphView === 'retained'} disabled={!job?.retained_graph} onClick={() => setGraphView('retained')}>保留图</button></div>
      <span>v{graph.version} · {graph.nodes.length} 个节点 · {graph.edges.length} 条关系</span></div>
    <GraphInspector graph={graph} />
    <div className="pg-rounds">{job?.history.length ? <><div className="pg-round-list" aria-label="演化轮次">{job.history.map(r => <button key={r.iteration} aria-pressed={selection?.iteration === r.iteration} onClick={() => setRound(r.iteration)}>
      {r.status === 'ACCEPTED' ? <Check size={15} /> : <X size={15} />}第 {r.iteration} 轮<span>{r.status === 'ACCEPTED' ? '保留' : '拒绝'}</span></button>)}</div>{selection && <SelectionDetail selection={selection} />}</>
      : <div className="pg-empty"><GitBranch size={24} /><div><strong>尚无演化提案</strong><p>Refiner 读取训练轨迹后，新增或删除的节点、关系，以及验证取舍理由会出现在这里。</p></div></div>}</div>
    <SectionHeading index="03" title="挑战集覆盖" note={`数据指纹 ${frozen.dataset_sha256.slice(0, 16)}`} />
    <div className="pg-partitions" role="tablist" aria-label="数据分组">{partitions.map(([id, title, note]) => <button key={id} role="tab" aria-selected={partition === id} onClick={() => setPartition(id)}>
      <strong>{title} <span>{frozen.groups[id].length}</span></strong><small>{note}</small></button>)}</div>
    <div className="pg-task-grid">{(job?.cases ?? data.dataset.cases).filter(c => c.learning_partition === partition).map(c => <article key={c.id}><div><code>{c.id}</code><span>{c.category}</span></div><p>{c.description}</p><small>{c.scenario_family}</small></article>)}</div>
    <SectionHeading index="04" title="样本证据" note="包括失败、停止前已完成的样本" />
    {!samples.data?.samples.length ? <div className="pg-empty"><div><strong>尚无已完成样本</strong><p>每个样本结束后立即落盘，实验中断也能继续查看。</p></div></div> : <>
      <div className="pg-samples"><table><thead><tr><th>阶段</th><th>案例 / 重复</th><th>结果</th><th>判分原因</th><th>轨迹</th></tr></thead><tbody>{samples.data.samples.map(s => <tr key={key(s)}>
        <td>{phase(s.stage)}</td><td><code>{s.case_id} / {s.repetition}</code></td><td><span className={s.passed ? 'pg-pass' : 'pg-fail'}>{s.passed ? '通过' : '失败'}</span></td>
        <td>{s.failures.join('；') || (s.failure_stage ? `失败阶段：${s.failure_stage}` : '独立判据通过')}<small className="pg-sample-count">PG 干预 {s.interventions}</small></td><td><button aria-expanded={selectedSample === s} onClick={() => setSampleKey(sampleKey === key(s) ? '' : key(s))}>查看 {s.case_id} 证据</button></td></tr>)}</tbody></table></div>
      {selectedSample && <SampleEvidence jobId={jobId} sample={selectedSample} />}</>}
    {job && <details className="pg-raw"><summary>冻结配置、执行身份与完整实验记录</summary><JsonBlock value={job} /></details>}
  </div>
}
