import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { ExecutionWorkbench } from '../components/ExecutionWorkbench'
import { ResultArtifactViews } from '../components/CatalogResultView'
import type { RunDetail } from '../types'
import {
  DefinitionGrid,
  ErrorState,
  formatTime,
  JsonBlock,
  LoadingState,
  PageHeader,
  SegmentedTabs,
  SectionHeading,
  SPECIALIST_LABELS,
  STATUS_LABELS,
  TERMINAL_REASON_LABELS,
  TelemetryStrip,
  TOOL_LABELS,
  ViewTransition,
} from '../components/SystemUI'

type Tab = 'result' | 'evidence' | 'steering' | 'pg' | 'tools' | 'raw'

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

  return (
    <>
      <PageHeader code={businessMode ? '任务档案 / 01' : '实时运行 / 01'}
        title={businessMode ? '任务记录' : '运行检查'}
        subtitle={businessMode ? '查看每次业务处理的真实结果，并可进入工程视图检查执行证据。' : '沿执行主脊检查一次任务：配置身份、事件顺序、证据状态、工具调用和终止原因。'} />
      {runs.length === 0 ? <div className="empty-state"><b>尚无运行记录</b><p>请先在业务沙盒执行一个消费任务。</p></div> : <>
        <div className="filter-row run-filters">
          <label>状态<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="ALL">全部状态</option>{statuses.map((value) => <option key={value} value={value}>{STATUS_LABELS[value] ?? value}</option>)}</select></label>
          <label>配置档案<select value={profile} onChange={(e) => setProfile(e.target.value)}><option value="ALL">全部配置档案</option>{profiles.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>运行<select value={runId ?? ''} onChange={(e) => navigate(`${businessMode ? '/sandbox/tasks' : '/runs'}/${e.target.value}`)}>
            {filtered.map((run) => <option value={run.run_id} key={run.run_id}>{formatTime(run.created_at)} · {run.request}</option>)}
          </select></label>
        </div>
        {detailQuery.isLoading && <LoadingState label="正在装载运行证据" />}
        {detailQuery.isError && <ErrorState error={detailQuery.error} retry={() => void detailQuery.refetch()} />}
        {detail && <ViewTransition viewKey={detail.run_id}>
          <TelemetryStrip items={[
            { label: '运行状态', value: STATUS_LABELS[detail.status] ?? detail.status, note: detail.terminal_reason ? (TERMINAL_REASON_LABELS[detail.terminal_reason] ?? detail.terminal_reason) : '尚未终止', tone: 'teal' },
            { label: '运行配置档案', value: detail.profile.profile_id, note: detail.profile.legacy_profile_reference ? '历史运行未记录' : '已记录', tone: 'violet' },
            { label: '工具 / 模型', value: `${detail.tool_calls} / ${detail.model_calls}`, note: `共 ${detail.event_count} 个事件`, tone: 'blue' },
            { label: '令牌用量', value: detail.tokens ?? '未报告', note: '模型服务商未报告时不伪造', tone: 'ink' },
          ]} />
          <SectionHeading index="01" title="执行主脊" note={`${detail.run_id.slice(0, 12)} / ${formatTime(detail.created_at)}`} />
          <div className="run-prompt">{detail.request}</div>
          <ExecutionWorkbench key={detail.run_id} events={events} selected={selectedEvent} onSelect={setSelectedEvent}
            running={detail.status === 'RUNNING'} stale={detailQuery.isError} />
          <SegmentedTabs value={tab} onChange={(value) => setTab(value as Tab)} layoutId="run-detail-tab"
            items={
              [['result', '结果与计划'], ['evidence', '上下文与证据'], ['steering', '在线监控与纠偏'], ['pg', '程序图指导'], ['tools', '工具账本'], ['raw', '原始记录']]
            } />
          <ViewTransition viewKey={tab} className="tab-panel">
            {tab === 'result' && <ResultTab detail={detail} />}
            {tab === 'evidence' && <EvidenceTab detail={detail} />}
            {tab === 'steering' && <SteeringTab detail={detail} />}
            {tab === 'pg' && <ProceduralGraphTab detail={detail} />}
            {tab === 'tools' && <ToolsTab detail={detail} />}
            {tab === 'raw' && <JsonBlock value={{ run_id: detail.run_id, harness: detail.profile, events, budget: detail.result.budget, usage: detail.result.usage, commands: detail.commands }} />}
          </ViewTransition>
        </ViewTransition>}
      </>}
    </>
  )
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

const CASE_EXECUTION_STATUS_LABELS: Record<string, string> = {
  EXECUTE: '执行可信读取', WAIT_USER_INPUT: '等待用户补充材料', WAIT_APPROVAL: '等待用户明确确认',
  BLOCKED: '能力边界阻断', COMPLETE: '无需后续动作', ABSTAINED: '主动弃权',
}

const CASE_EXECUTION_REASON_LABELS: Record<string, string> = {
  AUTHORITATIVE_EVIDENCE_AVAILABLE: '政策允许读取权威订单事实',
  CUSTOMER_EVIDENCE_REQUIRED: '政策要求用户补充材料',
  EXPLICIT_APPROVAL_REQUIRED: '退款等写操作必须由用户明确确认',
  ACTION_COMPLETED_WITH_PROGRESS: '工具结果已核验，案件版本已经推进',
  UNSUPPORTED_CAPABILITY: '当前能力不能可靠完成该动作',
  AUTONOMOUS_STEP_BUDGET_EXHAUSTED: '已达到本轮自主动作上限',
  CASE_STATE_NOT_EXECUTABLE: '案件、裁决或计划身份不一致，已失败关闭',
  POLICY_OBLIGATION_NOT_AUTOMATABLE: '该政策义务不允许自动执行',
  NO_FURTHER_AUTONOMOUS_ACTION: '当前没有待执行的自主动作',
  NO_SUPPORTED_NEXT_ACTION: '计划没有当前执行器支持的下一步',
  CASE_TERMINAL: '案件已进入终态，无需继续执行',
  TRUSTED_AUTONOMOUS_ACTION_DOMINATES: '权威工具动作无需打断用户，优先于其它候选',
  HIGHEST_INFORMATION_GAIN_QUESTION: '选择结构化信息增益上界最高的问题',
  ACTIVE_CLARIFICATION_MAX_STRUCTURAL_VALUE: '从非支配问题中选择政策解锁与未知消除价值最高的一项',
  CLARIFICATION_PORTFOLIO_VERIFIED: '澄清问题、语义缺口和反事实答案分支已完成绑定核验',
  POLICY_BOUND_APPROVAL_READY: '全部政策义务满足，进入明确授权边界',
  DEFERRED_OBLIGATION_PRESERVED: '用户已明确延期，保留等待状态而不重复追问',
  CAPABILITY_BOUNDARY_PRESERVED: '没有可安全自动执行的能力，保持阻断边界',
}

function PolicyAwareCaseExecution({ detail }: { detail: RunDetail }) {
  const events = detail.result.events ?? []
  const projections = events.filter((item) => item.type === 'case_state_projected')
  const graphEvents = events.filter((item) => item.type === 'case_obligation_graph_solved')
  const clarificationEvents = events.filter((item) => item.type === 'case_clarification_portfolio_solved')
  const clarificationOpened = events.filter((item) => item.type === 'case_clarification_opened')
  const clarificationBindings = events.filter((item) => item.type === 'case_clarification_answer_bound')
  const actions = events.filter((item) => item.type === 'case_autonomous_action_selected')
  const completed = events.filter((item) => item.type === 'case_autonomous_action_completed')
  const pause = events.filter((item) => item.type === 'case_execution_paused').at(-1)
  if (!projections.length && !graphEvents.length && !clarificationEvents.length
      && !clarificationOpened.length && !clarificationBindings.length && !actions.length && !pause) return null
  const state = record(projections.at(-1)?.state)
  const latestGraphEvent = graphEvents.at(-1)
  const latestGraph = record(latestGraphEvent?.graph)
  const latestNodes = Array.isArray(latestGraph.nodes) ? latestGraph.nodes : []
  const latestCandidates = Array.isArray(latestGraph.candidates) ? latestGraph.candidates : []
  const latestPortfolio = record(clarificationEvents.at(-1)?.portfolio)
  const latestQuestions = Array.isArray(latestPortfolio.questions) ? latestPortfolio.questions : []
  const policyIdentity = [state.policy_version_id, state.policy_decision_id].filter(Boolean).map(String).join(' / ')
  const planIdentity = [state.plan_id, state.plan_revision].filter((value) => value !== undefined && value !== null).map(String).join(' / ')
  const pauseStatus = String(pause?.status ?? 'ABSTAINED')
  const reason = String(pause?.reason_code ?? 'NO_SUPPORTED_NEXT_ACTION')
  const ledgerItems: Array<Record<string, unknown>> = [
    ...graphEvents.map((item) => ({ ...item, displayKind: '义务求解' })),
    ...clarificationEvents.map((item) => ({ ...item, displayKind: '澄清求解' })),
    ...clarificationOpened.map((item) => ({ ...item, displayKind: '问题入账' })),
    ...clarificationBindings.map((item) => ({ ...item, displayKind: '回答绑定' })),
    ...actions.map((item) => ({ ...item, displayKind: '选择动作' })),
    ...completed.map((item) => ({ ...item, displayKind: '完成动作' })),
    ...(pause ? [{ ...pause, displayKind: '边界暂停' }] : []),
  ]
  return <>
    <SectionHeading index="P" title="政策感知自主执行" note="只执行无需审批的权威读取" />
    <DefinitionGrid items={[
      ['循环状态', CASE_EXECUTION_STATUS_LABELS[pauseStatus] ?? pauseStatus],
      ['自主工具动作', completed.length],
      ['当前义务节点', latestGraphEvent ? latestNodes.filter((item) => record(item).kind === 'REQUIREMENT').length : '未启用'],
      ['当前候选行动', latestGraphEvent ? latestCandidates.length : '未启用'],
      ['当前澄清问题', clarificationEvents.length ? latestQuestions.length : '未启用'],
      ['持久化待回答', clarificationOpened.length],
      ['已绑定回答', clarificationBindings.length],
      ['案件版本', state.case_id ? `${String(state.case_id)} / REV ${String(state.case_revision ?? '—')}` : '未记录'],
      ['政策裁决', policyIdentity || '未记录'],
      ['案件计划', planIdentity || '未记录'],
    ]} />
    <p className="muted">退款提交不会由该循环自动执行；澄清信息价值只表示结构上界，不是回答概率或实验收益。需要用户材料、明确确认或系统不具备的能力时，循环都会持久化原因并暂停。</p>
    <div className="steering-ledger">{ledgerItems.sort((left, right) => Number(left.sequence ?? 0) - Number(right.sequence ?? 0)).map((item, index) => {
      const isGraph = item.displayKind === '义务求解'
      const isClarification = item.displayKind === '澄清求解'
      const isClarificationOpened = item.displayKind === '问题入账'
      const isClarificationBinding = item.displayKind === '回答绑定'
      const graph = record(item.graph)
      const candidates = Array.isArray(graph.candidates) ? graph.candidates : []
      const portfolio = record(item.portfolio)
      const questions = Array.isArray(portfolio.questions) ? portfolio.questions : []
      const decision = record(item.decision)
      const selection = isClarification ? decision : record(item.selection)
      const operation = String(item.operation ?? '')
      const itemReason = String(selection.reason_code ?? item.reason_code ?? (item.displayKind === '完成动作' ? 'ACTION_COMPLETED_WITH_PROGRESS' : reason))
      const selectedQuestion = isClarification
        ? record(questions.find((question) => record(question).question_id === decision.selected_question_id))
        : {}
      const title = isGraph
        ? `${candidates.length} 个候选行动`
        : isClarification
          ? `${questions.length} 个澄清问题`
          : isClarificationOpened
            ? '待回答问题已持久化'
            : isClarificationBinding
              ? '回答已绑定，等待独立核验'
              : operation
                ? (TOOL_LABELS[operation] ?? operation)
                : (CASE_EXECUTION_STATUS_LABELS[String(item.status)] ?? String(item.status))
      const description = isClarificationOpened
        ? '问题绑定案件版本、政策指纹和内容指纹，不代表用户授权。'
        : isClarificationBinding
          ? '本轮原文只作为证据候选，是否满足义务仍由语义核验与政策裁决决定。'
          : (CASE_EXECUTION_REASON_LABELS[itemReason] ?? itemReason)
      const evidence = isClarificationOpened
        ? `问题：${String(item.question ?? '未记录')} · ${String(item.question_id ?? '').slice(0, 12)}`
        : isClarificationBinding
          ? `义务 ${String(item.requirement_id ?? '未记录')} · 授权：否`
          : isClarification && selectedQuestion.question
            ? `选中问题：${String(selectedQuestion.question)}`
            : isGraph && selection.selected_candidate_id
              ? `候选 ${String(selection.selected_candidate_id).slice(0, 12)}`
              : item.from_revision !== undefined
                ? `REV ${String(item.from_revision)} → REV ${String(item.to_revision)}`
                : `案件版本 REV ${String(item.case_revision ?? state.case_revision ?? '—')}`
      return <article key={`${String(item.sequence ?? index)}-${String(item.displayKind)}`}>
        <span>{String(item.sequence ?? '—')} / {String(item.displayKind)}</span>
        <b>{title}</b>
        <p>{description}</p>
        <small>{evidence}</small>
      </article>
    })}</div>
  </>
}

function ProceduralGraphTab({ detail }: { detail: RunDetail }) {
  const events = detail.result.events ?? []
  const guidance = events.filter((item) => item.type === 'pg_guidance')
  const graph = record(record(detail.checkpoint.context).procedural_graph)
  const labels: Record<string, string> = { READY: '建议已生成', EXPOSED: '已送入决策', ABSTAINED: '图不适用，已弃权', SKIPPED: '已跳过', FAILED: '指导失败', CONTEXT_OMITTED: '上下文不足，未送入决策' }
  const reasons: Record<string, string> = { CONTEXT_BUDGET: '图或轨迹超出上下文预算', LOCALIZATION_ABSTAINED: '轨迹证据不足，定位器主动弃权', NO_OUTGOING_EDGES: '当前节点没有后继关系', NO_APPLICABLE_EDGES: '候选关系均未获得当前证据支持', RESERVED_DECISION_AND_AUDIT: '为决策和独立核验保留预算', GUIDANCE_SCHEMA_INVALID: '指导格式或引用无效' }
  if (!guidance.length && !Object.keys(graph).length) return <div className="empty-inline">该运行尚无程序图指导记录。</div>
  return <>
    <DefinitionGrid items={[
      ['图标识', String(graph.graph_id ?? guidance[0]?.graph_id ?? '未记录')],
      ['冻结图版本', String(graph.version ?? guidance[0]?.graph_version ?? '未记录')],
      ['指导记录', guidance.length],
      ['指导模型调用', events.filter((item) => ['model_started', 'model_retry'].includes(item.type) && item.purpose === 'pg_guidance').length],
    ]} />
    <p className="muted">图在本次运行中保持冻结。候选后继保持建议性质；只有已有无进展证据的完全相同重放可以在提交前阻断，任何单次行动都不代表已经证明效果改善。</p>
    <details><summary>查看冻结图指纹</summary><JsonBlock value={graph.sha256 ?? guidance[0]?.graph_sha256 ?? null} /></details>
    {!guidance.length && <div className="empty-inline">已绑定图，尚未生成指导；无法判断是否实际参与决策。</div>}
    <div className="steering-ledger pg-guidance-ledger">{guidance.map((item, index) => {
      const exposure = events.filter((event) => event.type === 'evidence_context' && event.pg_decision_key === item.decision_key).at(-1)
      const review = events.filter((event) => event.type === 'pg_decision_reviewed' && event.decision_key === item.decision_key).at(-1)
      const outcome = review?.review_id ? events.filter((event) => event.type === 'pg_action_outcome'
        && event.review_id === review.review_id).at(-1) : undefined
      const next = events.find((event) => event.type === 'adaptive_decision' && Number(event.sequence) > Number(item.sequence)
        && (index + 1 === guidance.length || Number(event.sequence) < Number(guidance[index + 1].sequence)))
      const decision = record(next?.decision)
      const status = String(exposure?.pg_status ?? item.status)
      const edgeIds = Array.isArray(item.edge_ids) ? item.edge_ids.map(String) : []
      const subgraph = record(item.subgraph)
      const edges = Array.isArray(subgraph.edges) ? subgraph.edges.map(record) : []
      const localization = record(item.localization)
      const shadow = record(localization.shadow)
      const confidence = typeof localization.confidence === 'number' ? `${Math.round(localization.confidence * 100)}%` : '未记录'
      const scope = item.abstained ? '主动弃权' : item.fallback ? '全图回退' : '有界局部图'
      const generation = record(item.generation)
      const contract = record(item.guidance_contract)
      const applicability = record(item.applicability)
      const evaluations = Array.isArray(applicability.evaluations) ? applicability.evaluations.map(record) : []
      const preplay = record(item.preplay)
      const selectedPath = record(preplay.selected_path)
      const preplayEdges = Array.isArray(selectedPath.edge_ids) ? selectedPath.edge_ids.map(String) : []
      const predictionOutcome = preplay.plan_id ? events.filter((event) => event.type === 'pg_prediction_outcome'
        && event.preplay_plan_id === preplay.plan_id).at(-1) : undefined
      return <article key={String(item.sequence ?? index)}>
        <span>事件 {String(item.sequence ?? '—')} · 定位 {String(item.anchor ?? '主动弃权')} · {scope}</span>
        <b>{labels[status] ?? status}</b>
        <p>{String(item.guidance || reasons[String(item.reason)] || item.reason || '没有生成可用建议')}</p>
        <small>生成方式：{generation.mode === 'deterministic-contract-v1' ? '确定性指导契约' : '模型情境化'} · 额外模型调用：{String(generation.model_calls ?? '未记录')}{Object.keys(contract).length ? ` · 契约 ${String(contract.contract_id ?? '').slice(0, 8)}` : ''}</small>
        <small>定位器：{String(localization.algorithm ?? 'action-anchor-v1')} · 置信度：{confidence}{Object.keys(shadow).length ? ` · v1 影子：${String(shadow.anchor ?? '未知')}${shadow.fallback ? '（全图回退）' : ''}` : ''}</small>
        {Object.keys(applicability).length > 0 && <small>条件证据：{String(applicability.status ?? '未记录')} · TRUE {Array.isArray(applicability.selected_edge_ids) ? applicability.selected_edge_ids.length : 0} · FALSE {Array.isArray(applicability.false_edge_ids) ? applicability.false_edge_ids.length : 0} · UNKNOWN {Array.isArray(applicability.unknown_edge_ids) ? applicability.unknown_edge_ids.length : 0}</small>}
        {Object.keys(preplay).length > 0 && <div className="run-preplay"><div><strong>有界程序预演</strong><span>{String(preplay.status ?? '未记录')} · {preplay.replanned_from ? `由 ${String(preplay.replanned_from).slice(0, 8)} 重规划` : '初始预演'}</span></div>
          {preplayEdges.length ? <div className="run-preplay-path">{preplayEdges.map((edgeId, edgeIndex) => <span className={edgeIndex === 0 ? 'is-current' : 'is-future'} key={`${edgeId}-${edgeIndex}`}>{edgeId}<i>{edgeIndex === 0 ? '当前可执行' : '预测路径'}</i></span>)}</div> : <p>{String(preplay.reason ?? '没有可用路径')}</p>}
          <small>首步：{String(selectedPath.first_action ?? '无额外行动')} · 路径分数：{typeof selectedPath.score === 'number' ? selectedPath.score.toFixed(3) : '未记录'} · 预测不是事实或授权</small>
          {predictionOutcome && <small className={`run-prediction-${String(predictionOutcome.status).toLowerCase()}`}>预测核对：{String(predictionOutcome.status)} · 期望 {String(predictionOutcome.expected_action ?? '无')} / 实际 {String(predictionOutcome.actual_action ?? '无')}</small>}
        </div>}
        <small>引用关系：{edgeIds.join(' / ') || '无'} · 后续行动：{String(decision.tool ?? decision.action ?? review?.actual_action ?? '尚无已提交行动')}{review ? ` · 采用审查：${String(review.status)}${review.blocked ? '（提交前阻断）' : ''}` : ''}</small>
        {outcome && <small>行动进展：{String(outcome.progress_status ?? '未记录')} · {Array.isArray(outcome.signals) && outcome.signals.length ? outcome.signals.map(String).join(' / ') : String(outcome.progress_reason ?? '没有可核验信号')}</small>}
        {evaluations.length > 0 && <details><summary>查看每条关系的条件求值证据</summary><div className="table-wrap"><table><thead><tr><th>关系</th><th>判定</th><th>原因</th><th>证据绑定</th></tr></thead><tbody>
          {evaluations.map((evaluation) => <tr key={String(evaluation.edge_id)}><td>{String(evaluation.source)} → {String(evaluation.target)}<br /><code>{String(evaluation.edge_id)}</code></td><td>{String(evaluation.status)}</td><td>{String(evaluation.reason)}</td><td><JsonBlock value={record(evaluation.evaluation).evidence_bindings ?? []} /></td></tr>)}
        </tbody></table></div></details>}
        {edges.length > 0 && <details><summary>查看本次检索的行动关系与条件</summary><div className="table-wrap"><table><thead><tr><th>关系</th><th>条件</th><th>指导</th><th>常见错误</th></tr></thead><tbody>
          {edges.map((edge) => <tr key={String(edge.id)}><td>{String(edge.source)} → {String(edge.target)}<br /><code>{String(edge.relation)}</code></td><td>{String(edge.condition || '无条件')}</td><td>{String(edge.guidance)}</td><td>{String(edge.pitfalls || '未记录')}</td></tr>)}
        </tbody></table></div></details>}
        <details><summary>查看检索与指导原始证据</summary><JsonBlock value={item} /></details>
      </article>
    })}</div>
  </>
}

function SteeringTab({ detail }: { detail: RunDetail }) {
  const events = detail.result.events ?? []
  const monitorEvents = events.filter((item) => item.type === 'trajectory_monitor_evaluated')
  const driftEvents = events.filter((item) => item.type === 'trajectory_drift_detected')
  const adviceEvents = events.filter((item) => item.type === 'corrective_advice_generated')
  const interventionEvents = events.filter((item) => item.type === 'intervention_applied')
  const rejected = interventionEvents.filter((item) => item.outcome === 'ACTION_REJECTED')
  const adaptive = record(detail.checkpoint.adaptive)
  const liveplan = record(adaptive.liveplan)
  const lastAdvice = record(liveplan.last_advice)
  const active = record(liveplan.active_intervention)
  const isEnabled = detail.profile.profile_id.includes('liveplan') || monitorEvents.length > 0 || Object.keys(liveplan).length > 0

  if (!isEnabled) return <div className="empty-inline">该运行未启用在线轨迹监控；不会伪造漂移、建议或恢复数据。</div>
  return <>
    <DefinitionGrid items={[
      ['轨迹评估次数', monitorEvents.length],
      ['漂移触发次数', driftEvents.length],
      ['纠偏模型调用', Number(liveplan.advisor_calls ?? 0)],
      ['执行前阻断', rejected.length],
    ]} />
    {Object.keys(lastAdvice).length > 0 ? <section className="steering-advice">
      <div><span>最近纠偏建议</span><b>{String(lastAdvice.reason_code ?? '未分类')}</b></div>
      <p>{String(lastAdvice.guidance ?? '建议内容未记录')}</p>
      <small>{String(lastAdvice.source ?? 'predefined')} · {String(lastAdvice.recommended_action ?? '—')} · 置信度 {String(lastAdvice.confidence ?? '—')}</small>
    </section> : <div className="empty-inline">Monitor 尚未检测到需要纠偏的轨迹漂移。</div>}
    <SectionHeading index="M" title="Monitor 漂移证据" note="确定性规则，不调用模型" />
    {driftEvents.length > 0 ? <div className="table-wrap"><table><thead><tr><th>事件</th><th>漂移信号</th><th>证据序号</th><th>判断依据</th></tr></thead><tbody>
      {driftEvents.map((item, index) => {
        const signals = Array.isArray(item.signals) ? item.signals.map(record) : []
        return <tr key={`${String(item.sequence)}-${index}`}><td>{String(item.sequence ?? '—')}</td><td>{signals.map((signal) => String(signal.code ?? 'UNKNOWN')).join(' / ')}</td><td>{signals.flatMap((signal) => Array.isArray(signal.evidence_sequences) ? signal.evidence_sequences : []).join(', ') || '—'}</td><td>{signals.map((signal) => String(signal.detail ?? '')).join('；') || '—'}</td></tr>
      })}
    </tbody></table></div> : <div className="empty-inline">当前轨迹没有确定性漂移证据。</div>}
    <SectionHeading index="I" title="Advisor 与干预结果" note="建议无执行权限，阻断仅发生在动作提交前" />
    {adviceEvents.length + interventionEvents.length > 0 ? <div className="steering-ledger">
      {[...adviceEvents, ...interventionEvents].sort((left, right) => Number(left.sequence ?? 0) - Number(right.sequence ?? 0)).map((item, index) => {
        const advice = record(item.advice)
        return <article key={`${String(item.sequence)}-${index}`}><span>{String(item.sequence ?? '—')} / {item.type === 'corrective_advice_generated' ? '建议' : '干预'}</span><b>{String(item.outcome ?? advice.reason_code ?? '已记录')}</b><p>{String(advice.guidance ?? (item.outcome === 'ACTION_REJECTED' ? '候选动作与已确认的无进展动作完全相同，已在执行前阻止。' : '干预状态已持久化。'))}</p><small>{String(item.source ?? item.mode ?? 'deterministic')}</small></article>
      })}
    </div> : <div className="empty-inline">尚无 Advisor 或控制器事件。</div>}
    {Object.keys(active).length > 0 && <details><summary>查看当前未消费干预</summary><JsonBlock value={active} /></details>}
  </>
}

function ResultTab({ detail }: { detail: RunDetail }) {
  const steps = detail.result.plan?.steps ?? []
  return <>
    <SectionHeading index="A" title="最终结果" />
    <div className="result-copy">{detail.result.reply ?? '该运行尚未生成最终回复。'}</div>
    <ResultArtifactViews artifacts={detail.result.artifacts} />
    <PolicyAwareCaseExecution detail={detail} />
    {steps.length > 0 ? <div className="table-wrap"><table><thead><tr><th>步骤</th><th>执行角色</th><th>操作</th><th>状态</th></tr></thead><tbody>
      {steps.map((step, index) => <tr key={String(step.step_id ?? index)}><td>{String(step.step_id ?? index + 1)}</td><td>{SPECIALIST_LABELS[String(step.specialist ?? '')] ?? String(step.specialist ?? '—')}</td><td>{TOOL_LABELS[String(step.tool ?? '')] ?? String(step.tool ?? '—')}</td><td>{STATUS_LABELS[String(step.status ?? '')] ?? String(step.status ?? '—')}</td></tr>)}
    </tbody></table></div> : <div className="empty-inline">该路径没有记录结构化计划</div>}
  </>
}

function EvidenceTab({ detail }: { detail: RunDetail }) {
  const checkpoint = detail.checkpoint
  const adaptive = (checkpoint.adaptive ?? {}) as Record<string, unknown>
  const bindingTransaction = (adaptive.binding_tx ?? {}) as Record<string, unknown>
  const requestIr = detail.result.request_ir ?? adaptive.request_ir
  const requestRequirements = requestIr && typeof requestIr === 'object'
    ? (requestIr as { requirements?: unknown[] }).requirements ?? []
    : []
  const failures = Object.values(record(checkpoint.failure_ledger)).map(record)
  const recoveryLabels: Record<string, string> = {
    REPAIR_PROTOCOL: '修正协议', REPAIR_ARGUMENTS: '修正参数', REVISE_INTERPRETATION: '修订需求解释',
    REVISE_BINDING: '修订约束映射', CHANGE_CANDIDATE_OR_EVIDENCE: '更换候选或补充证据', STOP: '停止重复恢复',
  }
  return <>
    <DefinitionGrid items={[
      ['目标修订版本', String(adaptive.goal_revision ?? 0)],
      ['约束映射版本', String(adaptive.binding_revision ?? 0)],
      ['映射事务', bindingTransaction.status ? `${String(bindingTransaction.status)} / ${String(bindingTransaction.source ?? '—')}` : '尚无事务'],
      ['模型已读证据', Array.isArray(adaptive.read_evidence_refs) ? adaptive.read_evidence_refs.length : 0],
      ['目标数量', Array.isArray(adaptive.objective_ledger) ? adaptive.objective_ledger.length : 0],
      ['语义要求', requestRequirements.length],
      ['失败身份', failures.length],
      ['上下文', checkpoint.context ? '已记录' : '未记录'],
    ]} />
    <div className="split-json"><div><h3>统一请求语义</h3><JsonBlock value={requestIr ?? { status: '尚未生成 RequestIR' }} /></div><div><h3>目标、映射事务与证据</h3><JsonBlock value={{ goal: adaptive.goal, objective_ledger: adaptive.objective_ledger ?? [], constraint_bindings: adaptive.constraint_bindings ?? {}, binding_transaction: adaptive.binding_tx ?? null, binding_resolution: adaptive.binding_resolution ?? null, binding_evidence_access: adaptive.binding_evidence_access ?? {}, working_memory: adaptive.working_memory, read_evidence_refs: adaptive.read_evidence_refs ?? [] }} /></div></div>
    <SectionHeading index="R" title="失败与有界恢复" />
    {failures.length ? <div className="table-wrap"><table><thead><tr><th>阶段 / 错误</th><th>失败次数</th><th>恢复动作</th><th>身份指纹</th></tr></thead><tbody>
      {failures.map((failure) => {
        const signature = record(failure.signature)
        return <tr key={String(failure.signature_sha256)}>
          <td><small>{String(signature.stage ?? '—')}</small><br /><code>{String(signature.code ?? '—')}</code></td>
          <td>{String(failure.occurrence ?? '—')} 次 · 最多恢复 {String(failure.max_recoveries ?? '—')} 次</td>
          <td>{recoveryLabels[String(failure.recovery_action)] ?? String(failure.recovery_action ?? '—')}</td>
          <td><code title={String(failure.signature_sha256)}>{String(failure.signature_sha256 ?? '').slice(0, 16)}</code></td>
        </tr>
      })}
    </tbody></table></div> : <div className="empty-inline">本次运行尚未记录共享恢复事件。</div>}
    <details><summary>恢复账本与结构化上下文</summary><JsonBlock value={{ failure_ledger: checkpoint.failure_ledger ?? {}, context: checkpoint.context ?? null }} /></details>
  </>
}

function ToolsTab({ detail }: { detail: RunDetail }) {
  const trace = detail.result.trace ?? []
  return <>
    <DefinitionGrid items={[["轨迹条目", trace.length], ["持久化账本条目", detail.tool_ledger.length]]} />
    {trace.length === 0 && detail.tool_ledger.length === 0 && <div className="empty-inline">该运行没有工具调用</div>}
    <div className="tool-grid">{trace.map((entry, index) => <article key={`${entry.tool}-${index}`}><span>{String(index + 1).padStart(2, '0')} / 工具</span><h3>{TOOL_LABELS[entry.tool ?? ''] ?? entry.tool}</h3><code>{entry.tool}</code><div><small>调用参数</small><JsonBlock value={entry.args ?? {}} /></div><div><small>执行结果</small><JsonBlock value={entry.result} /></div></article>)}</div>
    {detail.tool_ledger.length > 0 && <><SectionHeading index="L" title="持久化工具账本" /><JsonBlock value={detail.tool_ledger} /></>}
  </>
}
