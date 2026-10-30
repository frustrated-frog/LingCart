import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { RunDetail, RunSummary } from '../types'
import { RunsPage } from './RunsPage'

const summary: RunSummary = {
  run_id: 'run-liveplan', conversation_id: 'conversation-1', request: '推荐预算 500 元的耳机',
  created_at: '2026-10-20T10:00:00Z', status: 'WAITING_CLARIFICATION', agent: 'adaptive',
  terminal_reason: 'MODEL_STOPPED', profile: { profile_id: 'adaptive-liveplan-v1' }, event_count: 5,
  tool_calls: 2, model_calls: 3, latency_ms: 100, tokens: 80,
}

const detail: RunDetail = {
  ...summary,
  result: {
    reply: '需要补充佩戴偏好。',
    events: [
      { type: 'trajectory_monitor_evaluated', sequence: 1 },
      { type: 'trajectory_drift_detected', sequence: 2, signals: [{
        code: 'ACTION_OSCILLATION', evidence_sequences: [1, 2], detail: '相同查询没有增加权威证据。',
      }] },
      { type: 'corrective_advice_generated', sequence: 3, source: 'predefined', advice: {
        reason_code: 'ACTION_OSCILLATION', guidance: '改用不同检索条件。',
      } },
      { type: 'intervention_applied', sequence: 4, outcome: 'ACTIVATED', mode: 'CONSTRAIN_NEXT_ACTION' },
      { type: 'intervention_applied', sequence: 5, outcome: 'ACTION_REJECTED', mode: 'CONSTRAIN_NEXT_ACTION' },
    ],
    trace: [],
  },
  checkpoint: { adaptive: { liveplan: {
    advisor_calls: 0,
    last_advice: {
      reason_code: 'ACTION_OSCILLATION', guidance: '改用不同检索条件。', source: 'predefined',
      recommended_action: 'CHANGE_APPROACH', confidence: 1,
    },
  } } },
  tool_ledger: [], commands: [],
  trace_projection: { integrity: {}, summary: {}, timeline: [] },
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/runs/run-liveplan']}>
        <Routes><Route path="/runs/:runId" element={<RunsPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => vi.restoreAllMocks())

describe('RunsPage LivePlan evidence', () => {
  it('shows persisted monitor evidence, advice and precommit rejection', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(detail)

    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '在线监控与纠偏' }))

    expect(screen.getByText('相同查询没有增加权威证据。')).toBeInTheDocument()
    expect(screen.getAllByText('ACTION_OSCILLATION').length).toBeGreaterThan(0)
    expect(screen.getAllByText('改用不同检索条件。').length).toBeGreaterThan(0)
    expect(screen.getByText('候选动作与已确认的无进展动作完全相同，已在执行前阻止。')).toBeInTheDocument()
  })

  it('shows the versioned request semantics and grounded source span', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue({ ...detail, result: {
      ...detail.result,
      request_ir: {
        version: 'request-ir/v1', sha256: 'a'.repeat(64), requirements: [{
          requirement_id: 'context:0', kind: 'PROHIBITION', description: '禁止支付',
          objective_ids: [], constraint_refs: [], source: {
            source: 'CURRENT_TURN', history_index: null, start: 16, end: 22,
            text: '不要执行支付', source_sha256: 'b'.repeat(64),
          },
        }],
      },
    } })

    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '上下文与证据' }))

    expect(screen.getByText('统一请求语义')).toBeInTheDocument()
    expect(screen.getAllByText('1').length).toBeGreaterThan(0)
    expect(screen.getByText(/PROHIBITION/)).toBeInTheDocument()
    expect(screen.getByText(/不要执行支付/)).toBeInTheDocument()
  })
})

describe('RunsPage policy-aware case execution evidence', () => {
  it('shows authoritative progress and the explicit approval boundary', async () => {
    const caseDetail: RunDetail = { ...detail, status: 'WAITING_APPROVAL', result: { trace: [], events: [
      { type: 'case_state_projected', sequence: 1, state: {
        case_id: 'CASE-1', case_revision: 3, policy_version_id: 'policy-v1',
        policy_decision_id: 'decision-1', plan_id: 'plan-1', plan_revision: 3,
      } },
      { type: 'case_obligation_graph_solved', sequence: 2, graph: {
        nodes: [{ kind: 'REQUIREMENT' }, { kind: 'REQUIREMENT' }, { kind: 'RESOLUTION' }],
        candidates: [{ mode: 'AUTONOMOUS' }, { mode: 'ASK_USER' }],
      }, selection: {
        selected_candidate_id: 'candidate-authoritative',
        reason_code: 'TRUSTED_AUTONOMOUS_ACTION_DOMINATES',
      } },
      { type: 'case_clarification_portfolio_solved', sequence: 3, portfolio: {
        questions: [{
          question_id: 'question-packaging',
          question: '请说明原包装是否完整，以及附件是否齐全？',
        }, {
          question_id: 'question-label',
          question: '请确认序列号标签是否清晰？',
        }],
      }, decision: {
        selected_question_id: 'question-packaging',
        reason_code: 'ACTIVE_CLARIFICATION_MAX_STRUCTURAL_VALUE',
      } },
      { type: 'case_clarification_opened', sequence: 4, question_id: 'question-packaging',
        question: '请说明原包装是否完整，以及附件是否齐全？', case_revision: 2 },
      { type: 'case_autonomous_action_selected', sequence: 5, operation: 'continue_after_sales_case',
        reason_code: 'AUTHORITATIVE_EVIDENCE_AVAILABLE', case_revision: 2 },
      { type: 'case_autonomous_action_completed', sequence: 6, operation: 'continue_after_sales_case',
        from_revision: 2, to_revision: 3 },
      { type: 'case_clarification_answer_bound', sequence: 7, requirement_id: 'policy:ORIGINAL_PACKAGING',
        authorization_granted: false },
      { type: 'case_execution_paused', sequence: 8, status: 'WAIT_APPROVAL',
        reason_code: 'EXPLICIT_APPROVAL_REQUIRED', case_revision: 3 },
    ] } }
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(caseDetail)

    renderPage()

    expect(await screen.findByText('政策感知自主执行')).toBeInTheDocument()
    expect(screen.getAllByText('等待用户明确确认').length).toBeGreaterThan(0)
    expect(screen.getByText('政策允许读取权威订单事实')).toBeInTheDocument()
    expect(screen.getByText('退款等写操作必须由用户明确确认')).toBeInTheDocument()
    expect(screen.getAllByText('2 个候选行动').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2 个澄清问题').length).toBeGreaterThan(0)
    expect(screen.getByText('权威工具动作无需打断用户，优先于其它候选')).toBeInTheDocument()
    expect(screen.getByText('从非支配问题中选择政策解锁与未知消除价值最高的一项')).toBeInTheDocument()
    expect(screen.getByText('选中问题：请说明原包装是否完整，以及附件是否齐全？')).toBeInTheDocument()
    expect(screen.getByText('待回答问题已持久化')).toBeInTheDocument()
    expect(screen.getByText('回答已绑定，等待独立核验')).toBeInTheDocument()
    expect(screen.getByText('本轮原文只作为证据候选，是否满足义务仍由语义核验与政策裁决决定。')).toBeInTheDocument()
    expect(screen.getByText('REV 2 → REV 3')).toBeInTheDocument()
  })
})

describe('RunsPage PG evidence', () => {
  it('distinguishes generated guidance, exposure, budget skip and committed action', async () => {
    const pgDetail: RunDetail = { ...detail, checkpoint: { context: { procedural_graph: {
      graph_id: 'lingcart-pg', version: 1, sha256: 'frozen-graph',
    } } }, result: { trace: [], events: [
      { type: 'model_started', sequence: 1, purpose: 'pg_guidance' },
      { type: 'pg_guidance', sequence: 2, decision_key: 'first', graph_sha256: 'frozen-graph',
        anchor: 'search_products', status: 'READY', guidance: '空结果后扩大检索范围', edge_ids: ['search-retry'], fallback: false },
      { type: 'evidence_context', sequence: 3, pg_decision_key: 'first', pg_status: 'EXPOSED' },
      { type: 'adaptive_decision', sequence: 4, decision: { action: 'tool', tool: 'search_products' } },
      { type: 'pg_guidance', sequence: 5, decision_key: 'second', anchor: 'unknown_action', fallback: true,
        status: 'SKIPPED', guidance: '', edge_ids: [], reason: 'RESERVED_DECISION_AND_AUDIT' },
    ] } }
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(pgDetail)
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('已送入决策')).toBeInTheDocument()
    expect(screen.getByText('空结果后扩大检索范围')).toBeInTheDocument()
    expect(screen.getByText('为决策和独立核验保留预算')).toBeInTheDocument()
    expect(screen.getByText(/引用关系：search-retry · 后续行动：search_products/)).toBeInTheDocument()
    expect(screen.getByText(/全图回退/)).toBeInTheDocument()
  })

  it('shows an empty state when no graph has been bound', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue(detail)
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('该运行尚无程序图指导记录。')).toBeInTheDocument()
  })

  it('does not imply graph exposure when guidance was omitted by context budget', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue({ ...detail, result: { trace: [], events: [
      { type: 'pg_guidance', sequence: 1, decision_key: 'omitted', status: 'READY', guidance: '建议内容', edge_ids: ['edge-a'] },
      { type: 'evidence_context', sequence: 2, pg_decision_key: 'omitted', pg_status: 'CONTEXT_OMITTED' },
    ] } })
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText('上下文不足，未送入决策')).toBeInTheDocument()
    expect(screen.queryByText('已送入决策')).not.toBeInTheDocument()
  })

  it('shows v2 low-confidence abstention and the non-acting v1 shadow result', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue({ ...detail, checkpoint: { context: { procedural_graph: {
      graph_id: 'lingcart-pg', version: 1, sha256: 'frozen-graph',
    } } }, result: { trace: [], events: [
      { type: 'pg_guidance', sequence: 1, decision_key: 'abstained', status: 'SKIPPED',
        anchor: null, fallback: false, abstained: true, guidance: '', edge_ids: [], reason: 'LOCALIZATION_ABSTAINED',
        localization: { algorithm: 'trajectory-state-v2', confidence: 0,
          shadow: { anchor: 'answer_faq', fallback: true } } },
    ] } })
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getAllByText(/主动弃权/).length).toBeGreaterThan(0)
    expect(screen.getByText('轨迹证据不足，定位器主动弃权')).toBeInTheDocument()
    expect(screen.getByText(/v1 影子：answer_faq（全图回退）/)).toBeInTheDocument()
  })

  it('shows deterministic contract adoption review and precommit blocking', async () => {
    vi.spyOn(api, 'runs').mockResolvedValue({ runs: [summary] })
    vi.spyOn(api, 'run').mockResolvedValue({ ...detail, checkpoint: { context: { procedural_graph: {
      graph_id: 'lingcart-pg', version: 1, sha256: 'frozen-graph',
    } } }, result: { trace: [], events: [
      { type: 'pg_guidance', sequence: 1, decision_key: 'contract-decision', status: 'READY',
        anchor: 'search_products', fallback: false, guidance: '不得原样重复无进展调用', edge_ids: ['search-inspect'],
        generation: { mode: 'deterministic-contract-v1', model_calls: 0 },
        applicability: { strategy: 'typed-evidence-v1', status: 'APPLICABLE', selected_edge_ids: ['search-inspect'],
          false_edge_ids: [], unknown_edge_ids: [], evaluations: [{ edge_id: 'search-inspect', source: 'search_products',
            target: 'inspect', status: 'TRUE', reason: 'CONDITION_SATISFIED', evaluation: { evidence_bindings: [
              { fact: 'trajectory.latest_outcome', actual: 'NON_EMPTY', source: 'trajectory_state.recent_actions[-1].outcome' },
            ] } }] },
        preplay: { plan_id: 'preplay-plan-1', status: 'READY', reason: 'BOUNDED_PATH_SELECTED',
          selected_path: { first_action: 'inspect', score: 0.811, edge_ids: ['search-inspect', 'inspect-recommend'] } },
        guidance_contract: { contract_id: 'contract-12345678', policy: 'ADVISORY_EXCEPT_EXACT_NO_PROGRESS_REPEAT' } },
      { type: 'evidence_context', sequence: 2, pg_decision_key: 'contract-decision', pg_status: 'EXPOSED' },
      { type: 'pg_decision_reviewed', sequence: 3, decision_key: 'contract-decision', status: 'BLOCKED',
        blocked: true, actual_action: 'search_products', review_id: 'review-1' },
      { type: 'pg_prediction_outcome', sequence: 4, preplay_plan_id: 'preplay-plan-1', status: 'DIVERGED',
        expected_action: 'inspect', actual_action: 'search_products' },
    ] } })
    renderPage()
    fireEvent.click(await screen.findByRole('tab', { name: '程序图指导' }))
    expect(screen.getByText(/生成方式：确定性指导契约/)).toBeInTheDocument()
    expect(screen.getByText(/额外模型调用：0/)).toBeInTheDocument()
    expect(screen.getByText(/后续行动：search_products · 采用审查：BLOCKED（提交前阻断）/)).toBeInTheDocument()
    expect(screen.getByText(/条件证据：APPLICABLE · TRUE 1 · FALSE 0 · UNKNOWN 0/)).toBeInTheDocument()
    expect(screen.getByText('有界程序预演')).toBeInTheDocument()
    expect(screen.getByText('当前可执行')).toBeInTheDocument()
    expect(screen.getByText('预测路径')).toBeInTheDocument()
    expect(screen.getByText(/预测核对：DIVERGED/)).toBeInTheDocument()
    expect(screen.getByText('查看每条关系的条件求值证据')).toBeInTheDocument()
  })
})
