import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CaseModelDecision } from './CaseModelDecision'
import type { RunEvent } from '../types'

const accepted: RunEvent = { type: 'case_model_decision', sequence: 2, version: 'case-model-decision-v1',
  case_id: 'case', case_revision: 1, state_id: 'state', graph_id: 'graph', directive_id: 'directive',
  status: 'ACCEPTED', observed: false, decision: { action: 'SELECT', candidate_id: 'candidate',
    source_quote: '请先核对原包装', reason: '先处理用户关注的材料。', question: null,
    expected_observation: '获得包装状态', replan_when: '收到真实回答后重新选择' } }
const graph: RunEvent = { type: 'case_obligation_graph_solved', sequence: 3, graph: { graph_id: 'graph',
  candidates: [{ candidate_id: 'candidate', mode: 'ASK_USER', requirement_id: 'packaging', question: '原包装是否完整？' }] } }

describe('CaseModelDecision', () => {
  it('shows source, selected action and hypothetical observation separately', () => {
    render(<CaseModelDecision events={[accepted, graph]} />)
    expect(screen.getByText('请先核对原包装')).toBeInTheDocument()
    expect(screen.getByText('原包装是否完整？')).toBeInTheDocument()
    expect(screen.getByText('后续计划 · 尚未发生')).toBeInTheDocument()
    expect(screen.getByText(/尚未观察到对应工具完成/)).toBeInTheDocument()
  })
  it('requires same case and directive for actual completion', () => {
    const done = { type: 'case_autonomous_action_completed', sequence: 4, case_id: 'other', directive_id: 'directive' }
    const { rerender } = render(<CaseModelDecision events={[accepted, graph, done]} />)
    expect(screen.getByText(/尚未观察到对应工具完成/)).toBeInTheDocument()
    rerender(<CaseModelDecision events={[accepted, graph, { ...done, case_id: 'case' }]} />)
    expect(screen.getByText(/已观察到对应工具完成/)).toBeInTheDocument()
  })
  it('shows failed validation and can inspect an earlier decision as stale', () => {
    render(<CaseModelDecision events={[accepted, graph,
      { type: 'case_state_projected', sequence: 4, state: { case_id: 'case', state_id: 'new' } },
      { type: 'case_model_decision_rejected', sequence: 5, case_id: 'case', state_id: 'new', code: 'INVALID_CASE_MODEL_DECISION', feedback: '未知动作' },
    ]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('未据此执行')
    fireEvent.change(screen.getByLabelText('选择模型决策记录'), { target: { value: '2' } })
    expect(screen.getByText('历史决策 · 后续状态已变化')).toBeInTheDocument()
  })
  it('shows empty, partial, malformed and goal-clarification states truthfully', () => {
    const { rerender } = render(<CaseModelDecision events={[]} />)
    expect(screen.getByText(/尚无模型行动决策/)).toBeInTheDocument()
    rerender(<CaseModelDecision events={[accepted]} />)
    expect(screen.getByRole('status')).toHaveTextContent('部分记录缺失')
    rerender(<CaseModelDecision events={[{ ...accepted, decision: {} }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('不完整')
    rerender(<CaseModelDecision events={[{ ...accepted, decision: { ...(accepted.decision as object), action: 'CLARIFY_GOAL', candidate_id: null, question: '你希望了解维修还是申请退款？' } }]} />)
    expect(screen.getByText(/本决策要求先澄清目标/)).toBeInTheDocument()
  })
})
