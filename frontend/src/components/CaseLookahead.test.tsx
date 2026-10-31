import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { LookaheadReport, RunEvent } from '../types'
import { CaseLookahead } from './CaseLookahead'

const report: LookaheadReport = {
  version: 'case-lookahead-v1', report_id: 'report', state_id: 'state-1', graph_id: 'graph', portfolio_id: 'portfolio',
  status: 'SELECTED', reason_code: 'BOUNDED_INFORMATION_COVERAGE', horizon: 2, expanded_paths: 4,
  selected_candidate_id: 'serial', assumption: 'SUFFICIENT_INFORMATION_NOT_OBSERVED',
  paths: [
    { candidate_ids: ['serial', 'packaging'], requirement_ids: ['SERIAL', 'PACKAGING'], remaining_if_sufficient: [],
      remaining_if_no_information: ['SERIAL', 'PACKAGING'], user_turns: 2, tool_calls: 0, response_burden: 3, first_action_burden: 1 },
    { candidate_ids: ['packaging', 'serial'], requirement_ids: ['PACKAGING', 'SERIAL'], remaining_if_sufficient: [],
      remaining_if_no_information: ['SERIAL', 'PACKAGING'], user_turns: 2, tool_calls: 0, response_burden: 3, first_action_burden: 2 },
  ],
}
const events: RunEvent[] = [
  { type: 'case_state_projected', sequence: 1, state: { case_id: 'case', state_id: 'state-1' } },
  { type: 'case_lookahead_solved', sequence: 2, case_id: 'case', case_revision: 1, report },
  { type: 'case_obligation_graph_solved', sequence: 3, graph: { graph_id: 'graph', candidates: [
    { candidate_id: 'serial', question: '序列号标签完整吗？', mode: 'ASK_USER' },
    { candidate_id: 'packaging', question: '原包装和附件是否齐全？', mode: 'ASK_USER' },
  ] } },
]

describe('CaseLookahead', () => {
  it('separates selected first action, hypothetical continuation and actual observation', () => {
    render(<CaseLookahead events={events} />)
    expect(screen.getByText('已选首步')).toBeInTheDocument()
    expect(screen.getByText('备选首步')).toBeInTheDocument()
    expect(screen.getAllByText('假设后续')).toHaveLength(2)
    expect(screen.getAllByText('仍有 2 项义务')).toHaveLength(2)
    expect(screen.getByText(/本记录未观察到首步工具完成/)).toBeInTheDocument()
    expect(screen.getByText(/不是成功概率/)).toBeInTheDocument()
  })

  it('switches historical records and marks observations that supersede the source state', () => {
    render(<CaseLookahead events={[...events,
      { type: 'case_state_projected', sequence: 4, state: { case_id: 'case', state_id: 'state-2' } },
      { type: 'case_lookahead_solved', sequence: 5, case_id: 'case', case_revision: 2,
        report: { ...report, state_id: 'state-2', status: 'NOT_REQUIRED', paths: [], selected_candidate_id: null, expanded_paths: 0 } },
    ]} />)
    expect(screen.getByText(/当前没有需要前瞻的信息动作/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('选择前瞻记录'), { target: { value: '2' } })
    expect(screen.getByText('历史状态 · 后续已重新投影')).toBeInTheDocument()
    expect(screen.getByText('已选首步')).toBeInTheDocument()
  })

  it('does not mistake another case or revision for completion', () => {
    const completion = { type: 'case_autonomous_action_completed', sequence: 4, candidate_id: 'serial', graph_id: 'graph', from_revision: 1 }
    const { rerender } = render(<CaseLookahead events={[...events, { ...completion, case_id: 'other' }]} />)
    expect(screen.getByText(/未观察到首步工具完成/)).toBeInTheDocument()
    rerender(<CaseLookahead events={[...events, { ...completion, case_id: 'case' }]} />)
    expect(screen.getByText(/首步工具已完成，后续以新状态为准/)).toBeInTheDocument()
  })

  it('shows a persisted clarification as a question, not completed evidence', () => {
    render(<CaseLookahead events={[...events, { type: 'case_clarification_opened', sequence: 4,
      case_id: 'case', state_id: 'state-1', portfolio_id: 'portfolio' }]} />)
    expect(screen.getByText(/首步澄清问题已记录；是否满足义务仍需回答核验/)).toBeInTheDocument()
  })

  it('shows missing, partial and invalid records without fabricating paths', () => {
    const { rerender } = render(<CaseLookahead events={[]} />)
    expect(screen.getByText(/尚无前瞻决策记录/)).toBeInTheDocument()
    rerender(<CaseLookahead events={[events[1]]} />)
    expect(screen.getByRole('status')).toHaveTextContent('部分记录缺失')
    rerender(<CaseLookahead events={[{ ...events[1], report: { ...report, expanded_paths: -1 } }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('前瞻记录不完整')
  })

  it('explains fallback without pretending to have searched a partial candidate set', () => {
    render(<CaseLookahead events={[{ ...events[1], report: { ...report, status: 'FALLBACK', paths: [], expanded_paths: 0,
      reason_code: 'LOOKAHEAD_CANDIDATE_LIMIT' } }]} />)
    expect(screen.getByText(/已回退原有安全策略/)).toBeInTheDocument()
    expect(screen.queryByText('已选首步')).not.toBeInTheDocument()
  })
})
