import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { CaseGoal, RunEvent } from '../types'
import { CaseGoalRevision } from './CaseGoalRevision'

const draft: CaseGoal = { case_id: 'case-a', order_id: 1001, mode: 'DRAFT', revision: 1, run_id: 'run-a', question: null,
  source: { source: 'CURRENT_TURN', history_index: null, start: 0, end: 5, text: '先保存草稿', source_sha256: 'hash' } }
const event: RunEvent = { type: 'case_goal_revised', sequence: 1, case_id: 'case-a', previous: null, current: draft,
  preserved: ['已入账材料'], suspended: ['提交确认'], next_steps: ['保存本轮明确材料'] }

describe('CaseGoalRevision', () => {
  it('shows source, changed goal and planned effects without inventing completed writes', () => {
    render(<CaseGoalRevision events={[event]} goals={{ 'case-a': draft }} />)
    expect(screen.getByText('先保存草稿')).toBeInTheDocument()
    expect(screen.getByText('仅保存草稿')).toBeInTheDocument()
    expect(screen.getByText('已入账材料')).toBeInTheDocument()
    expect(screen.getByText('目标已保存到会话')).toBeInTheDocument()
    expect(screen.getByText(/以工具账本为准/)).toBeInTheDocument()
    expect(screen.queryByText(/项材料操作完成/)).not.toBeInTheDocument()
  })
  it('matches execution by case, goal revision and originating run', () => {
    const applied: RunEvent = { type: 'case_goal_applied', case_id: 'other', goal: draft, material_observations: [{}, {}] }
    const { rerender } = render(<CaseGoalRevision events={[event, applied]} />)
    expect(screen.queryByText(/2 项材料操作完成/)).not.toBeInTheDocument()
    rerender(<CaseGoalRevision events={[event, { ...applied, case_id: 'case-a' }]} />)
    expect(screen.getByText(/2 项材料操作完成/)).toBeInTheDocument()
  })
  it('lets users inspect an older revision and marks it historical', () => {
    render(<CaseGoalRevision events={[event, { ...event, sequence: 3, previous: draft, current: { ...draft, revision: 2, mode: 'CONTINUE' } }]} />)
    expect(screen.getByText('继续处理')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('选择目标修订记录'), { target: { value: '1' } })
    expect(screen.getByText(/历史修订/)).toBeInTheDocument()
  })
  it('handles loading, unchanged state, missing and malformed evidence', () => {
    const { rerender } = render(<CaseGoalRevision events={[]} running />)
    expect(screen.getByRole('status')).toHaveTextContent('正在理解')
    rerender(<CaseGoalRevision events={[]} goals={{ 'case-a': draft }} />)
    expect(screen.getByText(/沿用仅保存草稿/)).toBeInTheDocument()
    rerender(<CaseGoalRevision events={[{ ...event, current: {} }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('不完整')
    rerender(<CaseGoalRevision events={[{ type: 'case_goal_revised', case_id: 'case-a', current: draft }]} />)
    expect(screen.getByText('历史记录不完整')).toBeInTheDocument()
    expect(screen.getAllByText('尚无对应记录。')).toHaveLength(3)
  })
})
