import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ExecutionModeSwitch, TaskPulseRail, WorkspaceEvidence } from './WorkspaceUI'

describe('WorkspaceUI', () => {
  it('exposes execution mode as a keyboard-operable switch', () => {
    const onChange = vi.fn()
    render(<ExecutionModeSwitch adaptive={false} disabled={false} onChange={onChange} />)

    const control = screen.getByRole('switch', { name: '自适应规划' })
    expect(control).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(control)
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('projects task phases only from recorded run events', () => {
    const { rerender } = render(<TaskPulseRail events={[]} />)
    const phaseItems = screen.getAllByRole('listitem')
    expect(phaseItems.every((item) => item.classList.contains('waiting'))).toBe(true)

    rerender(<TaskPulseRail status="SUCCEEDED" events={[
      { type: 'intake', sequence: 1 },
      { type: 'plan_created', sequence: 2 },
      { type: 'tool_result', sequence: 3 },
      { type: 'verification', sequence: 4 },
      { type: 'terminal', sequence: 5, status: 'SUCCEEDED' },
    ]} />)

    expect(screen.getAllByRole('listitem').every((item) => item.classList.contains('complete'))).toBe(true)
    expect(screen.getByText('5 条真实事件')).toBeInTheDocument()
    expect(screen.getByText('已完成')).toBeInTheDocument()
  })

  it('keeps the evidence rail empty until a real run exists', () => {
    render(<WorkspaceEvidence />)
    expect(screen.getByText('尚无执行证据')).toBeInTheDocument()
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })

  it('does not reuse evidence from a previous result while a new run is pending', () => {
    render(<WorkspaceEvidence pending result={{
      run_id: 'old-run', reply: '旧结果', agent: 'order', route: 'DIRECT', status: 'SUCCEEDED',
      terminal_reason: 'COMPLETED', events: [], trace: [{ tool: 'get_order_status' }],
      pending_action: null, usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
    }} />)

    expect(screen.getByText('待执行')).toBeInTheDocument()
    expect(screen.getByText('等待分配')).toBeInTheDocument()
    expect(screen.getAllByText('执行中')).toHaveLength(2)
    expect(screen.queryByText('old-run')).not.toBeInTheDocument()
    expect(screen.queryByText('5')).not.toBeInTheDocument()
  })
})
