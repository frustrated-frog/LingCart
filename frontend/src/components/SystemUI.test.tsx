import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { HarnessProfile } from '../types'
import { ExecutionSpine, localizeEngineeringText, MechanismRail, StatusBadge } from './SystemUI'

function profile(id: string, planning: string): HarnessProfile {
  return {
    profile_id: id,
    version: '1.0.0',
    display_name: id,
    description: 'test profile',
    engine: { id: 'runtime', version: '1' },
    mechanisms: {
      planning: { slot: 'planning', id: planning, version: '1' },
      context: { slot: 'context', id: 'bounded-context', version: '1' },
      memory: { slot: 'memory', id: 'durable-memory', version: '1' },
      verification: { slot: 'verification', id: 'evidence-gate', version: '1' },
      tool_protocol: { slot: 'tool_protocol', id: 'typed-tools', version: '1' },
    },
    capabilities: [],
    research: { status: 'baseline', source_type: 'native', paper_id: null, hypothesis: null },
    fingerprint: 'profile-fingerprint',
    execution_sha256: 'execution-fingerprint',
    source_revision: null,
    resolved_mechanisms: {},
    product_eligible: true,
  }
}

describe('MechanismRail', () => {
  it('renders five mechanism slots and marks an experimental delta', () => {
    const { container } = render(<MechanismRail profile={profile('candidate', 'paper-planner')} reference={profile('baseline', 'adaptive-planner')} />)
    expect(screen.getByText('paper-planner')).toBeInTheDocument()
    expect(container.querySelectorAll('.mechanism-cell')).toHaveLength(5)
    expect(container.querySelectorAll('.mechanism-cell.changed')).toHaveLength(1)
  })

  it('shows a Chinese mechanism name while retaining its reproducible identifier', () => {
    render(<MechanismRail profile={profile('baseline', 'bounded-model-loop')} />)
    expect(screen.getByText('有界模型循环')).toBeInTheDocument()
    expect(screen.getByText(/bounded-model-loop/)).toBeInTheDocument()
  })
})

describe('ExecutionSpine', () => {
  it('projects event semantics and reports selection', () => {
    const select = vi.fn()
    render(<ExecutionSpine events={[
      { type: 'intake', sequence: 1 },
      { type: 'tool_call', sequence: 2, tool: 'get_order_status' },
      { type: 'terminal', sequence: 3, status: 'SUCCEEDED' },
    ]} selected={0} onSelect={select} />)
    expect(screen.getByText('查询订单状态')).toBeInTheDocument()
    fireEvent.click(screen.getByText('已完成'))
    expect(select).toHaveBeenCalledWith(2)
  })

  it('keeps an explicit empty evidence state', () => {
    render(<ExecutionSpine events={[]} />)
    expect(screen.getByText('暂未记录执行事件')).toBeInTheDocument()
  })
})

describe('engineering copy localization', () => {
  it('translates descriptive engineering terms without changing protocol identifiers', () => {
    expect(localizeEngineeringText('Agent 通过 Harness Profile 加载 Playbook'))
      .toBe('智能体通过运行配置档案加载策略手册')
  })
})

describe('StatusBadge', () => {
  it('translates runtime status without hiding the semantic class', () => {
    const { container } = render(<StatusBadge status="WAITING_APPROVAL" />)
    expect(screen.getByText('等待确认')).toBeInTheDocument()
    expect(container.firstChild).toHaveClass('warn')
  })
})
