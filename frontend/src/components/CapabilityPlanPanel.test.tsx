import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { CapabilityPlanView, CapabilityPlanningProjection } from '../types'
import { CapabilityPlanPanel } from './CapabilityPlanPanel'

const initial: CapabilityPlanView = {
  schema_version: 1, plan_id: 'first', catalog_sha256: 'catalog', goal_revision: 0, revision: 1,
  previous_plan_id: null, status: 'READY', reason: '根据目标与工具能力建立初始计划', facts: {}, missing: [],
  delegated_objectives: [], objective_labels: { refund: '判断订单是否满足退款条件' }, expanded: 3, completion_verified: false,
  nodes: [
    { action_id: 'details', tool: 'get_order_details', scope: 'order:1001', objective_ids: ['refund'],
      requires: [], provides: ['order:1001/order.status'], depends_on: [], evidence_refs: [], status: 'READY', automatic: true },
    { action_id: 'refund', tool: 'check_refund_eligibility', scope: 'order:1001', objective_ids: ['refund'],
      requires: ['order:1001/order.status'], provides: ['order:1001/order.refund_check'], depends_on: ['details'],
      evidence_refs: [], status: 'PLANNED', automatic: true },
  ],
}
const current: CapabilityPlanView = { ...initial, plan_id: 'second', revision: 2, previous_plan_id: 'first',
  reason: '根据实际观察重算剩余依赖', facts: { 'order:1001/order.status': ['step-1'] },
  nodes: [{ ...initial.nodes[1], status: 'READY', depends_on: [], evidence_refs: ['step-1'] }],
}
const projection: CapabilityPlanningProjection = { status: 'READY', history: [initial, current], stale: false, invalid_records: 0 }

describe('CapabilityPlanPanel', () => {
  it('shows actual observations and permits inspecting historical dependencies', () => {
    render(<CapabilityPlanPanel projection={projection} />)
    expect(screen.getByText('根据实际观察重算剩余依赖')).toBeInTheDocument()
    expect(screen.getByText('step-1')).toBeInTheDocument()
    expect(screen.getByText('来源：step-1')).toBeInTheDocument()
    expect(screen.getByText('预计提供 · 尚未执行')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('计划版本'), { target: { value: 'first' } })
    expect(screen.getByText(/正在查看历史计划/)).toBeInTheDocument()
    const node = screen.getByRole('button', { name: /等待前置结果/ })
    fireEvent.click(node)
    expect(node).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('判断订单是否满足退款条件')).toBeInTheDocument()
    expect(screen.queryByText('来源：step-1')).not.toBeInTheDocument()
  })

  it('shows empty, malformed, stale and partial states without pretending success', () => {
    const view = render(<CapabilityPlanPanel />)
    expect(screen.getByRole('status')).toHaveTextContent('尚无能力规划记录')
    view.rerender(<CapabilityPlanPanel projection={{ ...projection, history: [], status: 'PARTIAL', invalid_records: 1 }} />)
    expect(screen.getByRole('status')).toHaveTextContent('记录无法解析')
    view.rerender(<CapabilityPlanPanel projection={{ ...projection, status: 'PARTIAL', invalid_records: 1, stale: true }} />)
    expect(screen.getByRole('alert')).toHaveTextContent('1 条规划记录不完整')
    expect(screen.getByRole('status')).toHaveTextContent('目标已经变化')
  })

  it('distinguishes observations ready from completion and displays missing paths', () => {
    const view = render(<CapabilityPlanPanel projection={{ ...projection, history: [{ ...current,
      status: 'OBSERVATIONS_READY', nodes: [], delegated_objectives: ['refund'] }] }} />)
    expect(screen.getByText('所需观察已齐备')).toBeInTheDocument()
    expect(screen.getByText(/仍需模型处理：判断订单/)).toBeInTheDocument()
    view.rerender(<CapabilityPlanPanel projection={{ ...projection, history: [{ ...current, status: 'UNREACHABLE', nodes: [],
      missing: ['order:1001/order.refund_check'] }] }} />)
    expect(screen.getByText('当前能力无法形成路径')).toBeInTheDocument()
    expect(screen.getByText(/缺少可行路径：订单 1001/)).toBeInTheDocument()
  })
})
