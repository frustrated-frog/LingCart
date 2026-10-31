import {
  Check,
  CircleDashed,
  FileCheck2,
  Gauge,
  ListTree,
  Route,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import type { ExecutionResult, RunEvent } from '../types'
import { STATUS_LABELS } from './SystemUI'

const phases = [
  {
    id: 'understand',
    label: '理解需求',
    note: '目标与约束',
    icon: Sparkles,
    events: new Set([
      'intake', 'dialogue_protocol_retry', 'route_decision', 'request_audited', 'request_ir_projected',
      'request_semantics_extracted', 'request_semantics_compiled', 'request_semantics_revised', 'execution_boundaries_updated', 'execution_boundary_blocked',
      'catalog_request_audited', 'constraint_binding_proposed', 'constraint_binding_unresolved',
      'constraint_binding_rejected', 'constraint_bindings_refined',
    ]),
  },
  {
    id: 'plan',
    label: '制定计划',
    note: '能力与步骤',
    icon: ListTree,
    events: new Set([
      'plan_created', 'compilation_coverage', 'adaptive_decision', 'decision_rejected',
      'decision_action_surface', 'plan_replanned', 'replan_requested',
    ]),
  },
  {
    id: 'execute',
    label: '执行任务',
    note: '模型与工具',
    icon: Route,
    events: new Set([
      'step_started', 'step_completed', 'step_skipped', 'model_started', 'model_call',
      'model_retry', 'model_reused', 'tool_requested', 'tool_call', 'tool_result',
      'tool_retry', 'tool_protocol', 'native_submission', 'response_started', 'response_delta',
    ]),
  },
  {
    id: 'verify',
    label: '核验交付',
    note: '结果与证据',
    icon: FileCheck2,
    events: new Set([
      'verification', 'verification_failed', 'catalog_goal_verified',
      'catalog_delivery_verified', 'response_completed', 'response_failed', 'terminal',
    ]),
  },
] as const

function reachedPhase(events: RunEvent[]): number {
  return events.reduce((highest, event) => {
    const index = phases.findIndex((phase) => phase.events.has(event.type))
    return Math.max(highest, index)
  }, events.length > 0 ? 0 : -1)
}
export function WorkspaceMasthead() {
  return (
    <header className="workspace-masthead">
      <div>
        <span className="workspace-kicker"><Sparkles size={14} aria-hidden="true" />消费智能体工作台</span>
        <h1>说清目标，剩下的每一步都有据可查。</h1>
        <p>从商品选择到订单售后，LingCart 会保留计划、工具调用、核验结果和需要你确认的动作。</p>
      </div>
      <div className="workspace-boundaries" aria-label="工作台可信边界">
        <span><ShieldCheck size={15} aria-hidden="true" />高影响操作先确认</span>
        <span><Gauge size={15} aria-hidden="true" />运行状态实时同步</span>
        <small>当前使用本地演示数据</small>
      </div>
    </header>
  )
}
export function ExecutionModeSwitch({ adaptive, disabled, onChange }: {
  adaptive: boolean
  disabled: boolean
  onChange: (adaptive: boolean) => void
}) {
  return (
    <div className="execution-mode-control">
      <div>
        <strong>{adaptive ? '自主规划' : '标准执行'}</strong>
        <small>{adaptive ? '允许 Agent 在边界内重规划' : '优先采用确定性业务路径'}</small>
      </div>
      <button
        type="button"
        className="mode-switch-button"
        role="switch"
        aria-label="自适应规划"
        aria-checked={adaptive}
        disabled={disabled}
        onClick={() => onChange(!adaptive)}
      >
        <span aria-hidden="true" />
      </button>
    </div>
  )
}

export function TaskPulseRail({ events, status }: { events: RunEvent[]; status?: string }) {
  const reached = reachedPhase(events)
  const terminal = [...events].reverse().find((event) => event.type === 'terminal')
  const failed = terminal?.status === 'FAILED' || terminal?.status === 'CANCELLED'
  const succeeded = terminal?.status === 'SUCCEEDED'

  return (
    <section className="task-pulse" aria-label="任务执行阶段">
      <div className="task-pulse-label">
        <span>任务脉冲</span>
        <small>{events.length > 0 ? `${events.length} 条真实事件` : '发送需求后开始记录'}</small>
      </div>
      <ol>
        {phases.map((phase, index) => {
          const state = succeeded && index <= reached ? 'complete'
            : index < reached ? 'complete'
            : index === reached ? (failed && index === phases.length - 1 ? 'failed' : 'current')
              : 'waiting'
          const Icon = state === 'complete' ? Check : phase.icon
          return (
            <li className={state} key={phase.id} aria-current={state === 'current' ? 'step' : undefined}>
              <span className="pulse-node"><Icon size={14} aria-hidden="true" /></span>
              <div><strong>{phase.label}</strong><small>{phase.note}</small></div>
            </li>
          )
        })}
      </ol>
      {status && <span className="task-pulse-status">{STATUS_LABELS[status] ?? status}</span>}
    </section>
  )
}
export function WorkspaceEvidence({ result, activeRunId, activeStatus, pending = false }: {
  result?: ExecutionResult
  activeRunId?: string | null
  activeStatus?: string
  pending?: boolean
}) {
  const hasCurrentRun = pending || Boolean(activeRunId)
  const currentResult = hasCurrentRun ? undefined : result
  const runId = activeRunId ?? (hasCurrentRun ? undefined : result?.run_id)
  const status = pending ? 'PENDING' : activeRunId ? activeStatus : result?.status

  return (
    <section className="workspace-evidence" aria-label="当前任务证据">
      <div className="context-card-heading">
        <div><span>当前任务</span><h2>证据摘要</h2></div>
        {status && <em>{STATUS_LABELS[status] ?? status}</em>}
      </div>
      {!runId && !currentResult && !pending ? (
        <div className="evidence-empty">
          <CircleDashed size={20} aria-hidden="true" />
          <strong>尚无执行证据</strong>
          <p>发送一个需求后，这里会显示真实 Run、工具调用和 Token 记录。</p>
        </div>
      ) : (
        <dl className="evidence-ledger">
          <div><dt>运行编号</dt><dd>{runId?.slice(0, 12) ?? '等待分配'}</dd></div>
          <div><dt>工具调用</dt><dd>{currentResult ? currentResult.trace.length : '执行中'}</dd></div>
          <div><dt>结构化产物</dt><dd>{currentResult ? currentResult.artifacts?.length ?? 0 : '执行中'}</dd></div>
          <div><dt>Token</dt><dd>{currentResult?.usage.total_tokens ?? '暂无'}</dd></div>
        </dl>
      )}
    </section>
  )
}
