import { useState } from 'react'
import type { ModelCaseDecision, RunEvent } from '../types'
import './case-lookahead.css'
import './case-model-decision.css'

const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' ? v as Record<string, unknown> : {}
function valid(v: unknown): v is ModelCaseDecision {
  const d = record(v)
  return ['source_quote', 'reason', 'expected_observation', 'replan_when'].every(k => typeof d[k] === 'string' && Boolean(d[k]))
    && ((d.action === 'SELECT' && typeof d.candidate_id === 'string' && d.question === null)
      || (d.action === 'CLARIFY_GOAL' && d.candidate_id === null && typeof d.question === 'string' && Boolean(d.question)))
}

export function CaseModelDecision({ events }: { events: RunEvent[] }) {
  const reports = events.filter(e => ['case_model_decision', 'case_model_decision_rejected'].includes(e.type))
  const [selected, setSelected] = useState('latest')
  const event = reports.find(e => String(e.sequence) === selected) ?? reports.at(-1)
  if (!event) return <div className="empty-inline">尚无模型行动决策；可在运行事件中查看当前进度。</div>
  const decision = event.decision
  const rejected = event.type === 'case_model_decision_rejected'
  const latest = events.filter(e => e.type === 'case_state_projected' && record(e.state).case_id === event.case_id).at(-1)
  const stale = latest && record(latest.state).state_id !== event.state_id
  const graph = record(events.find(e => e.type === 'case_obligation_graph_solved' && record(e.graph).graph_id === event.graph_id)?.graph)
  const candidates = Array.isArray(graph.candidates) ? graph.candidates.map(record) : []
  const candidate = candidates.find(c => c.candidate_id === record(decision).candidate_id)
  const completed = events.some(e => e.type === 'case_autonomous_action_completed' && e.case_id === event.case_id && e.directive_id === event.directive_id)
  const questionOpened = events.some(e => e.type === 'case_clarification_opened' && e.case_id === event.case_id
    && e.state_id === event.state_id && e.requirement_id === candidate?.requirement_id)
  const accepted = valid(decision) && event.status === 'ACCEPTED' && event.observed === false
  return <section className="case-lookahead case-model-decision" aria-label="模型行动决策">
    <header><div><span>自主决策 / 目标 · 观察 · 行动</span><h3>这一步，由当前任务决定</h3></div>
      <label>决策记录<select aria-label="选择模型决策记录" value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="latest">最近一次</option>{reports.map((e, i) => <option key={String(e.sequence ?? i)} value={String(e.sequence)}>事件 {e.sequence} · {e.type === 'case_model_decision_rejected' ? '未通过' : `案件版本 ${e.case_revision}`}</option>)}
      </select></label></header>
    <div className="lookahead-source"><b>{stale ? '历史决策 · 后续状态已变化' : '本轮最近决策'}</b><span>{rejected ? '校验拒绝' : '单步选择 · 实际反馈后重新决策'}</span></div>
    {rejected ? <p role="alert">模型提议未通过校验，未据此执行动作。错误码：{String(event.code)}。{String(event.feedback)}</p>
      : !accepted ? <p role="alert">决策记录不完整，请查看原始事件。</p>
      : <>
        <div className="model-decision-columns">
          <article><span className="model-decision-label">用户依据</span><blockquote>{decision.source_quote}</blockquote><p>{decision.reason}</p></article>
          <article className="model-decision-choice"><span className="model-decision-label">已接受的下一步</span>
            <h4>{decision.action === 'CLARIFY_GOAL' ? '先明确处理目标' : !candidate ? '等待候选说明' : candidate?.mode === 'AUTONOMOUS' ? '查询权威业务信息' : candidate?.mode === 'APPROVAL' ? '请求用户确认' : candidate?.mode === 'WAIT' ? '等待已延期的材料' : candidate?.mode === 'BLOCKED' ? '说明当前能力限制' : '询问当前所需材料'}</h4>
            <p>{decision.question ?? String(candidate?.question ?? candidate?.requirement_id ?? '候选说明尚未加载')}</p>
            {decision.action === 'SELECT' && !candidate && <p role="status">部分记录缺失：暂未找到对应候选动作。</p>}
          </article>
          <article><span className="model-decision-label">后续计划 · 尚未发生</span><h4>预计获得的信息</h4><p>{decision.expected_observation}</p><h4>何时重新判断</h4><p>{decision.replan_when}</p></article>
        </div>
        <footer>实际进度：{completed ? '已观察到对应工具完成；结果进入下一轮决策' : questionOpened ? '材料问题已持久化，等待真实回答' : decision.action === 'CLARIFY_GOAL' ? '本决策要求先澄清目标；是否已进入等待状态，请查看运行结果' : '尚未观察到对应工具完成；选择动作不等于完成任务'}</footer>
      </>}
  </section>
}
