import { useState } from 'react'
import { ArrowRight, GitBranch, Pause, Play, Save, HelpCircle } from 'lucide-react'
import type { CaseGoal, RunEvent } from '../types'
import './case-goal-revision.css'

const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' ? v as Record<string, unknown> : {}
const modes = { CONTINUE: '继续处理', DRAFT: '仅保存草稿', PAUSED: '暂停推进', CLARIFY: '等待目标澄清' }
const icons = { CONTINUE: Play, DRAFT: Save, PAUSED: Pause, CLARIFY: HelpCircle }
function goal(v: unknown): v is CaseGoal {
  const g = record(v)
  return typeof g.case_id === 'string' && typeof g.run_id === 'string' && Object.hasOwn(modes, String(g.mode))
    && Number.isInteger(g.revision) && Number(g.revision) > 0 && typeof record(g.source).text === 'string'
}
const list = (value: unknown) => Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []

export function CaseGoalRevision({ events, goals = {}, running = false }: {
  events: RunEvent[]; goals?: Record<string, CaseGoal>; running?: boolean
}) {
  const [selection, setSelection] = useState('latest')
  const revisions = events.filter(e => e.type === 'case_goal_revised')
  const event = revisions.find(e => String(e.sequence) === selection) ?? revisions.at(-1)
  const current = event?.current
  const prior = event?.previous
  const valid = goal(current) && current.case_id === event?.case_id
  const Icon = valid ? icons[current.mode] : GitBranch
  const newer = valid && (revisions.some(e => e.case_id === current.case_id && goal(e.current) && e.current.revision > current.revision)
    || (goals[current.case_id]?.revision ?? 0) > current.revision)
  const applied = valid && events.find(e => e.type === 'case_goal_applied' && e.case_id === current.case_id
    && goal(e.goal) && e.goal.revision === current.revision && e.goal.run_id === current.run_id)
  const stateSaved = valid && goal(goals[current.case_id]) && goals[current.case_id].revision === current.revision
    && goals[current.case_id].run_id === current.run_id
  const observations = applied && Array.isArray(applied.material_observations) ? applied.material_observations : []
  return <section className="goal-revision" aria-label="案件目标调整">
    <header><div><span className="goal-eyebrow"><GitBranch size={14} aria-hidden="true" /> 用户目标 / 执行方向</span>
      <h3>这一次，往哪个方向继续？</h3></div>
      {revisions.length > 0 && <label>目标修订<select aria-label="选择目标修订记录" value={selection} onChange={e => setSelection(e.target.value)}>
        <option value="latest">最近一次</option>{revisions.map((e, i) => <option key={String(e.sequence ?? i)} value={String(e.sequence)}>事件 {e.sequence}</option>)}
      </select></label>}
    </header>
    {!event ? <div className="empty-inline" role="status">{running ? '正在理解本轮目标，修订记录尚未生成。' : '本轮没有目标修订记录。'}
      {Object.values(goals).filter(goal).map(g => <p key={g.case_id}>案件 {g.case_id.slice(0, 8)} · 沿用{modes[g.mode]} · 目标版本 {g.revision}</p>)}
    </div> : !valid ? <p className="goal-notice" role="alert">目标记录不完整或案件不匹配，请查看原始事件。</p> : <>
      <div className="goal-route"><div><span>此前目标</span><b>{goal(prior) && prior.case_id === current.case_id ? modes[prior.mode] : prior === null ? '常规处理' : '历史记录不完整'}</b></div>
        <ArrowRight size={23} aria-label="调整为" /><div className={`goal-current ${current.mode.toLowerCase()}`}><Icon size={20} aria-hidden="true" /><div><span>目标版本 {current.revision}</span><b>{modes[current.mode]}</b></div></div>
        <span className="goal-case">案件 {current.case_id.slice(0, 8)}</span>
      </div>
      {newer && <p className="goal-notice" role="status">历史修订 · 后续目标已变化</p>}
      <blockquote><span>用户本轮原话</span>{current.source.text}</blockquote>
      <div className="goal-impact">{([['保留', event.preserved], ['暂停', event.suspended], ['下一步计划', event.next_steps]] as const).map(([title, rows]) =>
        <article key={title}><h4>{title}</h4>{list(rows).length ? <ul>{list(rows).map(text => <li key={text}>{text}</li>)}</ul>
          : <p>{title === '暂停' && current.mode === 'CONTINUE' ? '恢复案件评估；提交仍需明确确认。' : '尚无对应记录。'}</p>}</article>)}</div>
      <footer role="status"><b>{stateSaved ? '目标已保存到会话' : applied ? '执行已采用此目标' : '已记录目标修订'}</b>
        <span>{applied ? `已观察到 ${observations.length} 项材料操作完成，已停止主动推进。` : '材料是否保存及后续动作是否完成，以工具账本为准。'}</span>
      </footer>
    </>}
  </section>
}
