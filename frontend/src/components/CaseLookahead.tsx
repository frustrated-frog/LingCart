import { useState } from 'react'
import type { InformationPath, LookaheadReport, RunEvent } from '../types'
import './case-lookahead.css'

const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' ? value as Record<string, unknown> : {}
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(v => typeof v === 'string')
function validPath(value: unknown): value is InformationPath {
  const p = record(value)
  return ['candidate_ids', 'requirement_ids', 'remaining_if_sufficient', 'remaining_if_no_information'].every(k => strings(p[k]))
    && ['user_turns', 'tool_calls', 'response_burden', 'first_action_burden'].every(k => typeof p[k] === 'number' && Number.isInteger(p[k]) && Number(p[k]) >= 0)
    && (p.candidate_ids as string[]).length > 0
    && (p.candidate_ids as string[]).length <= 2
    && (p.candidate_ids as string[]).length === (p.requirement_ids as string[]).length
}

function validReport(value: unknown): value is LookaheadReport {
  const r = record(value)
  return r.version === 'case-lookahead-v1' && r.horizon === 2
    && r.assumption === 'SUFFICIENT_INFORMATION_NOT_OBSERVED'
    && ['report_id', 'state_id', 'graph_id', 'portfolio_id', 'reason_code'].every(k => typeof r[k] === 'string')
    && ['SELECTED', 'FALLBACK', 'INVALID', 'NOT_REQUIRED'].includes(String(r.status))
    && Number.isInteger(r.expanded_paths) && Number(r.expanded_paths) >= 0 && Number(r.expanded_paths) <= 144
    && (r.selected_candidate_id === null || typeof r.selected_candidate_id === 'string')
    && Array.isArray(r.paths) && r.paths.every(validPath)
    && (r.status !== 'SELECTED' || r.paths.some(p => p.candidate_ids[0] === r.selected_candidate_id))
}

export function CaseLookahead({ events }: { events: RunEvent[] }) {
  const reports = events.filter(e => e.type === 'case_lookahead_solved')
  const [selected, setSelected] = useState('latest')
  const event = reports.find(e => String(e.sequence) === selected) ?? reports.at(-1)
  if (!event) return <div className="empty-inline">尚无前瞻决策记录；未生成的路径不会展示为已规划。</div>
  const report = event.report
  if (!validReport(report)) {
    return <p role="alert">前瞻记录不完整，无法展示候选路径。可在原始事件中检查。</p>
  }
  const paths = report.paths
  const graph = record(events.find(e => e.type === 'case_obligation_graph_solved' && record(e.graph).graph_id === report.graph_id)?.graph)
  const candidates = Array.isArray(graph.candidates) ? graph.candidates.map(record) : []
  const candidate = (id: string) => candidates.find(c => c.candidate_id === id)
  const label = (id: string) => {
    const c = candidate(id)
    return c ? String(c.question ?? (c.mode === 'AUTONOMOUS' ? '查询权威订单信息' : c.requirement_id)) : `候选 ${id.slice(0, 8)}（说明缺失）`
  }
  const latestState = record(events.filter(e => e.type === 'case_state_projected' && record(e.state).case_id === event.case_id).at(-1)?.state)
  const stale = latestState.state_id && latestState.state_id !== report.state_id
  const completed = events.some(e => e.type === 'case_autonomous_action_completed' && e.case_id === event.case_id
    && e.graph_id === report.graph_id && e.from_revision === event.case_revision && e.candidate_id === report.selected_candidate_id)
  const questionOpened = events.some(e => e.type === 'case_clarification_opened' && e.case_id === event.case_id
    && e.state_id === report.state_id && e.portfolio_id === report.portfolio_id)
  return <section className="case-lookahead" aria-label="信息行动前瞻">
    <header><div><span>行动决策 / 前瞻与实况</span><h3>先做哪一步，为什么？</h3></div>
      <label>决策记录<select aria-label="选择前瞻记录" value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="latest">最近一次</option>{reports.map((e, i) => <option key={String(e.sequence ?? i)} value={String(e.sequence)}>事件 {e.sequence} · 案件版本 {String(e.case_revision)}</option>)}
      </select></label></header>
    <div className="lookahead-source"><b>{stale ? '历史状态 · 后续已重新投影' : '本轮最近决策'}</b>
      <span>案件版本 {String(event.case_revision)} · 最多 {String(report.horizon)} 步 · 已搜索 {String(report.expanded_paths)} 条路径</span></div>
    <p>充分信息分支是假设覆盖上界，不是成功概率。只执行当前一步，后续依据真实材料重新规划。</p>
    <p>选择顺序：优先权威查询，再比较剩余义务、交互与工具次数、总回答负担和首步负担。</p>
    {!candidates.length && paths.length > 0 && <p role="status">部分记录缺失：未找到对应义务图，暂时显示候选标识。</p>}
    {!paths.length ? <div className="empty-inline">{report.status === 'INVALID' ? '输入核验未通过，未采用前瞻动作' : report.status === 'FALLBACK' ? '前瞻不可用，已回退原有安全策略' : '当前没有需要前瞻的信息动作'} · {String(report.reason_code)}</div> :
      <div className="lookahead-paths">{paths.map(path => {
        const chosen = path.candidate_ids[0] === report.selected_candidate_id
        return <article className={chosen ? 'chosen' : ''} key={path.candidate_ids.join(':')}>
          <div className="lookahead-path-title"><b>{chosen ? '已选首步' : '备选首步'}</b><span>首步需答 {path.first_action_burden} 项 · 路径共 {path.response_burden} 项</span></div>
          <ol>{path.candidate_ids.map((id, index) => <li key={id}><span>{index === 0 ? '现在' : '假设后续'}</span>{label(id)}</li>)}</ol>
          <div className="lookahead-branches"><div><span>若各步信息充分</span><b>剩余 {path.remaining_if_sufficient.length} 项义务</b></div><div><span>若未获得有效信息</span><b>仍有 {path.remaining_if_no_information.length} 项义务</b></div></div>
          <small>计划内交互 {path.user_turns} 次 · 工具调用 {path.tool_calls} 次 · 均为假设计数</small>
        </article>
      })}</div>}
    <footer>真实执行：{completed ? '首步工具已完成，后续以新状态为准' : questionOpened
      ? '首步澄清问题已记录；是否满足义务仍需回答核验'
      : '本记录未观察到首步工具完成；提问或选择不等于义务已满足'}</footer>
  </section>
}
