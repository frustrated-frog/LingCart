import { useState } from 'react'
import { ArrowRight, Check, GitCompareArrows, RefreshCw } from 'lucide-react'
import type { CaseEvidenceTransition, EvidenceSemanticStatus, RunEvent } from '../types'
import './case-state-changes.css'

const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' ? value as Record<string, unknown> : {}
const labels: Record<EvidenceSemanticStatus, string> = { SUFFICIENT: '材料充分', INCOMPLETE: '待补充', IRRELEVANT: '不适用', CONFLICT: '存在冲突', UNKNOWN: '待核实' }
const status = (value: unknown) => typeof value === 'string' && Object.hasOwn(labels, value)
function validReport(value: unknown): value is CaseEvidenceTransition {
  const r = record(value)
  return r.version === 'case-evidence-transition-v1' && r.commit_status === 'COMMITTED'
    && ['report_id', 'case_id', 'assessment_id', 'assessment_report_id', 'input_hash'].every(k => typeof r[k] === 'string' && r[k])
    && Number.isInteger(r.from_revision) && Number(r.from_revision) >= 0
    && Number.isInteger(r.to_revision) && Number(r.to_revision) > Number(r.from_revision)
    && (r.error_code === null || typeof r.error_code === 'string')
    && Array.isArray(r.changes) && r.changes.every(value => {
      const c = record(value)
      return ['requirement_id', 'description', 'evidence_type', 'reason', 'reason_code'].every(k => typeof c[k] === 'string' && c[k])
        && (c.from_status === null || status(c.from_status)) && status(c.to_status)
        && ['PRESERVE', 'REASSESS'].includes(String(c.action))
        && (c.action !== 'PRESERVE' || (c.from_status === c.to_status && typeof c.prior_assessment_id === 'string'))
    })
    && new Set(r.changes.map(c => record(c).requirement_id)).size === r.changes.length
}

export function CaseStateChanges({ events }: { events: RunEvent[] }) {
  const [selected, setSelected] = useState('latest')
  const committed = events.filter(e => e.type === 'tool_result' && record(e.result).evidence_transition)
  const event = committed.find(e => String(e.sequence) === selected) ?? committed.at(-1)
  const report = record(event?.result).evidence_transition
  const repairs = events.filter(e => ['case_material_target_reconciled', 'case_turn_verification_conflict'].includes(e.type))
  const proposals = events.filter(e => e.type === 'case_evidence_transition_assessed')
  const valid = validReport(report)
  const laterRevision = valid && events.some(e => e.type === 'case_state_projected'
    && record(e.state).case_id === report.case_id && Number(record(e.state).case_revision) > report.to_revision)
  return <section className="case-changes" aria-label="案件材料状态变化">
    <header><div><span className="changes-eyebrow"><GitCompareArrows size={14} aria-hidden="true" /> 跨轮材料 / 状态演进</span>
      <h3>这一轮，哪些判断变了？</h3></div>
      {committed.length > 0 && <label>落账记录<select aria-label="选择材料变化记录" value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="latest">最近一次</option>{committed.map((e, i) => <option key={String(e.sequence ?? i)} value={String(e.sequence)}>事件 {e.sequence}</option>)}
      </select></label>}
    </header>
    {!event ? <div className="empty-inline">{proposals.length ? '已有核验提议，尚未观察到已提交的材料变化。' : '尚无已提交的增量核验记录。'}</div>
      : !valid ? <p role="alert">材料变化记录不完整，无法还原状态；请检查原始工具事件。</p>
      : <>
        <div className="changes-source"><b><Check size={13} aria-hidden="true" /> 已落账</b><span>案件 {report.case_id.slice(0, 8)}</span>
          <span>版本 {report.from_revision} <ArrowRight size={12} aria-label="到" /> {report.to_revision}</span>
          {laterRevision && <span role="status">历史版本 · 后续案件状态已变化</span>}</div>
        {report.error_code && <p role="alert">本轮核验未得到可靠结论，材料按待核实记录。错误：{report.error_code}</p>}
        <div className="changes-columns" aria-hidden="true"><span>材料要求</span><span>此前判断</span><span>本次处理</span><span>本次判断</span></div>
        <div className="changes-list">{report.changes.map(change => <article key={change.requirement_id} className={`change-row ${change.action.toLowerCase()}`}>
          <div className="change-requirement"><b>{change.description}</b><small>{change.evidence_type}</small></div>
          <div className="change-status"><span className="sr-only">此前判断：</span>{change.from_status === null ? '暂无有效结论' : labels[change.from_status]}</div>
          <div className="change-operation">{change.action === 'PRESERVE' ? <Check size={16} aria-hidden="true" /> : <RefreshCw size={16} aria-hidden="true" />}
            <b>{change.action === 'PRESERVE' ? '保留结论' : change.from_status === null ? '首次核验' : '重新核验'}</b><ArrowRight size={15} aria-hidden="true" /></div>
          <div className={`change-status current ${change.to_status.toLowerCase()}`}><span className="sr-only">本次判断：</span>{labels[change.to_status]}</div>
          <p className="change-reason">{change.reason}</p>
        </article>)}</div>
        {!report.changes.length && <div className="empty-inline">当前政策没有需要语义核验的材料要求。</div>}
        <footer><span>“材料充分”表示用户陈述满足要求，仍需后续政策判断与操作确认。</span>
          <details><summary>查看落账依据</summary><dl><dt>评估</dt><dd>{report.assessment_id}</dd><dt>变化报告</dt><dd>{report.report_id}</dd><dt>输入指纹</dt><dd>{report.input_hash}</dd></dl></details>
        </footer>
      </>}
    {repairs.length > 0 && <details className="changes-repairs"><summary>本轮理解校正 · {repairs.length} 条记录</summary>
      <p>以下为理解过程记录；材料是否更新以已落账结果为准。</p>
      {repairs.map((e, i) => <article key={String(e.sequence ?? i)}><b>{e.type === 'case_material_target_reconciled' ? '校正隐式材料归属' : e.terminal ? '需要进一步澄清' : '定向修复理解分歧'}</b>
        {e.type === 'case_material_target_reconciled' ? <p>{String(e.from_requirement_id)} → {String(e.to_requirement_id)}</p> : <p>{String(record(e.feedback).issue ?? e.question ?? '归属或执行限制不一致')}</p>}
        {typeof e.source_quote === 'string' && <blockquote>{e.source_quote}</blockquote>}
      </article>)}
    </details>}
  </section>
}
