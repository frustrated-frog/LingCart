import type { PGEdgeCredit, PGSelection } from '../types'

const labels: Record<string, string> = { SUPPORT: '支持', COUNTER: '反例', UNKNOWN: '未知', MIXED: '证据有分歧',
  ADOPTED_INFORMATION_GAIN: '采用后新增可信信息', ADOPTED_WITHOUT_INFORMATION: '采用后没有信息增益',
  INSUFFICIENT_CHAIN: '证据链不完整', NOT_EXPOSED: '未进入决策上下文', CONDITION_NOT_PROVEN: '条件未被证明',
  NOT_ADOPTED: '行动没有采用关系', NO_ATTRIBUTABLE_INFORMATION: '无法归因信息变化',
  NEW_INFORMATION: '新增信息', NO_NEW_INFORMATION: '没有新增信息',
  NO_INFORMATION_GAIN: '重复或无效观察', UNRESOLVED_GOAL: '目标未完成', ACTION_FAILURE: '行动失败',
  GUIDANCE_NOT_ADOPTED: '指导未采用', COVERAGE_GAP: '程序知识缺口', OBJECTIVE_PROGRESS: '推进目标',
  RECOVERY: '恢复执行', BETTER_ADOPTION: '改善采用', VALIDATION_REGRESSION: '验证退化',
  REPEATED_REJECTED_HYPOTHESIS: '重复已否定的修改方向', VALIDATION_NON_REGRESSION: '验证未退化',
  COST_REGRESSION: '成本退化', PG_NOT_EXPOSED: '指导未暴露' }
const label = (value: string) => labels[value] ?? value

export function EdgeCreditPanel({ credit, iteration, onEvidence }: {
  credit?: PGEdgeCredit; iteration?: number; onEvidence: (caseId: string, repetition: number) => void
}) {
  if (!credit) return <p className="pg-muted">当前图版本尚无训练贡献记录。选择有贡献报告的演化轮次，查看其训练图。</p>
  return <section className="pg-credit" aria-label="关系贡献证据">
    <header><strong>第 {iteration} 轮 · {label(credit.classification)}</strong><small>{credit.case_count} 个训练情境 · 关联证据，不代表因果</small></header>
    <div className="pg-credit-counts"><span>支持 <b>{credit.support}</b></span><span>反例 <b>{credit.counter}</b></span><span>未知 <b>{credit.unknown}</b></span></div>
    {!credit.evidence.length && <p>尚未观察到这条关系的干预链，保持未知。</p>}
    <div className="pg-credit-evidence">{credit.evidence.map(e => <article key={e.evidence_id}>
      <div><b className={`pg-credit-${e.classification.toLowerCase()}`}>{label(e.classification)}</b><code>{e.sample_ref} · 事件 {e.guidance_sequence}</code>
        <button type="button" onClick={() => onEvidence(e.case_id, e.repetition)}>查看训练轨迹 {e.sample_ref}</button></div>
      <p>{label(e.reason)} · 任务{e.task_passed ? '通过' : '失败'}</p>
      <small>条件 {e.condition} → {e.exposed ? '已曝光' : '未曝光'} → {e.adopted ? '已采用' : '未采用'} → {label(e.information_gain)}</small>
      <small>后续核验目标：{e.later_verified_objective_ids.join('、') || '尚无记录'}</small>
    </article>)}</div>
    {credit.contrasts.length ? <details><summary>同任务成败对照（{credit.contrasts.length} 组）</summary>{credit.contrasts.map(c => <div key={c.case_id} className="pg-credit-pair">
      <b>{c.case_id}</b><span>成功轨迹 {c.successful.samples.join('、')}<small>{Object.entries(c.successful.counts).map(([k, v]) => `${label(k)} ${v}`).join(' · ') || '没有该边证据'}</small></span>
      <span>失败轨迹 {c.failed.samples.join('、')}<small>{Object.entries(c.failed.counts).map(([k, v]) => `${label(k)} ${v}`).join(' · ') || '没有该边证据'}</small></span>
    </div>)}</details> : <p className="pg-muted">本轮尚无同一任务的成功／失败配对，不能比较轨迹差异。</p>}
  </section>
}

export function HypothesisPanel({ selection }: { selection: PGSelection }) {
  if (!selection.hypothesis && !selection.edge_credit) return null
  const hypothesis = selection.hypothesis
  return <section className="pg-hypothesis" aria-label="修改假设与失败记忆">
    <header><strong>修改假设</strong><small>训练证据提出 · 独立验证选择</small></header>
    {hypothesis ? <><p>{hypothesis.problem}</p><dl><dt>要解决的问题</dt><dd>{label(hypothesis.failure_mode)}</dd>
      <dt>预期行为变化</dt><dd>{label(hypothesis.expected_change)}</dd><dt>修改关系</dt><dd>{hypothesis.target_edges.join('、')}</dd>
      <dt>训练依据</dt><dd>{hypothesis.source_samples.join('、')} · {hypothesis.evidence_ids.length} 条具体干预证据</dd></dl></>
      : <p>未生成有效假设，已有训练贡献报告仍可查看。</p>}
    {selection.validation_changes?.length ? <details><summary>验证样本变化（仅供审查，不回流 Refiner）</summary>
      {selection.validation_changes.map(c => <p key={`${c.case_id}:${c.repetition}`}><code>{c.case_id}:{c.repetition}</code> · {c.before_passed ? '通过' : '失败'} → {c.after_passed ? '通过' : '失败'} · 模型尝试 {c.model_attempt_delta >= 0 ? '+' : ''}{c.model_attempt_delta}</p>)}
    </details> : <p className="pg-muted">没有可比较的候选验证结果。</p>}
    <h4>本轮收到的失败假设记忆</h4>
    {!selection.hypothesis_memory?.length && <p className="pg-muted">尚无历史假设；本轮不会凭空补充失败经验。</p>}
    {selection.hypothesis_memory?.map((m, i) => <article key={`${m.hypothesis_id}:${i}`}>
      <b>第 {m.iteration} 轮 · {m.blocks_retry ? '同父图下禁止原方向重试' : '保留观察，不封禁方向'}</b>
      <p>{m.hypothesis.problem} · {label(m.reason)}</p><small>比较 {m.validation_summary.compared_samples} 项 · 退化 {m.validation_summary.regressed_samples} 项 · 改善 {m.validation_summary.improved_samples} 项</small>
    </article>)}
    {selection.memory_entry && <p className="pg-memory-result">本轮记忆：{label(selection.memory_entry.reason)} · {selection.memory_entry.blocks_retry ? '下一轮须更换修改方向' : '保留记录，未否定该方向'}</p>}
  </section>
}
