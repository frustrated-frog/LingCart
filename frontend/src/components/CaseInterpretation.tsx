import { useState } from 'react'
import { Quote, ScanText } from 'lucide-react'
import type { ControlIntent, RunEvent } from '../types'
import './case-goal-revision.css'
import './case-interpretation.css'

const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' ? v as Record<string, unknown> : {}
const labels = { REQUIRE_CONFIRMATION: '继续核验，等待确认', PROHIBIT: '明确禁止执行', RELEASE_PROHIBITION: '解除指定禁令', REVOKE_APPROVAL: '撤回旧确认' }
const effects = { REQUIRE_CONFIRMATION: '可以重新核验、准备确认；当前没有提交授权。', PROHIBIT: '该范围的禁令持续生效，直到用户明确解除。',
  RELEASE_PROHIBITION: '解除这一项限制，其他禁令和提交前确认继续有效。', REVOKE_APPROVAL: '旧确认失效；后续继续处理需重新核验、取得新确认。' }
const restrictions: Record<string, string> = { READ_ONLY:'业务写入', NO_SUBMISSION:'提交与支付退款', NO_REFUND:'退款', NO_PAYMENT:'支付' }
function valid(v: unknown): v is ControlIntent {
  const c = record(v)
  return Object.hasOwn(labels, String(c.kind))
    && (c.scope === 'SESSION' ? c.case_id == null : c.scope === 'CASE' && typeof c.case_id === 'string' && c.case_id.length > 0)
    && (c.kind === 'PROHIBIT' || c.kind === 'RELEASE_PROHIBITION' ? Object.hasOwn(restrictions, String(c.restriction)) : c.restriction == null)
}

export function CaseInterpretation({ events, running = false }: { events: RunEvent[]; running?: boolean }) {
  const [selected, setSelected] = useState('latest')
  const controls = events.filter(e => e.type === 'case_control_interpreted')
  const event = controls.find(e => String(e.sequence) === selected) ?? controls.at(-1)
  const intent = event?.control
  const sources = new Map<string, { text: string; paths: string[]; start: number; end: number }>()
  for (const e of events.filter(e => e.type === 'request_sources_resolved')) {
    for (const item of Array.isArray(e.bindings) ? e.bindings : []) {
      const binding = record(item), source = record(binding.source)
      if (typeof source.source_ref !== 'string' || typeof source.text !== 'string' || source.origin !== 'CURRENT_TURN'
        || !Number.isInteger(source.start) || !Number.isInteger(source.end)) continue
      const row = sources.get(source.source_ref) ?? { text: source.text, paths: [], start: Number(source.start), end: Number(source.end) }
      const path = Array.isArray(binding.path) ? binding.path.join('.') : ''
      row.paths.push(path.includes('material_changes') ? '材料变更' : path.includes('control_intents') ? '控制意图' : e.purpose === 'request_semantics' ? '独立需求提取' : '请求理解')
      sources.set(source.source_ref, row)
    }
  }
  const sourceText = record(event?.source).text
  return <section className="goal-revision interpretation" aria-label="原话与执行含义">
    <header><div><span className="goal-eyebrow"><ScanText size={14} aria-hidden="true" /> 原文 / 意图 / 执行</span><h3>这句话，如何影响下一步？</h3></div>
      {controls.length > 0 && <label>解释记录<select aria-label="选择控制解释" value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="latest">最近一次</option>{controls.map((e, i) => <option key={e.sequence ?? i} value={String(e.sequence)}>事件 {e.sequence}</option>)}
      </select></label>}
    </header>
    {event ? valid(intent) && typeof sourceText === 'string' ? <>
      <div className="interpretation-intent"><span className="interpretation-state">{event.status === 'VERIFIED' ? '两次解释一致' : '尚未核验'}</span>
        <h4>{labels[intent.kind]}</h4><p>{effects[intent.kind]}</p><small>作用范围：{intent.scope === 'CASE' ? `案件 ${intent.case_id}` : '当前会话'}{intent.restriction ? ` · ${restrictions[intent.restriction] ?? intent.restriction}` : ''}</small>
        {event !== controls.at(-1) && <p role="status">历史解释 · 后续已有新的记录</p>}
      </div><blockquote><span>对应用户原话</span>{sourceText}</blockquote>
    </> : <p className="goal-notice" role="alert">控制解释记录不完整，无法判断作用范围。</p>
      : <p className="goal-notice" role="status">{running ? '正在理解本轮请求，控制解释尚未生成。' : '本轮没有新的控制意图解释。'}</p>}
    <div className="interpretation-sources"><h4><Quote size={14} aria-hidden="true" /> 原文定位</h4>
      {sources.size ? [...sources].map(([id, row], index) => <details key={id}><summary>片段 {index + 1} · {[...new Set(row.paths)].join(' / ')}</summary>
        <blockquote>{row.text}</blockquote><small>本轮用户消息 · 字符范围 {row.start}–{row.end}</small></details>)
        : <p role="status">尚无可定位的原文片段。</p>}
    </div>
    <footer>原文定位与解释一致不代表材料充分或已执行；最终结果以政策核验、确认与工具账本为准。</footer>
  </section>
}
