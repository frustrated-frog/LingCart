import { useState } from 'react'
import { ArrowRight, Database, GitBranch, ScanLine } from 'lucide-react'
import type { CapabilityPlanningProjection } from '../types'
import { TOOL_LABELS } from './SystemUI'
import '../styles/capability-plan.css'

const STATUSES = {
  READY: '已有可执行步骤', NEEDS_MODEL: '等待模型决策', OBSERVATIONS_READY: '所需观察已齐备',
  PARTIAL: '部分目标缺少路径', UNREACHABLE: '当前能力无法形成路径', SEARCH_LIMIT: '达到规划搜索上限',
}
const NODE_STATUSES = { READY: '输入已齐备', PLANNED: '等待前置结果', NEEDS_ARGUMENTS: '待模型绑定参数' }
const FACTS: Record<string, string> = {
  'order.status': '订单状态', 'order.details': '订单详情', 'order.shipment': '物流信息',
  'order.refund_check': '退款资格判断', 'order.payment': '支付状态',
  'catalog.candidates': '商品候选', 'catalog.comparison': '商品对比', 'policy.answer': '政策回答',
}
function factLabel(value: string) {
  const [scope, name = ''] = value.split('/')
  const target = scope.startsWith('order:') ? `订单 ${scope.slice(6)}` : scope === 'catalog' ? '商品目录' : '政策问题'
  return `${target} · ${FACTS[name] ?? (name.startsWith('comparison:') ? '指定商品对比' : name || value)}`
}

export function CapabilityPlanPanel({ projection }: { projection?: CapabilityPlanningProjection }) {
  const [version, setVersion] = useState('latest')
  const [selected, setSelected] = useState<string | null>(null)
  const history = projection?.history ?? []
  const current = (version === 'latest' ? history.at(-1) : history.find(p => p.plan_id === version)) ?? history.at(-1)
  if (!current) return <div className="empty-inline" role="status">
    {projection?.status === 'PARTIAL' ? '能力规划记录无法解析，请查看原始事件。' : '该运行尚无能力规划记录。使用 CGAP 研究 Profile 的任务会在这里显示真实计划。'}
  </div>
  const positions = new Map<string, { x: number; y: number; depth: number }>()
  const rows = new Map<number, number>()
  for (const node of current.nodes) {
    const depth = Math.max(0, ...node.depends_on.map(id => (positions.get(id)?.depth ?? -1) + 1))
    const row = rows.get(depth) ?? 0
    positions.set(node.action_id, { x: 28 + depth * 272, y: 28 + row * 130, depth })
    rows.set(depth, row + 1)
  }
  const width = Math.max(600, ...[...positions.values()].map(p => p.x + 250))
  const height = Math.max(220, ...[...positions.values()].map(p => p.y + 130))
  const active = current.nodes.find(n => n.action_id === selected) ?? current.nodes[0]
  const isHistory = current.plan_id !== history.at(-1)?.plan_id
  return <section className="capability-panel" aria-label="能力驱动动态规划">
    <header className="capability-heading">
      <div><span className="capability-kicker"><GitBranch size={14} /> 能力驱动规划</span>
        <h3>{STATUSES[current.status]}</h3><p>{current.reason}</p></div>
      <label>计划版本<select aria-label="计划版本" value={version} onChange={e => { setVersion(e.target.value); setSelected(null) }}>
        <option value="latest">最新 · 第 {history.at(-1)?.revision} 版</option>
        {history.map(p => <option key={p.plan_id} value={p.plan_id}>第 {p.revision} 版 · {p.reason}</option>)}
      </select></label>
    </header>
    {(isHistory || projection?.stale) && <p className="capability-notice" role="status">
      {projection?.stale ? '目标已经变化，当前显示的规划尚未更新。' : '正在查看历史计划；节点状态只代表生成该版本时的状态。'}
    </p>}
    {projection?.status === 'PARTIAL' && <p className="capability-notice" role="alert">有 {projection.invalid_records} 条规划记录不完整，当前仅展示可解析部分。</p>}
    <div className="capability-workspace">
      <aside className="capability-facts"><h4><Database size={15} /> 已获得的真实观察</h4>
        {Object.entries(current.facts).length ? Object.entries(current.facts).map(([name, refs]) =>
          <div className="capability-fact" key={name}><strong>{factLabel(name)}</strong><span>{refs.join(' · ')}</span></div>
        ) : <p>尚无可用于满足依赖的工具观察。</p>}
        <p className="capability-footnote">只有通过核验的实际返回值才能满足依赖。</p>
      </aside>
      <div className="capability-canvas" tabIndex={0} aria-label="工具依赖图，可横向滚动">
        {current.nodes.length ? <div className="capability-map" style={{ width, height }}>
          <svg width={width} height={height} aria-hidden="true">
            {current.nodes.flatMap(node => node.depends_on.map(parent => {
              const from = positions.get(parent), to = positions.get(node.action_id)
              if (!from || !to) return null
              const x = from.x + 224, y = from.y + 49
              return <path key={`${parent}-${node.action_id}`} d={`M ${x} ${y} C ${x + 25} ${y}, ${to.x - 25} ${to.y + 49}, ${to.x} ${to.y + 49}`} />
            }))}
          </svg>
          {current.nodes.map(node => {
            const position = positions.get(node.action_id)!
            return <button key={node.action_id} className={`capability-node ${node.status.toLowerCase()}`}
              aria-pressed={active?.action_id === node.action_id} onClick={() => setSelected(node.action_id)}
              style={{ left: position.x, top: position.y }}>
              <span>{NODE_STATUSES[node.status]}<ArrowRight size={13} /></span>
              <strong>{TOOL_LABELS[node.tool] ?? node.tool}</strong>
              <small>{node.scope.startsWith('order:') ? `订单 ${node.scope.slice(6)}` : node.scope === 'catalog' ? '商品目录' : '政策问题'}</small>
            </button>
          })}
        </div> : <div className="capability-no-plan"><ScanLine size={30} /><strong>当前没有剩余工具路径</strong>
          <p>{current.missing.length ? '查看缺失能力，或由模型选择澄清与其它可行行动。' : '继续由模型读取证据、完成推荐与最终核验。'}</p></div>}
      </div>
    </div>
    {active && <div className="capability-inspector" aria-live="polite">
      <div><small>为什么需要这一步</small><strong>{active.objective_ids.map(id => current.objective_labels[id] ?? id).join('；')}</strong></div>
      <div><small>前置条件</small><span>{active.requires.map(factLabel).join('；') || '使用已确认目标中的输入'}</span>
        {active.evidence_refs.length > 0 && <code>来源：{active.evidence_refs.join(' · ')}</code>}</div>
      <div><small>预计提供 · 尚未执行</small><span>{active.provides.map(factLabel).join('；')}</span></div>
    </div>}
    {current.missing.length > 0 && <p className="capability-notice">缺少可行路径：{current.missing.map(factLabel).join('；')}</p>}
    {current.delegated_objectives.length > 0 && <p className="capability-delegated">仍需模型处理：{current.delegated_objectives.map(id => current.objective_labels[id] ?? id).join('；')}</p>}
    <footer className="capability-footer"><span>目标版本 {current.goal_revision} · 搜索展开 {current.expanded} 个状态</span>
      <span>依赖图表示计划，任务完成以最终核验为准。</span></footer>
  </section>
}
