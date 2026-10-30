import { Component, lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowLeft, ArrowRight, Clock3, GitBranch, ListTree, Pause, Play, Radio, SkipBack, SkipForward } from 'lucide-react'
import type { PGGraph, RunEvent } from '../types'
import type { CanvasEdge, CanvasNode } from './GraphCanvas'
import { eventTone, guidanceMarks, historicalGuidance } from './executionGraph'
import { eventLabel, JsonBlock, TOOL_LABELS } from './SystemUI'
import '../styles/execution-workbench.css'

const GraphCanvas = lazy(() => import('./GraphCanvas'))
const actionLabel = (id: string) => TOOL_LABELS[id] ?? ({ Start: '任务起点', End: '任务终点', set_goal: '建立目标', inspect: '检查证据',
  clarify: '澄清需求', revise_plan: '修订计划', recommend: '推荐候选', complete: '完成核验', finish: '结束执行' }[id] ?? id)
class CanvasBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <div className="instrument-empty" role="alert"><b>画布暂时无法加载</b><p>可继续通过对象列表和事件带查看证据，刷新页面后重试。</p></div> : this.props.children }
}
function Canvas(props: Parameters<typeof GraphCanvas>[0]) {
  return <CanvasBoundary><Suspense fallback={<div className="instrument-empty" role="status">正在加载交互画布…</div>}><GraphCanvas {...props} /></Suspense></CanvasBoundary>
}
function pgNodes(graph: PGGraph, anchor?: string | null): CanvasNode[] {
  return graph.nodes.map(n => ({ id: n.id, title: actionLabel(n.id), subtitle: n.description || n.id,
    badge: n.kind === 'ACTION' ? '行动' : n.kind === 'STATE' ? '状态' : '推理 / 其他', tone: n.id === anchor ? 'cyan' : 'neutral', active: n.id === anchor }))
}
function pgEdges(graph: PGGraph, marks = guidanceMarks()): CanvasEdge[] {
  return graph.edges.map(e => ({ id: e.id, source: e.source, target: e.target, label: e.relation,
    kind: marks.predicted.includes(e.id) ? 'predicted' : marks.guidance.includes(e.id) ? 'guidance' : undefined }))
}

export function PGGraphExplorer({ graph, selectedEdge, onSelectEdge }: {
  graph: PGGraph; selectedEdge?: string; onSelectEdge: (id: string) => void
}) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const node = graph.nodes.find(n => n.id === selectedNode)
  return <section className="execution-workbench pg-static-canvas" aria-label="程序知识图画布">
    <header className="instrument-header"><div><span className="instrument-eyebrow">PROCEDURAL ATLAS</span><h3>程序知识图谱</h3></div>
      <div className="instrument-identity"><b>{graph.graph_id} · v{graph.version}</b><span>{graph.nodes.length} 节点 · {graph.edges.length} 关系 · 冻结结构</span></div></header>
    <Canvas nodes={pgNodes(graph)} edges={pgEdges(graph)} selectedId={selectedNode ?? selectedEdge}
      onSelectNode={setSelectedNode} onSelectEdge={id => { setSelectedNode(null); onSelectEdge(id) }} />
    <footer className="instrument-static-footer"><span><GitBranch size={14} />连线表示程序关系；实际采用情况见样本回放</span><span>拖动画布 · 滚动平移 · 右下角缩放</span></footer>
    {node && <div className="instrument-node-details"><b>{actionLabel(node.id)}</b><code>{node.id}</code><p>{node.description}</p>
      <div>{graph.edges.filter(e => e.source === node.id || e.target === node.id).map(e => <button type="button" key={e.id} onClick={() => { setSelectedNode(null); onSelectEdge(e.id) }}>{actionLabel(e.source)} → {actionLabel(e.target)}</button>)}</div></div>}
  </section>
}

export function ExecutionWorkbench({ events, selected, onSelect, running = false, stale = false, sourceLabel = '持久化运行事件' }: {
  events: RunEvent[]; selected: number; onSelect: (index: number) => void; running?: boolean; stale?: boolean; sourceLabel?: string
}) {
  const cursor = Math.max(0, Math.min(selected, events.length - 1))
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [follow, setFollow] = useState(false)
  const [view, setView] = useState<'events' | 'pg'>('events')
  const [object, setObject] = useState<{ kind: 'node' | 'edge'; id: string } | null>(null)
  const [wide, setWide] = useState(false)
  const event = events[cursor]
  const selectedTick = useRef<HTMLButtonElement>(null)
  const timeline = useRef<HTMLDivElement>(null)
  const latest = useRef({ cursor, onSelect })
  latest.current = { cursor, onSelect }
  useEffect(() => { setObject(null) }, [cursor, view])
  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => latest.current.onSelect(latest.current.cursor + 1), speed)
    return () => window.clearInterval(timer)
  }, [playing, speed])
  useEffect(() => { if (cursor >= events.length - 1) setPlaying(false) }, [cursor, events.length])
  useEffect(() => { if (follow && running && !stale && events.length) onSelect(events.length - 1) }, [follow, running, stale, events.length, onSelect])
  useEffect(() => {
    const pause = () => { if (document.hidden) setPlaying(false) }
    document.addEventListener('visibilitychange', pause)
    return () => document.removeEventListener('visibilitychange', pause)
  }, [])
  useEffect(() => {
    const parent = timeline.current, tick = selectedTick.current
    if (parent && tick) parent.scrollLeft = tick.offsetLeft - parent.offsetLeft - parent.clientWidth / 2 + tick.clientWidth / 2
  }, [cursor])
  const choose = (index: number) => { setPlaying(false); setFollow(false); onSelect(Math.max(0, Math.min(index, events.length - 1))) }
  const history = useMemo(() => historicalGuidance(events, cursor), [events, cursor])
  const marks = guidanceMarks(history?.event)
  const graph = history?.graph
  const start = Math.max(0, Math.min(cursor - 1, events.length - 3))
  const visible = events.slice(start, start + 3)
  const nodes = view === 'pg' && graph ? pgNodes(graph, marks.anchor) : view === 'pg' ? [] : visible.map((e, i): CanvasNode => ({
    id: `event-${start + i}`, title: eventLabel(e)[1], subtitle: `${e.type}${e.status ? ` · ${e.status}` : ''}`,
    badge: `${e.sequence == null ? '编号缺失' : `事件 ${e.sequence}`} · ${start + i < cursor ? '先前' : start + i === cursor ? '当前' : '后续'}`,
    tone: eventTone(e), active: start + i === cursor,
  }))
  const edges: CanvasEdge[] = view === 'pg' && graph ? pgEdges(graph, marks) : view === 'pg' ? [] : visible.slice(1).map((_, i) => ({
    id: `order-${start + i}`, source: `event-${start + i}`, target: `event-${start + i + 1}`, label: '记录先后', kind: 'sequence',
  }))
  const selectedNode = object?.kind === 'node' ? graph?.nodes.find(n => n.id === object.id) : null
  const selectedEdge = object?.kind === 'edge' ? graph?.edges.find(e => e.id === object.id) : null
  const payload = selectedNode ?? selectedEdge ?? event
  // 对象详情仅来自游标已经到达的快照；当前记录原文始终可检查。
  const selectedTitle = selectedNode ? actionLabel(selectedNode.id) : selectedEdge ? `${actionLabel(selectedEdge.source)} → ${actionLabel(selectedEdge.target)}` : event ? eventLabel(event)[1] : '尚无事件'
  const tickStart = Math.max(0, Math.min(cursor - 50, events.length - 100))
  return <section className={`execution-workbench ${wide ? 'is-tall' : ''}`} aria-label="执行图回放工作台">
    <header className="instrument-header"><div><span className="instrument-eyebrow">EXECUTION OBSERVATORY</span><h3>执行现场<span className="instrument-live-label">{running ? '持续采集' : '历史回放'}</span></h3></div>
      <div className="instrument-identity"><b>{sourceLabel}</b><span>{events.length} 条记录 · 按事件播放</span></div></header>
    <div className="instrument-toolbar"><div className="instrument-view-tabs" role="group" aria-label="图视图">
      <button type="button" aria-pressed={view === 'events'} onClick={() => setView('events')}><ListTree size={15} />执行事件</button>
      <button type="button" aria-pressed={view === 'pg'} onClick={() => setView('pg')}><GitBranch size={15} />历史程序图</button>
    </div><button type="button" className="instrument-expand" aria-pressed={wide} onClick={() => setWide(!wide)}>{wide ? '收起画布' : '展开画布'}</button></div>
    {stale && <div className="instrument-notice" role="status">刷新失败，正在展示上次读取的历史记录；自动跟随已暂停。</div>}
    <div className="instrument-stage">
      <div className="instrument-map">
        <div className="instrument-map-caption"><span>{view === 'events' ? '事件邻域 / 连线仅表示记录顺序' : graph ? `${graph.graph_id} · v${history?.event.graph_version ?? '未知'} / 历史局部图` : '此时没有可用的程序图快照'}</span><span>{view === 'events' ? `${events.length ? cursor + 1 : 0} / ${events.length}` : history ? `来源事件 ${history.event.sequence ?? '编号缺失'}` : '尚无记录'}</span></div>
        {view === 'pg' && !graph ? <div className="instrument-empty"><GitBranch size={32} /><b>当前游标之前没有完整程序图</b><p>继续播放到 PG 指导事件，或切回执行事件。</p></div> : <Canvas nodes={nodes} edges={edges} linear={view === 'events'}
          selectedId={object?.id ?? (view === 'events' ? `event-${cursor}` : undefined)}
          onSelectNode={id => { if (view === 'events') choose(Number(id.replace('event-', ''))); else setObject({ kind: 'node', id }) }}
          onSelectEdge={id => { if (view === 'pg') setObject({ kind: 'edge', id }) }} />}
        <div className="instrument-legend"><span><i className="cyan" />{view === 'events' ? '当前事件描边' : '指导选中关系'}</span><span><i className="violet" />{view === 'events' ? '模型 / PG' : '预演路径（预测）'}</span><span><i className="amber" />{view === 'events' ? '错误 / 等待 / 阻断' : '不代表已经执行'}</span></div>
      </div>
      <aside className="instrument-probe" aria-label="游标证据">
        <div className="instrument-probe-top"><span>EVIDENCE PROBE</span><span>{object ? '图对象' : '当前事件'}</span></div>
        <b className="instrument-probe-sequence">{event?.sequence == null ? '—' : String(event.sequence).padStart(3, '0')}<small>事件序号</small></b>
        <h4>{selectedTitle}</h4>
        <p className="instrument-probe-kind">{object?.id ?? event?.type ?? '执行后将显示持久化记录'}</p>
        <dl><dt><Clock3 size={12} />记录时间</dt><dd>{event?.timestamp ?? '未记录时间戳'}</dd>
          {view === 'pg' && history && <><dt>图指纹</dt><dd className="instrument-fingerprint">{String(history.event.graph_sha256 ?? '未记录指纹')}</dd></>}
          {event?.status && <><dt>记录状态</dt><dd>{event.status}</dd></>}
          {selectedEdge && <><dt>适用条件</dt><dd>{selectedEdge.condition || '未记录说明'}</dd><dt>执行指导</dt><dd>{selectedEdge.guidance || '未记录说明'}</dd><dt>常见错误</dt><dd>{selectedEdge.pitfalls || '未记录'}</dd></>}
          {selectedNode && <><dt>节点说明</dt><dd>{selectedNode.description || '未记录'}</dd></>}
          {!object && typeof event?.reason === 'string' && <><dt>记录原因</dt><dd>{event.reason}</dd></>}
        </dl>
        {payload && <details key={`${cursor}-${object?.id ?? ''}`} className="instrument-raw"><summary>查看{object ? '对象' : '事件'}原文</summary><JsonBlock value={payload} /></details>}
        {view === 'pg' && graph && <details className="instrument-object-list"><summary>选择节点或关系</summary>
          {graph.nodes.map(n => <button type="button" key={`n-${n.id}`} onClick={() => setObject({ kind: 'node', id: n.id })}>{actionLabel(n.id)}</button>)}
          {graph.edges.map(e => <button type="button" key={`e-${e.id}`} onClick={() => setObject({ kind: 'edge', id: e.id })}>{e.id} · {actionLabel(e.source)} → {actionLabel(e.target)}</button>)}
        </details>}
        <p className="instrument-probe-footnote">{view === 'pg' ? '快照只说明当时可见的程序关系；采用与结果以实际事件为准。' : '按存储顺序浏览；播放间隔不代表真实执行耗时。'}</p>
      </aside>
    </div>
    <div className="instrument-transport"><div className="instrument-play-controls">
      <button type="button" aria-label="回到首个事件" disabled={!events.length || cursor === 0} onClick={() => choose(0)}><SkipBack size={15} /></button>
      <button type="button" aria-label="上一个事件" disabled={!events.length || cursor === 0} onClick={() => choose(cursor - 1)}><ArrowLeft size={15} /></button>
      <button type="button" className="instrument-play" aria-label={playing ? '暂停回放' : '播放事件'} disabled={events.length < 2} onClick={() => { setFollow(false); if (!playing && cursor === events.length - 1) onSelect(0); setPlaying(!playing) }}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? '暂停' : '播放'}</button>
      <button type="button" aria-label="下一个事件" disabled={!events.length || cursor === events.length - 1} onClick={() => choose(cursor + 1)}><ArrowRight size={15} /></button>
      <button type="button" aria-label="跳到最新事件" disabled={!events.length || cursor === events.length - 1} onClick={() => choose(events.length - 1)}><SkipForward size={15} /></button>
    </div><label className="instrument-speed">间隔<select aria-label="播放间隔" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={1600}>1.6 秒</option><option value={1000}>1 秒</option><option value={400}>0.4 秒</option></select></label>
      {running && <button type="button" className="instrument-follow" aria-pressed={follow} disabled={stale} onClick={() => { setPlaying(false); setFollow(!follow) }}><Radio size={14} />跟随最新</button>}
      <output aria-label="浏览位置">{events.length ? cursor + 1 : 0}<span> / {events.length}</span></output>
    </div>
    <div className="instrument-timeline" ref={timeline} aria-label="事件时间带">{events.slice(tickStart, tickStart + 100).map((e, i) => {
      const index = tickStart + i
      return <button type="button" ref={index === cursor ? selectedTick : undefined} key={index} className={`instrument-tick tone-${eventTone(e)} ${index === cursor ? 'is-selected' : ''}`}
        aria-label={`浏览第 ${index + 1} 条记录：${eventLabel(e)[1]}`} aria-pressed={index === cursor} title={`${e.sequence ?? '编号缺失'} · ${eventLabel(e)[1]}`} onClick={() => choose(index)}><i /><span>{e.sequence ?? '—'}</span></button>
    })}</div>
    <div className="instrument-scrubber"><span>起点</span><input type="range" aria-label="回放位置" min={0} max={Math.max(0, events.length - 1)} value={cursor} disabled={!events.length} onChange={e => choose(Number(e.target.value))} /><span>最新记录</span></div>
  </section>
}

export function SamplePlayback({ turns }: { turns: { message?: string; result?: { events?: RunEvent[] } }[] }) {
  const [turn, setTurn] = useState(0)
  const [cursor, setCursor] = useState(0)
  const selected = turns[turn]
  return <div className="sample-playback"><label>回放轮次<select aria-label="回放轮次" value={turn} onChange={e => { setTurn(Number(e.target.value)); setCursor(0) }}>{turns.map((t, i) => <option key={i} value={i}>第 {i + 1} 轮 · {t.message ?? '任务输入未记录'}</option>)}</select></label>
    <ExecutionWorkbench key={turn} events={selected?.result?.events ?? []} selected={cursor} onSelect={setCursor} sourceLabel={`实验原始轨迹 · 第 ${turn + 1} 轮`} />
  </div>
}
