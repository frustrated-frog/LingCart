import { useEffect, useMemo, useState } from 'react'
import { Background, BackgroundVariant, Controls, Handle, MarkerType, Position, ReactFlow, useReactFlow,
  type Edge, type Node, type NodeProps } from '@xyflow/react'
import { GitBranch, ScanLine } from 'lucide-react'
import '@xyflow/react/dist/style.css'
import '../styles/execution-workbench.css'

export type CanvasNode = { id: string; title: string; subtitle: string; badge: string; tone?: string; active?: boolean }
export type CanvasEdge = { id: string; source: string; target: string; label?: string; kind?: 'guidance' | 'predicted' | 'sequence' }
type VisualNode = Node<CanvasNode, 'instrument'>
const NODE_WIDTH = 220
const NODE_HEIGHT = 104

function InstrumentNode({ data }: NodeProps<VisualNode>) {
  return <div className={`instrument-node tone-${data.tone ?? 'neutral'} ${data.active ? 'is-current' : ''}`}>
    <Handle type="target" position={Position.Left} />
    <div className="instrument-node-meta"><span>{data.badge}</span>{data.active ? <ScanLine size={14} /> : <GitBranch size={13} />}</div>
    <strong title={data.title}>{data.title}</strong><small title={data.subtitle}>{data.subtitle}</small>
    <Handle type="source" position={Position.Right} />
  </div>
}
const nodeTypes = { instrument: InstrumentNode }
function FitOnLayout({ revision }: { revision: string }) {
  const { fitView } = useReactFlow()
  useEffect(() => { const timer = setTimeout(() => { void fitView({ padding: .18, maxZoom: 1, duration: 0 }) }, 40); return () => clearTimeout(timer) }, [revision, fitView])
  return null
}

export default function GraphCanvas({ nodes, edges, selectedId, onSelectNode, onSelectEdge, linear = false }: {
  nodes: CanvasNode[]; edges: CanvasEdge[]; selectedId?: string
  onSelectNode: (id: string) => void; onSelectEdge?: (id: string) => void; linear?: boolean
}) {
  const structure = JSON.stringify({ nodes: nodes.map(n => n.id), edges: edges.map(e => [e.id, e.source, e.target]), linear })
  const [layout, setLayout] = useState<{ key: string; positions: Record<string, { x: number; y: number }>; error: boolean } | null>(null)
  useEffect(() => {
    let cancelled = false
    const data = JSON.parse(structure) as { nodes: string[]; edges: string[][]; linear: boolean }
    const fallback = Object.fromEntries(data.nodes.map((id, i) => [id, { x: data.linear ? i * 285 : (i % 3) * 285, y: data.linear ? 0 : Math.floor(i / 3) * 160 }]))
    if (data.linear || !data.nodes.length) { setLayout({ key: structure, positions: fallback, error: false }); return }
    import('elkjs/lib/elk.bundled.js').then(async ({ default: ELK }) => {
      const elk = new ELK()
      const result = await elk.layout({ id: 'root', layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT',
        'elk.spacing.nodeNode': '38', 'elk.layered.spacing.nodeNodeBetweenLayers': '90', 'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES' },
        children: data.nodes.map(id => ({ id, width: NODE_WIDTH, height: NODE_HEIGHT })),
        edges: data.edges.map(([id, source, target]) => ({ id, sources: [source], targets: [target] })) })
      if (!cancelled) setLayout({ key: structure, positions: Object.fromEntries((result.children ?? []).map(n => [n.id, { x: n.x ?? 0, y: n.y ?? 0 }])), error: false })
    }).catch(() => { if (!cancelled) setLayout({ key: structure, positions: fallback, error: true }) })
    return () => { cancelled = true }
  }, [structure])
  const flowNodes: VisualNode[] = useMemo(() => nodes.map(n => ({ id: n.id, type: 'instrument', data: n,
    position: layout?.positions[n.id] ?? { x: 0, y: 0 }, selected: selectedId === n.id,
    ariaLabel: `${n.badge}：${n.title}`, width: NODE_WIDTH, height: NODE_HEIGHT,
  })), [nodes, layout, selectedId, onSelectNode])
  const flowEdges: Edge[] = useMemo(() => edges.map(e => ({ ...e, type: 'smoothstep', selected: selectedId === e.id,
    interactionWidth: 24, focusable: true, ariaLabel: `关系：${e.label ?? e.id}`,
    className: `instrument-edge ${e.kind ?? 'relation'}`, markerEnd: { type: MarkerType.ArrowClosed, color: e.kind === 'predicted' ? '#b29cff' : e.kind === 'guidance' ? '#64dfdf' : '#6b819e' },
    style: { stroke: e.kind === 'predicted' ? '#b29cff' : e.kind === 'guidance' ? '#64dfdf' : '#6b819e', strokeWidth: selectedId === e.id ? 3 : 1.5, strokeDasharray: e.kind === 'predicted' ? '6 5' : undefined },
    labelStyle: { fill: '#d9e5f4', fontSize: 11 }, labelBgStyle: { fill: '#101c2d', fillOpacity: .95 }, labelBgPadding: [7, 4] as [number, number],
  })), [edges, selectedId, onSelectEdge])
  if (!nodes.length) return <div className="instrument-empty"><GitBranch size={30} /><b>没有可显示的图记录</b><p>选择已有事件，或查看下方原始证据。</p></div>
  return <div className="instrument-flow" aria-label="交互图画布" onKeyDownCapture={event => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    const target = event.target as Element
    const node = target.closest('.react-flow__node')
    const edge = target.closest('.react-flow__edge')
    const id = (node ?? edge)?.getAttribute('data-id')
    if (!id) return
    event.preventDefault()
    if (node) onSelectNode(id)
    else onSelectEdge?.(id)
  }}>
    {layout?.error && <div className="instrument-layout-warning" role="status">自动布局失败，已使用网格布局；所有记录仍可选择。</div>}
    {layout?.key !== structure ? <div className="instrument-empty" role="status">正在排列图结构…</div> :
      <ReactFlow nodes={flowNodes} edges={flowEdges} nodeTypes={nodeTypes} nodesDraggable={false} nodesConnectable={false}
        onNodeClick={(_, n) => onSelectNode(n.id)} onEdgeClick={(_, e) => onSelectEdge?.(e.id)}
        fitView minZoom={.15} maxZoom={1.5} colorMode="dark" deleteKeyCode={null} panOnScroll selectionOnDrag={false}
        ariaLabelConfig={{ 'controls.zoomIn.ariaLabel': '放大图', 'controls.zoomOut.ariaLabel': '缩小图', 'controls.fitView.ariaLabel': '适应画布',
          'node.a11yDescription.default': '按 Enter 选择节点，使用 Tab 在节点间移动。', 'edge.a11yDescription.default': '按 Enter 选择关系。' }}>
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#34465e" />
        <Controls showInteractive={false} /><FitOnLayout revision={structure} />
      </ReactFlow>}
  </div>
}
