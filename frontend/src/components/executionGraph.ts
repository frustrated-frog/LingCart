import type { PGEdge, PGGraph, RunEvent } from '../types'

export const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []

/** 仅恢复记录中的图，不把当前图或推测关系补进历史快照。 */
export function graphSnapshot(event: RunEvent): PGGraph | null {
  const snapshot = asRecord(event.subgraph)
  if (!Array.isArray(snapshot.nodes) || !Array.isArray(snapshot.edges)) return null
  const seen = new Set<string>()
  const nodes = snapshot.nodes.flatMap(value => {
    const n = asRecord(value)
    if (typeof n.id !== 'string' || seen.has(n.id)) return []
    seen.add(n.id)
    return [{ id: n.id, kind: String(n.kind ?? 'UNKNOWN'), description: String(n.description ?? '') }]
  })
  const edgeIds = new Set<string>()
  const edges: PGEdge[] = snapshot.edges.flatMap(value => {
    const e = asRecord(value)
    if (typeof e.id !== 'string' || edgeIds.has(e.id) || typeof e.source !== 'string' || typeof e.target !== 'string'
      || !seen.has(e.source) || !seen.has(e.target)) return []
    edgeIds.add(e.id)
    return [{ ...e, id: e.id, source: e.source, target: e.target, relation: String(e.relation ?? ''),
      condition: String(e.condition ?? ''), guidance: String(e.guidance ?? ''), pitfalls: String(e.pitfalls ?? '') } as PGEdge]
  })
  return { graph_id: String(event.graph_id ?? '未记录图身份'), version: Number(event.graph_version ?? 0), parent_sha256: null, nodes, edges }
}

export function historicalGuidance(events: RunEvent[], cursor: number) {
  for (let index = Math.min(cursor, events.length - 1); index >= 0; index--) {
    if (events[index].type === 'pg_guidance') return { event: events[index], index, graph: graphSnapshot(events[index]) }
  }
  return null
}

export function guidanceMarks(event?: RunEvent) {
  const preplay = asRecord(event?.preplay)
  const selected = asRecord(preplay.selected_path)
  const paths = Array.isArray(preplay.paths) ? preplay.paths : []
  return {
    guidance: strings(event?.edge_ids),
    predicted: [...new Set([...strings(selected.edge_ids), ...paths.flatMap(p => strings(asRecord(p).edge_ids))])],
    anchor: typeof event?.anchor === 'string' ? event.anchor : null,
  }
}

/** 色彩只标识显式事件类型；不从载荷猜测任务成功或 PG 因果收益。 */
export function eventTone(event: RunEvent): 'cyan' | 'violet' | 'amber' | 'green' | 'neutral' {
  if (['error', 'verification_failed', 'loop_blocked', 'approval_required', 'case_execution_paused'].includes(event.type)) return 'amber'
  if (['verification', 'catalog_goal_verified', 'case_obligation_verified', 'pg_prediction_outcome'].includes(event.type)) return 'green'
  if (event.type.startsWith('pg_') || event.type.startsWith('model_')) return 'violet'
  if (event.type.startsWith('tool_') || event.type === 'adaptive_decision') return 'cyan'
  return 'neutral'
}
