import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RunEvent } from '../types'
import { ExecutionWorkbench, SamplePlayback } from './ExecutionWorkbench'
import { graphSnapshot, guidanceMarks, historicalGuidance } from './executionGraph'

// jsdom 没有几何布局；真实画布由浏览器验收，组件测试检查用户可见回放语义。
vi.mock('./GraphCanvas', () => ({ default: ({ nodes, onSelectNode }: { nodes: { id: string; title: string }[]; onSelectNode: (id: string) => void }) =>
  <div aria-label="测试画布">{nodes.map(n => <button key={n.id} onClick={() => onSelectNode(n.id)}>{n.title}</button>)}</div> }))

const snapshot: RunEvent = { type: 'pg_guidance', sequence: 2, graph_id: 'frozen', graph_version: 3, graph_sha256: 'historical-sha', anchor: 'Start', edge_ids: ['a'],
  subgraph: { nodes: [{ id: 'Start', kind: 'STATE', description: '起点' }, { id: 'search', kind: 'ACTION', description: '搜索' }],
    edges: [{ id: 'a', source: 'Start', target: 'search', relation: 'LEADS_TO', condition: '条件证据', guidance: '先读取', pitfalls: '' }] },
  preplay: { selected_path: { edge_ids: ['a'] } } }
const events: RunEvent[] = [{ type: 'request_received', sequence: 1 }, snapshot,
  { type: 'tool_result', sequence: 3, tool: 'search', result: { evidence: 'future-only-evidence' } }]
function Player({ initial = 0, source = events, running = false, stale = false }: { initial?: number; source?: RunEvent[]; running?: boolean; stale?: boolean }) {
  const [selected, setSelected] = useState(initial)
  return <ExecutionWorkbench events={source} selected={selected} onSelect={setSelected} running={running} stale={stale} />
}
afterEach(() => vi.useRealTimers())

describe('execution replay', () => {
  it('steps, seeks and stops at the last recorded event without re-execution', () => {
    vi.useFakeTimers()
    render(<Player />)
    fireEvent.click(screen.getByRole('button', { name: '播放事件' }))
    act(() => { vi.advanceTimersByTime(1000) })
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('2 / 3')
    act(() => { vi.advanceTimersByTime(1000) })
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('3 / 3')
    expect(screen.getByRole('button', { name: '播放事件' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '下一个事件' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '播放事件' }))
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('1 / 3')
    fireEvent.change(screen.getByLabelText('回放位置'), { target: { value: '1' } })
    act(() => { vi.advanceTimersByTime(3000) })
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('2 / 3')
    expect(screen.getByRole('button', { name: '播放事件' })).toBeInTheDocument()
  })
  it('never uses a future PG snapshot or future result in the evidence probe', async () => {
    render(<Player />)
    fireEvent.click(screen.getByRole('button', { name: '历史程序图' }))
    expect(screen.getByText('当前游标之前没有完整程序图')).toBeInTheDocument()
    expect(screen.queryByText('historical-sha')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '下一个事件' }))
    expect(screen.getByText('historical-sha')).toBeInTheDocument()
    fireEvent.click(screen.getByText('查看事件原文'))
    expect(within(screen.getByLabelText('游标证据')).queryByText(/future-only-evidence/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('选择节点或关系'))
    fireEvent.click(screen.getByRole('button', { name: /a · 任务起点 → search/ }))
    expect(screen.getByText('先读取')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '回到首个事件' }))
    expect(screen.queryByText('先读取')).not.toBeInTheDocument()
  })
  it('does not invent sequence numbers and tolerates an empty stream', () => {
    const { rerender } = render(<Player source={[]} />)
    expect(screen.getByRole('button', { name: '播放事件' })).toBeDisabled()
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('0 / 0')
    rerender(<Player source={[{ type: 'tool_call' }]} />)
    expect(screen.getByText('未记录时间戳')).toBeInTheDocument()
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('1 / 1')
    expect(within(screen.getByLabelText('游标证据')).getByText('—')).toBeInTheDocument()
  })
  it('follows new records only when enabled and freezes on refresh errors or manual navigation', () => {
    const { rerender } = render(<Player running source={events.slice(0, 2)} />)
    fireEvent.click(screen.getByRole('button', { name: '跟随最新' }))
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('2 / 2')
    rerender(<Player running source={events} />)
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('3 / 3')
    fireEvent.click(screen.getByRole('button', { name: '上一个事件' }))
    rerender(<Player running source={[...events, { type: 'terminal', sequence: 4 }]} />)
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('2 / 4')
    fireEvent.click(screen.getByRole('button', { name: '跟随最新' }))
    rerender(<Player running stale source={[...events, { type: 'terminal', sequence: 4 }, { type: 'error', sequence: 5 }]} />)
    expect(screen.getByText(/刷新失败，正在展示/)).toBeInTheDocument()
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('4 / 5')
  })
  it('stops playback and clears the cursor when the sample turn changes', () => {
    vi.useFakeTimers()
    render(<SamplePlayback turns={[{ message: '第一轮', result: { events } }, { message: '第二轮', result: { events } }]} />)
    fireEvent.click(screen.getByRole('button', { name: '播放事件' }))
    act(() => { vi.advanceTimersByTime(1000) })
    fireEvent.change(screen.getByLabelText('回放轮次'), { target: { value: '1' } })
    act(() => { vi.advanceTimersByTime(3000) })
    expect(screen.getByLabelText('浏览位置')).toHaveTextContent('1 / 3')
    expect(screen.getByRole('button', { name: '播放事件' })).toBeInTheDocument()
  })
  it('bounds rendered timeline entries while allowing navigation across a long trace', () => {
    render(<Player source={Array.from({ length: 2000 }, (_, i) => ({ type: 'model_call', sequence: i + 1 }))} />)
    expect(screen.getAllByRole('button', { name: /^浏览第/ })).toHaveLength(100)
    fireEvent.change(screen.getByLabelText('回放位置'), { target: { value: '1999' } })
    expect(screen.getByRole('button', { name: /浏览第 2000 条记录/ })).toHaveAttribute('aria-pressed', 'true')
  })
})

describe('historical snapshot display adapters', () => {
  it('uses storage order for duplicate or missing sequence numbers and does not search past a missing newer snapshot', () => {
    const later = { type: 'pg_guidance', sequence: 2 }
    expect(historicalGuidance([snapshot, later], 1)?.graph).toBeNull()
    expect(historicalGuidance([snapshot, later], 0)?.graph?.graph_id).toBe('frozen')
    expect(historicalGuidance([{ type: 'tool_call' }, snapshot], 0)).toBeNull()
  })
  it('excludes dangling and duplicated graph objects without synthesizing absent nodes', () => {
    const corrupt = { ...snapshot, subgraph: { nodes: [{ id: 'a' }, { id: 'a' }, {}], edges: [{ id: 'x', source: 'a', target: 'missing' }] } }
    expect(graphSnapshot(corrupt)?.nodes).toHaveLength(1)
    expect(graphSnapshot(corrupt)?.edges).toHaveLength(0)
    expect(graphSnapshot({ type: 'pg_guidance' })).toBeNull()
  })
  it('keeps predicted edges separate from actual execution', () => {
    expect(guidanceMarks(snapshot)).toEqual({ guidance: ['a'], predicted: ['a'], anchor: 'Start' })
    expect(guidanceMarks({ type: 'tool_result', edge_ids: ['a'] }).predicted).toEqual([])
  })
})
