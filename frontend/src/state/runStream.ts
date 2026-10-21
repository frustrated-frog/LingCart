import type { RunEvent } from '../types'

export type StreamConnectionState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'closed'

export const AGENT_STATUS_LABELS: Record<string, string> = {
  QUEUED: '任务已进入队列',
  UNDERSTANDING: '正在理解你的需求',
  PLANNING: '正在制定执行计划',
  USING_TOOL: '正在调用工具获取证据',
  VERIFYING: '正在核验结果',
  RESPONDING: '正在组织回答',
  WAITING_USER: '等待你确认或补充信息',
  COMPLETED: '任务已完成',
  FAILED: '任务未完成',
  CANCELLED: '任务已取消',
}

export interface ProjectedAgentStatus {
  code: string
  label: string
  sequence: number
}

export interface ProjectedResponse {
  messageId: string | null
  content: string
  state: 'idle' | 'streaming' | 'completed' | 'failed'
  failureCode?: string
  chunkCount: number
}

function sequence(event: RunEvent): number {
  return typeof event.sequence === 'number' ? event.sequence : -1
}

export function projectAgentStatus(events: RunEvent[]): ProjectedAgentStatus | null {
  const event = [...events].reverse().find((item) => item.type === 'agent_status_changed')
  if (!event || typeof event.status !== 'string') return null
  return {
    code: event.status,
    label: typeof event.label === 'string' ? event.label : AGENT_STATUS_LABELS[event.status] ?? '任务正在执行',
    sequence: sequence(event),
  }
}

export function projectRunResponse(events: RunEvent[]): ProjectedResponse {
  const ordered = [...events].sort((left, right) => sequence(left) - sequence(right))
  const started = [...ordered].reverse().find((event) =>
    event.type === 'response_started' && typeof event.message_id === 'string')
  if (!started || typeof started.message_id !== 'string') {
    return { messageId: null, content: '', state: 'idle', chunkCount: 0 }
  }
  const messageId = started.message_id
  const chunks = new Map<number, string>()
  let terminal: RunEvent | undefined
  ordered.forEach((event) => {
    if (event.message_id !== messageId) return
    if (event.type === 'response_delta' && typeof event.chunk_index === 'number'
      && event.chunk_index >= 0 && typeof event.delta === 'string' && !chunks.has(event.chunk_index)) {
      chunks.set(event.chunk_index, event.delta)
    }
    if (event.type === 'response_completed' || event.type === 'response_failed') terminal = event
  })
  const contiguous: string[] = []
  while (chunks.has(contiguous.length)) contiguous.push(chunks.get(contiguous.length) as string)
  return {
    messageId,
    content: contiguous.join(''),
    state: terminal?.type === 'response_completed' ? 'completed'
      : terminal?.type === 'response_failed' ? 'failed' : 'streaming',
    failureCode: terminal?.type === 'response_failed' && typeof terminal.code === 'string' ? terminal.code : undefined,
    chunkCount: contiguous.length,
  }
}
