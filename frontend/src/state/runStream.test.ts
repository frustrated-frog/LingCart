import { describe, expect, it } from 'vitest'
import { projectAgentStatus, projectRunResponse } from './runStream'

describe('run stream projection', () => {
  it('projects public status without exposing internal reasoning', () => {
    const status = projectAgentStatus([
      { type: 'agent_status_changed', sequence: 1, status: 'UNDERSTANDING', label: '正在理解你的需求' },
      { type: 'model_started', sequence: 2, purpose: 'dialogue_understanding' },
      { type: 'agent_status_changed', sequence: 3, status: 'USING_TOOL', label: '正在查询订单状态' },
    ])

    expect(status).toEqual({ code: 'USING_TOOL', label: '正在查询订单状态', sequence: 3 })
  })

  it('rebuilds only contiguous chunks and deduplicates replayed events', () => {
    const response = projectRunResponse([
      { type: 'response_delta', sequence: 4, message_id: 'answer', chunk_index: 2, delta: '界' },
      { type: 'response_started', sequence: 1, message_id: 'answer' },
      { type: 'response_delta', sequence: 2, message_id: 'answer', chunk_index: 0, delta: '你' },
      { type: 'response_delta', sequence: 2, message_id: 'answer', chunk_index: 0, delta: '你' },
    ])

    expect(response).toMatchObject({ messageId: 'answer', content: '你', state: 'streaming', chunkCount: 1 })
  })

  it('selects the latest provider attempt after an early stream failure', () => {
    const response = projectRunResponse([
      { type: 'response_started', sequence: 1, message_id: 'failed' },
      { type: 'response_failed', sequence: 2, message_id: 'failed', code: 'MODEL_TIMEOUT' },
      { type: 'response_started', sequence: 3, message_id: 'answer' },
      { type: 'response_delta', sequence: 4, message_id: 'answer', chunk_index: 0, delta: '完成' },
      { type: 'response_completed', sequence: 5, message_id: 'answer' },
    ])

    expect(response).toMatchObject({ messageId: 'answer', content: '完成', state: 'completed' })
  })
})
