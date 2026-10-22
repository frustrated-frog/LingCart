import { waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamRunEvents } from './client'
import type { RunEvent } from '../types'

function eventStream(payload: string) {
  const chunks = [new TextEncoder().encode(payload)]
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: vi.fn(async () => chunks.length
          ? { done: false, value: chunks.shift() }
          : { done: true, value: undefined }),
      }),
    },
  } as unknown as Response
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamRunEvents', () => {
  it('reconnects from the last persisted sequence without replaying events', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(eventStream('id: 1\nevent: response_delta\ndata: {"type":"response_delta","sequence":1,"message_id":"m","chunk_index":0,"delta":"你"}\n\n'))
      .mockResolvedValueOnce(eventStream('id: 2\nevent: terminal\ndata: {"type":"terminal","sequence":2,"status":"SUCCEEDED"}\n\n'))
    vi.stubGlobal('fetch', fetchMock)
    const events: RunEvent[] = []
    const states: string[] = []

    streamRunEvents('run-1', 'session-1', (event) => events.push(event), 0, (state) => states.push(state))

    await waitFor(() => expect(events.map((event) => event.sequence)).toEqual([1, 2]), { timeout: 2_000 })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toContain('after=1')
    expect(fetchMock.mock.calls[1][1].headers['Last-Event-ID']).toBe('1')
    expect(states).toContain('reconnecting')
    expect(states.at(-1)).toBe('closed')
  })
})
