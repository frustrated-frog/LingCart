import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { RequestSemantics, RunDetail } from '../types'
import { RequestSemanticsPanel } from './RequestSemanticsPanel'

const source = { source: 'CURRENT_TURN' as const, history_index: null, start: 0, end: 5, text: '耳机改一款', source_sha256: 'c'.repeat(64) }
const contract: RequestSemantics = {
  version: 'request-semantics/v1', sha256: 'a'.repeat(64), previous_sha256: 'b'.repeat(64),
  requirements: [
    { source, requirement: { requirement_id: 'headphones', kind: 'OUTCOME', description: '推荐耳机', source_quote: source.text,
      targets: [], predicate: null, restriction: null,
      outcome: { kind: 'recommend', quantity: 1, order_id: null, question: null, product_ids: [] } } },
    { source: { ...source, text: '不超过400元' }, requirement: { requirement_id: 'budget', kind: 'CONDITION', description: '单件价格不超过400元',
      source_quote: '不超过400元', targets: ['headphones'], predicate: { field: 'price', op: 'lte', value: 400 }, outcome: null, restriction: null } },
    { source: { ...source, text: '键盘不变' }, requirement: { requirement_id: 'keyboard', kind: 'OUTCOME', description: '推荐键盘', source_quote: '键盘不变',
      targets: [], predicate: null, restriction: null,
      outcome: { kind: 'recommend', quantity: 1, order_id: null, question: null, product_ids: [] } } },
  ],
  revisions: [
    { operation: 'REPLACE', previous_id: 'headphones', requirement_id: 'headphones', source_quote: '耳机改一款' },
    { operation: 'PRESERVE', previous_id: 'keyboard', requirement_id: 'keyboard', source_quote: null },
    { operation: 'RETRACT', previous_id: 'old_preference', requirement_id: null, source_quote: '取消颜色偏好' },
  ],
}

function detail(result: RunDetail['result'] = {}, checkpoint: RunDetail['checkpoint'] = {}): RunDetail {
  return { run_id: 'semantic-run', conversation_id: 'session', request: source.text, created_at: '',
    status: 'PARTIAL', agent: 'shopping', terminal_reason: null, profile: { profile_id: 'adaptive-v1' }, event_count: 0, tool_calls: 0,
    model_calls: 0, latency_ms: null, tokens: null, result, checkpoint, tool_ledger: [], commands: [],
    trace_projection: { integrity: {}, summary: {}, timeline: [] } }
}

describe('RequestSemanticsPanel', () => {
  it('shows scoped conditions, accepted revisions and actual restrictions', () => {
    render(<RequestSemanticsPanel detail={detail({ request_semantics: contract,
      semantic_revision: { preserved: ['keyboard'], changed: ['headphones'], added: [], retracted: [],
        previous_sha256: contract.previous_sha256, current_sha256: contract.sha256 },
      execution_boundaries: [{ restriction: 'NO_PAYMENT', source: { ...source, text: '先别购买' }, origin: 'adaptive' }],
    })} />)
    expect(screen.getByRole('heading', { name: '已匹配执行目标' })).toBeInTheDocument()
    const ear = screen.getByRole('heading', { name: '推荐耳机' }).closest('article')!
    const keyboard = screen.getByRole('heading', { name: '推荐键盘' }).closest('article')!
    expect(within(ear).getByText('已修订 · 重新核验')).toBeInTheDocument()
    expect(within(ear).getByText('单件价格不超过400元')).toBeInTheDocument()
    expect(within(keyboard).queryByText('单件价格不超过400元')).not.toBeInTheDocument()
    expect(within(keyboard).getByText('需求保留')).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: '取消颜色偏好' })).toBeInTheDocument()
    expect(screen.getByText('禁止支付')).toBeInTheDocument()
    expect(screen.getByText('先别购买')).toBeInTheDocument()
  })

  it('does not label an uncompiled extraction as accepted', () => {
    render(<RequestSemanticsPanel detail={detail({}, { semantic_extraction: { contract } })} />)
    expect(screen.getByRole('heading', { name: '已提取 · 待匹配执行目标' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '已匹配执行目标' })).not.toBeInTheDocument()
  })

  it('does not invent semantics or restrictions for historical runs', () => {
    render(<RequestSemanticsPanel detail={detail()} />)
    expect(screen.getByText(/本次运行尚无独立需求契约/)).toBeInTheDocument()
    expect(screen.getByText(/本次运行未记录额外执行限制/)).toBeInTheDocument()
    expect(screen.queryByText('目标数量')).not.toBeInTheDocument()
  })

  it('ignores revision badges from a different contract', () => {
    render(<RequestSemanticsPanel detail={detail({ request_semantics: contract,
      semantic_revision: { preserved: [], changed: ['headphones'], added: [], retracted: [], previous_sha256: null, current_sha256: 'stale' },
    })} />)
    expect(screen.queryByText('已修订 · 重新核验')).not.toBeInTheDocument()
  })
})
