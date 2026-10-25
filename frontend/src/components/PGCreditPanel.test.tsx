import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { EdgeCreditPanel, HypothesisPanel } from './PGCreditPanel'
import type { PGEdgeCredit, PGSelection } from '../types'

const credit: PGEdgeCredit = {
  edge_id: 'retry', edge_sha256: 'sha', classification: 'MIXED', support: 1, counter: 1, unknown: 0, case_count: 1,
  evidence: [{ evidence_id: 'positive', sample_ref: 'PG01:1', case_id: 'PG01', repetition: 1, turn_index: 0,
    classification: 'SUPPORT', reason: 'ADOPTED_INFORMATION_GAIN', condition: 'TRUE', exposed: true, adopted: true,
    information_gain: 'NEW_INFORMATION', task_passed: false, later_verified_objective_ids: [], guidance_sequence: 1 },
  { evidence_id: 'negative', sample_ref: 'PG01:2', case_id: 'PG01', repetition: 2, turn_index: 0,
    classification: 'COUNTER', reason: 'ADOPTED_WITHOUT_INFORMATION', condition: 'TRUE', exposed: true, adopted: true,
    information_gain: 'NO_NEW_INFORMATION', task_passed: true, later_verified_objective_ids: [], guidance_sequence: 1 }],
  contrasts: [{ case_id: 'PG01', successful: { samples: ['PG01:2'], counts: { COUNTER: 1 } }, failed: { samples: ['PG01:1'], counts: { SUPPORT: 1 } } }],
}
it('shows conflicting evidence without whole-task reward and opens source', () => {
  const open = vi.fn()
  render(<EdgeCreditPanel credit={credit} iteration={2} onEvidence={open} />)
  expect(screen.getByText('第 2 轮 · 证据有分歧')).toBeInTheDocument()
  expect(screen.getByText('采用后新增可信信息 · 任务失败')).toBeInTheDocument()
  expect(screen.getByText('采用后没有信息增益 · 任务通过')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '查看训练轨迹 PG01:1' }))
  expect(open).toHaveBeenCalledWith('PG01', 1)
  expect(screen.getByText(/关联证据，不代表因果/)).toBeInTheDocument()
})
it('keeps missing historical credit unknown', () => {
  render(<EdgeCreditPanel onEvidence={vi.fn()} />)
  expect(screen.getByText(/当前图版本尚无训练贡献记录/)).toBeInTheDocument()
})
it('shows hypothesis, rejected direction and separate validation samples', () => {
  const hypothesis = { problem: '空结果之后反复读取', failure_mode: 'NO_INFORMATION_GAIN', expected_change: 'RECOVERY',
    target_edges: ['retry'], source_samples: ['PG01:1'], evidence_ids: ['positive'] }
  const selection: PGSelection = { iteration: 2, status: 'REJECTED', reason: 'REPEATED_REJECTED_HYPOTHESIS', parent_sha256: 'x', retained_sha256: 'x',
    hypothesis, hypothesis_memory: [{ hypothesis_id: 'h1', hypothesis, iteration: 1, status: 'REJECTED', reason: 'VALIDATION_REGRESSION',
      blocks_retry: true, validation_summary: { regressed_samples: 1, improved_samples: 0, compared_samples: 1 } }],
    validation_changes: [{ case_id: 'PG07', repetition: 1, before_passed: true, after_passed: false, model_attempt_delta: 2 }] }
  render(<HypothesisPanel selection={selection} />)
  expect(screen.getByText(/同父图下禁止原方向重试/)).toBeInTheDocument()
  expect(screen.getByText(/验证样本变化（仅供审查，不回流 Refiner）/)).toBeInTheDocument()
  expect(screen.getByText('PG07:1')).toBeInTheDocument()
})
