import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { expect, it } from 'vitest'
import type { PGJob, SIMAttempt, SIMMemory } from '../types'
import { SIMMemoryPanel } from './SIMMemoryPanel'

const attempt: SIMAttempt = {
  attempt_id: 'first', experiment_id: 'sim', run_id: 'source', iteration: 1,
  parent_sha256: 'parent', candidate_sha256: 'candidate', training_sha256: 'training', candidate_training_sha256: 'measured',
  environment: { models_sha256: 'model-before', dataset_sha256: 'dataset', protocol_sha256: 'protocol', source_sha256: 'source', limits_sha256: 'limits', train_cases: ['PG01'] },
  change: '先补充观察再调整恢复路径', expected_change: 'RECOVERY', patch_sha256: 'patch',
  patch: { reason: '恢复', source_cases: ['PG01'], add_nodes: [], delete_nodes: [], add_edges: [], delete_edges: [] },
  status: 'REJECTED', reason: 'VALIDATION_REGRESSION', predicted_fixes: ['PG01:1'], fixed: [], missed: ['PG01:1'], regressed: [],
  observation_status: 'MEASURED', validation_summary: { compared_samples: 1, improved_samples: 0, regressed_samples: 1 },
}
const memory: SIMMemory = { schema_version: 1, mechanism: 'pg-sim-v1', online_profile: 'adaptive-pg-context-v4',
  execution_source: 'test_harness', retained_sha256: 'parent', problems: [{ problem_id: 'sim-problem', diagnosis: '重复无效观察', failure_mode: 'NO_INFORMATION_GAIN', attempts: [attempt] }] }
// 该组件只读取这些投影字段，完整任务由页面测试覆盖。
const job = { frozen: { evolution_policy: 'sim-v1', sim_memory_mode: 'problems', sim_source_job_id: 'pg-source', sim_source_sha256: 'memory-sha' },
  history: [], sim_source: memory, evolution: null } as unknown as PGJob
const show = (data: PGJob) => render(<MemoryRouter><SIMMemoryPanel job={data} /></MemoryRouter>)

it('shows inherited failures and persisted partial attempts without inventing unmeasured outcomes', () => {
  show({ ...job, history: [{ iteration: 1, status: 'REJECTED', reason: 'INVALID_PROPOSAL_OR_REPORT', parent_sha256: 'parent', retained_sha256: 'parent',
    sim_problem_id: 'sim-problem', hypothesis: { problem: '重复无效观察', failure_mode: 'NO_INFORMATION_GAIN', expected_change: 'RECOVERY', target_edges: [], source_samples: [], evidence_ids: [] },
    sim_attempt: { ...attempt, attempt_id: 'second', run_id: 'target', environment: { ...attempt.environment, models_sha256: 'model-after' },
      observation_status: 'NOT_EVALUATED', candidate_training_sha256: null, fixed: [], missed: [],
      validation_summary: { compared_samples: 0, improved_samples: 0, regressed_samples: 0 } } }] })
  expect(screen.getByRole('status')).toHaveTextContent('尚未最终封存')
  expect(screen.getByRole('link', { name: '查看继承来源 ↗' })).toHaveAttribute('href', '/experiments/pg?job=pg-source')
  expect(screen.getByText('未修复 PG01:1')).toBeInTheDocument()
  expect(screen.getByText('未评测，效果未知')).toBeInTheDocument()
  expect(screen.getByText('环境已变化，历史结论需重新验证')).toBeInTheDocument()
  expect(screen.getAllByRole('listitem')).toHaveLength(2)
  expect(within(screen.getAllByRole('listitem')[1]).queryByText('已修复 无')).not.toBeInTheDocument()
})

it('uses the sealed memory once without duplicating source attempts', () => {
  show({ ...job, evolution: { sim_memory: memory } })
  expect(screen.queryByRole('status')).not.toBeInTheDocument()
  expect(screen.getAllByRole('listitem')).toHaveLength(1)
  expect(screen.getByText('拒绝修改 · 验证退化')).toBeInTheDocument()
})

it('shows an honest cold-start empty state', () => {
  show({ ...job, sim_source: null, frozen: { ...job.frozen, sim_source_job_id: null, sim_source_sha256: null } })
  expect(screen.getByText('尚无可继承的问题历史')).toBeInTheDocument()
  expect(screen.queryByRole('listitem')).not.toBeInTheDocument()
})
