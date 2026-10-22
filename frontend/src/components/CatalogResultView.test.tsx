import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ConstraintPlanResultArtifact } from '../types'
import { ResultArtifactViews } from './CatalogResultView'

function optimizedArtifact(
  status: 'BEST_KNOWN' | 'PROVEN_OPTIMAL',
): ConstraintPlanResultArtifact {
  const complete = status === 'PROVEN_OPTIMAL'
  return {
    artifact_type: 'constraint_plan_result',
    version: 'constraint-plan/v1',
    solution_set_id: `solution-${status}`,
    plan_spec: {
      goals: [
        { goal_id: 'left', source_quote: '一个甲', quantity: 1 },
        { goal_id: 'right', source_quote: '一个乙', quantity: 1 },
      ],
      constraints: [],
      optimization: { requirement_id: 'minimum_cost', direction: 'MIN' },
    },
    compilation_coverage: [],
    provider_registry_version: 'providers-v1',
    domains: [
      { goal_id: 'left', provider: 'TEST', provider_fingerprint: 'left-fp', candidate_count: 1, evidence: {} },
      { goal_id: 'right', provider: 'TEST', provider_fingerprint: 'right-fp', candidate_count: 1, evidence: {} },
    ],
    feasibility: 'FEASIBLE',
    solutions: [{
      solution_id: 'candidate-solution',
      optimization_value: 12,
      selections: [
        { goal_id: 'left', source_quote: '一个甲', quantity: 1, candidates: [] },
        { goal_id: 'right', source_quote: '一个乙', quantity: 1, candidates: [] },
      ],
      constraint_checks: [],
      resource_checks: [],
    }],
    search: {
      complete,
      nodes_examined: complete ? 20 : 10,
      feasible_count: 4,
      feasible_count_exact: complete,
      retained_feasible_count: 3,
      stored_solution_limit_reached: false,
      max_search_nodes: 10,
    },
    optimality: {
      status,
      objective_value: 12,
      proof: complete ? 'EXHAUSTIVE_SEARCH' : 'BOUNDED_SEARCH',
    },
    selection_policy: 'EXPLICIT_OPTIMIZATION',
    verification: {
      status: 'VERIFIED',
      verdict: complete ? 'SATISFIED' : 'BEST_KNOWN',
      checked_requirements: [],
      global_goal: complete,
    },
  }
}

describe('ConstraintArtifactView optimality evidence', () => {
  it('does not present a bounded best-known result as globally optimal', () => {
    render(<ResultArtifactViews artifacts={[optimizedArtifact('BEST_KNOWN')]} />)

    expect(screen.getByText('当前已知约束方案')).toBeInTheDocument()
    expect(screen.getByText('硬约束已核验 · 最优性未证明')).toBeInTheDocument()
    expect(screen.getByText('显式优化 · 当前已知最优')).toBeInTheDocument()
  })

  it('shows global optimality only with an exhaustive proof', () => {
    render(<ResultArtifactViews artifacts={[optimizedArtifact('PROVEN_OPTIMAL')]} />)

    expect(screen.getByText('硬约束与最优性已核验')).toBeInTheDocument()
    expect(screen.getByText('显式优化 · 已证明全局最优')).toBeInTheDocument()
  })
})
