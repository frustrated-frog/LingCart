import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CaseStateChanges } from './CaseStateChanges'
import type { CaseEvidenceTransition, RunEvent } from '../types'

const report: CaseEvidenceTransition = {
  version: 'case-evidence-transition-v1', report_id: 'report', assessment_report_id: 'proposal', assessment_id: 'assessment',
  input_hash: 'input', case_id: 'case', from_revision: 2, to_revision: 3, commit_status: 'COMMITTED', policy_outcome: 'ELIGIBLE', error_code: null,
  changes: [
    { requirement_id: 'serial', evidence_type: 'SERIAL_LABEL', description: '序列号标签状态', from_status: 'SUFFICIENT', to_status: 'SUFFICIENT',
      action: 'PRESERVE', reason: '新包装材料不影响标签结论', reason_code: 'UNCHANGED_DEPENDENCIES', prior_assessment_id: 'old' },
    { requirement_id: 'packaging', evidence_type: 'PACKAGING', description: '原包装完整性', from_status: 'UNKNOWN', to_status: 'SUFFICIENT',
      action: 'REASSESS', reason: '本轮包装材料已补全', reason_code: 'MATCHING_MATERIAL_ADDED', prior_assessment_id: 'old' },
  ],
}
const committed: RunEvent = { type: 'tool_result', sequence: 4, result: { evidence_transition: report } }

describe('CaseStateChanges', () => {
  it('renders actual committed transitions, reasons and source identities', () => {
    render(<CaseStateChanges events={[committed]} />)
    expect(screen.getByText('已落账')).toBeInTheDocument()
    expect(screen.getByText('保留结论')).toBeInTheDocument()
    expect(screen.getByText('重新核验')).toBeInTheDocument()
    expect(screen.getByText('新包装材料不影响标签结论')).toBeInTheDocument()
    fireEvent.click(screen.getByText('查看落账依据'))
    expect(screen.getByText('assessment')).toBeVisible()
  })
  it('never displays a preflight proposal as committed evidence', () => {
    render(<CaseStateChanges events={[{ type: 'case_evidence_transition_assessed', report }]} />)
    expect(screen.getByText(/已有核验提议，尚未观察到/)).toBeInTheDocument()
    expect(screen.queryByText('已落账')).not.toBeInTheDocument()
  })
  it('separates interpretation repair from material changes', () => {
    render(<CaseStateChanges events={[{ type: 'case_material_target_reconciled', from_requirement_id: 'packaging', to_requirement_id: 'serial', source_quote: '标签清晰' },
      { type: 'case_turn_verification_conflict', terminal: true, question: '确认材料归属' }]} />)
    fireEvent.click(screen.getByText(/本轮理解校正/))
    expect(screen.getByText('校正隐式材料归属')).toBeVisible()
    expect(screen.getByText('需要进一步澄清')).toBeVisible()
    expect(screen.queryByText('已落账')).not.toBeInTheDocument()
  })
  it('switches history and marks a superseded case revision', () => {
    render(<CaseStateChanges events={[committed, { type: 'case_state_projected', state: { case_id: 'case', case_revision: 4 } },
      { ...committed, sequence: 6, result: { evidence_transition: { ...report, from_revision: 3, to_revision: 4 } } }]} />)
    expect(screen.queryByText(/历史版本/)).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('选择材料变化记录'), { target: { value: '4' } })
    expect(screen.getByRole('status')).toHaveTextContent('历史版本')
  })
  it('handles empty, broken, failed and partial records explicitly', () => {
    const { rerender } = render(<CaseStateChanges events={[]} />)
    expect(screen.getByText(/尚无已提交/)).toBeInTheDocument()
    rerender(<CaseStateChanges events={[{ ...committed, result: { evidence_transition: { ...report, to_revision: 1 } } }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('记录不完整')
    rerender(<CaseStateChanges events={[{ ...committed, result: { evidence_transition: { ...report, error_code: 'SEMANTIC_MODEL_UNAVAILABLE',
      changes: report.changes.map(c => ({ ...c, action: 'REASSESS', to_status: 'UNKNOWN' })) } } }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('本轮核验未得到可靠结论')
    expect(screen.queryByText('保留结论')).not.toBeInTheDocument()
  })
})
