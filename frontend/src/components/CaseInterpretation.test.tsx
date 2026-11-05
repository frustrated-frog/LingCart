import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { RunEvent } from '../types'
import { CaseInterpretation } from './CaseInterpretation'

const control: RunEvent = { type:'case_control_interpreted', sequence:5, status:'VERIFIED',
  control:{kind:'REQUIRE_CONFIRMATION', scope:'CASE', case_id:'case-a', restriction:null},
  source:{text:'先继续检查，这不是同意提交'} }
const sources: RunEvent = { type:'request_sources_resolved', purpose:'dialogue_understanding', bindings:[
  {path:['control_intents',0], source:{source_ref:'current-a', origin:'CURRENT_TURN', text:'先继续检查，这不是同意提交', start:0,end:14}},
  {path:['objectives',0], source:{source_ref:'current-a', origin:'CURRENT_TURN', text:'先继续检查，这不是同意提交', start:0,end:14}},
] }

describe('CaseInterpretation', () => {
  it('shows scope, exact source and distinct non-authorizing effect', () => {
    render(<CaseInterpretation events={[sources,control]} />)
    expect(screen.getByText('继续核验，等待确认')).toBeInTheDocument()
    expect(screen.getByText(/作用范围：案件 case-a/)).toBeInTheDocument()
    expect(screen.getByText(/当前没有提交授权/)).toBeInTheDocument()
    expect(screen.getAllByText(/片段 1/)).toHaveLength(1)
    fireEvent.click(screen.getByText(/片段 1/))
    expect(screen.getByText(/字符范围 0–14/)).toBeVisible()
    expect(screen.getByText(/不代表材料充分或已执行/)).toBeInTheDocument()
  })
  it('marks selected older interpretation as historical', () => {
    render(<CaseInterpretation events={[control, {...control, sequence:8,
      control:{kind:'PROHIBIT',scope:'SESSION',case_id:null,restriction:'NO_REFUND'}}]} />)
    expect(screen.getByText(/作用范围：当前会话/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('选择控制解释'),{target:{value:'5'}})
    expect(screen.getByText(/历史解释/)).toHaveAttribute('role','status')
  })
  it('handles loading, empty and incomplete evidence without inventing outcomes', () => {
    const {rerender} = render(<CaseInterpretation events={[]} running />)
    expect(screen.getByText(/正在理解本轮请求/)).toBeInTheDocument()
    rerender(<CaseInterpretation events={[]} />)
    expect(screen.getByText(/本轮没有新的控制意图解释/)).toBeInTheDocument()
    rerender(<CaseInterpretation events={[{...control, control:{kind:'PROHIBIT', scope:'CASE'}}]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('记录不完整')
  })
})
