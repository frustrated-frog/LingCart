import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { RecoveryLabPage } from './RecoveryLabPage'
import type { RecoveryReplayDetail, RecoveryReplaySample } from '../types'

const sample: RecoveryReplaySample = {
  profile_id:'adaptive-recovery-search-v1',case_id:'R1',repetition:1,origin_sha256:'origin',database_sha256:'db',
  passed:false,failures:['任务仍未满足全部目标'],terminal_reason:'MODEL_STOPPED',tokens:250,model_calls:3,tool_calls:4,suffix_steps:1,gates:[],
  timeline:[{step_id:'step-2',tool:'search_products',status:'SUCCEEDED',failure_kind:null,new_records:3}],
  searches:[{search_id:'search',status:'SELECTED',chosen_branch:'b1',reason:'第二条路径读到了有效记录',probes:[
    {branch_id:'b0',probe_id:'probe-b0-0',status:'SUCCEEDED',tool:'search_products',code:null,product_records:0},
    {branch_id:'b1',probe_id:'probe-b1-0',status:'SUCCEEDED',tool:'discover_products',code:null,product_records:3},
  ]}],
}
const run: RecoveryReplayDetail = {
  run_id:'smoke',status:'COMPLETED',execution_source:'live_provider',source_revision:'abc12345',partial:false,warnings:[],expected_samples:6,
  profiles:[{profile_id:'adaptive-v1',display_name:'原始决策',role:'reference'},
    {profile_id:'adaptive-recovery-control-v1',display_name:'配对控制',role:'control'},
    {profile_id:'adaptive-recovery-search-v1',display_name:'有界恢复搜索',role:'candidate'}],
  samples:[sample,{...sample,profile_id:'adaptive-recovery-control-v1',searches:[],passed:true,failures:[],tokens:200},
    {...sample,repetition:2,searches:[],passed:true,failures:[],tokens:null}],
}
function page() {
  render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter><RecoveryLabPage/></MemoryRouter></QueryClientProvider>)
}
beforeEach(() => {
  vi.restoreAllMocks()
  vi.spyOn(api,'recoveryReplays').mockResolvedValue({runs:[run]})
  vi.spyOn(api,'recoveryReplay').mockResolvedValue(run)
})
it('separates actual probes from final task success',async () => {
  page()
  expect(await screen.findByText('第二条路径读到了有效记录')).toBeInTheDocument()
  expect(screen.getByText('任务未完成')).toBeInTheDocument()
  expect(screen.getByText('任务完成')).toBeInTheDocument()
  expect(screen.getByText('250 Token')).toBeInTheDocument()
  expect(screen.getByText('已完成 · 3 条商品记录')).toBeInTheDocument()
  expect(screen.getByText('尚无这组回放结果。')).toBeInTheDocument()
  expect(screen.getByText(/分支读取成功不代表任务完成/)).toBeInTheDocument()
  fireEvent.click(screen.getByText('查看未通过原因'))
  expect(screen.getByText('任务仍未满足全部目标')).toBeInTheDocument()
})
it('switches matched repetitions and never invents missing cost',async () => {
  page()
  await screen.findByText('冻结的失败检查点')
  fireEvent.change(screen.getByLabelText('检查点样本'),{target:{value:'origin:2'}})
  expect(screen.getByText('未知 Token')).toBeInTheDocument()
  expect(screen.queryByText('第二条路径读到了有效记录')).not.toBeInTheDocument()
})
it('shows an empty experiment catalog without fake results',async () => {
  vi.mocked(api.recoveryReplays).mockResolvedValue({runs:[]})
  page()
  expect(await screen.findByText('尚无同状态回放')).toBeInTheDocument()
  expect(api.recoveryReplay).not.toHaveBeenCalled()
  expect(screen.queryByText('任务完成')).not.toBeInTheDocument()
})
it('preserves partial evidence and exposes read errors',async () => {
  vi.mocked(api.recoveryReplay).mockResolvedValue({...run,partial:true,warnings:['缺少配对样本']})
  page()
  expect(await screen.findByText(/缺少配对样本/)).toBeInTheDocument()
  expect(screen.getByText('250 Token')).toBeInTheDocument()
})
it('provides a retry when the catalog fails',async () => {
  vi.mocked(api.recoveryReplays).mockRejectedValue(new Error('回放产物读取失败'))
  page()
  expect(await screen.findByText('回放产物读取失败')).toBeInTheDocument()
  expect(screen.getByRole('button',{name:'重新读取'})).toBeInTheDocument()
})
it('uses contract profiles and distinguishes deferred gates from actual search',async () => {
  vi.mocked(api.recoveryReplay).mockResolvedValue({...run,
    profiles:[{profile_id:'adaptive-recovery-eager-v2',display_name:'立即恢复搜索 v2',role:'control'},
      {profile_id:'adaptive-recovery-search-v2',display_name:'按需恢复搜索 v2',role:'candidate'}],
    samples:[{...sample,profile_id:'adaptive-recovery-search-v2',searches:[],gates:[
      {gate_id:'g1',status:'DEFERRED',code:'ORDINARY_RECOVERY_FIRST',failure_count:1,threshold:2},
      {gate_id:'g2',status:'OPEN',code:'RECOVERY_STALLED',failure_count:2,threshold:2},
    ]}],
  })
  page()
  expect(await screen.findByRole('region',{name:'按需恢复搜索 v2'})).toBeInTheDocument()
  expect(screen.getByText('先尝试常规恢复')).toBeInTheDocument()
  expect(screen.getByText('允许展开搜索')).toBeInTheDocument()
  expect(screen.getByText('失败 2 / 阈值 2')).toBeInTheDocument()
  expect(screen.queryByText('已选择首步')).not.toBeInTheDocument()
  expect(screen.queryByText('原始决策')).not.toBeInTheDocument()
})
