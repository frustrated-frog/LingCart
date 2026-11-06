import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { RSILabPage } from './RSILabPage'
import type { RSIRunDetail, RSISampleSide } from '../types'
const side: RSISampleSide = { passed: false, tokens: 250, source: 'live_provider', recovery: {
 episodes: [],timeline:[{step_id:'s1',sequence:3,tool:'search_products',status:'SUCCEEDED',failure_kind:'empty_results',new_records:0}],
 task_passed:false,verified_objectives:0,repeated_without_new_records:0,note:'观察不代表成功' } }
const run: RSIRunDetail = { run_id:'smoke',status:'COMPLETED',target_module:'recovery',source_revision:'abcdef123',historical:true,execution_source:'live_provider',partial:false,warnings:[],spec:{},comparison:{status:'COMPLETED',observed_improvement:false},
 candidates:[{round:0,candidate:0,status:'COST_OR_NOISE',selected:false,hypothesis:'区分空结果和工具超时',parent_sha256:'seed',program_sha256:'child',measurement:null,
 changes:[{field:'recovery.trigger',before:'all_failures',after:'tool_failures'}],samples:[{case_id:'R1',repetition:1,parent:side,candidate:{...side,passed:true,tokens:500}}]},
 {round:1,candidate:0,status:'INVALID_PROPOSAL',selected:false,hypothesis:'',parent_sha256:'seed',program_sha256:null,measurement:null,changes:[],samples:[]}] }
function page() { render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter><RSILabPage/></MemoryRouter></QueryClientProvider>) }
beforeEach(() => { vi.restoreAllMocks(); vi.spyOn(api,'rrsiRuns').mockResolvedValue({runs:[run]}); vi.spyOn(api,'rrsiRun').mockResolvedValue(run) })
it('shows paired evidence and rejected candidates',async () => {
 page();expect(await screen.findByText('区分空结果和工具超时')).toBeInTheDocument()
 expect(screen.getByText('任务未通过')).toBeInTheDocument();expect(screen.getByText('任务通过')).toBeInTheDocument()
 expect(screen.getByText('250 Token')).toBeInTheDocument();expect(screen.getByText('500 Token')).toBeInTheDocument()
 expect(screen.getByText(/历史版本记录/)).toBeInTheDocument();expect(screen.getByText('尚未满足收益条件')).toBeInTheDocument()
 fireEvent.click(screen.getByRole('button',{name:/第 2 轮/}))
 expect(await screen.findByText('提案未通过协议校验，没有可用的结构化修改假设。')).toBeInTheDocument()
 expect(screen.getAllByText(/尚无该版本的验证轨迹/)).toHaveLength(2)
})
it('does not invent metrics for empty history',async () => {
 vi.mocked(api.rrsiRuns).mockResolvedValue({runs:[]});page()
 expect(await screen.findByText('尚无演化证据')).toBeInTheDocument();expect(screen.queryByText('0%')).not.toBeInTheDocument()
 expect(api.rrsiRun).not.toHaveBeenCalled()
})
it('shows read failure',async () => {
 vi.mocked(api.rrsiRuns).mockRejectedValue(new Error('文件读取失败'));page()
 expect(await screen.findByText('文件读取失败')).toBeInTheDocument()
})
it('keeps partial results inspectable',async () => {
 vi.mocked(api.rrsiRun).mockResolvedValue({...run,partial:true,warnings:['验证报告尚未完整写入'],comparison:null});page()
 expect(await screen.findByText(/验证报告尚未完整写入/)).toBeInTheDocument()
 expect(screen.getByText('尚未完成留出对照')).toBeInTheDocument()
})
