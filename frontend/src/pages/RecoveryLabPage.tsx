import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { GitBranch, RefreshCw, ScanLine } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, STATUS_LABELS, TERMINAL_REASON_LABELS, TOOL_LABELS } from '../components/SystemUI'
import type { RecoveryReplayProfile, RecoveryReplaySample, RecoverySearch } from '../types'
import '../styles/recovery-lab.css'

const names: Record<string, string> = {
  'adaptive-v1': '原始决策', 'adaptive-recovery-control-v1': '配对控制', 'adaptive-recovery-search-v1': '有界恢复搜索',
}
const roles = { reference: '参考策略', control: '配对控制', candidate: '研究候选', unknown: '合同身份待核实' }
const searchStatus: Record<string, string> = {
  SELECTED: '已选择首步', ABSTAINED: '未选出方案', INVALID: '提案或选择无效', BUDGET_SKIPPED: '预算不足，跳过搜索', PARTIAL: '探查进行中',
}

function SearchBranches({ search }: { search: RecoverySearch }) {
  const branches = [...new Set(search.probes.map(p => p.branch_id))]
  return <section className="recovery-search" aria-label="恢复探查分支">
    <header><GitBranch size={16}/><b>{searchStatus[search.status] ?? search.status}</b><code>{search.search_id.slice(0, 8)}</code></header>
    <div className="recovery-branch-grid">{branches.map(branch => <article key={branch} className={search.chosen_branch === branch ? 'chosen' : ''}>
      <div><b>{branch.toUpperCase()}</b><span>{search.chosen_branch === branch ? '选择首步' : '备选探查'}</span></div>
      {search.probes.filter(p => p.branch_id === branch).map(probe => <p key={probe.probe_id}>
        <strong>{probe.tool ? TOOL_LABELS[probe.tool] ?? probe.tool : '未执行工具'}</strong>
        <span>{STATUS_LABELS[probe.status] ?? probe.status}{probe.product_records !== null ? ` · ${probe.product_records} 条商品记录` : ''}</span>
        {probe.code && <code>{probe.code}</code>}
      </p>)}
    </article>)}</div>
    {!branches.length && <p>尚无实际探查记录。</p>}
    {search.reason && <p className="recovery-choice"><b>选择依据</b>{search.reason}</p>}
  </section>
}

function ReplayCard({ sample, profile }: { sample?: RecoveryReplaySample; profile: RecoveryReplayProfile }) {
  const name = profile.display_name !== profile.profile_id ? profile.display_name : names[profile.profile_id] ?? profile.profile_id
  return <section className={`recovery-arm ${profile.role === 'candidate' ? 'search-arm' : ''}`} aria-label={name}>
    <header><span>{name}</span><small>{roles[profile.role]} · {profile.profile_id}</small></header>
    {!sample ? <p className="recovery-empty">尚无这组回放结果。</p> : <>
      <div className={`recovery-result ${sample.passed ? 'passed' : 'failed'}`}><b>{sample.passed ? '任务完成' : '任务未完成'}</b><span>{sample.tokens?.toLocaleString() ?? '未知'} Token</span></div>
      <dl className="recovery-cost"><div><dt>模型调用</dt><dd>{sample.model_calls}</dd></div><div><dt>工具调用</dt><dd>{sample.tool_calls}</dd></div><div><dt>正式步骤</dt><dd>{sample.suffix_steps}</dd></div></dl>
      {sample.gates.length > 0 && <section className="recovery-gates" aria-label="搜索触发记录"><h3>搜索触发记录</h3>
        <ol>{sample.gates.map(gate => <li key={gate.gate_id} className={gate.status === 'OPEN' ? 'gate-open' : 'gate-deferred'}>
          <div><b>{gate.status === 'OPEN' ? '允许展开搜索' : gate.status === 'DEFERRED' ? '先尝试常规恢复' : gate.status}</b>
            <span>失败 {gate.failure_count} / 阈值 {gate.threshold}</span></div>
          <p>{gate.code === 'ORDINARY_RECOVERY_FIRST' ? '尚未达到触发阈值，本次门控不调用模型。' : gate.code === 'RECOVERY_STALLED' ? '连续失败且没有新增有效观察，继续检查搜索预算。' : gate.code}</p>
        </li>)}</ol>
      </section>}
      {sample.searches.map(search => <SearchBranches key={search.search_id} search={search}/>)}
      <h3>失败后的正式执行</h3>
      <ol className="recovery-timeline">{sample.timeline.map((step, i) => <li key={`${step.step_id}-${i}`} className={step.failure_kind ? 'failed-step' : ''}>
        <small>{step.step_id}</small><b>{TOOL_LABELS[step.tool] ?? step.tool}</b><span>{STATUS_LABELS[step.status] ?? step.status}</span>
      </li>)}</ol>
      {!sample.timeline.length && <p className="recovery-empty">没有后续正式步骤。</p>}
      <footer>{sample.terminal_reason ? TERMINAL_REASON_LABELS[sample.terminal_reason] ?? sample.terminal_reason : '终止原因未知'}</footer>
      {!!sample.failures.length && <details><summary>查看未通过原因</summary><ul>{sample.failures.map((failure, i) => <li key={i}>{failure}</li>)}</ul></details>}
    </>}
  </section>
}

export function RecoveryLabPage() {
  const [selectedRun, setSelectedRun] = useState('')
  const [selectedPoint, setSelectedPoint] = useState('')
  const list = useQuery({ queryKey: ['recovery-replays'], queryFn: api.recoveryReplays, refetchInterval: 15000 })
  const runId = list.data?.runs.find(r => r.run_id === selectedRun)?.run_id ?? list.data?.runs[0]?.run_id
  const detail = useQuery({ queryKey: ['recovery-replay', runId], queryFn: () => api.recoveryReplay(runId!), enabled: !!runId, refetchInterval: 15000 })
  const run = detail.data
  const profiles = run?.profiles ?? []
  const points = [...new Map((run?.samples ?? []).map(s => [`${s.origin_sha256}:${s.repetition}`, s])).entries()]
  const pointId = points.find(([key]) => key === selectedPoint)?.[0] ?? points[0]?.[0]
  const samples = run?.samples.filter(s => `${s.origin_sha256}:${s.repetition}` === pointId) ?? []
  return <div className="recovery-lab">
    <PageHeader code="RECOVERY / SEARCH" title="从同一次失败，走向不同结果" subtitle="固定起点，展开探查，再回到真实执行。比较每种策略如何使用剩余预算。" marker="研究候选"/>
    <div className="recovery-toolbar"><Link to="/experiments">← 论文实验室</Link>
      <label>回放实验<select value={runId ?? ''} onChange={e => { setSelectedRun(e.target.value); setSelectedPoint('') }}>
        {!list.data?.runs.length && <option value="">尚无记录</option>}{list.data?.runs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
      </select></label><button onClick={() => { void list.refetch(); if (runId) void detail.refetch() }} aria-label="刷新回放记录"><RefreshCw size={16}/></button>
    </div>
    {list.isPending && <LoadingState label="正在读取回放实验"/>}
    {list.isError && <ErrorState error={list.error} retry={() => void list.refetch()}/>}
    {list.data?.runs.length === 0 && <div className="recovery-empty"><ScanLine size={30}/><h2>尚无同状态回放</h2><p>隔离评测采集失败检查点后，这里会显示真实策略对照。</p></div>}
    {runId && detail.isPending && <LoadingState label="正在读取分支与恢复轨迹"/>}
    {detail.isError && <ErrorState error={detail.error} retry={() => void detail.refetch()}/>}
    {run && <>
      <div className="recovery-provenance"><span>{run.execution_source === 'live_provider' ? '真实模型实验' : '测试或未知来源'}</span><code>实验源码 {run.source_revision?.slice(0, 8) ?? '未知'}</code><span>{run.status === 'COMPLETED' ? '回放已结束' : run.status === 'INVALID' ? '对照证据不完整或无效' : '部分记录'}</span></div>
      {(run.partial || detail.isRefetchError) && <p className="recovery-notice">部分证据尚不可用或刷新失败，保留已读取内容。{run.warnings.join('；')}</p>}
      <div className="recovery-point-picker"><h2>同起点 · 策略对照</h2><label>检查点样本<select value={pointId ?? ''} onChange={e => setSelectedPoint(e.target.value)}>
        {!points.length && <option value="">尚无后续执行</option>}{points.map(([key, sample]) => <option key={key} value={key}>{sample.case_id} · 第 {sample.repetition} 次</option>)}
      </select></label></div>
      {samples[0] ? <div className="recovery-origin"><ScanLine size={21}/><div><b>冻结的失败检查点</b><span>目标、证据、业务数据、剩余故障与前缀预算</span></div><code>{samples[0].origin_sha256.slice(0, 12)}</code></div> : <p className="recovery-empty">尚未取得可比较的回放样本；可能未遇到可恢复失败，或实验仍在执行。</p>}
      {!!samples.length && <div className="recovery-arms">{profiles.map(profile => <ReplayCard key={profile.profile_id} profile={profile} sample={samples.find(s => s.profile_id === profile.profile_id)}/>)}</div>}
      <p className="recovery-footnote">指标仅统计失败点之后，包含全部搜索探查成本。分支读取成功不代表任务完成；同起点对照仍含模型随机性。本页展示对应源码的实验记录，不自动宣布收益或晋升。</p>
    </>}
  </div>
}
