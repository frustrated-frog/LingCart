import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, GitBranch, RefreshCw, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, TOOL_LABELS, STATUS_LABELS } from '../components/SystemUI'
import type { RSICandidate, RSISampleSide } from '../types'
import '../styles/rsi-lab.css'

const reasons: Record<string, string> = {
  ACCEPTED: '通过选择条件', RSI_TARGET_NOT_EXERCISED: '目标模块未实际介入', INVALID_PROPOSAL: '提案协议无效',
  RSI_MODULE_SCOPE: '修改超出单模块范围', COST_OR_NOISE: '成本或收益不足', STABILITY_REGRESSION: '重复稳定性退化',
  BELOW_HISTORICAL_FLOOR: '低于历史成绩下限', RSI_CRITIC_REJECTED: '可复用性审查未通过',
  SAFETY_OR_PROTOCOL: '安全或协议检查未通过', COMPLETED: '已完成', FAILED: '未完成', PARTIAL: '部分产物',
}
const failures: Record<string, string> = { empty_results: '结果为空', tool_failure: '工具失败', decision_rejected: '决策被拒绝', verification_rejected: '结果核验被拒绝' }
const fields: Record<string, string> = { mode: '分析调用方式', trigger: '触发时机', instruction: '决策指令' }
const values: Record<string, string> = { inline: '在原决策中分析', deliberate: '增加独立分析调用', initial: '首次理解目标', after_goal: '目标建立后', every_decision: '每次决策', all_failures: '所有失败', tool_failures: '工具失败', empty_results: '结果为空', decision_rejections: '决策被拒绝' }
const armNames: Record<string, string> = { baseline: '原始 Agent', fixed: '固定策略', schedule_control: '相同调用策略', evolved: '最终保留', seed: '初始策略', retained: '最终保留' }
const pretty = (value: string) => values[value] ?? value
const reason = (value: string) => reasons[value] ?? value

function Trajectory({ side, title, sha }: { side?: RSISampleSide | null; title: string; sha: string | null }) {
  return <section className="rsi-trajectory" aria-label={title}>
    <header><span>{title}</span><code>{sha?.slice(0, 8) ?? '未生成'}</code></header>
    {!side ? <p className="rsi-empty">尚无该版本的验证轨迹。候选可能在执行前被拒绝。</p> : <>
      <div className={`rsi-outcome ${side.passed ? 'positive' : 'negative'}`}><b>{side.passed ? '任务通过' : '任务未通过'}</b><span>{side.tokens?.toLocaleString() ?? '未知'} Token</span></div>
      <p className="rsi-caption">{side.source === 'live_provider' ? '真实模型记录' : '测试或未知来源'} · 核验完成目标 {side.recovery.verified_objectives}</p>
      <ol className="rsi-timeline">{side.recovery.timeline.map((step, index) => <li key={`${step.step_id}-${index}`} className={step.failure_kind ? 'has-failure' : ''}>
        <span className="rsi-node" /><div><small>{step.step_id} · 事件 {step.sequence ?? '未知'}</small><b>{TOOL_LABELS[step.tool] ?? step.tool}</b>
          <span>{step.failure_kind ? failures[step.failure_kind] ?? step.failure_kind : STATUS_LABELS[step.status] ?? step.status}</span>
          <em>新增商品记录 {step.new_records}</em></div>
      </li>)}</ol>
      {!side.recovery.timeline.length && <p className="rsi-empty">尚无工具步骤，不能推断恢复有效。</p>}
      <div className="rsi-episodes">{side.recovery.episodes.map((episode, index) => <details key={index}>
        <summary>{failures[episode.failure_kind] ?? episode.failure_kind} · {episode.observation_improved ? '后续获得新商品记录' : '尚无新增商品记录'}</summary>
        {episode.error_code && <code>{episode.error_code}</code>}
        {episode.recovery_actions.map((action, i) => <p key={i}>{TOOL_LABELS[action.tool] ?? action.tool}：{action.same_arguments === true ? '重复原参数' : action.same_arguments === false ? '更换行动或参数' : '参数对比未知'}，新增 {action.new_records} 条观察</p>)}
        {!episode.recovery_actions.length && <p>没有后续行动记录。</p>}
      </details>)}</div>
    </>}
  </section>
}

function CandidateCompare({ candidate }: { candidate: RSICandidate }) {
  const [pairId, setPairId] = useState('')
  const pair = candidate.samples.find(p => `${p.case_id}:${p.repetition}` === pairId) ?? candidate.samples[0]
  return <>
    <div className="rsi-comparison-heading"><h2>同任务 · 两个执行版本</h2><label>验证样本 <select value={pair ? `${pair.case_id}:${pair.repetition}` : ''} onChange={e => setPairId(e.target.value)}>
      {!candidate.samples.length && <option value="">尚无验证样本</option>}
      {candidate.samples.map(p => <option key={`${p.case_id}:${p.repetition}`} value={`${p.case_id}:${p.repetition}`}>{p.case_id} · 第 {p.repetition} 次</option>)}
    </select></label></div>
    <div className="rsi-compare-grid">
      <Trajectory side={pair?.parent} title="父版本" sha={candidate.parent_sha256} />
      <section className="rsi-delta" aria-label="修改与选择依据"><span className="rsi-kicker">变更假设</span><h3>{candidate.selected ? '保留为下一代' : reason(candidate.status)}</h3>
        <p>{candidate.hypothesis || '提案未通过协议校验，没有可用的结构化修改假设。'}</p>
        {candidate.changes.map(change => <article key={change.field}><b>{fields[change.field.split('.')[1]] ?? change.field}</b><div className="rsi-before">{pretty(change.before)}</div><ArrowRight size={14} aria-hidden="true"/><div className="rsi-after">{pretty(change.after)}</div></article>)}
        {!candidate.changes.length && <p className="rsi-caption">没有可审查的程序修改。</p>}
        <footer><ShieldCheck size={17}/><span>原始目标、工具权限与最终核验保持固定。</span></footer>
      </section>
      <Trajectory side={pair?.candidate} title="候选版本" sha={candidate.program_sha256} />
    </div>
    <p className="rsi-caption">新增记录仅统计后续三步首次出现的商品，不覆盖其他信息增益。观察改善不等于任务完成，也不证明修改的因果收益。</p>
  </>
}

export function RSILabPage() {
  const list = useQuery({ queryKey: ['rrsi-runs'], queryFn: api.rrsiRuns, refetchInterval: 15000 })
  const [selected, setSelected] = useState('')
  const [candidateIndex, setCandidateIndex] = useState(0)
  const runId = list.data?.runs.some(r => r.run_id === selected) ? selected : list.data?.runs[0]?.run_id
  const detail = useQuery({ queryKey: ['rrsi-run', runId], queryFn: () => api.rrsiRun(runId!), enabled: !!runId, refetchInterval: 15000 })
  const run = detail.data
  const candidate = run?.candidates[candidateIndex] ?? run?.candidates[0]
  return <div className="rsi-lab">
    <PageHeader code="RSI / RECOVERY" title="看见 Agent 如何改变" subtitle="从一次失败，到一个可验证的新版本。所有轨迹来自保存的实验产物。" marker="研究候选" />
    <div className="rsi-toolbar"><Link to="/experiments">← 论文实验室</Link><label>实验记录 <select aria-label="实验记录" value={runId ?? ''} onChange={e => { setSelected(e.target.value); setCandidateIndex(0) }}>
      {!list.data?.runs.length && <option value="">尚无实验记录</option>}{list.data?.runs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
    </select></label><button onClick={() => { void list.refetch(); void detail.refetch() }} disabled={!runId}><RefreshCw size={14}/>刷新证据</button></div>
    {list.isPending && <LoadingState label="正在读取实验产物"/>}
    {list.isError && <ErrorState error={list.error} retry={() => void list.refetch()}/>}
    {list.data?.runs.length === 0 && <div className="rsi-empty"><GitBranch size={30}/><h2>尚无演化证据</h2><p>通过研究评测入口运行一次实验后，这里会显示真实版本与任务对照。</p></div>}
    {runId && detail.isPending && <LoadingState label="正在读取版本与轨迹"/>}
    {detail.isError && <ErrorState error={detail.error} retry={() => void detail.refetch()}/>}
    {run && <>
      <div className="rsi-provenance"><span>{run.target_module === 'recovery' ? '失败恢复' : '规划'} · {reason(run.status)}</span><code>源码 {run.source_revision?.slice(0, 8) ?? '未知'}</code><span>{run.execution_source === 'live_provider' ? '真实模型实验' : '非真实模型或未知来源'}</span></div>
      {run.historical && <p className="rsi-notice">历史版本记录：实验源码提交与当前不同，不能把这些数字视为当前版本的效果。</p>}
      {(run.partial || detail.isRefetchError) && <p className="rsi-notice">部分证据尚不可用或刷新失败，保留已读取内容。{run.warnings.join('；')}</p>}
      <div className="rsi-generations" aria-label="候选版本">{run.candidates.map((c, i) => <button key={`${c.round}-${c.candidate}`} aria-pressed={candidate === c} onClick={() => setCandidateIndex(i)}>
        <GitBranch size={20}/><span><b>第 {c.round + 1} 轮 · 候选 {c.candidate + 1}</b><small>{c.selected ? '已保留' : reason(c.status)}</small></span><ArrowRight size={18}/>
      </button>)}</div>
      {candidate ? <CandidateCompare key={`${runId}-${candidate.round}-${candidate.candidate}`} candidate={candidate}/> : <p className="rsi-empty">尚无候选提案，已有记录仍可保留检查。</p>}
      <section className="rsi-holdout"><div><span className="rsi-kicker">独立留出 / 同预算对照</span><h2>{!run.comparison ? '尚未完成留出对照' : run.comparison.status !== 'COMPLETED' ? '对照证据无效' : run.comparison.observed_improvement ? '观察到改善，等待复核' : '尚未满足收益条件'}</h2><p>一次成功不代表稳定提升；候选不会自动进入产品。</p>{run.comparison?.identical_arms?.map((arms, i) => <p className="rsi-notice" key={i}>{arms.map(a => armNames[a] ?? a).join("、")}使用同一程序；这些运行的差异不能归因于策略演化。</p>)}</div>
        {run.comparison?.measurements && <table><thead><tr><th>对照版本</th><th>任务成功</th><th>重复稳定性</th><th>Token</th></tr></thead><tbody>{Object.entries(run.comparison.measurements).map(([name, m]) => <tr key={name}><td>{armNames[name] ?? name}</td><td>{(m.score*100).toFixed(0)}%</td><td>{(m.stable*100).toFixed(0)}%</td><td>{m.tokens.toLocaleString()}</td></tr>)}</tbody></table>}
      </section>
    </>}
  </div>
}
