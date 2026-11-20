import { Link } from 'react-router-dom'
import { GitBranch, History } from 'lucide-react'
import { JsonBlock } from './SystemUI'
import type { PGJob, SIMAttempt, SIMProblem } from '../types'

const modes = { none: '无记忆', problems: '仅问题', full: '完整历史' }
const reasons: Record<string, string> = {
  VALIDATION_NON_REGRESSION: '验证未退化', VALIDATION_REGRESSION: '验证退化', COST_REGRESSION: '成本退化',
  PG_NOT_EXPOSED: '指导未覆盖', SAFETY_OR_PROTOCOL_VIOLATION: '安全或协议未通过',
  INCOMPARABLE_TRAINING: '训练条件变化', INCOMPARABLE_VALIDATION: '验证条件变化',
  REPEATED_REJECTED_STRUCTURE: '重复已拒绝结构', INVALID_PROPOSAL_OR_REPORT: '提案或报告无效',
}
const short = (value: string) => value.slice(0, 12)
const refs = (values: string[]) => values.join('、') || '无'

// 只拼接持久化记录；状态和修复结论均由后端提供。
function problemsFor(job: PGJob): SIMProblem[] {
  if (job.evolution?.sim_memory) return job.evolution.sim_memory.problems
  const problems = (job.sim_source?.problems ?? []).map(p => ({ ...p, attempts: [...p.attempts] }))
  for (const row of job.history) {
    if (!row.sim_attempt || !row.sim_problem_id || !row.hypothesis) continue
    let problem = problems.find(p => p.problem_id === row.sim_problem_id)
    if (!problem) {
      problem = { problem_id: row.sim_problem_id, diagnosis: row.hypothesis.problem, failure_mode: row.hypothesis.failure_mode, attempts: [] }
      problems.push(problem)
    }
    if (!problem.attempts.some(a => a.attempt_id === row.sim_attempt!.attempt_id)) problem.attempts.push(row.sim_attempt)
  }
  return problems
}

function Attempt({ attempt, previous, inherited }: { attempt: SIMAttempt; previous?: SIMAttempt; inherited: boolean }) {
  const measured = attempt.observation_status === 'MEASURED'
  const changed = previous && JSON.stringify(previous.environment) !== JSON.stringify(attempt.environment)
  return <li className={`sim-attempt sim-${attempt.status.toLowerCase()}`}>
    <header><span>{inherited ? '继承记录' : '本次实验'} · 第 {attempt.iteration} 轮</span>
      <strong>{attempt.status === 'ACCEPTED' ? '保留修改' : '拒绝修改'} · {reasons[attempt.reason] ?? attempt.reason}</strong></header>
    <p className="sim-change">{attempt.change}</p>
    <div className="sim-outcome-grid">
      <div><small>评测前预测</small><b>{refs(attempt.predicted_fixes)}</b><span>预计从失败变为通过</span></div>
      <div><small>训练复测</small>{measured ? <><b className="pg-pass">已修复 {refs(attempt.fixed)}</b><span>未修复 {refs(attempt.missed)}</span></> : <b>未评测，效果未知</b>}</div>
      <div><small>其他训练退化</small><b className={measured && attempt.regressed.length ? 'pg-fail' : ''}>{measured ? refs(attempt.regressed) : '尚无证据'}</b><span>原通过样本变为失败</span></div>
      <div><small>独立验证</small><b>{attempt.validation_summary.compared_samples ? `改善 ${attempt.validation_summary.improved_samples} · 退化 ${attempt.validation_summary.regressed_samples}` : '尚无可比反馈'}</b><span>仅聚合反馈进入学习记忆</span></div>
    </div>
    <footer><span>模型 <code title={attempt.environment.models_sha256}>{short(attempt.environment.models_sha256)}</code></span>
      <span>数据 <code title={attempt.environment.dataset_sha256}>{short(attempt.environment.dataset_sha256)}</code></span>
      {changed && <b>环境已变化，历史结论需重新验证</b>}</footer>
    <details><summary>查看修改与证据身份</summary><JsonBlock value={{ patch: attempt.patch, parent_sha256: attempt.parent_sha256,
      candidate_sha256: attempt.candidate_sha256, training_sha256: attempt.training_sha256,
      candidate_training_sha256: attempt.candidate_training_sha256, environment: attempt.environment }} /></details>
  </li>
}

export function SIMMemoryPanel({ job }: { job: PGJob }) {
  const problems = problemsFor(job)
  const inheritedIds = new Set(job.sim_source?.problems.flatMap(p => p.attempts.map(a => a.attempt_id)))
  return <section className="sim-memory" aria-label="持续自改进记忆">
    <header className="sim-heading"><div><span><History size={15} />SELF-IMPROVEMENT MEMORY</span><h2>问题会延续，修改有回声</h2>
      <p>优化器读取的条件：<b>{modes[job.frozen.sim_memory_mode ?? 'full']}</b> · 下方保留完整审计历史。</p></div>
      <div className="sim-source"><GitBranch size={17} />{job.frozen.sim_source_job_id ? <Link to={`/experiments/pg?job=${job.frozen.sim_source_job_id}`}>查看继承来源 ↗</Link> : <span>从原始种子图开始</span>}
        <small>{job.frozen.sim_source_sha256 ? `记忆 ${short(job.frozen.sim_source_sha256)}` : '尚未继承历史'}</small></div></header>
    {!job.evolution?.sim_memory && <p className="sim-partial" role="status">记忆尚未最终封存，以下仅展示已持久化的来源与已完成轮次。</p>}
    {!problems.length ? <div className="pg-empty"><History size={24} /><div><strong>尚无可继承的问题历史</strong><p>合法候选经过评测后，会在这里留下修改、预测与实际反馈。</p></div></div>
      : problems.map(problem => <article className="sim-problem" key={problem.problem_id}>
        <header><code>{problem.problem_id}</code><h3>{problem.diagnosis}</h3></header>
        <ol>{problem.attempts.map((attempt, index) => <Attempt key={attempt.attempt_id} attempt={attempt}
          previous={problem.attempts[index - 1]} inherited={inheritedIds.has(attempt.attempt_id)} />)}</ol>
      </article>)}
    <p className="sim-note">训练复测解释修改是否符合预期；验证决定是否保留。这里的结果不能替代独立留出测试，也不代表已经证明持续自改进收益。</p>
  </section>
}
