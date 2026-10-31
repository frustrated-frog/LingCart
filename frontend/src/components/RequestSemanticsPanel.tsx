import type { ExecutionBoundary, ExecutionRestriction, RequestSemantics, RunDetail, SemanticRevision } from '../types'
import { JsonBlock, SectionHeading } from './SystemUI'

const restrictionLabels: Record<ExecutionRestriction, string> = {
  READ_ONLY: '仅允许读取', NO_PAYMENT: '禁止支付', NO_REFUND: '禁止退款', NO_SUBMISSION: '禁止最终提交',
}
const operationLabels = { PRESERVE: '保留', REPLACE: '替换', RETRACT: '撤销', ADD: '新增' }
const kindLabels = { OUTCOME: '任务结果', CONDITION: '硬条件', PROHIBITION: '禁止事项',
  EXECUTION_BOUNDARY: '执行要求', USER_FACT: '用户陈述', SOFT_PREFERENCE: '偏好', UNKNOWN: '待澄清' }

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function semantics(value: unknown): RequestSemantics | undefined {
  const data = record(value)
  return data.version === 'request-semantics/v1' && Array.isArray(data.requirements) && Array.isArray(data.revisions)
    ? data as unknown as RequestSemantics : undefined
}

export function RequestSemanticsPanel({ detail }: { detail: RunDetail }) {
  const adaptive = record(detail.checkpoint.adaptive)
  const accepted = semantics(detail.result.request_semantics ?? adaptive.request_semantics)
  const revisionOnly = detail.result.events?.some((event) => event.type === 'request_semantics_revised') ?? false
  const contract = accepted ?? semantics(record(detail.checkpoint.semantic_extraction).contract)
  const revision = (detail.result.semantic_revision ?? adaptive.semantic_revision) as SemanticRevision | undefined
  const currentRevision = revision?.current_sha256 === contract?.sha256 ? revision : undefined
  const boundaries = (detail.result.execution_boundaries ?? detail.checkpoint.execution_boundaries ?? []) as ExecutionBoundary[]
  const outcomes = contract?.requirements.filter(({ requirement }) => requirement.kind === 'OUTCOME') ?? []
  const contextRows = contract?.requirements.filter(({ requirement }) => !['OUTCOME', 'CONDITION'].includes(requirement.kind)) ?? []
  const label = (identity: string) => contract?.requirements.find(({ requirement }) => requirement.requirement_id === identity)?.requirement.description ?? identity

  return <section aria-label="需求与修订" className="semantic-panel">
    <SectionHeading index="S" title="需求与修订" />
    {!contract ? <div className="empty-inline">本次运行尚无独立需求契约；既有 RequestIR 投影不代表已完成独立语义匹配。</div> : <>
      <div className="semantic-header">
        <div><span className="semantic-kicker">原文 → 独立提取 → 执行目标</span><h3>{revisionOnly ? '已保存需求修订 · 未启动执行' : accepted ? '已匹配执行目标' : '已提取 · 待匹配执行目标'}</h3>
          <p>{accepted ? '以下为本次接受的需求。任务是否完成，仍以最终证据核验为准。' : '以下为提取候选，尚不能据此执行或宣称完成。'}</p></div>
        <code title={contract.sha256}>{contract.version}<br />{contract.sha256.slice(0, 16)}</code>
      </div>
      <div className="semantic-objectives">
        {outcomes.map(({ requirement, source }) => {
          const constraints = contract.requirements.filter(({ requirement: row }) => row.kind === 'CONDITION' && row.targets.includes(requirement.requirement_id))
          const changed = currentRevision?.changed.includes(requirement.requirement_id)
          const preserved = currentRevision?.preserved.includes(requirement.requirement_id)
          return <article className={`semantic-objective${changed ? ' is-revised' : ''}`} key={requirement.requirement_id}>
            <div className="semantic-objective-heading"><code>{requirement.requirement_id}</code>
              <span>{changed ? '已修订 · 重新核验' : preserved ? '需求保留' : '任务结果'}</span></div>
            <h4>{requirement.description}</h4>
            <div className="semantic-quantity">目标数量 <b>{requirement.outcome?.quantity}</b></div>
            <blockquote>{source.text}</blockquote>
            {constraints.length ? <ul className="semantic-constraints">{constraints.map(({ requirement: condition }) =>
              <li key={condition.requirement_id}><span>{condition.description}</span>
                <small>{condition.targets.length > 1 ? `共同适用：${condition.targets.map(label).join('、')}` : '仅适用于此结果'}</small></li>,
            )}</ul> : <p className="semantic-note">契约未记录该结果的商品硬条件。</p>}
          </article>
        })}
      </div>
      {contextRows.length > 0 && <div className="semantic-context">{contextRows.map(({ requirement }) =>
        <div key={requirement.requirement_id} className={requirement.kind === 'UNKNOWN' ? 'is-uncertain' : ''}>
          <b>{kindLabels[requirement.kind]}</b><span>{requirement.description}</span><q>{requirement.source_quote}</q>
        </div>,
      )}</div>}
      {contract.previous_sha256 && <details className="semantic-revisions" open>
        <summary>本轮修订记录 · {accepted ? '已接受' : '待匹配'}</summary>
        <div className="table-wrap"><table><thead><tr><th>操作</th><th>需求</th><th>修改依据</th></tr></thead>
          <tbody>{contract.revisions.map((change, index) => <tr key={`${change.previous_id ?? change.requirement_id}-${index}`}>
            <td><span className={`semantic-operation operation-${change.operation.toLowerCase()}`}>{operationLabels[change.operation]}</span></td>
            <td>{label(change.requirement_id ?? change.previous_id ?? '—')}</td>
            <td>{change.source_quote ?? (change.operation === 'PRESERVE' ? '沿用此前已接受的要求' : '见需求原文')}</td>
          </tr>)}</tbody></table></div>
      </details>}
      <details><summary>独立契约与修订原始记录</summary><JsonBlock value={{ contract, revision: currentRevision ?? null }} /></details>
    </>}
    <div className="semantic-boundaries"><h4>当前执行限制</h4>
      {boundaries.length ? <ul>{boundaries.map((boundary, index) => <li key={`${boundary.restriction}-${index}`}>
        <b>{restrictionLabels[boundary.restriction]}</b><q>{boundary.source.text}</q>
      </li>)}</ul> : <p>本次运行未记录额外执行限制。支付、退款仍须明确确认。</p>}
    </div>
  </section>
}
