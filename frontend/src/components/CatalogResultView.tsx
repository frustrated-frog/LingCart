import { AlertTriangle, Boxes, CheckCircle2, Clock3, Database, GitMerge, Layers3, SearchCheck } from 'lucide-react'
import type { CatalogResultArtifact, CatalogResultProduct, ConstraintPlanResultArtifact, ResultArtifact } from '../types'

function isCatalogArtifact(artifact: ResultArtifact): artifact is CatalogResultArtifact {
  return artifact.artifact_type === 'catalog_query_result'
    && Array.isArray(artifact.products)
    && Array.isArray(artifact.groups)
    && Array.isArray(artifact.member_ids)
    && typeof artifact.delivery === 'object'
}

function isConstraintArtifact(artifact: ResultArtifact): artifact is ConstraintPlanResultArtifact {
  return artifact.artifact_type === 'constraint_plan_result'
    && Array.isArray(artifact.domains)
    && Array.isArray(artifact.solutions)
    && typeof artifact.search === 'object'
}

function price(value: number, currency: string): string {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency', currency, minimumFractionDigits: 2,
  }).format(value)
}

function fieldLabel(field: { core?: string | null; concept_id?: string | null }): string {
  const labels: Record<string, string> = {
    product_id: '商品编号', name: '商品名称', category: '类目', price: '价格', stock: '库存', summary: '商品摘要',
  }
  const identity = field.core ?? field.concept_id ?? '字段'
  return labels[identity] ?? identity
}

function metricLabel(
  alias: string,
  definitions: CatalogResultArtifact['query_spec']['metrics'] = [],
): string {
  const definition = definitions.find((item) => item.alias === alias)
  if (!definition) return alias
  const functions = { COUNT: '数量', MIN: '最低', MAX: '最高', AVG: '平均', SUM: '合计' }
  return `${functions[definition.function]}${definition.field ? fieldLabel(definition.field) : ''}`
}

function ProductTable({ products }: { products: CatalogResultProduct[] }) {
  return <div className="catalog-table-wrap"><table className="catalog-result-table">
    <thead><tr><th>商品</th><th>价格</th><th>库存</th><th>已验证目录信息</th></tr></thead>
    <tbody>{products.map((product) => <tr key={product.product_id}>
      <td><b>{product.name}</b><code>{product.product_id}</code></td>
      <td className="catalog-price">{price(product.price, product.currency)}</td>
      <td>{product.stock > 0 ? `${product.stock} 件` : '暂无库存'}</td>
      <td>{product.summary}</td>
    </tr>)}</tbody>
  </table></div>
}

function verificationStatus(artifact: CatalogResultArtifact) {
  if (artifact.verification?.status) return artifact.verification.status
  if (artifact.verification?.goal?.verdict === 'SATISFIED'
      && artifact.verification?.delivery?.verdict === 'SATISFIED') return 'VERIFIED'
  return 'PENDING'
}

function CatalogArtifactView({ artifact }: { artifact: CatalogResultArtifact }) {
  const verification = verificationStatus(artifact)
  if (verification !== 'VERIFIED') {
    const rejected = verification === 'REJECTED'
    const label = rejected ? '目录结果未通过核验' : verification === 'PARTIAL' ? '目录结果仅部分核验' : '目录结果尚未核验'
    return <section className={`catalog-result catalog-result-${verification.toLowerCase()}`} aria-label={label}>
      <header className="catalog-result-head">
        <div><span>CATALOG EVIDENCE</span><h3>{label}</h3></div>
        <div className="catalog-result-proof"><AlertTriangle size={14} /><span>{verification}</span></div>
      </header>
      <div className="catalog-verification-warning" role="alert">
        <AlertTriangle size={16} />
        <div><strong>未将该产物作为可信商品结果交付</strong>
          <p>{artifact.verification?.error?.message ?? '目标或交付核验尚未全部完成，商品行仅保留为运行诊断证据。'}</p>
          {artifact.verification?.error?.code && <code>{artifact.verification.error.code}</code>}
        </div>
      </div>
      <footer><code>{artifact.result_set_id}</code><span>{artifact.data_source} · 诊断证据</span></footer>
    </section>
  }
  const productById = new Map(artifact.products.map((product) => [product.product_id, product]))
  const groups = artifact.groups.length > 0
    ? artifact.groups.map((group) => ({ ...group, products: group.product_ids
      .map((id) => productById.get(id)).filter((product): product is CatalogResultProduct => Boolean(product)) }))
    : [{ key: '查询结果', product_ids: artifact.member_ids, count: artifact.products.length, products: artifact.products }]
  const ordering = artifact.query_spec.ordering.map((item) =>
    `${fieldLabel(item.field)}${item.direction === 'ASC' ? '升序' : '降序'}`,
  ).join('、')
  const metricEntries = Object.entries(artifact.metrics)
  const populationComplete = artifact.delivery.population_complete
    ?? (artifact.delivery.requested_complete && artifact.delivery.complete)
  const deliveryLabel = artifact.delivery.mode === 'PAGE'
    ? `第 ${artifact.delivery.page ?? 1} 页`
    : artifact.delivery.mode === 'TOP_K'
      ? '前 K 项'
      : populationComplete
        ? '完整'
        : '目标结果'
  const deliveryDetail = artifact.delivery.has_more
    ? `仍有后续${artifact.delivery.next_page ? ` · 下一页 ${artifact.delivery.next_page}` : ''}`
    : ordering || '已完成所选范围'

  return <section className="catalog-result" aria-label="结构化商品目录结果">
    <header className="catalog-result-head">
      <div><span>CATALOG RESULT</span><h3>目录结果集</h3></div>
      <div className="catalog-result-proof"><CheckCircle2 size={14} /><span>目标与交付已核验</span></div>
    </header>
    <div className="catalog-result-summary">
      <div><Database size={15} /><span>结果规模</span><strong>{artifact.delivered_count}</strong><small>匹配 {artifact.matched_count} 款</small></div>
      <div><Layers3 size={15} /><span>结果组织</span><strong>{groups.length}</strong><small>{artifact.groups.length ? '按请求分组' : '未分组'}</small></div>
      <div><Clock3 size={15} /><span>交付状态</span><strong>{deliveryLabel}</strong><small>{deliveryDetail}</small></div>
    </div>
    {metricEntries.length > 0 && <dl className="catalog-metrics">{metricEntries.map(([name, value]) =>
      <div key={name}><dt>{metricLabel(name, artifact.query_spec.metrics)}</dt><dd>{String(value ?? '无可用值')}</dd></div>)}</dl>}
    <div className="catalog-groups">{groups.map((group, index) => <section className="catalog-group" key={group.key}>
      <div className="catalog-group-head"><div><span>{String(index + 1).padStart(2, '0')} / GROUP</span><h4>{group.key}</h4></div><b>{group.products.length} 款</b></div>
      <ProductTable products={group.products} />
    </section>)}</div>
    <footer><code>{artifact.result_set_id}</code><span>{artifact.data_source} · {artifact.catalog_snapshot.version ?? '目录快照'}</span></footer>
  </section>
}

export function CatalogResultViews({ artifacts = [] }: { artifacts?: ResultArtifact[] }) {
  const catalogArtifacts = artifacts.filter(isCatalogArtifact)
  if (catalogArtifacts.length === 0) return null
  return <>{catalogArtifacts.map((artifact) => <CatalogArtifactView artifact={artifact} key={artifact.result_set_id} />)}</>
}

function constraintValue(value: unknown): string {
  if (typeof value === 'number') return new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 }).format(value)
  if (Array.isArray(value)) return value.join('、')
  return value == null ? '未解析' : String(value)
}

function constraintCheckLabel(check: ConstraintPlanResultArtifact['solutions'][number]['constraint_checks'][number]) {
  const operators: Record<string, string> = {
    EQ: '=', NE: '≠', LT: '<', LTE: '≤', GT: '>', GTE: '≥',
  }
  const operator = operators[check.operator ?? ''] ?? check.operator ?? '—'
  if (check.kind === 'COMPARISON') {
    return `${constraintValue(check.left)} ${operator} ${constraintValue(check.right)}`
  }
  if (check.kind === 'ALL_DIFFERENT') return `属性 ${check.attribute ?? 'candidate_id'} 均不相同`
  return `${constraintValue(check.observed)} ${operator} ${constraintValue(check.expected)}`
}

function ConstraintArtifactView({ artifact }: { artifact: ConstraintPlanResultArtifact }) {
  const verified = artifact.verification?.status === 'VERIFIED'
  if (!verified) {
    const label = artifact.verification?.status === 'REJECTED' ? '约束结果未通过核验' : '约束结果尚未核验'
    return <section className="catalog-result catalog-result-rejected" aria-label={label}>
      <header className="catalog-result-head">
        <div><span>CONSTRAINT EVIDENCE</span><h3>{label}</h3></div>
        <div className="catalog-result-proof"><AlertTriangle size={14} /><span>{artifact.verification?.status ?? 'PENDING'}</span></div>
      </header>
      <div className="catalog-verification-warning" role="alert"><AlertTriangle size={16} /><div>
        <strong>未将候选组合作为可信方案交付</strong>
        <p>{artifact.verification?.error?.message ?? '全局目标核验尚未完成，候选数据只保留为运行诊断证据。'}</p>
        {artifact.verification?.error?.code && <code>{artifact.verification.error.code}</code>}
      </div></div>
      <footer><code>{artifact.solution_set_id}</code><span>诊断证据</span></footer>
    </section>
  }

  const infeasible = artifact.feasibility === 'INFEASIBLE'
  const optimalityStatus = artifact.optimality?.status
    ?? (artifact.plan_spec.optimization
      ? artifact.search.complete ? 'PROVEN_OPTIMAL' : 'BEST_KNOWN'
      : 'NOT_REQUESTED')
  const optimalityLabels = {
    NOT_REQUESTED: ['未请求', '仅验证硬约束'],
    NO_FEASIBLE_SOLUTION: ['不适用', '已证明没有可行方案'],
    BEST_KNOWN: ['未证明', '当前已知最优'],
    PROVEN_OPTIMAL: ['已证明', '全局最优'],
  } as const
  const optimalityLabel = optimalityLabels[optimalityStatus]
  const proofLabel = infeasible
    ? '无解结论已核验'
    : optimalityStatus === 'BEST_KNOWN'
      ? '硬约束已核验 · 最优性未证明'
      : optimalityStatus === 'PROVEN_OPTIMAL'
        ? '硬约束与最优性已核验'
        : '全部硬约束已核验'
  const goalLabels = new Map(artifact.plan_spec.goals.map((goal) => [goal.goal_id, goal.source_quote]))
  return <section className="catalog-result constraint-result" aria-label="已核验约束方案">
    <header className="catalog-result-head">
      <div><span>GLOBAL GOAL PROOF</span><h3>{infeasible ? '当前约束下没有可行方案' : optimalityStatus === 'BEST_KNOWN' ? '当前已知约束方案' : '全局约束方案'}</h3></div>
      <div className="catalog-result-proof"><CheckCircle2 size={14} /><span>{proofLabel}</span></div>
    </header>
    <div className="catalog-result-summary">
      <div><GitMerge size={15} /><span>目标节点</span><strong>{artifact.plan_spec.goals.length}</strong><small>{artifact.domains.map((domain) => domain.provider).join(' · ')}</small></div>
      <div><Boxes size={15} /><span>交付方案</span><strong>{artifact.solutions.length}</strong><small>{artifact.search.feasible_count_exact ? `共 ${artifact.search.feasible_count} 组可行` : `至少 ${artifact.search.feasible_count} 组可行`}</small></div>
      <div><SearchCheck size={15} /><span>求解边界</span><strong>{artifact.search.complete ? '完整' : '有界'}</strong><small>检查 {artifact.search.nodes_examined} 个完整赋值</small></div>
      <div><CheckCircle2 size={15} /><span>最优性</span><strong>{optimalityLabel[0]}</strong><small>{optimalityLabel[1]}{artifact.optimality?.objective_value != null ? ` · 目标值 ${constraintValue(artifact.optimality.objective_value)}` : ''}</small></div>
    </div>
    <div className="constraint-domain-rail" aria-label="候选数据域">
      {artifact.domains.map((domain) => <div key={domain.goal_id}>
        <span>{goalLabels.get(domain.goal_id) ?? domain.goal_id}</span><strong>{domain.candidate_count}</strong><small>个已核验候选 · {domain.provider}</small>
      </div>)}
    </div>
    {infeasible ? <div className="constraint-empty">
      <SearchCheck size={22} /><div><strong>搜索已完成，没有同时满足全部硬约束的组合</strong>
        <p>系统没有放宽预算、删除目标或把单项结果伪装成完整方案。</p></div>
    </div> : <div className="constraint-solutions">{artifact.solutions.map((solution, solutionIndex) =>
      <article className="constraint-solution" key={solution.solution_id}>
        <div className="constraint-solution-index"><span>SOLUTION</span><strong>{String(solutionIndex + 1).padStart(2, '0')}</strong><code>{solution.solution_id.slice(-8)}</code></div>
        <div className="constraint-goal-lanes">{solution.selections.map((selection) => <section key={selection.goal_id}>
          <div><span>{selection.goal_id}</span><small>{selection.source_quote} · {selection.quantity} 件</small></div>
          {selection.candidates.map((candidate, index) => <div className="constraint-candidate" key={`${candidate.candidate_id}-${index}`}>
            <div><strong>{candidate.label}</strong><code>{candidate.candidate_id}</code></div>
            {typeof candidate.attributes.price === 'number'
              ? <b>{price(candidate.attributes.price, String(candidate.evidence.currency ?? 'CNY'))}</b>
              : <b>{constraintValue(candidate.attributes.price)}</b>}
            <p>{String(candidate.evidence.summary ?? '已由数据域提供身份与属性证据')}</p>
          </div>)}
        </section>)}</div>
        <div className="constraint-proof-list">
          {solution.constraint_checks.map((check) => <div key={check.requirement_id}>
            <CheckCircle2 size={13} /><span>{check.requirement_id}</span><strong>{constraintCheckLabel(check)}</strong>
          </div>)}
          {(solution.resource_checks ?? []).map((check) => <div key={`resource-${check.resource_id}`}>
            <CheckCircle2 size={13} /><span>RESOURCE · {check.resource_id}</span><strong>占用 {check.used} / 容量 {check.capacity} · 剩余 {check.remaining}</strong>
          </div>)}
        </div>
      </article>)}</div>}
    <footer><code>{artifact.solution_set_id}</code><span>{artifact.selection_policy === 'EXPLICIT_OPTIMIZATION'
      ? optimalityStatus === 'PROVEN_OPTIMAL' ? '显式优化 · 已证明全局最优' : '显式优化 · 当前已知最优'
      : '确定性差异覆盖 · 非个人偏好排序'}</span></footer>
  </section>
}

export function ResultArtifactViews({ artifacts = [] }: { artifacts?: ResultArtifact[] }) {
  const catalogArtifacts = artifacts.filter(isCatalogArtifact)
  const constraintArtifacts = artifacts.filter(isConstraintArtifact)
  if (catalogArtifacts.length === 0 && constraintArtifacts.length === 0) return null
  return <>
    {constraintArtifacts.map((artifact) => <ConstraintArtifactView artifact={artifact} key={artifact.solution_set_id} />)}
    {catalogArtifacts.map((artifact) => <CatalogArtifactView artifact={artifact} key={artifact.result_set_id} />)}
  </>
}
