import { CheckCircle2, Database, Layers3 } from 'lucide-react'
import type { CatalogResultArtifact, CatalogResultProduct, ResultArtifact } from '../types'

function isCatalogArtifact(artifact: ResultArtifact): artifact is CatalogResultArtifact {
  return artifact.artifact_type === 'catalog_query_result'
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

function CatalogArtifactView({ artifact }: { artifact: CatalogResultArtifact }) {
  const productById = new Map(artifact.products.map((product) => [product.product_id, product]))
  const groups = artifact.groups.length > 0
    ? artifact.groups.map((group) => ({ ...group, products: group.product_ids
      .map((id) => productById.get(id)).filter((product): product is CatalogResultProduct => Boolean(product)) }))
    : [{ key: '查询结果', product_ids: artifact.member_ids, count: artifact.products.length, products: artifact.products }]
  const ordering = artifact.query_spec.ordering.map((item) =>
    `${fieldLabel(item.field)}${item.direction === 'ASC' ? '升序' : '降序'}`,
  ).join('、')
  const metricEntries = Object.entries(artifact.metrics)
  const deliveryLabel = artifact.delivery.mode === 'PAGE'
    ? `第 ${artifact.delivery.page} 页`
    : artifact.delivery.mode === 'TOP_K'
      ? '前 K 项'
      : artifact.delivery.population_complete
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
      <div><CheckCircle2 size={15} /><span>交付状态</span><strong>{deliveryLabel}</strong><small>{deliveryDetail}</small></div>
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
