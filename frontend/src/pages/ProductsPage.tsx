import { useQuery } from '@tanstack/react-query'
import { GitCompareArrows, Search, X } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, SectionHeading } from '../components/SystemUI'

export function ProductsPage() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const [budget, setBudget] = useState(1000)
  const [availableOnly, setAvailableOnly] = useState(true)
  const [compare, setCompare] = useState<string[]>([])
  const products = useQuery({ queryKey: ['products', search, category, budget, availableOnly],
    queryFn: () => api.products({ query: search, category: category || undefined, max_price: budget || undefined, available_only: availableOnly }) })
  if (products.isLoading) return <LoadingState label="正在读取商品目录" />
  if (products.isError) return <ErrorState error={products.error} retry={() => void products.refetch()} />
  const items = products.data?.products ?? []
  const selected = items.filter((item) => compare.includes(item.product_id))
  const toggle = (id: string) => setCompare((current) => current.includes(id) ? current.filter((item) => item !== id) : current.length < 3 ? [...current, id] : current)

  return <>
    <PageHeader code="PRODUCT SANDBOX / 11" title="商品发现" subtitle="筛选 60 条真实演示记录，比较取舍，再把明确目标交给 Agent。" marker="DEMO CATALOG" />
    <div className="filter-row product-filters">
      <label>搜索商品<div className="input-with-icon"><Search size={14} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="商品名、编号或参数" /></div></label>
      <label>品类<select value={category} onChange={(e) => setCategory(e.target.value)}><option value="">全部</option><option>耳机</option><option>键盘</option><option>扩展坞</option></select></label>
      <label>预算上限<input type="number" min="0" step="100" value={budget} onChange={(e) => setBudget(Number(e.target.value))} /></label>
      <label className="check-filter"><input type="checkbox" checked={availableOnly} onChange={(e) => setAvailableOnly(e.target.checked)} />只看有货</label>
    </div>
    <div className="catalog-note">找到 <b>{products.data?.total ?? 0}</b> 款商品 · 演示品牌、价格与规格，不是实时行情</div>
    {selected.length > 0 && <section className="comparison-dock"><div><span>COMPARE / {selected.length} OF 3</span><strong>{selected.map((item) => item.name).join(' · ')}</strong></div><button onClick={() => setCompare([])}><X size={14} />清空</button>{selected.length >= 2 && <button className="primary" onClick={() => navigate(`/sandbox?prompt=${encodeURIComponent(`对比商品 ${selected.map((item) => item.product_id).join(' 和 ')}`)}`)}><GitCompareArrows size={14} />交给 Agent 对比</button>}</section>}
    <SectionHeading index="01" title="商品目录" note={`${items.length} VISIBLE RECORDS`} />
    <div className="product-grid">{items.map((product) => <article className={`product-card${compare.includes(product.product_id) ? ' selected' : ''}`} key={product.product_id}>
      <button className="product-visual" aria-label={`选择 ${product.name} 对比`} onClick={() => toggle(product.product_id)}><span>{product.category}</span><b>{product.product_id.slice(1, 3)}</b><i>{compare.includes(product.product_id) ? '已选择' : '加入对比'}</i></button>
      <div className="product-content"><small>{product.category} · {product.product_id}</small><h3>{product.name}</h3><p>{product.summary}</p><div className="product-price">¥{product.price.toFixed(2)}</div><div className="stock-line">库存 {product.stock}<span>{product.stock > 0 ? 'AVAILABLE' : 'OUT OF STOCK'}</span></div>
        <details><summary>商品参数</summary><dl>{Object.entries(product.specs).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl></details>
      </div>
    </article>)}</div>
  </>
}
