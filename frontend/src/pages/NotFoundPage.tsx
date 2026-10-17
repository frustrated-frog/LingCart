import { ArrowLeft } from 'lucide-react'
import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return <section className="not-found"><code>HTTP 404 / ROUTE NOT FOUND</code><h1>这里没有可用界面</h1><p>地址可能已变更，工程数据没有受到影响。</p><Link to="/"><ArrowLeft size={15} />返回 Agent 控制台</Link></section>
}
