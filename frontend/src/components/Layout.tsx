import {
  Box,
  Boxes,
  CircleUserRound,
  FlaskConical,
  History,
  LayoutDashboard,
  PackageSearch,
  Route,
  Settings2,
  ShoppingBag,
  Store,
} from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'

const engineering = [
  { to: '/', label: '工程总览', icon: LayoutDashboard, end: true },
  { to: '/runs', label: '运行检查', icon: Route },
  { to: '/experiments', label: '论文实验室', icon: FlaskConical },
  { to: '/mechanisms', label: '机制与配置', icon: Boxes },
  { to: '/datasets', label: '评测数据集', icon: Box },
]

const consumer = [
  { to: '/sandbox', label: '消费工作台', icon: Store, end: true },
  { to: '/sandbox/products', label: '商品目录', icon: PackageSearch },
  { to: '/sandbox/orders', label: '订单与售后', icon: ShoppingBag },
  { to: '/sandbox/tasks', label: '任务记录', icon: History },
  { to: '/sandbox/preferences', label: '我的偏好', icon: Settings2 },
]

function NavSection({ title, items }: { title: string; items: typeof engineering }) {
  return (
    <section className="nav-section">
      <div className="nav-section-title">{title}</div>
      <nav aria-label={title}>
        {items.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <Icon size={15} strokeWidth={1.7} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </section>
  )
}

export function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/sandbox" className="brand" aria-label="LingCart 消费工作台">
          <span className="brand-mark">序</span>
          <span><b>LingCart</b><small>消费任务智能体</small></span>
        </NavLink>
        <div className="nav-scroll">
          <NavSection title="消费体验" items={consumer} />
          <NavSection title="研发与证据" items={engineering} />
        </div>
        <div className="sidebar-runtime">
          <div className="runtime-line"><i /><span>本地工作区</span><b>演示数据</b></div>
          <NavLink className="new-session" to="/sandbox?new=1"><CircleUserRound size={15} />新建业务对话</NavLink>
          <div className="build-note"><span>LINGCART / LOCAL</span><p>可恢复执行 · 证据可追溯 · 高影响操作需确认</p></div>
        </div>
      </aside>
      <main className="main-canvas">
        <Outlet />
      </main>
    </div>
  )
}
