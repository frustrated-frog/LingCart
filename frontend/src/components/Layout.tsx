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
  { to: '/', label: '智能体控制台', icon: LayoutDashboard, end: true },
  { to: '/runs', label: '运行检查', icon: Route },
  { to: '/experiments', label: '实验契约', icon: FlaskConical },
  { to: '/mechanisms', label: '机制注册表', icon: Boxes },
  { to: '/datasets', label: '数据集覆盖', icon: Box },
]

const sandbox = [
  { to: '/sandbox', label: '消费工作台', icon: Store, end: true },
  { to: '/sandbox/products', label: '商品发现', icon: PackageSearch },
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
        <NavLink to="/" className="brand" aria-label="LingCart 首页">
          <span className="brand-mark">序</span>
          <span><b>LingCart</b><small>智能体工程</small></span>
        </NavLink>
        <div className="nav-scroll">
          <NavSection title="智能体工程" items={engineering} />
          <NavSection title="业务沙盒" items={sandbox} />
        </div>
        <div className="sidebar-runtime">
          <div className="runtime-line"><i /><span>本地运行时</span><b>已连接</b></div>
          <NavLink className="new-session" to="/sandbox?new=1"><CircleUserRound size={15} />新建业务对话</NavLink>
          <div className="build-note"><span>LINGCART / 网页界面</span><p>本地数据 · 可追溯执行 · 桌面工程界面</p></div>
        </div>
      </aside>
      <main className="main-canvas">
        <Outlet />
      </main>
    </div>
  )
}
