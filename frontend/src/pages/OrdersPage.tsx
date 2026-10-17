import { useQuery } from '@tanstack/react-query'
import { CreditCard, RotateCcw, Search, Truck } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, STATUS_LABELS, StatusBadge } from '../components/SystemUI'

const ITEM_LABELS: Record<string, string> = {
  'Wireless Headphones': '无线耳机', 'Mechanical Keyboard': '机械键盘', 'USB-C Hub': 'USB-C 扩展坞',
  'Standing Desk': '升降桌', 'Desk Lamp': '台灯', 'Bluetooth Speaker': '蓝牙音箱',
  'Laptop Stand': '笔记本支架', 'Webcam 1080p': '1080p 摄像头', 'Ergonomic Mouse': '人体工学鼠标',
  'Noise-Cancelling Earbuds': '降噪耳塞',
}

export function OrdersPage() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('')
  const orders = useQuery({ queryKey: ['orders', search, status], queryFn: () => api.orders({ query: search, status: status || undefined }) })
  if (orders.isLoading) return <LoadingState label="正在读取订单" />
  if (orders.isError) return <ErrorState error={orders.error} retry={() => void orders.refetch()} />
  const ask = (prompt: string) => navigate(`/sandbox?prompt=${encodeURIComponent(prompt)}`)
  return <>
    <PageHeader code="ORDER SANDBOX / 12" title="订单与售后" subtitle="查看订单、支付和售后状态，再把具体业务操作交给 Agent。" marker="DEMO ORDERS" />
    <div className="filter-row two"><label>搜索订单<div className="input-with-icon"><Search size={14} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="订单号、商品名称或演示客户" /></div></label><label>订单状态<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="">全部状态</option><option value="placed">已下单</option><option value="shipped">运输中</option><option value="delivered">已送达</option><option value="cancelled">已取消</option></select></label></div>
    <div className="catalog-note">{orders.data?.orders.length ?? 0} 条订单 · 共享本地演示数据，未接入真实客户账户</div>
    <div className="order-list">{orders.data?.orders.map((order) => <article className="order-card" key={order.order_id}>
      <div className="order-main"><div className="order-symbol">↗</div><div><small>ORDER #{order.order_id}</small><h3>{ITEM_LABELS[order.item] ?? order.item}</h3><p>{order.order_date} · ¥{order.amount.toFixed(2)} · 支付 {STATUS_LABELS[order.payment_status ?? 'none'] ?? order.payment_status}</p></div></div><StatusBadge status={order.status} />
      <div className="order-actions"><button onClick={() => ask(`查询订单 ${order.order_id} 的状态和物流`)}><Truck size={14} />状态与物流</button><button onClick={() => ask(`查询订单 ${order.order_id} 是否可以退款`)}><RotateCcw size={14} />检查退款资格</button><button className="primary" disabled={order.status === 'cancelled'} onClick={() => ask(order.payment_status === 'pending' ? `帮我支付订单 ${order.order_id}` : `订单 ${order.order_id} 商品有问题，我想退款`)}><CreditCard size={14} />{order.payment_status === 'pending' ? '准备支付' : '申请退款'}</button></div>
    </article>)}</div>
  </>
}
