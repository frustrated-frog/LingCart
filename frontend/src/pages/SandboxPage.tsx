import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUp, Check, CircleStop, Search, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { PageHeader, StatusBadge } from '../components/SystemUI'
import { getPreferenceProfileId, getSessionId, resetSessionId } from '../state/localSession'
import type { ChatResponse, PendingAction } from '../types'

interface Message {
  role: 'user' | 'assistant'
  content: string
  result?: ChatResponse
}

const quickPrompts = [
  ['按预算选购', '推荐降噪耳机，预算 500 元以内'],
  ['查物流', '查询订单 1002 的状态和物流'],
  ['检查退款资格', '查询订单 1001 是否可以退款'],
] as const

export function SandboxPage() {
  const [params, setParams] = useSearchParams()
  const client = useQueryClient()
  const initialSession = useRef(getSessionId())
  const [sessionId, setSessionId] = useState(initialSession.current)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState(params.get('prompt') ?? '')
  const [adaptive, setAdaptive] = useState(false)
  const [paymentMethod, setPaymentMethod] = useState('card')
  const conversation = useQuery({ queryKey: ['conversation', sessionId], queryFn: () => api.conversation(sessionId) })

  useEffect(() => {
    if (params.get('new') === '1') {
      const next = resetSessionId()
      setSessionId(next)
      setMessages([])
      setParams({}, { replace: true })
    }
  }, [params, setParams])
  useEffect(() => {
    if (conversation.data && messages.length === 0) setMessages(conversation.data.messages.map((item) => ({ role: item.role, content: item.content })))
  }, [conversation.data, messages.length])

  const send = useMutation({
    mutationFn: async ({ message, action }: { message: string; action?: PendingAction }) => {
      const profileId = getPreferenceProfileId()
      if (profileId) await api.bindPreferenceProfile(profileId, sessionId)
      return api.chat({ message, session_id: sessionId, request_id: crypto.randomUUID().replaceAll('-', ''),
        action_id: action?.action_id, harness_profile: adaptive ? 'adaptive-v1' : 'baseline-v1' })
    },
    onSuccess: (result) => {
      setMessages((current) => [...current, { role: 'assistant', content: result.reply, result }])
      void client.invalidateQueries({ queryKey: ['runs'] })
      void client.invalidateQueries({ queryKey: ['engineering-overview'] })
    },
  })

  const submit = (message: string, action?: PendingAction) => {
    const value = message.trim()
    if (!value || send.isPending) return
    setMessages((current) => [...current, { role: 'user', content: value }])
    setDraft('')
    send.mutate({ message: value, action })
  }
  const onSubmit = (event: FormEvent) => { event.preventDefault(); submit(draft) }
  const latestResult = [...messages].reverse().find((item) => item.result)?.result
  const pending = latestResult?.pending_action ?? null

  return <>
    <PageHeader code="BUSINESS SANDBOX / 10" title="消费工作台"
      subtitle="用真实业务路径产生可检查的运行证据。商品、订单、支付和退款均连接本地演示数据库。" marker="DEMO DATA" />
    <div className="sandbox-layout">
      <section className="conversation-panel">
        <div className="conversation-toolbar">
          <div><span>CONVERSATION</span><code>{sessionId.slice(0, 12)}</code></div>
          <label className="mode-switch"><input type="checkbox" checked={adaptive} onChange={(event) => setAdaptive(event.target.checked)} /><span>Adaptive planning</span></label>
        </div>
        <div className="message-list">
          {messages.length === 0 && <div className="conversation-empty"><Sparkles size={24} /><h2>告诉 Agent 你的目标</h2><p>从明确的业务请求开始，或启用 Adaptive planning 观察自主规划链路。</p></div>}
          {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
            <div className="message-role">{message.role === 'user' ? 'YOU' : 'LINGCART AGENT'}</div>
            <div className="message-copy">{message.content}</div>
            {message.result && <div className="message-meta"><StatusBadge status={message.result.status} /><span>{message.result.agent} · {message.result.run_id.slice(0, 10)} · {message.result.trace.length} tools</span></div>}
          </article>)}
          {send.isPending && <article className="message assistant pending"><div className="message-role">LINGCART AGENT</div><div className="thinking-line"><i /><span>正在执行并核验结果</span></div></article>}
          {send.isError && <div className="inline-error">{send.error.message}</div>}
        </div>
        {pending && <ApprovalPanel action={pending} paymentMethod={paymentMethod} setPaymentMethod={setPaymentMethod}
          onConfirm={() => submit(pending.type === 'PROCESS_PAYMENT' ? `确认使用 ${paymentMethod} 支付` : '确认继续', pending)}
          onCancel={() => submit('取消，先不要执行', pending)} disabled={send.isPending} />}
        <form className="composer" onSubmit={onSubmit}>
          <textarea aria-label="消费需求" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="继续提问，或补充你的需求…" rows={3} />
          <button aria-label="发送需求" disabled={!draft.trim() || send.isPending}><ArrowUp size={18} /></button>
        </form>
      </section>
      <aside className="sandbox-aside">
        <div className="aside-block"><span>QUICK STARTS</span><h3>常用任务</h3>{quickPrompts.map(([label, prompt]) => <button onClick={() => submit(prompt)} disabled={send.isPending} key={label}><Search size={14} /><div><b>{label}</b><small>{prompt}</small></div></button>)}</div>
        <div className="aside-block evidence-note"><span>EXECUTION EVIDENCE</span><h3>每次处理都可追溯</h3><p>运行完成后会进入 Run Inspector，保留 Profile、事件、工具参数、目标证据与终止原因。</p></div>
      </aside>
    </div>
  </>
}

function ApprovalPanel({ action, paymentMethod, setPaymentMethod, onConfirm, onCancel, disabled }: {
  action: PendingAction; paymentMethod: string; setPaymentMethod: (value: string) => void
  onConfirm: () => void; onCancel: () => void; disabled: boolean
}) {
  const title = action.type === 'PROCESS_PAYMENT' ? '支付' : '退款'
  return <div className="approval-panel"><div><span>APPROVAL REQUIRED</span><h3>确认订单 #{action.order_id} 的{title}操作</h3><p>只有明确确认后才会修改本地演示数据库。</p></div>
    {action.type === 'PROCESS_PAYMENT' && <label>支付方式<select value={paymentMethod} onChange={(event) => setPaymentMethod(event.target.value)}><option value="card">银行卡</option><option value="paypal">PayPal</option></select></label>}
    <div className="approval-actions"><button className="primary" onClick={onConfirm} disabled={disabled}><Check size={14} />确认{title}</button><button onClick={onCancel} disabled={disabled}><CircleStop size={14} />取消操作</button></div>
  </div>
}
