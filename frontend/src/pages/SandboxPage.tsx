import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUp, Check, CircleStop, CornerDownRight, RotateCcw, Search, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, ApiError, streamRunEvents } from '../api/client'
import { ResultArtifactViews } from '../components/CatalogResultView'
import { ExecutionSpine, PageHeader, StatusBadge, TERMINAL_REASON_LABELS } from '../components/SystemUI'
import { getPreferenceProfileId, getSessionId, resetSessionId } from '../state/localSession'
import { clearRunOperation, loadRunOperation, saveRunOperation } from '../state/runOperation'
import type { RunOperation } from '../state/runOperation'
import { projectAgentStatus, projectRunResponse } from '../state/runStream'
import type { StreamConnectionState } from '../state/runStream'
import type { ExecutionResult, PendingAction, RunEvent } from '../types'

interface Message {
  role: 'user' | 'assistant'
  content: string
  runId: string
  result?: ExecutionResult
}

const quickPrompts = [
  ['按预算选购', '推荐降噪耳机，预算 500 元以内'],
  ['查物流', '查询订单 1002 的状态和物流'],
  ['检查退款资格', '查询订单 1001 是否可以退款'],
] as const

function operationId(): string {
  return crypto.randomUUID().replaceAll('-', '')
}

function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] {
  const keyed = new Map(current.map((event, index) => [event.sequence ?? index + 1, event]))
  incoming.forEach((event, index) => keyed.set(event.sequence ?? current.length + index + 1, event))
  return [...keyed.entries()].sort(([left], [right]) => left - right).map(([, event]) => event)
}

function conciseFailure(message: string): string {
  return message.split('\n', 1)[0].slice(0, 240)
}

export function SandboxPage() {
  const [params, setParams] = useSearchParams()
  const queryClient = useQueryClient()
  const initialSession = useRef(getSessionId())
  const [sessionId, setSessionId] = useState(initialSession.current)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState(params.get('prompt') ?? '')
  const [adaptive, setAdaptive] = useState(false)
  const [paymentMethod, setPaymentMethod] = useState('card')
  const [operation, setOperation] = useState<RunOperation | null>(() => loadRunOperation(initialSession.current))
  const [activeRunId, setActiveRunId] = useState<string | null>(() => operation?.runId ?? null)
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([])
  const [controlDraft, setControlDraft] = useState('')
  const [streamConnection, setStreamConnection] = useState<StreamConnectionState>('idle')
  const messageListRef = useRef<HTMLDivElement>(null)
  const followOutputRef = useRef(true)
  const streamCursorRef = useRef({ runId: '', sequence: 0 })

  const conversation = useQuery({
    queryKey: ['conversation', sessionId],
    queryFn: () => api.conversation(sessionId),
    refetchInterval: activeRunId ? 1_500 : false,
  })
  const activeRun = useQuery({
    queryKey: ['product-run', activeRunId, sessionId],
    queryFn: () => api.productRun(activeRunId as string, sessionId),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) => query.state.data?.result === null ? 750 : false,
  })
  const commands = useQuery({
    queryKey: ['run-commands', activeRunId, sessionId],
    queryFn: () => api.runCommands(activeRunId as string, sessionId),
    enabled: Boolean(activeRunId) && activeRun.data?.result === null,
    refetchInterval: activeRun.data?.result === null ? 1_000 : false,
  })

  useEffect(() => {
    if (params.get('new') !== '1') return
    const next = resetSessionId()
    setSessionId(next)
    setMessages([])
    setOperation(loadRunOperation(next))
    setActiveRunId(null)
    setLiveEvents([])
    setParams({}, { replace: true })
  }, [params, setParams])

  useEffect(() => {
    const saved = conversation.data
    if (!saved) return
    setMessages((current) => {
      const results = new Map(current.filter((item) => item.result).map((item) => [item.runId, item.result]))
      return saved.messages.map((item) => ({
        role: item.role,
        content: item.content,
        runId: item.run_id,
        result: item.role === 'assistant' ? item.result ?? results.get(item.run_id) : undefined,
      }))
    })
    if (saved.active_run_id) {
      setActiveRunId(saved.active_run_id)
      setOperation((current) => {
        if (!current) return current
        const recovered = { ...current, runId: saved.active_run_id as string }
        saveRunOperation(sessionId, recovered)
        return recovered
      })
    }
  }, [conversation.data, sessionId])

  const activeRunStatus = activeRun.data?.status
  const activeRunFinished = Boolean(activeRun.data?.result)
  useEffect(() => {
    if (!activeRunId || activeRunStatus === 'INTERRUPTED' || activeRunFinished) return
    if (streamCursorRef.current.runId !== activeRunId) {
      streamCursorRef.current = { runId: activeRunId, sequence: 0 }
      setLiveEvents(activeRun.data?.events ?? [])
    } else if (activeRun.data?.events) {
      setLiveEvents((current) => mergeEvents(current, activeRun.data?.events ?? []))
    }
    const persistedLast = activeRun.data?.events.at(-1)?.sequence ?? 0
    streamCursorRef.current.sequence = Math.max(streamCursorRef.current.sequence, persistedLast)
    const stream = streamRunEvents(activeRunId, sessionId,
      (event) => {
        if (typeof event.sequence === 'number') streamCursorRef.current.sequence = event.sequence
        setLiveEvents((current) => mergeEvents(current, [event]))
      }, streamCursorRef.current.sequence, setStreamConnection)
    return () => stream.abort()
  }, [activeRunFinished, activeRunId, activeRunStatus, sessionId])

  useEffect(() => {
    if (activeRun.data?.events) setLiveEvents((current) => mergeEvents(current, activeRun.data?.events ?? []))
  }, [activeRun.data?.events])

  useEffect(() => {
    const node = messageListRef.current
    if (node && followOutputRef.current) node.scrollTop = node.scrollHeight
  }, [liveEvents, messages])

  useEffect(() => {
    const result = activeRun.data?.result
    if (!result || !activeRunId) return
    setMessages((current) => {
      const existing = current.findIndex((item) => item.role === 'assistant' && item.runId === activeRunId)
      if (existing >= 0) return current.map((item, index) => index === existing ? { ...item, result } : item)
      return [...current, { role: 'assistant', content: result.reply, runId: activeRunId, result }]
    })
    clearRunOperation(sessionId)
    setOperation(null)
    setActiveRunId(null)
    setControlDraft('')
    void queryClient.invalidateQueries({ queryKey: ['conversation', sessionId] })
    void queryClient.invalidateQueries({ queryKey: ['runs'] })
    void queryClient.invalidateQueries({ queryKey: ['engineering-overview'] })
  }, [activeRun.data?.result, activeRunId, queryClient, sessionId])

  const send = useMutation({
    mutationFn: async (next: RunOperation) => {
      const profileId = getPreferenceProfileId()
      if (profileId) await api.bindPreferenceProfile(profileId, sessionId)
      return api.startRun({
        message: next.message,
        session_id: sessionId,
        request_id: next.requestId,
        action_id: next.actionId,
        harness_profile: next.harnessProfile,
        execution_mode: next.executionMode,
      })
    },
    retry: (count, error) => !(error instanceof ApiError) && count < 1,
    onSuccess: (handle, current) => {
      const accepted = { ...current, runId: handle.run_id }
      saveRunOperation(sessionId, accepted)
      setOperation(accepted)
      setActiveRunId(handle.run_id)
      void queryClient.invalidateQueries({ queryKey: ['conversation', sessionId] })
    },
    onError: async () => {
      const saved = await api.conversation(sessionId).catch(() => null)
      if (!saved?.active_run_id) return
      setActiveRunId(saved.active_run_id)
      setOperation((current) => {
        if (!current) return current
        const recovered = { ...current, runId: saved.active_run_id as string }
        saveRunOperation(sessionId, recovered)
        return recovered
      })
      send.reset()
    },
  })

  const control = useMutation({
    mutationFn: (input: { kind: 'steer' | 'cancel' | 'follow-up'; message: string; commandId: string; revision: number }) =>
      api.controlRun(activeRunId as string, input.kind, {
        session_id: sessionId,
        message: input.message,
        command_id: input.commandId,
        expected_revision: input.revision,
      }),
    retry: (count, error) => !(error instanceof ApiError) && count < 1,
    onSuccess: async () => {
      setControlDraft('')
      await commands.refetch()
    },
  })
  const resume = useMutation({
    mutationFn: () => api.resumeRun(activeRunId as string, sessionId),
    onSuccess: (handle) => {
      setActiveRunId(handle.run_id)
      void activeRun.refetch()
    },
  })

  const running = Boolean(activeRunId && activeRun.data?.result === null && activeRun.data?.status === 'RUNNING')
  const interrupted = Boolean(activeRunId && activeRun.data?.result === null && activeRun.data?.status === 'INTERRUPTED')
  const pendingRun = Boolean(activeRunId && !activeRun.data?.result && !interrupted)
  const busy = send.isPending || pendingRun || interrupted
  const submit = (message: string, action?: PendingAction) => {
    const value = message.trim()
    if (!value || busy) return
    const next: RunOperation = {
      requestId: operationId(),
      message: value,
      actionId: action?.action_id,
      harnessProfile: adaptive ? 'adaptive-v1' : 'baseline-v1',
      executionMode: adaptive ? 'adaptive' : 'auto',
    }
    saveRunOperation(sessionId, next)
    setOperation(next)
    setLiveEvents([])
    setStreamConnection('idle')
    setMessages((current) => [...current, { role: 'user', content: value, runId: `pending:${next.requestId}` }])
    setDraft('')
    send.mutate(next)
  }
  const onSubmit = (event: FormEvent) => { event.preventDefault(); submit(draft) }
  const latestResult = [...messages].reverse().find((item) => item.result)?.result
  const pending = latestResult?.pending_action ?? activeRun.data?.result?.pending_action
    ?? conversation.data?.state.pending_action ?? null
  const events = liveEvents.length ? liveEvents : activeRun.data?.events ?? []
  const liveResponse = useMemo(() => projectRunResponse(events), [events])
  const agentStatus = useMemo(() => projectAgentStatus(events), [events])
  const liveStatusLabel = send.isPending ? '正在提交可恢复任务'
    : agentStatus?.label ?? '任务已接收，正在恢复执行状态'
  const revision = commands.data?.revision
  const runControl = (kind: 'steer' | 'cancel' | 'follow-up') => {
    const message = kind === 'cancel' ? '' : controlDraft.trim()
    if ((kind !== 'cancel' && !message) || control.isPending || revision === undefined) return
    control.mutate({ kind, message, commandId: operationId(), revision })
  }
  const retryOperation = () => { if (operation && !operation.runId) send.mutate(operation) }

  return <>
    <PageHeader code="BUSINESS SANDBOX / 10" title="消费工作台"
      subtitle="用真实业务路径产生可检查的运行证据。任务异步执行，可实时观察、调整、排队后续请求或中止。" marker="DEMO DATA" />
    <div className="sandbox-layout">
      <section className="conversation-panel">
        <div className="conversation-toolbar">
          <div><span>CONVERSATION</span><code>{sessionId.slice(0, 12)}</code></div>
          <label className="mode-switch"><input type="checkbox" checked={adaptive} disabled={busy} onChange={(event) => setAdaptive(event.target.checked)} /><span>自适应规划</span></label>
        </div>
        <div className="message-list" ref={messageListRef} onScroll={(event) => {
          const node = event.currentTarget
          followOutputRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 96
        }}>
          {messages.length === 0 && <div className="conversation-empty"><Sparkles size={24} /><h2>告诉 Agent 你的目标</h2><p>从明确的业务请求开始，或启用 Adaptive planning 观察自主规划链路。</p></div>}
          {messages.map((message, index) => <article className={`message ${message.role}${message.result?.artifacts?.some((artifact) => ['catalog_query_result', 'constraint_plan_result'].includes(artifact.artifact_type)) ? ' has-catalog' : ''}`} key={`${message.runId}-${message.role}-${index}`}>
            <div className="message-role">{message.role === 'user' ? 'YOU' : 'LINGCART AGENT'}</div>
            <div className="message-copy">{message.content}</div>
            {message.result && <ResultArtifactViews artifacts={message.result.artifacts} />}
            {message.result?.error && <div className="message-error-evidence" role="alert" aria-label="运行失败详情">
              <div><strong>{TERMINAL_REASON_LABELS[message.result.error.code] ?? '运行协议拒绝'}</strong><code>{message.result.error.code}</code></div>
              <p>{conciseFailure(message.result.error.message)}</p>
              <small>{message.result.error.retryable ? '该错误允许使用同一请求编号重试' : '该错误需要修正协议或输入后再执行'}</small>
            </div>}
            {message.result && <div className="message-meta"><StatusBadge status={message.result.status} /><span>{message.result.agent} · {message.result.run_id.slice(0, 10)} · {message.result.trace.length} tools</span></div>}
          </article>)}
          {(send.isPending || pendingRun) && <article className="message assistant live-response" aria-live="polite" aria-busy={liveResponse.state !== 'completed'}>
            <div className="message-role"><span>LINGCART AGENT</span>{streamConnection === 'reconnecting' && <em>连接恢复中</em>}</div>
            <div className="agent-progress"><i /><span>{liveStatusLabel}</span></div>
            {liveResponse.content && <div className="message-copy stream-copy">{liveResponse.content}{liveResponse.state === 'streaming' && <span className="stream-cursor" aria-hidden="true" />}</div>}
            {!liveResponse.content && <div className="stream-placeholder">执行进度会持续同步，回答生成后将在这里逐步显示。</div>}
            {liveResponse.state === 'failed' && <div className="stream-warning">回答生成中断，已输出内容和运行证据仍然保留。</div>}
          </article>}
          {send.isError && <div className="inline-error">{send.error.message}<button onClick={retryOperation} disabled={!operation || Boolean(operation.runId)}>使用同一请求编号重试</button></div>}
        </div>

        {activeRunId && <section className="live-run-panel">
          <div className="live-run-head"><div><span>LIVE RUN</span><code>{activeRunId.slice(0, 12)}</code></div><StatusBadge status={activeRun.data?.status ?? 'RUNNING'} /></div>
          <ExecutionSpine events={events} />
          {running && <div className="run-control-composer">
            <textarea aria-label="运行时指令" rows={2} value={controlDraft} onChange={(event) => setControlDraft(event.target.value)} placeholder="补充约束，或排队下一项任务…" />
            <div><button onClick={() => runControl('steer')} disabled={!controlDraft.trim() || control.isPending || revision === undefined}><CornerDownRight size={14} />调整当前目标</button><button onClick={() => runControl('follow-up')} disabled={!controlDraft.trim() || control.isPending || revision === undefined}>排队后续任务</button><button className="danger" onClick={() => runControl('cancel')} disabled={control.isPending || revision === undefined}><CircleStop size={14} />停止运行</button></div>
          </div>}
          {interrupted && <div className="run-recovery"><p>任务在完成前中断，检查点和已完成工具结果仍然保留。</p><button className="primary" onClick={() => resume.mutate()} disabled={resume.isPending}><RotateCcw size={14} />从检查点恢复</button></div>}
          {(control.isError || resume.isError) && <div className="inline-error">{(control.error ?? resume.error)?.message}</div>}
        </section>}

        {pending && <ApprovalPanel action={pending} paymentMethod={paymentMethod} setPaymentMethod={setPaymentMethod}
          onConfirm={() => submit(pending.type === 'PROCESS_PAYMENT' ? `确认使用 ${paymentMethod} 支付` : '确认继续', pending)}
          onCancel={() => submit('取消，先不要执行', pending)} disabled={busy} />}
        <form className="composer" onSubmit={onSubmit}>
          <textarea aria-label="消费需求" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={busy ? '当前任务结束后可发送新需求；运行时约束请在上方补充' : '继续提问，或补充你的需求…'} rows={3} disabled={busy} />
          <button aria-label="发送需求" disabled={!draft.trim() || busy}><ArrowUp size={18} /></button>
        </form>
      </section>
      <aside className="sandbox-aside">
        <div className="aside-block"><span>QUICK STARTS</span><h3>常用任务</h3>{quickPrompts.map(([label, prompt]) => <button onClick={() => submit(prompt)} disabled={busy} key={label}><Search size={14} /><div><b>{label}</b><small>{prompt}</small></div></button>)}</div>
        <div className="aside-block evidence-note"><span>DURABLE EXECUTION</span><h3>刷新页面不会丢任务</h3><p>请求编号、活动 Run、检查点、审批状态和事件序列均由持久层恢复；网络重试复用同一幂等键。</p></div>
      </aside>
    </div>
  </>
}

function ApprovalPanel({ action, paymentMethod, setPaymentMethod, onConfirm, onCancel, disabled }: {
  action: PendingAction; paymentMethod: string; setPaymentMethod: (value: string) => void
  onConfirm: () => void; onCancel: () => void; disabled: boolean
}) {
  const caseSubmission = action.type === 'SUBMIT_AFTER_SALES_CASE'
  const title = action.type === 'PROCESS_PAYMENT' ? '支付' : caseSubmission ? '售后退款提交' : '退款'
  const confirmLabel = caseSubmission ? '确认提交退款申请' : `确认${title}`
  return <div className="approval-panel"><div><span>APPROVAL REQUIRED</span><h3>确认订单 #{action.order_id} 的{title}操作</h3><p>{caseSubmission
    ? `案件 ${action.case_id ?? '—'} 已通过政策核验；确认后才会创建退款申请。`
    : '审批状态已持久化；刷新后仍会恢复，只有明确确认才会修改本地演示数据库。'}</p></div>
    {action.type === 'PROCESS_PAYMENT' && <label>支付方式<select value={paymentMethod} onChange={(event) => setPaymentMethod(event.target.value)}><option value="card">银行卡</option><option value="paypal">PayPal</option></select></label>}
    <div className="approval-actions"><button className="primary" onClick={onConfirm} disabled={disabled}><Check size={14} />{confirmLabel}</button><button onClick={onCancel} disabled={disabled}><CircleStop size={14} />取消操作</button></div>
  </div>
}
