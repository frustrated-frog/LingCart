import { useQuery, useQueryClient } from '@tanstack/react-query'
import { KeyRound, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { Outlet } from 'react-router-dom'
import { api, ApiError, setEngineeringKey } from '../api/client'
import { ErrorState, LoadingState } from './SystemUI'

export function EngineeringGate() {
  const queryClient = useQueryClient()
  const [key, setKey] = useState('')
  const access = useQuery({ queryKey: ['engineering-overview'], queryFn: api.overview, retry: false })
  if (access.isLoading) return <LoadingState label="正在验证工程控制台访问权限" />
  if (access.isError && (!(access.error instanceof ApiError) || access.error.status !== 403)) {
    return <ErrorState error={access.error} retry={() => void access.refetch()} />
  }
  if (access.isError) {
    const submit = (event: FormEvent) => {
      event.preventDefault()
      setEngineeringKey(key)
      void queryClient.invalidateQueries({ queryKey: ['engineering-overview'] })
    }
    return <section className="engineering-access">
      <div className="access-mark"><KeyRound size={22} /></div>
      <span>TRUSTED RESEARCH ENVIRONMENT</span>
      <h1>工程控制台需要访问凭据</h1>
      <p>运行轨迹包含用户请求、上下文、工具参数和检查点。远程访问时必须提供服务端配置的 Engineering Key；本机直连无需凭据。</p>
      <form onSubmit={submit}><label>ENGINEERING KEY<input type="password" autoComplete="off" value={key} onChange={(event) => setKey(event.target.value)} /></label><button className="primary" disabled={!key.trim()}><ShieldCheck size={14} />验证并进入</button></form>
    </section>
  }
  return <Outlet />
}
