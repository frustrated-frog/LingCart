import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, Plus, ShieldCheck, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, SectionHeading } from '../components/SystemUI'
import { getPreferenceProfileId, savePreferenceProfileId } from '../state/localSession'

const featureOptions = ['主动降噪', '通话降噪', '三模', '双模', '有线', '无线', '蓝牙', '头戴式', '开放式', '入耳式', '4K', '60Hz', '便携', '通勤', '办公', '防水', '红轴', '茶轴', '双屏', 'HDMI', '运动', '视频会议', '麦克风', '空间音频']

const preferenceLabels: Record<string, string> = {
  max_price: '预算上限',
  category: '常用品类',
  features: '关注特征',
}

export function PreferencesPage() {
  const queryClient = useQueryClient()
  const [profileId, setProfileId] = useState(getPreferenceProfileId() ?? '')
  const [budget, setBudget] = useState(1000)
  const [category, setCategory] = useState('耳机')
  const [features, setFeatures] = useState<string[]>([])
  const preferences = useQuery({
    queryKey: ['preferences', profileId],
    queryFn: () => api.preferences(profileId),
    enabled: Boolean(profileId),
  })

  useEffect(() => {
    const values = Object.fromEntries((preferences.data?.preferences ?? []).map((item) => [item.key, item.value.value]))
    if (typeof values.max_price === 'number') setBudget(values.max_price)
    if (typeof values.category === 'string') setCategory(values.category)
    if (Array.isArray(values.features)) setFeatures(values.features.filter((item): item is string => typeof item === 'string'))
  }, [preferences.data])

  const createProfile = useMutation({
    mutationFn: api.createPreferenceProfile,
    onSuccess: ({ profile_id }) => {
      savePreferenceProfileId(profile_id)
      setProfileId(profile_id)
    },
  })
  const save = useMutation({
    mutationFn: async () => {
      if (!profileId) throw new Error('请先创建偏好档案')
      await api.savePreference(profileId, { key: 'max_price', value: budget, confirmed: true })
      await api.savePreference(profileId, { key: 'category', value: category, confirmed: true })
      return api.savePreference(profileId, { key: 'features', value: features, confirmed: true })
    },
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['preferences', profileId] }),
  })
  const forget = useMutation({
    mutationFn: (key: string) => api.forgetPreference(profileId, key),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['preferences', profileId] }),
  })
  const current = useMemo(() => preferences.data?.preferences ?? [], [preferences.data])

  if (profileId && preferences.isLoading) return <LoadingState label="正在读取偏好档案" />
  if (preferences.isError) return <ErrorState error={preferences.error} retry={() => void preferences.refetch()} />

  return <>
    <PageHeader code="PREFERENCE MEMORY / 14" title="我的偏好"
      subtitle="只有明确确认的预算、品类和选购特征才会进入长期记忆；可以随时查看版本或忘记单项。" marker="USER CONTROLLED" />
    {!profileId ? <section className="preference-onboarding">
      <div className="onboarding-index">01</div>
      <div><span>NO PROFILE BOUND</span><h2>创建一份本地偏好档案</h2><p>档案使用随机标识保存在本地数据库。Agent 不会从对话中静默推断并写入偏好。</p></div>
      <button className="primary" onClick={() => createProfile.mutate()} disabled={createProfile.isPending}><Plus size={15} />创建档案</button>
    </section> : <>
      <div className="profile-identity"><div><span>PROFILE ID</span><code>{profileId}</code></div><div><ShieldCheck size={17} /><span>仅保存已确认字段</span></div></div>
      <div className="preference-layout">
        <section>
          <SectionHeading index="01" title="编辑偏好" note="EXPLICIT CONFIRMATION REQUIRED" />
          <div className="panel preference-form">
            <label>预算上限（元）<input type="number" min="1" max="1000000" step="100" value={budget} onChange={(event) => setBudget(Number(event.target.value))} /></label>
            <label>常用品类<select value={category} onChange={(event) => setCategory(event.target.value)}><option>耳机</option><option>键盘</option><option>扩展坞</option></select></label>
            <fieldset><legend>关注特征（最多 6 项）</legend><div className="feature-options">{featureOptions.map((feature) => {
              const selected = features.includes(feature)
              return <button type="button" className={selected ? 'selected' : ''} key={feature} onClick={() => setFeatures((items) => selected ? items.filter((item) => item !== feature) : items.length < 6 ? [...items, feature] : items)}>{selected && <Check size={12} />}{feature}</button>
            })}</div></fieldset>
            <div className="confirmation-note"><ShieldCheck size={16} /><p><b>本次保存会覆盖这三个字段。</b><span>点击按钮代表你明确同意把这些值用于后续已绑定会话。</span></p></div>
            <button className="primary save-preferences" onClick={() => save.mutate()} disabled={save.isPending || budget <= 0}><Check size={15} />确认并保存</button>
            {save.isSuccess && <div className="inline-success">偏好已保存，下一次业务对话会读取最新版本。</div>}
            {save.isError && <div className="inline-error">{save.error.message}</div>}
          </div>
        </section>
        <section>
          <SectionHeading index="02" title="已保存字段" note={`${current.length} ACTIVE RECORDS`} />
          <div className="preference-records">{current.length === 0 && <div className="empty-inline">尚未保存偏好</div>}{current.map((item) => <article key={item.key}>
            <div><span>{preferenceLabels[item.key] ?? item.key}</span><code>{item.key}</code></div>
            <strong>{Array.isArray(item.value.value) ? item.value.value.join(' · ') : String(item.value.value)}</strong>
            <small>VERSION {item.version} · {new Date(item.updated_at).toLocaleString('zh-CN', { hour12: false })}</small>
            <button aria-label={`忘记${preferenceLabels[item.key] ?? item.key}`} onClick={() => forget.mutate(item.key)} disabled={forget.isPending}><Trash2 size={14} />忘记</button>
          </article>)}</div>
        </section>
      </div>
    </>}
  </>
}
