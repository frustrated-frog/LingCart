const SESSION_KEY = 'lingcart.session_id'
const PROFILE_KEY = 'lingcart.preference_profile_id'

function id(): string {
  return crypto.randomUUID().replaceAll('-', '')
}

export function getSessionId(): string {
  const saved = localStorage.getItem(SESSION_KEY)
  if (saved) return saved
  const created = id()
  localStorage.setItem(SESSION_KEY, created)
  return created
}

export function resetSessionId(): string {
  const created = id()
  localStorage.setItem(SESSION_KEY, created)
  return created
}

export function getPreferenceProfileId(): string | null {
  return localStorage.getItem(PROFILE_KEY)
}

export function savePreferenceProfileId(profileId: string): void {
  localStorage.setItem(PROFILE_KEY, profileId)
}
