export interface RunOperation {
  requestId: string
  message: string
  actionId?: string
  harnessProfile: string
  executionMode: 'auto' | 'adaptive'
  runId?: string
}

function key(sessionId: string): string {
  return `lingcart:run-operation:${sessionId}`
}

export function loadRunOperation(sessionId: string): RunOperation | null {
  try {
    const raw = localStorage.getItem(key(sessionId))
    return raw ? JSON.parse(raw) as RunOperation : null
  } catch {
    localStorage.removeItem(key(sessionId))
    return null
  }
}

export function saveRunOperation(sessionId: string, operation: RunOperation): void {
  localStorage.setItem(key(sessionId), JSON.stringify(operation))
}

export function clearRunOperation(sessionId: string): void {
  localStorage.removeItem(key(sessionId))
}
