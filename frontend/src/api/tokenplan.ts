export type JsonRecord = Record<string, unknown>

export type ChatRequest = {
  message: string
  user_id: string
  conv_id?: string
}

export type IntentExecution = {
  intent_id?: string
  intent?: string
  status?: string
  reason_code?: string
  agent_type?: string
  latency_ms?: number
  evidence_count?: number
  selected_skill_ids?: string[]
  tool_names?: string[]
  stage_index?: number
  [key: string]: unknown
}

export type ToolEvent = {
  tool_name?: string
  success?: boolean
  fallback_used?: boolean
  latency_ms?: number
  skill_id?: string
  [key: string]: unknown
}

export type ChatResponse = {
  conv_id: string
  trace_id?: string | null
  response: string
  supervisor: JsonRecord
  agent_type: string
  escalated: boolean
  latency_ms: number
  knowledge_used: boolean
  agent_types: string[]
  status: string
  overall_status: string
  response_action: string
  reason_code: string
  evidence_ids: string[]
  tool_events: ToolEvent[]
  intent_dispatch: JsonRecord
  intent_executions: IntentExecution[]
  intent_result_summary: JsonRecord
  request_control: JsonRecord
  stage_timings_ms: Record<string, number>
  memory_persisted: boolean
  memory_error_code: string
}

export function supervisorIntentLabels(result: ChatResponse): string[] {
  const analysis = isRecord(result.supervisor.analysis) ? result.supervisor.analysis : {}
  const intents = Array.isArray(analysis.intents) ? analysis.intents : []
  return intents.flatMap((item) => {
    if (!isRecord(item) || typeof item.label !== 'string') return []
    return [item.label]
  })
}

export function supervisorRewrite(result: ChatResponse): JsonRecord {
  const analysis = isRecord(result.supervisor.analysis) ? result.supervisor.analysis : {}
  return isRecord(analysis.rewrite) ? analysis.rewrite : {}
}

export type HealthResponse = {
  status: string
  agents?: JsonRecord
  skills?: JsonRecord
  resource_limits?: JsonRecord
}

const configuredBase = String(import.meta.env.VITE_TOKENPLAN_API_BASE_URL ?? '').trim()
const API_BASE_URL = (configuredBase || '/api').replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorDetail(payload: unknown, status: number): string {
  if (isRecord(payload)) {
    if (typeof payload.detail === 'string') return payload.detail
    if (typeof payload.message === 'string') return payload.message
  }
  return `请求失败（HTTP ${status}）`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError('无法连接 TokenPlan 后端，请确认 8000 端口服务已启动。', 0)
  }

  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ApiError(errorDetail(payload, response.status), response.status)
  }
  return payload as T
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

export function sendChat(payload: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })
}
