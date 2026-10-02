import {
  Activity,
  AlertCircle,
  ArrowRight,
  BadgeHelp,
  Bot,
  BrainCircuit,
  Building2,
  ChevronRight,
  CircleCheckBig,
  ClipboardList,
  Database,
  FileSearch,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  Network,
  PanelLeftClose,
  Radar,
  Route,
  Send,
  ShieldCheck,
  Wrench,
} from 'lucide-react'
import { type FormEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ApiError, getHealth, sendChat, supervisorIntentLabels, type ChatResponse } from './api/urbanops'
import { ConversationList, type ConversationSummary } from './components/ConversationList'
import { ExecutionPanel } from './components/ExecutionPanel'
import { labelAgent, labelIntent } from './utils/labels'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  result?: ChatResponse
}

type Conversation = ConversationSummary & {
  messages: ChatMessage[]
}

type BackendStatus = 'checking' | 'online' | 'offline'

const CONVERSATIONS_KEY = 'urbanops_conversations_v1'
const USER_ID_KEY = 'urbanops_user_id'

const quickActions = [
  {
    label: '设备巡检',
    description: '查询设施档案、巡检记录与维护规范',
    prompt: '请查询泵站 P-102 的设备档案、最近巡检记录和日常维护要求。',
    icon: Layers3,
    tone: 'violet',
  },
  {
    label: '工单跟进',
    description: '衔接异常记录、维修申请与处理进度',
    prompt: '路灯 LT-208 连续离线，请查询最近巡检记录并说明如何创建维修工单。',
    icon: ClipboardList,
    tone: 'amber',
  },
  {
    label: '应急处置',
    description: '检索预案、升级条件与协同流程',
    prompt: '主干道出现大面积积水时，应按什么预案处置并通知哪些岗位？',
    icon: ShieldCheck,
    tone: 'emerald',
  },
  {
    label: '故障排查',
    description: '结合告警现象与知识规范生成排查路径',
    prompt: '泵站 P-102 持续高温告警，应该按什么顺序排查？',
    icon: Wrench,
    tone: 'blue',
  },
  {
    label: '复合问题示例',
    description: '一次触发多个运维任务协作',
    prompt: '泵站 P-102 高温告警，请先查询最近巡检记录，再给出排查步骤并生成维修工单。',
    icon: BrainCircuit,
    tone: 'rose',
  },
]

const processingSteps = ['识别完整 Query 中的多个意图', 'Supervisor 正在委派能力 Agent', '按需加载运维 Skill 与检索知识', '汇总各任务处理结果']

function buildDemoConversation(): Conversation {
  const result: ChatResponse = {
    conv_id: 'urbanops-demo',
    trace_id: 'trace-demo-p102-001',
    response: [
      '### P-102 高温告警处置建议',
      '',
      '1. **先核验巡检记录**：最近一次巡检记录显示冷却风道存在积尘，建议优先检查通风口与风机状态。',
      '2. **按顺序排查**：确认温度传感器读数 → 检查润滑与负载 → 检查冷却系统 → 复核控制柜散热。',
      '3. **工单衔接**：已生成维修申请草稿，需由值班人员确认设备编号、风险等级和停机窗口后提交。',
      '',
      '> 当前结论来自巡检记录与维护规范，现场操作仍需遵循安全规程。',
    ].join('\n'),
    supervisor: {
      analysis: {
        rewrite: {
          status: 'resolved',
          effective_query: '查询泵站 P-102 最近巡检记录，检索高温故障排查规范，并准备维修工单申请',
        },
        intents: [
          { intent_id: 'demo-1', label: 'inspection_standard_query' },
          { intent_id: 'demo-2', label: 'facility_troubleshooting' },
          { intent_id: 'demo-3', label: 'work_order_handling' },
        ],
      },
    },
    agent_type: 'rag_knowledge',
    escalated: false,
    latency_ms: 1280,
    knowledge_used: true,
    agent_types: ['business_data_query', 'rag_knowledge', 'business_operation'],
    status: 'COMPLETED',
    overall_status: 'COMPLETED',
    response_action: 'ANSWER',
    reason_code: 'OK',
    evidence_ids: ['inspection-p102-20261001', 'manual-pump-cooling-v3'],
    tool_events: [
      { tool_name: 'business_data_query', success: true, latency_ms: 190 },
      { tool_name: 'knowledge_search', success: true, latency_ms: 260 },
      { tool_name: 'business_operation', success: true, latency_ms: 210 },
    ],
    intent_dispatch: {},
    intent_executions: [
      {
        intent_id: 'demo-1',
        intent: 'inspection_standard_query',
        status: 'COMPLETED',
        agent_type: 'business_data_query',
        latency_ms: 320,
        selected_skill_ids: ['inspection-standards'],
      },
      {
        intent_id: 'demo-2',
        intent: 'facility_troubleshooting',
        status: 'COMPLETED',
        agent_type: 'rag_knowledge',
        latency_ms: 410,
        selected_skill_ids: ['facility-troubleshooting'],
      },
      {
        intent_id: 'demo-3',
        intent: 'work_order_handling',
        status: 'COMPLETED',
        agent_type: 'business_operation',
        latency_ms: 360,
        selected_skill_ids: ['work-order-process'],
      },
    ],
    intent_result_summary: {},
    request_control: {},
    stage_timings_ms: {
      few_shot_retrieval_ms: 45,
      supervisor_ms: 170,
      tool_execution_ms: 660,
      response_guard_ms: 82,
      total_ms: 1280,
    },
    memory_persisted: true,
    memory_error_code: '',
  }

  return {
    id: 'urbanops-demo',
    title: 'P-102 高温告警协同处置',
    updatedAt: new Date().toISOString(),
    messageCount: 2,
    messages: [
      {
        id: 'urbanops-demo-user',
        role: 'user',
        content: '泵站 P-102 出现高温告警，请查询最近巡检记录，给出排查步骤并准备维修工单。',
      },
      {
        id: 'urbanops-demo-assistant',
        role: 'assistant',
        content: result.response,
        result,
      },
    ],
  }
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter((item): item is Conversation => (
      typeof item === 'object'
      && item !== null
      && typeof (item as Conversation).id === 'string'
      && Array.isArray((item as Conversation).messages)
    ))
  } catch {
    return []
  }
}

function loadUserId(): string {
  try {
    return localStorage.getItem(USER_ID_KEY) || 'local-user'
  } catch {
    return 'local-user'
  }
}

function saveConversations(conversations: Conversation[]) {
  try {
    const compact = conversations.slice(0, 20).map((conversation) => ({
      ...conversation,
      messages: conversation.messages.slice(-60),
    }))
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(compact))
  } catch {
    // 浏览器存储不可用时仍允许当前会话继续。
  }
}

function conversationTitle(message: string): string {
  const compact = message.replace(/\s+/g, ' ').trim()
  return compact.length > 24 ? `${compact.slice(0, 24)}…` : compact || '新对话'
}

function newConversation(seed = ''): Conversation {
  const now = new Date().toISOString()
  return {
    id: crypto.randomUUID(),
    title: conversationTitle(seed),
    updatedAt: now,
    messageCount: 0,
    messages: [],
  }
}

function readableError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) return '请求太快了，请稍等片刻再试。'
    if (error.status === 503) return 'UrbanOps 正在启动，请稍后重试。'
    return error.message
  }
  if (error instanceof Error) return error.message
  return '请求失败，请稍后重试。'
}

function App() {
  const [demoMode] = useState(() => new URLSearchParams(window.location.search).has('demo'))
  const [initialConversations] = useState(() => (demoMode ? [buildDemoConversation()] : loadConversations()))
  const [conversations, setConversations] = useState<Conversation[]>(initialConversations)
  const [currentId, setCurrentId] = useState<string | undefined>(initialConversations[0]?.id)
  const [input, setInput] = useState('')
  const [userId, setUserId] = useState(loadUserId)
  const [isSending, setIsSending] = useState(false)
  const [processingStep, setProcessingStep] = useState(0)
  const [notice, setNotice] = useState('')
  const [backendStatus, setBackendStatus] = useState<BackendStatus>(demoMode ? 'online' : 'checking')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const currentConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === currentId),
    [conversations, currentId],
  )
  const messages = currentConversation?.messages ?? []

  useEffect(() => {
    if (!demoMode) saveConversations(conversations)
  }, [conversations, demoMode])

  useEffect(() => {
    try {
      localStorage.setItem(USER_ID_KEY, userId.trim() || 'local-user')
    } catch {
      // 用户标识仍保留在本次页面状态中。
    }
  }, [userId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, isSending])

  useEffect(() => {
    if (demoMode) return
    let active = true
    const controller = new AbortController()

    const check = async () => {
      try {
        const health = await getHealth(controller.signal)
        if (active) setBackendStatus(health.status === 'ok' ? 'online' : 'offline')
      } catch {
        if (active) setBackendStatus('offline')
      }
    }

    void check()
    const timer = window.setInterval(() => void check(), 30_000)
    return () => {
      active = false
      controller.abort()
      window.clearInterval(timer)
    }
  }, [demoMode])

  useEffect(() => {
    if (!isSending) return
    const timer = window.setInterval(() => {
      setProcessingStep((step) => Math.min(step + 1, processingSteps.length - 1))
    }, 1500)
    return () => window.clearInterval(timer)
  }, [isSending])

  function updateConversation(id: string, update: (conversation: Conversation) => Conversation) {
    setConversations((items) => items.map((item) => (item.id === id ? update(item) : item)))
  }

  function startNewConversation() {
    const conversation = newConversation()
    setConversations((items) => [conversation, ...items])
    setCurrentId(conversation.id)
    setInput('')
    setNotice('')
    setSidebarOpen(false)
  }

  function deleteConversation(id: string) {
    const remaining = conversations.filter((item) => item.id !== id)
    setConversations(remaining)
    if (id === currentId) setCurrentId(remaining[0]?.id)
  }

  function pickConversation(id: string) {
    setCurrentId(id)
    setNotice('')
    setSidebarOpen(false)
  }

  async function submitMessage(rawMessage: string) {
    const message = rawMessage.trim()
    if (!message || isSending) return

    let conversationId = currentId
    if (!conversationId) {
      const conversation = newConversation(message)
      conversationId = conversation.id
      setConversations((items) => [conversation, ...items])
      setCurrentId(conversationId)
    }

    const activeId = conversationId
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
    }
    const now = new Date().toISOString()
    updateConversation(activeId, (conversation) => ({
      ...conversation,
      title: conversation.messageCount === 0 ? conversationTitle(message) : conversation.title,
      updatedAt: now,
      messageCount: conversation.messageCount + 1,
      messages: [...conversation.messages, userMessage],
    }))

    setInput('')
    setNotice('')
    setProcessingStep(0)
    setIsSending(true)

    try {
      const result = await sendChat({
        message,
        user_id: userId.trim() || 'local-user',
        conv_id: activeId,
      })
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.response,
        result,
      }
      updateConversation(activeId, (conversation) => ({
        ...conversation,
        updatedAt: new Date().toISOString(),
        messageCount: conversation.messageCount + 1,
        messages: [...conversation.messages, assistantMessage],
      }))
      setBackendStatus('online')
    } catch (error) {
      setNotice(readableError(error))
      if (error instanceof ApiError && (error.status === 0 || error.status === 503)) {
        setBackendStatus('offline')
      }
    } finally {
      setIsSending(false)
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    void submitMessage(input)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submitMessage(input)
    }
  }

  const statusCopy = demoMode ? '演示数据' : backendStatus === 'online' ? '服务在线' : backendStatus === 'offline' ? '后端未连接' : '正在连接'

  return (
    <div className="app-shell">
      {sidebarOpen ? <button type="button" aria-label="关闭侧边栏" className="sidebar-scrim" onClick={() => setSidebarOpen(false)} /> : null}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Building2 aria-hidden="true" /></div>
          <div>
            <strong>UrbanOps</strong>
            <span>市政运维智能体</span>
          </div>
          <button type="button" className="mobile-close" onClick={() => setSidebarOpen(false)} aria-label="关闭侧边栏">
            <PanelLeftClose aria-hidden="true" />
          </button>
        </div>

        <button type="button" className="new-chat-button" onClick={startNewConversation}>
          <MessageSquarePlus aria-hidden="true" />
          新建对话
        </button>

        <section className="sidebar-section quick-prompts">
          <div className="sidebar-heading">常用场景</div>
          {quickActions.slice(0, 4).map((action) => {
            const Icon = action.icon
            return (
              <button type="button" key={action.label} onClick={() => { setInput(action.prompt); setSidebarOpen(false) }}>
                <Icon aria-hidden="true" />
                <span>{action.label}</span>
                <ChevronRight aria-hidden="true" />
              </button>
            )
          })}
        </section>

        <section className="sidebar-section history-section">
          <div className="sidebar-heading">对话历史</div>
          <div className="history-scroll">
            <ConversationList
              conversations={conversations}
              currentId={currentId}
              onSelect={pickConversation}
              onDelete={deleteConversation}
            />
          </div>
        </section>

        <div className="sidebar-footer">
          <label htmlFor="user-id">运维人员标识</label>
          <input id="user-id" value={userId} maxLength={128} onChange={(event) => setUserId(event.target.value)} />
          <p><Database aria-hidden="true" />用于区分会话记忆与运维上下文</p>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div className="header-title">
            <button type="button" className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="打开侧边栏">
              <Menu aria-hidden="true" />
            </button>
            <div>
              <h1>{currentConversation?.title || 'UrbanOps 市政运维工作台'}</h1>
              <p>设备巡检 · 故障排查 · 工单协同</p>
            </div>
          </div>
          <div className={`service-status ${backendStatus}`}>
            <span />
            {statusCopy}
          </div>
        </header>

        {notice ? (
          <div className="notice" role="alert">
            <AlertCircle aria-hidden="true" />
            <span>{notice}</span>
            <button type="button" onClick={() => setNotice('')} aria-label="关闭提示">×</button>
          </div>
        ) : null}

        <div className="conversation-stage">
          <section className="chat-stage">
            <div className="messages-viewport">
              {messages.length === 0 ? (
                <Welcome onPick={(prompt) => setInput(prompt)} />
              ) : (
                <div className="message-list">
                  {messages.map((message) => <MessageBubble key={message.id} message={message} />)}
                  {isSending ? <ProcessingMessage step={processingStep} /> : null}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            <form className="composer-wrap" onSubmit={handleSubmit}>
              <div className="composer">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  maxLength={8000}
                  rows={3}
                  placeholder="描述设备、巡检、工单或应急处置问题…"
                  aria-label="输入运维问题"
                />
                <div className="composer-footer">
                  <span><BrainCircuit aria-hidden="true" />支持在一条消息中提出多个问题</span>
                  <button type="submit" disabled={!input.trim() || isSending} aria-label="发送消息">
                    {isSending ? <LoaderCircle className="spin" aria-hidden="true" /> : <Send aria-hidden="true" />}
                  </button>
                </div>
              </div>
              <p className="composer-hint">Enter 发送 · Shift + Enter 换行 · 重要操作需人工确认 · 全流程可追踪</p>
            </form>
          </section>

        </div>
      </main>
    </div>
  )
}

function Welcome({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="welcome">
      <section className="operations-hero">
        <div className="hero-copy">
          <div className="hero-eyebrow"><Radar aria-hidden="true" />Municipal Operations Copilot</div>
          <h2>市政设施运维协同中枢</h2>
          <p>围绕设备巡检、故障排查和工单跟进，将知识检索、多轮上下文与任务编排整合到同一条可追踪链路。</p>
          <div className="capability-row" aria-label="支持能力">
            <span><Route aria-hidden="true" />多意图识别</span>
            <span><BrainCircuit aria-hidden="true" />Supervisor 编排</span>
            <span><Database aria-hidden="true" />知识与会话记忆</span>
          </div>
        </div>
        <div className="workflow-card" aria-label="Agent 处理链">
          <div className="workflow-card-header">
            <span><Activity aria-hidden="true" />Agent 处理链</span>
            <small><CircleCheckBig aria-hidden="true" />全程可追踪</small>
          </div>
          <ol>
            <li><span>01</span><div><strong>问题识别</strong><small>识别设备、告警与任务意图</small></div></li>
            <li><span>02</span><div><strong>知识检索</strong><small>匹配规范、手册与历史记录</small></div></li>
            <li><span>03</span><div><strong>任务编排</strong><small>协调查询、排障与工单 Agent</small></div></li>
            <li><span>04</span><div><strong>统一处置</strong><small>汇总证据、建议与后续动作</small></div></li>
          </ol>
        </div>
      </section>

      <section className="operations-strip" aria-label="运维场景">
        <div><Layers3 aria-hidden="true" /><span><strong>设备巡检</strong><small>档案、记录与维护规范</small></span></div>
        <div><Wrench aria-hidden="true" /><span><strong>故障排查</strong><small>告警归因与处置路径</small></span></div>
        <div><Network aria-hidden="true" /><span><strong>工单协同</strong><small>申请、跟进与状态衔接</small></span></div>
        <div><FileSearch aria-hidden="true" /><span><strong>规范检索</strong><small>版本、范围与证据校验</small></span></div>
      </section>

      <div className="prompt-heading">
        <strong>开始一个运维任务</strong>
        <span>选择场景，或直接在下方描述问题</span>
      </div>
      <section className="quick-grid">
        {quickActions.map((action) => {
          const Icon = action.icon
          return (
            <button type="button" key={action.label} className={`quick-card ${action.tone}`} onClick={() => onPick(action.prompt)}>
              <span className="quick-icon"><Icon aria-hidden="true" /></span>
              <span className="quick-copy"><strong>{action.label}</strong><small>{action.description}</small></span>
              <ArrowRight className="quick-arrow" aria-hidden="true" />
            </button>
          )
        })}
      </section>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const user = message.role === 'user'
  const result = message.result
  const agents = result?.agent_types.length ? result.agent_types : result?.agent_type ? [result.agent_type] : []

  return (
    <article className={`message-row ${user ? 'user' : 'assistant'}`}>
      {!user ? <div className="message-avatar"><Bot aria-hidden="true" /></div> : null}
      <div className="message-column">
        <div className="message-author">{user ? '你' : 'UrbanOps'}</div>
        <div className="message-bubble">
          {user ? (
            <p>{message.content}</p>
          ) : (
            <div className="assistant-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>
          )}
        </div>
        {result ? (
          <>
            <div className="message-meta">
              {supervisorIntentLabels(result).map((intent) => <span key={intent}><Route aria-hidden="true" />{labelIntent(intent)}</span>)}
              {agents.map((agent) => <span key={agent}><BrainCircuit aria-hidden="true" />{labelAgent(agent)}</span>)}
              {result.knowledge_used ? <span><Database aria-hidden="true" />已检索知识</span> : null}
              {result.escalated ? <span className="warning"><BadgeHelp aria-hidden="true" />需要人工</span> : null}
            </div>
            <ExecutionPanel result={result} />
          </>
        ) : null}
      </div>
    </article>
  )
}

function ProcessingMessage({ step }: { step: number }) {
  return (
    <article className="message-row assistant processing-row" aria-live="polite">
      <div className="message-avatar"><Bot aria-hidden="true" /></div>
      <div className="message-column">
        <div className="message-author">UrbanOps</div>
        <div className="processing-card">
          <LoaderCircle className="spin" aria-hidden="true" />
          <div>
            <strong>正在处理你的请求</strong>
            <span>{processingSteps[step]}</span>
          </div>
          <div className="processing-dots"><i /><i /><i /><i /></div>
        </div>
      </div>
    </article>
  )
}

export default App
