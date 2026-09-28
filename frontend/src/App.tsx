import {
  AlertCircle,
  ArrowRight,
  BadgeHelp,
  Bot,
  BrainCircuit,
  ChevronRight,
  ClipboardList,
  Database,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  Route,
  Send,
  ShieldCheck,
  Sparkles,
  Wrench,
} from 'lucide-react'
import { type FormEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ApiError, getHealth, sendChat, supervisorIntentLabels, type ChatResponse } from './api/tokenplan'
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
    label: '设备与点位',
    description: '查询设施档案、位置与维护要求',
    prompt: '请查询泵站 P-102 的设备类型、所在点位和日常维护要求。',
    icon: Layers3,
    tone: 'violet',
  },
  {
    label: '巡检与工单',
    description: '跟进巡检记录与维修工单',
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
    label: '设备故障排查',
    description: '根据告警与现象检索排查步骤',
    prompt: '泵站 P-102 持续高温告警，应该按什么顺序排查？',
    icon: Wrench,
    tone: 'blue',
  },
  {
    label: '复合问题演示',
    description: '一次触发多个运维任务协作',
    prompt: '泵站 P-102 高温告警，请先查询最近巡检记录，再给出排查步骤并生成维修工单。',
    icon: BrainCircuit,
    tone: 'rose',
  },
]

const processingSteps = ['识别完整 Query 中的多个意图', 'Supervisor 正在委派能力 Agent', '按需加载运维 Skill 与检索知识', '汇总各任务处理结果']

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
    return localStorage.getItem(USER_ID_KEY) || 'demo-user'
  } catch {
    return 'demo-user'
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
  const [initialConversations] = useState(loadConversations)
  const [conversations, setConversations] = useState<Conversation[]>(initialConversations)
  const [currentId, setCurrentId] = useState<string | undefined>(initialConversations[0]?.id)
  const [input, setInput] = useState('')
  const [userId, setUserId] = useState(loadUserId)
  const [isSending, setIsSending] = useState(false)
  const [processingStep, setProcessingStep] = useState(0)
  const [notice, setNotice] = useState('')
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const currentConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === currentId),
    [conversations, currentId],
  )
  const messages = currentConversation?.messages ?? []

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  useEffect(() => {
    try {
      localStorage.setItem(USER_ID_KEY, userId.trim() || 'demo-user')
    } catch {
      // 用户标识仍保留在本次页面状态中。
    }
  }, [userId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, isSending])

  useEffect(() => {
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
  }, [])

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
        user_id: userId.trim() || 'demo-user',
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

  const statusCopy = backendStatus === 'online' ? '服务在线' : backendStatus === 'offline' ? '后端未连接' : '正在连接'

  return (
    <div className="app-shell">
      {sidebarOpen ? <button type="button" aria-label="关闭侧边栏" className="sidebar-scrim" onClick={() => setSidebarOpen(false)} /> : null}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Sparkles aria-hidden="true" /></div>
          <div>
            <strong>UrbanOps</strong>
            <span>Municipal Operations Agent</span>
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
              <p>多意图识别 · Supervisor 调度 · Agentic RAG</p>
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
              <p className="composer-hint">Enter 发送 · Shift + Enter 换行 · 当前项目未接入真实设备与工单平台</p>
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
      <section className="empty-welcome">
        <div className="empty-welcome-icon"><Bot aria-hidden="true" /></div>
        <h2>你好，我是 UrbanOps</h2>
        <p>直接描述设备、巡检、工单或应急处置问题，也可以在一条消息里同时提出多个诉求。</p>
        <div className="capability-row" aria-label="支持能力">
          <span><Route aria-hidden="true" />13 类业务意图</span>
          <span><BrainCircuit aria-hidden="true" />3 个能力 Agent</span>
          <span><Database aria-hidden="true" />知识与会话记忆</span>
        </div>
      </section>

      <div className="prompt-heading">
        <strong>试试这些问题</strong>
        <span>点击后可以继续编辑</span>
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
