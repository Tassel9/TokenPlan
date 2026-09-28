import { MessageSquare, Trash2 } from 'lucide-react'

export type ConversationSummary = {
  id: string
  title: string
  updatedAt: string
  messageCount: number
}

type ConversationListProps = {
  conversations: ConversationSummary[]
  currentId?: string
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime()
  if (!Number.isFinite(then)) return ''
  const minutes = Math.max(0, Math.floor((Date.now() - then) / 60_000))
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days} 天前`
  return new Date(iso).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

export function ConversationList({ conversations, currentId, onSelect, onDelete }: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <div className="history-empty">
        <MessageSquare aria-hidden="true" />
        <p>还没有对话</p>
        <span>从一个真实运维问题开始</span>
      </div>
    )
  }

  return (
    <div className="conversation-list">
      {conversations.map((conversation) => (
        <div
          key={conversation.id}
          className={`conversation-item ${conversation.id === currentId ? 'active' : ''}`}
        >
          <button type="button" className="conversation-main" onClick={() => onSelect(conversation.id)}>
            <MessageSquare className="conversation-icon" aria-hidden="true" />
            <span className="conversation-copy">
              <strong>{conversation.title || '新对话'}</strong>
              <small>{relativeTime(conversation.updatedAt)} · {conversation.messageCount} 条</small>
            </span>
          </button>
          <button
            type="button"
            aria-label={`删除对话：${conversation.title}`}
            className="conversation-delete"
            onClick={(event) => {
              event.stopPropagation()
              onDelete(conversation.id)
            }}
          >
            <Trash2 aria-hidden="true" />
          </button>
        </div>
      ))}
    </div>
  )
}
