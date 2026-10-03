import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatResponse } from './api/tokenplan'

vi.mock('./api/tokenplan', async () => {
  const actual = await vi.importActual<typeof import('./api/tokenplan')>('./api/tokenplan')
  return {
    ...actual,
    getHealth: vi.fn(),
    sendChat: vi.fn(),
  }
})

import App from './App'
import { getHealth, sendChat } from './api/tokenplan'

const responseFixture: ChatResponse = {
  conv_id: 'conv-1',
  trace_id: 'trace-demo-001',
  response: '我会分别处理技术故障和重复扣款问题。',
  supervisor: {
    analysis: {
      rewrite: { status: 'resolved', effective_query: '插件 401 排查；账单重复扣款处理' },
      intents: [
        { intent_id: 'intent-1-technical_troubleshooting', label: 'technical_troubleshooting', supporting_text: ['插件报 401'] },
        { intent_id: 'intent-2-payment_issue', label: 'payment_issue', supporting_text: ['账单重复扣款'] },
      ],
    },
  },
  agent_type: 'technical',
  escalated: false,
  latency_ms: 680,
  knowledge_used: true,
  agent_types: ['technical', 'billing'],
  status: 'COMPLETED',
  overall_status: 'COMPLETED',
  response_action: 'ANSWER',
  reason_code: 'OK',
  evidence_ids: ['kb-1'],
  tool_events: [],
  intent_dispatch: {},
  intent_executions: [
    {
      intent: 'technical_troubleshooting',
      status: 'COMPLETED',
      agent_type: 'technical',
      latency_ms: 320,
      selected_skill_ids: ['technical-troubleshooting'],
    },
    {
      intent: 'payment_issue',
      status: 'COMPLETED',
      agent_type: 'billing',
      latency_ms: 290,
      selected_skill_ids: ['payment-support'],
    },
  ],
  intent_result_summary: {},
  request_control: {},
  stage_timings_ms: { few_shot_retrieval_ms: 12, supervisor_ms: 60, total_ms: 680 },
  memory_persisted: true,
  memory_error_code: '',
}

const mockedGetHealth = vi.mocked(getHealth)
const mockedSendChat = vi.mocked(sendChat)

beforeEach(() => {
  vi.clearAllMocks()
  mockedGetHealth.mockResolvedValue({ status: 'ok' })
  mockedSendChat.mockResolvedValue(responseFixture)
})

describe('TokenPlan workspace', () => {
  it('fills a quick prompt without sending it immediately', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('服务在线')).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: /套餐怎么选/ })[1])

    expect(screen.getByRole('textbox', { name: '输入客服问题' })).toHaveValue(
      '基础版和专业版有什么区别？请结合使用场景帮我选择。',
    )
  })

  it('renders the answer and exposes a readable execution trace', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('服务在线')).toBeInTheDocument()
    const textbox = screen.getByRole('textbox', { name: '输入客服问题' })
    await user.type(textbox, '插件报 401，而且账单重复扣款')
    await user.click(screen.getByRole('button', { name: '发送消息' }))

    expect(await screen.findByText('我会分别处理技术故障和重复扣款问题。')).toBeInTheDocument()
    expect(screen.queryByRole('complementary', { name: '本轮执行详情' })).not.toBeInTheDocument()
    expect(mockedSendChat).toHaveBeenCalledWith(expect.objectContaining({
      message: '插件报 401，而且账单重复扣款',
      user_id: 'demo-user',
      conv_id: expect.any(String),
    }))

    await user.click(screen.getByRole('button', { name: /查看本次处理过程/ }))
    expect(screen.getAllByText('技术排障').length).toBeGreaterThan(0)
    expect(screen.getAllByText('账单 Agent').length).toBeGreaterThan(0)
    expect(screen.getAllByText('technical-troubleshooting、payment-support').length).toBeGreaterThan(0)
    expect(screen.getAllByText('trace-demo-001').length).toBeGreaterThan(0)
  })
})
