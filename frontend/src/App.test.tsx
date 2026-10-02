import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatResponse } from './api/urbanops'

vi.mock('./api/urbanops', async () => {
  const actual = await vi.importActual<typeof import('./api/urbanops')>('./api/urbanops')
  return {
    ...actual,
    getHealth: vi.fn(),
    sendChat: vi.fn(),
  }
})

import App from './App'
import { getHealth, sendChat } from './api/urbanops'

const responseFixture: ChatResponse = {
  conv_id: 'conv-1',
  trace_id: 'trace-local-001',
  response: '我会先排查泵站高温告警，再处理维修工单。',
  supervisor: {
    analysis: {
      rewrite: { status: 'resolved', effective_query: '泵站 P-102 高温告警排查；创建维修工单' },
      intents: [
        { intent_id: 'intent-1-facility_troubleshooting', label: 'facility_troubleshooting', supporting_text: ['泵站 P-102 高温告警'] },
        { intent_id: 'intent-2-work_order_handling', label: 'work_order_handling', supporting_text: ['创建维修工单'] },
      ],
    },
  },
  agent_type: 'rag_knowledge',
  escalated: false,
  latency_ms: 680,
  knowledge_used: true,
  agent_types: ['rag_knowledge', 'business_operation'],
  status: 'COMPLETED',
  overall_status: 'COMPLETED',
  response_action: 'ANSWER',
  reason_code: 'OK',
  evidence_ids: ['kb-1'],
  tool_events: [],
  intent_dispatch: {},
  intent_executions: [
    {
      intent: 'facility_troubleshooting',
      status: 'COMPLETED',
      agent_type: 'rag_knowledge',
      latency_ms: 320,
      selected_skill_ids: ['facility-troubleshooting'],
    },
    {
      intent: 'work_order_handling',
      status: 'COMPLETED',
      agent_type: 'business_operation',
      latency_ms: 290,
      selected_skill_ids: ['work-order-process'],
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
  window.history.replaceState({}, '', '/')
  vi.clearAllMocks()
  mockedGetHealth.mockResolvedValue({ status: 'ok' })
  mockedSendChat.mockResolvedValue(responseFixture)
})

describe('UrbanOps workspace', () => {
  it('fills a quick prompt without sending it immediately', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('服务在线')).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: /设备巡检/ })[1])

    expect(screen.getByRole('textbox', { name: '输入运维问题' })).toHaveValue(
      '请查询泵站 P-102 的设备档案、最近巡检记录和日常维护要求。',
    )
  })

  it('renders the deterministic project demo without calling the backend', async () => {
    window.history.replaceState({}, '', '/?demo')
    render(<App />)

    expect(screen.getByText('演示环境')).toBeInTheDocument()
    expect(screen.getByText('P-102 高温告警处置建议')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /查看本次处理过程/ })).toBeInTheDocument()
    expect(mockedGetHealth).not.toHaveBeenCalled()
  })

  it('renders the answer and exposes a readable execution trace', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('服务在线')).toBeInTheDocument()
    const textbox = screen.getByRole('textbox', { name: '输入运维问题' })
    await user.type(textbox, '泵站 P-102 高温告警，请排查并创建维修工单')
    await user.click(screen.getByRole('button', { name: '分析问题' }))

    expect(await screen.findByText('我会先排查泵站高温告警，再处理维修工单。')).toBeInTheDocument()
    expect(screen.queryByRole('complementary', { name: '本轮执行详情' })).not.toBeInTheDocument()
    expect(mockedSendChat).toHaveBeenCalledWith(expect.objectContaining({
      message: '泵站 P-102 高温告警，请排查并创建维修工单',
      user_id: 'local-user',
      conv_id: expect.any(String),
    }))

    await user.click(screen.getByRole('button', { name: /查看本次处理过程/ }))
    expect(screen.getAllByText('设备故障排查').length).toBeGreaterThan(0)
    expect(screen.getAllByText('工单服务').length).toBeGreaterThan(0)
    expect(screen.getAllByText('设备故障排查、巡检工单流程').length).toBeGreaterThan(0)
    expect(screen.getAllByText('trace-local-001').length).toBeGreaterThan(0)
  })
})
