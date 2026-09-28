import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, sendChat } from './urbanops'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('UrbanOps API client', () => {
  it('sends the complete chat request to the backend', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ response: 'ok' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await sendChat({
      message: '泵站 P-102 高温告警，请排查并创建维修工单',
      user_id: 'local-user',
      conv_id: 'conv-1',
    })

    expect(fetchMock).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        message: '泵站 P-102 高温告警，请排查并创建维修工单',
        user_id: 'local-user',
        conv_id: 'conv-1',
      }),
    }))
  })

  it('keeps backend error details for the UI', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '请求频率过高' }), {
      status: 429,
      headers: { 'Content-Type': 'application/json' },
    })))

    await expect(getHealth()).rejects.toEqual(new ApiError('请求频率过高', 429))
  })
})
