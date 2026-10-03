import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, sendChat } from './tokenplan'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('TokenPlan API client', () => {
  it('sends the complete chat request to the backend', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ response: 'ok' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await sendChat({
      message: '插件报 401，而且账单重复扣款',
      user_id: 'demo-user',
      conv_id: 'conv-1',
    })

    expect(fetchMock).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        message: '插件报 401，而且账单重复扣款',
        user_id: 'demo-user',
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
