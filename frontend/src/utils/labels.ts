const intentLabels: Record<string, string> = {
  subscription_info_query: '套餐咨询',
  subscription_purchase: '订阅开通',
  subscription_change: '套餐变更',
  subscription_cancel: '取消订阅',
  payment_issue: '支付问题',
  invoice_handling: '发票处理',
  refund_handling: '退款处理',
  account_login_issue: '登录问题',
  account_security_request: '账号安全',
  entitlement_change_request: '权益变更',
  technical_troubleshooting: '技术排障',
  service_complaint: '服务投诉',
  service_feedback: '服务反馈',
}

const agentLabels: Record<string, string> = {
  subscription: '套餐与权益 Agent',
  support: '用户支持 Agent',
  general: '通用 Agent',
  billing: '交易与账务 Agent',
  technical: '技术 Agent',
}

export function labelIntent(value: string): string {
  return intentLabels[value] ?? value.replaceAll('_', ' ')
}

export function labelAgent(value: string): string {
  return agentLabels[value] ?? value
}
