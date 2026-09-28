const intentLabels: Record<string, string> = {
  subscription_info_query: '运维规范查询',
  subscription_purchase: '巡检任务创建',
  subscription_change: '巡检计划变更',
  subscription_cancel: '巡检任务取消',
  payment_issue: '设备异常上报',
  invoice_handling: '工单处理',
  refund_handling: '工单撤回',
  account_login_issue: '终端接入故障',
  account_security_request: '终端安全管理',
  entitlement_change_request: '设备权限变更',
  technical_troubleshooting: '设备故障排查',
  service_complaint: '运维投诉',
  service_feedback: '运维反馈',
}

const agentLabels: Record<string, string> = {
  general: '通用 Agent',
  billing: '工单 Agent',
  technical: '故障诊断 Agent',
  rag_knowledge: '知识检索 Agent',
  business_data_query: '设施数据查询 Agent',
  business_operation: '工单操作 Agent',
}

const skillLabels: Record<string, string> = {
  'plan-benefits': '巡检维护规范',
  'billing-policy': '巡检工单流程',
  'refund-policy': '工单撤回规则',
  'account-security': '终端接入安全',
  'technical-troubleshooting': '设备故障排查',
}

export function labelIntent(value: string): string {
  return intentLabels[value] ?? value.replaceAll('_', ' ')
}

export function labelAgent(value: string): string {
  return agentLabels[value] ?? value
}

export function labelSkill(value: string): string {
  return skillLabels[value] ?? value
}
