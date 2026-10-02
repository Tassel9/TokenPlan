const intentLabels: Record<string, string> = {
  inspection_standard_query: '运维规范查询',
  inspection_task_create: '巡检任务创建',
  inspection_task_update: '巡检计划变更',
  inspection_task_cancel: '巡检任务取消',
  alert_report: '设备异常上报',
  work_order_handling: '工单处理',
  work_order_withdrawal: '工单撤回',
  terminal_access_issue: '智慧路灯终端接入故障',
  terminal_security_request: '智慧路灯终端安全管理',
  operations_permission_change: '设备权限变更',
  facility_troubleshooting: '设备故障排查',
  operations_complaint: '运维投诉',
  operations_feedback: '运维反馈',
}

const agentLabels: Record<string, string> = {
  rag_knowledge: '知识检索',
  business_data_query: '设施数据',
  business_operation: '工单服务',
}

const skillLabels: Record<string, string> = {
  'inspection-standards': '巡检维护规范',
  'work-order-process': '巡检工单流程',
  'work-order-return': '工单撤回规则',
  'streetlight-security': '智慧路灯终端接入安全',
  'facility-troubleshooting': '设备故障排查',
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
