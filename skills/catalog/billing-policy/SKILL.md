---
name: billing-policy
description: >-
  解释 UrbanOps 巡检任务、异常上报和维修工单的公开流转规则。用于工单流程咨询；不用于核验实时设施或具体工单状态。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# 巡检与工单流转规则

## 核心契约

### 输入约束

- 必需：巡检任务、异常上报、工单创建、派发、转派、催办或关闭主题；必要时补充设施编号、区域、告警等级和时间。
- 禁止索取终端密钥、平台口令和完整接入凭证；明确工单撤回或退回主题优先匹配 `refund-policy`。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对设施类型、区域、告警等级、生效时间和规范版本。
2. 将通用规则与例外分开；资料冲突、过期或适用范围缺失时停止推断。
3. 实时告警、巡检记录和具体工单进度需要授权业务 Tool；工具缺失时转人工运维人员。

### 输出约束

- 顺序为：适用规则、关键条件、例外、下一步。
- 不承诺派单、接单或关闭结果，不用用户描述反推工单平台状态。

### 工具与权限

- `knowledge.retrieve` 不能证明工单已经创建、派发、接单或关闭。
- 按需资源只在本 Skill 激活时可读，最终调用仍受 Tool 权限校验。

## 按需资源

- 判断公开流程与实时业务记录边界时读取 `references/policy-boundaries.md`。
- 需要生成条件清晰的工单规则答复时读取 `assets/billing-response-template.md`。
