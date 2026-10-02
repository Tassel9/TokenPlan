---
name: work-order-return
description: >-
  解释 UrbanOps 工单撤回、退回和驳回的条件、材料、时限与例外。用于工单回退规则咨询；不用于核验或修改具体工单状态。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  urbanops-owner-agent: rag_knowledge
---

# 工单撤回与退回规则

## 核心契约

### 输入约束

- 必需：工单撤回、退回或驳回诉求；规则结论会受影响时，再追问工单阶段、设施类型、区域、提交时间和原因。
- 禁止索取终端密钥、平台口令或完整接入凭证；用户自述只能作为检索条件。
- 先区分公开规则、操作资格、提交操作和工单进度。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对生效时间、工单阶段、设施类型和区域。
2. 资料冲突、过期或未覆盖当前场景时停止推断，明确当前无法确认。
3. 操作资格、提交和进度只能使用授权业务 Tool；当前没有对应 Tool 时转人工运维人员。

### 输出约束

- 顺序为：规则结论、适用条件、所需信息与时限、例外、下一步。
- 不承诺“肯定能撤回”“工单已经退回”、审批结果或具体完成时间。

### 工具与权限

- `knowledge.retrieve` 只能证明公开规则，不能证明操作资格或实时工单状态。
- Skill 命中不扩大 Agent 权限；缺少业务 Tool 时安全转人工。

## 按需资源

- 判断工单回退诉求边界时读取 `references/request-boundaries.md`。
- 组织最终规则答复时读取 `assets/work-order-return-template.md`。
