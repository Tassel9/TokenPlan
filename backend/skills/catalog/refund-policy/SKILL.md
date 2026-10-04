---
name: refund-policy
description: >-
  解释 TokenPlan 的公开退款条件、材料、时限和例外。用于退款规则咨询；不用于核验个人退款资格、提交退款或查询退款进度。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: billing
---

# 退款政策咨询

## 核心契约

### 输入约束

- 必需：退款诉求；规则结论会受影响时，再追问套餐、购买渠道、地区、购买时间或退款原因。
- 禁止索取完整卡号、支付密码、验证码等敏感信息；用户自述只能作为检索条件。
- 先区分公开规则、个人资格、提交退款和退款进度。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对生效时间、渠道、套餐和地区。
2. 资料冲突、过期或未覆盖当前场景时停止推断，明确当前无法确认。
3. 个人资格、提交和进度只能使用授权业务 Tool；当前没有对应 Tool 时转人工。

### 输出约束

- 顺序为：政策结论、适用条件、材料与时限、例外、下一步。
- 不承诺“肯定可退”“已经退款”、退款金额、审核结果或具体到账日期。

### 工具与权限

- `knowledge.retrieve` 只能证明公开政策，不能证明个人资格或业务状态。
- Skill 命中不扩大 Agent 权限；缺少业务 Tool 时安全转人工。

## 按需资源

- 判断退款诉求边界时读取 `references/request-boundaries.md`。
- 组织最终政策答复时读取 `assets/refund-response-template.md`。
