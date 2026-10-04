---
name: billing-policy
description: >-
  解释 TokenPlan 购买、套餐变更、续费、扣款、支付、发票和账单的公开政策。用于账单规则咨询；不用于核验个人订单、支付记录或发票状态。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: billing
---

# 订阅账单规则咨询

## 核心契约

### 输入约束

- 必需：购买、套餐变更、续费、扣款、支付、发票或账单主题；必要时补充套餐、地区、币种、渠道和时间。
- 禁止索取完整卡号、支付密码和验证码；明确退款主题优先匹配 `refund-policy`。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对生效时间、地区、币种、渠道和套餐。
2. 将通用规则与例外分开；资料冲突、过期或适用范围缺失时停止推断。
3. 个人账单、重复扣款、支付失败或发票进度需要授权业务 Tool；工具缺失时转人工。

### 输出约束

- 顺序为：适用规则、关键条件、例外、下一步。
- 不承诺审核或到账结果，不用用户描述反推后台账单状态。

### 工具与权限

- `knowledge.retrieve` 不能证明已经支付、扣款成功或发票已经开具。
- 按需资源只在本 Skill 激活时可读，最终调用仍受 Tool 权限校验。

## 按需资源

- 判断公开政策与个人记录边界时读取 `references/policy-boundaries.md`。
- 需要生成条件清晰的账单规则答复时读取 `assets/billing-response-template.md`。
