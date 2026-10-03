---
name: account-security
description: >-
  处理 TokenPlan 账号的登录、密码重置、绑定邮箱、两步验证、设备会话与第三方账号绑定流程。用于账号安全指导；不用于直接修改账号或认证信息。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# TokenPlan 账号安全

## 核心契约

### 输入约束

- 必需：账号问题类型和故障现象；必要时补充错误提示、发生时间、已尝试步骤及是否仍能使用备用验证方式。
- 禁止索取密码、短信验证码、恢复码、API Key、私钥或完整支付信息。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对对应账号问题的自助入口、安全要求和人工渠道。
2. 先判断是密码错误、账号锁定、两步验证异常、第三方绑定问题还是服务不可用，再给出最短恢复路径。
3. 需要后台解锁、修改绑定信息、撤销设备会话或查看个人认证状态时，必须使用授权业务 Tool；当前无相应 Tool 时转人工。

### 输出约束

- 顺序为：当前判断、自助处理步骤、安全注意事项、无法自助时的人工渠道。
- 不声称账号已解锁、密码已重置、会话已撤销或身份核验已通过。

### 工具与权限

- `knowledge.retrieve` 只提供公开流程，不能证明个人账号状态。
- Skill 命中不扩大 Agent 权限；未注册或未授权的账号操作 Tool 不会暴露给模型。

## 按需资源

- 需要区分自助恢复、后台操作和服务故障时读取 `references/security-boundaries.md`。
- 需要整理转人工信息时读取 `assets/account-handoff-template.md`。
