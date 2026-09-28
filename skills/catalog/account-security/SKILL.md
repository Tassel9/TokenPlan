---
name: account-security
description: >-
  处理 UrbanOps 终端接入、设备证书、接入凭证、绑定关系和安全策略。用于终端安全指导；不用于直接修改真实设备配置。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# UrbanOps 终端接入与安全

## 核心契约

### 输入约束

- 必需：终端或设备编号、接入问题类型和故障现象；必要时补充错误提示、发生时间、已尝试步骤及网络状态。
- 禁止索取平台口令、完整私钥、设备证书原文或完整接入令牌。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，核对对应终端类型的接入流程、安全要求和人工渠道。
2. 先判断是网络中断、证书异常、凭证失效、网关故障还是平台服务不可用，再给出最短恢复路径。
3. 需要更新证书、修改绑定关系、撤销终端会话或查看真实认证状态时，必须使用授权业务 Tool；当前无相应 Tool 时转人工运维人员。

### 输出约束

- 顺序为：当前判断、安全检查步骤、预期现象、无法自助时的人工渠道。
- 不声称终端已恢复、证书已更新、会话已撤销或设备认证已通过。

### 工具与权限

- `knowledge.retrieve` 只提供公开流程，不能证明实时终端状态。
- Skill 命中不扩大 Agent 权限；未注册或未授权的设备操作 Tool 不会暴露给模型。

## 按需资源

- 需要区分自助恢复、后台操作和服务故障时读取 `references/security-boundaries.md`。
- 需要整理转人工信息时读取 `assets/account-handoff-template.md`。
