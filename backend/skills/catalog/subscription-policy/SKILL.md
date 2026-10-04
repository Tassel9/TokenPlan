---
name: subscription-policy
description: 解释 TokenPlan 购买、开通、套餐变更、退订和自动续费流程，只提供咨询与官方自助指引。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: subscription
---

# 订阅办理流程咨询

## 核心契约

- 核对套餐、计费周期、购买渠道和规则生效时间；条件不足时只追问影响答案的字段。
- FAQ 使用简单 RAG；单跳规则使用混合检索加重排；涉及多份政策、前后条件或证据缺口时使用 Agentic RAG。
- 个人现有套餐与权益通过 Agentic RAG 调用只读业务查询核验，公开规则不能证明个人状态。
- 不购买、变更或取消订阅。用户要求代操作时先回答可回答的咨询，再询问是否需要转人工。
- 输出适用规则、关键条件、官方自助入口和待确认事项；不承诺办理成功。
