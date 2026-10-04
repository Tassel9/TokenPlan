---
name: technical-troubleshooting
description: >-
  诊断 TokenPlan 插件、IDE、代码补全、仓库索引、模型调用、API 和常见错误码。用于低风险故障排查与转人工信息整理；不用于解释账单结果或承诺故障已经修复。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: support
---

# TokenPlan 技术故障排查

## 核心契约

### 输入约束

- 优先收集错误原文、错误码、IDE/插件版本、模型、网络环境、发生时间和最短复现步骤。
- 信息不足时只追问最能区分原因的字段；禁止索取密码、验证码、API Key、私钥或完整凭据。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，查询与错误码、客户端版本、模型和环境匹配的依据。
2. 先判断服务公告、登录状态、代理网络、插件配置和项目索引状态，再建议重装、清理或重置等高成本操作。
3. 每轮只给少量验证步骤和预期 Observation；根据结果进入下一分支，不重复已失败步骤。
4. 需要后台日志、服务端配置、账号操作或连续排查失败时，保留证据并转人工技术支持。

### 输出约束

- 顺序为：当前判断、验证步骤、预期现象、结果分支、转人工所需信息。
- 没有 Observation 不声称根因或修复完成；扣款、退款和个人额度子诉求交给对应 Skill/Agent。

### 工具与权限

- `knowledge.retrieve` 提供排障依据，降级检索结果不能作为根因证明。
- 按需资源只在本 Skill 激活时可读；需要后台日志或配置写入时转人工。

## 按需资源

- 需要低风险排查顺序和分支条件时读取 `references/diagnostic-branches.md`。
- 需要整理转人工信息时读取 `assets/troubleshooting-handoff-template.md`。
