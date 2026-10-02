---
name: facility-troubleshooting
description: >-
  诊断 UrbanOps 管理的泵站、路灯、井盖、传感器、网关和平台接口故障。用于低风险排查与转人工信息整理；不用于承诺现场故障已经修复。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  urbanops-owner-agent: rag_knowledge
---

# UrbanOps 设备故障排查

## 核心契约

### 输入约束

- 优先收集设施编号、告警原文、告警码、传感数据、网络环境、发生时间和最短复现步骤。
- 信息不足时只追问最能区分原因的字段；禁止索取平台口令、完整私钥、设备证书或接入令牌。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，查询与设施类型、告警码、设备版本和工况匹配的依据。
2. 先判断供电、通信、传感器、网关、环境条件和平台服务状态，再建议重启、复位或更换部件等高成本操作。
3. 每轮只给少量验证步骤和预期 Observation；根据结果进入下一分支，不重复已失败步骤。
4. 需要现场检测、平台日志、设备控制或连续排查失败时，保留证据并转人工运维支持。

### 输出约束

- 顺序为：当前判断、验证步骤、预期现象、结果分支、转人工所需信息。
- 没有 Observation 不声称根因或修复完成；实时状态、工单创建和权限变更子诉求交给对应 Skill/Agent。

### 工具与权限

- `knowledge.retrieve` 提供排障依据，降级检索结果不能作为根因证明。
- 按需资源只在本 Skill 激活时可读；需要现场检测、设备控制或配置写入时转人工。

## 按需资源

- 需要低风险排查顺序和分支条件时读取 `references/diagnostic-branches.md`。
- 需要整理转人工信息时读取 `assets/troubleshooting-handoff-template.md`。
