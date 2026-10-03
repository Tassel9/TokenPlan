---
name: plan-benefits
description: >-
  比较 TokenPlan 套餐、价格、模型、Token 额度、团队席位和功能权益等公开信息。用于套餐差异、选型、权益与额度规则咨询；不用于查询个人订阅状态或剩余额度。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# 套餐权益与额度咨询

## 核心契约

### 输入约束

- 必需：套餐咨询目标；做比较时还需要套餐名，以及价格、模型、额度、席位或功能等目标维度。
- 条件不足时只追问会改变检索结论的最少信息；“我的套餐、我的额度”属于个人业务记录。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，查询目标套餐和同一比较维度。
2. 核对地区、币种、计费周期、生效时间与版本；只有证据覆盖相同维度时才横向比较。
3. 区分公开权益、个人可用额度和实际功能开通状态；后两者没有授权业务 Tool 时转人工或官方控制台。

### 输出约束

- 顺序为：直接结论、关键差异、适用条件、额度口径、未确认信息。
- 不把宣传描述推断为个人权益，也不声称升级、降级或功能开通已经生效。

### 工具与权限

- `knowledge.retrieve` 只能提供公开且带适用范围的规则，不能证明个人订阅、额度或工作区状态。
- Skill 命中不扩大 Agent 权限；最终 Tool 调用仍受 Agent 与 Tool 权限交集限制。

## 按需资源

- 需要套餐对比维度、知识冲突处理和正反例时读取 `references/comparison-boundaries.md`。
- 需要输出紧凑套餐对比时读取 `assets/plan-comparison-template.md`。
