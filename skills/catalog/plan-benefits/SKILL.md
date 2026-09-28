---
name: plan-benefits
description: >-
  查询 UrbanOps 设备巡检规范、维护周期、检查项目和适用范围。用于公开运维标准咨询；不用于查询实时设备状态或巡检结果。
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# 设备巡检与维护规范

## 核心契约

### 输入约束

- 必需：设施类型或设备编号，以及希望查询的巡检、维护或处置主题。
- 条件不足时只追问会改变检索结论的最少信息；实时状态、最近巡检结果和当前工单属于业务记录。

### 工作流

1. 使用 `knowledge.retrieve` 能力绑定的检索工具，查询对应设施和作业类型的规范。
2. 核对设施类型、区域、作业条件、生效时间与文档版本。
3. 区分公开规范、现场状态和实际任务进度；后两者没有授权业务 Tool 时转人工运维人员或业务平台。

### 输出约束

- 顺序为：直接结论、关键检查项、适用条件、安全要求、未确认信息。
- 不把通用规范推断为现场状态，也不声称巡检或维护操作已经完成。

### 工具与权限

- `knowledge.retrieve` 只能提供带适用范围的规范，不能证明实时设备、巡检或工单状态。
- Skill 命中不扩大 Agent 权限；最终 Tool 调用仍受 Agent 与 Tool 权限交集限制。

## 按需资源

- 需要比较不同设施或工况的规范、处理知识冲突时读取 `references/comparison-boundaries.md`。
- 需要输出紧凑巡检对照表时读取 `assets/plan-comparison-template.md`。
