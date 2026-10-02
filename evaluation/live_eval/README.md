# UrbanOps Live Eval

这是 UrbanOps 当前统一的端到端评测入口。它参考 TaskMind 的场景化 Live Eval 结构，复用
UrbanOps 已有的真实 `ChatService`、Trace、工具事件和 Token 采集链路。

## 评测契约

一次套件运行按以下顺序执行：

1. 递归加载并严格校验 `scenarios/**/*.yaml`；未知字段、重复 ID 和无行为断言的场景直接失败。
2. 在执行前冻结场景集合、重复次数和 `scenario_digest`，声明预期样本数。
3. 每个 `scenario × run` 创建独立的 Chroma、会话 SQLite 和 Trace SQLite，并重新组装应用服务。
4. 通过真实 `ChatService.handle()` 执行完整多轮会话，保存 `sample.json` 与 `trace.json`。
5. 对答案、意图、Agent、工具、证据、Trace 和安全禁区执行确定性断言；可选启用 LLM Judge 补充语义质量判断。
6. 同时报告样本通过率和稳定通过率。稳定通过要求同一场景的所有重复运行全部通过。
7. Baseline 只允许与相同模型、层级、Judge 模式、重复次数和场景摘要的完整报告比较。

规则断言是行为与安全门禁，LLM Judge 不能覆盖规则失败。所有产物均为受控离线评测证据，
`production_evidence=false`。Judge 默认与运行时使用同一模型提供商，可能存在同源偏差；因此
它只补充事实性、完整性、连续性与安全语义判断，不单独构成发布证据。

## 场景结构

```yaml
schema_version: 1
id: urbanops-inspection-standard-001
name: 泵站巡检规范查询
group: retrieval
tier: smoke
tags: [single_turn, knowledge]
turns:
  - 泵站日常巡检需要记录哪些基础信息？
expect:
  answer:
    contains_any: [设施编号, 点位, 时间]
  route:
    intents_exact: [inspection_standard_query]
    agents_exact: [rag_knowledge]
  tools:
    successful: [knowledge_search]
  evidence:
    min_count: 1
judge:
  goal: 回答巡检记录要求，不得声称已经完成现场巡检。
```

`intents_exact` / `agents_exact` 会阻止“必需路由命中了，但又多派了无关能力”的假通过。
`contains_none`、`must_not_call` 和空的精确路由集合用于安全及越权场景。

## 运行

在仓库根目录使用完整依赖环境：

```powershell
$py = ".\.venv-win\Scripts\python.exe"

# 只校验场景与选样计划，不调用模型
& $py evaluation\live_eval\run_suite.py --dry-run --tier smoke

# 快速套件；每题独立运行 3 次
& $py evaluation\live_eval\run_suite.py --tier smoke --runs 3

# 回归套件并启用语义 Judge
& $py evaluation\live_eval\run_suite.py --tier regression --runs 3 --judge

# 保存基线
& $py evaluation\live_eval\run_suite.py --tier smoke --runs 3 `
  --save-baseline evaluation\live_eval\baselines\deepseek-smoke.json

# 与同口径基线比较
& $py evaluation\live_eval\run_suite.py --tier smoke --runs 3 `
  --baseline evaluation\live_eval\baselines\deepseek-smoke.json
```

默认报告写入 `evaluation/reports/live_eval/<timestamp>/`，运行现场保存在系统临时目录。
用 `--root <dir>` 可以把隔离现场保留到指定父目录。退出码 `1` 表示样本失败、样本不完整
或 Baseline 阻断；排查时可用 `--allow-failures` 只生成证据而不阻断命令。

## 与旧评测的关系

- `evaluation/benchmarks/evaluate_end_to_end_tasks.py` 保留为历史 v1 运行器和兼容适配层；其冻结题集仍是迁移前业务语料，不代表当前 UrbanOps 效果。
- `evaluation/live_eval/` 是当前市政运维场景的统一回归入口。
- 检索微基准、意图专项集和记忆专项实验仍保留在原位置；它们回答局部机制问题，不等同于端到端任务成功率。
