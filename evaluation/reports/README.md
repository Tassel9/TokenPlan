# TokenPlan 评测报告目录

当前默认意图评测输入为 TokenPlan 的 AI Coding 订阅客服场景，覆盖套餐与权益、购买与变更、账单与发票、退款、账户安全、技术排障及服务反馈等 13 类意图。

新报告应由 `evaluation/live_eval/` 或相应 benchmark 基于当前冻结夹具生成。生成物默认仅作受控离线实验记录；只有同时保留数据集版本、运行配置、重复次数和原始结果时，才可作为可复核基线。历史报告不得被表述为线上生产效果。

2026-10-04 简单 FAQ 调用优化对照保存在 `llm_call_fast_paths_20261004/`：专项 8 个任务各运行两次，调用 68→54；原 28 个任务本轮严格通过 14/28，上轮为 15/28。局部调用节省与整体质量分别报告，开发中失败的版本也保留，详见[调用优化评测](../../docs/llm-call-fast-paths-evaluation-20261004.md)。

2026-10-04 后续上下文、引用、融合校准和密钥处理修复保存在 `context_repairs_20261004/`。原 28 个任务保持 v1 判分；专项另有行为 v2 和新问法复测，不混入原集合。所有中间失败版本保留，详见[修复评测](../../docs/context-repairs-evaluation-20261004.md)。

## 历史意图识别基线（2026-10-03）

2026-10-04 默认识别器输出单个业务路由或 `orchestrate`。当前新旧模型与端到端对照保存在 `orchestrate_routing_20261004/`，口径和结论见 [评测记录](../../docs/orchestrate-routing-evaluation-20261004.md)。以下报告对应历史多标签协议；原样本、金标和报告保持冻结。

咨询职责的测试预期另存于 `evaluation/fixtures/intent_natural_consultation_v2.json`：代操作请求先询问是否转人工，混合请求先回答咨询再询问；不期待实际购买、退款或账户修改。该 100 条行为金标仍未完整跑模型，不能用本次语义标签评测替代；端到端行为另外使用冻结的 28 个任务与独立专项诊断。

- 数据集：`evaluation/fixtures/supervisor_intent_final_v2.json`，100 条版本化合成离线样例。
- 策略：`intent-recognizer-v4-simple-fusion`，BGE 定义向量权重 `0.10`，意图树权重 `0.90`，确认/低置信阈值为 `0.70/0.40`。
- 两次运行主指标一致：Intent Set Exact Match `94.0%`、Macro-F1 `96.5234%`、Micro-F1 `97.0874%`。
- 当前报告：`supervisor_intent_current_v2_live.json`；首轮复现文件：`supervisor_intent_current_v2_run1.json`；哈希与运行参数由 `supervisor_intent_latest_manifest.json` 锁定。
- 边界：`production_evidence=false`，独立业务标签复核仍为 `pending`；原始报告默认被 `.gitignore` 排除，只作为本地受控实验记录。
