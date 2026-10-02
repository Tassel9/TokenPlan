# UrbanOps 局部机制评测

当前统一端到端回归入口是 `evaluation/live_eval/run_suite.py`。本目录保留检索、记忆、并发与架构对照等专项实验，用来解释局部机制，不应把单项分数直接表述为生产任务成功率。

## 智慧路灯检索基准

检索脚本统一读取 `evaluation/fixtures/urbanops_agentic_rag_ragas_cases_v1.json`。该夹具包含 60 篇智慧路灯运维文档和 200 个策展合成问题，holdout 为 150 例。

| 脚本 | 用途 |
| --- | --- |
| `bench_retrieval_latency.py` | 测量检索链路延迟与逐例召回 |
| `diagnose_recall_gap.py` | 拆分稠密、词法、融合与重排阶段的召回缺口 |
| `sweep_rerank_candidates.py` | 扫描重排候选数的质量与延迟权衡 |
| `bench_concurrency.py` | 测量不同并发档位的吞吐与尾延迟 |
| `compare_embedding_models.py` | 比较嵌入模型的召回与编码成本 |
| `verify_concurrent_equivalence.py` | 验证并发双路召回与顺序实现的结果一致性 |
| `verify_bf16_rerank.py` | 比较 fp32 与 bf16 重排结果和性能 |

## 记忆与端到端专项

| 脚本 | 用途 |
| --- | --- |
| `evaluate_memory_update_matrix.py` | 验证事实新增、变更、撤回、乱序与幂等 |
| `evaluate_long_term_memory_retrieval.py` | 测量长期事实检索与误召 |
| `evaluate_memory_cross_session.py` | 验证跨会话事实写入、召回和使用 |
| `evaluate_short_term_memory_probe.py` | 验证上下文压缩后的事实保真 |
| `evaluate_end_to_end_tasks.py` | 执行 28 个智慧路灯任务的完整多轮链路 |
| `evaluate_multi_vs_single_agent.py` | 对比 Supervisor 多 Agent 与单 Agent 评测臂 |

## 运行约束

- 离线模型实验需要本机已有 BGE 模型缓存；真实 Judge 或事实抽取需要本地配置的模型 API。
- RabbitMQ 专项默认连接 `amqp://urbanops:urbanops123@127.0.0.1:5672/`，可用 `BENCH_RABBITMQ_URL` 覆盖。
- 结果写入 `evaluation/reports/`。每次报告必须记录当前 fixture SHA 和代码版本，不能复用旧领域结果。
- 当前夹具是合成数据且 `production_evidence=false`；任何百分比都必须注明样本数、重复次数和通过判据。

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
$py = ".\.venv-win\Scripts\python.exe"

& $py evaluation\live_eval\run_suite.py --dry-run --tier smoke
& $py evaluation\benchmarks\evaluate_end_to_end_tasks.py --k 3
```
