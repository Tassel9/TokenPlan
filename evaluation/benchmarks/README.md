# 检索链路基准脚本

> 当前统一端到端回归入口已迁移到 `evaluation/live_eval/run_suite.py`。本目录保留检索、记忆、
> 性能等局部机制基准，以及迁移前端到端 v1 的复现脚本；局部指标和历史题集结果不能当作
> 当前 UrbanOps 的端到端任务成功率。新口径见 `evaluation/live_eval/README.md`。

本目录是 `docs/rag-retrieval-performance.md` 中全部数字的**可执行证据**。
所有脚本：离线运行（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），使用
`evaluation/fixtures/agentic_rag_ragas_cases_campuscare_v1.json` 的 **holdout 150 例**，
结果写入 `evaluation/reports/retrieval_optimization/`。

> 运行环境：仓库内 `.venv-win`（含 chromadb 0.5.23 / torch 2.8.0+cpu / sentence-transformers 5.1.2）。
> 冷启动会加载 BGE 模型（重排 ~2.2 s、嵌入 ~4 s），已从测量中排除。

## 核心基准（报告主表来源）

| 脚本 | 用途 | 产出 |
| --- | --- | --- |
| `bench_retrieval_latency.py` | 端到端检索延迟 + 逐例召回；两种口径（`chain` = fair 实验的 `fixed_rag` 路径；`production_tool_call` = 生产工具调用路径） | `latency_bench_*.json` |
| `diagnose_recall_gap.py` | 召回归因：融合/稠密/词法/链路 各自 R@5、R@12、R@50，按类别拆分，定位「金标是否根本进不了候选池」 | `recall_attribution.json` |
| `sweep_rerank_candidates.py` | 重排候选数扫描（12/10/8/6）：延迟 vs 召回拐点 | `rerank_candidate_sweep.json` |
| `bench_concurrency.py` | 并发档位压测（1/2/4/8/16）：吞吐 + P50/P95，用不同 query 保持缓存冷态 | `concurrency_profile*.json` |
| `compare_embedding_models.py` | 中文 BGE 模型选型（small vs base）：召回相同下的延迟差 | `embedding_model_compare.json` |
| `bench_profile_update_cost.py` | 画像更新链路的真实成本（含真实 DeepSeek 抽取调用）；**单轮旧口径**，与 `bench_rabbitmq_offload.py` 的 3 轮口径同量级 | `profile_update_cost.json` |
| `bench_rabbitmq_offload.py` | 主链路耗时对照：画像更新「同步内联」vs「真实 RabbitMQ 入队（publish confirm）」+ worker 完成延迟 + 持久化验证（需 RabbitMQ） | `rabbitmq_offload_sqlite.json` |
| `bench_fact_extraction_context.py` | 长期事实抽取的上下文协同 A/B（`off` = 仅当前消息 / `on` = 附同会话最近用户发言），两臂都跑真实 DeepSeek 单次抽取 + 确定性准入；冻结集 46 例 | `extraction_context_ab.json` |

## 等价性 / 精度验证

| 脚本 | 验证内容 | 产出 |
| --- | --- | --- |
| `verify_concurrent_equivalence.py` | 并发双路召回与顺序路径**逐字节等价**，以及真实重叠时间 | 控制台 |
| `verify_bf16_rerank.py` | fp32 vs bf16：分数漂移、top-5 顺序变化、逐例召回、延迟 | `bf16_vs_fp32.json` |
| `verify_fixed_rag_ids.py` | 与历史 fair 报告逐例对齐（provenance 校验：同一数据集、不同代码版本） | `verify_fixed_rag_ids.json` |
| `ab_pool_size.py` | 单变量 A/B：候选池深度（48/80/200）是否影响召回 | `pool_size_ab.json` |

## 探针（回答「为什么」，非报告主表）

| 脚本 | 问题 |
| --- | --- |
| `probe_rerank_budget.py` | 重排算力花在哪：token 形状、线程数、dtype、对数、ONNX |
| `probe_int8_rerank.py` | 动态 int8 量化能否替代 bf16（结论：不能，精度扰动大） |
| `probe_rerank_batch_scaling.py` | 重排是否算力饱和（决定微批处理是否有意义） |
| `probe_embedding_encode_cost.py` | 单次 query 编码成本（冷缓存） |
| `probe_chroma_embedding_function.py` | chromadb 0.5.23 对自定义 embedding_function 的校验行为（换 BGE 的前提验证） |

## 复现全部结果

```powershell
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
Set-Location <repo-root>

$py = ".\.venv-win\Scripts\python.exe"
& $py evaluation\benchmarks\diagnose_recall_gap.py
& $py evaluation\benchmarks\bench_retrieval_latency.py
& $py evaluation\benchmarks\sweep_rerank_candidates.py
& $py evaluation\benchmarks\bench_concurrency.py
& $py evaluation\benchmarks\verify_concurrent_equivalence.py
& $py evaluation\benchmarks\verify_bf16_rerank.py
& $py evaluation\benchmarks\verify_fixed_rag_ids.py

# 需先启动 RabbitMQ：docker compose up -d rabbitmq
& $py evaluation\benchmarks\bench_rabbitmq_offload.py

# 需 .env 中的 DEEPSEEK_API_KEY；无需 Chroma 服务
& $py evaluation\benchmarks\bench_fact_extraction_context.py
```

## 注意

- 本机为 CPU-only（AVX512）；bf16 路径依赖 AVX512 能力探测，不满足时自动回退 fp32，届时延迟数字显著变差。
- 并发档位的绝对吞吐受本机 CPU 限制（约 8–12 req/s），换机器需重测。
- `bench_rabbitmq_offload.py` 需要本机 RabbitMQ（默认 `amqp://tokenplan:tokenplan123@127.0.0.1:5672/`），可用 `BENCH_RABBITMQ_URL` 覆盖；SQLite 会话库写在临时目录。新报告单独保存，不与旧版数据合并。
- `bench_fact_extraction_context.py` 的冻结集为 `evaluation/fixtures/long_term_memory_extraction_v1.json`（直接陈述 / 更新 / 指代 / 撤回 / 无操作线索 / 敏感 / 无事实等 46 例）；报告在 `evaluation/reports/long_term_memory/`，提示词迭代轮次以 `_v2_round1` / `_v2.1_round2` / `_v2.1_round3` 后缀留档，勿混用数字。
- 历史对比数字（91.33%、218.6 ms）来自 git 提交 `a257a5e` 时期的报告与代码口径，引用时请参照
  `docs/rag-retrieval-performance.md` §3 的血缘说明，不要与当前数字混用。

## 记忆评测（2026-09-24，A–D）

| 脚本 | 用途 | 产出 |
| --- | --- | --- |
| `evaluate_memory_update_matrix.py` | **B 更新语义矩阵**（确定性、无 LLM）：新增/变更/撤回/防复活/乱序/幂等/跨键隔离/过期/冲突 | `reports/long_term_memory/update_matrix.json` |
| `evaluate_long_term_memory_retrieval.py` | **C 检索侧**：query→事实 recall@3、误召、`FACT_MAX_DISTANCE` 阈值扫描（真实 BGE + 内嵌 Chroma） | `reports/long_term_memory/retrieval_threshold_sweep.json` |
| `evaluate_short_term_memory_probe.py` | **A 短期探针**：全量历史 vs 压缩视图的保真率与 Token 开销（20 轮合成对话 + 真实压缩链路） | `reports/short_term_memory/probe_fidelity.json` |
| `evaluate_memory_cross_session.py` | **D 端到端跨会话**：no_memory / stale / current 三臂，真实抽取→落库→读取→回答，确定性判分 + 撤回场景判定模型 | `reports/long_term_memory/cross_session_e2e.json` |
| `bench_fact_extraction_context.py` | 抽取侧上下文协同 A/B（写入侧，46 例冻结集） | `reports/long_term_memory/extraction_context_ab.json` |

```powershell
# 记忆评测（B/C/D 用 .venv-win；A 需要 tokenizer 缓存；D 需要 .env 的 DEEPSEEK_API_KEY）
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
$py = ".\.venv-win\Scripts\python.exe"
& $py evaluation\benchmarks\evaluate_memory_update_matrix.py
& $py evaluation\benchmarks\evaluate_long_term_memory_retrieval.py
& E:\anaconda3\envs\my_env\python.exe evaluation\benchmarks\evaluate_short_term_memory_probe.py
& $py evaluation\benchmarks\evaluate_memory_cross_session.py
```

- B/C/D 使用临时目录内嵌 Chroma + 真实 `bge-small-zh-v1.5`，不需要 RabbitMQ / 队列；A 用 SQLite 临时库驱动真实压缩链路。
- A 的对话含**合成填充段**（8 段轮换）用于把热窗口推过 6000 Token 的压缩触发线；探针事实埋在第
  2/4/6/7/11/15/17/18 轮，回答与判分为确定性关键词/长度断言。
- D 的失败会归类为 `write_gap` / `recall_gap` / `use_gap` / `false_claim` / `check_gap`（先看库内已存事实，
  再区分写入、召回与使用缺口；`check_gap` 表示链路完整但判分口径未过）。
- 冻结集：`evaluation/fixtures/long_term_memory_update_matrix_v1.json`、`long_term_memory_retrieval_v1.json`、
  `memory_cross_session_v1.json`、`long_term_memory_extraction_v1.json`。
- 2026-09-24 修复（R1 全量事实注入 + R2d 预算驱动视图）后的对照基线：修复前报告留档为
  `reports/**/*_pre_fix.json`（A 覆盖 7/8、D current 6/9 recall_gap 3）；修复后 A 覆盖 8/8
  （视图 Token 5,672 ≈ -44.3%）、D current 8/9、recall_gap 0（复跑 2 次一致）。
  背景与口径说明见 `deliverables/记忆评测根因分析.md`。

## 端到端任务评测（2026-09-24）

任务级全链路评测：一个评测单元 = 一个多轮会话任务（2~5 轮），走 `ChatService.handle()`
完整链路（记忆读取 → Supervisor 决策 → 多意图调度 → 专业 Agent ReAct → 工具/RAG →
响应护栏 → 记忆写回）。k 次独立重复（不同 user_id 隔离），结果 / 过程 / 成本 / 安全四类
指标同报。

| 文件 | 说明 |
| --- | --- |
| `evaluate_end_to_end_tasks.py` | harness：离线组装（内嵌 Chroma + 临时 SQLite，无容器依赖）、k 次重复、trace 事件与 LLM 用量采集、确定性判分、LLM Judge 集成；`--analyze` 看报告、`--rejudge` 离线按新口径重判 |
| `end_to_end_judge.py` | LLM Judge（rubric 四准则：事实/完整/连续/安全-veto；temperature=0；判分在用量采集之外） |
| `fixtures/end_to_end_tasks_v1.json` | 冻结任务集：28 个（咨询 7 / 多步 6 / 多意图 5 / 记忆 5 / 对抗 5） |

```powershell
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
$py = ".\.venv-win\Scripts\python.exe"
# 全量（28 任务 × k=3 + judge，约 60 分钟）
& $py evaluation\benchmarks\evaluate_end_to_end_tasks.py --k 3
# 单点排查：--only <task_id,...>；小样本：--limit N；跳过 judge：--no-judge
# 报告分析：--analyze <report>；口径修订后离线重判：--rejudge <report> --out <new>
```

口径与首跑结果（2026-09-24，28×3=84 次运行、0 错误）：

- **单次成功率 66.7%；pass@k = 25/28（至少一次成功）；pass^k = 12/28（三次全过，可靠性口径）**
- 单轮端到端延迟 p50 7.1s / p95 10.5s；agent 链路 411 次 LLM 调用、193 万 tokens（judge 不计入）
- **权威报告：`reports/end_to_end/e2e_tasks_v1_rejudged.json`**（在该文件 `meta.criteria_note`
  说明重判来源；原始跑为 `e2e_tasks_v1.json`，两者仅确定性判据词表不同）
- 已识别的系统缺陷（均有 trace 证据）：① Supervisor 多轮决策反复被拒后降级转人工
  （`supervisor_coordination_failed`；触发含空 action、实体不接地、歧义改写单候选、
  同 Agent 一阶段多消息等）；② Agent 结构化决策解析失败仅 repair 一次即降级
  （`invalid_agent_action`）；③ 用户粘贴完整 Key 时未提醒密钥安全（安全缺口，3/3 复现）；
  ④ 域外请求拒绝话术生硬（"无可委派意图"，无礼貌引导）。
- 任务级失败明细与归因见 `deliverables/端到端评测方案.md` §9。
