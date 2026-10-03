# RAG 检索链路性能与召回报告

> 单一权威口径。本文档取代此前散落的一次性草稿（`.lark-tmp/` 下的临时报告已废弃，不入库）。
> 全部数字可由 `evaluation/benchmarks/` 下的脚本在本机复现，原始结果在 `evaluation/reports/retrieval_optimization/`。

## 1. 结论速览

| 指标 | 数值 | 口径 | 证据 |
| --- | --- | --- | --- |
| 参考文档 Recall@5（端到端链路） | **98.00%** | 冻结 holdout 150 例，`kb.search_async` → 去重 → BGE rerank top-5 | `recall_attribution.json`、`verify_fixed_rag_ids.json` |
| 同上，优化前（Chroma 默认英文嵌入） | 89.33% | 同一数据集、同一代码，唯一变量为嵌入模型 | `latency_bench_fair.json`、`bf16_vs_fp32.json` |
| 历史报告基线（2026-09-01, `fixed_rag` 臂） | 91.33% | 同一数据集，但为旧代码版本（加权融合，非 RRF） | `fair_rrf_bge_full.json` |
| 端到端检索延迟（候选 8） | **mean 83.8 ms / P95 90.3 ms** | 本地 CPU，不含网络、不含答案生成 | `rerank_candidate_sweep.json`、`concurrency_profile_c8.json` |
| 同上（候选 12，优化前） | mean 110.8 ms | 同上 | `rerank_candidate_sweep.json` |
| 同上（历史基线） | mean 218.6 ms | 同上（fp32 重排 + 候选 12） | `latency_bench_fair.json` |
| 单并发吞吐（候选 12 → 8） | 9.46 → **11.97 req/s** | 32 请求/档，本机 CPU | `concurrency_profile.json`、`concurrency_profile_c8.json` |
| 用户画像更新链路（RabbitMQ 异步化收益） | 同一 job：同步执行 **839 ms/轮** → 请求侧入队 **1.36 ms/轮**（3 轮实测：732–947 ms vs 1.35–1.38 ms；入队 P95 1.95 ms），其中 LLM 抽取 828 ms（约 99%） | **实现对照，非改造前后实测差**（旧实现 fire-and-forget，请求路径仅 1.6 µs）：「主链路全程 await 抽取+向量化+落库」vs「publish+confirm 后即返回」；worker 空闲时画像异步生效延迟 665 ms | `rabbitmq_offload.json`、`profile_update_cost.json` |

**一句话**：检索链路把中文嵌入从 Chroma 内置英文模型换成 BGE 中文模型后召回从 89.33% 提升到 98.00%（单变量、同数据集）；配合候选池收敛与 bf16 重排，端到端检索延迟从 218.6 ms 降到 83.8 ms（P95 90.3 ms）。记忆链路上，把单次 839 ms 的画像抽取与向量化交给 RabbitMQ 后台执行后，请求侧只付 1.36 ms 入队开销（真实 broker，3 轮实测；实现对照口径见 §2）。

## 2. 测量口径与边界

**计入延迟的部分**：`hybrid_dense_bm25_rrf`（Chroma 稠密向量召回 + SQLite FTS5/BM25 词法召回 → RRF 融合）+ `rerank_bge`（BGE cross-encoder 重排）。即「检索链路」本身。

**不计入的部分**：
- 网络往返、LLM 答案生成、Supervisor 意图路由与置信度判定（属主管链，另有测量）；
- 服务冷启动：重排模型冷加载约 2.2–2.4 s，测量时已预热排除；
- Reranker 首次调用含懒加载的额外 ~170 ms，已在 300 次采样前完成预热。

**明确不成立的指标口径**：
- **本项目没有 TTFT（首 token 时间）**。`POST /chat` 是非流式接口（`response_model=ChatResponse`，无 SSE），不存在「首 token」概念。相关指标应表述为**主链路响应时间**。
- 优化前的口径混淆：文献中常见的「38.9% 降幅」来自**另一组实验**（首检索前置 A/B：2663.8 ms → 1628.2 ms，同时 recall 90.67% → 96.00%），口径为「端到端含生成」，不可与本文表格中的检索延迟混用。若引用必须注明「首检索前置」限定词。
- **画像更新的「同步内联」对照口径**：`rabbitmq_offload.json` 的 A 臂是「主链路全程 await 抽取+向量化+落库」的实现成本（839 ms/轮），不代表改造前的历史实现——改造前主链路用 `asyncio.create_task` 触发同一份工作（请求路径成本实测仅 **1.6 µs**），不阻塞响应但工作仍在本进程内（不持久化、无重试/背压）。因此 RabbitMQ 的收益应分两部分讲：**工作重量（相对同步实现的 839 ms/轮）+ 持久化/重试/死信 + prefetch 背压**；注意消费者默认与 API **同进程**（`LONG_TERM_MEMORY_WORKER_ENABLED=true`，单容器 uvicorn，compose 无独立 worker），不存在进程隔离，且 worker 的 BGE 嵌入/Chroma 写入是同步调用（`embed_sync`），每条任务在共享事件循环上阻塞约 3–11 ms（p50 3.3 / max 48）。

**硬件边界**：
- 本机 CPU-only（AMD64，AVX512，32 逻辑核），torch 线程数 16；无 GPU。
- 语料为策展合成数据集（CampusCare/TokenPlan 域，60 篇文档、150 例 holdout）。**该规模下的结论不能外推到百万级语料**（见 §6）。
- 单位：平均值为 300 次采样（150 例 × 2 轮）与并发压测 32 请求/档的组合。

## 3. 数据集与血缘

- 冻结数据集：`evaluation/fixtures/agentic_rag_ragas_cases_campuscare_v1.json`
  - `dataset_id = campuscare-agentic-rag-ragas-curated-v1`，200 例（holdout 150 例参与评测）
  - 该文件与 git 提交 `a257a5e` 中的 `evaluation/fixtures/agentic_rag_ragas_cases.json` **字节一致**（文件 sha256 `f9279366…`）
  - 历史报告记录的 `dataset.sha256 = 6179a17c…` 是仓库流水线对 `{documents, cases}` 的**规范化哈希**（`evaluate_agentic_rag_ragas_pipeline._canonical_sha256`），已用同一算法复核一致 —— 即历史基线与本次测量**是同一份数据**。
- 历史基线报告：`evaluation/reports/retrieval_optimization/fair_rrf_bge_full.json`（`campuscare-agentic-rag-ragas-pipeline-report-v3`，`production_evidence=false`），其 `fixed_rag` 臂 Recall@5 = 91.33%。
- 环境：Python 3.9.13，Chroma 0.5.23，sentence-transformers 5.1.2，离线模式（`HF_HUB_OFFLINE=1`）。

## 4. 优化项与前后对比（逐项单变量）

### 4.1 并发双路召回（asyncio.gather）

原先稠密与词法两路是**顺序**执行的。改为 `asyncio.gather(to_thread(dense), to_thread(lexical), return_exceptions=True)`，并做单路失败降级（单路异常 → 降级为单路结果；两路皆失败 → `KnowledgeRetrievalUnavailable`）。

- 等价性：150/150 例输出**逐字节一致**（`verify_concurrent_equivalence.py`）
- 并发时双路重叠 0.76 ms，融合阶段整体 15.3 ms（`latency_bench_bge.json`）
- 说明：融合阶段只占链路 ~10%，此项是**结构性正确**（去掉伪并发）而非主要提速项

### 4.2 中文 BGE 嵌入替换（主要召回提升）

原链路的 Chroma 使用内置 `all-MiniLM-L6-v2`（英文语料训练）处理中文文档，稠密召回 Recall@5 仅 34.3%。（注：该 34.3% 是**旧数据集**下的分路诊断，与下表口径不同，仅用于定位问题。）

同数据集、同代码、仅替换嵌入模型（`bge-small-zh-v1.5`，查询侧加 BGE 指令前缀）：

| 指标 | Chroma 默认（英文） | BGE 中文 | 证据 |
| --- | --- | --- | --- |
| 稠密召回 R@5 | 34.3%（旧集诊断） | **98.67%** | `recall_attribution.json` |
| 词法召回 R@5 | — | 93.67% | 同上 |
| RRF 融合 R@12 | — | 99.33% | 同上 |
| 融合 R@50（上界） | — | 100.00% | 同上 |
| 端到端链路 R@5 | 89.33% | **98.00%** | 同上 |
| 链路平均延迟 | 106.1 ms | 103.5 ms（基本持平） | `latency_bench_fair_optimized.json` / `latency_bench_bge.json` |

- 剩余 2pp 损失定位：150 例中有 **2 例**金标文档在候选 12 内但被重排挤出 top-5；候选 50 内金标缺失数为 **0**。
- 模型尺寸选择：`bge-small-zh-v1.5`（512 维/24M）与 `bge-base-zh-v1.5`（768 维/102M）**召回相同（均为 98.0%）**，但 query 编码 15.6 ms vs 40.4 ms、链路 112.9 ms vs 138.9 ms → 选 small（`embedding_model_compare.json`）。
- 兼容性：BGE 集合名独立为 `tokenplan_knowledge_base_v3`，可通过 `RAG_EMBEDDING_BACKEND=chroma-default` 回退（见 `docs/configuration.md`）。

### 4.3 重排改 bf16 推理

BGE cross-encoder 重排占链路 ~90%（92.7/103.5 ms）。在 AVX512 CPU 上启用 bf16 autocast（能力探测 + 可回退）：

| 精度 | 延迟 mean / P95 | Recall@5 | 证据 |
| --- | --- | --- | --- |
| fp32 | 201.9 / 240.6 ms | 89.33% | `bf16_vs_fp32.json` |
| bf16 | **88.3 / 104.3 ms** | 89.33%（逐例不变） | 同上 |

- 加速 2.29×；受影响的是第 4–5 位的**排序细节**（9 例顺序变化、6 例集合边界变化），**Recall@5 逐例不变**。
- 该对比在换 BGE 前完成，因此召回列显示为 89.33%；bf16 与嵌入模型互不影响。

### 4.4 重排候选数 12 → 8

`AdaptiveRetrievalConfig.rerank_candidate_limit` 默认 12 → 8（重排成本与候选数近似线性）：

| 候选数 | 重排 mean / P95 | 链路 mean | Recall@5 | 证据 |
| --- | --- | --- | --- | --- |
| 12 | 96.1 / 112.5 ms | 110.8 ms | 98.0% | `rerank_candidate_sweep.json` |
| **8** | 69.1 / 75.2 ms | **83.8 ms** | 98.0% | 同上 |
| 6 | 57.8 / 64.4 ms | 72.5 ms | 98.0% | 同上（更激进，未采用） |

- 选 8 为拐点：召回无损失，链路 −24%；取 6 有继续收益但候选池过窄，鲁棒性风险未验证。
- 并发侧收益：单并发吞吐 9.46 → **11.97 req/s**（+26.5%），各级 P50 延迟 −22%~−25%。

### 4.5 用户画像更新异步化（RabbitMQ）

把用户画像抽取与向量写入交给队列后台执行（**同步执行口径**的成本对照）：

- 历史单轮实测（`profile_update_cost.py`，仅抽取+向量化+落库）：mean 812.0 / P50 585.1 / max 1553.5 ms，其中 LLM 抽取 675.0 ms（83%）——**与下方 3 轮口径同量级，两套数字勿混用**
- SQLite 会话存储已替换旧版暂存实现；`bench_rabbitmq_offload.py` 的主链路成本对照需要按新实现重新测量，旧数字不用于当前性能声明。
- 持久化与背压验证：无消费者时消息驻留队列（深度 = 3），消费者启动后 0.26–0.27 s 清空；prefetch=1 串行消费下积压延迟线性累积（6 条积压 → 末条 4.1–5.3 s），即队列承担了流量整形
- 口径提醒：「同步内联」是**实现对照**，不是历史实现——改造前主链路用 `asyncio.create_task` 触发同一份工作（实测请求路径成本 1.6 µs，不阻塞响应，但重活仍在本进程内、不持久化、无重试与背压）；「移出请求路径」成立，但**消费者默认与 API 同进程、无进程隔离**，且 worker 的嵌入段会在共享事件循环上阻塞数毫秒/条
- 幂等性验证：6 条消息 → 4 行（重复消息折叠、敏感字段拒绝、无事实消息空写），事件 ID 幂等生效

## 5. 负结果（已尝试但**不采用**，避免重复踩坑）

| 尝试 | 结果 | 结论 |
| --- | --- | --- |
| int8 量化重排 | 78.6 ms，仅比 bf16 快 13%，但 150 例中 82 例 top-5 集合被扰动 | 收益/风险不匹配，**不采用** |
| 跨请求微批处理（`RerankBatcher`） | CPU 上无吞吐增益（158 批合并 1932 对，均 12.2 对/批，最大 36 对；吞吐与未批处理持平甚至略降） | 默认关闭（`RAG_RERANK_BATCH_WINDOW_MS=0`），保留实现供 GPU 场景 |
| 加深候选池（48 → 80 → 200） | 召回均为 98.0%，链路 136.9 → 94.9 → 94.7 ms | 换 BGE 后池深对召回**无影响**，此前「应加深池」的建议**作废** |
| 并发批内融合 | 各级吞吐稳定在 8–12 req/s，c8/c16 时 P50 从 ~83 ms 升至 661 ms / 1313 ms | 瓶颈是 CPU 重排算力，非调度；**扩容需 GPU 或模型蒸馏** |

`RerankBatcher` 之所以无收益：单机 CPU 上重排已饱和，合并批次只是把同一份算力换了个排队方式；其价值在 GPU 批量推理（batch 利用率）场景。

## 6. 边界与不适用场景

- **不适用百万级语料**：本报告语料 60 篇 / 150 例。稠密召回延迟随语料线性增长（Chroma HNSW 索引），候选池与重排成本的关系在数据量放大后需要重新标定。
- **CPU-only 结论**：bf16 加速依赖 AVX512（代码内有能力探测，不满足自动回退 fp32）；GPU 上应重新测量，int8/批处理的结论可能反转。
- **单机单进程**：并发压测为单进程 asyncio + 线程池，不代表多实例部署下的表现（未测分布式缓存与共享向量库）。
- **召回口径**：Recall@5 基于「参考文档 ID 命中」，不评价答案正确性；RAGAS 端到端评测见 `evaluation/evaluate_agentic_rag_ragas_*.py`。

## 7. 复现步骤

```powershell
# 环境（本机 .venv-win 已装齐依赖；离线运行）
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
Set-Location <repo>

# 1) 召回归因（融合/稠密/词法/链路 各 R@5 + 分类别）
.\.venv-win\Scripts\python.exe evaluation\benchmarks\diagnose_recall_gap.py

# 2) 端到端延迟基准（含逐例 recall）
.\.venv-win\Scripts\python.exe evaluation\benchmarks\bench_retrieval_latency.py

# 3) 重排候选数扫描
.\.venv-win\Scripts\python.exe evaluation\benchmarks\sweep_rerank_candidates.py

# 4) 并发档位压测
.\.venv-win\Scripts\python.exe evaluation\benchmarks\bench_concurrency.py

# 5) 与历史 fair 报告逐例对齐（provenance 校验）
.\.venv-win\Scripts\python.exe evaluation\benchmarks\verify_fixed_rag_ids.py
```

脚本清单与用途见 `evaluation/benchmarks/README.md`；脚本把结果写入 `evaluation/reports/retrieval_optimization/`。

## 8. 相关配置开关

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_EMBEDDING_BACKEND` | `bge` | `bge` \| `chroma-default`（回退英文内置模型） |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文嵌入模型 |
| `RAG_EMBEDDING_DEVICE` | 自动 | 嵌入模型设备 |
| `RAG_RERANKER_DTYPE` | `auto` | `auto` \| `bf16` \| `fp32` |
| `RAG_RERANKER_BATCH_WINDOW_MS` | `0`（关闭） | 跨请求微批窗口 |
| `RAG_RERANKER_MAX_BATCH_PAIRS` | `48` | 微批最大对数 |
| `RAG_RERANK_CANDIDATE_LIMIT` | `8` | 重排候选数 |

完整说明见 `docs/configuration.md`。
