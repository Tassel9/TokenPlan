# 配置参考

`.env.example` 只列**必填项**与**本地运行所需的地址覆盖**；其余全部有代码默认值，列在本文件。共 80+ 个变量，按功能分组。

先跑自检，它会告诉你缺什么、能否降级：

```bash
python backend/cli.py doctor          # 人类可读
python backend/cli.py doctor --json   # 机器可读（verdict: ok / degraded / blocked）
```

## 1. 三层配置模型

| 层 | 变量 | 不设置时会怎样 |
|---|---|---|
| ① 必填 | `DEEPSEEK_API_KEY` | 启动即抛 `RuntimeError: 未设置 DEEPSEEK_API_KEY` |
| ② 本地运行（容器外） | `SESSION_DB_PATH`、`RABBITMQ_URL`、`CHROMA_HOST`、`CHROMA_PORT`、`CHROMA_PERSIST_DIRECTORY` | SQLite 使用本地文件；RabbitMQ、ChromaDB 的代码默认值是容器内主机名，宿主机运行时需覆盖 |
| ③ 可选 | 其余全部 | 使用下表默认值，功能不变 |

> 用 `docker compose` 起应用时，compose 会用 `environment:` 覆盖 ② 里的地址（容器内固定值），所以同一份 `.env` 两种跑法都能用。

## 2. 应用与入口

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | uvicorn 监听地址 |
| `APP_ENV` | 空 | 仅当值等于 `development` 时开启 uvicorn 热重载 |
| `MONITOR_INTERVAL` | `10` | 性能监控采样间隔（秒） |

## 3. 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **必填**；兼容读取 `ANTHROPIC_API_KEY` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/anthropic` | 兼容 `ANTHROPIC_BASE_URL` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 传入已停用的 `deepseek-chat` / `deepseek-reasoner` 会自动回退并告警 |

## 4. 进程级并发闸门

| 变量 | 默认值 |
|---|---|
| `LLM_MAX_CONCURRENCY` | `8` |
| `RETRIEVAL_MAX_CONCURRENCY` | `16` |
| `TOOL_MAX_CONCURRENCY` | `32` |

## 5. 短期记忆与会话归档（SQLite）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SESSION_DB_PATH` | `./data/session/conversations.sqlite3` | 短期会话窗口、增量摘要、有界原始归档、CaseState、轮次与并发提交记录 |
| `SESSION_HISTORY_MAX_MESSAGES` | `100` | 每会话保留的原始归档消息上限 |
| `SESSION_HISTORY_PAGE_SIZE` | `50` | 历史读取默认条数 |
| `SESSION_HOT_MEMORY_MAX_MESSAGES` | `40` | 摘要持续失败时的近期对话硬上限 |
| `SHORT_TERM_TOKEN_LIMIT` | `6000` | 短期记忆按最终渲染文本计的 Token 预算 |
| `SHORT_TERM_RECENT_TURNS` | `5` | 兼容保留：视图按 Token 预算保留未摘要轮次后，该值不再作为硬性轮数上限 |
| `SHORT_TERM_SUMMARY_MAX_TOKENS` | `2048` | 单次增量摘要的输出上限（实测合法摘要约 900~1600 token，过小会截断 Tool Call） |

运行时从 SQLite 读取当前窗口与摘要，默认保留 24 小时。窗口按最终上下文的 Token 预算保留完整轮次，移出的消息与旧摘要一起生成增量摘要。新摘要和近期窗口在同一事务中更新；发布前校验会话轮次及有效租约，拒绝迟到的旧轮次摘要。

每轮在 SQLite 中一次提交用户消息、助手消息和可选 CaseState，再按预算更新短期视图。数据库启用 WAL 和写事务，按 `user_id + conv_id` 隔离会话；归档与 CaseState 默认保留 7 天。窗口过期或缺失后，可从仍有效的有界归档重建；普通读取不会延长 TTL，重建视图会重新设置短期 TTL。进程重启后继续使用同一个 `SESSION_DB_PATH` 即可读取已有窗口、摘要和归档，无需迁移原有 SQLite 数据。SQLite 使用 Python 标准库，不需要独立缓存服务。

## 6. 对话入口容量保护

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHAT_RATE_LIMIT_ENABLED` | `true` | 请求级限流开关 |
| `CHAT_RATE_LIMIT_REQUESTS` | `30` | 令牌桶容量（同时是单用户突发上限） |
| `CHAT_RATE_LIMIT_WINDOW_SECONDS` | `60` | 令牌补充周期（秒）；补充速率 = REQUESTS / WINDOW_SECONDS |
| `CHAT_TURN_GATE_ENABLED` | `true` | 会话级轮次闸门（防同一会话并发串扰） |
| `CHAT_TURN_LEASE_SECONDS` | `30` | 租约时长 |
| `CHAT_TURN_RENEW_SECONDS` | `10` | 续租间隔 |

## 7. 长期记忆写入队列（RabbitMQ）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RABBITMQ_URL` | `amqp://tokenplan:tokenplan123@rabbitmq:5672/` | 容器内地址 |
| `LONG_TERM_MEMORY_QUEUE_ENABLED` | `true` | 置 `false` 只关闭长期记忆写入（在线问答不受影响），也是"不想跑 MQ"的降级开关 |
| `LONG_TERM_MEMORY_WORKER_ENABLED` | `true` | API 与 Worker 暂同进程；拆独立 Worker 后 API 侧置 `false` |
| `LONG_TERM_MEMORY_WORKER_PREFETCH` | `1` | 预取数 |
| `LONG_TERM_MEMORY_MAX_ATTEMPTS` | `3` | 最大投递尝试次数 |
| `LONG_TERM_MEMORY_RETRY_BACKOFF_S` | `0.5` | 重试退避基数 |
| `LONG_TERM_MEMORY_PENDING_TTL_SECONDS` | `86400` | Pending 画像覆盖的安全兜底期限 |

## 8. 长期事实（ChromaDB）与 Embedding

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_HOST` / `CHROMA_PORT` | `chromadb` / `8000` | 容器内地址；本地用 `localhost` / `8001` |
| `CHROMA_PERSIST_DIRECTORY` | `/app/data/chroma` | 容器内路径；本地用 `./data/chroma` |
| `MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK` | `false` | 置 `true` 允许内嵌 ChromaDB 降级（生产默认连接失败即停止启动） |
| `RAG_CHROMA_COLLECTION_NAME` | `tokenplan_knowledge_base_v2` | 知识库集合名 |
| `RAG_LEXICAL_INDEX_PATH` | 跟随 `CHROMA_PERSIST_DIRECTORY` 下的 `tokenplan_lexical_v2.sqlite3` | 复用旧 FTS5 索引时显式指定 |
| `LONG_TERM_MEMORY_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | BGE 模型 |
| `LONG_TERM_MEMORY_EMBEDDING_REVISION` | 固定 revision | 与知识库索引保持一致 |
| `LONG_TERM_MEMORY_EMBEDDING_DEVICE` | 跟随意图通道的 `INTENT_EMBEDDING_DEVICE` | `cpu` / `cuda` |
| `LONG_TERM_MEMORY_CONTEXT_EXTRACTION_ENABLED` | `true` | 抽取时附带同会话最近用户发言用于消解指代；置 `false` 回到仅看当前消息的旧行为 |
| `LONG_TERM_MEMORY_CONTEXT_TURNS` | `3` | 附带的最大用户发言条数 |
| `LONG_TERM_MEMORY_CONTEXT_MAX_CHARS` | `1200` | 附带上下文的字符预算 |
| `LONG_TERM_MEMORY_FACT_INJECTION` | `full` | `full`=读取时注入全部当前有效非全局事实（闭集 ≤3 条，与查询无关）；`recall`=仅距离门控 top-3 语义召回 |

抽取上下文只用于消解“它 / 这个 / 刚才说的”类指代：跨轮证据只允许变更（`supersede`）或撤回（`retract`），且变更 / 撤回措辞必须出现在当前消息；助手措辞永远不能作为事实证据。
默认 `full` 注入模式下非全局事实条数受限（`style.*` 走全局注入），对 Token 成本影响可忽略；`recall` 模式保留给按键数敏感或事实类型扩展后的场景。

## 9. Skills

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SKILL_CATALOG_PATH` | `<仓库>/backend/skills/catalog` | 相对 `backend/core/doctor.py` 解析，与工作目录无关；容器内为 `/app/backend/skills/catalog` |
| `KNOWLEDGE_API_INGEST_AUTHORITY` | `unknown` | 知识入库接口的调用方标识白名单 |

## 10. 意图识别

默认识别器只输出一个业务路由或 `orchestrate` 控制路由。普通业务路由直接进入父领域 Agent；复合请求进入 Supervisor，拆解后按每项原文证据分别计算 Embedding / LLM 融合分数。业务标签仍为 13 个，编排路由单独增加一个定义向量。

`SINGLE_INTENT_FAST_PATH_ENABLED` 默认为 `true`，使已确认的普通业务路由直接派发父领域 Agent；`orchestrate` 始终进入 Supervisor。关闭此项或注入缺少默认父领域 Agent 的旧团队时，普通路由使用保留的 Supervisor 检查路径。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INTENT_EMBEDDING_TOP_K` | `6` | 全标签 Embedding 分数中保留的诊断候选数；不裁剪 LLM 意图树 |
| `INTENT_EMBEDDING_MODEL` / `_REVISION` / `_DEVICE` | `BAAI/bge-base-zh-v1.5` / 固定 revision / 自动 | 意图相似度编码器；每个意图只缓存一个定义向量 |
| `INTENT_EMBEDDING_CACHE_SIZE` | 内置默认 | 编码结果 LRU 大小 |
| `INTENT_EMBEDDING_PRELOAD` | `true` | 启动时预热 13 个业务定义和 1 个 orchestrate 控制路由向量 |
| `INTENT_FUSION_ALPHA` | `0.05` | 应用组合层的 Embedding 权重；`final_score = α × embedding_score + (1-α) × tree_score`，2026-10-04 离线校准，非正确概率；可用环境变量覆盖 |
| `INTENT_CLEAR_THRESHOLD` | `0.70` | 融合分达到该值且 LLM 给出原文证据时冻结执行 |
| `INTENT_LOW_THRESHOLD` | `0.40` | 融合分位于该值与 CLEAR 阈值之间时向用户澄清；更低视为未匹配 |

默认主链路同时启动全标签 Embedding 和 LLM 完整意图树推理，然后只做一次逐标签融合。Embedding 不生成可执行意图，也不裁剪 LLM 的标签空间；LLM 通道失败时系统不会仅凭相似度自动执行。历史 few-shot、Platt 校准和 Jev 模块保留给离线对照评测，不接入默认应用装配。

上下文整理、来源校验与识别现已[拆成独立边界](intent-context-boundaries.md)。没有来源时直接保留原句；少量完整公共 FAQ 模板（套餐价格、退款条件、发票入口等）在没有待澄清事项时也直接保留原句。有指代、省略、待澄清事项或未命中模板时，仍先独立整理并校验，再让两路识别读取同一整理后问题。模板只减少上下文调用，不取代意图识别、来源校验或融合门禁。

拆分前策略使用 `evaluation/fixtures/supervisor_intent_final_v2.json` 的 100 条版本化离线集评测，两次运行的主指标一致：Intent Set Exact Match 为 `94.0%`，Macro-F1 为 `96.5234%`，Micro-F1 为 `97.0874%`。报告保存在 `evaluation/reports/supervisor_intent_current_v2_live.json`，并由 `supervisor_intent_latest_manifest.json` 锁定数据、配置与 SHA256；它不证明拆分后新模型边界的效果。该数据集为合成离线回归集，独立业务复核仍为 `pending`，且 `production_evidence=false`；修改模型输入输出边界、`α`、阈值、意图定义或模型后必须重新运行评测。旧 90 条 few-shot 报告及 manifest 继续作为历史基线保留，不能与当前结果混算。

## 11. RAG 检索与重排

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_ADAPTIVE_SEARCH_ENABLED` | `true` | 自适应检索（高置信度快速返回） |
| `RAG_EMBEDDING_BACKEND` | `bge` | 向量通道编码器：`bge`（中文 BGE）或 `chroma-default`（Chroma 内置英文 MiniLM，旧集合） |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | BGE 模型；改 `bge-base-zh-v1.5` 可与意图识别共用同一编码器（单条编码 5 ms → 23 ms） |
| `RAG_EMBEDDING_DEVICE` / `_CACHE_SIZE` / `_REVISION` | 空 / `512` / 空 | 编码设备、查询向量 LRU 容量、模型 revision（留空按模型默认） |
| `RAG_CHROMA_COLLECTION_NAME` | 按后端自动 | `bge` → `tokenplan_knowledge_base_v3`；`chroma-default` → `..._v2`（维数不同不可共用） |
| `RAG_HYBRID_RRF_K` | `20` | RRF 融合常数 |
| `RAG_RERANK_CANDIDATE_LIMIT` | `8` | 送入重排的候选数；实测 12/8/6 召回同为 0.980，候选减半重排成本线性下降（96→69→58 ms） |
| `RAG_RERANKER_BACKEND` | `bge` | 重排后端 |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | 重排模型 |
| `RAG_RERANKER_DEVICE` / `_BATCH_SIZE` / `_MAX_LENGTH` / `_PRELOAD` | `cpu` / `12` / `512` / `false` | 重排运行参数 |
| `RAG_RERANKER_DTYPE` | `auto` | 重排计算精度：`auto`（AVX512 CPU / bf16 GPU 自动启用 bf16 autocast，其余 fp32）、`bf16`、`fp32` |
| `RAG_RERANKER_BATCH_WINDOW_MS` | `0` | 跨请求批处理窗口；`0` = 关闭。CPU 上实测无端到端收益（算力受限），GPU / 多核机器可设 `4` 启用 |
| `RAG_RERANKER_MAX_BATCH_PAIRS` | `48` | 合并批次的单次前向上限（对数）；仅在批处理开启时生效 |
| `RAG_PIPELINE_CACHE_TTL_S` | `60` | 检索管线缓存 |
| `RAG_REWRITE_CACHE_TTL_S` | `300` | 查询改写缓存 |
| `RAG_FAST_PATH_MIN_SCORE` / `_MARGIN` / `_CHANNEL_SCORE` | `0.78` / `0.12` / `0.35` | 仅给非 RRF 自定义检索器的兼容回退 |
| `RAG_HEADING_LEXICAL_WEIGHT` | `0.50` | 标题字段的词法权重 |
| `RAG_VECTOR_CANDIDATE_MULTIPLIER` / `_MIN` | `4` / `20` | 向量候选数 = max(倍数×top_k, 下限) |
| `AGENT_INITIAL_RETRIEVAL_ENABLED` | `false` | 保留历史调用兼容；领域 Agent 不统一前置单跳检索，已确认的简单公共 FAQ 可按模板前置 FAQ 检索 |
| `AGENTIC_RAG_REFLECTION_ENABLED` | `true` | `agentic_rag` 工具内部检索后的结构化证据判断；FAQ 与单跳路径不强制多跳反思 |
| `AGENTIC_RAG_MAX_SEARCH_CALLS` | `2` | 外层工具调用与 Agentic 工具内部知识检索各自的上限，范围 1–3；内部上限包含首次取证 |

领域 Agent 对命中完整公共 FAQ 模板、且与冻结的单个业务标签一致的单诉求请求，先通过受控 ToolBinding 执行 `faq_search`，再让模型根据证据作答，省去首次工具选择调用。`orchestrate` 请求的子任务全部保留原模型工具选择路径。缺少 FAQ 工具、多个标签、个人记录、复合问题或未命中模板时，仍由模型选择检索工具。预取的 FAQ 也计入预算，保留文档作用域、证据治理和响应护栏；证据不足时允许按原预算补搜。执行元数据记录 `simple-public-faq-v1` / `faq_prefetch`。
Agentic 工具内部另有请求级预算，默认执行一次混合检索，证据不足时最多补搜一次；个人记录查询先调用只读业务工具，按需补充公开规则。
内部仅暴露混合检索与当前用户范围内的只读查询工具，不允许递归 Agentic 调用或写操作。关闭结构化反思也不会绕过预算。
历史 Runtime 显式注入多个初始检索的兼容调用仍按首轮次数加一次缺口补搜计数；当前生产 Agentic 工具使用显式预算覆盖该规则。
默认应用没有真实业务查询后台，需通过 `build_app_services(readonly_business_query=...)` 注入受控只读适配器；未接入时不生成个人状态结论。

## 12. Trace、健康检查与可观测性

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRACE_ENABLED` | `true` | 执行轨迹开关 |
| `TRACE_DB_PATH` | `./data/execution_traces.sqlite3` | 轨迹库路径 |
| `TRACE_RETENTION_DAYS` | `7` | 保留天数 |
| `TRACE_INTERRUPTION_THRESHOLD_SECONDS` | `300` | 判定"疑似中断"的阈值 |
| `TRACE_FINGERPRINT_KEY` | 空 | 载荷指纹 HMAC 盐；对外部署前建议设置 |
| `AGENT_HEALTH_ENABLED` | `true` | 领域 Agent 健康检查 |
| `AGENT_HEALTH_COOLDOWN_SECONDS` | `60` | 熔断冷却时长 |
| `AGENT_HEALTH_HALF_OPEN_MAX_CALLS` | `1` | 半开探测并发 |

## 13. docker compose / 部署脚本专用

这些变量**不被 Python 读取**，只用于 compose 插值。compose 中全部带 `:-` 默认值，因此可以完全不出现在 `.env` 里。

| 变量 | compose 默认值 |
|---|---|
| `RABBITMQ_USER` / `RABBITMQ_PASSWORD` | `tokenplan` / `tokenplan123` |
| `IMAGE_NAME` / `VERSION` / `REGISTRY` | 见 `build-image.sh`、`run-image.sh` |
| `CONFIG_DIR` / `DATA_DIR` / `LOGS_DIR` | 见 `run-image.sh` |

> 如果改了 `RABBITMQ_*`，请同步修改 `RABBITMQ_URL`。

## 14. 遗留变量（可删）

| 变量 | 现状 |
|---|---|
| `APP_NAME` | 已无任何代码读取（历史项目名遗留） |
| `EVAL_BASELINE_PATH` | `docker-compose.yml` 仍注入，但主线代码不读取 |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_OPENAI_BASE_URL` | 仅 `evaluation/evaluate_agentic_rag_ragas_judge.py` 读取，属评测脚本专用 |

## 15. 常见场景配方

**A. 只跑依赖容器 + 本地 Python（最省事）**

```bash
cp .env.example .env          # 填 DEEPSEEK_API_KEY 即可
docker compose up -d chromadb rabbitmq
python backend/cli.py doctor          # 应输出 ok / degraded
python backend/cli.py "我的账单为什么多了 20 元"
```

**B. 全容器**

```bash
cp .env.example .env          # 只填 DEEPSEEK_API_KEY
docker compose up -d
```

**C. 不跑 RabbitMQ（关闭长期记忆写入）**

```bash
# .env 追加
LONG_TERM_MEMORY_QUEUE_ENABLED=false
```

**D. 不跑 ChromaDB（仅本地调试）**

```bash
# .env 追加
MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK=true
```
