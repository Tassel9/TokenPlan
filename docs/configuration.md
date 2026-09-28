# 配置参考

`.env.example` 只列**必填项**与**本地运行所需的地址覆盖**；其余全部有代码默认值，列在本文件。共 80+ 个变量，按功能分组。

先跑自检，它会告诉你缺什么、能否降级：

```bash
python -m cli doctor          # 人类可读
python -m cli doctor --json   # 机器可读（verdict: ok / degraded / blocked）
```

## 1. 三层配置模型

| 层 | 变量 | 不设置时会怎样 |
|---|---|---|
| ① 必填 | `DEEPSEEK_API_KEY` | 启动即抛 `RuntimeError: 未设置 DEEPSEEK_API_KEY` |
| ② 本地运行（容器外） | `SESSION_DB_PATH`、`RABBITMQ_URL`、`CHROMA_HOST`、`CHROMA_PORT`、`CHROMA_PERSIST_DIRECTORY` | RabbitMQ、ChromaDB 的代码默认值是容器内主机名，宿主机运行时需覆盖 |
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

## 5. 短期记忆（SQLite）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SESSION_DB_PATH` | `./data/session/conversations.sqlite3` | 会话、摘要、CaseState 和并发提交记录的本地文件 |
| `SESSION_HISTORY_MAX_MESSAGES` | `100` | 每会话保留的原始归档消息上限 |
| `SESSION_HISTORY_PAGE_SIZE` | `50` | 历史读取默认条数 |
| `SESSION_HOT_MEMORY_MAX_MESSAGES` | `40` | 摘要持续失败时的近期对话硬上限 |
| `SHORT_TERM_TOKEN_LIMIT` | `6000` | 短期记忆按最终渲染文本计的 Token 预算 |
| `SHORT_TERM_RECENT_TURNS` | `5` | 兼容保留：视图按 Token 预算保留未摘要轮次后，该值不再作为硬性轮数上限 |
| `SHORT_TERM_SUMMARY_MAX_TOKENS` | `2048` | 单次增量摘要的输出上限（实测合法摘要约 900~1600 token，过小会截断 Tool Call） |

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
| `RAG_CHROMA_COLLECTION_NAME` | `urbanops_knowledge_base_chroma_v1` | 知识库集合名 |
| `RAG_LEXICAL_INDEX_PATH` | 跟随 `CHROMA_PERSIST_DIRECTORY` 下的 `urbanops_lexical_v1.sqlite3` | 复用旧 FTS5 索引时显式指定 |
| `LONG_TERM_MEMORY_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | BGE 模型 |
| `LONG_TERM_MEMORY_EMBEDDING_REVISION` | 固定 revision | 与知识库索引保持一致 |
| `LONG_TERM_MEMORY_EMBEDDING_DEVICE` | 跟随 Supervisor 的 `SUPERVISOR_FEW_SHOT_EMBEDDING_DEVICE` | `cpu` / `cuda` |
| `LONG_TERM_MEMORY_CONTEXT_EXTRACTION_ENABLED` | `true` | 抽取时附带同会话最近用户发言用于消解指代；置 `false` 回到仅看当前消息的旧行为 |
| `LONG_TERM_MEMORY_CONTEXT_TURNS` | `3` | 附带的最大用户发言条数 |
| `LONG_TERM_MEMORY_CONTEXT_MAX_CHARS` | `1200` | 附带上下文的字符预算 |
| `LONG_TERM_MEMORY_FACT_INJECTION` | `full` | `full`=读取时注入全部当前有效非全局事实（闭集 ≤3 条，与查询无关）；`recall`=仅距离门控 top-3 语义召回 |

抽取上下文只用于消解“它 / 这个 / 刚才说的”类指代：跨轮证据只允许变更（`supersede`）或撤回（`retract`），且变更 / 撤回措辞必须出现在当前消息；助手措辞永远不能作为事实证据。
默认 `full` 注入模式下非全局事实条数受限（`style.*` 走全局注入），对 Token 成本影响可忽略；`recall` 模式保留给按键数敏感或事实类型扩展后的场景。

## 9. Skills

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SKILL_CATALOG_PATH` | `<仓库>/skills/catalog` | 相对 `core/doctor.py` 解析，与工作目录无关；容器内为 `/app/skills/catalog` |
| `KNOWLEDGE_API_INGEST_AUTHORITY` | `unknown` | 知识入库接口的调用方标识白名单 |

## 10. Supervisor 意图识别

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SUPERVISOR_FEW_SHOT_PATH` | `evaluation/fixtures/supervisor_few_shots_v1.json` | 正反例素材 |
| `SUPERVISOR_INTENT_CANDIDATE_TOP_N` | 跟随 `SUPERVISOR_FEW_SHOT_TOP_K`（`6`） | Embedding 通道输出的候选数；LLM 通道独立遍历完整意图树，不使用该 Top-N 剪枝 |
| `SUPERVISOR_FEW_SHOT_TOP_K` | `6` | Embedding 通道候选与诊断素材上限 |
| `SUPERVISOR_FEW_SHOT_MAX_CHARS` | `8000` | Embedding 候选携带的诊断素材预算；不注入独立 LLM 意图树通道 |
| `SUPERVISOR_FEW_SHOT_EMBEDDING_MODEL` / `_REVISION` / `_DEVICE` | `BAAI/bge-base-zh-v1.5` / 固定 revision / 自动 | BGE 编码器 |
| `SUPERVISOR_FEW_SHOT_EMBEDDING_CACHE_SIZE` | 内置默认 | 编码结果 LRU 大小 |
| `SUPERVISOR_FEW_SHOT_PRELOAD` | `true` | 启动时预热模型 |
| `SUPERVISOR_INTENT_EMBEDDING_CALIBRATION_SCALE` / `_BIAS` | `1.0` / `0.0` | Embedding 原始分的 Platt 校准参数；默认恒等映射，需在开发校准集拟合后冻结 |
| `SUPERVISOR_INTENT_TREE_CALIBRATION_SCALE` / `_BIAS` | `1.0` / `0.0` | LLM 意图树原始分的 Platt 校准参数；默认恒等映射，需与 Embedding 通道分别拟合 |
| `SUPERVISOR_INTENT_FUSION_ALPHA` | `0.50` | `final_score = α × emb_score + (1-α) × tree_score` 中的 Embedding 权重；必须在校准集上选择后冻结 |
| `SUPERVISOR_INTENT_RECALL_THRESHOLD` | `0.40` | 融合分达到该值且存在 LLM 原文证据 → `CLEAR` |
| `SUPERVISOR_INTENT_RECOMMENDATION_THRESHOLD` | `0.34` | 融合分位于该值与 CLEAR 阈值之间 → `AMBIGUOUS`；更低 → `LOW` |
| `SUPERVISOR_UNMATCHED_HANDOFF_TURNS` | `3` | 连续未匹配轮次达到该值 → 转人工 |
| `SUPERVISOR_INTENT_TOOL_BACKEND` | `disabled` | 默认关闭；`jev` 是保留的对照实验模式，启用后不走默认的 Embedding + LLM 意图树并行融合主链路 |
| `TYPESAFE_API_KEY` | 空 | Jev Tool 凭据；仅在 backend=`jev` 时必填，不写入日志 |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | TypeSafe API 根地址；Tool 调用 `POST /v1/systemone` |
| `SUPERVISOR_JEV_MODEL` | `jev-1.13.0` | 固定 Jev 版本，避免模型别名更新后阈值漂移 |
| `SUPERVISOR_JEV_CANDIDATE_THRESHOLD` | `0.20` | 单标签概率达到该值后进入 Supervisor 可选候选集；上线前必须用独立中文样本校准 |
| `SUPERVISOR_JEV_RECOMMENDATION_THRESHOLD` | `0.80` | 高概率提示阈值；Supervisor 仍需校验范围、原文证据和最小标签集合 |
| `SUPERVISOR_JEV_TIMEOUT_SECONDS` | `10` | Jev 请求超时秒数 |

启用 Jev 前额外安装 `pip install -r requirements/intent-jev.txt`。当前消息、受限最近历史和 CaseState 意图上下文会发送给 TypeSafe API，应按部署环境的数据策略决定是否启用。

默认主链路会同时启动 Embedding 多源打分和 LLM 意图树推理；两路完成后先分别做 Platt 校准，再按逐标签融合分进入 `CLEAR / AMBIGUOUS / LOW`。四个校准参数的默认值只是恒等映射，用于保证链路可运行，不代表已经完成统计校准。当前仓库中的旧意图报告是在改造前生成的，不能直接作为该融合策略的效果结论；修改校准参数、`α` 或两个阈值后必须重新运行独立冻结测试集。

## 11. RAG 检索与重排

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_ADAPTIVE_SEARCH_ENABLED` | `true` | 自适应检索（高置信度快速返回） |
| `RAG_EMBEDDING_BACKEND` | `bge` | 向量通道编码器：`bge`（中文 BGE）或 `chroma-default`（Chroma 内置英文 MiniLM，旧集合） |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | BGE 模型；改 `bge-base-zh-v1.5` 可与意图识别共用同一编码器（单条编码 5 ms → 23 ms） |
| `RAG_EMBEDDING_DEVICE` / `_CACHE_SIZE` / `_REVISION` | 空 / `512` / 空 | 编码设备、查询向量 LRU 容量、模型 revision（留空按模型默认） |
| `RAG_CHROMA_COLLECTION_NAME` | 按后端自动 | `bge` → `urbanops_knowledge_base_v1`；`chroma-default` → `urbanops_knowledge_base_chroma_v1`（维数不同不可共用） |
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
| `AGENT_INITIAL_RETRIEVAL_ENABLED` | `true` | 首检索前置（Runtime 先检索再决策） |
| `AGENTIC_RAG_REFLECTION_ENABLED` | `true` | 检索后的结构化证据判断 |
| `AGENTIC_RAG_MAX_SEARCH_CALLS` | `2` | 每请求最多检索次数 |
| `SINGLE_INTENT_FAST_PATH_ENABLED` | `true` | 单意图知识问题（巡检规范 / 终端接入 / 设备故障排查）跳过 Supervisor 的派发+收口两次 LLM 规划，直接委派知识型 Agent；`false` 回退全量编排 |

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
| `IMAGE_NAME` / `VERSION` / `REGISTRY` | 见 `scripts/build-image.sh`、`scripts/run-image.sh` |
| `CONFIG_DIR` / `DATA_DIR` / `LOGS_DIR` | 见 `scripts/run-image.sh` |

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
python -m cli doctor          # 应输出 ok / degraded
python -m cli "泵站 3 号泵出现高温告警，应该怎么排查"
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
