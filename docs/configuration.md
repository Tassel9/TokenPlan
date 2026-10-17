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
| ② 本地运行（容器外） | `REDIS_HOST`、`REDIS_PORT`、`CHROMA_HOST`、`CHROMA_PORT`、`CHROMA_PERSIST_DIRECTORY` | Redis 保存短期会话，ChromaDB 保存长期事实；Redis、ChromaDB 的代码默认值是容器内主机名，宿主机运行时需覆盖 |
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

## 5. 短期记忆与会话归档（Redis）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MEMORY_BACKEND` | `redis` | 默认 Redis + ChromaDB；`sqlite` 为显式离线兼容 |
| `REDIS_HOST` | `redis` | 容器内地址；宿主机运行设 localhost |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `REDIS_DB` | `0` | Redis 数据库编号 |
| `REDIS_PASSWORD` | 空 | 与服务端认证配置一致 |
| `SESSION_DB_PATH` | `./data/session/conversations.sqlite3` | 显式 SQLite 离线后端与旧数据迁移源文件 |
| `SESSION_HISTORY_MAX_MESSAGES` | `100` | 每会话保留的原始归档消息上限 |
| `SESSION_HISTORY_TTL_SECONDS` | `604800` | 原始归档与约束变更的逻辑保留期，默认 7 天；普通读取不续期 |
| `SESSION_HISTORY_PAGE_SIZE` | `50` | 历史读取默认条数 |
| `SESSION_HOT_MEMORY_MAX_MESSAGES` | `20` | 摘要持续失败时的近期消息兜底上限；按完整轮次保留，条数包含用户消息和回答 |
| `SHORT_TERM_TOKEN_LIMIT` | `6000` | 约束原文、摘要、近期完整轮次及同会话召回按最终渲染文本计的 Token 预算 |
| `SHORT_TERM_RECENT_TURNS` | `5` | 超限压缩后保留的最近完整对话轮数；后续对话继续追加，普通读取不按五轮截断 |
| `SHORT_TERM_SUMMARY_TOKEN_LIMIT` | `1000` | 注入上下文的摘要子预算，含摘要标题，用同一模型 tokenizer 计数；必须小于总预算 |
| `SHORT_TERM_SUMMARY_MAX_TOKENS` | `2048` | 单次摘要生成的输出上限，包含 Tool JSON 与来源字段，不用于窗口预留 |

运行时从 Redis 同一快照读取当前窗口与摘要，默认保留 24 小时。窗口按最终上下文的 Token 预算保留完整轮次，移出的消息与旧摘要一起生成增量摘要。新摘要和近期窗口通过 Lua CAS 一起更新；发布前校验会话轮次及有效租约，拒绝迟到的旧轮次摘要。

摘要与近期对话统一用模型 tokenizer 计数：超过 6000 Token 才触发压缩，压缩后默认保留最近 5 轮完整对话，较早轮次与旧摘要合并重写；之后新对话继续追加，直到再次超限，不因出现第六轮而压缩或截断。摘要最多 1000 Token，计入总预算。生成时按摘要子预算预留空间，而非预留完整 JSON 输出额度；最近五轮过长时按完整轮次减少保留数量，最新单轮仍完整保留。最终摘要必须比所覆盖的旧内容更短。超限会要求更短摘要并重试，失败保留原始记录。旧库中的超限摘要会重新压缩，修复失败时当前请求使用有界原始归档，避免注入超限摘要。详见[摘要 Token 预算](short-term-summary-token-budget-20261004.md)。

约束原文与撤回记录独立于摘要保存，由本轮提交折叠更新；常见限制使用规则，其他条件复用上下文整理模型提出逐字引用并校验。作用域默认当前任务，明确整个会话有效才提升作用域。当前变更优先于旧摘要。下游按 Token 选择完整问答，不再按角色优先级或字符数裁剪。用户明确回看历史时，在同一 `(user_id, conv_id)` 的有效归档中按关键词选取最多两轮完整原话，预算不足则不追加。保护区本身或最新一轮超出预算时仍保留原文，因此 6000 是正常选择预算，并非整个模型请求的硬上限。详见[短期记忆保真边界](short-term-fidelity-20261005.md)。

每轮通过 Redis Lua 一次提交消息、CaseState、咨询索引、长期更新 Outbox 与显式偏好 Pending；摘要发布校验源轮次、派生视图版本、行实例和有效租约。归档与 CaseState 默认保留 7 天，普通读取不续期。窗口缺失时从仍有效的有界归档恢复。Redis 持久化与迁移、最终一致和异常停机边界见[说明](redis-chroma-memory-20261004.md)。

会话后端保存近期未解决咨询索引，默认随 CaseState 保留 7 天。明确跨会话指向时检索同一用户最近 20 条有效记录，澄清最多展示 5 条，待选列表保存在当前会话中。唯一匹配后恢复咨询对象与用户原话，变化的业务状态需重新查询；普通读取不延长索引 TTL。详见[跨会话承接说明](cross-session-consultations-20261004.md)。

## 6. 对话入口容量保护

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHAT_RATE_LIMIT_ENABLED` | `true` | 请求级限流开关 |
| `CHAT_RATE_LIMIT_REQUESTS` | `30` | 令牌桶容量（同时是单用户突发上限） |
| `CHAT_RATE_LIMIT_WINDOW_SECONDS` | `60` | 令牌补充周期（秒）；补充速率 = REQUESTS / WINDOW_SECONDS |
| `CHAT_TURN_GATE_ENABLED` | `true` | 会话级轮次闸门（防同一会话并发串扰） |
| `CHAT_TURN_LEASE_SECONDS` | `30` | 租约时长 |
| `CHAT_TURN_RENEW_SECONDS` | `10` | 续租间隔 |

## 7. 长期记忆异步写入

会话提交后唤醒进程内后台协程；模型抽取、候选判断和事实写入不在请求路径中等待。用户消息按用户和会话分别累计，默认每 10 轮进行一次批量抽取；`mem0_flow` 候选通过来源校验后，逐条召回相关旧记忆，只做 `ADD/SKIP` 判断。同一作用域内重复已有记忆项直接跳过，不生成确认或关系记录；新事情、新原因或条件独立保存，不同作用域分别保存。普通抽取不改变已有记忆项的单值，明确更正走修改流程。存在比较对象时通常增加一次判断模型调用，超预算分组；精确重复、已知投影值变更或没有比较对象时无需语义调用。规则能明确识别的偏好变更和撤回直接处理，完整原话仍参与批次。Redis Outbox 与会话原子提交，成功发布后清理；少轮 `mem0_flow` 会话默认空闲 120 秒处理，关闭在十秒预算内尝试剩余批次。legacy 少轮会话继续累计，不强制关闭抽取。Redis 来源和完成的判断可供重启重试复用，SQLite 待处理来源和判断缓存仅在进程内；多进程首次竞争可能重复调用模型。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LONG_TERM_MEMORY_ASYNC_ENABLED` | `true` | 置 `false` 关闭长期记忆写入 |
| `LONG_TERM_MEMORY_ENGINE` | `mem0_flow` | 应用默认按主题抽取与召回；`legacy` 回到五字段引擎。直接构造 MemoryManager 的兼容默认值仍为 legacy |
| `LONG_TERM_MEMORY_EXTRACTION_INTERVAL_TURNS` | `10` | 每用户、每会话的批量抽取轮数，最小为 1；有比较对象时另做批量语义比较 |
| `LONG_TERM_MEMORY_IDLE_FLUSH_SECONDS` | `120` | 新引擎少轮会话空闲后提前处理；legacy 不启用。空闲时间从最后任务入队算起 |
| `LONG_TERM_MEMORY_EXTRACTION_TOKEN_LIMIT` | `12000` | 新引擎抽取及每次比较的输入上限；保留完整轮次/比较分组，单条仍超限保留失败来源 |
| `LONG_TERM_MEMORY_CONTEXT_TOKEN_BUDGET` | `800` | 新引擎生成的长期上下文总 Token 上限；不截断单条记忆的归属信息 |
| `LONG_TERM_MEMORY_RECALL_MAX` | `5` | 新引擎最多注入的主题正文数量 |
| `LONG_TERM_MEMORY_FACT_TTL_DAYS` | `180` | 通用用户事实保留天数 |
| `LONG_TERM_MEMORY_PREFERENCE_TTL_DAYS` | `365` | 通用偏好保留天数 |
| `LONG_TERM_MEMORY_PLAN_TTL_DAYS` | `30` | 无原文明写截止日期的用户计划保留天数 |
| `LONG_TERM_MEMORY_ADVICE_TTL_DAYS` | `30` | 独立归属的历史助手建议保留天数 |
| `LONG_TERM_MEMORY_DECISION_TTL_DAYS` | `180` | 项目决策保留天数 |
| `LONG_TERM_MEMORY_CLEANUP_INTERVAL_S` | `300` | 后台物理清理过期事实的间隔，启动时也执行；与抽取轮数独立 |
| `LONG_TERM_MEMORY_MAX_ATTEMPTS` | `3` | 后台任务最大处理尝试次数 |
| `LONG_TERM_MEMORY_RETRY_BACKOFF_S` | `0.5` | 重试退避基数，指数增长 |
| `LONG_TERM_MEMORY_SHUTDOWN_TIMEOUT_S` | `10` | 关闭时等待当前批次的上限，超时取消；Redis 待处理记录保留供下次启动恢复 |
| `LONG_TERM_MEMORY_PENDING_TTL_SECONDS` | `86400` | 无未完成记录时的 Pending 期限；未完成或失败任务继续保留覆盖 |

旧 `RABBITMQ_URL`、`LONG_TERM_MEMORY_QUEUE_ENABLED`、`LONG_TERM_MEMORY_WORKER_ENABLED`、`LONG_TERM_MEMORY_WORKER_PREFETCH` 已不再读取。历史关闭写入配置请改为 `LONG_TERM_MEMORY_ASYNC_ENABLED=false`。

重复表达和普通读取均不续期，新引擎及旧五字段投影都不再生成确认事件。已存在的历史确认仍可读取，兼容其原有效期；本次不执行现有用户数据迁移。后台清理到期旧值并保留不含原值的失效边界，覆盖到期前的迟到来源；到期后用户重新表达可新增。Chroma 物理删除失败时，已发布标记仍隐藏过期值，待清理 ID 保留供后续重试。

新流程把完整用户/助手轮次作为输入，普通操作追加按主题正文；压缩仅触发处理，不以摘要作来源。候选校验后缓存，Chroma/CAS 写入重试不重新抽取。默认十轮、空闲、压缩及“记住”共享任务完成状态。生命周期、内部管理方法、只读迁移和实际离线验证见[新流程说明](long-term-memory-mem0-runtime.md)。旧五个投影字段仍沿用固定 TTL，不受上述通用类型 TTL 配置影响。

## 8. 长期事实与知识库存储（ChromaDB）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_HOST` / `CHROMA_PORT` | `chromadb` / `8000` | 容器内地址；本地用 `localhost` / `8001` |
| `CHROMA_PERSIST_DIRECTORY` | `/app/data/chroma` | 容器内路径；本地用 `./data/chroma` |
| `RAG_CHROMA_COLLECTION_NAME` | `tokenplan_knowledge_base_v2` | 知识库集合名 |
| `RAG_LEXICAL_INDEX_PATH` | 跟随 `CHROMA_PERSIST_DIRECTORY` 下的 `tokenplan_lexical_v2.sqlite3` | 复用旧 FTS5 索引时显式指定 |
| `LONG_TERM_MEMORY_CONTEXT_EXTRACTION_ENABLED` | `true` | legacy 引擎抽取时附带同会话最近用户发言；新引擎附带有限原始消息和相关旧记忆 |
| `LONG_TERM_MEMORY_CONTEXT_TURNS` | `3` | legacy 附带的最大用户发言条数 |
| `LONG_TERM_MEMORY_CONTEXT_MAX_CHARS` | `1200` | legacy 附带上下文的字符预算 |
| `LONG_TERM_MEMORY_FACT_INJECTION` | `recall` | 表达偏好固定注入，其他当前有效事实按当前问题语义召回（余弦距离 ≤0.58，最多三条）；`full` 为显式全量注入兼容模式 |

legacy 抽取上下文只用于消解“它 / 这个 / 刚才说的”类指代：跨轮证据只允许变更（`supersede`）或撤回（`retract`），且变更 / 撤回措辞必须出现在当前消息；助手措辞不能作为用户事实证据。新引擎可保存带独立归属的助手建议，每条新增内容仍须有新原始消息证据。

新引擎使用 `user_memories_v1` 和独立 memory-manifest，旧 `user_facts_v3` 保留为兼容投影。当前问题与有依据的补充 Query（最多三条）召回语义候选，关键词仅辅助排序；距离阈值沿用 ≤0.58，数量与总 Token 预算由上表控制。读取过滤作用域、有效期、Pending 与删除边界。当前账户状态仍由业务查询获取。
长期事实默认使用 `recall`，先归并同用户已发布事实，再仅在当前有效的非全局事实中调用 ChromaDB 向量查询；使用 `user_facts_v3` 集合与默认 BGE 文档向量，Redis 维护可见事件 ID、版本与 Pending。空问题不召回非全局事实，检索失败不降级为全量注入。`MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK` 不会把长期事实切到另一份本地库。SQLite 导入使用[显式迁移命令](redis-chroma-memory-20261004.md)。

## 9. Skills

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SKILL_CATALOG_PATH` | `<仓库>/backend/skills/catalog` | 相对 `backend/core/doctor.py` 解析，与工作目录无关；容器内为 `/app/backend/skills/catalog` |
| `KNOWLEDGE_API_INGEST_AUTHORITY` | `unknown` | 知识入库接口的调用方标识白名单 |

## 10. 意图识别

默认 `IntentRecognizer` 对完整问题一次输出全部当前诉求及各自原文证据，随后按意图分别计算 Embedding / LLM 融合分数并冻结。Supervisor 只消费冻结意图，负责父领域 Agent 选择、依赖编排和汇总，不再次识别或拆分用户诉求。业务标签仍为 13 个；`RoutingIntentRecognizer` 的 `orchestrate` 控制路由只在显式兼容或对照模式中使用。

`SINGLE_INTENT_FAST_PATH_ENABLED` 默认为 `true`，使有限的、已确认的单个知识诉求直接派发父领域 Agent。关闭此项或遇到多意图、个人数据请求及其他非快速路径时，同一冻结意图集合进入 Supervisor 编排，不触发第二次意图识别。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INTENT_EMBEDDING_TOP_K` | `6` | 全标签 Embedding 分数中保留的诊断候选数；不裁剪 LLM 意图树 |
| `INTENT_EMBEDDING_MODEL` / `_REVISION` / `_DEVICE` | `BAAI/bge-base-zh-v1.5` / 固定 revision / 自动 | 意图相似度编码器；每个意图只缓存一个定义向量 |
| `INTENT_EMBEDDING_CACHE_SIZE` | 内置默认 | 编码结果 LRU 大小 |
| `INTENT_EMBEDDING_PRELOAD` | `true` | 启动时预热业务意图定义；显式路由兼容模式还会使用 orchestrate 控制路由向量 |
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
| `AGENT_INITIAL_RETRIEVAL_ENABLED` | `true` | 领域 Agent 首轮自动执行固定混合检索；多子句最多拆成三路 |
| `AGENTIC_RAG_REFLECTION_ENABLED` | `true` | 保留兼容变量名；默认混合检索后由当前领域 Agent 提交证据反思，再决定是否补查 |
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
| `IMAGE_NAME` / `VERSION` / `REGISTRY` | 见 `build-image.sh`、`run-image.sh` |
| `CONFIG_DIR` / `DATA_DIR` / `LOGS_DIR` | 见 `run-image.sh` |


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
docker compose up -d redis chromadb
python backend/cli.py doctor          # 应输出 ok / degraded
python backend/cli.py "我的账单为什么多了 20 元"
```

**B. 全容器**

```bash
cp .env.example .env          # 只填 DEEPSEEK_API_KEY
docker compose up -d
```

**C. 关闭长期记忆写入**

```bash
# .env 追加
LONG_TERM_MEMORY_ASYNC_ENABLED=false
```

**D. 知识库使用本地 ChromaDB（仅本地调试）**

```bash
# .env 设置；知识库在远端不可用时尝试本地持久化目录
CHROMA_HOST=localhost
CHROMA_PERSIST_DIRECTORY=./data/chroma
```
