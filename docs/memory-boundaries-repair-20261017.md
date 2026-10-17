# 记忆写入、作用域和执行上下文修复

## 当前行为

- 相关长期记忆进入 `AgentContextSnapshot.memory_context`，并在实际业务 Agent 的执行器上下文中序列化。内容标记为历史背景，不是用户授权、已核验业务事实或已完成操作。
- 多任务请求分别按委派任务的问题检索记忆。没有检索回调的直接编排调用，只允许单个请求项使用预加载记忆，不向多任务广播整段画像。短期原话继续通过任务绑定的引用、实体和约束传递，不恢复全量历史下发。
- 先提取本轮明确的 workspace 标识，再选择记忆范围；后台记录绑定当前轮次确定的范围。批次按用户、会话、workspace 隔离，直接提交跨 workspace 批次会被拒绝，辅助原话不跨 workspace 引用。
- 固定字段的 Pending 和快捷修改共用正向语义检查。否定、他人情况、假设和疑问不能仅凭出现某个值就覆盖用户设置。
- “以后不要年付”通过语义抽取保存否定偏好正文，不推断月付；成功保存后，撤回已有且与它矛盾的正向年付设置。之后用户明确改成年付时，撤回匹配的旧否定偏好。不同作用域不互相更正。
- “我现在只用 macOS 开发”等明确本人新状态可以走受控更正；“也用”仍是补充。普通抽取仍遵守 ADD/SKIP，不采用最新陈述无条件覆盖。
- 操作条件使用独立的 `action_permission:动作` 键。旧通用键在读取更新时按原话拆分并保留来源；修改退款条件不会删除取消订阅条件。
- 新会话消息使用带时区的 UTC 时间；读取旧的无时区历史时，沿用既有 `_as_utc` 本地时间转换规则，不改写已存原话，避免辅助证据读取使抽取任务失败。
- 明确的“年付也可以了”“预算不限了”等放宽表达可撤回对应旧限制，不因此创建正向偏好；疑问、假设不会放宽条件。约束引用仍保留用户原文，执行提示明确每条只对其 key 指定的条件生效。

## 验证

`tests/test_memory_revision_boundaries.py` 覆盖真实业务 Agent 执行器收到的上下文、错误否定/主体/假设写入、否定偏好的保存与再次修改、自然语言更正、独立权限更新、放宽条件、首次指定与切换 workspace、后台批次隔离。多任务的正向记忆传递和隐私隔离由 `tests/test_intent_orchestrator.py` 同时验证。

2026-10-17 公开工作区完整测试：**928 passed、1 skipped（47.25 秒）**。跳过项要求本地评测报告，而这些报告按公开仓库策略不上传。新增边界文件包含 16 项测试，另在既有编排测试中验证逐任务记忆传递、无召回回调时的隔离和快照不可变性。54 个新增或修改的 Python 文件通过语法与 Ruff 关键错误检查（E9/F63/F7/F82），差异空白检查通过。Compose 使用示例环境配置校验通过，未连接真实用户服务。

复跑命令（仓库根目录，先安装应用和测试依赖）：

```powershell
.venv-win/Scripts/python.exe -m pip install -r requirements/base.txt -r requirements-test.txt
.venv-win/Scripts/python.exe -B -m pytest -q -rs -p no:cacheprovider tests
```

此前私有工作区的 19 个相关测试文件为 361 passed；该数字不作为本次公开提交的验证结果。公开集成还检查了暂存后源码的语法，避免仅验证工作区而遗漏分块暂存的位置错误。

测试使用模拟模型/向量、临时 Chroma 与 fakeredis，不访问现有用户记忆库，不构成线上模型准确率证据。真实模型的复杂否定、多主体及省略表达仍需要单独评测。未执行真实 Redis 断电恢复或线上灰度，未迁移已有用户数据。

## 公开仓库同步范围

公开 main 原为 SQLite 会话与 RabbitMQ 队列。本次按用户确认，补齐 Redis 会话、Chroma 主题记忆、进程内后台更新、持久 Outbox、短期约束、任务上下文快照及必要迁移工具，同时更新依赖、Compose、README 和配置说明。移除旧 RabbitMQ 队列实现及对应测试；SQLite 作为显式离线兼容后端保留。

公开版沿用现有整句多意图识别、阶段委派和 RAG 执行流程；快照由已校验委派生成，未引入私有规划重试、RAG 实验、知识数据集或本地评测报告。任务快照中的历史背景不增加 Agent 的工具权限。

新增主体：Redis/Chroma 适配器、主题记忆与 ADD/SKIP 判断、后台更新器、约束保护、只读快照、显式迁移工具、相关回归测试及设计说明。修改主体：聊天入口、记忆管理器、CaseState、上下文整理、编排与 Specialist 接入、会话闸门和限流、应用工厂、部署依赖与配置。删除主体：`backend/memory/profile_update_queue.py` 和 `tests/test_profile_update_queue.py`。

升级时将 `LONG_TERM_MEMORY_QUEUE_ENABLED=false` 改为 `LONG_TERM_MEMORY_ASYNC_ENABLED=false`；旧 MQ 变量已不再读取。启动 Redis 与 Chroma 后再启用新默认配置。迁移工具默认只读预览，显式导入步骤见[Redis/Chroma 迁移](redis-chroma-memory-20261004.md)和[主题记忆迁移](long-term-memory-mem0-runtime.md)。
