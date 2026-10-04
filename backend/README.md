# Backend 模块说明

`backend/` 是 TokenPlan 的 Python 运行时源码根目录。按职责可分为：

| 分组 | 模块 | 职责 |
|---|---|---|
| 入口与应用编排 | `api/`、`cli.py`、`app_services.py`、`application/` | HTTP/CLI 入口、依赖装配和稳定的应用流程 |
| Agent 决策 | `agents/`、`core/` | Supervisor、套餐与权益 / 交易与账务 / 用户支持三个领域 Agent、意图识别和模型客户端 |
| 执行与治理 | `runtime/`、`response/`、`skills/` | 受控执行、工具绑定、响应检查和 Skill 目录 |
| 数据与集成 | `mcp/`、`memory/` | FAQ / 单跳 / Agentic RAG 工具、只读业务查询接口、会话状态和长期记忆 |
| 可观测性 | `monitor/` | Trace、指标和性能监控 |
| 通用扩展 | `tools/` | 后端通用工具的预留位置 |

`memory/sqlite_session_store.py` 保存短期窗口、增量摘要、有界原始归档、CaseState
和轮次提交校验；ChromaDB 保存长期用户事实。短期窗口过期或缺失时，可从仍有效的
有界归档重建；已有 SQLite 会话文件可以直接继续使用。

根目录只保留前端、评测、测试、文档、部署配置等项目级模块。后端内部导入仍以
`agents`、`core`、`runtime` 等为源码根包，因此本地工具应把 `backend/` 加入
Python 搜索路径；仓库内的 Pytest、Docker 和文档命令已经完成对应配置。
