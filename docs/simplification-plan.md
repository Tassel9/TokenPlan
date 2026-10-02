# 工程改造方案（精简与借鉴）

> 这是工程侧清单，不含面试话术。来源：与开源项目 `XianyuAutoAgent` 的逐项对照（2026-09-23）。
> 目标：把项目从"多依赖、多配置"改成"最少依赖可跑、按需升级"，同时**不动证据体系**。> 最近复核：2026-09-26（测试 498 例全绿；`离线模式`、`提示词外置` 等仍待做，见 §2 状态列）。
## 1. 结论

| | 对照组 XianyuAutoAgent | 本仓库（改造前） | 目标 |
|---|---|---|---|
| 配置面 | 6 个环境变量 | **83 个生效变量** | 必填 1 个；本地运行额外 5 个；其余有默认值 |
| 容器 | 0 | **6 个服务** | 保持可用，但依赖降级为可选 |
| 离线可跑 | 纯函数，零依赖 | **无** | 提供 `--offline` 模式 |
| 提示词 | 外置 `prompts/*.txt` + 热重载 | 代码内 | 外置 + 版本化 |
| 测试 / 评测 | 0 / 0 | **54 个测试文件 / 6 个评测脚本** | 保留（这是本仓库的强项） |

**要学的是减法习惯，要保的是证据体系。**

改造后（2026-09-23 实测）：`.env.example` 从 **83 项 / 145 行**收到 **1 项必填 + 6 项本地地址 + 7 项注释可选**（53 行，含校准与权限说明注记）；新增 `python backend/cli.py doctor`；测试 **498 例全绿**（64 个测试文件，含 `tests/test_cli_doctor.py` 16 例）。〔2026-09-26 复核〕

## 2. 借鉴清单（9 条）

| # | 做法 | 落点 | 状态 |
|---|---|---|---|
| 1 | 配置分层：示例只留必填，其余回落代码默认值 | `.env.example`、`docs/configuration.md` | ✅ 已完成 |
| 2 | 队列与 Chroma 降级配置 | `backend/app_services.py`（已支持 `LONG_TERM_MEMORY_QUEUE_ENABLED` 等） | 部分具备，文档化 |
| 3 | 提示词外置 + 版本化 + 热重载 | `prompts/`、`backend/agents/supervisor_lead.py`（已有 `POLICY_VERSION`） | 待做 |
| 4 | 离线可跑：纯函数/假依赖跑通端到端 | `backend/cli.py`、`backend/app_services.py` | 待做（见第 4 节验收） |
| 5 | 启动自检：缺什么、怎么补、能否降级 | `backend/core/doctor.py`、`python backend/cli.py doctor` | ✅ 已完成 |
| 6 | 接管状态机：接管期间只记录 + 超时交还 | `backend/agents/intent_router.py` 附近（已有 `HandoffPolicy`） | 待做 |
| 7 | 入口卫生清单化（哪些输入不进 LLM） | `backend/core/request_control.py` | 待做 |
| 8 | 输出安全分两层：合规黑名单（硬替换）+ 真实性质疑（证据不足转人工） | `backend/response/guard.py` | 待做 |
| 9 | 模型接入与代码解耦（换模型只改环境变量） | `backend/core/deepseek_client.py`（已配置化） | 已完成 |

## 3. 数据审计（先审计再下刀，不要拍脑袋删）

| 审计 | 问题 | 产出 |
|---|---|---|
| ① 标签使用率 | 13 个意图标签中，评测集中样本 <3 条的有几个？ | 合并尾部标签的候选名单 |
| ② 多轮触发率 | 多少比例会话真正用到第 2 轮 `SEND_MESSAGES`？若 <5%，`max_rounds` 6 → 2~3 并加单意图快速路径 | 轮次分布表 |
| ③ 队列价值 | MQ 当前同进程部署，价值是否只剩"持久化 + 重试"？ | 是否用 SQLite 表 + 后台任务替代的结论 |
| ④ 配置利用率 | 83 个变量里有多少用过非默认值？ | 可删除清单 |

复现方式：① 用 `evaluation/` 现有冻结集统计标签频次；② 用 `backend/monitor/execution_trace.py` 的 trace 统计每会话轮次；③ 检查 `.env` 是否有非默认值。

## 4. 验收标准

- [x] `.env.example` 必填段 ≤ 8 项（现为 1 项必填 + 6 项本地地址）
- [x] `python backend/cli.py doctor` 可在依赖未启动时给出 blocked / degraded / ok 三种结论
- [x] 删除 `.env` 中的非必填项后，容器路径（compose 注入）与本地路径都能启动
- [x] 回归：`python -m unittest discover -s tests -t .` → **498 例全绿**（2026-09-26 复核）
- [ ] `python backend/cli.py "在吗" --offline` 无 Chroma / 无密钥也能出一份 JSON 回执
- [ ] 提示词改动不需要修改 Python 文件

### 4.1 已交付

| 文件 | 说明 |
|---|---|
| `.env.example` | 三层结构：必填 1 项 / 本地地址 5 项 / 注释可选 7 项 |
| `docs/configuration.md` | 80+ 个可读变量的分组参考（含默认值与来源） |
| `docs/simplification-plan.md` | 本文 |
| `backend/core/doctor.py` | 配置自检（只读环境变量 + 探测端口，不构造服务图） |
| `backend/cli.py` | 新增 `python backend/cli.py doctor [--json]`；原聊天命令改为 `python backend/cli.py "消息"` |
| `tests/test_cli_doctor.py` | 16 例：必填缺失 / 占位值 / 各依赖降级判定 / JSON 契约 / 命令分发 / 不启动服务图 |

### 4.2 需要留意的约束

`tests/test_supervisor_intent_latest_manifest.py` 把 `.env.example` 与评测报告锁在一起：

```python
self.assertIn("SUPERVISOR_INTENT_CANDIDATE_TOP_N=6", env_example)
```

这是证据链（manifest 同时锁定 dataset / few_shots / report 的 sha256）。因此该行**保留在示例中**并注明“被报告锁定，改动需重跑评测并更新 manifest”，不要为了精简删掉它。

## 5. 不砍清单（证据三支柱）

1. **候选召回 + 置信度门控**：90 条冻结集 EM 94.4% / Macro-F1 95.6%（`evaluation/reports/supervisor_intent_final_live_top6.json`）
2. **评测体系**：`evaluation/` 6 个脚本 + 冻结集 + 报告
3. **Skill 治理与 handoff 测试**：`tests/test_dynamic_skills.py`、`tests/test_conversation_case_state.py`

## 6. 边界

- 对照组是 **GPL-3.0**：只借鉴设计，**不复制代码**。
- 不引入任何新数字；所有结论必须能用仓库内命令复现。
- 本次未改动 compose 服务清单与评测流程。
