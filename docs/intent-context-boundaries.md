# 上下文、识别、校验分离

默认调用链已接入 `IntentRecognitionPipeline`，职责不是只换文件名：识别器的输入参数、模型提示和输出 schema 均不再包含历史整理、实体提取或改写。

```text
原始问题 + 受控历史 / CaseState
  → QueryContextProcessor：整理指代、实体和引用，提出 effective_query
  → ContextResultValidator：检查结构与引用来源
  → RoutingIntentRecognizer：并行计算 Embedding / LLM，只选择一个业务路由或 orchestrate
  → RouteResultValidator + 融合门禁：检查路由、当前原文证据、范围和分数
  → 普通业务路由：按父领域直接派发咨询 Agent
  → orchestrate：Supervisor 拆解业务诉求并确定 primary_intent_id
     → IntentResultValidator：校验全部标签、原文证据和引用编号
     → 每项诉求分别计算 Embedding / LLM 融合，冻结后分阶段执行
```

## 边界与兼容

- `QueryContextProcessor` 只接受当前问题与上下文，输出 `rewrite`；不判断标签、分数或 Agent。原生模型只选择来源编号，代码回填用户原文并提取显式邮箱、订单、金额和错误码；不让模型重新抄写来源或生成实体。没有历史和有效案件事实时直接保留原句，不调用上下文模型。服务入口附带受信任的 TokenPlan 产品范围，不能以此虚构个人事实或将明确外部业务纳入范围。
- 少量完整公共 FAQ 模板在没有待澄清事项时，即使有历史也保留当前原句，不调用上下文模型；`rewrite.reason_code=complete_public_question`，继承实体和引用为空。模板整句匹配，不靠“没有代词”推断完整性。指代、省略、多个问题、个人记录、待澄清事项和未命中模板时保留原上下文路径；校验重试改由模型整理。
- 其他有受控历史或有效案件事实的请求独立调用上下文模型。历史预算优先保留用户消息；CaseState 的 `discussion_messages` 保存最近讨论原文，即使低置信度尚未派发，也能在追问时恢复对象。它不是已执行事项或已核验账户状态；当前明确新话题优先。引用按实际传给模型的历史快照重新编号，来源目录只包含用户话语和案件实体。真正的多对象歧义在上游停止，两路识别均不启动；格式或来源校验失败有限重试后追问对象，不据此宣称已转人工。
- 默认 `RoutingIntentRecognizer.recognize(IntentRecognitionInput(...))` 只产出一个未经校验的业务路由或 `orchestrate`，不输出子任务数组。多个业务词不直接等于多个诉求；背景、参数、否定或已完成事项不另拆。原生模型从当前原文目录选择 `supporting_source_ids`，代码回填证据，编排入口至少提供两段不同且互不包含的当前诉求原文。逗号、顿号等切分仅提供候选片段，不自动确定诉求数量。
- 外部校验器负责结构、合法标签和来源检查；上下文校验还约束指代引用来自当前消息、继承实体绑定到已引用来源、澄清候选有来源。它不是业务权限、账户归属、退款条件或金额正确性的校验；业务校验仍在领域 Agent / 工具边界。
- 外层流水线负责有限的格式 / 证据重试、路由融合与门禁。`orchestrate` 独立于 13 个业务标签；它的融合分数写入 `route_fusion`，不能当成一个可执行业务意图。
- 已确认的普通业务路由默认按父领域直接进入 Agent。只有 `orchestrate` 进入 Supervisor 拆解；它在同一次决策中提出诉求集合、主诉求和首个阶段，不重新整理历史或改写问题。
- Supervisor 的 `primary_intent_id` 必须引用已输出诉求。用户明确强调重点时据此选择，否则选择最先提出的当前诉求；分数高低不决定重要程度。最终汇总围绕主诉求展开，各项结论分别依据各自 Observation，不能省略冲突和未解决事项。主诉求也优先写入会话的 `last_intents`。
- 每项业务诉求基于自己的 `supporting_text` 计算 Embedding 分数，与自己的 LLM 分数融合，后续轮次冻结结果。同标签合并或编号规范化时，主诉求、派发和待人工确认的引用同步映射；不能以新编号派发旧引用。
- 复合请求派发时，领域 Agent 的问题正文由绑定诉求的已校验原文证据组装；不能把其他诉求夹进 Supervisor 的自然语言委派文本。指代相关事实仍通过已校验实体与 AgentMemory 提供。显式顺序请求的 Tool schema 与本地门禁都限制每阶段一条消息、一个业务意图和 `all_success`。
- 独立的不确定事项单独澄清，其余已确认事项全部回答后保留结果。若不确定项是明确顺序请求中的前置事项，则停止后续依赖。首轮计划只涉及不确定项时，冻结确认集合并重新规划；不执行未确认诉求。
- 普通请求由领域 Agent 区分咨询与代操作；复合请求还由 Supervisor 在 `handoff_confirmation_intent_ids` 中记录需询问用户是否转人工的诉求。退款条件、材料和入口仍是咨询；“替我退款”属于退款领域，但不能直接办理。允许直接 `ASK_USER`，不强制先派发操作。
- 混合请求先回答可回答的咨询，再 `ASK_USER` 询问是否转人工；这些待确认诉求保留在识别结果中，不能被当作已完成，也不因尚未派发而自动转人工。执行团队按三个父意图划分：套餐与权益 → `subscription`、交易与账务 → `billing`、用户支持 → `support`；业务办理 Agent 已移除。领域 Agent 自行选择 FAQ、单跳或 Agentic RAG 工具；个人记录查询由 Agentic 工具调用受控只读后台。用户明确要求转人工时，入口已有 `RequestControlPolicy` 接收人工请求；该状态不证明外部客服已经接单。

`SupervisorAnalysis.rewrite` 作为旧消费端的兼容字段，由外层流水线组装，不是识别模型产生的内容。显式注入历史合并式 `SupervisorLead` 的兼容路径仍保留；默认应用装配走新链路，不宣称已删除所有旧接口。

## 调用方式

```python
from core.intent_pipeline import IntentRecognitionPipeline
from core.intent_recognizer import IntentRecognizer
from core.single_intent_recognizer import SingleIntentRecognizer

pipeline = IntentRecognitionPipeline(context, embedding_index=embedding_index)
outcome = await pipeline.recognize(query, history=history, case_state=case_state)
# outcome.route 是单个业务标签或 "orchestrate"。
# 默认 IntentOrchestrator 直接派发普通请求，仅为 orchestrate 启动拆解。

# 以下显式保留历史单 / 多标签对照，两者都不是生产路由识别器。
single = IntentRecognitionPipeline(
    context, recognizer=SingleIntentRecognizer(context, embedding_index=embedding_index))
multi = IntentRecognitionPipeline(context, embedding_index=embedding_index, recognizer_type=IntentRecognizer)
# 对照实验固定上游整理结果，双方看到同一问题。
prepared = await multi.prepare_query(query, history=history, case_state=case_state)
single_outcome = await single.recognize_prepared(prepared)
multi_outcome = await multi.recognize_prepared(prepared)
```

## 配置与证据边界

当前为单路由识别 + 复合请求拆解；业务标签仍为 13 个，另有 1 个控制路由定义。2026-10-04 校准后，默认应用装配使用 Embedding 权重 `0.05`、确认门限 `0.70`、低分门限 `0.40`，业务路由、控制路由和拆出的诉求均收到同组配置。环境变量可以覆盖；独立实例化历史策略类仍保留 `0.10` 的兼容默认值。融合分数不是概率，LLM 失败时不能仅凭 Embedding 执行业务。校准划分、候选选择与独立验证结果见[修复评测记录](context-repairs-evaluation-20261004.md)。

Supervisor 拆解与首轮规划共用一次请求。普通业务路由省去 Supervisor 规划和汇总调用，复合请求增加逐项 Embedding 校验。独立上下文模型按上述条件调用。领域 Agent 对已确认标签与简单公共 FAQ 模板一致的单诉求请求，前置受控 FAQ 检索后让模型作答；`orchestrate` 的全部子任务及其他复杂请求仍由模型选工具。耗时、费用与识别效果依据模型评测，不能根据 Mock 回归推断。Trace 分别记录 `query_context_ms`、`intent_recognition_ms` 和 `supervisor_ms`。关闭 `SINGLE_INTENT_FAST_PATH_ENABLED` 或注入旧团队时仍保留原 Supervisor 检查路径。

原 100 条样本与金标保持冻结，旧报告不覆盖。`compare_single_multi_intent.py` 和 `evaluate_supervisor_semantics.py` 仍显式测历史协议。当前对照用 `evaluate_orchestrate_routing.py` 跑真实模型语义，用 `run_routing_e2e.py` 跑 ChatService、模型与工具的端到端链路。路由改造的历史结果见[初始离线评测](orchestrate-routing-evaluation-20261004.md)，本轮上下文修复结果见[修复评测记录](context-repairs-evaluation-20261004.md)。结构和引用检查不能确定性证明模型没有漏诉求、误判指代或生成语义错误；测试通过也不替代模型效果评测。

完整 API Key 的检测在 ChatService 查询长期记忆前执行，消息、历史、摘要和案件状态进入模型或追踪前脱敏；当前消息含密钥时以固定安全答复提示用户自行撤销轮换，不调用识别与回答模型，也不替用户配置账户。已进入系统前的客户端或基础设施日志不由该检测器覆盖。回复检查区分公开规则中的“已开票等情况”和无依据的“已为您退款”；名词状态描述放行不代表个人业务状态已核验。

职责预期版本见 [intent_natural_consultation_v2.json](../evaluation/fixtures/intent_natural_consultation_v2.json)：保留原 100 条输入、历史上下文及业务话题金标，另标注咨询、人工确认与禁止操作能力。29 条要求先询问人工，8 条要求先回答咨询再询问人工。行为预期仍待独立业务复核，尚未跑当前 Supervisor 的真实模型行为评测。
