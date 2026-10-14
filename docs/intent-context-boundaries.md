# 上下文、意图识别与 Supervisor 编排边界

默认调用链使用 `IntentRecognitionPipeline`。它先独立整理上下文，再对完整问题做一次多意图识别；Supervisor 只消费冻结结果，不重复识别用户意图。

```text
原始问题 + 受控历史 / CaseState
  → QueryContextProcessor：整理指代、实体和引用，提出 effective_query
  → ContextResultValidator：检查结构与引用来源
  → IntentRecognizer：对完整问题输出全部当前诉求、各自原文证据和分数
     ↘ 全标签 Embedding：独立计算 13 个业务标签的相似度
  → IntentResultValidator：校验全部标签、原文证据和范围
  → IntentFusionPolicy：按意图逐项融合并冻结可执行意图集合
  → 可直接委派的单个知识诉求：按固定父领域进入 Agent
  → 其他诉求：Supervisor 只决定 Agent、主次、依赖、执行阶段和汇总
```

## 默认职责边界

- `QueryContextProcessor` 只处理指代、实体、受控引用和问题改写，不判断意图标签或选择 Agent。没有历史和有效案件事实时直接保留原句；少量完整公共 FAQ 在没有待澄清事项时也可跳过上下文模型。
- `IntentRecognizer` 读取原始问题、同一份 `effective_query` 和产品范围，不读取历史、CaseState、团队列表或执行 Observation。它一次返回当前消息中的完整意图集合，每项必须绑定当前原文证据。
- Embedding 与 LLM 意图树是并行信号。Embedding 不生成可执行意图，也不裁剪 LLM 的标签空间；LLM 通道失败时不能仅凭相似度执行。
- 外部校验器检查结构、合法标签、原文证据和范围；融合策略按每项诉求自己的证据计算分数。通过门禁的结果写入 `frozen_analysis` 与 `frozen_execution_analysis`，之后不能由 Supervisor 增删、改名或替换。
- Supervisor 收到冻结集合后，`analysis_required=false`、`intent_review_required=false`。它只负责把意图映射到三个父领域 Agent，确定 `primary_intent_id`、并发或依赖顺序、澄清、人工确认以及最终汇总；若再次输出 `analysis` 会被本地校验拒绝。
- 执行侧要求每个冻结意图被覆盖并收敛后才能 `FINAL`。这能防止调度阶段漏派，但不能证明上游识别器没有遗漏原始诉求，因此意图集合完整性仍需单独评测。

## 上下文和业务边界

- 历史预算优先保留用户消息；CaseState 的 `discussion_messages` 可保存最近讨论原文，即使上一轮因低置信度未派发，也能辅助后续指代恢复。讨论内容不等于已执行事项或已核验账户状态。
- 上下文模型只选择受控来源编号，代码回填用户原文并提取显式邮箱、订单、金额和错误码；模型不能自由生成来源或个人事实。多对象歧义在上游停止，不进入意图识别和业务工具。
- 领域 Agent 按套餐与权益、交易与账务、用户支持三个父领域分工。咨询可检索公开知识或调用受控只读查询；代购、退款提交、退订和账户修改需要询问是否转人工，不能声称已经操作。
- 可直接委派的单个高置信知识诉求只省略 Supervisor 规划调用，不省略上游意图识别、校验、融合、Agent 证据处理或最终回复检查。多意图及不适合快速委派的单意图仍进入 Supervisor。
- 独立的不确定事项单独澄清，其他已确认事项可继续回答；若不确定项是明确依赖链的前置条件，则停止后续任务。

## 兼容与对照模式

`RoutingIntentRecognizer` 仍保留给历史兼容和显式对照实验。只有调用方主动注入该识别器时，才使用“单个业务路由或 `orchestrate`”协议，并允许 Supervisor 对 `orchestrate` 做诉求拆解。它不再是默认应用装配，相关行为不能用于描述默认主链路。

显式注入历史一体化 `SupervisorLead` 的兼容路径也仍存在。`SupervisorAnalysis.rewrite` 是外层流水线组装的兼容字段，不是默认意图识别模型生成的内容。

## 调用方式

```python
from core.intent_pipeline import IntentRecognitionPipeline
from core.routing_intent_recognizer import RoutingIntentRecognizer

# 默认：一次识别完整意图集合，交给 Supervisor 时语义已冻结。
pipeline = IntentRecognitionPipeline(context, embedding_index=embedding_index)
outcome = await pipeline.recognize(
    query,
    history=history,
    case_state=case_state,
)

# 显式兼容 / 对照模式：单个业务路由或 orchestrate。
route_pipeline = IntentRecognitionPipeline(
    context,
    embedding_index=embedding_index,
    recognizer_type=RoutingIntentRecognizer,
)
```

## 配置与证据边界

默认应用使用 13 个业务标签，Embedding 权重、确认门限和低分门限由 `INTENT_FUSION_ALPHA`、`INTENT_CLEAR_THRESHOLD`、`INTENT_LOW_THRESHOLD` 配置。融合分数是门禁信号，不是正确概率。

`SINGLE_INTENT_FAST_PATH_ENABLED` 只控制有限单知识诉求是否直接委派；关闭后仍使用同一冻结意图集合进入 Supervisor，不会触发第二次意图识别。

Trace 分别记录 `query_context_ms`、`intent_recognition_ms` 和 `supervisor_ms`。结构校验、逐项派发和 Mock 回归不能证明模型语义识别完整，也不能代表生产效果；调整模型、提示、意图定义、融合权重或阈值后，应重新运行语义识别和端到端评测。
