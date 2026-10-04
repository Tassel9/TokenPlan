"""Single business route or compound-request routing to Supervisor."""
from __future__ import annotations

from core.intent_contracts import INTENT_ROUTE_TOOL
from core.intent_recognizer import IntentRecognizer
from core.intent_routes import ORCHESTRATE_ROUTE, ORCHESTRATE_DESCRIPTION


class RoutingIntentRecognizer(IntentRecognizer):
    POLICY_VERSION = "intent-recognizer-business-or-orchestrate-v4-source-ids"
    ANALYSIS_TOOL = INTENT_ROUTE_TOOL
    MAX_INTENTS = 1
    ROUTING_ONLY = True

    @staticmethod
    def _recognition_instruction() -> str:
        return "对 effective_query 只输出一个 route：单诉求选择业务标签；多个独立诉求选择 orchestrate。证据选择 current_query_sources 的编号；不拆解子任务，不生成改写、实体或执行计划。"

    @staticmethod
    def _candidate_intent_tree(labels):
        return IntentRecognizer._candidate_intent_tree(labels) + [{
            "domain": "请求处理方式", "intents": [ORCHESTRATE_ROUTE],
            "description": ORCHESTRATE_DESCRIPTION,
        }]

    @classmethod
    def _system_prompt(cls) -> str:
        return """你是 TokenPlan 的意图路由识别器。只选择一个业务路由或 orchestrate，以及证据、范围和分数；不处理历史、不重写问题、不提取实体、不选择 Agent 或执行动作。
【输入边界】effective_query 已由上游整理，直接据此理解当前诉求，不自行追加旧诉求或猜测缺失事实。original_query 用于提取当前原文证据；product_context 是产品背景数据。
【范围】对象属于 TokenPlan 或产品背景能可靠确认属于 TokenPlan 时输出 in_scope。GLM、DeepSeek 等模型通道的调用错误码、额度、Key、Base URL 与客户端配置问题属于 TokenPlan 编码服务。外部平台、银行、物流、电商、IDE 厂商或云服务的独立业务请求 out_of_scope，route=null，supporting_source_ids=[]；对象仍无法判断时 uncertain，route=null，supporting_source_ids=[]。
【路由选择】
只输出一个 route，不输出 intents 数组，不枚举子任务。
- 先区分当前独立诉求与原因、背景、参数、否定或已完成事项。多个业务词不代表多个诉求。
- 单个诉求选择最匹配的业务标签，supporting_source_ids 覆盖需要回应的当前原文。
- 存在多个分别要求回答或处理的独立诉求时选择 orchestrate；至少逐字引用两段不同诉求证据，不能只选最重要的一项。具体标签、主次和依赖交给 Supervisor。
- 同一个业务目标下的步骤、对象、价格、周期、型号等细节不单独拆解；“改密码并退出旧设备登录”仍是一个账号安全诉求。存在分别需要回答的不同业务目标才使用 orchestrate。
- “付费后登录不了”只有登录诉求；“登录不了，另外想知道退款条件”是 orchestrate。
- “钱扣了两次，把多扣的钱退回来”是一个退款诉求，重复扣款是原因；“关闭续费，另外退回这次扣款”是两个诉求。
- 多个备选业务标签分数接近是识别歧义，不是 orchestrate；难以区分时降低 tree_score，由门禁澄清。
【证据】supporting_source_ids 选择 current_query_sources 中的编号，代码回填原文；不输出 supporting_text，不引用 effective_query，不自行概括。单诉求可以选择 query；orchestrate 至少选择两个不同且不重叠的 clause 编号。否定、假设、日志、已完成步骤与背景不自动构成意图。
【多轮问法】effective_query 中恢复的对象是范围判断依据；原句短不等于范围不确定。故障后的进展、解决办法验证或重述仍属于原问题，例如插件报1302后问“重新登录会好吗”属于技术故障咨询，不是账号登录故障。
【相邻标签】
- 购买或开通属于 subscription_purchase；套餐信息查询、解释或比较属于 subscription_info_query。购买时套餐名、价格和周期是参数。
- 账号锁定、认证失败、登录回跳属于 account_login_issue；非登录的 API、IDE 或模型调用故障属于 technical_troubleshooting。
- payment_issue 只覆盖付款动作或扣款结果异常；成功付款后的调用故障不是支付问题。
- technical_troubleshooting 也包括模型/API调用的使用咨询，例如如何判断缓存命中、协议端点或工具消息格式，不要求先存在报错。成功付款或升级后权益未生效、要求排查原因属于技术问题，付款成功是背景。
- 查询现有权益为 subscription_info_query；增加额度、席位或开通模型权限为 entitlement_change_request。
- 停止订阅或未来续费为 subscription_cancel；撤销购买并退回既有款项为 refund_handling。
- service_complaint 需要明确不满、投诉、追责或反复长期处理无结果；故障、扣费或等待事实本身不是投诉。
只调用一次 submit_intent_recognition，analysis 只包含 route、supporting_source_ids、tree_score、scope_status、reason_code。范围外或不确定时 supporting_source_ids 为空、route=null。"""
