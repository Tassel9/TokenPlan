"""Targeted questions for validated but unresolved routing; no model call."""
from __future__ import annotations

import re
from core.supervisor_decision import FineGrainedIntent, SupervisorAnalysis


def routing_clarification(analysis: SupervisorAnalysis | None) -> str:
    if analysis is None:
        return "请说明你要咨询的 TokenPlan 套餐、账务或技术问题。"
    if analysis.rewrite.clarification_question:
        return analysis.rewrite.clarification_question
    query = analysis.rewrite.effective_query
    # Safe boundaries may be stated without accepting or executing an intent.
    if '保证' in query and any(word in query for word in ('补偿', '今天', '修复')):
        return "我无法保证今天内修复，也不能承诺补偿结果。是否需要转人工客服核验故障进度和补偿诉求？"
    if re.search(r'(?:直接|替我|帮我).{0,16}(?:退款|订阅退|注销|取消订阅|关闭续费)', query):
        return "当前只能提供办理规则和流程咨询，无法直接退款、取消订阅或注销账户。是否需要转人工客服办理？"
    labels = {item.label for item in analysis.intents}
    if FineGrainedIntent.TECHNICAL_TROUBLESHOOTING in labels:
        missing = []
        if not any(word.casefold() in query.casefold() for word in
                   ("IDE", "插件", "CLI", "编码工具", "客户端", "API", "工具调用")):
            missing.append("使用的客户端或调用方式")
        if not any(word in query for word in ("报", "错误", "超时", "拒绝", "失败", "异常", "缓存")):
            missing.append("具体报错或异常现象")
        if missing:
            return "为继续排查，请补充" + "、".join(missing) + "。"
        return "请补充问题发生时间及脱敏后的关键错误日志，方便进一步排查。不要提供完整密钥。"
    if FineGrainedIntent.ACCOUNT_LOGIN_ISSUE in labels:
        return "请说明是 TokenPlan 控制台登录失败，还是客户端/API 调用认证失败，并提供错误提示；不要提供密码或密钥。"
    if FineGrainedIntent.REFUND_HANDLING in labels:
        return "你是想了解退款条件和流程，还是需要转人工客服办理退款？"
    if FineGrainedIntent.SERVICE_COMPLAINT in labels:
        return "我无法承诺修复时间或补偿结果。是否需要转人工客服核验你反馈的问题及补偿诉求？"
    if FineGrainedIntent.SUBSCRIPTION_INFO_QUERY in labels:
        return "你想了解公开套餐规则，还是查询自己账户的套餐或用量记录？"
    return "你指的是 TokenPlan 的哪个套餐、账号或调用问题？请补充具体对象。"
