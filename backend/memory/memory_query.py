"""Grounded supplemental memory queries, without an additional model call."""
from __future__ import annotations

import re

_REFERENCE = re.compile(r"这个|那个|刚才|之前|当前|现在|它|那我的|\b(?:this|that|current|previous)\b", re.I)
_TOPICS = {
    "套餐": "套餐选择偏好 付费周期 预算 使用需求",
    "订阅": "套餐选择偏好 付费周期 预算 使用需求",
    "年付": "账单周期偏好 年付 月付",
    "月付": "账单周期偏好 年付 月付",
    "插件": "用户一直使用什么操作系统和开发工具进行开发？",
    "开发": "用户一直使用什么操作系统和开发工具进行开发？",
    "方案": "项目决策 用户计划 历史助手建议",
}


def memory_queries(query: str, recent_messages=(), case_state=None) -> list[str]:
    query = str(query or "").strip()
    if not query:
        return []
    result = [query]
    for topic, supplement in _TOPICS.items():
        if topic in query and supplement not in result:
            result.append(supplement)
    if _REFERENCE.search(query):
        # Actual conversation text is a query hint, never a new user fact.
        for message in reversed(list(recent_messages)[-4:]):
            content = str(getattr(message, "content", "") or "")
            if content:
                result.append(query + "；此前对话：" + content[:400])
                break
        for key, values in getattr(case_state, "entities", {}).items():
            if key in {"workspace_id", "plan", "plan_name"} and values:
                result.append(query + "；已知对象：" + " ".join(map(str, values[:2])))
    return list(dict.fromkeys(result))[:3]
