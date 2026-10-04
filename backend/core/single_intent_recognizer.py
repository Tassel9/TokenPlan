"""Matched, independently prompted single-intent baseline (not output truncation)."""
from __future__ import annotations

from copy import deepcopy
from core.intent_recognizer import INTENT_ANALYSIS_TOOL, IntentRecognizer


_SINGLE_INTENT_TOOL = deepcopy(INTENT_ANALYSIS_TOOL)
_SINGLE_INTENT_TOOL["input_schema"]["properties"]["analysis"]["properties"]["intents"]["maxItems"] = 1


class SingleIntentRecognizer(IntentRecognizer):
    """Use the same prepared input, model and labels, but one goal.

    This class is injectable into IntentOrchestrator and does not change its
    default recognizer. It selects a current primary goal in a separate LLM
    request; it never calls the multi-intent recognizer and truncates its output.
    """

    POLICY_VERSION = "intent-recognizer-single-v2-candidates-only"
    ANALYSIS_TOOL = _SINGLE_INTENT_TOOL
    MAX_INTENTS = 1

    @staticmethod
    def _intent_selection_rule() -> str:
        return (
            "只识别当前一个主要诉求，intents 最多一个标签。每个标签必须对应用户要求回答或完成的结果；"
            "名称、金额、套餐、错误码、操作参数和背景描述不能单独激活标签。"
            "若有多个独立诉求，优先用户明确指定的当前重点；未指定重点时选择原文中最先提出的独立诉求。"
            "即使还有其他诉求，也不要增加标签。多个问题若属于同一标签，合并 supporting_text。"
            "唯一 intent 仍须输出基于意图树边界与原文证据的 tree_score。"
            "不要为了返回一个标签而忽略范围或指代不确定性；out_of_scope 或 uncertain 时 intents=[]。"
        )
