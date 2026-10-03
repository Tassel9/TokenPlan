"""One-pass fusion for the independent embedding and LLM intent-tree channels."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Sequence

from core.intent_embedding import IntentEmbeddingResult
from core.supervisor_decision import (
    FineGrainedIntent,
    ScopeStatus,
    SupervisorAnalysis,
    SupervisorIntent,
)


class IntentFusionBand(str, Enum):
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    LOW = "low"


INTENT_DISPLAY_NAMES: Dict[FineGrainedIntent, str] = {
    FineGrainedIntent.SUBSCRIPTION_INFO_QUERY: "查询套餐信息",
    FineGrainedIntent.SUBSCRIPTION_PURCHASE: "购买订阅",
    FineGrainedIntent.SUBSCRIPTION_CHANGE: "变更订阅",
    FineGrainedIntent.SUBSCRIPTION_CANCEL: "取消订阅",
    FineGrainedIntent.PAYMENT_ISSUE: "处理支付或扣款问题",
    FineGrainedIntent.INVOICE_HANDLING: "处理发票",
    FineGrainedIntent.REFUND_HANDLING: "处理退款",
    FineGrainedIntent.ACCOUNT_LOGIN_ISSUE: "处理登录问题",
    FineGrainedIntent.ACCOUNT_SECURITY_REQUEST: "变更账号安全设置",
    FineGrainedIntent.ENTITLEMENT_CHANGE_REQUEST: "变更额度或权限",
    FineGrainedIntent.TECHNICAL_TROUBLESHOOTING: "排查技术故障",
    FineGrainedIntent.SERVICE_COMPLAINT: "投诉或追责",
    FineGrainedIntent.SERVICE_FEEDBACK: "提交建议或评价",
}

_INTENT_ANCHORS: Dict[FineGrainedIntent, tuple[str, ...]] = {
    FineGrainedIntent.SUBSCRIPTION_INFO_QUERY: ("了解", "查询", "比较", "区别", "价格", "额度", "多少"),
    FineGrainedIntent.SUBSCRIPTION_PURCHASE: ("购买", "开通", "买"),
    FineGrainedIntent.SUBSCRIPTION_CHANGE: ("升级", "降级", "换成", "改成", "变更", "调低", "调高"),
    FineGrainedIntent.SUBSCRIPTION_CANCEL: ("取消", "停止订阅", "关闭续费", "停止续费", "停掉"),
    FineGrainedIntent.PAYMENT_ISSUE: ("重复扣", "多扣", "扣款", "支付失败", "付款失败", "金额异常"),
    FineGrainedIntent.INVOICE_HANDLING: ("发票", "开票", "抬头"),
    FineGrainedIntent.REFUND_HANDLING: ("退款", "退钱", "退回款", "退回"),
    FineGrainedIntent.ACCOUNT_LOGIN_ISSUE: ("无法登录", "不能登录", "登录失败", "账号锁定", "密码错误"),
    FineGrainedIntent.ACCOUNT_SECURITY_REQUEST: ("改密码", "更换邮箱", "两步验证", "退出设备", "注销账号"),
    FineGrainedIntent.ENTITLEMENT_CHANGE_REQUEST: ("增加额度", "增加席位", "开通权限", "添加席位", "补一个位置"),
    FineGrainedIntent.TECHNICAL_TROUBLESHOOTING: ("报错", "故障", "超时", "无法使用", "不可用", "错误"),
    FineGrainedIntent.SERVICE_COMPLAINT: ("投诉", "追责", "要个说法", "不满"),
    FineGrainedIntent.SERVICE_FEEDBACK: ("建议", "希望", "评价", "很好用"),
}

_NEGATION_PREFIX = re.compile(
    r"(?:不是(?:来|要|想|为了)?|并非|不要|不想|不需要|无需|不用|别|先别|"
    r"没有(?:要求|打算|想|要)|没(?:有)?(?:要求|打算|想|要)|未(?:要求|打算))"
    r"[^，,。；;！？!?\n]{0,7}$"
)
_BACKGROUND_PREFIX = re.compile(
    r"(?:如果|假如|假设|比如|例如|示例|文档(?:里|中)?|日志(?:里|中)?|"
    r"邮件(?:里|中|写着)?|截图(?:里|中)?|测试数据(?:里|中)?|别人说)"
    r"[^，,。；;！？!?\n]{0,16}$"
)


@dataclass(frozen=True)
class IntentFusionDecision:
    intent: SupervisorIntent
    band: IntentFusionBand
    embedding_score: float
    tree_score: float
    final_score: float
    fusion_alpha: float
    reason_code: str

    @property
    def intent_id(self) -> str:
        return self.intent.intent_id

    @property
    def label(self) -> str:
        return self.intent.label.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "label": self.label,
            "supporting_text": list(self.intent.supporting_text),
            "band": self.band.value,
            "embedding_score": round(self.embedding_score, 6),
            "tree_score": round(self.tree_score, 6),
            "final_score": round(self.final_score, 6),
            "fusion_alpha": self.fusion_alpha,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class IntentFusionAssessment:
    status: str
    decisions: tuple[IntentFusionDecision, ...]
    clear_threshold: float
    low_threshold: float
    fusion_alpha: float
    active_channels: tuple[str, ...] = ("embedding", "intent_tree")
    degraded: bool = False
    error: str = ""

    @property
    def confirmed(self) -> tuple[SupervisorIntent, ...]:
        return tuple(
            item.intent for item in self.decisions
            if item.band == IntentFusionBand.CONFIRMED
        )

    @property
    def clarification_candidates(self) -> tuple[SupervisorIntent, ...]:
        return tuple(
            item.intent for item in self.decisions
            if item.band == IntentFusionBand.AMBIGUOUS
        )

    @property
    def rejected(self) -> tuple[SupervisorIntent, ...]:
        return tuple(
            item.intent for item in self.decisions
            if item.band == IntentFusionBand.LOW
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": "parallel_embedding_llm_tree_fusion_v2",
            "status": self.status,
            "clear_threshold": self.clear_threshold,
            "low_threshold": self.low_threshold,
            "fusion_alpha": self.fusion_alpha,
            "active_channels": list(self.active_channels),
            "degraded": self.degraded,
            "confirmed_intent_ids": [item.intent_id for item in self.confirmed],
            "clarification_intent_ids": [
                item.intent_id for item in self.clarification_candidates
            ],
            "rejected_intent_ids": [item.intent_id for item in self.rejected],
            "decisions": [item.to_dict() for item in self.decisions],
            "error": self.error,
        }


class IntentFusionPolicy:
    """Fuse each tree-selected label once; embedding never dispatches by itself."""

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        clear_threshold: float = 0.70,
        low_threshold: float = 0.40,
    ) -> None:
        alpha = float(alpha)
        clear = float(clear_threshold)
        low = float(low_threshold)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("intent fusion alpha must be between 0 and 1")
        if not 0.0 <= low < clear <= 1.0:
            raise ValueError(
                "intent thresholds must satisfy 0 <= low < clear <= 1"
            )
        self.alpha = alpha
        self.clear_threshold = clear
        self.low_threshold = low

    def assess(
        self,
        original_query: str,
        analysis: SupervisorAnalysis,
        embedding: IntentEmbeddingResult,
    ) -> IntentFusionAssessment:
        embedding_ok = embedding.status == "ok"
        active_channels = (
            ("embedding", "intent_tree")
            if embedding_ok else ("intent_tree",)
        )
        effective_alpha = self.alpha if embedding_ok else 0.0
        if analysis.scope_status != ScopeStatus.IN_SCOPE or not analysis.intents:
            return IntentFusionAssessment(
                "not_applicable",
                (),
                self.clear_threshold,
                self.low_threshold,
                effective_alpha,
                active_channels,
                not embedding_ok,
                embedding.error,
            )

        if embedding_ok:
            missing = [
                item.label.value for item in analysis.intents
                if embedding.score_for(item.label.value) is None
            ]
            if missing:
                return IntentFusionAssessment(
                    "failed",
                    (),
                    self.clear_threshold,
                    self.low_threshold,
                    effective_alpha,
                    active_channels,
                    False,
                    "embedding score coverage mismatch: " + ", ".join(missing),
                )

        decisions = []
        for intent in analysis.intents:
            embedding_score = embedding.score_for(intent.label.value) or 0.0
            tree_score = max(0.0, min(1.0, float(intent.tree_score)))
            final_score = (
                effective_alpha * embedding_score
                + (1.0 - effective_alpha) * tree_score
            )
            conflict = _evidence_conflict(original_query, intent)
            if conflict:
                band = IntentFusionBand.LOW
                reason_code = conflict
            elif final_score >= self.clear_threshold:
                band = IntentFusionBand.CONFIRMED
                reason_code = "above_clear_threshold"
            elif final_score >= self.low_threshold:
                band = IntentFusionBand.AMBIGUOUS
                reason_code = "between_thresholds"
            else:
                band = IntentFusionBand.LOW
                reason_code = "below_low_threshold"
            decisions.append(IntentFusionDecision(
                intent,
                band,
                embedding_score,
                tree_score,
                final_score,
                effective_alpha,
                reason_code,
            ))

        return IntentFusionAssessment(
            "ok",
            tuple(decisions),
            self.clear_threshold,
            self.low_threshold,
            effective_alpha,
            active_channels,
            not embedding_ok,
            embedding.error,
        )

    @staticmethod
    def clarification_question(
        candidates: Sequence[SupervisorIntent],
        *,
        confirmed: bool,
    ) -> str:
        names = [f"“{INTENT_DISPLAY_NAMES[item.label]}”" for item in candidates]
        joined = "、".join(names)
        if confirmed:
            return f"另外，我还不能确认你是否也需要{joined}，请明确一下。"
        return f"我还不能确认你是否需要{joined}，请明确一下你希望我处理哪些诉求。"


def _evidence_conflict(original_query: str, intent: SupervisorIntent) -> str:
    anchors = _INTENT_ANCHORS[intent.label]
    positions = list(
        _supported_anchor_positions(
            original_query,
            intent.supporting_text,
            anchors,
        )
    )
    if not positions:
        return ""
    contexts = [_position_context(original_query, position) for position in positions]
    if all(context == "negated" for context in contexts):
        return "explicit_negation_conflict"
    if all(context in {"negated", "background"} for context in contexts):
        return "background_only_conflict"
    return ""


def _supported_anchor_positions(
    query: str,
    supporting_text: Sequence[str],
    anchors: Sequence[str],
) -> Iterable[int]:
    lowered_query = query.casefold()
    for evidence in supporting_text:
        lowered_evidence = evidence.casefold()
        start = 0
        while True:
            support_at = lowered_query.find(lowered_evidence, start)
            if support_at < 0:
                break
            found_anchor = False
            for anchor in anchors:
                anchor_start = 0
                lowered_anchor = anchor.casefold()
                while True:
                    relative = lowered_evidence.find(lowered_anchor, anchor_start)
                    if relative < 0:
                        break
                    found_anchor = True
                    yield support_at + relative
                    anchor_start = relative + max(1, len(lowered_anchor))
            if not found_anchor:
                yield support_at
            start = support_at + max(1, len(lowered_evidence))


def _position_context(query: str, position: int) -> str:
    clause_start = max(
        query.rfind(marker, 0, position)
        for marker in ("，", ",", "。", "；", ";", "！", "!", "？", "?", "\n")
    ) + 1
    prefix = query[clause_start:position]
    if _NEGATION_PREFIX.search(prefix):
        return "negated"
    if _BACKGROUND_PREFIX.search(prefix):
        return "background"
    return "affirmative"
