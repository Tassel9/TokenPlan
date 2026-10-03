"""Post-recognition confidence routing for Supervisor intent candidates."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence

from core.supervisor_decision import (
    FineGrainedIntent,
    ScopeStatus,
    SupervisorAnalysis,
    SupervisorIntent,
)
from core.supervisor_few_shot_retriever import FewShotRetrieval, SupervisorFewShotRetriever


class IntentConfidenceBand(str, Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    LOW = "low"

    # Compatibility aliases for existing internal call sites.
    CONFIRMED = "clear"
    CLARIFY = "ambiguous"
    REJECTED = "low"


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
class IntentScoreCalibration:
    """Platt-style score calibration; the defaults are an identity mapping."""

    embedding_scale: float = 1.0
    embedding_bias: float = 0.0
    tree_scale: float = 1.0
    tree_bias: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.embedding_scale,
            self.embedding_bias,
            self.tree_scale,
            self.tree_bias,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("intent calibration parameters must be finite")
        if self.embedding_scale <= 0.0 or self.tree_scale <= 0.0:
            raise ValueError("intent calibration scales must be greater than zero")

    def embedding(self, score: float) -> float:
        return _platt_calibrate(score, self.embedding_scale, self.embedding_bias)

    def tree(self, score: float) -> float:
        return _platt_calibrate(score, self.tree_scale, self.tree_bias)

    def to_dict(self) -> Dict[str, float]:
        return {
            "embedding_scale": self.embedding_scale,
            "embedding_bias": self.embedding_bias,
            "tree_scale": self.tree_scale,
            "tree_bias": self.tree_bias,
        }


@dataclass(frozen=True)
class IntentConfidenceDecision:
    intent: SupervisorIntent
    band: IntentConfidenceBand
    matching_score: float
    raw_embedding_score: float
    raw_tree_score: float
    embedding_score: float
    tree_score: float
    final_score: float
    fusion_alpha: float
    similarity_margin: float
    positive_similarity: float
    negative_similarity: float
    positive_source: str
    negative_source: str
    reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.intent.to_dict(),
            "band": self.band.value,
            "matching_score": round(self.matching_score, 6),
            "raw_embedding_score": round(self.raw_embedding_score, 6),
            "raw_tree_score": round(self.raw_tree_score, 6),
            "embedding_score": round(self.embedding_score, 6),
            "tree_score": round(self.tree_score, 6),
            "final_score": round(self.final_score, 6),
            "fusion_alpha": round(self.fusion_alpha, 6),
            "similarity_margin": round(self.similarity_margin, 6),
            "positive_similarity": round(self.positive_similarity, 6),
            "negative_similarity": round(self.negative_similarity, 6),
            "positive_source": self.positive_source,
            "negative_source": self.negative_source,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class IntentConfidenceAssessment:
    status: str
    decisions: tuple[IntentConfidenceDecision, ...]
    recall_threshold: float
    recommendation_threshold: float
    fusion_alpha: float
    calibration: IntentScoreCalibration
    error: str = ""

    def intents_for(self, band: IntentConfidenceBand) -> tuple[SupervisorIntent, ...]:
        return tuple(item.intent for item in self.decisions if item.band == band)

    @property
    def confirmed(self) -> tuple[SupervisorIntent, ...]:
        return self.intents_for(IntentConfidenceBand.CLEAR)

    @property
    def clarification_candidates(self) -> tuple[SupervisorIntent, ...]:
        return self.intents_for(IntentConfidenceBand.AMBIGUOUS)

    @property
    def rejected(self) -> tuple[SupervisorIntent, ...]:
        return self.intents_for(IntentConfidenceBand.LOW)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "strategy": "parallel_embedding_llm_tree_calibrated_fusion_v1",
            "recall_threshold": self.recall_threshold,
            "recommendation_threshold": self.recommendation_threshold,
            "fusion_alpha": self.fusion_alpha,
            "calibration": self.calibration.to_dict(),
            "confirmed_intent_ids": [item.intent_id for item in self.confirmed],
            "clarification_intent_ids": [
                item.intent_id for item in self.clarification_candidates
            ],
            "rejected_intent_ids": [item.intent_id for item in self.rejected],
            "decisions": [item.to_dict() for item in self.decisions],
            "error": self.error,
        }


class SupervisorIntentConfidencePolicy:
    """Fuse independent Embedding and LLM-tree scores into three-state routing."""

    def __init__(
        self,
        retriever: SupervisorFewShotRetriever,
        *,
        recall_threshold: float = 0.40,
        recommendation_threshold: float = 0.34,
        fusion_alpha: float = 0.50,
        embedding_calibration_scale: float = 1.0,
        embedding_calibration_bias: float = 0.0,
        tree_calibration_scale: float = 1.0,
        tree_calibration_bias: float = 0.0,
    ) -> None:
        recall = float(recall_threshold)
        recommendation = float(recommendation_threshold)
        if not 0.0 <= recommendation < recall <= 1.0:
            raise ValueError(
                "intent confidence thresholds must satisfy "
                "0 <= recommendation < recall <= 1"
            )
        alpha = float(fusion_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("intent fusion alpha must be between 0 and 1")
        # Keep the dependency explicit: this policy consumes the retriever's
        # independently produced, all-label score vector.
        if retriever is None:
            raise ValueError("intent confidence policy requires a retriever")
        self.recall_threshold = recall
        self.recommendation_threshold = recommendation
        self.fusion_alpha = alpha
        self.calibration = IntentScoreCalibration(
            embedding_scale=float(embedding_calibration_scale),
            embedding_bias=float(embedding_calibration_bias),
            tree_scale=float(tree_calibration_scale),
            tree_bias=float(tree_calibration_bias),
        )

    async def assess(
        self,
        original_query: str,
        analysis: SupervisorAnalysis,
        retrieval: FewShotRetrieval,
    ) -> IntentConfidenceAssessment:
        if analysis.scope_status != ScopeStatus.IN_SCOPE or not analysis.intents:
            return IntentConfidenceAssessment(
                "not_applicable",
                (),
                self.recall_threshold,
                self.recommendation_threshold,
                self.fusion_alpha,
                self.calibration,
            )
        try:
            if retrieval.status != "ok":
                raise RuntimeError("embedding channel did not complete successfully")
            scores = {item.label: item for item in retrieval.intent_scores}
            required_labels = {
                item.label.value for item in analysis.intents
            } | set(retrieval.candidate_intents)
            missing = sorted(required_labels - set(scores))
            if missing:
                raise RuntimeError(
                    "embedding score coverage mismatch: " + ", ".join(missing)
                )
        except Exception as ex:
            return IntentConfidenceAssessment(
                "failed",
                (),
                self.recall_threshold,
                self.recommendation_threshold,
                self.fusion_alpha,
                self.calibration,
                f"{type(ex).__name__}: {str(ex)[:240]}",
            )
        decisions = []
        tree_intents = {item.label.value: item for item in analysis.intents}
        ordered_labels = [item.label.value for item in analysis.intents]
        ordered_labels.extend(
            label for label in retrieval.candidate_intents
            if label not in tree_intents
        )
        for index, label in enumerate(ordered_labels, start=1):
            score = scores[label]
            intent = tree_intents.get(label)
            raw_embedding_score = score.matching_score
            raw_tree_score = intent.tree_score if intent is not None else 0.0
            if intent is None:
                intent = SupervisorIntent(
                    intent_id=f"intent-{index}-{label}",
                    label=FineGrainedIntent(label),
                    supporting_text=(original_query,),
                    tree_score=0.0,
                )
            embedding_score = self.calibration.embedding(raw_embedding_score)
            tree_score = self.calibration.tree(raw_tree_score)
            final_score = (
                self.fusion_alpha * embedding_score
                + (1.0 - self.fusion_alpha) * tree_score
            )
            conflict = _evidence_conflict(original_query, intent)
            if conflict:
                band = IntentConfidenceBand.LOW
                reason_code = conflict
            elif (
                label in tree_intents
                and final_score >= self.recall_threshold
            ):
                band = IntentConfidenceBand.CLEAR
                reason_code = "above_fused_clear_threshold"
            elif final_score >= self.recommendation_threshold:
                band = IntentConfidenceBand.AMBIGUOUS
                reason_code = (
                    "between_fused_thresholds"
                    if label in tree_intents
                    else "embedding_only_requires_tree_evidence"
                )
            else:
                band = IntentConfidenceBand.LOW
                reason_code = "below_fused_low_threshold"
            decisions.append(IntentConfidenceDecision(
                intent=intent,
                band=band,
                matching_score=final_score,
                raw_embedding_score=raw_embedding_score,
                raw_tree_score=raw_tree_score,
                embedding_score=embedding_score,
                tree_score=tree_score,
                final_score=final_score,
                fusion_alpha=self.fusion_alpha,
                similarity_margin=score.similarity_margin,
                positive_similarity=score.positive_similarity,
                negative_similarity=score.negative_similarity,
                positive_source=score.positive_source,
                negative_source=score.negative_source,
                reason_code=reason_code,
            ))
        return IntentConfidenceAssessment(
            "ok",
            tuple(decisions),
            self.recall_threshold,
            self.recommendation_threshold,
            self.fusion_alpha,
            self.calibration,
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


def _platt_calibrate(score: float, scale: float, bias: float) -> float:
    """Map a bounded raw score to a calibrated probability."""
    raw = min(1.0, max(0.0, float(score)))
    if scale == 1.0 and bias == 0.0:
        return raw
    epsilon = 1e-6
    bounded = min(1.0 - epsilon, max(epsilon, raw))
    logit = math.log(bounded / (1.0 - bounded))
    transformed = scale * logit + bias
    if transformed >= 0.0:
        return 1.0 / (1.0 + math.exp(-transformed))
    exp_value = math.exp(transformed)
    return exp_value / (1.0 + exp_value)


def _evidence_conflict(original_query: str, intent: SupervisorIntent) -> str:
    """Reject evidence that occurs only in explicit negation or quoted background."""
    anchors = _INTENT_ANCHORS[intent.label]
    positions = list(_supported_anchor_positions(original_query, intent.supporting_text, anchors))
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
