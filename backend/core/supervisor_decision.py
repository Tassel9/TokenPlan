"""Validated semantic analysis emitted by the IntentRecognizer."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FineGrainedIntent(str, Enum):
    SUBSCRIPTION_INFO_QUERY = "subscription_info_query"
    SUBSCRIPTION_PURCHASE = "subscription_purchase"
    SUBSCRIPTION_CHANGE = "subscription_change"
    SUBSCRIPTION_CANCEL = "subscription_cancel"
    PAYMENT_ISSUE = "payment_issue"
    INVOICE_HANDLING = "invoice_handling"
    REFUND_HANDLING = "refund_handling"
    ACCOUNT_LOGIN_ISSUE = "account_login_issue"
    ACCOUNT_SECURITY_REQUEST = "account_security_request"
    ENTITLEMENT_CHANGE_REQUEST = "entitlement_change_request"
    TECHNICAL_TROUBLESHOOTING = "technical_troubleshooting"
    SERVICE_COMPLAINT = "service_complaint"
    SERVICE_FEEDBACK = "service_feedback"


@dataclass(frozen=True)
class IntentSpec:
    """One intent's domain, retrieval language, decision boundary, and confidence anchor."""

    domain: str
    retrieval_text: str
    decision_text: str
    confidence_text: str


INTENT_SPECS: Dict[FineGrainedIntent, IntentSpec] = {
    FineGrainedIntent.SUBSCRIPTION_INFO_QUERY: IntentSpec(
        domain="订阅管理",
        retrieval_text="套餐价格、多少钱、试用规则、免费试用、版本区别、套餐对比、模型权限、Token额度、用量配额、席位、支持范围、套餐包含内容",
        decision_text="查询套餐价格、试用规则、版本差异、模型权限、Token额度、席位或支持范围；购买参数本身不构成独立查询诉求",
        confidence_text="查询套餐价格、试用规则、版本差异、模型权限、Token额度、席位或支持范围",
    ),
    FineGrainedIntent.SUBSCRIPTION_PURCHASE: IntentSpec(
        domain="订阅管理",
        retrieval_text="购买套餐、买订阅、开通订阅、新购套餐、订阅服务、下单购买",
        decision_text="购买或开通新的订阅；升级、降级或变更已有订阅不属于新购",
        confidence_text="购买或开通订阅",
    ),
    FineGrainedIntent.SUBSCRIPTION_CHANGE: IntentSpec(
        domain="订阅管理",
        retrieval_text="升级套餐、降级套餐、更换套餐、调整套餐、变更订阅、切换版本",
        decision_text="升级、降级或变更现有订阅；首次购买归入订阅购买，增加额度或席位归入权益变更",
        confidence_text="升级、降级或变更现有订阅",
    ),
    FineGrainedIntent.SUBSCRIPTION_CANCEL: IntentSpec(
        domain="订阅管理",
        retrieval_text="取消订阅、退订、停止订阅、关闭自动续费、不再续费、终止套餐",
        decision_text="明确停止现有订阅或关闭未来自动续费；仅撤销刚发生的购买并要求原路退回款项归入退款处理",
        confidence_text="明确停止现有订阅或关闭自动续费；仅撤销刚发生的购买并原路退款归入退款处理",
    ),
    FineGrainedIntent.PAYMENT_ISSUE: IntentSpec(
        domain="支付与账务",
        retrieval_text="支付失败、付款失败、扣款异常、金额不对、重复扣款、扣了两次、付款方式故障、银行卡支付、付款页面、支付按钮",
        decision_text="付款动作失败、金额异常、重复扣款或付款方式故障；付款成功后的业务API、IDE、索引或模型调用故障不属于支付问题",
        confidence_text="支付失败、金额异常、重复扣款或付款方式故障",
    ),
    FineGrainedIntent.INVOICE_HANDLING: IntentSpec(
        domain="支付与账务",
        retrieval_text="发票、开发票、开票、为订单开票、补差价开票、公司抬头发票、电子发票、发票抬头、税号、重开发票、修改发票、发票状态",
        decision_text="查询或办理发票规则、状态、开具、重开或修改",
        confidence_text="查询或办理发票规则、状态、开具、重开或修改",
    ),
    FineGrainedIntent.REFUND_HANDLING: IntentSpec(
        domain="支付与账务",
        retrieval_text="退款、退钱、原路退回、撤销购买、退款条件、退款材料、退款进度、退款未到账、多久到账、钱什么时候回来",
        decision_text="办理退款，或查询退款条件、材料、进度和到账时间；明确否定退款不得成立，停止未来自动续费归入取消订阅",
        confidence_text="查询或办理退款条件、材料、进度或到账时间",
    ),
    FineGrainedIntent.ACCOUNT_LOGIN_ISSUE: IntentSpec(
        domain="账号与权益",
        retrieval_text="无法登录、登录不了、忘记密码、账号锁定、认证失败、验证失败、进不去账号、登录报错",
        decision_text="忘记密码、账号锁定、认证失败或无法登录；能够登录后主动修改密码属于账号安全请求",
        confidence_text="忘记密码、账号锁定、认证失败或无法登录",
    ),
    FineGrainedIntent.ACCOUNT_SECURITY_REQUEST: IntentSpec(
        domain="账号与权益",
        retrieval_text="修改密码、更换邮箱、两步验证、双重验证、2FA、设备会话、退出设备、账号注销、安全设置",
        decision_text="查询或主动变更密码、邮箱、两步验证、设备会话或账号注销；因忘记密码或认证失败而无法登录归入登录问题",
        confidence_text="查询或变更密码、邮箱、两步验证、设备会话或账号注销",
    ),
    FineGrainedIntent.ENTITLEMENT_CHANGE_REQUEST: IntentSpec(
        domain="账号与权益",
        retrieval_text="增加额度、提升Token额度、扩容、开通模型权限、增加席位、添加席位、修改工作区资格、变更权益",
        decision_text="明确要求增加额度、开通模型权限、增加席位或修改工作区资格；只查询现有权益归入订阅信息，只报告现有权益不可用不算变更请求",
        confidence_text="明确要求增加额度、开通模型权限、增加席位或修改工作区资格；只报告现有权益不可用不算变更请求",
    ),
    FineGrainedIntent.TECHNICAL_TROUBLESHOOTING: IntentSpec(
        domain="技术支持",
        retrieval_text="IDE插件、代码补全、业务API、接口报错、索引故障、网络错误、请求超时、4xx、5xx、模型调用失败、技术排查",
        decision_text="排查IDE插件、代码补全、业务API、索引、网络、超时或4xx/5xx故障；支付按钮或付款页面失败只归支付问题，忘记密码、账号锁定、登录认证失败或登录后回跳只归登录问题",
        confidence_text="排查IDE插件、代码补全、业务API、索引、网络、超时或4xx/5xx故障；支付按钮或付款页面失败只归支付问题",
    ),
    FineGrainedIntent.SERVICE_COMPLAINT: IntentSpec(
        domain="服务体验",
        retrieval_text="投诉、不满、追责、差评、服务太差、客服不处理、反复处理无结果、长期没有结果、一直没人解决、账单不合理、产品很糟糕、要个说法",
        decision_text="对产品、账单或客服处理表达明确评价性不满、投诉或追责，或说明同一服务问题经反复、长期处理仍无结果；单纯陈述故障、扣费或等待事实不等于投诉",
        confidence_text="对产品、账单或客服处理表达明确评价性不满、投诉或追责；故障或金额异常事实本身不等于投诉",
    ),
    FineGrainedIntent.SERVICE_FEEDBACK: IntentSpec(
        domain="服务体验",
        retrieval_text="建议、改进建议、问题已解决后的改进建议、建议以后、希望新增、增加展示、显示状态、意见反馈、产品反馈、体验优化、表扬、好评、做得不错",
        decision_text="提供正面评价或改进建议，但不要求处理具体故障、账务或账号问题",
        confidence_text="提供正面评价或改进建议，但不要求处理具体故障",
    ),
}


# Compatibility view for the existing Supervisor prompt and validators.  The
# richer INTENT_SPECS mapping remains the single source of truth.
INTENT_DEFINITIONS: Dict[FineGrainedIntent, str] = {
    intent: spec.decision_text for intent, spec in INTENT_SPECS.items()
}


class RewriteStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class ScopeStatus(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNCERTAIN = "uncertain"


ENTITY_KEYS = {
    "order_id", "account_email", "workspace_id", "plan", "model", "ide",
    "date", "amount", "error_code",
}
EntityKey = Literal[
    "order_id", "account_email", "workspace_id", "plan", "model", "ide",
    "date", "amount", "error_code",
]
_INTENT_ID = re.compile(r"^intent-(\d+)-([a-z_]+)$")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ORDER_RE = re.compile(
    r"(?:订单号|订单|order(?:\s*id)?)\s*[#：:]?\s*([A-Za-z0-9-]{4,})",
    flags=re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"(?:¥|￥|\$)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*元")
_ERROR_CODE_RE = re.compile(r"(?<!\d)([45]\d{2})(?!\d)")


class RewriteReferenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    source: str = Field(
        pattern=r"^(case\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*(?:\[\d+\])?|history\[\d+\])$"
    )
    value: str = Field(min_length=1)


class SupervisorRewriteContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RewriteStatus
    effective_query: str
    references: List[RewriteReferenceContract]
    extracted_entities: Dict[EntityKey, List[str]]
    inherited_entities: Dict[EntityKey, List[str]]
    ambiguity_candidates: Dict[EntityKey, List[str]]
    clarification_question: str
    reason_code: str


class SupervisorIntentContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(pattern=r"^intent-\d+-[a-z_]+$")
    label: FineGrainedIntent
    supporting_text: List[str] = Field(min_length=1)
    tree_score: float = Field(ge=0.0, le=1.0)


class SupervisorAnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewrite: SupervisorRewriteContract
    intents: List[SupervisorIntentContract]
    scope_status: ScopeStatus
    reason_code: str


def _inline_local_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline Pydantic local refs before nesting this schema in a tool."""

    definitions = dict(schema.get("$defs") or {})

    def expand(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            if name not in definitions:
                raise ValueError(f"unknown local schema reference: {reference}")
            if name in stack:
                raise ValueError(f"recursive local schema reference: {reference}")
            merged = deepcopy(definitions[name])
            merged.update({key: item for key, item in value.items() if key != "$ref"})
            return expand(merged, stack + (name,))
        return {
            key: expand(item, stack)
            for key, item in value.items()
            if key != "$defs"
        }

    expanded = expand(deepcopy(schema))
    if not isinstance(expanded, dict):
        raise ValueError("expanded schema must be an object")
    return expanded


SUPERVISOR_ANALYSIS_SCHEMA: Dict[str, Any] = SupervisorAnalysisContract.model_json_schema()
SUPERVISOR_ANALYSIS_TOOL_SCHEMA: Dict[str, Any] = _inline_local_schema_refs(
    SUPERVISOR_ANALYSIS_SCHEMA
)


@dataclass(frozen=True)
class RewriteReference:
    mention: str
    source: str
    value: str


@dataclass(frozen=True)
class SupervisorRewrite:
    status: RewriteStatus
    effective_query: str
    references: tuple[RewriteReference, ...] = ()
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    inherited_entities: Dict[str, List[str]] = field(default_factory=dict)
    ambiguity_candidates: Dict[str, List[str]] = field(default_factory=dict)
    clarification_question: str = ""
    reason_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "effective_query": self.effective_query,
            "references": [asdict(item) for item in self.references],
            "extracted_entities": {k: list(v) for k, v in self.extracted_entities.items()},
            "inherited_entities": {k: list(v) for k, v in self.inherited_entities.items()},
            "ambiguity_candidates": {k: list(v) for k, v in self.ambiguity_candidates.items()},
            "clarification_question": self.clarification_question,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class SupervisorIntent:
    intent_id: str
    label: FineGrainedIntent
    supporting_text: tuple[str, ...]
    tree_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "label": self.label.value,
            "supporting_text": list(self.supporting_text),
            "tree_score": round(self.tree_score, 6),
        }


@dataclass(frozen=True)
class SupervisorAnalysis:
    rewrite: SupervisorRewrite
    intents: tuple[SupervisorIntent, ...]
    scope_status: ScopeStatus
    reason_code: str = ""

    @property
    def intent_labels(self) -> List[FineGrainedIntent]:
        return [item.label for item in self.intents]

    @property
    def intent_rows(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.intents]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rewrite": self.rewrite.to_dict(),
            "intents": self.intent_rows,
            "scope_status": self.scope_status.value,
            "reason_code": self.reason_code,
        }


class SupervisorDecisionValidator:
    """Fail-closed validation for first-round Supervisor semantics."""

    @classmethod
    def validate_analysis(
        cls,
        raw: Any,
        *,
        original_query: str,
        case_state: Optional[Mapping[str, Any]] = None,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> SupervisorAnalysis:
        if not isinstance(raw, Mapping):
            raise ValueError("first-round Supervisor decision requires analysis")
        try:
            raw = SupervisorAnalysisContract.model_validate(raw).model_dump(mode="json")
        except ValidationError as ex:
            first = ex.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first.get("loc", ())) or "analysis"
            raise ValueError(
                f"invalid Supervisor analysis structure at {location}: {first.get('msg', 'invalid value')}"
            ) from ex
        allowed = {"rewrite", "intents", "scope_status", "reason_code"}
        if set(raw) - allowed:
            raise ValueError("Supervisor analysis contains unknown fields")
        rewrite = cls._validate_rewrite(
            raw.get("rewrite"), original_query=original_query,
            case_state=case_state or {}, history=history or [],
        )
        try:
            scope_status = ScopeStatus(str(raw.get("scope_status", "")).strip())
        except ValueError as ex:
            raise ValueError("invalid scope_status") from ex
        intents = cls._validate_intents(raw.get("intents"), original_query)
        if scope_status == ScopeStatus.IN_SCOPE and not intents:
            raise ValueError("in_scope analysis requires at least one intent")
        if scope_status != ScopeStatus.IN_SCOPE and intents:
            raise ValueError("out_of_scope or uncertain analysis cannot contain intents")
        return SupervisorAnalysis(
            rewrite=rewrite,
            intents=tuple(intents),
            scope_status=scope_status,
            reason_code=cls._clean(raw.get("reason_code"))[:200],
        )

    @classmethod
    def _validate_rewrite(
        cls,
        raw: Any,
        *,
        original_query: str,
        case_state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> SupervisorRewrite:
        if not isinstance(raw, Mapping):
            raise ValueError("analysis.rewrite must be an object")
        allowed = {
            "status", "effective_query", "references", "extracted_entities",
            "inherited_entities", "ambiguity_candidates", "clarification_question",
            "reason_code",
        }
        if set(raw) - allowed:
            raise ValueError("rewrite contains unknown fields")
        try:
            status = RewriteStatus(str(raw.get("status", "")).strip())
        except ValueError as ex:
            raise ValueError("invalid rewrite status") from ex
        effective_query = cls._clean(raw.get("effective_query"))
        references = cls._references(raw.get("references", []))
        extracted = cls._entities(raw.get("extracted_entities", {}), original_query)
        evidence_text = cls._evidence_text(original_query, case_state, history)
        inherited = cls._entities(raw.get("inherited_entities", {}), evidence_text)
        ambiguity = cls._ambiguity(raw.get("ambiguity_candidates", {}))
        clarification = cls._clean(raw.get("clarification_question"))[:1000]

        if status == RewriteStatus.NOT_NEEDED:
            if effective_query != original_query or references or inherited or ambiguity:
                raise ValueError("not_needed rewrite must preserve the original query")
        elif status == RewriteStatus.RESOLVED:
            if not effective_query or effective_query == original_query or not references:
                raise ValueError("resolved rewrite requires a changed query and references")
            for reference in references:
                source_evidence = cls._reference_evidence(reference.source, case_state, history)
                if not source_evidence or not cls._reference_grounded(
                    reference.value, source_evidence
                ):
                    raise ValueError("rewrite reference is not grounded in its source")
            for value in cls._guarded_literals(effective_query):
                if value.casefold() not in evidence_text.casefold():
                    raise ValueError("rewrite introduced an ungrounded sensitive literal")
            explicit_orders = {value.casefold() for value in _ORDER_RE.findall(original_query)}
            resolved_orders = {value.casefold() for value in _ORDER_RE.findall(effective_query)}
            if explicit_orders and not resolved_orders.issubset(explicit_orders):
                raise ValueError("rewrite changed an explicit order id")
        elif status == RewriteStatus.AMBIGUOUS:
            if not ambiguity or not clarification:
                raise ValueError("ambiguous rewrite requires candidates and a question")
            if any(len(values) < 2 for values in ambiguity.values()):
                raise ValueError("ambiguous rewrite requires multiple candidates")
            effective_query = original_query
            references = []
            inherited = {}
        else:
            effective_query = original_query
            references = []
            inherited = {}

        return SupervisorRewrite(
            status=status,
            effective_query=effective_query,
            references=tuple(references),
            extracted_entities=extracted,
            inherited_entities=inherited,
            ambiguity_candidates=ambiguity,
            clarification_question=clarification,
            reason_code=cls._clean(raw.get("reason_code"))[:200],
        )

    @classmethod
    def _validate_intents(
        cls,
        raw: Any,
        original_query: str,
    ) -> List[SupervisorIntent]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("analysis.intents must be an array")
        result: List[SupervisorIntent] = []
        ids: set[str] = set()
        labels: set[FineGrainedIntent] = set()
        lowered = original_query.casefold()
        for index, row in enumerate(raw, start=1):
            if not isinstance(row, Mapping) or set(row) != {
                "intent_id", "label", "supporting_text", "tree_score"
            }:
                raise ValueError("invalid Supervisor intent object")
            try:
                label = FineGrainedIntent(str(row.get("label", "")).strip())
            except ValueError as ex:
                raise ValueError("unknown Supervisor intent label") from ex
            intent_id = cls._clean(row.get("intent_id"))
            match = _INTENT_ID.fullmatch(intent_id)
            if not match or match.group(2) != label.value:
                # 位置编号只做格式归一：模型偶尔按“估计序号”编号（跳号/乱序），
                # 此类纯格式问题不再 fail-closed，输出端会统一重排为
                # intent-{position}-{label}，避免整轮识别被误伤。
                raise ValueError("intent_id must embed its label")
            support_raw = row.get("supporting_text")
            if not isinstance(support_raw, Sequence) or isinstance(support_raw, (str, bytes)):
                raise ValueError("supporting_text must be an array")
            support = tuple(dict.fromkeys(
                cls._clean(value) for value in support_raw if cls._clean(value)
            ))
            if not support or any(value.casefold() not in lowered for value in support):
                raise ValueError("intent supporting_text must quote the current query")
            tree_score = row.get("tree_score")
            if (
                isinstance(tree_score, bool)
                or not isinstance(tree_score, (int, float))
                or not 0.0 <= float(tree_score) <= 1.0
            ):
                raise ValueError("intent tree_score must be between 0 and 1")
            if intent_id in ids:
                raise ValueError("duplicate Supervisor intent")
            ids.add(intent_id)
            if label in labels:
                # 同一标签重复出现（模型常按“每个诉求一条”拆分输出）：规范形态是
                # 一条 intent 携带多个 supporting_text；这里合并而非拒绝
                #（支持文本按序合并去重、tree_score 取较大值）。
                for position, existing in enumerate(result):
                    if existing.label != label:
                        continue
                    result[position] = SupervisorIntent(
                        existing.intent_id,
                        existing.label,
                        tuple(dict.fromkeys(existing.supporting_text + support)),
                        max(existing.tree_score, float(tree_score)),
                    )
                    break
                continue
            labels.add(label)
            result.append(SupervisorIntent(intent_id, label, support, float(tree_score)))
        # 合并可能令 intent_id 与位置脱节，统一重排为 "intent-{i}-{label}" 规范形
        #（无合并时为恒等变换）。
        return [
            SupervisorIntent(
                f"intent-{position}-{item.label.value}",
                item.label,
                item.supporting_text,
                item.tree_score,
            )
            for position, item in enumerate(result, start=1)
        ]

    @classmethod
    def extract_explicit_entities(cls, text: str) -> Dict[str, List[str]]:
        values = {
            "account_email": list(dict.fromkeys(_EMAIL_RE.findall(text))),
            "order_id": list(dict.fromkeys(_ORDER_RE.findall(text))),
            "amount": list(dict.fromkeys(_AMOUNT_RE.findall(text))),
            "error_code": list(dict.fromkeys(_ERROR_CODE_RE.findall(text))),
        }
        return {key: bucket for key, bucket in values.items() if bucket}

    @classmethod
    def _entities(cls, raw: Any, evidence: str) -> Dict[str, List[str]]:
        if not isinstance(raw, Mapping) or set(raw) - ENTITY_KEYS:
            raise ValueError("invalid entity object")
        lowered = evidence.casefold()
        result: Dict[str, List[str]] = {}
        for key, values in raw.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("entity values must be arrays")
            bucket = list(dict.fromkeys(cls._clean(value) for value in values if cls._clean(value)))
            if any(value.casefold() not in lowered for value in bucket):
                raise ValueError("entity value is not grounded")
            if bucket:
                result[str(key)] = bucket
        return result

    @classmethod
    def _references(cls, raw: Any) -> List[RewriteReference]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("rewrite.references must be an array")
        result: List[RewriteReference] = []
        for row in raw:
            if not isinstance(row, Mapping) or set(row) != {"mention", "source", "value"}:
                raise ValueError("invalid rewrite reference")
            source = cls._clean(row.get("source"))
            if not re.fullmatch(
                r"case\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*(?:\[\d+\])?|history\[\d+\]",
                source,
            ):
                raise ValueError("invalid rewrite reference source")
            item = RewriteReference(
                mention=cls._clean(row.get("mention")),
                source=source,
                value=cls._clean(row.get("value")),
            )
            if not item.mention or not item.value:
                raise ValueError("incomplete rewrite reference")
            result.append(item)
        return result

    @classmethod
    def _ambiguity(cls, raw: Any) -> Dict[str, List[str]]:
        if not isinstance(raw, Mapping) or set(raw) - ENTITY_KEYS:
            raise ValueError("invalid ambiguity_candidates")
        result: Dict[str, List[str]] = {}
        for key, values in raw.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("ambiguity candidates must be arrays")
            bucket = list(dict.fromkeys(cls._clean(value) for value in values if cls._clean(value)))
            if bucket:
                result[str(key)] = bucket
        return result

    @classmethod
    def _reference_grounded(cls, value: str, evidence: str) -> bool:
        """引用值是否可在来源证据中核验（允许大小写与空白差异）。"""
        if value.casefold() in evidence.casefold():
            return True
        normalized = re.sub(r"\s+", "", value).casefold()
        return bool(normalized) and normalized in re.sub(r"\s+", "", evidence).casefold()

    @classmethod
    def _reference_evidence(
        cls,
        source: str,
        case_state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        if source.startswith("case."):
            match = re.fullmatch(
                r"case\.(?P<path>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)(?:\[(?P<index>\d+)\])?",
                source,
            )
            if not match:
                return ""
            current: Any = case_state
            for part in match.group("path").split("."):
                if not isinstance(current, Mapping) or part not in current:
                    return ""
                current = current[part]
            if match.group("index") is not None:
                if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                    return ""
                index = int(match.group("index"))
                if index >= len(current):
                    return ""
                current = current[index]
            return json.dumps(current, ensure_ascii=False) if not isinstance(current, str) else current
        match = re.fullmatch(r"history\[(\d+)\]", source)
        if not match:
            return ""
        index = int(match.group(1))
        if index >= len(history):
            return ""
        return cls._clean(history[index].get("content", ""))

    @classmethod
    def _evidence_text(
        cls,
        query: str,
        case_state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        return "\n".join((
            query,
            json.dumps(case_state, ensure_ascii=False, sort_keys=True),
            *[cls._clean(item.get("content", "")) for item in history],
        ))

    @classmethod
    def _guarded_literals(cls, text: str) -> List[str]:
        return list(dict.fromkeys(
            _EMAIL_RE.findall(text)
            + _ORDER_RE.findall(text)
            + _AMOUNT_RE.findall(text)
        ))

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").strip()
