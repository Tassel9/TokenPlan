"""Validated semantic analysis emitted by the IntentRecognizer."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FineGrainedIntent(str, Enum):
    INSPECTION_STANDARD_QUERY = "inspection_standard_query"
    INSPECTION_TASK_CREATE = "inspection_task_create"
    INSPECTION_TASK_UPDATE = "inspection_task_update"
    INSPECTION_TASK_CANCEL = "inspection_task_cancel"
    ALERT_REPORT = "alert_report"
    WORK_ORDER_HANDLING = "work_order_handling"
    WORK_ORDER_WITHDRAWAL = "work_order_withdrawal"
    TERMINAL_ACCESS_ISSUE = "terminal_access_issue"
    TERMINAL_SECURITY_REQUEST = "terminal_security_request"
    OPERATIONS_PERMISSION_CHANGE = "operations_permission_change"
    FACILITY_TROUBLESHOOTING = "facility_troubleshooting"
    OPERATIONS_COMPLAINT = "operations_complaint"
    OPERATIONS_FEEDBACK = "operations_feedback"


@dataclass(frozen=True)
class IntentSpec:
    """One intent's domain, retrieval language, decision boundary, and confidence anchor."""

    domain: str
    retrieval_text: str
    decision_text: str
    confidence_text: str


INTENT_SPECS: Dict[FineGrainedIntent, IntentSpec] = {
    FineGrainedIntent.INSPECTION_STANDARD_QUERY: IntentSpec(
        domain="巡检管理",
        retrieval_text="巡检规范、检查项目、维护周期、保养要求、设备手册、设施标准、处置流程、适用范围",
        decision_text="查询设备巡检规范、维护周期、检查项目、处置流程或适用范围；创建任务所附带的设备参数不构成独立规范查询",
        confidence_text="查询设备巡检规范、维护周期、检查项目、处置流程或适用范围",
    ),
    FineGrainedIntent.INSPECTION_TASK_CREATE: IntentSpec(
        domain="巡检管理",
        retrieval_text="创建巡检任务、新建巡检、安排巡检、发起巡查、生成巡检计划",
        decision_text="创建新的巡检任务或巡查计划；调整已有计划不属于新建",
        confidence_text="创建巡检任务或巡查计划",
    ),
    FineGrainedIntent.INSPECTION_TASK_UPDATE: IntentSpec(
        domain="巡检管理",
        retrieval_text="调整巡检计划、变更巡检时间、更换巡检人员、修改巡检范围、调整任务优先级",
        decision_text="调整现有巡检任务的时间、人员、范围或优先级；首次创建归入巡检任务创建",
        confidence_text="调整现有巡检任务或巡查计划",
    ),
    FineGrainedIntent.INSPECTION_TASK_CANCEL: IntentSpec(
        domain="巡检管理",
        retrieval_text="取消巡检任务、终止巡查、停止巡检计划、撤销巡检安排",
        decision_text="明确取消尚未完成的巡检任务或终止巡查计划；撤回已提交工单归入工单撤回",
        confidence_text="取消巡检任务或终止巡查计划",
    ),
    FineGrainedIntent.ALERT_REPORT: IntentSpec(
        domain="异常与工单",
        retrieval_text="设备告警、异常上报、重复告警、状态异常、数据突变、离线告警、高温告警、水位告警",
        decision_text="上报设备告警、传感数据异常、设备离线或重复告警；需要分析根因和排查步骤时同时归入设备故障排查",
        confidence_text="设备告警、状态异常、数据突变或异常上报",
    ),
    FineGrainedIntent.WORK_ORDER_HANDLING: IntentSpec(
        domain="异常与工单",
        retrieval_text="维修工单、创建工单、派发工单、工单进度、工单状态、转派工单、关闭工单、催办工单",
        decision_text="查询或办理维修工单的创建、派发、转派、进度、关闭或催办",
        confidence_text="查询或办理维修工单",
    ),
    FineGrainedIntent.WORK_ORDER_WITHDRAWAL: IntentSpec(
        domain="异常与工单",
        retrieval_text="撤回工单、退回工单、驳回工单、取消报修、工单退回原因、撤回进度",
        decision_text="撤回、退回或驳回已提交的维修工单，或查询相关条件与进度；取消尚未执行的巡检计划归入巡检任务取消",
        confidence_text="查询或办理工单撤回、退回或驳回",
    ),
    FineGrainedIntent.TERMINAL_ACCESS_ISSUE: IntentSpec(
        domain="终端与权限",
        retrieval_text="终端离线、设备无法接入、网关连接失败、认证失败、北斗终端掉线、遥测中断、数据不上报",
        decision_text="设备、网关或移动终端无法接入平台，或认证、通信、遥测上报失败；已接入后的安全配置归入终端安全管理",
        confidence_text="终端离线、设备无法接入、认证失败或遥测中断",
    ),
    FineGrainedIntent.TERMINAL_SECURITY_REQUEST: IntentSpec(
        domain="终端与权限",
        retrieval_text="终端密钥、设备证书、接入凭证、访问控制、终端解绑、会话撤销、安全策略",
        decision_text="查询或变更终端密钥、设备证书、接入凭证、绑定关系或安全策略；设备无法接入归入终端接入故障",
        confidence_text="查询或变更终端凭证、证书、绑定关系或安全策略",
    ),
    FineGrainedIntent.OPERATIONS_PERMISSION_CHANGE: IntentSpec(
        domain="终端与权限",
        retrieval_text="开通设备权限、增加点位权限、修改区域权限、调整角色、授权巡检、变更工单权限",
        decision_text="明确要求新增或调整设备、区域、巡检或工单操作权限；只查询现有规范不属于权限变更",
        confidence_text="新增或调整设备、区域、巡检或工单权限",
    ),
    FineGrainedIntent.FACILITY_TROUBLESHOOTING: IntentSpec(
        domain="故障处置",
        retrieval_text="设备故障、泵站异常、路灯故障、井盖告警、传感器漂移、振动异常、温度异常、通信超时、排障步骤",
        decision_text="分析市政设施、传感器、网关或平台接口的故障现象并给出排查步骤；单纯上报告警归入设备异常上报",
        confidence_text="分析设备、传感器、网关或平台接口故障并给出排查步骤",
    ),
    FineGrainedIntent.OPERATIONS_COMPLAINT: IntentSpec(
        domain="运维服务",
        retrieval_text="投诉、不满、追责、处置太慢、工单无人处理、反复报修无结果、长期未解决、要个说法",
        decision_text="对运维处置、工单流转或值守响应表达明确不满、投诉或追责，或说明同一问题反复、长期处理仍无结果；单纯陈述故障或等待事实不等于投诉",
        confidence_text="对运维处置、工单流转或值守响应表达明确不满、投诉或追责",
    ),
    FineGrainedIntent.OPERATIONS_FEEDBACK: IntentSpec(
        domain="运维服务",
        retrieval_text="建议、改进建议、希望新增、增加告警展示、优化巡检流程、意见反馈、运维评价、表扬",
        decision_text="提供运维流程、工作台或处置体验的评价和改进建议，但不要求处理具体故障或工单",
        confidence_text="提供运维流程或处置体验的评价和改进建议",
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
    "facility_id", "work_order_id", "inspection_task_id", "terminal_id",
    "operator_id", "team_id", "permission_scope", "location", "asset_type",
    "alert_code", "date", "error_code",
}
EntityKey = Literal[
    "facility_id", "work_order_id", "inspection_task_id", "terminal_id",
    "operator_id", "team_id", "permission_scope", "location", "asset_type",
    "alert_code", "date", "error_code",
]
_INTENT_ID = re.compile(r"^intent-(\d+)-([a-z_]+)$")
_ERROR_CODE_RE = re.compile(r"(?<!\d)([45]\d{2})(?!\d)")
_FACILITY_RE = re.compile(
    r"(?:设备|设施|泵站|路灯|井盖|终端)(?:编号|ID|号)?\s*[#：:]?\s*([A-Za-z]+-[A-Za-z0-9-]+)",
    flags=re.IGNORECASE,
)
_WORK_ORDER_RE = re.compile(
    r"(?:工单号|工单|work\s*order)\s*[#：:]?\s*([A-Za-z]+-[A-Za-z0-9-]+)",
    flags=re.IGNORECASE,
)
_INSPECTION_TASK_RE = re.compile(
    r"(?:巡检任务|巡查任务|任务)(?:编号|ID|号)?\s*[#：:]?\s*([A-Za-z]+-[A-Za-z0-9-]+)",
    flags=re.IGNORECASE,
)
_TERMINAL_RE = re.compile(
    r"(?:终端|网关)(?:编号|ID|号)?\s*[#：:]?\s*([A-Za-z]+-[A-Za-z0-9-]+)",
    flags=re.IGNORECASE,
)
_ALERT_CODE_RE = re.compile(
    r"(?:告警码|告警代码|alarm\s*code)\s*[#：:]?\s*([A-Za-z0-9-]{3,})",
    flags=re.IGNORECASE,
)


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


SUPERVISOR_ANALYSIS_SCHEMA: Dict[str, Any] = SupervisorAnalysisContract.model_json_schema()


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
            explicit_work_orders = {
                value.casefold() for value in _WORK_ORDER_RE.findall(original_query)
            }
            resolved_work_orders = {
                value.casefold() for value in _WORK_ORDER_RE.findall(effective_query)
            }
            if (
                explicit_work_orders
                and not resolved_work_orders.issubset(explicit_work_orders)
            ):
                raise ValueError("rewrite changed an explicit work order id")
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
            "error_code": list(dict.fromkeys(_ERROR_CODE_RE.findall(text))),
            "facility_id": list(dict.fromkeys(_FACILITY_RE.findall(text))),
            "work_order_id": list(dict.fromkeys(_WORK_ORDER_RE.findall(text))),
            "inspection_task_id": list(dict.fromkeys(_INSPECTION_TASK_RE.findall(text))),
            "terminal_id": list(dict.fromkeys(_TERMINAL_RE.findall(text))),
            "alert_code": list(dict.fromkeys(_ALERT_CODE_RE.findall(text))),
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
            _FACILITY_RE.findall(text)
            + _WORK_ORDER_RE.findall(text)
            + _INSPECTION_TASK_RE.findall(text)
            + _TERMINAL_RE.findall(text)
            + _ALERT_CODE_RE.findall(text)
            + _ERROR_CODE_RE.findall(text)
        ))

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").strip()
