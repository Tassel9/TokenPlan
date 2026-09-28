"""Deterministic evidence and safety checks applied before sending a reply."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set


@dataclass
class GuardResult:
    response: str
    passed: bool
    escalated: bool
    reason_code: str
    findings: List[Dict[str, Any]] = field(default_factory=list)


class ResponseGuard:
    """Block unsafe knowledge, unsupported claims, and secret requests."""

    SAFE_KNOWLEDGE_CONFLICT_RESPONSE = (
        "知识库中关于该信息的资料存在版本冲突，当前无法可靠确认哪一版仍然生效。"
        "请以 UrbanOps 运维规范库或业务平台为准，并转人工运维人员核验后再作结论。"
    )
    SAFE_STALE_KNOWLEDGE_RESPONSE = (
        "现有知识资料缺少可验证的生效时间或已超过复核周期，当前无法确认它仍是最新规则。"
        "请以 UrbanOps 运维规范库或业务平台为准，并转人工运维人员核验。"
    )
    SAFE_NOT_APPLICABLE_RESPONSE = (
        "检索到的规范不适用于当前区域、设施类型或工况，系统不会据此给出确定结论。"
        "请补充设施类型、所在区域与工况信息，或通过 UrbanOps 业务平台核验。"
    )
    SAFE_NOT_EFFECTIVE_RESPONSE = (
        "检索到的规则在所查询时间尚未生效，系统不会把它作为当时的有效规则。"
        "请通过 UrbanOps 规范库或人工运维人员核验对应时间的规则。"
    )
    SAFE_VERSION_LOOKUP_UNAVAILABLE_RESPONSE = (
        "当前无法补齐并核验同一规则的全部版本，因此不能确认初检资料是否仍然有效。"
        "请稍后重试，或通过 UrbanOps 规范库与人工运维人员核验。"
    )

    WRITE_CLAIM = re.compile(
        r"(?:已经|已)(?:为您)?(?:创建工单|派发工单|转派工单|关闭工单|撤回工单|退回工单|"
        r"取消巡检|修改巡检计划|远程启动设备|远程停止设备|复位设备|解除告警|更新证书|变更权限)"
    )
    BUSINESS_READ_CLAIM = re.compile(
        r"(?:已经|已)(?:为您)?(?:查到|查询到|核实到|确认).{0,16}"
        r"(?:设备状态|设施状态|遥测数据|巡检记录|告警状态|工单状态|工单进度|"
        r"派单记录|处置记录|终端状态|权限状态)"
    )
    SENSITIVE = re.compile(r"(?:请|需要).{0,10}(?:提供|发送|告知|输入).{0,12}(?:平台口令|设备密码|完整私钥|设备证书|API密钥|访问令牌|接入令牌)")
    SECRET_WARNING = re.compile(
        r"(?:请)?(?:不要|切勿|勿|不应|无需|不需要|不必|不得|不能)"
        r"(?![^，,。；;！？!?\n]{0,16}(?:拒绝|忘记|停止|避免))"
        r"[^，,。；;！？!?\n]{0,16}?(?:提供|发送|告知|输入)"
        r"[^，,。；;！？!?\n]{0,12}?(?:平台口令|设备密码|密码|完整私钥|设备证书|API密钥|访问令牌|接入令牌)"
        # 警告句常带同类密名词枚举（如“…密码、验证码、恢复码或完整卡号”）；
        # 必须整段剥离，否则残留的“、验证码”会被 SENSITIVE 误读为索要验证码。
        # 枚举只吞“分隔符+密名词”，遇到后续真实索要语句（如“，请提供登录密码”）仍会保留并拦截。
        r"(?:\s*(?:、|,|，|或|以及|和)\s*"
        r"(?:平台口令|设备密码|密码|完整私钥|设备证书|API密钥|访问令牌|接入令牌))*"
    )
    # 引用式警告（如：请勿相信任何声称“已关闭工单”的说法）不应被
    # WRITE_CLAIM 视作“声称已完成写操作”；仅剔除警告子句，警告句之前真正
        # 的声称句（“已为您关闭工单。请勿相信……”）仍会被保留并拦截。
    DISBELIEF_WARNING = re.compile(
        r"(?:请勿|不要|切勿|勿|别|不应|不得|不能|无需)"
        r"[^。；;！？!?\n]{0,12}?(?:相信|轻信|误信)"
        r"[^。；;！？!?\n]{0,40}"
    )
    # 条件/疑问式提及的剥离规则：仅用于写操作声称检查。“是否已派单”“若已关闭工单”
    # 属于假设或询问，不是“声称已完成操作”；真正的肯定式声称（“已为您派单”）
    # 仍会被 WRITE_CLAIM 拦截。
    CONDITIONAL_CLAUSE = re.compile(
        r"(?:是否|若|倘若|如果|假如|要是|一旦|万一)"
        r"(?:已经|已)(?:为您)?"
        r"(?:创建工单|派发工单|转派工单|关闭工单|撤回工单|退回工单|取消巡检|修改巡检计划|"
        r"远程启动设备|远程停止设备|复位设备|解除告警|更新证书|变更权限)"
    )
    # 描述性状态片段（如“已关闭的工单”）：其中的“已+X”是名词修饰，
    # 不是“声称已完成操作”；同样只在写操作声称检查前剥离。
    STATE_DESCRIPTION = re.compile(
        r"(?:已经|已)(?:为您)?"
        r"(?:创建工单|派发工单|转派工单|关闭工单|撤回工单|退回工单|取消巡检|修改巡检计划|"
        r"远程启动设备|远程停止设备|复位设备|解除告警|更新证书|变更权限)"
        r"(?:[、或和及与][^。；;！？!?\n]{0,6}?)?"
        r"的?(?:工单|任务|设备|设施|记录|告警|状态|情况|终端|权限)"
    )
    ABSOLUTE_PROMISE = re.compile(r"(?:一定成功|百分百成功|保证修复|马上恢复|立即恢复|一定恢复)")

    SAFE_WRITE_RESPONSE = (
        "当前系统未接入 UrbanOps 真实设备、巡检和工单平台，不能完成设备控制或工单状态变更。"
        "可以查询运维规范；具体业务操作请转人工运维人员处理。"
    )

    def check(self, response: str, *, tool_events: Iterable[Dict[str, Any]] = ()) -> GuardResult:
        tool_events = list(tool_events)
        evidence = self._evidence_types(tool_events)
        findings: List[Dict[str, Any]] = []

        governance = self._knowledge_governance(tool_events)
        if governance.get("status") == "version_lookup_unavailable":
            findings.append({
                "rule": "knowledge_version_lookup_unavailable",
                "knowledge_keys": governance.get("knowledge_keys", []),
            })
            return GuardResult(
                response=self.SAFE_VERSION_LOOKUP_UNAVAILABLE_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="knowledge_version_lookup_unavailable",
                findings=findings,
            )
        if governance.get("status") == "conflict":
            findings.append({
                "rule": "knowledge_conflict",
                "conflict_keys": governance.get("conflict_keys", []),
            })
            return GuardResult(
                response=self.SAFE_KNOWLEDGE_CONFLICT_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="knowledge_conflict",
                findings=findings,
            )
        if governance.get("status") == "not_applicable":
            findings.append({
                "rule": "knowledge_scope_not_applicable",
                "not_applicable_keys": governance.get(
                    "not_applicable_keys", []
                ),
            })
            return GuardResult(
                response=self.SAFE_NOT_APPLICABLE_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="knowledge_scope_not_applicable",
                findings=findings,
            )
        if governance.get("status") == "not_effective":
            findings.append({
                "rule": "knowledge_not_effective",
                "not_effective_keys": governance.get(
                    "not_effective_keys", []
                ),
            })
            return GuardResult(
                response=self.SAFE_NOT_EFFECTIVE_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="knowledge_not_effective",
                findings=findings,
            )
        if governance.get("status") in {"stale", "freshness_unverified"}:
            findings.append({
                "rule": "knowledge_freshness_unverified",
                "stale_keys": governance.get("stale_keys", []),
                "unverified_keys": governance.get("unverified_keys", []),
            })
            return GuardResult(
                response=self.SAFE_STALE_KNOWLEDGE_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="knowledge_freshness_unverified",
                findings=findings,
            )

        # Remove only the warning clause, not the entire sentence: a later
        # affirmative request for another secret must still be rejected.
        if self.SENSITIVE.search(self.SECRET_WARNING.sub("", response or "")):
            findings.append({"rule": "sensitive_secret_request"})
            return GuardResult(
                response="请不要提供平台口令、完整私钥、设备证书或接入令牌。需要核验权限时请通过 UrbanOps 授权流程办理。",
                passed=False,
                escalated=True,
                reason_code="sensitive_secret_request",
                findings=findings,
            )
        write_claim_text = self.CONDITIONAL_CLAUSE.sub(
            "", self.DISBELIEF_WARNING.sub("", response or "")
        )
        write_claim_text = self.STATE_DESCRIPTION.sub("", write_claim_text)
        if self.WRITE_CLAIM.search(write_claim_text) and "verified_state_change" not in evidence:
            findings.append({"rule": "unsupported_write_claim", "missing": "verified_state_change"})
            return GuardResult(
                response=self.SAFE_WRITE_RESPONSE,
                passed=False,
                escalated=True,
                reason_code="unsupported_write_claim",
                findings=findings,
            )
        if self.ABSOLUTE_PROMISE.search(response or ""):
            findings.append({"rule": "absolute_promise"})
            return GuardResult(
                response=(
                    "具体处理结果和完成时间需要以 UrbanOps 业务平台的实际状态"
                    "或人工运维人员核验结果为准。"
                ),
                passed=False,
                escalated=False,
                reason_code="absolute_promise",
                findings=findings,
            )
        if self.BUSINESS_READ_CLAIM.search(response or "") and "verified_record_lookup" not in evidence:
            findings.append({"rule": "unsupported_read_claim", "missing": "verified_record_lookup"})
            return GuardResult(
                response=(
                    "当前没有可靠的查询结果，无法确认设施或工单状态。"
                    "请通过 UrbanOps 业务平台查询或转人工运维人员核验。"
                ),
                passed=False,
                escalated=False,
                reason_code="unsupported_read_claim",
                findings=findings,
            )
        return GuardResult(
            response=response,
            passed=True,
            escalated=False,
            reason_code="guard_passed",
            findings=findings,
        )

    @classmethod
    def knowledge_conflict_keys(
        cls,
        tool_events: Iterable[Dict[str, Any]],
    ) -> List[str]:
        governance = cls._knowledge_governance(tool_events)
        if governance.get("status") != "conflict":
            return []
        return list(dict.fromkeys(
            str(value).strip()
            for value in governance.get("conflict_keys", [])
            if value is not None and str(value).strip()
        ))

    @staticmethod
    def _knowledge_governance(tool_events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        priority = {
            "unknown": 0,
            "verified": 1,
            "resolved": 1,
            "not_applicable": 2,
            "not_effective": 2,
            "freshness_unverified": 3,
            "stale": 4,
            "conflict": 5,
            "version_lookup_unavailable": 6,
        }
        selected: Dict[str, Any] = {}
        for event in tool_events:
            if (
                not isinstance(event, dict)
                or event.get("tool_name") != "knowledge_search"
                or not event.get("success")
                or event.get("fallback_used")
            ):
                continue
            evidence_metadata = event.get("evidence_metadata")
            if not isinstance(evidence_metadata, dict):
                continue
            governance = evidence_metadata.get("knowledge_governance")
            if not isinstance(governance, dict):
                continue
            current_status = str(selected.get("status") or "unknown")
            candidate_status = str(governance.get("status") or "unknown")
            if priority.get(candidate_status, 0) > priority.get(current_status, 0):
                selected = dict(governance)
        return selected

    @staticmethod
    def _evidence_types(events: Iterable[Dict[str, Any]]) -> Set[str]:
        evidence: Set[str] = set()
        for event in events:
            if not event.get("success") or event.get("fallback_used"):
                continue
            evidence.update(str(value) for value in event.get("evidence_types", []))
        return evidence
