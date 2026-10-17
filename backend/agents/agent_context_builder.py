"""One context builder shared by direct routing and Supervisor delegation."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

from agents.intent_router import AgentMessageRoute
from agents.specialist_agents import AgentInput
from core.agent_context import AgentContextSnapshot
from core.context_sources import context_sources, current_query_sources
from core.supervisor_decision import SupervisorAnalysis
from core.task_planning import default_task, request_items, validate_task
from memory.agent_memory import AgentMemoryStore
from runtime.intent_execution import IntentInvocation, IntentResult


def _contains(text: str, value: str) -> bool:
    # ORDER-1 and 401 must not match ORDER-10 or 4010.
    return bool(re.search(r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])",
                          text, re.IGNORECASE))


def _mention_key(text: str) -> str:
    """Normalize only sentence-ending punctuation for reference matching."""
    return str(text).strip().rstrip("。！？!?.").rstrip()


def _mentions_source(mention: str, source: str) -> bool:
    value, text = _mention_key(mention), _mention_key(source)
    return bool(value and text and _contains(text, value))


class MissingTaskContext(ValueError):
    def __init__(self, fields: Sequence[str]) -> None:
        self.fields = tuple(fields)
        super().__init__("task context is missing: " + ", ".join(fields))

    @property
    def question(self) -> str:
        names = {"order_id": "订单号", "account_email": "账户邮箱", "workspace_id": "工作区",
                 "plan": "套餐", "date": "时间", "error_code": "错误码", "amount": "金额"}
        return "请明确当前问题对应的" + "、".join(names.get(key, "必要信息") for key in self.fields) + "，以免混用其他问题的信息。"


class AgentContextBuilder:
    def build(self, *, invocation: IntentInvocation, message: AgentMessageRoute,
              analysis: SupervisorAnalysis, original_query: str, request_id: str,
              user_id: str, conv_id: str, case_id: str, memory: AgentMemoryStore,
              completed: Mapping[str, IntentResult], case_state: Mapping[str, Any],
               history: Sequence[Mapping[str, Any]], faq_prefetch_allowed: bool,
               public_read_only: bool = False,
               approval_id: str = "", idempotency_key: str = "", trace_recorder: Any = None,
               protected_context: Sequence[Mapping[str, Any]] = (),
               long_term_context: str = "") -> AgentInput:
        items = request_items(analysis.intents, original_query)
        candidates = [item for item in items if item.intent_id in message.intent_ids]
        selected = [item for item in candidates if _mentions_source(item.source_text, invocation.query)]
        if not selected:
            selected = candidates
        task_id = invocation.task_id if invocation.task_id.startswith("task-") else "task-" + invocation.task_id
        task = default_task(task_id, selected)
        task = task.model_copy(update={"read_only": public_read_only})
        projected = memory.project_entities(invocation)
        sources = current_query_sources(original_query)
        historical = context_sources(case_state, history)
        # Rewrite IDs use the bounded history selected upstream. Recover their
        # original user-message IDs before recording snapshot provenance.
        grounded_references = []
        for ref in analysis.rewrite.references:
            source_id = ref.source
            if not _contains(historical.get(source_id, ""), ref.value):
                source_id = next((sid for sid, text in historical.items()
                                  if ref.source.startswith("history[") and sid.startswith("history[")
                                  and _contains(text, ref.value)), "")
            if source_id:
                grounded_references.append((ref, source_id))
        bindings: list[dict[str, Any]] = []
        entities: dict[str, list[str]] = {}
        unresolved: list[str] = []
        for key, values in projected.items():
            pending = []
            for value in values:
                own = [item.request_item_id for item in selected if _contains(item.source_text, value)]
                other = [item for item in items if item not in selected and _contains(item.source_text, value)]
                if _contains(original_query, value):
                    if own:
                        origin = "USER_MESSAGE"
                    elif other:
                        continue
                    else:
                        pending.append((value, "USER_MESSAGE", next(
                            (sid for sid, text in sources.items() if sid != "query" and _contains(text, value)), "query")))
                        continue
                    source_id = next((sid for sid, text in sources.items()
                                      if sid != "query" and _contains(text, value)), "query")
                else:
                    refs = [(ref, sid) for ref, sid in grounded_references
                            if _contains(ref.value, value) and any(_mentions_source(ref.mention, item.source_text) for item in selected)
                            and _contains(historical[sid], value)]
                    grounded_sources = [sid for sid, text in historical.items() if _contains(text, value)]
                    if not grounded_sources:
                        unresolved.append(key)
                        continue
                    # A historical entity is a user-context reference, not a verified business fact.
                    if refs:
                        own = [item.request_item_id for item in selected
                               if any(_mentions_source(ref.mention, item.source_text) for ref, _ in refs)]
                        source_id = refs[0][1]
                        origin = "CONTEXT_REFERENCE"
                    else:
                        pending.append((value, "CONTEXT_REFERENCE", grounded_sources[0]))
                        continue
                bindings.append({"key": key, "value": value, "request_item_ids": own,
                                 "source_type": origin, "source_id": source_id,
                                 "authority": "user_context", "business_fact_confirmed": False})
            # Only a unique background value can be shared with a focused task.
            # Competing orders/parameters with no question binding are withheld.
            if pending and len(values) == 1:
                value, origin, source_id = pending[0]
                bindings.append({"key": key, "value": value,
                                 "request_item_ids": [item.request_item_id for item in selected],
                                 "source_type": origin, "source_id": source_id,
                                 "authority": "user_context", "business_fact_confirmed": False})
            elif pending:
                unresolved.append(key)
        for binding in bindings:
            if binding["source_type"] == "CONTEXT_REFERENCE" and any(
                other["key"] == binding["key"] and other["value"] != binding["value"]
                and other["source_type"] == "CONTEXT_REFERENCE"
                and set(other["request_item_ids"]) & set(binding["request_item_ids"])
                for other in bindings
            ):
                unresolved.append(binding["key"])
                continue
            bucket = entities.setdefault(binding["key"], [])
            if binding["value"] not in bucket:
                bucket.append(binding["value"])
        bindings = [binding for binding in bindings if binding["value"] in entities.get(binding["key"], ())]
        references = [{"mention": ref.mention, "value": ref.value, "source_id": source_id,
                       "authority": "user_context", "business_fact_confirmed": False}
                      for ref, source_id in grounded_references
                      if any(_mentions_source(ref.mention, item.source_text) for item in selected)
                      and not any(_mention_key(other.mention) == _mention_key(ref.mention) and other.value != ref.value
                                  for other in analysis.rewrite.references)]
        user_constraints = [{"source_id": sid, "text": text} for sid, text in sources.items()
                            if sid != "query" and re.search(r"不要|不需要|不允许|禁止|^别(?:执行|照做|遵循)|仅|只(?:想|要|看|查询|咨询)|必须|不能|预算|截至|限于", text)
                            and not any(_contains(text, item.source_text) for item in items if item not in selected)]
        accepted = {key: getattr(result, "item_results", ()) for key, result in completed.items()
                    if result.status == "COMPLETED"}
        missing = [key for key in task.required_inputs if not entities.get(key)
                   and key not in {ref.name for ref in task.input_refs}]
        if missing:
            raise MissingTaskContext(missing)
        resolved = validate_task(task, items, intent_ids=message.intent_ids,
                                 entities=entities, completed=accepted)
        dependencies = []
        for ref in task.input_refs:
            row = next(item for item in accepted[ref.task_id] if item.request_item_id == ref.request_item_id)
            dependencies.append({"name": ref.name, "value": resolved[ref.name],
                                 "task_id": ref.task_id, "request_item_id": ref.request_item_id,
                                 "field": ref.field, "status": row.status, "evidence_ids": list(row.evidence_ids),
                                 "authority": "accepted_agent_output", "is_user_confirmation": False})
        view = memory.context_for(replace(invocation, entities=entities), request_id=request_id, case_id=case_id)
        memory_payload = view.to_runtime_payload()
        memory_payload["entities"] = entities
        memory_payload["scope"]["entity_keys"] = sorted(entities)
        snapshot = AgentContextSnapshot(
            request_id=request_id, task_id=task.task_id, attempt_no=1,
            assigned_agent=invocation.agent, case_id=case_id, task=task.model_dump(mode="json"),
            request_items=tuple(item.model_dump(mode="json") for item in selected),
            entity_bindings=tuple(bindings), dependency_inputs=tuple(dependencies),
            context_references=tuple(references),
            constraints={"read_only": task.read_only, "task_scope_frozen": True,
                          "public_read_only": public_read_only,
                          "agent_result_is_user_confirmation": False, "user_constraints": user_constraints,
                          "protected_context": list(protected_context)},
            unresolved_entities=tuple(key for key in dict.fromkeys(unresolved) if key not in entities),
            memory_context=({"long_term": long_term_context,
                             "authority": "historical_background",
                             "current_request_precedes": True,
                             "is_user_confirmation": False,
                             "business_fact_confirmed": False}
                            if long_term_context else {}),
        )
        return AgentInput(
            request_id=f"{request_id}-{invocation.intent_id}", message=invocation.query,
            execution_query=invocation.query, user_id=user_id, conv_id=conv_id,
            intent_id=invocation.intent_id, intent=invocation.intent, focus=invocation.focus,
            planning_task=task, attempt_no=1, resolved_inputs=resolved,
            faq_prefetch_allowed=faq_prefetch_allowed, public_read_only=public_read_only, entities=entities,
            agent_memory_context=memory_payload, context_snapshot=snapshot,
            approval_id=approval_id, idempotency_key=idempotency_key, trace_recorder=trace_recorder,
        )
