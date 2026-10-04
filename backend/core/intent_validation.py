"""External context/intent validators. Neither validation belongs to recognition."""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from core.intent_contracts import IntentAnalysisContract, IntentRouteContract, ValidatedIntentAnalysis
from core.context_contracts import ContextSelectionContract
from core.context_sources import context_sources, current_query_sources
from core.query_context import QueryContextDraft
from core.intent_routes import ORCHESTRATE_ROUTE
from core.supervisor_decision import (
    ScopeStatus, SupervisorDecisionValidator, SupervisorRewrite, SupervisorRewriteContract,
)


def _validated(contract: Any, raw: Any) -> dict[str, Any]:
    try:
        return contract.model_validate(raw).model_dump(mode="json")
    except ValidationError as ex:
        first = ex.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "result"
        raise ValueError(f"invalid structure at {location}: {first['msg']}") from ex


class ContextResultValidator:
    def validate(self, draft: QueryContextDraft) -> SupervisorRewrite:
        if not isinstance(draft.raw_response, Mapping) or set(draft.raw_response) != {"rewrite"}:
            raise ValueError("context result must contain only rewrite")
        proposed = draft.raw_response["rewrite"]
        if isinstance(proposed, Mapping) and "ambiguity_sources" in proposed:
            chosen = _validated(ContextSelectionContract, proposed)
            catalog = context_sources(draft.case_state, draft.history)
            references = []
            for row in chosen["references"]:
                if row["source"] not in catalog:
                    raise ValueError("rewrite source must name a supplied user fact")
                references.append(dict(row, value=catalog[row["source"]]))
            ambiguity = {}
            for key, ids in chosen["ambiguity_sources"].items():
                if any(source not in catalog for source in ids):
                    raise ValueError("ambiguity source must name a supplied user fact")
                ambiguity[key] = list(dict.fromkeys(catalog[source] for source in ids))
            if chosen["status"] == "ambiguous" and any(len(values) < 2 for values in ambiguity.values()):
                raise ValueError("ambiguous rewrite requires multiple candidates; only one distinct user source was selected. Resolve the user's object as worded; an unknown plan name or entitlement is not reference ambiguity")
            inherited = SupervisorDecisionValidator.extract_explicit_entities("\n".join(row["value"] for row in references))
            raw = dict(chosen)
            raw.pop("ambiguity_sources")
            raw.update(references=references, ambiguity_candidates=ambiguity,
                       extracted_entities=SupervisorDecisionValidator.extract_explicit_entities(draft.original_query),
                       inherited_entities=inherited)
        else:
            raw = _validated(SupervisorRewriteContract, proposed)
        rewrite = SupervisorDecisionValidator.validate_rewrite(
            raw, original_query=draft.original_query, case_state=draft.case_state, history=draft.history,
        )
        if rewrite.status.value == "not_needed" and context_sources(draft.case_state, draft.history):
            if re.match(r"^(?:那(?:它|款|套|个套餐|这个|这种|我现在|怎么|可能|重新)|这(?:款|套|个套餐|种情况|个问题)|该套餐|它|刚才那个|具体(?:要)?(?:检查|怎么)|日志我已经|环境变量我检查|错误信息说|我的对话历史确实|我确认过)", draft.original_query.strip()):
                raise ValueError("follow-up omitted its object; resolve it using supplied user source ids")
        # The inherited facts must be bound to the cited source, not merely
        # appear somewhere in unrelated old state/history.
        for reference in rewrite.references:
            if reference.mention.casefold() not in draft.original_query.casefold():
                raise ValueError("rewrite reference mention must quote the current query")
        cited_values = "\n".join(reference.value for reference in rewrite.references).casefold()
        if any(value.casefold() not in cited_values
               for values in rewrite.inherited_entities.values() for value in values):
            raise ValueError("inherited entity must be grounded in a cited reference value")
        evidence = SupervisorDecisionValidator._evidence_text(
            draft.original_query, draft.case_state, draft.history).casefold()
        if any(value.casefold() not in evidence
               for values in rewrite.ambiguity_candidates.values() for value in values):
            raise ValueError("ambiguity candidate is not grounded in context sources")
        return rewrite


class IntentResultValidator:
    def validate(self, raw_response: Any, *, original_query: str,
                 max_intents: Optional[int] = None) -> ValidatedIntentAnalysis:
        if not isinstance(raw_response, Mapping) or set(raw_response) != {"analysis"}:
            raise ValueError("intent recognition output must contain only analysis")
        raw = _validated(IntentAnalysisContract, raw_response["analysis"])
        intents = tuple(SupervisorDecisionValidator.validate_intents(raw["intents"], original_query))
        scope = ScopeStatus(raw["scope_status"])
        if scope == ScopeStatus.IN_SCOPE and not intents:
            raise ValueError("in_scope analysis requires at least one intent")
        if scope != ScopeStatus.IN_SCOPE and intents:
            raise ValueError("out_of_scope or uncertain analysis cannot contain intents")
        if max_intents is not None and len(intents) > max_intents:
            raise ValueError("single-intent recognition requires at most one distinct label")
        return ValidatedIntentAnalysis(intents, scope, raw["reason_code"][:200])


class RouteResultValidator:
    def validate(self, raw_response: Any, *, original_query: str,
                 max_intents: Optional[int] = None) -> ValidatedIntentAnalysis:
        if not isinstance(raw_response, Mapping) or set(raw_response) != {"analysis"}:
            raise ValueError("intent routing output must contain only analysis")
        proposed = raw_response["analysis"]
        if isinstance(proposed, Mapping) and "supporting_source_ids" in proposed:
            from core.intent_contracts import IntentRouteSelectionContract
            selected = _validated(IntentRouteSelectionContract, proposed)
            catalog = current_query_sources(original_query)
            ids = selected.pop("supporting_source_ids")
            if any(source not in catalog for source in ids):
                raise ValueError("route evidence must select current-query source ids")
            selected["supporting_text"] = [catalog[source] for source in ids]
            raw = _validated(IntentRouteContract, selected)
        else:
            raw = _validated(IntentRouteContract, proposed)
        scope = ScopeStatus(raw["scope_status"])
        route = raw["route"] or ""
        support = tuple(dict.fromkeys(text.strip() for text in raw["supporting_text"] if text.strip()))
        if scope != ScopeStatus.IN_SCOPE:
            if route or support:
                raise ValueError("out_of_scope or uncertain routing cannot contain a route or evidence")
            return ValidatedIntentAnalysis((), scope, raw["reason_code"][:200])
        if not route or not support or any(text not in original_query for text in support):
            raise ValueError("route supporting_text must quote the current query")
        if route == ORCHESTRATE_ROUTE:
            if len(support) < 2 or any(a in b for a in support for b in support if a != b):
                raise ValueError("orchestrate requires at least two distinct non-overlapping request spans")
            return ValidatedIntentAnalysis((), scope, raw["reason_code"][:200],
                                           route, raw["tree_score"], support)
        intents = tuple(SupervisorDecisionValidator.validate_intents([{
            "intent_id": f"intent-1-{route}", "label": route,
            "supporting_text": list(support), "tree_score": raw["tree_score"],
        }], original_query))
        return ValidatedIntentAnalysis(intents, scope, raw["reason_code"][:200],
                                       route, raw["tree_score"], support)
