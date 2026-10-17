"""Application-owned ADD-only extraction, explicit lifecycle operations and bounded recall."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re

from core.deepseek_client import deepseek_request_options, extract_text
from runtime.resource_limits import optional_slot
from memory.long_term_facts import (
    MemoryFactCandidate, MEMORY_FIELDS, detect_pending_memory_mutations, has_retract_cue,
    has_update_cue, resolve_profile,
)
from memory.long_term_memories import (
    MEMORY_SCHEMA, accept_proposals, blocks, build_extraction_prompt, messages_for_job,
    expiry_for, normalized, projections, render_memory, terms, utc, visible_rows,
    _HYPOTHESIS, _ONCE,
)
from memory.memory_comparison import (
    ComparisonStale, MemoryComparison, active_rows, admission_action, same_dimensions, snapshot_id,
)

logger = logging.getLogger(__name__)
_ID = re.compile(r"mem-[a-f0-9]{32}")
_ALL = re.compile(r"(?:全部|所有).{0,10}(?:记忆|偏好|信息)|(?:forget|delete) all", re.I)
_EDIT = re.compile(r"修改|更新|替换|\b(?:update|replace)\b", re.I)
_WORKSPACE = re.compile(r"项目|工作区|\b(?:project|workspace)\b", re.I)
_CONVERSATION = re.compile(r"会话|这段对话|本次对话|\bconversation\b", re.I)
_NEGATED_ERASE = re.compile(r"(?:不要|别|不用|不必|无需).{0,5}(?:忘记|删除|撤回|清除)|\bdo not (?:forget|delete)\b", re.I)


def _retract_requested(text):
    return has_retract_cue(text) and not _NEGATED_ERASE.search(text)


def _scoped_request(text):
    return "workspace" if _WORKSPACE.search(text) else "conversation" if _CONVERSATION.search(text) else ""


def _erase_all_request(text):
    return _retract_requested(text) and bool(_ALL.search(text)) and not _scoped_request(text)


def rejected_billing_values(text):
    """An explicit negative is a preference in its own right, never its opposite."""
    if (not has_update_cue(text) or _scoped_request(text) or _HYPOTHESIS.search(text)
            or _ONCE.search(text) or _retract_requested(text)
            or re.search(r"同事|朋友|别人|他人|他说|她说|[？?]|是否|能否", text)):
        return ()
    return tuple(value for value in ("月付", "年付")
                 if re.search(r"(?:不要|不接受|不考虑|不喜欢|不想)(?:再)?(?:使用|选择|推荐|订阅)?\s*" + value, text))


class MemoryFlow:
    def __init__(self, owner, store, *, token_budget=800, max_memories=5,
                 extraction_token_limit=12000, retention_days=None):
        self.owner, self.store = owner, store
        self.token_budget, self.max_memories = max(1, token_budget), max(1, max_memories)
        self.extraction_token_limit = max(1, extraction_token_limit)
        self.retention_days = retention_days or {}
        self._local_results = {}
        self._comparison = MemoryComparison(owner, store, token_limit=self.extraction_token_limit)

    @staticmethod
    def explicit_candidates(text):
        if (_HYPOTHESIS.search(text) or _ONCE.search(text) or _NEGATED_ERASE.search(text)
                or re.search(r"如何|怎么(?:样)?(?:删除|忘记|修改)", text)):
            return []
        if _scoped_request(text):
            return []
        if _erase_all_request(text):
            return [MemoryFactCandidate(key, "", "retract", text) for key in MEMORY_FIELDS]
        return detect_pending_memory_mutations(text)

    @staticmethod
    def explicit_request(text):
        if (_HYPOTHESIS.search(text) or _ONCE.search(text) or _NEGATED_ERASE.search(text)
                or re.search(r"如何|怎么(?:样)?(?:删除|忘记|修改)", text)):
            return False
        return bool(MemoryFlow.explicit_candidates(text) or rejected_billing_values(text) or _retract_requested(text)
                    or ((_ID.search(text) or re.search(r"记忆|偏好|memory", text, re.I))
                        and (has_update_cue(text) or _EDIT.search(text))))

    async def prepare_mutation(self, user_id, conv_id, text, *, workspace_id=""):
        if not self.explicit_request(text) or self.explicit_candidates(text) or rejected_billing_values(text):
            return (), ""
        published = await asyncio.to_thread(self.store.list_profile_events, user_id)
        rows = visible_rows(published, conv_id=conv_id, workspace_id=workspace_id)
        ids = _ID.findall(text)
        if ids:
            owned = {row["id"] for row in published if row["metadata"].get("operation") == "ADD"
                     and (row["metadata"]["scope_type"] == "user"
                          or (row["metadata"]["scope_type"] == "conversation" and row["metadata"]["scope_id"] == conv_id)
                          or (row["metadata"]["scope_type"] == "workspace" and row["metadata"]["scope_id"] == workspace_id))}
            if not set(ids) <= owned:
                return (), "没有找到属于当前用户和作用域的这条记忆，请提供正确的记忆 ID。"
            return tuple(dict.fromkeys(ids)), ""
        scope = _scoped_request(text)
        if scope == "workspace" and not workspace_id:
            return (), "请明确当前工作区，才能操作这个项目的记忆。"
        if scope:
            rows = [row for row in rows if row["metadata"]["scope_type"] == scope]
        if _retract_requested(text) and _ALL.search(text):
            if scope:
                scope_id = workspace_id if scope == "workspace" else conv_id
                rows = [row for row in published if row["metadata"].get("operation") == "ADD"
                        and row["metadata"]["scope_type"] == scope and row["metadata"]["scope_id"] == scope_id]
            return tuple(row["id"] for row in rows), ""
        target = re.sub(r"忘记|删除|撤回|清除|我的|长期|记忆|请|关于|那个|这条|之前", "", text)
        query_terms = terms(target)
        ranked = [(len(query_terms & terms(row["document"])), row) for row in rows]
        ranked = sorted((item for item in ranked if item[0] > 0), key=lambda item: -item[0])
        if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][0] > ranked[1][0]):
            return (ranked[0][1]["id"],), ""
        choices = "\n".join(f"- {row['id']}：{row['document'][:100]}" for _, row in ranked[:5])
        return (), "请明确要修改或忘记哪条长期记忆，提供主题或记忆 ID。" + (
            "\n" + choices if choices else "")

    def pending_events(self, job):
        erase_all = _erase_all_request(job.user_message)
        target_ids = ("*",) if erase_all else job.target_ids
        if not target_ids:
            return []
        micros = int(utc(job.effective_at).timestamp() * 1_000_000)
        return [{"key": "memory:" + target, "event_id": job.job_id, "micros": micros,
                 "ttl": self.owner._pending_profile_ttl_seconds,
                 "payload": json.dumps({"schema_version": "memory-pending-v1",
                    "user_id": job.user_id, "event_id": job.job_id, "target_id": target,
                    "effective_at": job.effective_at, "erase_all": erase_all}, ensure_ascii=False)}
                for target in target_ids]

    def _pending(self, user_id):
        source = self.owner.session_store
        values = source.pending_profile(user_id)
        return [json.loads(value) for value in values]

    async def _context(self, jobs):
        first = jobs[0]
        history = await asyncio.to_thread(self.owner._read_history_messages, first.user_id, first.conv_id)
        before = [message for message in history
                  if int(message.metadata.get("turn_seq", 0) or 0) < int(first.turn_seq or 0)
                  and message.metadata.get("memory_workspace_id", "") == first.workspace_id]
        result = []
        for message in before[-4:]:
            role = message.role.value
            if role not in {"user", "assistant"}:
                continue
            message_id = hashlib.sha256(
                f"{first.user_id}|{first.conv_id}|{message.timestamp.isoformat()}|{role}|{message.content}".encode()
            ).hexdigest()[:32]
            result.append({"message_id": message_id, "content": message.content, "role": role,
                           "job_id": "context-" + message_id, "conv_id": first.conv_id,
                           "turn_seq": str(message.metadata.get("turn_seq", 0)),
                           "observed_at": self.owner._as_utc(message.timestamp).isoformat(), "workspace_id": first.workspace_id})
        return result

    async def process_batch(self, jobs, *, clear_pending=True, projection_operation="set", candidate_filter=None):
        if not jobs:
            return
        if len({(job.user_id, job.conv_id) for job in jobs}) != 1:
            raise ValueError("memory batch crosses user/conversation")
        if len({job.workspace_id for job in jobs}) != 1:
            raise ValueError("memory batch crosses workspace scope")
        jobs = sorted(jobs, key=lambda job: (job.effective_at, int(job.turn_seq or 0), job.job_id))
        user_id, source = jobs[0].user_id, self.owner.session_store
        batch_id = hashlib.sha256(json.dumps([job.job_id for job in jobs]).encode()).hexdigest()
        cached = (await asyncio.to_thread(source.get_memory_batch, user_id, jobs[0].job_id)
                  if hasattr(source, "get_memory_batch") else self._local_results.get((user_id, batch_id)))
        if cached is None:
            messages = [message for job in jobs for message in messages_for_job(job)]
            context = await self._context(jobs)
            previous = visible_rows(await asyncio.to_thread(self.store.list_profile_events, user_id),
                                    conv_id=jobs[0].conv_id, workspace_id=jobs[0].workspace_id)
            related = []
            if previous:
                hits = await asyncio.to_thread(self.store.recall_memories, user_id,
                    "\n".join(job.user_message for job in jobs), previous, n_results=10)
                by_id = {row["id"]: row for row in previous}
                related = [by_id[hit["id"]] for hit in hits if hit["id"] in by_id]
            prompt = build_extraction_prompt(messages, context, related,
                                             allow_replacement=candidate_filter is not None)
            while self.owner._count_text_tokens(prompt) > self.extraction_token_limit and (context or related):
                if context:
                    context.pop(0)
                else:
                    related.pop()
                prompt = build_extraction_prompt(messages, context, related,
                                                 allow_replacement=candidate_filter is not None)
            if self.owner._count_text_tokens(prompt) > self.extraction_token_limit:
                if len(jobs) > 1:
                    middle = len(jobs) // 2
                    first = await self.process_batch(jobs[:middle], clear_pending=clear_pending,
                        projection_operation=projection_operation, candidate_filter=candidate_filter)
                    second = await self.process_batch(jobs[middle:], clear_pending=clear_pending,
                        projection_operation=projection_operation, candidate_filter=candidate_filter)
                    return first + second
                raise ValueError("complete memory turn exceeds extraction token budget")
            async with optional_slot(self.owner._llm_bulkhead):
                response = await self.owner._client.messages.create(
                    model=self.owner._model, max_tokens=4096, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}], **deepseek_request_options())
            rows = accept_proposals(json.loads(extract_text(response)), messages=messages, context=context,
                                    retention_days=self.retention_days)
            if candidate_filter is not None:
                rows = [row for row in rows if candidate_filter(row)]
                if not rows:
                    raise ValueError("explicit update has no compatible, evidenced replacement")
            for row in rows:
                row["metadata"].update(user_id=user_id, memory_id=row["id"])
            cached = {"job_ids": [job.job_id for job in jobs], "rows": rows}
            if hasattr(source, "put_memory_batch"):
                cached = await asyncio.to_thread(source.put_memory_batch, user_id, batch_id, cached)
            else:
                self._local_results[(user_id, batch_id)] = cached
        accepted_rows = ([row for row in cached["rows"] if candidate_filter(row)]
                         if candidate_filter is not None else cached["rows"])
        if candidate_filter is not None and not accepted_rows:
            raise ValueError("cached update has no compatible replacement")
        if candidate_filter is None and projection_operation == "set":
            review_batch_id = cached.get("batch_id") or hashlib.sha256(json.dumps(cached["job_ids"]).encode()).hexdigest()
            accepted_rows = await self._publish_compared_rows(user_id, accepted_rows, review_batch_id)
        else:
            # Explicit replacement keeps its application-owned target semantics.
            await self._apply_rows(user_id, accepted_rows)
        # The same accepted records, rather than a second extractor, feed legacy
        # categorical projections so explicit fallback never revives forgotten values.
        for row in accepted_rows:
            meta = row["metadata"]
            if not projections(meta):
                continue
            user_evidence = next((ref for ref in json.loads(meta["evidence"]) if ref["role"] == "user"), None)
            if user_evidence is None:
                continue
            from memory.conversation_memory import Message, MsgRole
            candidates = [MemoryFactCandidate(key, value, projection_operation, user_evidence["quote"])
                          for key, value in projections(meta).items()]
            await asyncio.to_thread(self.owner._apply_profile_candidates,
                user_id=user_id, conv_id=meta["source_conv_id"],
                source_message=Message(MsgRole.USER, user_evidence["quote"], utc(meta["observed_at"])),
                candidates=candidates)
        if clear_pending:
            for job in jobs:
                await asyncio.to_thread(self.owner._clear_profile_pending_event, user_id, job.job_id)
        remaining = [job for job in jobs if job.job_id not in cached["job_ids"]]
        if remaining:
            await self.process_batch(remaining, clear_pending=clear_pending,
                                     projection_operation=projection_operation, candidate_filter=candidate_filter)
        return accepted_rows

    async def _publish_compared_rows(self, user_id, additions, batch_id):
        for _ in range(4):
            published = await asyncio.to_thread(self.store.list_profile_events, user_id)
            pending = await asyncio.to_thread(self._pending, user_id)
            snapshot = snapshot_id(published, pending)
            erased_ids = await self._comparison.erased_ids(user_id, additions, published, batch_id)
            publishable = [row for row in additions if row["id"] not in erased_ids]
            barriers = [row["metadata"] for row in published
                        if row["metadata"].get("operation") not in {"ADD", "CONFIRM"}]
            if erased_ids:
                # Deletion may have scrubbed caches before a concurrent review
                # finished. Remove the newly identified paraphrase bodies too.
                source = self.owner.session_store
                if hasattr(source, "scrub_memory_batches"):
                    for marker in barriers:
                        await asyncio.to_thread(source.scrub_memory_batches, user_id, marker)
                for key, value in self._local_results.items():
                    if key[0] == user_id:
                        value["rows"] = [row for row in value["rows"] if row["id"] not in erased_ids]
            completed = {row["id"] for row in published} | {
                row["metadata"].get("source_candidate_id") for row in published
                if row["metadata"].get("operation") == "CONFIRM"}
            candidates = []
            for row in publishable:
                if row["metadata"].get("user_id") != user_id:
                    raise ValueError("cached memory belongs to another user")
                if (row["id"] not in completed and expiry_for(row, published) > utc()
                        and not any(blocks(row, marker) for marker in barriers)):
                    candidates.append(row)
            self._comparison.token_limit = self.extraction_token_limit
            decisions = await self._comparison.compare(user_id, candidates, published, pending, batch_id=batch_id)
            try:
                return await self._apply_rows(user_id, publishable, decisions=decisions, snapshot=snapshot)
            except ComparisonStale:
                continue
        raise ComparisonStale("memory changed too often during semantic comparison")

    async def _apply_rows(self, user_id, additions, *, decisions=None, snapshot=None):
        effective = []
        def mutate(rows):
            effective.clear()
            if snapshot is not None and snapshot_id(rows, self._pending(user_id)) != snapshot:
                raise ComparisonStale("memory changed before comparison publication")
            barriers = [row["metadata"] for row in rows if row["metadata"].get("operation") not in {"ADD", "CONFIRM"}]
            current = {row["id"]: row for row in active_rows(rows, self._pending(user_id))}
            seen = {row["metadata"]["content_hash"]: row for row in current.values()}
            ids = {row["id"] for row in rows}
            completed = {row["metadata"].get("source_candidate_id") for row in rows
                         if row["metadata"].get("operation") == "CONFIRM"}
            aliases = {}
            for row in additions:
                if row["metadata"].get("user_id") != user_id:
                    raise ValueError("cached memory belongs to another user")
                if any(blocks(row, marker) for marker in barriers) or expiry_for(row, rows) <= utc():
                    continue
                if row["id"] in ids:
                    effective.append(row)
                    continue
                if row["id"] in completed:
                    continue
                decision = (decisions or {}).get(row["id"], {})
                action = decision.get("action", "ADD")
                target_id = decision.get("existing_id", "")
                target_id = aliases.get(target_id, target_id)
                original = current.get(target_id) if target_id else seen.get(row["metadata"]["content_hash"])
                if target_id and (original is None or not same_dimensions(row, original)):
                    raise ComparisonStale("memory comparison target is no longer active")
                if original is not None and (action == "SKIP"
                        or row["metadata"]["content_hash"] == original["metadata"]["content_hash"]):
                    action = admission_action(row, original)
                if original is not None and action == "SKIP":
                    aliases[row["id"]] = original["id"]
                    # A repeat is a no-op: no body, vector, confirmation,
                    # relation metadata, projection update or TTL renewal.
                    continue
                rows.append(row)
                ids.add(row["id"])
                seen[row["metadata"]["content_hash"]] = row
                current[row["id"]] = row
                aliases[row["id"]] = row["id"]
                effective.append(row)
            return rows
        await asyncio.to_thread(self.store.mutate_profile_events, user_id, mutate)
        return list(effective)

    async def erase(self, user_id, target_ids=(), *, observed_at=None, profile_key="", erase_all=False,
                    operation="retract"):
        observed = utc(observed_at)
        before = await asyncio.to_thread(self.store.list_profile_events, user_id)
        targets = [row for row in before if row["metadata"].get("operation") == "ADD" and (
            row["id"] in target_ids or erase_all or
            (profile_key and profile_key in projections(row["metadata"])))]
        # Preserve existing barriers on retry, even if physical deletion already
        # succeeded and only cache/projection cleanup failed.
        markers = [row for row in before if row["metadata"].get("operation") == operation
                   and row["metadata"].get("observed_at") == observed.isoformat()
                   and (row["metadata"].get("target_id") in target_ids
                        or (profile_key and row["metadata"].get("profile_key") == profile_key)
                        or (erase_all and row["metadata"].get("erase_all")))]
        missing_ids = set(target_ids) - {row["id"] for row in targets} - {
            row["metadata"].get("target_id") for row in markers}
        for target in targets or ([None] if erase_all or profile_key else []):
            original = target["metadata"] if target else {}
            source_ids = set(json.loads(original.get("source_message_ids", "[]")))
            if target:
                for confirmation in before:
                    meta = confirmation["metadata"]
                    if meta.get("operation") == "CONFIRM" and meta.get("confirmed_id") == target["id"]:
                        source_ids.update(json.loads(meta.get("source_message_ids", "[]")))
            metadata = {"schema_version": MEMORY_SCHEMA, "user_id": user_id,
                        "operation": operation, "observed_at": observed.isoformat(),
                        "target_id": target["id"] if target else "",
                        "content_hash": original.get("content_hash", ""),
                        "profile_key": profile_key or original.get("profile_key", ""),
                        "erased_profile_keys": json.dumps(sorted(projections(original))),
                        "source_message_ids": json.dumps(sorted(source_ids)),
                        "erase_all": bool(erase_all)}
            marker_id = "mark-" + hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:32]
            metadata["memory_id"] = marker_id
            markers.append({"id": marker_id, "document": "[记忆已失效]", "metadata": metadata})
        for target_id in sorted(missing_ids):
            metadata = {"schema_version": MEMORY_SCHEMA, "user_id": user_id,
                        "operation": operation, "observed_at": observed.isoformat(),
                        "target_id": target_id, "content_hash": "", "profile_key": "",
                        "source_message_ids": "[]", "erase_all": False}
            marker_id = "mark-" + hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:32]
            metadata["memory_id"] = marker_id
            markers.append({"id": marker_id, "document": "[记忆已失效]", "metadata": metadata})
        def mutate(rows):
            # Re-evaluate on each CAS retry; a stale snapshot must not keep a value.
            # A confirmation may have arrived after the initial read. Its new
            # source must also be covered by the durable deletion boundary.
            for index, marker in enumerate(markers):
                metadata = dict(marker["metadata"])
                sources = set(json.loads(metadata.get("source_message_ids", "[]")))
                for row in rows:
                    meta = row["metadata"]
                    if (meta.get("operation") == "CONFIRM"
                            and meta.get("confirmed_id") == metadata.get("target_id")
                            and utc(meta["observed_at"]) <= observed):
                        sources.update(json.loads(meta.get("source_message_ids", "[]")))
                if sources != set(json.loads(metadata.get("source_message_ids", "[]"))):
                    metadata["source_message_ids"] = json.dumps(sorted(sources))
                    metadata.pop("memory_id", None)
                    marker_id = "mark-" + hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:32]
                    metadata["memory_id"] = marker_id
                    markers[index] = dict(marker, id=marker_id, metadata=metadata)
            result = [row for row in rows if operation == "supersede"
                      or row["metadata"].get("operation") not in {"ADD", "CONFIRM"}
                      or not any(blocks(row, marker["metadata"]) for marker in markers)]
            ids = {row["id"] for row in result}
            return result + [marker for marker in markers if marker["id"] not in ids]
        await asyncio.to_thread(self.store.mutate_profile_events, user_id, mutate)
        if hasattr(self.owner.session_store, "scrub_memory_batches"):
            for marker in markers:
                await asyncio.to_thread(self.owner.session_store.scrub_memory_batches, user_id, marker["metadata"])
        for key, value in list(self._local_results.items()):
            if key[0] == user_id:
                value["rows"] = [row for row in value["rows"]
                                 if not any(blocks(row, marker["metadata"]) for marker in markers)]
        # Metadata is enough for compatibility cleanup on a partially completed retry.
        return targets or markers

    async def process_explicit(self, job):
        candidates = self.explicit_candidates(job.user_message)
        rejected = rejected_billing_values(job.user_message)
        if _erase_all_request(job.user_message):
            await self.erase(job.user_id, observed_at=job.effective_at, erase_all=True)
            candidates = [MemoryFactCandidate(key, "", "retract", job.user_message)
                          for key in MEMORY_FIELDS]
        current_values = (resolve_profile(await asyncio.to_thread(self.owner._get_profile_rows, job.user_id))
                          .profile.get("facts", {}) if candidates or rejected else {})
        if rejected:
            # First admit the sourced negative text through the semantic gate.
            # Only invalidate a contradictory positive setting, never infer a
            # replacement or remove an unrelated existing monthly preference.
            def negative_preference(row):
                meta = row["metadata"]
                return (meta["memory_type"] == "user_preference" and meta["attributed_to"] == "user"
                        and not projections(meta)
                        and all(re.search(r"(?:不接受|不考虑|不要|不喜欢|不想|拒绝|避免).{0,12}" + value,
                                          row["document"]) for value in rejected))
            rows = await self.process_batch([job], clear_pending=False, projection_operation="supersede",
                                            candidate_filter=negative_preference)
            if not rows:
                raise ValueError("negative preference has no admitted source memory")
            key = "preference.billing_cycle"
            if (current_values.get(key) in rejected
                    and any(row["metadata"]["scope_type"] == "user" for row in rows)):
                from memory.conversation_memory import Message, MsgRole
                await self.erase(job.user_id, observed_at=job.effective_at, profile_key=key)
                await asyncio.to_thread(self.owner._apply_profile_candidates,
                    user_id=job.user_id, conv_id=job.conv_id,
                    source_message=Message(MsgRole.USER, job.user_message, utc(job.effective_at)),
                    candidates=[MemoryFactCandidate(key, "", "retract", job.user_message)])
            await asyncio.to_thread(self.owner._clear_profile_pending_event, job.user_id, job.job_id)
            return
        for candidate in candidates:
            if candidate.operation == "supersede" and candidate.memory_key == "preference.billing_cycle":
                # A later explicit positive choice also retires a matching
                # negative preference. Compatible exclusions remain intact.
                rows = visible_rows(await asyncio.to_thread(self.store.list_profile_events, job.user_id))
                conflicts = [row["id"] for row in rows
                    if row["metadata"].get("scope_type") == "user"
                    and row["metadata"].get("memory_type") == "user_preference"
                    and row["metadata"].get("attributed_to") == "user"
                    and any(candidate.value in rejected_billing_values(ref["quote"])
                            for ref in json.loads(row["metadata"].get("evidence", "[]"))
                            if ref.get("role") == "user")]
                if conflicts:
                    await self.erase(job.user_id, conflicts, observed_at=job.effective_at, operation="supersede")
            if (candidate.operation == "supersede" and
                    str(current_values.get(candidate.memory_key, "")).casefold() == candidate.value.casefold()):
                continue
            await self.erase(job.user_id, observed_at=job.effective_at,
                             profile_key=candidate.memory_key,
                             operation="supersede" if candidate.operation == "supersede" else "retract")
        if candidates:
            from memory.conversation_memory import Message, MsgRole
            await asyncio.to_thread(self.owner._apply_profile_candidates,
                user_id=job.user_id, conv_id=job.conv_id,
                source_message=Message(MsgRole.USER, job.user_message, utc(job.effective_at)),
                candidates=candidates)
        if job.target_ids:
            if _retract_requested(job.user_message):
                targets = await self.erase(job.user_id, job.target_ids, observed_at=job.effective_at)
                # A deleted categorical record must not reappear through the old reader.
                keys = {key for row in targets for key in projections(row["metadata"])}
                if keys:
                    from memory.conversation_memory import Message, MsgRole
                    await asyncio.to_thread(self.owner._apply_profile_candidates,
                        user_id=job.user_id, conv_id=job.conv_id,
                        source_message=Message(MsgRole.USER, job.user_message, utc(job.effective_at)),
                        candidates=[MemoryFactCandidate(key, "", "retract", job.user_message) for key in keys])
            elif has_update_cue(job.user_message) or _EDIT.search(job.user_message):
                # An explicit update adds its new source first, then replaces the target.
                targets = [row for row in await asyncio.to_thread(self.store.list_profile_events, job.user_id)
                           if row["id"] in job.target_ids and row["metadata"].get("operation") == "ADD"]
                dimensions = ("memory_type", "attributed_to", "scope_type", "scope_id")
                def compatible(row):
                    return any(all(row["metadata"][key] == target["metadata"][key] for key in dimensions)
                               for target in targets)
                replacement = await self.process_batch([job], clear_pending=False,
                    projection_operation="supersede", candidate_filter=compatible)
                if not replacement:
                    raise ValueError("explicit update has no admissible replacement")
                await self.erase(job.user_id, job.target_ids, observed_at=job.effective_at, operation="supersede")
        await asyncio.to_thread(self.owner._clear_profile_pending_event, job.user_id, job.job_id)

    async def recall(self, user_id, query, *, conv_id="", workspace_id="", queries=None):
        from memory.conversation_memory import LongTermMemoryContext
        if not query:
            result = await self.owner._get_legacy_long_term_memory(user_id, query=query)
            return result if self.owner._count_text_tokens(result.to_text()) <= self.token_budget else LongTermMemoryContext({})
        legacy = LongTermMemoryContext({})
        try:
            for _ in range(4):
                source = self.owner.session_store
                legacy_stamp = (await asyncio.to_thread(source.profile_read_stamp, user_id)
                                if hasattr(source, "profile_read_stamp") else None)
                legacy = await self.owner._get_legacy_long_term_memory(user_id, query=query)
                stamp = await asyncio.to_thread(self.store.read_stamp, user_id)
                pending = await asyncio.to_thread(self._pending, user_id)
                rows = visible_rows(await asyncio.to_thread(self.store.list_profile_events, user_id),
                                    conv_id=conv_id, workspace_id=workspace_id)
                suppressed_ids = {item.get("target_id") for item in pending}
                suppressed_keys = {item.get("memory_key") for item in pending}
                rows = [row for row in rows if not any(item.get("erase_all") for item in pending)
                        and row["id"] not in suppressed_ids
                        and not (set(projections(row["metadata"])) & suppressed_keys)]
                # Single-value projections remain authoritative when ordinary
                # appended statements conflict with an earlier explicit setting.
                profile = await self.owner._get_profile(user_id)
                current_values = profile.get("facts", {})
                rows = [row for row in rows if all(current_values.get(key) == value
                        for key, value in projections(row["metadata"]).items())]
                by_id = {row["id"]: row for row in rows}
                scores = {}
                for search_query in list(dict.fromkeys(queries or [query]))[:3]:
                    hits = await asyncio.to_thread(self.store.recall_memories, user_id,
                                                  search_query, rows, n_results=20)
                    query_terms = terms(search_query)
                    for hit in hits:
                        row = by_id.get(hit["id"])
                        if (not row or not math.isfinite(hit["distance"]) or hit["distance"] < -1e-6
                                or hit["distance"] > self.owner.FACT_MAX_DISTANCE):
                            continue
                        overlap = len(query_terms & terms(row["document"])) / max(1, len(query_terms))
                        score = 1 - hit["distance"] + min(0.15, 0.15 * overlap)
                        scores[hit["id"]] = max(scores.get(hit["id"], 0), score)
                if stamp != await asyncio.to_thread(self.store.read_stamp, user_id) or pending != await asyncio.to_thread(self._pending, user_id):
                    continue
                if legacy_stamp is not None and legacy_stamp != await asyncio.to_thread(source.profile_read_stamp, user_id):
                    continue
                # Retain confirmation metadata for the final expiry check.
                published_rows = await asyncio.to_thread(self.store.list_profile_events, user_id)
                active_ids = {row["id"] for row in visible_rows(published_rows, conv_id=conv_id, workspace_id=workspace_id)}
                if stamp != await asyncio.to_thread(self.store.read_stamp, user_id):
                    continue
                selected = []
                text_seen = set()
                for key in sorted(scores, key=lambda key: (scores[key], by_id[key]["metadata"]["observed_at"]), reverse=True):
                    row = by_id[key]
                    if key not in active_ids or normalized(row["document"]) in text_seen:
                        continue
                    selected.append(row)
                    text_seen.add(normalized(row["document"]))
                    if len(selected) >= self.max_memories:
                        break
                covered = {key for row in selected for key in projections(row["metadata"])}
                legacy_facts = [item for item in legacy.recalled_facts if item.split("=", 1)[0] not in covered]
                # Never cut a memory in the middle: preserve attribution and time.
                rendered = []
                used = self.owner._count_text_tokens(LongTermMemoryContext(legacy.current_profile).to_text())
                for row in selected:
                    text = render_memory(row)
                    tokens = self.owner._count_text_tokens(text + "\n")
                    if used + tokens <= max(1, self.token_budget - 100):
                        rendered.append(text)
                        used += tokens
                result = LongTermMemoryContext(legacy.current_profile, legacy_facts[:3], rendered)
                if self.owner._count_text_tokens(result.to_text()) > self.token_budget:
                    result = LongTermMemoryContext(legacy.current_profile, [], rendered)
                if self.owner._count_text_tokens(result.to_text()) > self.token_budget:
                    result = LongTermMemoryContext({})
                return result
            raise RuntimeError("memory changed repeatedly during recall")
        except Exception as ex:
            logger.warning("通用长期记忆召回失败，保留有效表达偏好: %s", type(ex).__name__)
            fresh = await self.owner._get_legacy_long_term_memory(user_id, query="")
            result = LongTermMemoryContext(fresh.current_profile)
            return result if self.owner._count_text_tokens(result.to_text()) <= self.token_budget else LongTermMemoryContext({})

    async def cleanup(self, *, now=None):
        now = utc(now)
        count = 0
        for user_id in await asyncio.to_thread(lambda: list(self.store.iter_profile_users())):
            all_rows = await asyncio.to_thread(self.store.list_profile_events, user_id)
            expired = [row for row in all_rows if row["metadata"].get("operation") == "ADD"
                       and expiry_for(row, all_rows, now=now) <= now]
            for row in expired:
                await self.erase(user_id, [row["id"]], observed_at=expiry_for(row, all_rows, now=now), operation="expire")
                count += 1
            if hasattr(self.store, "has_pending_profile_cleanup") and self.store.has_pending_profile_cleanup(user_id):
                await asyncio.to_thread(self.store.mutate_profile_events, user_id, lambda rows: rows)
        return count
