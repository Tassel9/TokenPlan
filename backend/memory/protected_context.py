"""Evidence-backed working constraints, independent of generated summaries.

Updates are folded in the session transaction. Rules cover common constraints;
the existing context model may propose other verbatim user quotes. This is not
a guarantee of exhaustive semantic extraction.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory.long_term_facts import _contains_sensitive_marker


class ProtectedUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    key: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1)
    kind: Literal["constraint", "decision", "date", "amount"] = "constraint"
    scope: Literal["task", "conversation"] = "task"
    operation: Literal["set", "retract"] = "set"


NEW_TASK = re.compile(r"^(?:另外[，,：:]?\s*(?:一个|有个|问个)|另一个问题|新问题|换个(?:问题|话题)|换一个(?:问题|话题)|先不(?:讨论|聊|处理).{0,30}[，,；;]|different (?:question|topic)|new (?:question|topic))", re.I)
CONTINUE_TASK = re.compile(r"^(?:这个套餐|该套餐|同一个|继续|接着|关于刚才|前面(?:那个|的)|刚才(?:那个|的)|针对(?:这个|刚才)|那(?:这个|个|它))")
_HYPOTHESIS = re.compile(r"^(?:如果|假如|假设|比如|例如)|(?:假设|如果).{0,8}(?:预算|截止)")
_ONCE = re.compile(r"这(?:次|一轮)(?:回答|回复)|仅(?:本轮|这条回复)")
_RETRACT = re.compile(r"(?:撤回|取消|删除|不再保留|不用遵守).{0,12}(?:限制|约束|预算|要求)|(?:预算|限制|约束).{0,8}(?:取消|撤回|不再适用)")
_UNSAFE = re.compile(r"忽略.{0,12}(?:系统|指令)|系统提示|ignore.{0,20}(?:system|instructions)", re.I)
_CONVERSATION = re.compile(r"整个(?:会话|对话)|本次(?:会话|对话)|后续所有")
_COMMITMENT = re.compile(r"必须|一定要|禁止|不要|不能|不得|不允许|不超过|最多|上限|最晚|截止|期限|只(?:考虑|比较|接受|需要|要|选|想|能)|仅|决定|选择|确认采用|预算.{0,12}(?:\d|有限)")
_BUSINESS_STATUS = re.compile(r"(?:退款|订单|支付|扣款|账户).{0,10}(?:已到账|已完成|成功|处理中|已取消)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:元|美元|人民币|天|日|月|年|%|小时|分钟))?")
_DATE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}月\d{1,2}[日号]")
_ACTION = re.compile(r"取消订阅|退款|购买|下单|操作|执行")
_RELAX = re.compile(r"(?:年付|月付|预算|期限|截止|限制|约束).{0,16}(?:也可以(?:了)?|也行(?:了)?|不限(?:了)?|不再限制|不作限制)|(?:取消|撤回).{0,8}(?:年付|月付).{0,8}(?:限制|要求)")


def _specific_updates(update: dict) -> list[dict]:
    """Migrate old generic permissions and normalize model/rule proposals alike."""
    if update["key"] == "action_permission":
        actions = list(dict.fromkeys(_ACTION.findall(update["quote"])))
        if actions:
            return [{**update, "key": "action_permission:" + action} for action in actions]
    return [update]


def _relaxation_keys(quote: str) -> set[str]:
    if not _RELAX.search(quote) or re.search(r"[？?]|是否|能否|不可以|不能", quote):
        return set()
    keys = set()
    if re.search(r"年付|月付", quote):
        keys.add("billing_period")
    if re.search(r"预算", quote):
        keys.add("budget")
    if re.search(r"期限|截止", quote):
        keys.add("deadline")
    return keys
_RULES = (
    ("budget", "amount", re.compile(r"预算|费用上限|价格上限|每月(?:最多|不超过)|月费(?:不能|不得|不超过)")),
    ("deadline", "date", re.compile(r"截止|最晚|期限|必须.{0,15}(?:之前|以前)")),
    ("billing_period", "constraint", re.compile(r"(?:只|仅|不)(?:考虑|比较|接受|需要|要|选).{0,15}(?:月付|年付)|不要.{0,10}(?:年付|月付)")),
    ("action_permission", "constraint", re.compile(r"(?:不要|不能|禁止|不允许).{0,12}(?:购买|取消订阅|退款|下单|操作|执行)|只(?:想|要|能)?(?:咨询|解释|比较|了解)|(?:我(?:同意|允许)|现在(?:可以|允许)|改为(?:可以|允许)).{0,12}(?:购买|取消订阅|退款|下单|操作|执行)")),
    ("platform_requirement", "constraint", re.compile(r"(?:必须|一定要|仅|只).{0,18}(?:支持|兼容|运行).{0,20}(?:Windows|Linux|macOS|iOS|Android)", re.I)),
)


def source_id(message: dict) -> str:
    metadata = message.get("metadata") or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    return hashlib.sha256(json.dumps([message.get("role"), message.get("timestamp", message.get("ts")),
                                     message.get("content"), metadata.get("turn_seq", 0)],
                                    ensure_ascii=False).encode()).hexdigest()[:24]


def rule_updates(text: str) -> list[dict]:
    result = []
    for part in re.split(r"[，,。；;\n]+", text):
        quote = part.strip()
        if not quote or _HYPOTHESIS.search(quote) or _ONCE.search(quote):
            continue
        for key in _relaxation_keys(quote):
            result.append({"key": key, "quote": quote, "kind": "constraint",
                           "scope": "task", "operation": "retract"})
        for key, kind, pattern in _RULES:
            if pattern.search(quote):
                result.extend(_specific_updates({"key": key, "quote": quote, "kind": kind,
                    "scope": "conversation" if _CONVERSATION.search(quote) else "task",
                    "operation": "retract" if _RETRACT.search(quote) else "set"}))
    return result


def validate_updates(updates: list, text: str) -> list[dict]:
    if not isinstance(updates, (list, tuple)):
        raise ValueError("protected context updates must be a list")
    accepted = []
    for raw in updates:
        item = ProtectedUpdate.model_validate(raw)
        if item.quote not in text:
            raise ValueError("protected context quote is not in the current user message")
        if item.key.startswith("action_permission:") and item.key.split(":", 1)[1] not in _ACTION.findall(item.quote):
            raise ValueError("protected permission key does not match the quoted action")
        # Evaluate the enclosing user clause too: a model must not remove
        # 'if' or 'this reply only' from its quote to turn an example into a fact.
        clause = next((part for part in re.split(r"[。；;\n]+", text) if item.quote in part), text)
        if (_contains_sensitive_marker(item.quote) or _UNSAFE.search(item.quote)
                or _HYPOTHESIS.search(clause.strip()) or _ONCE.search(clause)
                or _BUSINESS_STATUS.search(item.quote)):
            continue
        if item.operation == "retract" and not (_RETRACT.search(item.quote)
                                               or item.key in _relaxation_keys(item.quote)):
            raise ValueError("protected context retraction lacks an explicit user cue")
        if item.operation == "set" and not (_COMMITMENT.search(item.quote)
                or re.search(r"我(?:同意|允许)|现在(?:可以|允许)|改为(?:可以|允许)", item.quote)):
            continue
        if re.search(r"[？?]|是否|能否|可不可以", item.quote) and not re.search(
                r"必须|禁止|不要|不能|不超过|上限|最多|只考虑|仅考虑", item.quote):
            continue
        item.scope = "conversation" if _CONVERSATION.search(clause) else "task"
        accepted.extend(_specific_updates(item.model_dump()))
    return accepted


def fold_protected(previous: str | dict, payloads: list[str], *, reset=False) -> str:
    state = json.loads(previous or "{}") if isinstance(previous, str) else dict(previous)
    items = [specific for item in state.get("items", []) for specific in _specific_updates(item)]
    retractions = [specific for item in state.get("retractions", []) for specific in _specific_updates(item)]
    generation = int(state.get("task_generation", 0))
    for payload in payloads:
        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            continue  # Legacy stores also accepted opaque message strings.
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if reset or metadata.get("case_update_mode") == "replace" or NEW_TASK.search(text.strip()):
            items = [item for item in items if item["scope"] == "conversation"]
            retractions = [item for item in retractions if item["scope"] == "conversation"]
            generation += 1
            reset = False
        proposals = [*rule_updates(text), *metadata.get("protected_updates", [])]
        for update in validate_updates(proposals, text):
            same_key = [item for item in [*items, *retractions] if item["key"] == update["key"]]
            if (update["scope"] == "task" and len(same_key) == 1 and same_key[0]["scope"] == "conversation"
                    and (update["operation"] == "retract" or same_key[0].get("task_generation") == generation)):
                update["scope"] = "conversation"
            identity = (update["scope"], update["key"])
            # Relax only the identified old condition. "Annual is also OK"
            # must not remove an unrelated "monthly only" restriction.
            if update["operation"] == "retract" and _relaxation_keys(update["quote"]):
                targets = [item for item in items if (item["scope"], item["key"]) == identity]
                periods = set(re.findall(r"年付|月付", update["quote"]))
                if not targets or (periods and not any(
                        periods & set(re.findall(r"年付|月付", item["quote"]))
                        or re.search(r"(?:只|仅)(?:考虑|接受|要|选).{0,8}(?:年付|月付)", item["quote"])
                        for item in targets)):
                    continue
            items = [item for item in items if (item["scope"], item["key"]) != identity]
            retractions = [item for item in retractions if (item["scope"], item["key"]) != identity]
            target = items if update["operation"] == "set" else retractions
            target.append({**update, "source_turn_seq": int(metadata.get("turn_seq", 0) or 0),
                           "source_message_id": source_id(message), "task_generation": generation})
    return json.dumps({"items": items, "retractions": retractions, "task_generation": generation}, ensure_ascii=False)


def working_protected_context(items, text, *, turn_seq=0, updates=(), reset=False) -> list[dict]:
    """Apply validated current-turn changes before freezing an Agent's input."""
    message = dict(role="user", content=text, timestamp="current-turn",
                   metadata=dict(turn_seq=turn_seq, protected_updates=list(updates)))
    state = {"items": [item for item in items if item.get("operation") != "retract"],
             "retractions": [item for item in items if item.get("operation") == "retract"],
             "task_generation": max((item.get("task_generation", 0) for item in items), default=0)}
    folded = json.loads(fold_protected(state, [json.dumps(message, ensure_ascii=False)], reset=reset))
    return folded["items"] + folded["retractions"]


def render_protected(items: list[dict]) -> str:
    if not items:
        return ""
    return "[约束与决策变更（用户原文；优先于旧摘要）]\n" + "\n".join(
        f"- [key={item['key']}][turn_seq={item['source_turn_seq']}][scope={item['scope']}][{ '已撤回' if item.get('operation') == 'retract' else '有效' }] {item['quote']}" for item in items)


def validate_summary_fidelity(summary, messages, previous, protected=()):
    """Check grounded literals and conflicts detectable by the constraint rules.

Verbatim protected quotes are retained separately by code; a summary need not
repeat them. Semantic facts without recognizable literals still need evaluation.
"""
    sources: dict[int, str] = {}
    for message in messages:
        seq = int(message.metadata.get("turn_seq", 0) or 0)
        sources[seq] = sources.get(seq, "") + "\n" + message.content
    old_items = ([previous.current_goal, *previous.confirmed_information, *previous.open_questions]
                 if previous else [])
    for item in old_items:
        for seq in item.source_turn_seqs:
            sources[seq] = sources.get(seq, "") + "\n" + item.text
    for item in protected:
        seq = item["source_turn_seq"]
        sources[seq] = sources.get(seq, "") + "\n" + item["quote"]
    active = {item["key"]: item for item in protected}
    for item in [summary.current_goal, *summary.confirmed_information, *summary.open_questions]:
        evidence = "\n".join(sources.get(seq, "") for seq in item.source_turn_seqs)
        literals = [*_NUMBER.findall(item.text), *_DATE.findall(item.text)]
        if evidence and any(re.sub(r"\s+", "", value) not in re.sub(r"\s+", "", evidence) for value in literals):
            raise ValueError("summary changed an evidenced number, unit or date")
        for update in rule_updates(item.text):
            original = active.get(update["key"])
            if (original and original.get("operation") == "retract" and _COMMITMENT.search(update["quote"])
                    and not _RETRACT.search(update["quote"])):
                raise ValueError("summary revived an explicitly retracted constraint")
            if original and update["kind"] in {"amount", "date"}:
                if set(_NUMBER.findall(update["quote"])) != set(_NUMBER.findall(original["quote"])):
                    raise ValueError("summary conflicts with an active protected constraint")
                if re.search(r"不超过|上限|最多|≤", original["quote"]) and re.search(r"至少|不低于|≥", update["quote"]):
                    raise ValueError("summary reversed an upper-bound constraint")
                monthly = re.search(r"每月|/月|月付", original["quote"])
                if monthly and re.search(r"每年|/年|年付", update["quote"]):
                    raise ValueError("summary changed the billing period")
        permissions = [item for key, item in active.items() if key == "action_permission" or key.startswith("action_permission:")]
        for permission in permissions:
            if permission.get("operation") == "retract" or not re.search(r"不要|不能|禁止|不允许", permission["quote"]):
                continue
            actions = ([permission["key"].split(":", 1)[1]] if ":" in permission["key"] else _ACTION.findall(permission["quote"]))
            for action in actions:
                if (re.search(r"(?:可以|允许|同意|应当|需要).{0,8}" + action, item.text)
                        and not re.search(r"不要|不能|禁止|不允许", item.text)):
                    raise ValueError("summary reversed a prohibition")


def recall_turns(messages: list, query: str, recent: list, *, max_turns=2) -> list:
    """Bounded same-conversation lexical recall, returning complete raw turns."""
    if not re.search(r"之前|以前|最开始|当时|早先|我说过|刚才|前面|原来|原话|记得", query):
        return []
    terms = set(re.findall(r"[A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query.casefold()))
    chinese = re.sub(r"之前|以前|最开始|当时|早先|我说过|刚才|前面|原来|原话|记得|多少|什么|是多少|的|我", "", query)
    terms.update(chinese[i:i+2] for i in range(len(chinese)-1)
                 if re.fullmatch(r"[\u4e00-\u9fff]{2}", chinese[i:i+2]))
    starts = [i for i, message in enumerate(messages) if message.role.value == "user"]
    turns = [messages[start:(starts[i+1] if i+1 < len(starts) else len(messages))]
             for i, start in enumerate(starts)]
    recent_ids = {(message.timestamp, message.role, message.content) for message in recent}
    scored = []
    for i, turn in enumerate(turns):
        if (turn[0].timestamp, turn[0].role, turn[0].content) in recent_ids:
            continue
        text = turn[0].content.casefold()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, i, turn))
    chosen = sorted(scored, key=lambda entry: (entry[0], entry[1]), reverse=True)[:max_turns]
    return [message for _, _, turn in sorted(chosen, key=lambda entry: entry[1]) for message in turn]
