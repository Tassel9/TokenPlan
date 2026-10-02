# -*- coding: utf-8 -*-
"""A. 短期记忆探针：压缩视图 vs 全量历史的信息保真评测。

做法：20 轮模拟对话（关键事实刻意埋在第 2/4/7/15/17/18 轮），真实触发增量摘要压缩后，
对同一组探针问题分别用两份上下文回答：
  * ``full``：全量对话记录（不压缩，对照上限）；
  * ``compressed``：真实压缩视图（结构化摘要 + 最近完整轮次，``get_short_term_memory``）。
保真率 = compressed 答对 / full 答对；同时报告两种视图的 Token 开销。

运行环境：需要 anthropic、SQLite 和 transformers/tokenizer 缓存。
    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & E:\\anaconda3\\envs\\my_env\\python.exe evaluation\\benchmarks\\evaluate_short_term_memory_probe.py

产物：``evaluation/reports/short_term_memory/probe_fidelity.json``。
⚠️ 口径：合成对话；轻量构造（``__new__`` + 注入 SQLite/LLM/tokenizer），只驱动短期记忆真实代码路径。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from anthropic import AsyncAnthropic  # noqa: E402

from core.deepseek_client import (  # noqa: E402
    DEEPSEEK_DEFAULT_MODEL,
    deepseek_request_options,
    extract_text,
    load_deepseek_tokenizer,
)
from memory.conversation_memory import MemoryManager, MsgRole  # noqa: E402
from memory.sqlite_session_store import SQLiteSessionStore  # noqa: E402

REPORT_PATH = (
    ROOT / "evaluation" / "reports" / "short_term_memory" / "probe_fidelity.json"
)

# ── 20 轮模拟对话：关键事实埋点（轮序号 1 起）────────────────────────────────
# 2: 工单号 WO-2026-0917 ｜ 4: 联系人周航 / 班组滨江照明班 ｜ 7: 责任班组
# 15: 巡检点位更新（文三路 → 江南大道）｜ 17: 上门时间 ｜ 18: 值守口令
CONVERSATION: List[Tuple[str, str]] = [
    ("你好，我想安排 A 区智慧路灯专项巡检。", "可以。请提供路灯范围、巡检时段、责任班组和优先级，系统会据此创建巡检任务。"),
    ("维修工单号是 WO-2026-0917，请帮我核对。", "已记录工单 WO-2026-0917，后续可以继续查询受理、派发、处理和验收状态。"),
    ("另外我比较关心告警怎么和工单关联。", "同一故障的告警应保留路灯编号、告警码和时间窗口，创建工单时引用告警记录，避免重复派单。"),
    ("我叫周航，负责滨江照明班。", "已记录联系人周航和责任班组滨江照明班，仅用于本次巡检与工单协同。"),
    ("巡检点位先写西湖区文三路 90 号。", "已记录初始巡检点位为西湖区文三路 90 号，后续变更会覆盖旧点位。"),
    ("紧急故障的响应时限是多少？", "紧急漏电或大面积熄灯工单要求 30 分钟内响应，并优先完成现场安全隔离。"),
    ("这张工单的责任班组就写滨江照明班。", "已记录责任班组为滨江照明班，派发前仍需核对片区权限和值班状态。"),
    ("一批路灯同时离线时怎么安排巡检？", "应先检查共同网关和公共通信链路，再按片区拆分任务，避免逐盏重复排查。"),
    ("遥测中断后如何补齐历史数据？", "先确认控制命令是否可达，再核对采集、缓存和上报链路；补传时要保留原始采样时间。"),
    ("区域权限怎么分配？", "按最小权限原则授予点位、巡检和工单权限，高风险远程控制需要额外确认并记录审计。"),
    ("审计日志保留多久？", "审计日志默认保留 180 天，可按时间、路灯编号和操作类型检索并导出。"),
    ("告警高峰期会不会限流？", "重复告警会按路灯编号和时间窗口聚合；超过阈值时进入队列，但紧急安全告警不得被静默丢弃。"),
    ("路灯终端认证用什么机制？", "智慧路灯终端使用设备证书和绑定关系认证，同时核对设备时间、编号和证书链。"),
    ("终端凭证怎么保护？", "不得在对话或工单中提交完整私钥、设备证书或接入令牌；轮换必须由授权人员在设备平台执行。"),
    ("点位改成滨江区江南大道 588 号，之前那个不用了。", "已将当前巡检点位更新为滨江区江南大道 588 号，西湖区文三路 90 号不再使用。"),
    ("紧急工单怎么预约上门？", "在工单中选择紧急级别和期望时段，值班组确认后安排现场人员，并保留到场和验收记录。"),
    ("上门时间定在周五下午 3 点，可以吗？", "可以，已记录上门时间为周五下午 3 点；如需改期应及时通知值班组。"),
    ("这次夜班值守口令是 NIGHT-888，帮我记住。", "已在本次会话中记录夜班值守口令 NIGHT-888；它不是设备密钥，不用于终端认证。"),
    ("好的，那先这样。", "好的，后续可在本会话继续查询工单、巡检和告警进度。"),
    ("最后确认一下，周五之前能完成巡检吗？", "需以班组接单和现场结果为准；当前已记录周五下午 3 点上门，不提前承诺完工。"),
]

# 合成填充段（轮换复用）：把热窗口推过 6000 Token 的压缩触发线，控制评测成本。
# 注意：填充段刻意不含任何探针事实（工单号/姓名/点位/响应时限/上门时间/值守口令/日志天数）。
DETAIL_BLOCKS: List[str] = [
    "在数据安全方面，平台采用多副本存储与异地容灾相结合的设计：生产数据在主区域多副本保存，副本在独立可用区异步同步，遇到区域性故障时可以整体切换。备份策略为按小时增量、按天全量的滚动执行，备份文件统一加密存储并定期做完整性校验，校验失败的备份会自动标记并从轮换序列中剔除。恢复演练按季度组织，覆盖单表恢复、整库恢复与跨区域拉起三类场景，每次演练都会输出恢复时间与数据一致性的记录。如需把备份下载到本地留存，可以在管理后台提交导出申请，导出文件默认加密，需要用贵司提供的公钥进行封装后再交付。",
    "权限体系上，平台提供组织、角色与成员三层模型：组织对应独立的资源空间，角色决定可操作的资源类型与动作范围，成员可以同时拥有多个角色并按最宽范围生效。角色支持自定义，也内置了管理员、只读审计与实施人员三套常用模板；对高危操作（批量删除、密钥轮换、对外导出）默认要求二次确认并强制写入审计。权限变更建议按最小化原则推进，先在小范围验证再全量生效；如果对接了外部身份提供商，可以按部门自动同步角色。",
    "合规与审计方面，平台的审计记录覆盖登录行为、权限变更、数据导出与配置修改四类操作，记录字段包含操作人、操作对象、结果与来源地址，不包含业务数据正文。审计记录支持按时间范围与操作类型检索，并可导出为表格文件供内部审查使用；导出动作本身也会生成一条新的审计记录。如果贵司有独立的日志平台，可以把审计流通过接口推送到你们自己的存储上，避免两个系统之间的口径差异。",
    "网络接入方面，平台对外服务开启传输层加密，内部服务之间也启用双向校验；公网入口与企业专线可以同时存在，便于分阶段迁移。防火墙需要放行的是平台对外提供的域名与端口，反向代理场景下请保留原始来源地址，否则风控与审计记录会失真。白名单功能按项目维度维护，支持单个地址与网段两种粒度；变更后通常几分钟内生效，如果遇到缓存尚未刷新的情况，重新建立连接即可。",
    "客户端兼容性上，网页端支持主流现代浏览器，桌面端提供独立安装包并支持自动更新，移动端以查看与审批为主。不同端之间的数据实时同步，同一路灯终端可以在多个端同时在线；如需强制单端登录，可在安全设置里开启限制项。桌面端的文件缓存默认加密存储，退出登录时会清理临时目录；当桌面端与网页端同时编辑同一份内容时，平台会保留两个版本并在界面上提示冲突位置，由使用人自行合并。",
    "工单与服务流程方面，提交工单时需要选择影响范围与期望时间：影响范围决定优先级，期望时间用于安排响应。工单状态从待受理、处理中、待验证到已关闭，每一步都会触发通知；如果超过约定时限未得到响应，系统会自动升级提醒。验证阶段建议由提交人确认结果后再关闭，避免同一问题反复开单；超出支持范围的定制需求会作为建议单记录，由产品侧定期评审。",
    "迁移与切换方面，正式操作前建议先在测试环境做一次演练：按同样的数据量与网络条件跑一遍完整流程，记录耗时与异常点，再确定正式切换的具体步骤。切换窗口选择业务低峰，切换前必须完成备份并验证可恢复性；若切换后发现数据异常，应优先回滚到切换前状态，再分析原因。演练与正式切换都应保留操作日志，便于复盘时对照每一步的实际结果。",
    "巡检记录治理方面，系统按路灯编号、点位、巡检时间和责任班组保存结果；发现记录与现场现象不一致时，应保留遥测时间窗口、现场照片和复测结论。重复记录要区分重复提交、任务重试与同一故障的多次复发，不能直接删除原始证据。巡检规范调整后只影响新任务，历史记录仍保留当时适用的规范版本，便于后续审计和事故复盘。",
]


def _expanded_rounds() -> List[Tuple[str, str]]:
    rounds: List[Tuple[str, str]] = []
    for index, (user_text, assistant_text) in enumerate(CONVERSATION):
        blocks = "".join(
            f"\n\n{DETAIL_BLOCKS[(index + offset) % len(DETAIL_BLOCKS)]}"
            for offset in (0, 3, 5, 6)
        )
        rounds.append((user_text, f"{assistant_text}{blocks}"))
    return rounds


PROBES: List[Dict[str, Any]] = [
    {"id": "work-order-no", "turn": 2, "question": "我的工单号是多少？", "expect": ["WO-2026-0917"]},
    {"id": "contact", "turn": 4, "question": "联系人和责任班组分别是什么？", "expect": ["周航", "滨江照明班"], "expect_mode": "all"},
    {"id": "responsible-team", "turn": 7, "question": "这张工单的责任班组是什么？", "expect": ["滨江照明班"]},
    {"id": "location-current", "turn": 15, "question": "当前巡检点位在哪里？", "expect": ["江南大道"], "forbidden": ["文三路"]},
    {"id": "response-sla", "turn": 6, "question": "紧急故障响应时限是多久？", "expect": ["30 分钟", "30分钟"], "expect_mode": "any"},
    {"id": "onsite-time", "turn": 17, "question": "上门时间定在什么时候？", "expect": ["周五", "3 点"], "expect_mode": "all"},
    {"id": "shift-code", "turn": 18, "question": "夜班值守口令是什么？", "expect": ["NIGHT-888"]},
    {"id": "audit-retention", "turn": 11, "question": "审计日志默认保留多久？", "expect": ["180 天", "180天"]},
]


def _load_env_file() -> Dict[str, str]:
    env: Dict[str, str] = {}
    env_path = ROOT / ".env"
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _normalize(text: str) -> str:
    return "".join(str(text or "").split()).replace("，", "").replace("。", "").replace("？", "").lower()


def _score(answer: str, probe: Dict[str, Any]) -> bool:
    normalized = _normalize(answer)
    expects = [_normalize(item) for item in probe.get("expect") or []]
    if not expects:
        return False
    mode = probe.get("expect_mode", "any")
    if mode == "all":
        hit = all(item in normalized for item in expects)
    else:
        hit = any(item in normalized for item in expects)
    if not hit:
        return False
    for forbidden in probe.get("forbidden") or []:
        if _normalize(forbidden) in normalized:
            return False
    return True


def _build_manager(client: AsyncAnthropic, model: str) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._session_store = SQLiteSessionStore(":memory:")
    manager._client = client
    manager._model = model
    manager._llm_bulkhead = None
    manager._tokenizer = load_deepseek_tokenizer(model)
    manager._short_term_token_limit = 6000
    manager._recent_turns = 5
    manager._summary_output_token_limit = 2048
    manager._hot_memory_max_messages = 40
    manager._history_max_messages = 100
    return manager


def _render_full_history(rounds: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for user_text, assistant_text in rounds:
        lines.append(f"用户：{user_text}")
        lines.append(f"客服：{assistant_text}")
    return "\n".join(lines)


async def _answer(
    client: AsyncAnthropic,
    model: str,
    *,
    context: str,
    question: str,
) -> str:
    prompt = (
        "以下是一段客服会话记录。请只根据记录内容回答用户问题；"
        "如果记录中没有相关信息，请回答“不知道”。\n\n"
        f"=== 记录开始 ===\n{context}\n=== 记录结束 ===\n\n"
        f"问题：{question}\n只输出答案本身。"
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=80,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        **deepseek_request_options(),
    )
    return extract_text(resp).strip()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    env = _load_env_file()
    api_key = (env.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY，无法运行")
        return 1
    base_url = (env.get("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "").strip() or None
    model = (env.get("DEEPSEEK_MODEL") or os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL).strip()

    client = AsyncAnthropic(api_key=api_key, base_url=base_url)
    manager = _build_manager(client, model)

    user_id, conv_id = "probe-user", "probe-conv"
    rounds = _expanded_rounds()
    compression_events: List[int] = []
    previous_summary = ""
    try:
        for index, (user_text, assistant_text) in enumerate(rounds, start=1):
            await manager.add_turn(
                user_id,
                conv_id,
                user_content=user_text,
                assistant_content=assistant_text,
            )
            summary_model, current_summary = manager._read_short_term_summary(user_id, conv_id)  # noqa: SLF001
            if len(current_summary or "") > len(previous_summary):
                compression_events.append(index)
            previous_summary = current_summary or ""
            print(
                f"[{index:>2}/{len(rounds)}] hot_tokens="
                f"{manager._count_short_term_tokens(manager._read_short_term_messages(user_id, conv_id), summary_model.to_context_text() if summary_model else '')} "  # noqa: SLF001
                f"summary={'yes' if summary_model else 'no'}"
                f"{' (compressed)' if compression_events and compression_events[-1] == index else ''}"
            )

        context = await manager.get_short_term_memory(user_id, conv_id)
        compressed_view = context.to_text()
        _, summary_text = manager._read_short_term_summary(user_id, conv_id)  # noqa: SLF001
        full_view = _render_full_history(rounds)

        full_tokens = manager._count_text_tokens(full_view)  # noqa: SLF001
        compressed_tokens = manager._count_text_tokens(compressed_view)  # noqa: SLF001

        results: List[Dict[str, Any]] = []
        for probe in PROBES:
            full_answer = await _answer(client, model, context=full_view, question=str(probe["question"]))
            compressed_answer = await _answer(client, model, context=compressed_view, question=str(probe["question"]))
            results.append({
                "id": probe["id"],
                "turn": probe["turn"],
                "question": probe["question"],
                "expected": probe.get("expect"),
                "full_answer": full_answer,
                "compressed_answer": compressed_answer,
                "full_ok": _score(full_answer, probe),
                "compressed_ok": _score(compressed_answer, probe),
                "fact_in_view": any(
                    _normalize(item) in _normalize(compressed_view)
                    for item in probe.get("expect") or []
                ),
            })

        full_hits = sum(1 for item in results if item["full_ok"])
        compressed_hits = sum(1 for item in results if item["compressed_ok"])
        fidelity = (
            round(compressed_hits / full_hits, 4) if full_hits else None
        )
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "conversation_turns": len(rounds),
            "compression_events": compression_events,
            "summary_triggered": bool(summary_text),
            "summary_chars": len(summary_text or ""),
            "full_tokens": full_tokens,
            "compressed_tokens": compressed_tokens,
            "token_reduction": (
                round(1 - compressed_tokens / full_tokens, 4) if full_tokens else None
            ),
            "probes_total": len(results),
            "coverage_hits": sum(1 for item in results if item["fact_in_view"]),
            "full_hits": full_hits,
            "compressed_hits": compressed_hits,
            "fidelity": fidelity,
            "probes": results,
            "compressed_view": compressed_view,
        }
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== 逐题结果 ===")
        for item in results:
            print(
                f"[{'✓' if item['compressed_ok'] else '✗'}] {item['id']:<16} "
                f"full={'✓' if item['full_ok'] else '✗'} | "
                f"full={item['full_answer'][:24]!r} compressed={item['compressed_answer'][:24]!r}"
            )
        print("\n=== 汇总 ===")
        print(f"压缩事件轮次: {compression_events}（摘要 {report['summary_chars']} 字）")
        print(f"视图 Token: full={full_tokens} compressed={compressed_tokens}（-{report['token_reduction']:.1%}）")
        print(f"事实在最终视图中的覆盖: {report['coverage_hits']}/{len(results)}")
        print(f"探针命中: full {full_hits}/{len(results)}，compressed {compressed_hits}/{len(results)}")
        print(f"保真率（compressed/full）: {fidelity}")
        print("\n报告 ->", out_path)
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
