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
# 2: 订单号 TP-2026-0917 ｜ 4: 联系人周航 / 公司杭州云杉科技 ｜ 7: 发票抬头
# 15: 收货地址更新（文三路 → 江南大道）｜ 17: 上门时间 ｜ 18: 推荐码
CONVERSATION: List[Tuple[str, str]] = [
    ("你好，我想咨询一下企业版套餐的购买流程。", "您好！企业版支持在线下单与对公转账两种方式，下单后 1 个工作日内会有专属顾问与您联系确认席位数量、开票信息与实施排期。首次购买建议先确认使用人数，方便匹配折扣档位。"),
    ("我的订单号是 TP-2026-0917，请帮我核对一下。", "已为您核对该订单：企业版基础套装，含 50 席位与一年原厂支持。后续查询进度时提供这个订单号即可，我已在本会话中记录。"),
    ("另外我比较关心发票的问题。", "企业版默认开具增值税专用发票，抬头与税号在订单确认后不可随意变更，如需调整请在开票前联系顾问。电子发票一般在下单后 3 个工作日内推送至您的邮箱，纸质发票由挂靠服务商寄出。"),
    ("我叫周航，公司是杭州云杉科技。", "周航您好！已记录您的联系信息。为保护隐私，我们只会在订单履约、发票与售后场景使用这些信息，不会用于其他用途。"),
    ("收货地址先用西湖区文三路 90 号吧。", "好的，已记录收货地址为西湖区文三路 90 号。如发货前需要调整，随时在本会话里告诉我即可，我们会同步给仓库。"),
    ("售后保修期是多少？", "企业版硬件的标准保修期为 3 年，自签收之日起算；软件侧含一年原厂支持与版本更新。人为损坏与自然灾害不在保修范围内，如需上门服务可在保修期内预约，超期后按次收费。"),
    ("发票抬头就写杭州云杉科技，税号我稍后给你。", "收到，发票抬头已登记为杭州云杉科技。税号可以稍后补充，但请在开票流程启动前提供，否则专票需要作废重开，会影响报销时间。"),
    ("你们支持批量采购折扣吗？", "支持。企业版按席位数分档：50 席位以内为标准价，50 至 200 席位享 9 折，200 席位以上可以走年度框架协议，折扣与账期一并协商。批量采购还支持分批交付，第一批发货后 30 天内补齐余量即可。"),
    ("如果我们要做数据迁移，一般怎么收费？", "标准迁移服务覆盖 500GB 以内数据，超出部分按每 TB 计费。迁移由实施团队远程操作，需要您提供测试环境与只读账号；正式迁移前我们会在测试环境做一次演练，确认无误后再执行正式切换，避免影响生产。迁移窗口建议安排在非业务高峰，通常选择周末。"),
    ("权限体系可以对接我们的 AD 吗？", "可以。企业版支持 SAML 2.0 与 LDAP，与 AD 对接后可以用组织架构自动同步部门与角色。首次对接需要在管理后台下载元数据文件，由您的 IT 配置信任关系，一般 1 个工作日内可以完成联调。同步周期默认每小时一次，也支持手动触发。"),
    ("审计日志保留多久？", "审计日志默认保留 180 天，可在管理后台导出为 CSV。需要更长保留期可以在订单中增加日志归档模块，最长可扩展到 3 年。日志内容覆盖登录、权限变更、数据导出与配置修改四类操作，不包含业务数据正文，满足常规合规审计要求。"),
    ("高峰期并发会不会被限流？", "企业版按席位提供并发配额，50 席位的并发上限为 100，超出后会进入排队而不是直接拒绝。高峰期如果持续触顶，建议提前联系顾问做临时扩容，扩容生效一般在 30 分钟以内。限流策略对 API 与网页端一致，均按主账号维度计算。"),
    ("SSO 单点登录支持哪些协议？", "支持 SAML 2.0、OAuth 2.0 与 OIDC 三种协议，其中 SAML 2.0 的兼容性最广，适配主流身份提供商。配置时需要提供实体 ID、回调地址与签名证书，我们侧会生成对应的元数据供您导入。启用 SSO 后仍保留本地应急账号，避免身份提供商故障导致无法登录。"),
    ("我们还关心数据加密的问题。", "传输层默认启用 TLS 1.3，存储侧使用 AES-256 加密，密钥由独立的密钥管理服务托管并定期轮换。您可以按项目申请专属密钥，轮换周期最短 30 天。敏感字段还支持应用层加密，密钥由您自行保管，我们无法解密。"),
    ("对了，收货地址改成滨江区江南大道 588 号，之前那个不用了。", "好的，收货地址已变更为滨江区江南大道 588 号，之前的西湖区地址不再使用。我们会同步更新到仓库与物流系统，如已发货请联系顾问拦截改址。"),
    ("保修期内上门服务怎么预约？", "保修期内可在工单系统提交上门申请，选择期望时间段后由服务商确认。一般会在 1 个工作日内致电确认，紧急工单可以走优先通道，24 小时内响应。上门只负责硬件检修与系统重装，数据恢复不在标准服务范围内。"),
    ("安装师傅上门的时间定在周五下午 3 点，可以吗？", "可以，已将上门安装时间记录为周五下午 3 点。师傅出发前 1 小时会电话联系您，请保持手机畅通；如需改期请至少提前 4 小时告知，避免产生空跑费用。"),
    ("我有一个朋友推荐码 YUNSHAN-888，还能叠加吗？", "可以叠加，推荐码 YUNSHAN-888 会在订单结算时抵扣相应金额，与批量折扣互不冲突，但不可与其他优惠券叠加使用。结算页输入后立即生效，订单完成后推荐双方都会收到权益。"),
    ("好的，那先这样，有需要我再联系。", "好的，感谢咨询！专属顾问会在 1 个工作日内与您联系。后续在本会话继续提出即可，我会接续之前的记录，无需重复说明。"),
    ("最后确认一下，周五之前能发货吗？", "正常库存情况下，确认订单后 2 个工作日内发货，周五前可以完成出库。发货后物流单号会推送至您的手机与邮箱，签收时请当面检查外包装，如有破损请拍照留证并 24 小时内反馈。"),
]

# 合成填充段（轮换复用）：把热窗口推过 6000 Token 的压缩触发线，控制评测成本。
# 注意：填充段刻意不含任何探针事实（订单号/姓名/地址/保修/上门时间/推荐码/日志天数）。
DETAIL_BLOCKS: List[str] = [
    "在数据安全方面，平台采用多副本存储与异地容灾相结合的设计：生产数据在主区域多副本保存，副本在独立可用区异步同步，遇到区域性故障时可以整体切换。备份策略为按小时增量、按天全量的滚动执行，备份文件统一加密存储并定期做完整性校验，校验失败的备份会自动标记并从轮换序列中剔除。恢复演练按季度组织，覆盖单表恢复、整库恢复与跨区域拉起三类场景，每次演练都会输出恢复时间与数据一致性的记录。如需把备份下载到本地留存，可以在管理后台提交导出申请，导出文件默认加密，需要用贵司提供的公钥进行封装后再交付。",
    "权限体系上，平台提供组织、角色与成员三层模型：组织对应独立的资源空间，角色决定可操作的资源类型与动作范围，成员可以同时拥有多个角色并按最宽范围生效。角色支持自定义，也内置了管理员、只读审计与实施人员三套常用模板；对高危操作（批量删除、密钥轮换、对外导出）默认要求二次确认并强制写入审计。权限变更建议按最小化原则推进，先在小范围验证再全量生效；如果对接了外部身份提供商，可以按部门自动同步角色。",
    "合规与审计方面，平台的审计记录覆盖登录行为、权限变更、数据导出与配置修改四类操作，记录字段包含操作人、操作对象、结果与来源地址，不包含业务数据正文。审计记录支持按时间范围与操作类型检索，并可导出为表格文件供内部审查使用；导出动作本身也会生成一条新的审计记录。如果贵司有独立的日志平台，可以把审计流通过接口推送到你们自己的存储上，避免两个系统之间的口径差异。",
    "网络接入方面，平台对外服务开启传输层加密，内部服务之间也启用双向校验；公网入口与企业专线可以同时存在，便于分阶段迁移。防火墙需要放行的是平台对外提供的域名与端口，反向代理场景下请保留原始来源地址，否则风控与审计记录会失真。白名单功能按项目维度维护，支持单个地址与网段两种粒度；变更后通常几分钟内生效，如果遇到缓存尚未刷新的情况，重新建立连接即可。",
    "客户端兼容性上，网页端支持主流现代浏览器，桌面端提供独立安装包并支持自动更新，移动端以查看与审批为主。不同端之间的数据实时同步，同一账号可以在多个端同时在线；如需强制单端登录，可在安全设置里开启限制项。桌面端的文件缓存默认加密存储，退出登录时会清理临时目录；当桌面端与网页端同时编辑同一份内容时，平台会保留两个版本并在界面上提示冲突位置，由使用人自行合并。",
    "工单与服务流程方面，提交工单时需要选择影响范围与期望时间：影响范围决定优先级，期望时间用于安排响应。工单状态从待受理、处理中、待验证到已关闭，每一步都会触发通知；如果超过约定时限未得到响应，系统会自动升级提醒。验证阶段建议由提交人确认结果后再关闭，避免同一问题反复开单；超出支持范围的定制需求会作为建议单记录，由产品侧定期评审。",
    "迁移与切换方面，正式操作前建议先在测试环境做一次演练：按同样的数据量与网络条件跑一遍完整流程，记录耗时与异常点，再确定正式切换的具体步骤。切换窗口选择业务低峰，切换前必须完成备份并验证可恢复性；若切换后发现数据异常，应优先回滚到切换前状态，再分析原因。演练与正式切换都应保留操作日志，便于复盘时对照每一步的实际结果。",
    "账单核对与对账方面，每月结算后可以在后台下载对账文件，包含使用权、结算周期与金额明细；对账文件与发票金额一致，如存在差异请先核对应付金额与调整项再发起申诉。对账异常常见原因包括席位变更未生效、优惠到期未续期与子账号重复计算，提交申诉时附上对账文件的行号可以加快处理。结算规则如有调整会提前在公告里说明，历史结算不受影响；需要变更结算方式时请在对账日前提出，以免跨周期生效。",
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
    {"id": "order-no", "turn": 2, "question": "我的订单号是多少？", "expect": ["TP-2026-0917"]},
    {"id": "contact", "turn": 4, "question": "我的联系人和公司分别是什么？", "expect": ["周航", "云杉"]},
    {"id": "invoice", "turn": 7, "question": "发票抬头写的是什么？", "expect": ["杭州云杉科技"]},
    {"id": "address-current", "turn": 15, "question": "我现在的收货地址是什么？", "expect": ["江南大道"], "forbidden": ["文三路"]},
    {"id": "warranty", "turn": 6, "question": "保修期是多久？", "expect": ["3 年", "三年"], "expect_mode": "any"},
    {"id": "onsite-time", "turn": 17, "question": "上门安装时间定在什么时候？", "expect": ["周五", "3 点"], "expect_mode": "all"},
    {"id": "referral", "turn": 18, "question": "推荐码是多少？", "expect": ["YUNSHAN-888"]},
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
