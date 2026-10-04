# -*- coding: utf-8 -*-
"""主链路耗时对照：画像更新「同步内联」vs「RabbitMQ 异步入队」。

背景：改造前的主链路在每轮对话末尾触发用户画像更新（真实 LLM 抽取 + BGE 向量化 +
Chroma 写入）；改造后同一份工作经 RabbitMQ 交给独立 worker 进程。

四个口径（全部真实依赖，无 stub）：
  A 内联同步     ：stage(SQLite) + process_profile_update(真实 LLM 抽取 + BGE + Chroma) + pending 清理
                   —— 即「不做异步化」时主链路每轮要多付的成本，也是异步化可从主链路移走的量
  B 异步入队     ：真实 RabbitMQ publish（publisher confirm）+ SQLite staging，主链路不等抽取结果
                   —— 即当前实现主链路每轮真实成本
  C worker 完成  ：真实 consumer 消费同一批消息，测「响应返回后画像多久真正落库」
  D 持久化验证   ：无消费者时消息驻留 broker（进程重启不丢），消费者启动后清空

用法：.venv-win\\Scripts\\python.exe evaluation/benchmarks/bench_rabbitmq_offload.py
依赖：RabbitMQ(默认 amqp://urbanops:urbanops123@127.0.0.1:5672/)、DeepSeek API。

口径边界（必读，避免误引）：
  * A 臂「同步内联」是**实现对照**（主链路全程 await），不是改造前历史实现——
    改造前是 asyncio.create_task(update_profile(...)) 的 fire-and-forget，
    请求路径成本实测仅 ~1.6 µs。因此 839 → 1.4 ms 应表述为
    「工作重量 vs 请求侧入队开销」，不要写成「改造前后请求耗时的实测差」。
  * 消费者默认与 API 同进程（LONG_TERM_MEMORY_WORKER_ENABLED=true，单容器 uvicorn），
    不存在进程隔离；worker 的嵌入/Chroma 写入为同步调用，会占用共享事件循环数毫秒/条。
  * 入队成本为同一宿主 broker、无 TLS；跨网络部署需重新测量。
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from memory.conversation_memory import MemoryManager  # noqa: E402
from memory.profile_update_queue import (  # noqa: E402
    ProfileUpdateJob,
    RabbitMQProfileUpdateQueue,
)

BENCH_USER = "u-bench"

# 与 bench_profile_update_cost.py 保持同一批消息，便于口径对齐。
MESSAGES: List[Tuple[str, str, str]] = [
    (BENCH_USER, "以后回答我都希望简洁一点，别长篇大论。", "命中 style.response_length"),
    (BENCH_USER, "我平时用 macOS，开发用 PyCharm。", "命中 environment.*"),
    (BENCH_USER, "巡检班次我希望改成夜班，以后就选夜班吧。", "命中 preference.inspection_shift（触发 staging）"),
    (BENCH_USER, "我这张工单的工单撤回到底什么时候到账？", "无长期事实（应落空）"),
    (BENCH_USER, "回答简洁一点，不要长篇大论。", "与第 1 条重复（验幂等）"),
    (BENCH_USER, "我的验证码是 8848，帮我记一下。", "敏感信息（应被拒）"),
]

ENQUEUE_OPS = 120
CONCURRENCY = 8
CONCURRENCY_WAVES = 5
INLINE_ROUNDS = 2


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("no samples collected")
    ordered = sorted(values)

    def _pct(p: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
        return ordered[index]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _pct(0.5),
        "p95": _pct(0.95),
        "max": max(values),
    }


def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def _resolve_rabbit_url() -> str:
    return os.getenv(
        "BENCH_RABBITMQ_URL",
        "amqp://urbanops:urbanops123@127.0.0.1:5672/",
    )


async def _queue_depth(channel: Any, queue_name: str) -> int:
    declared = await channel.declare_queue(queue_name, passive=True)
    return int(declared.declaration_result.message_count or 0)


async def _purge_queue(channel: Any, queue_name: str) -> None:
    declared = await channel.declare_queue(queue_name, passive=True)
    await declared.purge()


def _load_runs(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read previously recorded runs so repeated invocations accumulate samples."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    runs = payload.get("runs")
    if isinstance(runs, list):
        return runs
    if "arms" in payload:  # 旧单次格式 → 迁移成一条 run
        return [{
            "measured_at": payload.get("measured_at"),
            "arms": payload.get("arms"),
            "headline": payload.get("headline"),
        }]
    return []


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(path: List[str]) -> List[float]:
        values: List[float] = []
        for run in runs:
            node: Any = run
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, (int, float)):
                values.append(float(node))
        return values

    def summarise(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "runs": len(runs),
        "main_chain_inline_ms": summarise(
            collect(["headline", "main_chain_inline_ms"])
        ),
        "main_chain_enqueue_ms": summarise(
            collect(["headline", "main_chain_enqueue_ms"])
        ),
        "offloaded_ms": summarise(collect(["headline", "offloaded_ms"])),
        "llm_extraction_ms": summarise(
            collect(["headline", "llm_extraction_ms"])
        ),
        "worker_idle_lag_ms": summarise(
            collect(["headline", "worker_idle_lag_ms"])
        ),
        "worker_processing_ms": summarise(
            collect(["headline", "worker_processing_ms"])
        ),
        "enqueue_p95_ms": summarise(
            collect(["arms", "enqueue_rabbitmq", "enqueue_ms", "p95"])
        ),
        "enqueue_concurrency_c8_p95_ms": summarise(
            collect([
                "arms",
                "enqueue_rabbitmq",
                "concurrency_c8",
                "per_op_ms",
                "p95",
            ])
        ),
    }


async def _drain_and_wait(
    queue: RabbitMQProfileUpdateQueue,
    *,
    expected: int,
    timeout_s: float,
) -> float:
    """Block until the broker reports an empty queue; returns elapsed seconds."""

    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        channel = queue._consumer_channel  # noqa: SLF001
        if channel is not None:
            depth = await _queue_depth(channel, queue.QUEUE_NAME)
            if depth == 0:
                return time.perf_counter() - started
        await asyncio.sleep(0.25)
    raise TimeoutError(f"队列在 {timeout_s:.0f}s 内未清空（期望处理 {expected} 条）")


async def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY（.env），无法测量真实 LLM 抽取")
        return 1

    rabbit_url = _resolve_rabbit_url()
    print(f"RabbitMQ : {rabbit_url}")

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-rabbit-bench-"))
    embedding = BGEEmbeddingProvider(
        model_name="BAAI/bge-small-zh-v1.5",
        revision=None,
    )
    memory = MemoryManager(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_password=os.getenv("REDIS_PASSWORD") or None,
        session_db_path=str(temp_root / "sessions.sqlite3"),
        chroma_host="127.0.0.1",
        chroma_port=1,  # 强制本地内嵌 Chroma，避免占用宿主 8001（已被其他项目占用）
        chroma_path=str(temp_root / "chroma"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        profile_embedding_provider=embedding,
        allow_embedded_chroma_fallback=True,
    )
    await asyncio.to_thread(memory.preload_profile_embedding)

    # 每条 LLM 抽取调用单独计时（同步内联路径的分解项）。
    llm_ms: List[float] = []
    original_create = memory._client.messages.create  # noqa: SLF001

    async def timed_create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original_create(*args, **kwargs)
        finally:
            llm_ms.append((time.perf_counter() - started) * 1000.0)

    memory._client.messages.create = timed_create  # noqa: SLF001

    producer = RabbitMQProfileUpdateQueue(
        url=rabbit_url,
        handler=memory.process_profile_update,
        stage_handler=memory.stage_profile_update,
        cleanup_handler=memory.clear_staged_profile_update,
        worker_enabled=False,
        prefetch_count=1,
    )
    publish_ms: List[float] = []
    original_publish = producer._publish_job  # noqa: SLF001

    async def timed_publish(job: ProfileUpdateJob, *, attempt: int) -> None:
        started = time.perf_counter()
        try:
            return await original_publish(job, attempt=attempt)
        finally:
            publish_ms.append((time.perf_counter() - started) * 1000.0)

    producer._publish_job = timed_publish  # noqa: SLF001

    try:
        await producer.start()
    except Exception as ex:
        print(f"RabbitMQ 不可用（{type(ex).__name__}: {ex}）；先启动 docker compose up -d rabbitmq")
        return 1
    await _purge_queue(producer._publish_channel, producer.QUEUE_NAME)  # noqa: SLF001

    report: Dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "session_backend": "sqlite",
        "rabbitmq_url": rabbit_url,
        "messages": [note for _, _, note in MESSAGES],
    }

    try:
        # ── A 内联同步：不做异步化时主链路要多付的成本 ─────────────────────────
        print("\n=== A 同步内联（stage + 抽取 + 向量化 + 落库，主链路全程 await）===")
        inline_rows: List[Dict[str, Any]] = []
        inline_total: List[float] = []
        inline_stage: List[float] = []
        inline_llm: List[float] = []
        for round_index in range(1, INLINE_ROUNDS + 1):
            for index, (user_id, message, note) in enumerate(MESSAGES, start=1):
                conv_id = f"inline-{round_index}-{index}"
                event_id = f"inline-{round_index}-{index}"
                effective_at = datetime.now(timezone.utc)
                llm_before = len(llm_ms)
                started = time.perf_counter()
                stage_started = time.perf_counter()
                staged = await memory.stage_profile_update(
                    user_id,
                    conv_id,
                    user_message=message,
                    effective_at=effective_at,
                    event_id=event_id,
                )
                stage_elapsed = (time.perf_counter() - stage_started) * 1000.0
                await memory.process_profile_update(
                    user_id,
                    conv_id,
                    user_message=message,
                    effective_at=effective_at,
                    event_id=event_id,
                )
                total_elapsed = (time.perf_counter() - started) * 1000.0
                llm_elapsed = llm_ms[llm_before] if len(llm_ms) > llm_before else 0.0
                inline_rows.append({
                    "round": round_index,
                    "note": note,
                    "staged": staged,
                    "stage_ms": round(stage_elapsed, 3),
                    "llm_ms": round(llm_elapsed, 3),
                    "total_ms": round(total_elapsed, 3),
                })
                inline_total.append(total_elapsed)
                inline_stage.append(stage_elapsed)
                inline_llm.append(llm_elapsed)
                print(
                    f"  r{round_index}.{index} {note[:22]:<24} "
                    f"总计 {total_elapsed:8.1f} ms（LLM {llm_elapsed:7.1f} / "
                    f"stage {stage_elapsed:5.2f}）"
                )
        inline_stats = _stats(inline_total)
        report["arms"] = {
            "inline_sync": {
                "rows": inline_rows,
                "total_ms": inline_stats,
                "stage_ms": _stats(inline_stage),
                "llm_ms": _stats(inline_llm),
                "chroma_rows": memory._profile.count(),  # noqa: SLF001
            }
        }
        print(
            f"  小计：mean {_fmt(inline_stats['mean'])} ms / p50 {_fmt(inline_stats['p50'])} "
            f"/ max {_fmt(inline_stats['max'])}（LLM 占 {statistics.fmean(inline_llm):.0f} ms）"
        )

        # ── B 异步入队：真实 broker publish + confirm ──────────────────────────
        print(f"\n=== B RabbitMQ 异步入队（staging + publish confirm，{ENQUEUE_OPS} 次）===")
        enqueue_ops: List[float] = []
        enqueue_stage: List[float] = []
        stage_hits = 0
        for index in range(ENQUEUE_OPS):
            user_id, message, _ = MESSAGES[index % len(MESSAGES)]
            conv_id = f"enqueue-{index}"
            effective_at = datetime.now(timezone.utc)
            stage_started = time.perf_counter()
            staged = await memory.stage_profile_update(
                user_id,
                conv_id,
                user_message=message,
                effective_at=effective_at,
                event_id=f"enqueue-{index}",
            )
            stage_elapsed = (time.perf_counter() - stage_started) * 1000.0
            started = time.perf_counter()
            await producer.enqueue(
                user_id=user_id,
                conv_id=conv_id,
                user_message=message,
                effective_at=effective_at,
            )
            enqueue_ops.append((time.perf_counter() - started) * 1000.0)
            enqueue_stage.append(stage_elapsed)
            stage_hits += 1 if staged else 0
        enqueue_stats = _stats(enqueue_ops)
        publish_stats = _stats(publish_ms)
        report["arms"]["enqueue_rabbitmq"] = {
            "ops": ENQUEUE_OPS,
            "stage_hits": stage_hits,
            "enqueue_ms": enqueue_stats,
            "publish_confirm_ms": publish_stats,
            "stage_ms": _stats(enqueue_stage),
        }
        print(
            f"  入队总成本：mean {_fmt(enqueue_stats['mean'])} ms / p50 {_fmt(enqueue_stats['p50'])} "
            f"/ p95 {_fmt(enqueue_stats['p95'])}"
        )
        print(
            f"  其中 publish+confirm：mean {_fmt(publish_stats['mean'])} ms / "
            f"p95 {_fmt(publish_stats['p95'])}；staging 命中 {stage_hits}/{ENQUEUE_OPS} 次"
        )

        # 序列化单独计时（不受 broker 往返影响）
        serialize_ms: List[float] = []
        sample_job = ProfileUpdateJob.create(
            user_id=BENCH_USER,
            conv_id="serialize",
            user_message=MESSAGES[0][1],
            effective_at=datetime.now(timezone.utc),
        )
        for _ in range(500):
            started = time.perf_counter()
            sample_job.to_body()
            serialize_ms.append((time.perf_counter() - started) * 1000.0)
        report["arms"]["enqueue_rabbitmq"]["serialize_ms"] = _stats(serialize_ms)
        print(f"  纯序列化：mean {_fmt(statistics.fmean(serialize_ms))} ms")

        # 并发入队：主链路在并发下是否仍为毫秒级
        async def one_op(index: int) -> float:
            user_id, message, _ = MESSAGES[index % len(MESSAGES)]
            started = time.perf_counter()
            await producer.enqueue(
                user_id=user_id,
                conv_id=f"concurrent-{index}",
                user_message=message,
                effective_at=datetime.now(timezone.utc),
            )
            return (time.perf_counter() - started) * 1000.0

        concurrent_ms: List[float] = []
        wall_started = time.perf_counter()
        for wave in range(CONCURRENCY_WAVES):
            wave_ms = await asyncio.gather(
                *[one_op(wave * CONCURRENCY + i) for i in range(CONCURRENCY)]
            )
            concurrent_ms.extend(wave_ms)
        wall_elapsed = time.perf_counter() - wall_started
        concurrent_stats = _stats(concurrent_ms)
        report["arms"]["enqueue_rabbitmq"]["concurrency_c8"] = {
            "ops": len(concurrent_ms),
            "per_op_ms": concurrent_stats,
            "wall_s": wall_elapsed,
            "throughput_ops_per_s": len(concurrent_ms) / wall_elapsed,
        }
        print(
            f"  并发 c={CONCURRENCY}：单次 mean {_fmt(concurrent_stats['mean'])} ms / "
            f"p95 {_fmt(concurrent_stats['p95'])}；吞吐 "
            f"{len(concurrent_ms) / wall_elapsed:.0f} ops/s"
        )
        await _purge_queue(producer._publish_channel, producer.QUEUE_NAME)  # noqa: SLF001

        # ── D 持久化：无消费者时消息驻留 broker ───────────────────────────────
        print("\n=== D 持久化验证（无消费者 → 启动消费 → 队列清空）===")
        for index in range(3):
            user_id, message, _ = MESSAGES[index % len(MESSAGES)]
            await producer.enqueue(
                user_id=user_id,
                conv_id=f"durable-{index}",
                user_message=message,
                effective_at=datetime.now(timezone.utc),
            )
        depth_before = await _queue_depth(producer._publish_channel, producer.QUEUE_NAME)  # noqa: SLF001
        print(f"  发布 3 条、无消费者：队列深度 = {depth_before}")
        drained_jobs: List[str] = []

        async def drain_handler(
            user_id: str,
            conv_id: str,
            *,
            user_message: str = "",
            effective_at: Any = None,
            event_id: str = "",
        ) -> None:
            drained_jobs.append(event_id)

        drain_consumer = RabbitMQProfileUpdateQueue(
            url=rabbit_url,
            handler=drain_handler,
            worker_enabled=True,
            prefetch_count=1,
        )
        await drain_consumer.start()
        drained_s = await _drain_and_wait(drain_consumer, expected=3, timeout_s=30)
        await drain_consumer.close()
        depth_after = await _queue_depth(producer._publish_channel, producer.QUEUE_NAME)  # noqa: SLF001
        report["arms"]["durability"] = {
            "depth_without_consumer": depth_before,
            "depth_after_consumer": depth_after,
            "consumed": len(drained_jobs),
            "drain_s": drained_s,
        }
        print(f"  消费者启动后 {drained_s:.2f}s 清空（剩余 {depth_after}），处理 {len(drained_jobs)} 条")

        # ── C worker 完成延迟：响应返回后画像多久真正落库 ──────────────────────
        print("\n=== C worker 完成延迟（真实抽取跑在 worker 侧，prefetch=1 串行消费）===")
        completion: Dict[str, float] = {}
        processing_ms: List[float] = []

        async def measured_handler(
            user_id: str,
            conv_id: str,
            *,
            user_message: str = "",
            effective_at: Any = None,
            event_id: str = "",
        ) -> None:
            process_started = time.perf_counter()
            await memory.process_profile_update(
                user_id,
                conv_id,
                user_message=user_message,
                effective_at=effective_at,
                event_id=event_id,
            )
            processing_ms.append((time.perf_counter() - process_started) * 1000.0)
            completion[event_id] = time.perf_counter()

        async def wait_job(job_id: str, timeout_s: float) -> Optional[float]:
            deadline_at = time.perf_counter() + timeout_s
            while job_id not in completion:
                if time.perf_counter() > deadline_at:
                    return None
                await asyncio.sleep(0.05)
            return completion[job_id]

        worker = RabbitMQProfileUpdateQueue(
            url=rabbit_url,
            handler=measured_handler,
            worker_enabled=True,
            prefetch_count=1,
        )
        await worker.start()

        # C1 空闲 worker：单条 job 从入队到落库的真实延迟（线上稳态）
        print("  [C1] 空闲 worker，单条 job 的异步生效延迟")
        idle_rows: List[Dict[str, Any]] = []
        idle_lag_ms: List[float] = []
        for index, (user_id, message, note) in enumerate(MESSAGES, start=1):
            conv_id = f"worker-idle-{index}"
            effective_at = datetime.now(timezone.utc)
            job = ProfileUpdateJob.create(
                user_id=user_id,
                conv_id=conv_id,
                user_message=message,
                effective_at=effective_at,
            )
            published_at = time.perf_counter()
            await producer.enqueue(
                user_id=user_id,
                conv_id=conv_id,
                user_message=message,
                effective_at=effective_at,
            )
            finished_at = await wait_job(job.job_id, timeout_s=30)
            if finished_at is None:
                idle_rows.append({"note": note, "lag_ms": None})
                continue
            lag = (finished_at - published_at) * 1000.0
            idle_lag_ms.append(lag)
            idle_rows.append({"note": note, "lag_ms": round(lag, 1)})
            print(f"    {note[:24]:<26} 入队→落库 {lag:8.1f} ms")
        idle_stats = _stats(idle_lag_ms) if idle_lag_ms else None

        # C2 积压：一次性发布 6 条 → 串行消费，观察背压下的排队长尾
        print("  [C2] 积压：一次性发布 6 条后串行消费")
        pending_jobs: List[Tuple[str, float, str]] = []
        for index, (user_id, message, note) in enumerate(MESSAGES, start=1):
            conv_id = f"worker-backlog-{index}"
            effective_at = datetime.now(timezone.utc)
            job = ProfileUpdateJob.create(
                user_id=user_id,
                conv_id=conv_id,
                user_message=message,
                effective_at=effective_at,
            )
            await producer.enqueue(
                user_id=user_id,
                conv_id=conv_id,
                user_message=message,
                effective_at=effective_at,
            )
            pending_jobs.append((job.job_id, time.perf_counter(), note))
        backlog_started = time.perf_counter()
        backlog_rows: List[Dict[str, Any]] = []
        backlog_lag_ms: List[float] = []
        for job_id, published_at, note in pending_jobs:
            finished_at = await wait_job(job_id, timeout_s=120)
            if finished_at is None:
                backlog_rows.append({"note": note, "lag_ms": None})
                continue
            lag = (finished_at - published_at) * 1000.0
            backlog_lag_ms.append(lag)
            backlog_rows.append({"note": note, "lag_ms": round(lag, 1)})
            print(f"    {note[:24]:<26} 入队→落库 {lag:8.1f} ms")
        backlog_s = time.perf_counter() - backlog_started
        await worker.close()

        processing_stats = _stats(processing_ms) if processing_ms else None
        report["arms"]["worker_completion"] = {
            "idle_rows": idle_rows,
            "idle_lag_ms": idle_stats,
            "backlog_rows": backlog_rows,
            "backlog_lag_ms": _stats(backlog_lag_ms) if backlog_lag_ms else None,
            "backlog_drain_s": backlog_s,
            "processing_ms": processing_stats,
            "processed": len(processing_ms),
            "chroma_rows_total": memory._profile.count(),  # noqa: SLF001
        }
        if idle_stats:
            print(
                f"    C1 小计：mean {_fmt(idle_stats['mean'])} ms / "
                f"max {_fmt(idle_stats['max'])}（空闲 worker 下画像异步生效延迟）"
            )
        if processing_stats:
            print(f"    单条处理耗时（worker 侧）：mean {_fmt(processing_stats['mean'])} ms")
        if backlog_lag_ms:
            print(
                f"    C2 积压 6 条：{backlog_s:.2f}s 清空，末条延迟 "
                f"{_fmt(backlog_lag_ms[-1])} ms（串行消费的背压下延迟线性累积）"
            )

        # ── 汇总口径 ──────────────────────────────────────────────────────────
        headline = {
            "main_chain_inline_ms": inline_stats["mean"],
            "main_chain_enqueue_ms": enqueue_stats["mean"],
            "offloaded_ms": inline_stats["mean"] - enqueue_stats["mean"],
            "offloaded_ratio": 1 - enqueue_stats["mean"] / inline_stats["mean"],
            "llm_extraction_ms": statistics.fmean(inline_llm),
            "worker_idle_lag_ms": idle_stats["mean"] if idle_stats else None,
            "worker_processing_ms": (
                processing_stats["mean"] if processing_stats else None
            ),
        }
        report["headline"] = headline
        print("\n=== 汇总（口径：同步执行对照 vs 请求侧入队，非改造前后实测差）===")
        print(
            f"  同一 job：同步执行 {_fmt(headline['main_chain_inline_ms'])} ms/轮 → "
            f"请求侧入队 {_fmt(headline['main_chain_enqueue_ms'])} ms/轮"
        )
        print(
            f"  工作重量与请求侧开销之差：{_fmt(headline['offloaded_ms'])} ms/轮"
            f"（{headline['offloaded_ratio'] * 100:.1f}%，其中 LLM 抽取 "
            f"{_fmt(headline['llm_extraction_ms'])} ms；若写成简历句请用"
            f"「把 839ms 的抽取与向量化交给队列后台执行，请求侧仅付 1.4ms 入队」）"
        )
        if idle_stats and processing_stats:
            print(
                f"  画像异步生效延迟：入队后 mean {_fmt(idle_stats['mean'])} ms"
                f"（worker 单条处理 mean {_fmt(processing_stats['mean'])} ms）"
            )
    finally:
        await producer.close()
        # 会话库和 Chroma 均位于临时目录。
        try:
            await _purge_queue(producer._publish_channel, producer.QUEUE_NAME)  # noqa: SLF001
        except Exception:
            pass
        memory.session_store.close()
        shutil.rmtree(temp_root, ignore_errors=True)

    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "rabbitmq_offload_sqlite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(out)
    runs.append({
        "measured_at": report["measured_at"],
        "arms": report["arms"],
        "headline": report["headline"],
    })
    aggregate = _aggregate(runs)
    out.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "environment": {
                    "session_backend": "sqlite",
                    "rabbitmq_url": rabbit_url,
                    "messages": report["messages"],
                },
                "runs": runs,
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    inline_agg = aggregate.get("main_chain_inline_ms") or {}
    enqueue_agg = aggregate.get("main_chain_enqueue_ms") or {}
    if inline_agg and enqueue_agg:
        print(
            f"  累积 {aggregate['runs']} 轮：内联 {_fmt(inline_agg['min'])}–"
            f"{_fmt(inline_agg['max'])} ms（均值 {_fmt(inline_agg['mean'])}）→ "
            f"入队 {_fmt(enqueue_agg['min'])}–{_fmt(enqueue_agg['max'])} ms"
            f"（均值 {_fmt(enqueue_agg['mean'])}）"
        )
    print("\ndetail ->", out)
    return 0


if __name__ == "__main__":
    # Windows ProactorEventLoop 在解释器退出阶段的 GC 噪音与结果无关，
    # 这里显式刷新输出后直接退出，保证脚本退出码可用于 CI 断言。
    exit_code = asyncio.run(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
