"""Instance-scoped SDK usage capture for isolated Supervisor evaluations.

No global SDK patches; no prompt, response text or credential is recorded here.
One recorder belongs to one run, including its concurrent child Agent calls.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class UsageRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @contextmanager
    def instrument(self, *clients: tuple[str, Any]) -> Iterator["UsageRecorder"]:
        originals = []
        seen = set()
        active = True
        try:
            for label, client in clients:
                if id(client) in seen:
                    continue
                seen.add(id(client))
                messages = client.messages
                original = messages.create
                originals.append((messages, original))

                def wrap(create, component):
                    async def measured(*args, **kwargs):
                        if not active:
                            return await create(*args, **kwargs)
                        row = {"component": component, "model": kwargs.get("model", ""),
                               "usage_available": False, "error_type": ""}
                        self.calls.append(row)
                        started = time.perf_counter()
                        try:
                            response = await create(*args, **kwargs)
                            try:
                                usage = getattr(response, "usage", None)
                                if hasattr(usage, "model_dump"):
                                    usage = usage.model_dump()
                            except Exception:
                                # Observability must not replace a valid model response.
                                usage = None
                            if isinstance(usage, dict):
                                keys = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                        "cache_creation_input_tokens")
                                counts = {key: usage.get(key) for key in keys}
                                valid = all(type(counts[k]) is int and counts[k] >= 0
                                            for k in keys[:2])
                                valid = valid and all(counts[k] is None or
                                    (type(counts[k]) is int and counts[k] >= 0) for k in keys[2:])
                                if valid:
                                    row.update({key: counts[key] or 0 for key in keys})
                                    row["usage_available"] = True
                            return response
                        except BaseException as ex:
                            row["error_type"] = type(ex).__name__
                            raise
                        finally:
                            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    return measured

                messages.create = wrap(original, label)
            yield self
        finally:
            active = False
            for messages, original in reversed(originals):
                messages.create = original

    def summary(self) -> dict[str, Any]:
        known = [row for row in self.calls if row["usage_available"]]
        complete = len(known) == len(self.calls)
        totals = {key: sum(row[key] for row in known) for key in (
            "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        # Anthropic Messages usage: cache input fields are separate from input_tokens.
        observed = sum(totals.values())
        return {**totals, "llm_calls": len(self.calls), "known_usage_calls": len(known),
                "usage_complete": complete, "observed_tokens": observed,
                "total_tokens": observed if complete else None,
                "accounting": "anthropic_messages_input_output_plus_cache_input",
                "calls": list(self.calls)}
