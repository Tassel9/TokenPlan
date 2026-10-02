"""Scoped keyed fingerprints for sensitive diagnostic payloads."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from typing import Any, Optional, Union


_PROCESS_KEY = secrets.token_bytes(32)


def payload_hmac_sha256(
    value: Any,
    *,
    scope: str,
    key: Optional[Union[str, bytes]] = None,
) -> str:
    """Return a canonical HMAC fingerprint scoped to one request or run.

    A configured key keeps fingerprints stable across restarts. Without one,
    the process uses an ephemeral random key, which is safer than silently
    falling back to an unkeyed digest.
    """

    configured_key = os.getenv("TRACE_FINGERPRINT_KEY", "").encode("utf-8")
    secret = key if key is not None else (configured_key or _PROCESS_KEY)
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    if not secret:
        raise ValueError("payload fingerprint key must not be empty")
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        canonical = json.dumps({"type": type(value).__name__})
    message = f"{scope}\0{canonical}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()
