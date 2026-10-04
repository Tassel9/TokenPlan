"""Redact exposed credential-shaped literals before memory, models and traces."""
from __future__ import annotations

import re
from typing import Any

SECRET = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])|(?i:(?:api[_ -]?key|访问令牌|API密钥|密钥)\s*[:：=]\s*)[A-Za-z0-9_-]{12,}")
SAFE_RESPONSE = (
    "请不要分享完整 API Key。你贴出的密钥可能已经暴露，请到签发该密钥的官方控制台自行撤销并轮换，"
    "再更新本地环境变量或客户端配置。不要再次发送新密钥。"
    "我不能替你直接配置账户或发起验证请求，可以说明 Key、Base URL 和客户端的配置检查步骤。"
    "你使用的是哪个客户端或编码工具？"
)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET.sub("[REDACTED_API_KEY]", value)
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value
