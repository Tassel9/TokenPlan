"""Shared DeepSeek configuration and response helpers."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_V4_FLASH_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_PRO_TOKENIZER = "deepseek-ai/DeepSeek-V4-Pro"
DEEPSEEK_V4_FLASH_TOKENIZER_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
DEEPSEEK_V4_PRO_TOKENIZER_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
_LEGACY_MODELS = {"deepseek-chat", "deepseek-reasoner"}


def load_deepseek_config(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Load a DeepSeek-only runtime config without ever logging the API key."""
    env = environ if environ is not None else os.environ
    api_key = (env.get("DEEPSEEK_API_KEY") or env.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")

    base_url = (
        env.get("DEEPSEEK_BASE_URL")
        or env.get("ANTHROPIC_BASE_URL")
        or DEEPSEEK_ANTHROPIC_BASE_URL
    ).strip()
    model = (
        env.get("DEEPSEEK_MODEL")
        or env.get("ANTHROPIC_MODEL")
        or DEEPSEEK_DEFAULT_MODEL
    ).strip()
    if model in _LEGACY_MODELS:
        logger.warning("检测到已停用的 DeepSeek 模型名，自动切换为 %s", DEEPSEEK_DEFAULT_MODEL)
        model = DEEPSEEK_DEFAULT_MODEL

    return {
        "provider": "deepseek",
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def deepseek_request_options() -> Dict[str, Any]:
    """Keep structured UrbanOps calls deterministic."""
    return {"extra_body": {"thinking": {"type": "disabled"}}}


def load_deepseek_tokenizer(model: str = DEEPSEEK_DEFAULT_MODEL) -> Any:
    """Load the revision-pinned official tokenizer without model weights."""
    from transformers import AutoTokenizer

    is_pro = "pro" in str(model or "").casefold()
    repository = DEEPSEEK_V4_PRO_TOKENIZER if is_pro else DEEPSEEK_V4_FLASH_TOKENIZER
    revision = (
        DEEPSEEK_V4_PRO_TOKENIZER_REVISION
        if is_pro
        else DEEPSEEK_V4_FLASH_TOKENIZER_REVISION
    )
    return AutoTokenizer.from_pretrained(repository, revision=revision)


def extract_text(response: Any) -> str:
    """Extract text blocks while safely ignoring optional thinking blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content or "")

    parts = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text") if block.get("type") == "text" else None
        else:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None) if block_type in (None, "text") else None
        if text:
            parts.append(str(text))
    return "\n".join(parts)
