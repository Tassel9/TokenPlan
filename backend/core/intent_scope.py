"""TokenPlan support scope and explicit out-of-scope markers."""
from __future__ import annotations

from typing import Tuple


BUSINESS_SCOPE = (
    "TokenPlan AI 编程订阅服务，包括套餐与权益、模型和 Token 额度、订阅账单、"
    "账号安全，以及 IDE 插件、代码补全和 API 调用相关技术支持。"
)

OUT_OF_SCOPE_MARKERS: Tuple[str, ...] = (
    "快递",
    "物流",
    "收货地址",
    "发货",
    "配送",
    "签收",
    "退货",
    "换货",
    "尺码",
    "鞋码",
    "上门取件",
    "航班",
    "机票",
    "酒店",
    "天气",
    "气温",
    "天气预报",
    "外卖",
    "体检报告",
    "股票",
    "播放音乐",
    "洗衣机",
)


def strong_out_of_scope_markers(text: str) -> Tuple[str, ...]:
    normalized = str(text).lower()
    return tuple(marker for marker in OUT_OF_SCOPE_MARKERS if marker in normalized)
