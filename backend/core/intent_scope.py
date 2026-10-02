"""UrbanOps support scope and explicit out-of-scope markers."""
from __future__ import annotations

from typing import Tuple


BUSINESS_SCOPE = (
    "UrbanOps 市政运维服务，包括设施台账、设备巡检、异常告警、故障排查、"
    "维修工单、应急预案、智慧路灯终端接入与运维权限管理。"
)

OUT_OF_SCOPE_MARKERS: Tuple[str, ...] = (
    "购物",
    "股票交易",
    "电商订单",
    "酒店预订",
    "电影购票",
    "快递寄件",
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
    "代码补全",
    "编程工具配置",
)


def strong_out_of_scope_markers(text: str) -> Tuple[str, ...]:
    normalized = str(text).lower()
    return tuple(marker for marker in OUT_OF_SCOPE_MARKERS if marker in normalized)
