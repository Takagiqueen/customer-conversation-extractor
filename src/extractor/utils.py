"""
通用工具函数 —— 文本拼接、正则识别、关键词匹配、产品提取。

所有函数为纯函数，不依赖外部状态，便于单独测试。
"""

from __future__ import annotations

import re
from typing import List, Optional


# ============================================================================
# 文本拼接
# ============================================================================


def join_user_messages(turns: list[dict]) -> str:
    """拼接所有用户消息，用空格分隔"""
    return " ".join(t["content"] for t in turns if t["role"] == "user")


def join_agent_messages(turns: list[dict]) -> str:
    """拼接所有客服消息，用空格分隔"""
    return " ".join(t["content"] for t in turns if t["role"] == "agent")


def join_all_text(turns: list[dict]) -> str:
    """拼接全部对话文本"""
    return " ".join(t["content"] for t in turns)


def get_first_user_msg(turns: list[dict]) -> str:
    """获取用户第一条消息，用于情绪初始判断"""
    for t in turns:
        if t["role"] == "user":
            return t["content"]
    return ""


def get_last_user_msg(turns: list[dict]) -> str:
    """获取用户最后一条消息，用于情绪最终判断"""
    for t in reversed(turns):
        if t["role"] == "user":
            return t["content"]
    return ""


def get_first_agent_msg(turns: list[dict]) -> str:
    """获取客服第一条消息"""
    for t in turns:
        if t["role"] == "agent":
            return t["content"]
    return ""


# ============================================================================
# 关键词匹配
# ============================================================================


def has_any_keyword(text: str, keywords: list[str]) -> bool:
    """检查文本是否包含任意关键词"""
    return any(kw in text for kw in keywords)


def count_keywords(text: str, keywords: list[str]) -> int:
    """统计文本中命中的关键词种数"""
    return sum(1 for kw in keywords if kw in text)


def match_first_keyword(text: str, keywords: list[str]) -> Optional[str]:
    """返回第一个命中的关键词，未命中返回 None"""
    for kw in keywords:
        if kw in text:
            return kw
    return None


# ============================================================================
# 正则识别
# ============================================================================

# 订单号格式：DD + 8位日期 + - + 4位以上编号
_ORDER_ID_RE = re.compile(r"DD\d{8}-\d{4,}")

# 手机号（含脱敏格式 138xxxx5521）
_PHONE_RE = re.compile(r"1[3-9]\d(?:\d{8}|x{3,4}\d{2,4})")

# 地址特征：区/路/街/号/层/楼/栋 等
_ADDRESS_PATTERNS = [
    re.compile(r"[一-龥]{2,}(?:区|路|街|大道|道)[一-龥\d]+(?:号|楼|层|栋|幢|座)"),
    re.compile(r"(?:区|路|街)\s*[\d]+(?:号|楼|层)"),
]


def find_order_ids(text: str) -> list[str]:
    """提取所有订单号（去重）"""
    return list(dict.fromkeys(_ORDER_ID_RE.findall(text)))


def find_phone_numbers(text: str) -> list[str]:
    """提取所有手机号模式（去重）"""
    return list(dict.fromkeys(_PHONE_RE.findall(text)))


def has_address(text: str) -> bool:
    """检测文本是否包含地址信息"""
    return any(p.search(text) for p in _ADDRESS_PATTERNS)


# ============================================================================
# 产品提取
# ============================================================================

# 按优先级排列：长词在前避免短词误匹配（如"蓝牙耳机"要在"耳机"之前匹配）
_PRODUCT_KEYWORDS: list[str] = [
    "扫地机器人", "蓝牙耳机", "黑色真皮双肩包", "白色T恤", "蓝色衬衫",
    "蓝色杯子", "手机壳", "充电宝", "面膜", "碗",
]


def extract_products(text: str) -> list[str]:
    """从文本中提取提及的产品名称"""
    found = []
    consumed_ranges: list[tuple[int, int]] = []

    for kw in _PRODUCT_KEYWORDS:
        idx = text.find(kw)
        if idx == -1:
            continue
        # 检查是否已被更长的关键词覆盖
        if any(start <= idx < end for start, end in consumed_ranges):
            continue
        found.append(kw)
        consumed_ranges.append((idx, idx + len(kw)))

    return found


# ============================================================================
# 对话特征判断
# ============================================================================


def has_delayed_response(turns: list[dict]) -> bool:
    """检测用户是否抱怨响应延迟"""
    user_text = join_user_messages(turns)
    return has_any_keyword(user_text, [
        "等了", "半天", "没人理", "怎么不回了",
        "搞快点", "快一点", "等了半天",
    ])


def has_bot_complaint(turns: list[dict]) -> bool:
    """检测用户是否抱怨智能客服/机器人"""
    user_text = join_user_messages(turns)
    return has_any_keyword(user_text, [
        "智能客服", "机器人", "智障", "答非所问",
    ])


def user_gave_up(turns: list[dict]) -> bool:
    """检测用户是否中途放弃"""
    last = get_last_user_msg(turns)
    return has_any_keyword(last, [
        "算了", "不用了", "不看了", "自己", "别的地方",
    ])


# ============================================================================
# 证据摘取
# ============================================================================


def extract_quotes(turns: list[dict], keywords: list[str],
                   role: str = "user", max_count: int = 3) -> list[str]:
    """
    从对话中摘取包含指定关键词的原文语句作为证据。

    Args:
        turns: 对话消息列表
        keywords: 需要匹配的关键词列表
        role: 从哪个角色的消息中摘取 ("user" 或 "agent")
        max_count: 最多摘取条数

    Returns:
        匹配到的原文列表（按对话顺序），最多 max_count 条
    """
    quotes: list[str] = []
    for t in turns:
        if t["role"] != role:
            continue
        if has_any_keyword(t["content"], keywords):
            quotes.append(t["content"])
            if len(quotes) >= max_count:
                break
    return quotes


def pick_evidence_quotes(turns: list[dict],
                         user_keywords: list[str],
                         agent_keywords: list[str]) -> tuple[list[str], list[str]]:
    """
    同时从用户和客服消息中摘取证据。

    Returns:
        (user_quotes, agent_quotes)
    """
    return (
        extract_quotes(turns, user_keywords, role="user"),
        extract_quotes(turns, agent_keywords, role="agent"),
    )
