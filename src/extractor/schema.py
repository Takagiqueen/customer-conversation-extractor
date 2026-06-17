"""
客服对话结构化提取 —— Pydantic Schema 定义

设计目标：为"客服主管周报"提供结构化数据支撑，
覆盖问题分类、解决效率、情绪变化、运营指标、风险预警、隐私合规等维度。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# 枚举定义 —— 所有可控字段的取值范围
# ============================================================================


class Channel(str, Enum):
    """对话渠道"""

    ONLINE = "在线"
    PHONE = "电话"
    EMAIL = "邮件"
    SOCIAL = "社交媒体"


class PrimaryIntent(str, Enum):
    """用户主要意图 —— 用户联系客服想要达成的目标"""

    REFUND_RETURN = "退款/退货"
    EXCHANGE = "换货"
    ORDER_CANCEL = "取消订单"
    LOGISTICS = "物流查询"
    PRODUCT_INQUIRY = "商品咨询"
    PROMOTION_COUPON = "优惠券/促销"
    ACCOUNT_SECURITY = "账号安全"
    COMPLAINT = "投诉"
    SUGGESTION = "建议反馈"
    SYSTEM_USAGE = "系统使用"
    URGE_FOLLOWUP = "催促/跟进"
    NO_SUBSTANTIVE = "无实质问题"
    OTHER = "其他"


class IssueCategory(str, Enum):
    """
    问题分类 —— 从业务角度对问题进行归类。

    与 PrimaryIntent 的差异：
    - Intent 是"用户想达成什么"（退款、咨询、投诉…）
    - Category 是"问题出在哪个业务环节"（质量、物流、政策…）

    同一个 Intent 可能对应不同 Category。例如"退款"可能因为
    质量问题（商品质量）或七天无理由（售后政策）。
    """

    PRODUCT_QUALITY = "商品质量"
    LOGISTICS_DELIVERY = "物流配送"
    REFUND_TIMELINESS = "退款到账"
    ACCOUNT_SECURITY = "账号安全"
    PROMOTION_RULES = "促销规则"
    PRODUCT_INFO = "产品信息"
    AFTERSALES_POLICY = "售后政策"
    SYSTEM_EXPERIENCE = "系统体验"
    INVENTORY = "商品库存"
    SERVICE_RESPONSE = "服务响应"
    COLOR_MISMATCH = "色差/描述不符"
    OTHER = "其他"


class ResolutionStatus(str, Enum):
    """解决状态 —— 单次对话结束时的问题处理结果"""

    RESOLVED = "已解决"
    PARTIALLY_RESOLVED = "部分解决"
    UNRESOLVED = "未解决"
    NO_ACTION_NEEDED = "无需解决"
    PENDING_FOLLOWUP = "待跟进"


class Sentiment(str, Enum):
    """
    用户情绪标签。

    说明：标签按强度递增排列（positive < neutral < negative < anxious < angry），
    方便周报做情绪变化趋势分析（例如「初始 angry → 最终 neutral」）。
    """

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANXIOUS = "anxious"
    ANGRY = "angry"


class UrgencyLevel(str, Enum):
    """
    紧急程度 —— 从业务影响角度评估。

    - low: 一般咨询，不影响交易
    - medium: 影响用户体验但可延迟处理
    - high: 涉及退款/金钱/时效，需尽快解决
    - critical: 涉及安全/法律/大规模投诉，需立即升级
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFlag(str, Enum):
    """
    风险标签 —— 用于周报中的风险预警模块。

    一条对话可同时打多个标签。
    """

    CHURN_RISK = "用户流失风险"
    PR_RISK = "舆情风险"
    LEGAL_RISK = "法律风险"
    ACCOUNT_RISK = "账号安全风险"
    ESCALATION_RISK = "投诉升级风险"
    BATCH_QUALITY = "批量质量问题"
    PRIVACY_RISK = "隐私泄露风险"


class PIIType(str, Enum):
    """隐私信息类型 —— 用于合规审计"""

    PHONE = "phone"
    ADDRESS = "address"
    EMAIL = "email"
    ID_CARD = "id_card"
    BANK_ACCOUNT = "bank_account"
    REAL_NAME = "real_name"


# ============================================================================
# 子模型
# ============================================================================


class SubIssue(BaseModel):
    """
    子诉求 —— 当一条对话包含多个独立问题时使用。

    例如：用户同时询问「退货进度」和「新快递到哪了」，
    这是两个独立的子诉求，需分别记录。
    """

    issue_summary: str = Field(
        description="子诉求一句话摘要，≤60 字",
    )
    issue_category: IssueCategory = Field(
        description="该子诉求的业务分类",
    )
    resolution_status: ResolutionStatus = Field(
        description="该子诉求的解决状态",
    )
    resolution_action: Optional[str] = Field(
        default=None,
        description="针对该子诉求的具体处理措施",
    )


class Evidence(BaseModel):
    """
    提取证据 —— 记录支持提取结论的原文片段。

    设计理由：客服主管审核周报时需要回溯原文验证结论，
    证据字段让审核过程可追溯、可审计。
    """

    user_quotes: list[str] = Field(
        default_factory=list,
        description="用户原文中支持判断的关键语句（≤3 条）",
    )
    agent_quotes: list[str] = Field(
        default_factory=list,
        description="客服原文中支持判断的关键语句（≤3 条）",
    )
    reasoning: str = Field(
        default="",
        description="提取该条记录的整体推理依据，≤150 字",
    )


# ============================================================================
# 主模型 —— 单条对话的完整提取结果
# ============================================================================


class ExtractionResult(BaseModel):
    """
    单条客服对话的结构化提取结果。

    ## 设计原则
    - **面向周报**：每个字段都服务于周报的某个统计维度
    - **可验证**：evidence 字段确保每项判断都可追溯到原文
    - **可审计**：confidence + needs_manual_review 支持人机协同质检
    """

    # ---- 基础元信息 ----
    conversation_id: str = Field(
        description="对话唯一标识，对应原始数据中的 id 字段",
    )
    channel: Channel = Field(
        description="对话渠道，用于周报中按渠道统计对话量",
    )
    agent: str = Field(
        description="客服人员名称，用于周报中按客服统计绩效",
    )
    turn_count: int = Field(
        description="原始 turns 数组的消息条数，用于衡量对话长度",
    )

    # ---- 用户问题 ----
    primary_intent: PrimaryIntent = Field(
        description="用户主要意图，用于周报中按意图分布分析用户需求结构",
    )
    issue_category: IssueCategory = Field(
        description="问题归属分类，用于周报中按业务环节定位高频问题",
    )
    issue_summary: str = Field(
        description="用户问题一句话摘要，≤80 字，用于周报中快速浏览问题清单",
    )
    products_mentioned: list[str] = Field(
        default_factory=list,
        description="对话中提及的商品名称，用于周报中统计商品相关投诉/咨询热度",
    )

    # ---- 多诉求处理 ----
    is_multi_issue: bool = Field(
        description="是否包含多个独立诉求，用于衡量对话复杂度",
    )
    sub_issues: list[SubIssue] = Field(
        default_factory=list,
        description="子诉求列表。单诉求时为空列表；多诉求时每项对应一个独立问题",
    )

    # ---- 解决情况 ----
    resolution_status: ResolutionStatus = Field(
        description="整体解决状态，用于周报中计算解决率（核心 KPI）",
    )
    resolution_action: str = Field(
        description="客服采取的具体解决措施摘要，≤100 字",
    )
    customer_next_action: Optional[str] = Field(
        default=None,
        description="用户接下来需要做的事（如「寄回商品」「等待退款到账」）",
    )
    agent_next_action: Optional[str] = Field(
        default=None,
        description="客服接下来需要跟进的事（如「24小时内回电」「到货后通知」）",
    )

    # ---- 情绪与风险 ----
    user_sentiment_initial: Sentiment = Field(
        description="用户首次发言时的情绪，用于评估用户进线时的初始状态",
    )
    user_sentiment_final: Sentiment = Field(
        description="用户最后发言时的情绪，与 initial 对比反映客服处理效果",
    )
    urgency_level: UrgencyLevel = Field(
        description="问题紧急程度，用于周报中标记需优先关注的高优对话",
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="风险标签列表，用于周报中的风险预警板块",
    )

    # ---- 运营统计 ----
    complaint_related: bool = Field(
        description="是否涉及投诉（含明确投诉词和行为），用于周报中统计投诉率",
    )
    human_transfer: bool = Field(
        description="是否发生了人工转接（机器人→人工或跨客服转接），用于评估自助解决率",
    )
    compensation_offered: bool = Field(
        description="是否提供了补偿（优惠券/退款/赠品/免运费等），用于核算服务成本",
    )

    # ---- 隐私识别 ----
    order_ids: list[str] = Field(
        default_factory=list,
        description="对话中出现的所有订单号，用于订单维度的关联分析",
    )
    pii_detected: bool = Field(
        description="是否检测到个人隐私信息（手机号/地址/身份证等）",
    )
    pii_types: list[PIIType] = Field(
        default_factory=list,
        description="检测到的隐私信息类型，用于合规审计",
    )

    # ---- 可验证性 ----
    evidence: Evidence = Field(
        description="支持提取结论的原文证据，使审核过程可追溯",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="本次提取的置信度。≥0.8 可直接采用；0.5-0.8 建议抽检；<0.5 强制人工复核",
    )
    needs_manual_review: bool = Field(
        description="是否需要人工复核。当置信度低或涉及高风险标签时标记为 True",
    )
    manual_review_reason: Optional[str] = Field(
        default=None,
        description="需要人工复核的具体原因，如「置信度过低」「风险标签命中」",
    )


# ============================================================================
# 批量提取结果
# ============================================================================


class BatchExtractionResult(BaseModel):
    """
    批量提取结果 —— 用于一次处理多条对话时的汇总输出。

    设计理由：周报场景需要一次性处理 25-100+ 条对话，批量结果
    模型包含了整体统计信息，方便客服主管快速了解数据全貌。
    """

    total_count: int = Field(description="对话总数")
    success_count: int = Field(description="成功提取的条数")
    failed_count: int = Field(description="提取失败的条数")
    results: list[ExtractionResult] = Field(
        default_factory=list,
        description="成功提取的结果列表",
    )
    failed_ids: list[str] = Field(
        default_factory=list,
        description="提取失败的对话 ID 列表",
    )
    summary: str = Field(
        default="",
        description="批量提取的整体摘要，≤200 字，可直接用于周报开头",
    )
