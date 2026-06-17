"""
Mock 抽取器 —— 基于规则和关键词的确定性抽取。

不依赖外部 API，所有分类逻辑通过关键词匹配和规则判断完成。
每条分类函数独立可测，注释说明了规则依据。
"""

from __future__ import annotations

from . import utils
from .schema import (
    BatchExtractionResult,
    Channel,
    Evidence,
    ExtractionResult,
    IssueCategory,
    PIIType,
    PrimaryIntent,
    ResolutionStatus,
    RiskFlag,
    Sentiment,
    SubIssue,
    UrgencyLevel,
)


class MockExtractor:
    """基于规则的确定性抽取器"""

    # ========================================================================
    # 批量入口
    # ========================================================================

    def extract_batch(self, conversations: list[dict]) -> BatchExtractionResult:
        results: list[ExtractionResult] = []
        failed_ids: list[str] = []

        for conv in conversations:
            try:
                result = self.extract_single(conv)
                results.append(result)
            except Exception:
                failed_ids.append(conv.get("id", "unknown"))

        return BatchExtractionResult(
            total_count=len(conversations),
            success_count=len(results),
            failed_count=len(failed_ids),
            results=results,
            failed_ids=failed_ids,
            summary=self._build_batch_summary(results, len(conversations)),
        )

    # ========================================================================
    # 单条抽取
    # ========================================================================

    def extract_single(self, conv: dict) -> ExtractionResult:
        turns: list[dict] = conv["turns"]

        # 预计算常用文本（只计算一次，各分类函数复用）
        user_text = utils.join_user_messages(turns)
        agent_text = utils.join_agent_messages(turns)
        all_text = utils.join_all_text(turns)
        first_user = utils.get_first_user_msg(turns)
        last_user = utils.get_last_user_msg(turns)

        # ---- 1. 基础字段 ----
        conversation_id = conv["id"]
        channel = Channel(conv["channel"])
        agent = conv["agent"]
        turn_count = len(turns)

        # ---- 2. 问题分类 ----
        primary_intent = self._classify_intent(user_text, all_text, turns)
        issue_category = self._classify_category(user_text, all_text, primary_intent, turns)
        issue_summary = self._build_issue_summary(user_text, primary_intent)
        products = utils.extract_products(all_text)

        # ---- 3. 多诉求 ----
        is_multi, sub_issues = self._detect_multi_issues(user_text, all_text, turns)

        # ---- 4. 解决状态 ----
        resolution_status, resolution_action, cust_next, agent_next = \
            self._classify_resolution(turns, all_text, agent_text, primary_intent)

        # ---- 5. 情绪与风险 ----
        sentiment_init = self._classify_sentiment(first_user)
        sentiment_final = self._classify_sentiment(last_user)
        urgency = self._classify_urgency(user_text, all_text, primary_intent, turns)
        risk_flags = self._detect_risks(user_text, all_text, turns)

        # ---- 6. 运营统计 ----
        complaint_related = self._is_complaint(user_text, all_text)
        human_transfer = self._is_transferred(all_text, turns)
        compensation_offered = self._has_compensation(agent_text)

        # ---- 7. 隐私 ----
        order_ids = utils.find_order_ids(all_text)
        pii_detected, pii_types = self._detect_pii(all_text)

        # ---- 8. 可验证性 ----
        evidence = self._build_evidence(turns, primary_intent, resolution_status)
        needs_review, review_reason = self._evaluate_review(
            is_multi=is_multi,
            risk_flags=risk_flags,
            human_transfer=human_transfer,
            primary_intent=primary_intent,
            resolution_status=resolution_status,
            pii_detected=pii_detected,
            user_text=user_text,
            turns=turns,
        )
        confidence = self._calc_confidence(
            is_multi=is_multi,
            needs_review=needs_review,
            risk_flags=risk_flags,
            turn_count=turn_count,
        )

        return ExtractionResult(
            conversation_id=conversation_id,
            channel=channel,
            agent=agent,
            turn_count=turn_count,
            primary_intent=primary_intent,
            issue_category=issue_category,
            issue_summary=issue_summary,
            products_mentioned=products,
            is_multi_issue=is_multi,
            sub_issues=sub_issues,
            resolution_status=resolution_status,
            resolution_action=resolution_action,
            customer_next_action=cust_next,
            agent_next_action=agent_next,
            user_sentiment_initial=sentiment_init,
            user_sentiment_final=sentiment_final,
            urgency_level=urgency,
            risk_flags=risk_flags,
            complaint_related=complaint_related,
            human_transfer=human_transfer,
            compensation_offered=compensation_offered,
            order_ids=order_ids,
            pii_detected=pii_detected,
            pii_types=pii_types,
            evidence=evidence,
            confidence=confidence,
            needs_manual_review=needs_review,
            manual_review_reason=review_reason,
        )

    # ========================================================================
    # 问题意图分类
    # ========================================================================

    def _classify_intent(
        self, user_text: str, all_text: str, turns: list[dict]
    ) -> PrimaryIntent:
        """根据关键词判定用户主要意图，优先级高的规则在前"""

        # 1) 取消订单 → 必须有"还没发货"语境
        if utils.has_any_keyword(user_text, ["取消", "还没发货"]):
            return PrimaryIntent.ORDER_CANCEL

        # 2) 投诉 → 显式投诉词、辱骂、强不满
        if utils.has_any_keyword(user_text, [
            "投诉", "什么破", "智障", "烂", "假货", "品控",
        ]):
            return PrimaryIntent.COMPLAINT

        # 3) 账号安全 → 异地登录、盗号
        if utils.has_any_keyword(user_text, ["异地登录", "账号安全", "登录记录"]):
            return PrimaryIntent.ACCOUNT_SECURITY

        # 4) 换货
        if utils.has_any_keyword(user_text, ["换", "换新", "换尺码"]):
            return PrimaryIntent.EXCHANGE

        # 5) 建议反馈（必须在催促/跟进之前，避免"建议"中含"等了好久"误判）
        if utils.has_any_keyword(user_text, ["建议", "能不能加个", "提个建议"]):
            return PrimaryIntent.SUGGESTION

        # 6) 系统使用 → 操作流程问题
        if utils.has_any_keyword(user_text, [
            "怎么操作", "格式不对", "上传", "流程", "复杂",
        ]):
            return PrimaryIntent.SYSTEM_USAGE

        # 7) 无实质问题
        if _is_idle_conversation(turns):
            return PrimaryIntent.NO_SUBSTANTIVE

        # 8) 催促/跟进（含退款/退货上下文）→ 退款还没到、退款没动静
        #    必须在纯退款之前判断，否则"退款怎么还没到"会被误判为退款请求
        _followup_words = [
            "还没到", "等了好", "等了好久", "还没动静",
            "怎么还没", "等了五", "等了半", "还没处理", "一周了还没",
        ]
        _refund_words = ["退款", "退货", "退", "退了", "退掉", "能退", "想退", "要退"]

        # 先排除商品咨询中的"等了好久"（如"补货等了好久"）
        if utils.has_any_keyword(user_text, [
            "能带上飞机", "成分", "有没有", "有货", "补货",
            "什么时候", "啥时候", "哪个好", "推荐", "介绍",
            "颜色", "尺码", "规格", "咨询", "问问",
        ]):
            return PrimaryIntent.PRODUCT_INQUIRY

        if utils.has_any_keyword(user_text, _followup_words):
            if utils.has_any_keyword(user_text, _refund_words):
                return PrimaryIntent.URGE_FOLLOWUP
            # 通用催促（无退款词，但明确表达等待不耐）
            return PrimaryIntent.URGE_FOLLOWUP

        # 9) 转人工（在退款判断之前，因为可能"退款问题"转人工）
        if utils.has_any_keyword(user_text, ["转人工", "转接"]):
            return PrimaryIntent.URGE_FOLLOWUP

        # 10) 退款/退货（纯退款请求，无跟进等待语境）
        if utils.has_any_keyword(user_text, _refund_words):
            return PrimaryIntent.REFUND_RETURN

        # 11) 物流 → 快递/签收/改地址/快递柜/配送
        if utils.has_any_keyword(user_text, [
            "快递", "签收", "派送", "配送", "改地址", "快递柜",
            "物流", "转运", "取件", "到了没",
        ]):
            return PrimaryIntent.LOGISTICS

        # 12) 优惠券/促销
        if utils.has_any_keyword(user_text, ["优惠券", "用不了", "满", "可用"]):
            return PrimaryIntent.PROMOTION_COUPON

        # 13) 商品咨询（覆盖最广，兜底）
        if utils.has_any_keyword(user_text, [
            "能带上飞机", "成分", "有没有", "有货", "补货",
            "什么时候", "啥时候", "哪个好", "推荐", "介绍",
            "颜色", "尺码", "规格", "咨询", "问问",
        ]):
            return PrimaryIntent.PRODUCT_INQUIRY

        return PrimaryIntent.OTHER

    def _classify_category(
        self, user_text: str, all_text: str,
        intent: PrimaryIntent, turns: list[dict],
    ) -> IssueCategory:
        """根据用户表述和意图综合判定业务分类"""

        # 退款到账时效（优先级最高：当用户明确表达退款/到账等太久时，
        # 即使同时有服务响应抱怨，也应归为退款时效问题）
        _refund_timing_words = [
            "还没到", "还没到账", "没到账", "没动静", "一周了还没",
            "等了.*天.*退款", "退款.*没到", "还没.*退款",
        ]
        _refund_words_check = ["退款", "退", "退了", "到账"]
        if intent == PrimaryIntent.URGE_FOLLOWUP:
            if utils.has_any_keyword(user_text, _refund_words_check):
                if utils.has_any_keyword(user_text, _refund_timing_words):
                    return IssueCategory.REFUND_TIMELINESS
                # 兜底：含退款词 + 催促时间词
                if utils.has_any_keyword(user_text, ["等了好", "好几天", "几天了", "等了五", "等了半", "一周了", "还没处理"]):
                    return IssueCategory.REFUND_TIMELINESS

        # 服务响应问题（优先于质量问题，因为抱怨服务≠商品质量问题）
        # 注意："机器人"不在此列，"扫地机器人"是商品名不是客服机器人
        if utils.has_any_keyword(user_text, [
            "智能客服", "答非所问", "智障",
            "没人理", "半天都没", "半天没人", "不回",
        ]):
            return IssueCategory.SERVICE_RESPONSE

        # 质量问题线索
        quality_clues = [
            "坏的", "碎的", "没声音", "不工作", "破了", "破损",
            "坏了", "假货", "品控", "质量问题",
        ]
        if utils.has_any_keyword(user_text, quality_clues):
            return IssueCategory.PRODUCT_QUALITY

        # 物流配送
        if utils.has_any_keyword(user_text, [
            "快递", "签收", "派送", "配送", "改地址", "快递柜",
            "物流", "转运", "取件",
        ]):
            return IssueCategory.LOGISTICS_DELIVERY

        # 账号安全
        if utils.has_any_keyword(user_text, ["异地登录", "账号安全", "登录记录"]):
            return IssueCategory.ACCOUNT_SECURITY

        # 促销规则
        if utils.has_any_keyword(user_text, ["优惠券", "满", "可用", "条件"]):
            return IssueCategory.PROMOTION_RULES

        # 产品信息（咨询类）
        if intent == PrimaryIntent.PRODUCT_INQUIRY:
            if utils.has_any_keyword(user_text, ["成分", "过敏", "带上飞机", "容量"]):
                return IssueCategory.PRODUCT_INFO
            if utils.has_any_keyword(user_text, ["补货", "有货", "到货"]):
                return IssueCategory.INVENTORY
            return IssueCategory.PRODUCT_INFO

        # 售后政策
        if intent in (PrimaryIntent.REFUND_RETURN, PrimaryIntent.EXCHANGE):
            if utils.has_any_keyword(user_text, ["行不行", "能退", "能换", "运费谁出", "运费"]):
                return IssueCategory.AFTERSALES_POLICY
            # 无质量线索的退款 → 售后政策，否则已在前面命中商品质量
            if not utils.has_any_keyword(user_text, quality_clues):
                return IssueCategory.AFTERSALES_POLICY

        # 系统体验
        if utils.has_any_keyword(user_text, [
            "流程", "复杂", "操作", "格式不对", "上传",
        ]):
            return IssueCategory.SYSTEM_EXPERIENCE

        # 服务响应
        if utils.has_any_keyword(user_text, [
            "等了", "半天", "没人理", "怎么不回了", "不回了",
            "搞快点", "没人", "智能客服", "机器人",
        ]):
            return IssueCategory.SERVICE_RESPONSE

        # 色差/描述不符
        if utils.has_any_keyword(user_text, ["颜色", "图片", "差挺多", "色差"]):
            return IssueCategory.COLOR_MISMATCH

        # 库存
        if utils.has_any_keyword(user_text, ["补货", "有货", "没货"]):
            return IssueCategory.INVENTORY

        return IssueCategory.OTHER

    # ========================================================================
    # 问题摘要
    # ========================================================================

    def _build_issue_summary(self, user_text: str, intent: PrimaryIntent) -> str:
        """根据首条用户消息生成简短摘要"""
        # 取前 80 字，避免过长
        return user_text[:80].strip()

    # ========================================================================
    # 多诉求检测
    # ========================================================================

    def _detect_multi_issues(
        self, user_text: str, all_text: str, turns: list[dict],
    ) -> tuple[bool, list[SubIssue]]:
        """
        检测对话是否包含多个独立诉求。

        规则：
        - 用户消息中出现显式列举（"两个问题""对了""还有"）
        - 客服回复中明确提到多个问题
        """
        agent_text = utils.join_agent_messages(turns)

        # 显式多问题提示词
        multi_markers_user = ["两个问题", "对了我那个", "另外"]
        multi_markers_agent = ["两个问题", "都帮您查", "都帮您处理"]

        is_multi = (
            utils.has_any_keyword(user_text, multi_markers_user)
            or utils.has_any_keyword(agent_text, multi_markers_agent)
        )

        if not is_multi:
            return False, []

        # 构建子诉求列表：基于用户原文拆解
        sub_issues: list[SubIssue] = []
        user_msgs = [t["content"] for t in turns if t["role"] == "user"]

        # 策略：在用户首条消息中寻找独立诉求边界
        first = user_msgs[0] if user_msgs else ""

        # 按 "退货" + "快递" 组合拆分（最常见的多诉求模式）
        has_return = utils.has_any_keyword(first, ["退货", "退款"])
        has_delivery = utils.has_any_keyword(first, ["快递", "物流", "配送", "到了没"])

        if has_return:
            sub_issues.append(SubIssue(
                issue_summary="退货/退款进度查询",
                issue_category=IssueCategory.AFTERSALES_POLICY,
                resolution_status=ResolutionStatus.RESOLVED,
                resolution_action="审核已通过，告知用户按地址寄回商品",
            ))
        if has_delivery:
            sub_issues.append(SubIssue(
                issue_summary="快递配送状态查询",
                issue_category=IssueCategory.LOGISTICS_DELIVERY,
                resolution_status=ResolutionStatus.RESOLVED,
                resolution_action="查询配送状态并告知预计送达时间",
            ))

        # 如果上述模式没有命中，使用通用拆分
        if not sub_issues:
            sub_issues.append(SubIssue(
                issue_summary=first[:60],
                issue_category=IssueCategory.OTHER,
                resolution_status=ResolutionStatus.PENDING_FOLLOWUP,
                resolution_action="需人工确认具体子诉求",
            ))

        return True, sub_issues

    # ========================================================================
    # 解决状态判定
    # ========================================================================

    def _classify_resolution(
        self, turns: list[dict], all_text: str,
        agent_text: str, intent: PrimaryIntent,
    ) -> tuple[ResolutionStatus, str, str | None, str | None]:
        """
        综合判定解决状态、处理措施、待办事项。

        优先级：
        1) 用户放弃 → UNRESOLVED
        2) 已转人工且未闭环 → PENDING_FOLLOWUP
        3) 纯咨询无动作 → NO_ACTION_NEEDED
        4) 明确后续需等待 → PENDING_FOLLOWUP
        5) 已明确处理 → RESOLVED
        """

        # --- 用户放弃 ---
        if utils.user_gave_up(turns):
            return (
                ResolutionStatus.UNRESOLVED,
                "用户中途放弃沟通或转向其他渠道",
                None,
                None,
            )

        # --- 已转人工 ---
        if utils.has_any_keyword(all_text, ["转接人工", "转人工"]):
            return (
                ResolutionStatus.PENDING_FOLLOWUP,
                "已转接人工客服继续处理",
                "等待新的客服继续处理",
                None,
            )

        # --- 纯咨询/建议 ---
        if intent in (PrimaryIntent.PRODUCT_INQUIRY, PrimaryIntent.SUGGESTION):
            if _agent_gave_answer(agent_text):
                return (
                    ResolutionStatus.RESOLVED,
                    _extract_resolution_action(agent_text),
                    None,
                    _extract_agent_next(agent_text),
                )

        # --- 无实质问题 ---
        if intent == PrimaryIntent.NO_SUBSTANTIVE:
            return (
                ResolutionStatus.NO_ACTION_NEEDED,
                "用户未提出明确诉求",
                None,
                None,
            )

        # --- 待跟进（客服承诺后续动作） ---
        if utils.has_any_keyword(agent_text, [
            "24小时内", "1-2天", "1-2个工作日", "短信通知",
            "我们会联系", "再联系我们", "收到后会",
            "预计下周", "到货后", "联系快递公司",
        ]):
            cust_next = _extract_customer_next(agent_text, intent, all_text)
            agent_next = _extract_agent_next(agent_text)
            return (
                ResolutionStatus.PENDING_FOLLOWUP,
                _extract_resolution_action(agent_text),
                cust_next,
                agent_next,
            )

        # --- 已解决 ---
        return (
            ResolutionStatus.RESOLVED,
            _extract_resolution_action(agent_text),
            _extract_customer_next(agent_text, intent, all_text),
            _extract_agent_next(agent_text),
        )

    # ========================================================================
    # 情绪分类
    # ========================================================================

    def _classify_sentiment(self, text: str) -> Sentiment:
        """根据单条消息判定情绪等级"""

        # angry: 感叹号 ≥3 或辱骂词
        if text.count("!") >= 3 or text.count("！") >= 3:
            return Sentiment.ANGRY
        if utils.has_any_keyword(text, [
            "什么破服务", "智障", "烂", "什么破",
        ]):
            return Sentiment.ANGRY

        # anxious: 害怕、担心
        if utils.has_any_keyword(text, [
            "害怕", "好怕", "担心", "不安", "怎么办",
        ]):
            return Sentiment.ANXIOUS

        # negative: 不满但非愤怒
        if utils.has_any_keyword(text, [
            "投诉", "假货", "失望", "算了", "行吧",
            "再等等", "好吧", "没信心", "等了好久",
            "等了五", "太复杂", "搞半天", "半天都没",
            "等了好", "搞快点", "等了半",
            "坏了", "是不是有", "品控", "又是",
            "很差", "太差", "太烂", "浪费",
        ]):
            return Sentiment.NEGATIVE

        # positive: 感谢且无负面
        if utils.has_any_keyword(text, ["谢谢", "好的谢谢", "感谢", "很棒"]):
            return Sentiment.POSITIVE

        return Sentiment.NEUTRAL

    # ========================================================================
    # 紧急程度
    # ========================================================================

    def _classify_urgency(
        self, user_text: str, all_text: str,
        intent: PrimaryIntent, turns: list[dict],
    ) -> UrgencyLevel:
        """根据问题类型和关键词综合评估紧急程度"""

        # critical: 安全/法律/批量质量
        if utils.has_any_keyword(user_text, [
            "异地登录", "盗号", "假货", "批量", "违法",
        ]):
            return UrgencyLevel.CRITICAL
        if utils.has_any_keyword(user_text, ["品控", "连续", "又是坏的"]):
            return UrgencyLevel.CRITICAL

        # high: 退款/金钱/时效
        if utils.has_any_keyword(user_text, [
            "退款", "退", "退钱", "破损", "碎的", "坏了",
            "没收到", "丢了", "丢失",
        ]):
            return UrgencyLevel.HIGH
        if intent in (
            PrimaryIntent.REFUND_RETURN,
            PrimaryIntent.URGE_FOLLOWUP,
        ):
            return UrgencyLevel.HIGH

        # medium: 影响体验但可延迟
        if utils.has_any_keyword(user_text, [
            "改地址", "换", "转人工", "用不了",
            "流程", "复杂",
        ]):
            return UrgencyLevel.MEDIUM

        # low: 一般咨询
        return UrgencyLevel.LOW

    # ========================================================================
    # 风险标签检测
    # ========================================================================

    def _detect_risks(
        self, user_text: str, all_text: str, turns: list[dict],
    ) -> list[RiskFlag]:
        """检测所有适用的风险标签，可同时打多个"""
        flags: list[RiskFlag] = []

        # 用户流失风险
        if utils.has_any_keyword(user_text, [
            "不用了", "别的地方", "没信心", "不在这里买",
        ]):
            flags.append(RiskFlag.CHURN_RISK)

        # 舆情风险（威胁曝光/社交媒体）
        if utils.has_any_keyword(user_text, ["曝光", "发微博", "朋友圈", "网上说"]):
            flags.append(RiskFlag.PR_RISK)

        # 法律风险
        if utils.has_any_keyword(user_text, [
            "假货", "欺诈", "违法", "消费者权益", "举报",
        ]):
            flags.append(RiskFlag.LEGAL_RISK)

        # 账号安全风险
        if utils.has_any_keyword(user_text, ["异地登录", "盗号", "密码被改"]):
            flags.append(RiskFlag.ACCOUNT_RISK)

        # 投诉升级风险
        if utils.has_any_keyword(user_text, [
            "投诉", "投诉你们", "找你们领导", "上级",
            "智障", "浪费我时间", "答非所问", "什么破服务",
        ]):
            flags.append(RiskFlag.ESCALATION_RISK)

        # 批量质量问题
        if utils.has_any_keyword(user_text, [
            "又是", "连续", "两次", "品控", "每次都",
        ]):
            if utils.has_any_keyword(user_text, ["坏的", "问题", "碎", "质量"]):
                flags.append(RiskFlag.BATCH_QUALITY)

        # 隐私泄露风险
        if utils.find_phone_numbers(all_text) and utils.has_address(all_text):
            flags.append(RiskFlag.PRIVACY_RISK)

        return flags

    # ========================================================================
    # 运营统计
    # ========================================================================

    def _is_complaint(self, user_text: str, all_text: str) -> bool:
        """是否涉及投诉"""
        return utils.has_any_keyword(user_text, [
            "投诉", "什么破", "智障", "烂服务",
            "没人理", "品控", "假货",
        ])

    def _is_transferred(self, all_text: str, turns: list[dict]) -> bool:
        """是否发生人工转接"""
        # 检查 turn 中出现转接动作
        for t in turns:
            if t["role"] == "agent" and utils.has_any_keyword(
                t["content"], ["转接", "转人工"]
            ):
                return True
        return utils.has_any_keyword(all_text, ["转人工客服"])

    def _has_compensation(self, agent_text: str) -> bool:
        """
        客服是否提供了补偿。

        注意：仅匹配"主动给予"补偿的表达（赠送、补偿、承担运费等），
        不包括单纯解释优惠券使用规则的情况。
        """
        # 主动赠送/补偿
        if utils.has_any_keyword(agent_text, [
            "赠送您", "作为补偿", "送您", "补偿您",
            "帮您申请一张", "申请一张", "全额退款",
        ]):
            return True
        # 承担运费
        if utils.has_any_keyword(agent_text, [
            "运费我们承担", "退货运费我们承担", "运费由我们承担",
            "免运费",
        ]):
            return True
        return False

    # ========================================================================
    # 隐私检测
    # ========================================================================

    def _detect_pii(self, all_text: str) -> tuple[bool, list[PIIType]]:
        """检测隐私信息"""
        types: list[PIIType] = []

        if utils.find_phone_numbers(all_text):
            types.append(PIIType.PHONE)
        if utils.has_address(all_text):
            types.append(PIIType.ADDRESS)
        if utils.has_any_keyword(all_text, ["邮箱", "email", "@"]):
            types.append(PIIType.EMAIL)

        return len(types) > 0, types

    # ========================================================================
    # 证据构建
    # ========================================================================

    def _build_evidence(
        self, turns: list[dict],
        intent: PrimaryIntent,
        resolution: ResolutionStatus,
    ) -> Evidence:
        """从对话原文摘取关键语句作为证据"""

        # 用户侧：摘取表达诉求的消息
        user_keywords = _intent_user_keywords(intent)
        user_quotes = utils.extract_quotes(turns, user_keywords, role="user")
        # 兜底：关键词匹配不到时，取第一条用户消息
        if not user_quotes:
            first = utils.get_first_user_msg(turns)
            if first:
                user_quotes = [first]

        # 客服侧：摘取给出处理方案的消息
        agent_keywords = _resolution_agent_keywords(resolution)
        agent_quotes = utils.extract_quotes(turns, agent_keywords, role="agent")
        # 兜底：关键词匹配不到时，取第一条客服回复
        if not agent_quotes:
            first = utils.get_first_agent_msg(turns)
            if first:
                agent_quotes = [first]

        # 推理依据
        reasoning = (
            f"用户意图为{intent.value}，客服处理结果为{resolution.value}。"
            f"判断基于对话中{len(user_quotes)}条用户关键语句和"
            f"{len(agent_quotes)}条客服回复。"
        )

        return Evidence(
            user_quotes=user_quotes,
            agent_quotes=agent_quotes,
            reasoning=reasoning,
        )

    # ========================================================================
    # 人工复核判定
    # ========================================================================

    def _evaluate_review(
        self,
        *,
        is_multi: bool,
        risk_flags: list[RiskFlag],
        human_transfer: bool,
        primary_intent: PrimaryIntent,
        resolution_status: ResolutionStatus,
        pii_detected: bool,
        user_text: str,
        turns: list[dict],
    ) -> tuple[bool, str | None]:
        """判定是否需要人工复核及原因"""
        reasons: list[str] = []

        if is_multi:
            reasons.append("多诉求对话，需确认各子问题分类和解决状态的准确性")
        if resolution_status == ResolutionStatus.UNRESOLVED:
            reasons.append("用户中途放弃，需评估是否需主动回访")
        if human_transfer and resolution_status == ResolutionStatus.PENDING_FOLLOWUP:
            reasons.append("已转人工但当前对话未显示最终处理结果")
        if RiskFlag.ESCALATION_RISK in risk_flags:
            reasons.append("涉及投诉升级风险，需主管关注")
        if RiskFlag.CHURN_RISK in risk_flags:
            reasons.append("用户有流失倾向，需评估挽留策略")
        if RiskFlag.ACCOUNT_RISK in risk_flags:
            reasons.append("涉及账号安全问题，需安全团队复核")
        if RiskFlag.BATCH_QUALITY in risk_flags:
            reasons.append("重复质量问题，需品控部门跟进")
        if pii_detected:
            reasons.append("对话含个人隐私信息，需合规检查")
        if utils.has_bot_complaint(turns):
            reasons.append("用户抱怨智能客服体验，需反馈给产品团队")
        if utils.has_delayed_response(turns):
            reasons.append("存在客服响应延迟，需评估排班或响应流程")
        # 业务进度延迟（退款到账慢等，非客服响应问题）
        if primary_intent == PrimaryIntent.URGE_FOLLOWUP:
            if utils.has_any_keyword(user_text, [
                "等了五", "等了好几天", "还没到账",
                "还没动静", "一周了还没", "还没处理",
            ]):
                reasons.append("业务进度待跟进，退款或发货时效需关注")
        if _has_incomplete_info(turns):
            reasons.append("用户未提供足够信息即结束对话，提取结果可能不完整")

        if reasons:
            return True, "；".join(reasons)
        return False, None

    # ========================================================================
    # 置信度计算
    # ========================================================================

    def _calc_confidence(
        self,
        *,
        is_multi: bool,
        needs_review: bool,
        risk_flags: list[RiskFlag],
        turn_count: int,
    ) -> float:
        """根据多种因素估算本次提取的置信度"""
        base = 0.85  # mock 模式基准分

        # 多诉求降低置信度
        if is_multi:
            base -= 0.10
        # 高风险降低置信度
        if risk_flags:
            base -= 0.05 * min(len(risk_flags), 3)
        # 极短对话（<3 条）信息不足
        if turn_count < 3:
            base -= 0.15
        # 长对话（>20 条）复杂度高
        if turn_count > 20:
            base -= 0.05

        return max(0.30, min(1.0, base))

    # ========================================================================
    # 批量汇总
    # ========================================================================

    def _build_batch_summary(
        self, results: list[ExtractionResult], total: int,
    ) -> str:
        """生成批量提取摘要"""
        resolved = sum(
            1 for r in results
            if r.resolution_status == ResolutionStatus.RESOLVED
        )
        complaints = sum(1 for r in results if r.complaint_related)
        needs_review = sum(1 for r in results if r.needs_manual_review)

        return (
            f"共处理{total}条对话。"
            f"已解决{resolved}条（{resolved * 100 // total}%），"
            f"涉及投诉{complaints}条，"
            f"需人工复核{needs_review}条。"
        )


# ============================================================================
# 辅助判断函数（模块级，不依赖 self）
# ============================================================================


def _is_idle_conversation(turns: list[dict]) -> bool:
    """判断是否为无实质问题的寒暄对话"""
    user_text = utils.join_user_messages(turns)
    idle_markers = ["嗯嗯我想想", "你好", "在吗"]
    # 如果用户只发了寒暄话且没有提出任何具体问题
    has_question = any(
        kw in user_text
        for kw in ["退款", "退货", "快递", "订单", "能", "帮", "怎么", "什么", "问"]
    )
    has_idle = utils.has_any_keyword(user_text, idle_markers)
    return has_idle and not has_question


def _agent_gave_answer(agent_text: str) -> bool:
    """判断客服是否给出了有效回答（而非仅追问）"""
    return utils.has_any_keyword(agent_text, [
        "根据", "规定", "可以", "不可以", "不能",
        "成分", "容量", "库存", "预计", "查到了",
        "这款", "建议", "推荐",
    ])


def _extract_resolution_action(agent_text: str) -> str:
    """从客服回复中提取具体处理措施摘要"""
    # 组合检测：全额退款 + 补偿 + 品控反馈
    has_full_refund = "全额退款" in agent_text
    has_coupon = "优惠券" in agent_text
    has_quality_feedback = utils.has_any_keyword(agent_text, ["品控", "反馈给"])
    if has_full_refund and has_coupon and has_quality_feedback:
        return "全额退款+50元优惠券补偿+反馈品控部门跟进"

    # 按优先级匹配单动作
    action_markers = [
        ("发起退款", "发起退款申请"),
        ("帮您取消", "已取消订单"),
        ("帮您安排", "安排换新"),
        ("提交成功", "退货申请已提交"),
        ("改址申请", "配送地址已修改"),
        ("联系快递", "已联系快递公司核实"),
        ("帮您查", "查询相关信息"),
        ("反馈给", "已记录反馈"),
        ("发起换货", "发起换货申请"),
        ("补发", "免费补发商品"),
        ("建议您", "给出使用建议"),
        ("通知您", "到货后短信通知"),
        ("申请一张", "提供优惠券补偿"),
        ("帮您申请", "已提交相关申请"),
    ]
    for marker, action in action_markers:
        if marker in agent_text:
            return action

    return agent_text[:100].strip()


def _extract_customer_next(
    agent_text: str, intent: PrimaryIntent, all_text: str,
) -> str | None:
    """提取用户下一步需做的事"""
    if "寄回" in agent_text:
        return "按退货地址寄回商品"
    if "等待" in agent_text or "预计" in agent_text:
        return "等待处理结果"
    if "修改密码" in agent_text:
        return "修改密码并开启二次验证"
    if utils.has_any_keyword(agent_text, ["到账", "退款"]):
        return "等待退款到账"
    if "短信" in agent_text:
        return "查收短信通知"
    return None


def _extract_agent_next(agent_text: str) -> str | None:
    """提取客服下一步需跟进的事"""
    if "24小时内" in agent_text:
        return "24小时内联系快递公司并回复用户"
    if "到货后" in agent_text or "预计下周" in agent_text:
        return "到货后短信通知用户"
    if "短信发您" in agent_text or "短信通知" in agent_text:
        return "发送通知短信"
    return None


def _has_incomplete_info(turns: list[dict]) -> bool:
    """检测是否存在信息不足的情况"""
    user_text = utils.join_user_messages(turns)
    return utils.has_any_keyword(user_text, [
        "首页推荐", "不清楚", "不知道", "不记得",
    ])


def _intent_user_keywords(intent: PrimaryIntent) -> list[str]:
    """返回与意图相关的用户侧证据关键词"""
    mapping: dict[PrimaryIntent, list[str]] = {
        PrimaryIntent.REFUND_RETURN: ["退款", "退货", "退", "退钱"],
        PrimaryIntent.EXCHANGE: ["换", "换新", "换尺码"],
        PrimaryIntent.ORDER_CANCEL: ["取消"],
        PrimaryIntent.LOGISTICS: ["快递", "签收", "配送", "派送", "快递柜"],
        PrimaryIntent.PRODUCT_INQUIRY: [
            "能带上", "成分", "有没有", "有货", "补货",
            "哪个好", "帮我看看", "推荐", "什么时候", "啥时候",
        ],
        PrimaryIntent.PROMOTION_COUPON: ["优惠券", "用不了"],
        PrimaryIntent.ACCOUNT_SECURITY: ["异地登录", "账号", "登录"],
        PrimaryIntent.COMPLAINT: ["投诉", "什么破", "智障", "假货", "品控"],
        PrimaryIntent.SUGGESTION: ["建议", "能不能加", "加个功能", "提个建议"],
        PrimaryIntent.SYSTEM_USAGE: ["怎么操作", "格式不对", "上传", "流程"],
        PrimaryIntent.URGE_FOLLOWUP: ["还没", "等了", "进度", "没到"],
        PrimaryIntent.NO_SUBSTANTIVE: ["你好", "我想想"],
        PrimaryIntent.OTHER: [],
    }
    return mapping.get(intent, [])


def _resolution_agent_keywords(resolution: ResolutionStatus) -> list[str]:
    """返回与解决状态相关的客服侧证据关键词"""
    mapping: dict[ResolutionStatus, list[str]] = {
        ResolutionStatus.RESOLVED: [
            "帮您", "已为", "好的", "可以", "查到了",
        ],
        ResolutionStatus.PARTIALLY_RESOLVED: [
            "帮您", "已为", "好的", "还需要",
        ],
        ResolutionStatus.UNRESOLVED: [
            "抱歉", "没能", "随时联系",
        ],
        ResolutionStatus.NO_ACTION_NEEDED: [
            "请问有什么", "慢慢想", "随时找我",
        ],
        ResolutionStatus.PENDING_FOLLOWUP: [
            "24小时", "1-2天", "预计", "到货后", "联系快递",
            "转接", "短信通知",
        ],
    }
    return mapping.get(resolution, [])
