"""测试 Pydantic Schema 的合法性校验"""

import pytest
from pydantic import ValidationError

from src.extractor.schema import (
    Channel,
    Evidence,
    ExtractionResult,
    IssueCategory,
    PrimaryIntent,
    ResolutionStatus,
    Sentiment,
    UrgencyLevel,
)


# ---- 辅助函数：构造一条最小合法结果 ----
def _make_minimal_result(**overrides) -> ExtractionResult:
    kwargs = dict(
        conversation_id="conv_test_01",
        channel=Channel.ONLINE,
        agent="小王",
        turn_count=5,
        primary_intent=PrimaryIntent.REFUND_RETURN,
        issue_category=IssueCategory.PRODUCT_QUALITY,
        issue_summary="蓝牙耳机左耳无声，用户要求退款",
        resolution_status=ResolutionStatus.RESOLVED,
        resolution_action="发起退款申请",
        is_multi_issue=False,
        sub_issues=[],
        user_sentiment_initial=Sentiment.NEUTRAL,
        user_sentiment_final=Sentiment.NEUTRAL,
        urgency_level=UrgencyLevel.HIGH,
        complaint_related=False,
        human_transfer=False,
        compensation_offered=False,
        pii_detected=False,
        pii_types=[],
        evidence=Evidence(),
        confidence=0.95,
        needs_manual_review=False,
    )
    kwargs.update(overrides)
    return ExtractionResult(**kwargs)


class TestExtractionResult:
    """测试 ExtractionResult 的构造与序列化"""

    def test_valid_result_passes_validation(self):
        """构造一条合法的 ExtractionResult，应通过 Pydantic 校验"""
        r = _make_minimal_result()

        # 序列化为 JSON 再反序列化，验证往返正确
        json_str = r.model_dump_json()
        r2 = ExtractionResult.model_validate_json(json_str)

        assert r2.conversation_id == "conv_test_01"
        assert r2.primary_intent == PrimaryIntent.REFUND_RETURN
        assert r2.resolution_status == ResolutionStatus.RESOLVED
        assert r2.confidence == 0.95

    def test_confidence_above_one_raises(self):
        """confidence > 1.0 在构造时抛出 ValidationError"""
        with pytest.raises(ValidationError):
            _make_minimal_result(confidence=1.5)

    def test_confidence_below_zero_raises(self):
        """confidence < 0.0 在构造时抛出 ValidationError"""
        with pytest.raises(ValidationError):
            _make_minimal_result(confidence=-0.2)

    def test_risk_flags_can_be_empty(self):
        """risk_flags 允许为空列表"""
        r = _make_minimal_result()
        assert len(r.risk_flags) == 0

    def test_evidence_with_explicit_reasoning(self):
        """Evidence reasoning 字段可独立设置"""
        ev = Evidence(
            user_quotes=["测试用户语句"],
            agent_quotes=["测试客服语句"],
            reasoning="基于关键词匹配推断用户意图",
        )
        assert len(ev.user_quotes) == 1
        assert len(ev.agent_quotes) == 1
        assert "关键词匹配" in ev.reasoning

    def test_resolution_status_values(self):
        """验证 ResolutionStatus 枚举值符合预期"""
        assert ResolutionStatus.RESOLVED.value == "已解决"
        assert ResolutionStatus.UNRESOLVED.value == "未解决"
        assert ResolutionStatus.PENDING_FOLLOWUP.value == "待跟进"

    def test_sentiment_count(self):
        """验证 Sentiment 枚举包含 5 个值"""
        values = [e.value for e in Sentiment]
        assert len(values) == 5
        assert "positive" in values
        assert "angry" in values

    def test_multiple_risk_flags(self):
        """一条对话可同时打多个风险标签"""
        from src.extractor.schema import RiskFlag
        r = _make_minimal_result(
            risk_flags=[RiskFlag.CHURN_RISK, RiskFlag.BATCH_QUALITY],
        )
        assert len(r.risk_flags) == 2
        assert RiskFlag.CHURN_RISK in r.risk_flags

    def test_json_roundtrip_preserves_all_fields(self):
        """JSON 序列化往返后所有字段保持一致"""
        r = _make_minimal_result(
            products_mentioned=["蓝牙耳机"],
            order_ids=["DD20240301-0001"],
            pii_detected=True,
            pii_types=[],
            evidence=Evidence(
                user_quotes=["能退款吗？"],
                agent_quotes=["我帮您发起退款"],
                reasoning="用户请求退款，客服协助处理",
            ),
        )
        r2 = ExtractionResult.model_validate_json(r.model_dump_json())
        assert r2.conversation_id == r.conversation_id
        assert r2.products_mentioned == r.products_mentioned
        assert r2.order_ids == r.order_ids
        assert r2.evidence.user_quotes == r.evidence.user_quotes
        assert r2.evidence.agent_quotes == r.evidence.agent_quotes
