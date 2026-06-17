"""测试 MockExtractor 在 25 条真实对话上的抽取质量"""

import json
from pathlib import Path

import pytest

from src.extractor.mock_extractor import MockExtractor
from src.extractor.schema import ResolutionStatus, RiskFlag

# 数据文件路径
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CONVERSATIONS_PATH = _DATA_DIR / "task2_conversations.json"


@pytest.fixture(scope="module")
def conversations() -> list[dict]:
    """加载 25 条测试对话"""
    with open(_CONVERSATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def batch_result(conversations: list[dict]):
    """运行 MockExtractor 批量提取，供所有测试复用"""
    extractor = MockExtractor()
    return extractor.extract_batch(conversations)


class TestBatchExtraction:
    """批量提取的正确性测试"""

    def test_input_has_25_conversations(self, conversations):
        """输入数据共 25 条对话"""
        assert len(conversations) == 25

    def test_success_count_is_25(self, batch_result):
        """25 条全部成功提取"""
        assert batch_result.success_count == 25

    def test_failed_count_is_zero(self, batch_result):
        """无失败记录"""
        assert batch_result.failed_count == 0

    def test_results_length_is_25(self, batch_result):
        """结果列表长度为 25"""
        assert len(batch_result.results) == 25

    def test_all_confidence_in_range(self, batch_result):
        """所有结果的 confidence 在 [0, 1] 之间"""
        for r in batch_result.results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"{r.conversation_id}: confidence={r.confidence}"
            )

    def test_all_have_evidence(self, batch_result):
        """所有结果的 evidence 至少包含用户或客服原文"""
        for r in batch_result.results:
            has_evidence = bool(r.evidence.user_quotes or r.evidence.agent_quotes)
            assert has_evidence, (
                f"{r.conversation_id}: evidence is empty"
            )

    def test_all_have_non_empty_issue_summary(self, batch_result):
        """所有结果的 issue_summary 非空"""
        for r in batch_result.results:
            assert r.issue_summary.strip(), (
                f"{r.conversation_id}: issue_summary is empty"
            )

    def test_all_have_non_empty_resolution_action(self, batch_result):
        """所有结果的 resolution_action 非空"""
        for r in batch_result.results:
            assert r.resolution_action.strip(), (
                f"{r.conversation_id}: resolution_action is empty"
            )


class TestSpecificConversations:
    """对关键对话的字段级验证"""

    def _find(self, batch_result, conv_id: str):
        for r in batch_result.results:
            if r.conversation_id == conv_id:
                return r
        raise AssertionError(f"未找到对话 {conv_id}")

    def test_conv_06_is_multi_issue(self, batch_result):
        """conv_06 多诉求，sub_issues ≥ 2"""
        r = self._find(batch_result, "conv_06")
        assert r.is_multi_issue is True
        assert len(r.sub_issues) >= 2

    def test_conv_16_human_transfer_and_review(self, batch_result):
        """conv_16 有人工转接，需要复核"""
        r = self._find(batch_result, "conv_16")
        assert r.human_transfer is True
        assert r.needs_manual_review is True

    def test_conv_20_batch_quality_or_churn_risk(self, batch_result):
        """conv_20 包含批量质量问题或用户流失风险"""
        r = self._find(batch_result, "conv_20")
        has_target_risk = (
            RiskFlag.BATCH_QUALITY in r.risk_flags
            or RiskFlag.CHURN_RISK in r.risk_flags
        )
        assert has_target_risk, (
            f"conv_20 risk_flags={[rf.value for rf in r.risk_flags]}"
        )

    def test_conv_25_unresolved_and_churn(self, batch_result):
        """conv_25 resolution_status=未解决，包含用户流失风险"""
        r = self._find(batch_result, "conv_25")
        assert r.resolution_status == ResolutionStatus.UNRESOLVED
        assert RiskFlag.CHURN_RISK in r.risk_flags
