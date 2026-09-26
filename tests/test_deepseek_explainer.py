from __future__ import annotations

import json

from src.analysis.deepseek_explainer import (
    build_explanation_prompts,
    explain_market_assessment,
)
from src.analysis.market_assessment import build_market_assessment
from src.clients.deepseek import DeepSeekJsonResponse, DeepSeekTransportError
from tests.test_evidence import _complete_bundle


class FakeProvider:
    def __init__(self, data: dict | None = None, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.system_prompt = ""
        self.user_prompt = ""

    def create_json(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 2500):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.error:
            raise self.error
        return DeepSeekJsonResponse(
            data=self.data or {},
            request_id="chat-test",
            model="deepseek-flash",
            usage={},
        )


class SequenceProvider:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = 0

    def create_json(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 2500):
        data = self.responses[self.calls]
        self.calls += 1
        return DeepSeekJsonResponse(
            data=data,
            request_id=f"chat-{self.calls}",
            model="deepseek-flash",
            usage={},
        )


def _valid_response(state_label: str) -> dict:
    return {
        "state_label": state_label,
        "overview": "趋势与市场宽度共同偏弱[E-TREND-I01][E-BREADTH-I01]。",
        "primary_conflict": "短线情绪与主要方向并不完全一致[E-SENTIMENT-I01]。",
        "confidence": "样本覆盖及时点差异降低了证据质量[E-BREADTH-06]。",
        "uncertainty": "估值缺少历史分位，不能判断高低[E-VALUATION-U01]。",
        "switch_conditions": [
            {
                "condition_id": "SC-01",
                "explanation": "趋势和宽度共同改变后，综合状态才可能改变[E-TREND-01][E-BREADTH-04]。",
            },
            {
                "condition_id": "SC-02",
                "explanation": "成分股情绪变化只能修正短线描述[E-SENTIMENT-05]。",
            },
        ],
    }


def test_valid_grounded_explanation_is_accepted() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    data = _valid_response(assessment.state_label)
    data["overview"] = (
        f"{assessment.subject_name}趋势与市场宽度共同偏弱"
        "[E-TREND-I01][E-BREADTH-I01]。"
    )
    provider = FakeProvider(data)

    result = explain_market_assessment(assessment, bundle, provider)

    assert result.source == "deepseek"
    assert result.request_id == "chat-test"
    assert set(result.cited_evidence_ids) <= {item.id for item in bundle.evidence}
    assert "不得修改或重新计算" in provider.system_prompt
    assert assessment.to_dict()["confidence"] == assessment.to_dict()["confidence"]


def test_prompt_supplies_confidence_evidence_and_field_citation_contract() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    _, user_prompt = build_explanation_prompts(assessment, bundle)
    payload = json.loads(user_prompt.split("：\n", 1)[1])
    supplied_ids = {item["id"] for item in payload["referenced_evidence"]}
    confidence_ids = set(payload["citation_contract"]["confidence"])

    assert {"E-BREADTH-05", "E-BREADTH-06"} <= supplied_ids
    assert {"E-VALUATION-03", "E-VALUATION-04", "E-VALUATION-05"} <= supplied_ids
    assert {"E-BREADTH-06", "E-VALUATION-05"} <= confidence_ids
    assert set(payload["citation_contract"]["switch_condition:SC-02"]) == {
        "E-SENTIMENT-05",
        "E-SENTIMENT-06",
        "E-SENTIMENT-07",
    }


def test_field_rejects_existing_but_semantically_wrong_evidence_id() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    data = _valid_response(assessment.state_label)
    data["confidence"] = "置信度由样本覆盖决定[E-VALUATION-U01]。"

    result = explain_market_assessment(assessment, bundle, FakeProvider(data))

    assert result.source == "local_fallback"
    assert "不属于该字段" in result.fallback_reason


def test_unknown_evidence_reference_triggers_local_fallback() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    data = _valid_response(assessment.state_label)
    data["overview"] = "出现未知事实[E-FAKE-99]。"

    result = explain_market_assessment(assessment, bundle, FakeProvider(data))

    assert result.source == "local_fallback"
    assert "不存在的证据编号" in result.fallback_reason
    assert result.request_id == "chat-test"
    assert [item.status for item in result.attempts] == ["rejected", "rejected"]


def test_changed_state_or_investment_advice_triggers_fallback() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    changed = _valid_response("综合偏强")
    changed_result = explain_market_assessment(
        assessment, bundle, FakeProvider(changed)
    )
    advice = _valid_response(assessment.state_label)
    advice["overview"] = "因此可以加仓[E-TREND-I01]。"
    advice_result = explain_market_assessment(
        assessment, bundle, FakeProvider(advice)
    )

    assert changed_result.source == "local_fallback"
    assert "修改了程序计算" in changed_result.fallback_reason
    assert advice_result.source == "local_fallback"
    assert "投资建议" in advice_result.fallback_reason


def test_new_number_outside_known_subject_name_triggers_fallback() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    data = _valid_response(assessment.state_label)
    data["overview"] = "未来可能上涨百分之99[E-TREND-I01]。"

    result = explain_market_assessment(assessment, bundle, FakeProvider(data))

    assert result.source == "local_fallback"
    assert "数字" in result.fallback_reason


def test_transport_failure_returns_deterministic_local_explanation() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    provider = FakeProvider(error=DeepSeekTransportError("接口超时"))

    result = explain_market_assessment(assessment, bundle, provider)

    assert result.source == "local_fallback"
    assert result.state_label == assessment.state_label
    assert result.overview == assessment.conclusion
    assert "接口超时" in result.fallback_reason
    assert result.attempts[0].status == "transport_error"


def test_missing_citation_is_repaired_once_and_keeps_audit_trail() -> None:
    bundle = _complete_bundle()
    assessment = build_market_assessment(bundle)
    invalid = _valid_response(assessment.state_label)
    invalid["switch_conditions"][0]["explanation"] = "趋势和宽度共同改变。"
    provider = SequenceProvider(
        [invalid, _valid_response(assessment.state_label)]
    )

    result = explain_market_assessment(assessment, bundle, provider)

    assert result.source == "deepseek"
    assert provider.calls == 2
    assert result.request_id == "chat-2"
    assert [item.status for item in result.attempts] == ["rejected", "accepted"]
    assert result.attempts[0].request_id == "chat-1"
    assert "没有引用证据" in result.attempts[0].detail
