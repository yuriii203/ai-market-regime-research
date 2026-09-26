from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from src.analysis.evidence import EvidenceDirection, EvidenceType
from src.analysis.market_assessment import (
    build_market_assessment,
    calculate_confidence,
)
from tests.test_evidence import _complete_bundle


def test_assessment_is_deterministic_and_all_references_are_traceable() -> None:
    bundle = _complete_bundle()
    first = build_market_assessment(bundle)
    second = build_market_assessment(bundle)
    evidence_ids = {item.id for item in bundle.evidence}

    assert first.to_json() == second.to_json()
    assert "趋势趋势" not in first.conclusion
    assert set(first.supporting_evidence_ids) <= evidence_ids
    assert set(first.counter_evidence_ids) <= evidence_ids
    assert set(first.uncertainty_evidence_ids) <= evidence_ids
    assert all(
        set(condition.related_evidence_ids) <= evidence_ids
        for condition in first.switch_conditions
    )
    assert "不预测未来涨跌" in first.compliance_note


def test_confidence_is_an_auditable_100_point_quality_score() -> None:
    confidence = calculate_confidence(_complete_bundle())

    assert sum(item.max_points for item in confidence.components) == 100
    assert confidence.score == round(
        sum(item.awarded_points for item in confidence.components), 1
    )
    assert "不代表上涨概率" in confidence.meaning
    assert confidence.deduction_reasons
    assert all("扣" in reason for reason in confidence.deduction_reasons)


def test_missing_dimension_and_time_gap_reduce_confidence() -> None:
    bundle = _complete_bundle()
    baseline = calculate_confidence(bundle).score
    dimensions = tuple(
        replace(
            item,
            available=False,
            direction=EvidenceDirection.UNKNOWN,
            label="不可用",
        )
        if item.dimension == "breadth"
        else item
        for item in bundle.dimensions
    )
    missing = replace(bundle, dimensions=dimensions, missing_dimensions=("breadth",))

    newest = max(
        item.effective_date
        for item in bundle.evidence
        if item.evidence_type == EvidenceType.FACT and item.effective_date
    )
    delayed_date = (date.fromisoformat(newest) + timedelta(days=7)).isoformat()
    evidence = tuple(
        replace(item, effective_date=delayed_date)
        if item.id == "E-SENTIMENT-01"
        else item
        for item in missing.evidence
    )
    degraded = replace(missing, evidence=evidence, max_trading_day_gap=7)
    result = calculate_confidence(degraded)

    assert result.score < baseline
    assert any("数据维度完整性" in item for item in result.deduction_reasons)
    assert any("数据时点一致性" in item for item in result.deduction_reasons)
    assert any("接口成功" in item for item in result.deduction_reasons)


def test_stale_data_reduces_time_score_against_explicit_reference_date() -> None:
    bundle = _complete_bundle()
    fresh = calculate_confidence(
        bundle, evaluation_date=date.fromisoformat(bundle.assessment_date)
    )
    stale = calculate_confidence(
        bundle,
        evaluation_date=date.fromisoformat(bundle.assessment_date) + timedelta(days=31),
    )
    fresh_time = next(item for item in fresh.components if item.key == "time_alignment")
    stale_time = next(item for item in stale.components if item.key == "time_alignment")

    assert stale_time.awarded_points < fresh_time.awarded_points
    assert "距评估基准日" in stale_time.reason


def test_cross_dimension_conflict_is_explicit_counter_evidence() -> None:
    bundle = _complete_bundle()
    dimensions = tuple(
        replace(
            item,
            direction=EvidenceDirection.POSITIVE,
            label="全A偏活跃",
        )
        if item.dimension == "sentiment"
        else item
        for item in bundle.dimensions
    )
    assessment = build_market_assessment(replace(bundle, dimensions=dimensions))

    assert assessment.state_label == "综合偏弱"
    assert "短线情绪相反" in assessment.primary_conflict
    assert "E-SENTIMENT-I01" in assessment.counter_evidence_ids


def test_no_conflict_does_not_invent_coverage_or_time_problem() -> None:
    assessment = build_market_assessment(_complete_bundle())

    assert "当前未识别到显著的跨维度矛盾" in assessment.primary_conflict
    assert "数据覆盖和时点差异" not in assessment.primary_conflict


def test_sentiment_switch_uses_selected_index_constituent_evidence() -> None:
    assessment = build_market_assessment(_complete_bundle())
    condition = next(
        item for item in assessment.switch_conditions if item.id == "SC-02"
    )

    assert assessment.subject_name in condition.description
    assert "成分股" in condition.description
    assert set(condition.related_evidence_ids) == {
        "E-SENTIMENT-05",
        "E-SENTIMENT-06",
        "E-SENTIMENT-07",
    }


def test_sentiment_switch_falls_back_to_full_market_only_without_constituent_data() -> None:
    bundle = _complete_bundle()
    evidence = tuple(
        item
        for item in bundle.evidence
        if item.id not in {"E-SENTIMENT-05", "E-SENTIMENT-06", "E-SENTIMENT-07"}
    )
    assessment = build_market_assessment(replace(bundle, evidence=evidence))
    condition = next(
        item for item in assessment.switch_conditions if item.id == "SC-02"
    )

    assert "成分情绪不可用时" in condition.description
    assert set(condition.related_evidence_ids) == {
        "E-SENTIMENT-01",
        "E-SENTIMENT-02",
        "E-SENTIMENT-03",
        "E-SENTIMENT-04",
    }


def test_switch_conditions_never_reference_missing_evidence() -> None:
    bundle = _complete_bundle()
    evidence = tuple(
        item for item in bundle.evidence if item.dimension != "sentiment"
    )
    dimensions = tuple(
        replace(
            item,
            available=False,
            direction=EvidenceDirection.UNKNOWN,
            inference_evidence_id=None,
        )
        if item.dimension == "sentiment"
        else item
        for item in bundle.dimensions
    )
    assessment = build_market_assessment(
        replace(
            bundle,
            evidence=evidence,
            dimensions=dimensions,
            missing_dimensions=("sentiment",),
        )
    )
    evidence_ids = {item.id for item in evidence}

    assert all(
        set(condition.related_evidence_ids) <= evidence_ids
        for condition in assessment.switch_conditions
    )
