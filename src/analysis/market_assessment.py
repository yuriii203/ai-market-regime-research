from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.analysis.evidence import (
    DIMENSION_ORDER,
    DimensionAssessment,
    EvidenceBundle,
    EvidenceDirection,
    EvidenceItem,
    EvidenceType,
)


CORE_WEIGHTS = {"trend": 0.45, "breadth": 0.35, "sentiment": 0.20}
COVERAGE_EVIDENCE_IDS = (
    "E-BREADTH-05",
    "E-BREADTH-06",
    "E-VALUATION-03",
    "E-VALUATION-04",
    "E-VALUATION-05",
)


@dataclass(frozen=True)
class ConfidenceComponent:
    key: str
    label: str
    max_points: float
    awarded_points: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "max_points": self.max_points,
            "awarded_points": self.awarded_points,
            "deducted_points": round(self.max_points - self.awarded_points, 1),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConfidenceAssessment:
    score: float
    level: str
    meaning: str
    components: tuple[ConfidenceComponent, ...]
    deduction_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "meaning": self.meaning,
            "components": [item.to_dict() for item in self.components],
            "deduction_reasons": list(self.deduction_reasons),
        }


@dataclass(frozen=True)
class StateSwitchCondition:
    id: str
    description: str
    related_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "related_evidence_ids": list(self.related_evidence_ids),
        }


@dataclass(frozen=True)
class MarketAssessment:
    schema_version: str
    subject_symbol: str
    subject_name: str
    assessment_date: str
    evaluation_date: str
    state_label: str
    conclusion: str
    primary_conflict: str
    supporting_evidence_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...]
    uncertainty_evidence_ids: tuple[str, ...]
    confidence: ConfidenceAssessment
    switch_conditions: tuple[StateSwitchCondition, ...]
    data_time_warnings: tuple[str, ...]
    dimension_summaries: tuple[str, ...]
    compliance_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_symbol": self.subject_symbol,
            "subject_name": self.subject_name,
            "assessment_date": self.assessment_date,
            "evaluation_date": self.evaluation_date,
            "state_label": self.state_label,
            "conclusion": self.conclusion,
            "primary_conflict": self.primary_conflict,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "counter_evidence_ids": list(self.counter_evidence_ids),
            "uncertainty_evidence_ids": list(self.uncertainty_evidence_ids),
            "confidence": self.confidence.to_dict(),
            "switch_conditions": [item.to_dict() for item in self.switch_conditions],
            "data_time_warnings": list(self.data_time_warnings),
            "dimension_summaries": list(self.dimension_summaries),
            "compliance_note": self.compliance_note,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def build_market_assessment(
    bundle: EvidenceBundle,
    *,
    evaluation_date: date | None = None,
    trading_dates: tuple[str, ...] = (),
) -> MarketAssessment:
    """Build a deterministic cross-dimensional assessment from traceable evidence."""
    reference_date = evaluation_date or date.fromisoformat(bundle.assessment_date)
    dimensions = {item.dimension: item for item in bundle.dimensions}
    evidence = {item.id: item for item in bundle.evidence}
    state_label, dominant_direction = _overall_state(dimensions)
    supporting, counter = _directional_evidence(
        dimensions, evidence, dominant_direction
    )
    conclusion = _build_conclusion(dimensions, state_label)
    primary_conflict = _primary_conflict(dimensions, dominant_direction)
    uncertainty_ids = tuple(
        item.id
        for item in bundle.evidence
        if item.evidence_type == EvidenceType.UNCERTAINTY
    )
    confidence = calculate_confidence(
        bundle,
        evaluation_date=reference_date,
        trading_dates=trading_dates,
    )
    time_warnings = bundle.time_conflicts or (
        "各维度事实数据归属于同一交易日；接口获取时间单独记录。",
    )
    summaries = tuple(
        dimensions[name].summary for name in DIMENSION_ORDER if name in dimensions
    )
    return MarketAssessment(
        schema_version="1.0",
        subject_symbol=bundle.subject_symbol,
        subject_name=bundle.subject_name,
        assessment_date=bundle.assessment_date,
        evaluation_date=reference_date.isoformat(),
        state_label=state_label,
        conclusion=conclusion,
        primary_conflict=primary_conflict,
        supporting_evidence_ids=supporting,
        counter_evidence_ids=counter,
        uncertainty_evidence_ids=uncertainty_ids,
        confidence=confidence,
        switch_conditions=_switch_conditions(
            dimensions,
            dominant_direction,
            set(evidence),
            bundle.subject_name,
        ),
        data_time_warnings=time_warnings,
        dimension_summaries=summaries,
        compliance_note="这是基于历史与截面数据的状态描述，不预测未来涨跌，不构成投资或仓位建议。",
    )


def calculate_confidence(
    bundle: EvidenceBundle,
    *,
    evaluation_date: date | None = None,
    trading_dates: tuple[str, ...] = (),
) -> ConfidenceAssessment:
    """Return a transparent data-quality score, not a return probability."""
    reference_date = evaluation_date or date.fromisoformat(bundle.assessment_date)
    dimensions = {item.dimension: item for item in bundle.dimensions}
    evidence = {item.id: item for item in bundle.evidence}

    available_count = sum(item.available for item in bundle.dimensions)
    completeness_ratio = available_count / len(DIMENSION_ORDER)
    completeness = _component(
        "completeness",
        "数据维度完整性",
        25,
        completeness_ratio,
        f"{available_count}/{len(DIMENSION_ORDER)}个维度可用。",
    )

    coverage_values = [
        _bounded_percent(evidence[item_id].raw_value)
        for item_id in COVERAGE_EVIDENCE_IDS
        if item_id in evidence
    ]
    coverage_ratio = (
        sum(coverage_values) / len(coverage_values) / 100
        if coverage_values
        else 0.0
    )
    coverage = _component(
        "coverage",
        "样本覆盖与有效率",
        20,
        coverage_ratio,
        (
            f"宽度及估值的机器可读覆盖指标平均为{coverage_ratio * 100:.1f}%。"
            if coverage_values
            else "没有可用的样本覆盖指标。"
        ),
    )

    fact_dates = sorted(
        {
            date.fromisoformat(item.effective_date)
            for item in bundle.evidence
            if item.evidence_type == EvidenceType.FACT and item.effective_date
        }
    )
    trading_gap = bundle.max_trading_day_gap
    alignment_ratio = (
        max(0.0, 1 - 0.25 * trading_gap)
        if fact_dates and trading_gap is not None
        else 0.5 if fact_dates else 0.0
    )
    age_trading_days = (
        _trading_day_age(fact_dates[-1], reference_date, trading_dates)
        if fact_dates
        else 0
    )
    freshness_ratio = (
        _freshness_ratio(age_trading_days) if fact_dates else 0.0
    )
    time_ratio = min(alignment_ratio, freshness_ratio)
    time_alignment = _component(
        "time_alignment",
        "数据时点一致性",
        20,
        time_ratio,
        (
            f"事实数据覆盖{len(fact_dates)}个有效日期，最大相差"
            f"{trading_gap if trading_gap is not None else '未知'}个交易日；"
            f"最新事实距评估基准日{age_trading_days}个交易日。"
            if fact_dates
            else "事实证据没有有效日期。"
        ),
    )

    directional = [
        dimensions[name].direction
        for name in CORE_WEIGHTS
        if name in dimensions
        and dimensions[name].available
        and dimensions[name].direction
        in {
            EvidenceDirection.POSITIVE,
            EvidenceDirection.NEGATIVE,
            EvidenceDirection.NEUTRAL,
        }
    ]
    counts = {direction: directional.count(direction) for direction in set(directional)}
    agreement_ratio = max(counts.values()) / len(directional) if directional else 0.0
    agreement = _component(
        "agreement",
        "核心证据方向一致性",
        20,
        agreement_ratio,
        (
            f"趋势、宽度、情绪中{max(counts.values())}/{len(directional)}个方向一致。"
            if directional
            else "没有可比较的核心方向证据。"
        ),
    )

    unavailable = len(bundle.missing_dimensions)
    dated_facts = [
        item for item in bundle.evidence
        if item.evidence_type == EvidenceType.FACT and item.effective_date
    ]
    undated_facts = sum(
        item.evidence_type == EvidenceType.FACT and not item.effective_date
        for item in bundle.evidence
    )
    health_ratio = max(0.0, 1 - 0.20 * unavailable - 0.10 * undated_facts)
    health = _component(
        "data_health",
        "接口成功与数据可追溯性",
        15,
        health_ratio if dated_facts else 0.0,
        f"{unavailable}个维度不可用，{undated_facts}条事实缺少日期。",
    )

    components = (completeness, coverage, time_alignment, agreement, health)
    score = round(sum(item.awarded_points for item in components), 1)
    level = "高" if score >= 85 else "中" if score >= 65 else "低"
    deductions = tuple(
        f"{item.label}扣{item.max_points - item.awarded_points:.1f}分：{item.reason}"
        for item in components
        if item.awarded_points < item.max_points
    )
    return ConfidenceAssessment(
        score=score,
        level=level,
        meaning="该分数衡量本次研判的数据和证据质量，不代表上涨概率或预测准确率。",
        components=components,
        deduction_reasons=deductions,
    )


def _overall_state(
    dimensions: dict[str, DimensionAssessment],
) -> tuple[str, EvidenceDirection]:
    score = 0.0
    available_weight = 0.0
    for name, weight in CORE_WEIGHTS.items():
        item = dimensions.get(name)
        if not item or not item.available:
            continue
        available_weight += weight
        if item.direction == EvidenceDirection.POSITIVE:
            score += weight
        elif item.direction == EvidenceDirection.NEGATIVE:
            score -= weight
    normalized = score / available_weight if available_weight else 0.0
    if normalized >= 0.35:
        return "综合偏强", EvidenceDirection.POSITIVE
    if normalized > 0.10:
        return "震荡偏强", EvidenceDirection.POSITIVE
    if normalized <= -0.35:
        return "综合偏弱", EvidenceDirection.NEGATIVE
    if normalized < -0.10:
        return "震荡偏弱", EvidenceDirection.NEGATIVE
    return "方向分化", EvidenceDirection.NEUTRAL


def _directional_evidence(
    dimensions: dict[str, DimensionAssessment],
    evidence: dict[str, EvidenceItem],
    dominant: EvidenceDirection,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    supporting: list[str] = []
    counter: list[str] = []
    for name in CORE_WEIGHTS:
        item = dimensions.get(name)
        if not item or not item.inference_evidence_id:
            continue
        evidence_id = item.inference_evidence_id
        if item.direction == dominant or (
            dominant == EvidenceDirection.NEUTRAL
            and item.direction == EvidenceDirection.NEUTRAL
        ):
            supporting.append(evidence_id)
        elif item.direction in {EvidenceDirection.POSITIVE, EvidenceDirection.NEGATIVE}:
            counter.append(evidence_id)
    return tuple(supporting), tuple(counter)


def _build_conclusion(
    dimensions: dict[str, DimensionAssessment], state_label: str
) -> str:
    parts = [state_label]
    for name in ("trend", "breadth", "sentiment"):
        item = dimensions.get(name)
        if item and item.available:
            parts.append(
                item.label if name == "trend" else f"{_dimension_name(name)}{item.label}"
            )
    liquidity = dimensions.get("liquidity")
    if liquidity and liquidity.available and liquidity.label != "基本持平":
        parts.append(f"交易活跃度{liquidity.label}")
    return "，".join(parts) + "。"


def _primary_conflict(
    dimensions: dict[str, DimensionAssessment], dominant: EvidenceDirection
) -> str:
    trend = dimensions.get("trend")
    breadth = dimensions.get("breadth")
    sentiment = dimensions.get("sentiment")
    if trend and breadth and trend.available and breadth.available:
        if {
            trend.direction,
            breadth.direction,
        } == {EvidenceDirection.POSITIVE, EvidenceDirection.NEGATIVE}:
            return "指数趋势与成分股市场宽度方向背离，是当前需要优先跟踪的矛盾。"
    if sentiment and sentiment.available and sentiment.direction not in {
        dominant,
        EvidenceDirection.NEUTRAL,
    }:
        return "趋势与宽度的主方向和短线情绪相反，短线活跃度尚未确认中期状态反转。"
    if dominant == EvidenceDirection.NEUTRAL:
        return "核心维度方向分化，当前没有形成占优的单一方向。"
    return "趋势、市场宽度与短线情绪方向基本一致，当前未识别到显著的跨维度矛盾。"


def _switch_conditions(
    dimensions: dict[str, DimensionAssessment],
    dominant: EvidenceDirection,
    available_evidence_ids: set[str],
    subject_name: str,
) -> tuple[StateSwitchCondition, ...]:
    if dominant == EvidenceDirection.NEGATIVE:
        cross = (
            "若20日收益率转正、收盘价重新站上MA20，且市场宽度改善至温和扩张以上，"
            "综合状态才具备转强条件。"
        )
    elif dominant == EvidenceDirection.POSITIVE:
        cross = (
            "若20日收益率转负、收盘价跌破MA20，且市场宽度转为温和收缩以下，"
            "综合状态将具备转弱条件。"
        )
    else:
        cross = (
            "若趋势和市场宽度同时转为同一方向，且至少达到偏强/温和扩张或"
            "偏弱/温和收缩，综合状态将结束分化。"
        )
    selected_sentiment_ids = _existing_ids(
        ("E-SENTIMENT-05", "E-SENTIMENT-06", "E-SENTIMENT-07"),
        available_evidence_ids,
    )
    if {"E-SENTIMENT-05", "E-SENTIMENT-06"} <= set(selected_sentiment_ids):
        sentiment_condition = (
            f"若{subject_name}成分股的涨跌停强弱关系反转，且炸板情况发生明显变化，"
            "短线情绪判断将改变；该条件只能修正情绪维度，不能单独推翻中期趋势。"
        )
        sentiment_evidence_ids = selected_sentiment_ids
    else:
        sentiment_condition = (
            "所选指数成分情绪不可用时，若全A涨跌停强弱关系反转并伴随炸板情况变化，"
            "短线情绪判断将改变；这是背景口径，不能单独推翻中期趋势。"
        )
        sentiment_evidence_ids = _existing_ids(
            (
                "E-SENTIMENT-01",
                "E-SENTIMENT-02",
                "E-SENTIMENT-03",
                "E-SENTIMENT-04",
            ),
            available_evidence_ids,
        )
    conditions = [
        StateSwitchCondition(
            id="SC-01",
            description=cross,
            related_evidence_ids=_existing_ids(
                ("E-TREND-01", "E-TREND-03", "E-BREADTH-04"),
                available_evidence_ids,
            ),
        ),
        StateSwitchCondition(
            id="SC-02",
            description=sentiment_condition,
            related_evidence_ids=sentiment_evidence_ids,
        ),
    ]
    return tuple(conditions)


def _component(
    key: str, label: str, max_points: float, ratio: float, reason: str
) -> ConfidenceComponent:
    return ConfidenceComponent(
        key=key,
        label=label,
        max_points=max_points,
        awarded_points=round(max_points * min(1.0, max(0.0, ratio)), 1),
        reason=reason,
    )


def _bounded_percent(value: object) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return min(100.0, max(0.0, float(value)))


def _freshness_ratio(age_days: int) -> float:
    if age_days <= 1:
        return 1.0
    if age_days <= 3:
        return 0.75
    if age_days <= 10:
        return 0.25
    return 0.0


def _trading_day_age(
    latest_effective_date: date,
    reference_date: date,
    trading_dates: tuple[str, ...],
) -> int:
    if reference_date <= latest_effective_date:
        return 0
    known = sorted(
        date.fromisoformat(value)
        for value in set(trading_dates)
        if latest_effective_date < date.fromisoformat(value) <= reference_date
    )
    if trading_dates:
        return len(known)
    # Unit-test and historical fallback when no exchange calendar was supplied.
    current = latest_effective_date
    weekdays = 0
    from datetime import timedelta

    while current < reference_date:
        current += timedelta(days=1)
        if current.weekday() < 5:
            weekdays += 1
    return weekdays


def _dimension_name(name: str) -> str:
    return {"trend": "趋势", "breadth": "市场宽度", "sentiment": "市场情绪"}[name]


def _existing_ids(
    candidates: tuple[str, ...], available: set[str]
) -> tuple[str, ...]:
    return tuple(item for item in candidates if item in available)
