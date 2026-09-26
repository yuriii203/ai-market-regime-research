from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any, Iterable

from src.analysis.trend_state import evaluate_trend_state
from src.clients.fuyao import (
    DataResult,
    PoolResult,
    PriceBar,
    ResponseMetadata,
    StockSnapshot,
    ValuationSnapshot,
)
from src.indicators.market_dimensions import (
    calculate_market_breadth,
    calculate_market_sentiment,
)
from src.indicators.trend import TrendIndicators


DIMENSION_ORDER = (
    "trend",
    "breadth",
    "style",
    "liquidity",
    "sentiment",
    "valuation",
)


class EvidenceType(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    UNCERTAINTY = "uncertainty"


class EvidenceDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CONTEXT = "context"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    dimension: str
    evidence_type: EvidenceType
    direction: EvidenceDirection
    description: str
    raw_value: float | int | str | None
    unit: str | None
    scope: str
    effective_date: str | None
    source: str
    endpoint: str | None
    request_ids: tuple[str, ...]
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimension": self.dimension,
            "evidence_type": self.evidence_type.value,
            "direction": self.direction.value,
            "description": self.description,
            "raw_value": self.raw_value,
            "unit": self.unit,
            "scope": self.scope,
            "effective_date": self.effective_date,
            "source": self.source,
            "endpoint": self.endpoint,
            "request_ids": list(self.request_ids),
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class DimensionAssessment:
    dimension: str
    label: str
    direction: EvidenceDirection
    summary: str
    available: bool
    fact_evidence_ids: tuple[str, ...]
    inference_evidence_id: str | None
    uncertainty_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "label": self.label,
            "direction": self.direction.value,
            "summary": self.summary,
            "available": self.available,
            "fact_evidence_ids": list(self.fact_evidence_ids),
            "inference_evidence_id": self.inference_evidence_id,
            "uncertainty_evidence_ids": list(self.uncertainty_evidence_ids),
        }


@dataclass(frozen=True)
class DimensionEvidence:
    assessment: DimensionAssessment
    evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class EvidenceBundle:
    schema_version: str
    subject_symbol: str
    subject_name: str
    assessment_date: str
    dimensions: tuple[DimensionAssessment, ...]
    evidence: tuple[EvidenceItem, ...]
    missing_dimensions: tuple[str, ...]
    time_conflicts: tuple[str, ...]
    effective_dates: tuple[str, ...]
    max_trading_day_gap: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_symbol": self.subject_symbol,
            "subject_name": self.subject_name,
            "assessment_date": self.assessment_date,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evidence": [item.to_dict() for item in self.evidence],
            "missing_dimensions": list(self.missing_dimensions),
            "time_conflicts": list(self.time_conflicts),
            "effective_dates": list(self.effective_dates),
            "max_trading_day_gap": self.max_trading_day_gap,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def build_evidence_bundle(
    *,
    subject_symbol: str,
    subject_name: str,
    indicators: TrendIndicators,
    trend_metadata: ResponseMetadata,
    style_histories: dict[str, DataResult[PriceBar]] | None = None,
    breadth_snapshots: list[StockSnapshot] | None = None,
    breadth_metadata: tuple[ResponseMetadata, ...] = (),
    breadth_universe_count: int | None = None,
    breadth_sampled_count: int | None = None,
    sentiment_pools: tuple[PoolResult, PoolResult, PoolResult] | None = None,
    sentiment_query_date: str | None = None,
    sentiment_universe_codes: set[str] | None = None,
    valuations: list[ValuationSnapshot] | None = None,
    valuation_metadata: tuple[ResponseMetadata, ...] = (),
    valuation_universe_count: int | None = None,
    valuation_sampled_count: int | None = None,
    breadth_effective_date: str | None = None,
    valuation_effective_date: str | None = None,
    trading_dates: tuple[str, ...] = (),
    errors: dict[str, str] | None = None,
) -> EvidenceBundle:
    errors = errors or {}
    results: list[DimensionEvidence] = []
    results.append(
        evaluate_trend_evidence(
            indicators, trend_metadata, subject_name, error=errors.get("trend")
        )
    )
    results.append(
        evaluate_breadth_evidence(
            breadth_snapshots,
            breadth_metadata,
            breadth_universe_count,
            breadth_sampled_count,
            subject_name,
            effective_date=breadth_effective_date,
            error=errors.get("breadth"),
        )
    )
    results.append(
        evaluate_style_evidence(style_histories, error=errors.get("style"))
    )
    results.append(
        evaluate_liquidity_evidence(
            indicators, trend_metadata, subject_name, error=errors.get("liquidity")
        )
    )
    results.append(
        evaluate_sentiment_evidence(
            sentiment_pools,
            sentiment_query_date,
            sentiment_universe_codes,
            subject_name,
            error=errors.get("sentiment"),
        )
    )
    results.append(
        evaluate_valuation_evidence(
            valuations,
            valuation_metadata,
            valuation_universe_count,
            valuation_sampled_count,
            subject_name,
            effective_date=valuation_effective_date,
            error=errors.get("valuation"),
        )
    )

    ordered = sorted(
        results, key=lambda item: DIMENSION_ORDER.index(item.assessment.dimension)
    )
    evidence = tuple(item for result in ordered for item in result.evidence)
    dimensions = tuple(result.assessment for result in ordered)
    dates = sorted(
        {
            item.effective_date
            for item in evidence
            if item.effective_date and item.evidence_type == EvidenceType.FACT
        }
    )
    trading_day_gap = _trading_day_span(dates, trading_dates)
    if len(dates) <= 1:
        time_conflicts = ()
    elif trading_day_gap is None:
        time_conflicts = (
            "各维度有效日期不完全一致，且交易日历不足以比较：" + "、".join(dates),
        )
    else:
        time_conflicts = (
            f"各维度有效日期相差{trading_day_gap}个交易日：" + "、".join(dates),
        )
    return EvidenceBundle(
        schema_version="1.0",
        subject_symbol=subject_symbol,
        subject_name=subject_name,
        assessment_date=indicators.as_of.isoformat(),
        dimensions=dimensions,
        evidence=evidence,
        missing_dimensions=tuple(
            item.dimension for item in dimensions if not item.available
        ),
        time_conflicts=time_conflicts,
        effective_dates=tuple(dates),
        max_trading_day_gap=trading_day_gap,
    )


def evaluate_trend_evidence(
    indicators: TrendIndicators,
    metadata: ResponseMetadata,
    scope: str,
    *,
    error: str | None = None,
) -> DimensionEvidence:
    if error:
        return _unavailable_dimension("trend", "趋势", scope, error)
    effective_date = indicators.as_of.isoformat()
    facts = (
        _fact("E-TREND-01", "trend", indicators.return_20d_pct, "%", scope,
              effective_date, metadata, "20日收益率", _sign_direction(indicators.return_20d_pct),
              "/api/a-share-index/prices/historical"),
        _fact("E-TREND-02", "trend", indicators.return_60d_pct, "%", scope,
              effective_date, metadata, "60日收益率", _sign_direction(indicators.return_60d_pct),
              "/api/a-share-index/prices/historical"),
        _fact("E-TREND-03", "trend", indicators.close_vs_ma20_pct, "%", scope,
              effective_date, metadata, "收盘价相对MA20", _sign_direction(indicators.close_vs_ma20_pct),
              "/api/a-share-index/prices/historical"),
        _fact("E-TREND-04", "trend", indicators.close_vs_ma60_pct, "%", scope,
              effective_date, metadata, "收盘价相对MA60", _sign_direction(indicators.close_vs_ma60_pct),
              "/api/a-share-index/prices/historical"),
        _fact("E-TREND-05", "trend", indicators.max_drawdown_60d_pct, "%", scope,
              effective_date, metadata, "近60日最大回撤", EvidenceDirection.CONTEXT,
              "/api/a-share-index/prices/historical"),
    )
    state = evaluate_trend_state(indicators)
    direction = _tone_direction(state.tone)
    inference = _inference(
        "E-TREND-I01", "trend", direction, scope, effective_date,
        f"趋势判断为{state.label}：{state.summary}", state.label,
    )
    uncertainties = tuple(
        _uncertainty(
            f"E-TREND-U{index:02d}", "trend", warning, scope, effective_date
        )
        for index, warning in enumerate(indicators.warnings, start=1)
    )
    return _available_dimension(
        "trend", state.label, direction, state.summary, facts, inference, uncertainties
    )


def evaluate_breadth_evidence(
    snapshots: list[StockSnapshot] | None,
    metadata: tuple[ResponseMetadata, ...],
    universe_count: int | None,
    sampled_count: int | None,
    scope: str,
    *,
    effective_date: str | None = None,
    error: str | None = None,
) -> DimensionEvidence:
    if error or snapshots is None or universe_count is None or sampled_count is None:
        return _unavailable_dimension(
            "breadth", "市场宽度", scope, error or "市场宽度输入不完整"
        )
    breadth = calculate_market_breadth(snapshots, universe_count)
    retrieved_at = _latest_retrieval_time(metadata)
    request_ids = _request_ids(metadata)
    endpoint = "/api/a-share/prices/snapshot"
    facts = (
        _plain_fact("E-BREADTH-01", "breadth", breadth.advancing, "只", scope,
                    effective_date, request_ids, "上涨家数", EvidenceDirection.POSITIVE, endpoint,
                    retrieved_at=retrieved_at),
        _plain_fact("E-BREADTH-02", "breadth", breadth.declining, "只", scope,
                    effective_date, request_ids, "下跌家数", EvidenceDirection.NEGATIVE, endpoint,
                    retrieved_at=retrieved_at),
        _plain_fact("E-BREADTH-03", "breadth", breadth.flat, "只", scope,
                    effective_date, request_ids, "平盘家数", EvidenceDirection.NEUTRAL, endpoint,
                    retrieved_at=retrieved_at),
        _plain_fact("E-BREADTH-04", "breadth", breadth.breadth_score_pct, "%", scope,
                    effective_date, request_ids, "宽度净值", _sign_direction(breadth.breadth_score_pct), endpoint,
                    retrieved_at=retrieved_at),
        _plain_fact("E-BREADTH-05", "breadth", breadth.valid_count / sampled_count * 100 if sampled_count else 0,
                    "%", scope, effective_date, request_ids, "抽样内有效覆盖率", EvidenceDirection.CONTEXT, endpoint,
                    retrieved_at=retrieved_at),
        _plain_fact("E-BREADTH-06", "breadth", sampled_count / universe_count * 100 if universe_count else 0,
                    "%", scope, effective_date, request_ids, "成分股抽样覆盖率", EvidenceDirection.CONTEXT, endpoint,
                    retrieved_at=retrieved_at),
    )
    label, direction = _breadth_label(breadth.breadth_score_pct)
    summary = (
        f"{scope}宽度{label}：上涨{breadth.advancing}只、"
        f"下跌{breadth.declining}只，宽度净值{breadth.breadth_score_pct:+.1f}%。"
    )
    inference = _inference(
        "E-BREADTH-I01", "breadth", direction, scope, effective_date,
        summary, label,
    )
    uncertainties: list[EvidenceItem] = []
    if sampled_count < universe_count:
        uncertainties.append(
            _uncertainty(
                "E-BREADTH-U01", "breadth",
                f"使用{sampled_count}只等距样本代表{universe_count}只A股成分，并非全量统计。",
                scope, effective_date,
            )
        )
    if breadth.valid_count < sampled_count:
        uncertainties.append(
            _uncertainty(
                "E-BREADTH-U02", "breadth",
                f"{sampled_count - breadth.valid_count}只样本缺少有效涨跌幅。",
                scope, effective_date,
            )
        )
    uncertainties.append(
        _uncertainty(
            "E-BREADTH-U03", "breadth",
            "快照接口未单独返回行情归属日，当前按交易日历最近已完成交易日归属。",
            scope, effective_date,
        )
    )
    return _available_dimension(
        "breadth", label, direction, summary, facts, inference, tuple(uncertainties)
    )


def evaluate_style_evidence(
    histories: dict[str, DataResult[PriceBar]] | None,
    *,
    error: str | None = None,
) -> DimensionEvidence:
    if error or not histories:
        return _unavailable_dimension(
            "style", "风格/宽基", "四个宽基指数", error or "宽基历史数据不可用"
        )
    rows: list[tuple[str, str, float, float, DataResult[PriceBar]]] = []
    facts: list[EvidenceItem] = []
    for index, (name, result) in enumerate(sorted(histories.items()), start=1):
        indicators = _simple_returns(result.items)
        code = f"E-STYLE-{index:02d}"
        date_value = result.items[-1].trading_date.isoformat()
        rows.append((name, code, indicators[0], indicators[1], result))
        facts.append(
            _fact(
                code, "style", indicators[0], "%", name, date_value,
                result.metadata, "20日收益率", _sign_direction(indicators[0]),
                "/api/a-share-index/prices/historical",
            )
        )
        facts.append(
            _fact(
                f"{code}-60D", "style", indicators[1], "%", name, date_value,
                result.metadata, "60日收益率", _sign_direction(indicators[1]),
                "/api/a-share-index/prices/historical",
            )
        )
    ranked = sorted(rows, key=lambda row: row[2], reverse=True)
    leader, laggard = ranked[0], ranked[-1]
    spread = leader[2] - laggard[2]
    absolute_note = "，但其20日收益率仍为负" if leader[2] < 0 else ""
    label = f"{leader[0]}相对领先"
    summary = (
        f"{leader[0]}过去20日相对领先{absolute_note}；"
        f"领先{laggard[0]} {spread:.2f}个百分点。"
    )
    effective_date = max(row[4].items[-1].trading_date.isoformat() for row in rows)
    inference = _inference(
        "E-STYLE-I01", "style", EvidenceDirection.CONTEXT,
        "四个宽基指数", effective_date, summary, label,
    )
    return _available_dimension(
        "style", label, EvidenceDirection.CONTEXT, summary,
        tuple(facts), inference, (),
    )


def evaluate_liquidity_evidence(
    indicators: TrendIndicators,
    metadata: ResponseMetadata,
    scope: str,
    *,
    error: str | None = None,
) -> DimensionEvidence:
    if error:
        return _unavailable_dimension("liquidity", "流动性", scope, error)
    date_value = indicators.as_of.isoformat()
    endpoint = "/api/a-share-index/prices/historical"
    facts: list[EvidenceItem] = []
    if indicators.turnover_20d_avg is not None:
        facts.append(
            _fact("E-LIQUIDITY-01", "liquidity", indicators.turnover_20d_avg,
                  "元", scope, date_value, metadata, "近20日平均成交额",
                  EvidenceDirection.CONTEXT, endpoint)
        )
    if indicators.turnover_5d_vs_20d_pct is None:
        uncertainty = _uncertainty(
            "E-LIQUIDITY-U01", "liquidity", "成交额变化不可用。", scope, date_value
        )
        assessment = DimensionAssessment(
            dimension="liquidity", label="不可用", direction=EvidenceDirection.UNKNOWN,
            summary="成交额变化数据不足，不能判断交易活跃度变化。", available=False,
            fact_evidence_ids=tuple(item.id for item in facts), inference_evidence_id=None,
            uncertainty_evidence_ids=(uncertainty.id,),
        )
        return DimensionEvidence(assessment, tuple(facts) + (uncertainty,))
    change = indicators.turnover_5d_vs_20d_pct
    facts.append(
        _fact("E-LIQUIDITY-02", "liquidity", change, "%", scope, date_value,
              metadata, "近5日成交额较20日均值", EvidenceDirection.CONTEXT, endpoint)
    )
    label = _liquidity_label(change)
    summary = f"{scope}近5日平均成交额较20日均值{change:+.2f}%，交易活跃度{label}。"
    inference = _inference(
        "E-LIQUIDITY-I01", "liquidity", EvidenceDirection.CONTEXT,
        scope, date_value, summary, label,
    )
    return _available_dimension(
        "liquidity", label, EvidenceDirection.CONTEXT, summary,
        tuple(facts), inference, (),
    )


def evaluate_sentiment_evidence(
    pools: tuple[PoolResult, PoolResult, PoolResult] | None,
    query_date: str | None,
    universe_codes: set[str] | None,
    subject_name: str,
    *,
    error: str | None = None,
) -> DimensionEvidence:
    if error or pools is None or query_date is None:
        return _unavailable_dimension(
            "sentiment", "市场情绪", "全A及所选指数", error or "情绪输入不完整"
        )
    up, down, broken = pools
    full = calculate_market_sentiment(up, down, broken)
    selected = (
        calculate_market_sentiment(
            up, down, broken, universe_codes=universe_codes
        )
        if universe_codes is not None
        else None
    )
    request_ids = tuple(
        request_id
        for pool in pools
        for request_id in (pool.request_ids or (pool.metadata.request_id,))
    )
    retrieved_at = max(pool.metadata.retrieval_time for pool in pools).isoformat()
    facts = [
        _plain_fact("E-SENTIMENT-01", "sentiment", full.limit_up_count, "只", "全A",
                    query_date, request_ids, "涨停家数", EvidenceDirection.POSITIVE,
                    "/api/a-share/special-data/limit-up-pool", retrieved_at=retrieved_at),
        _plain_fact("E-SENTIMENT-02", "sentiment", full.limit_down_count, "只", "全A",
                    query_date, request_ids, "跌停家数", EvidenceDirection.NEGATIVE,
                    "/api/a-share/special-data/limit-down-pool", retrieved_at=retrieved_at),
        _plain_fact("E-SENTIMENT-03", "sentiment", full.limit_break_count, "只", "全A",
                    query_date, request_ids, "炸板家数", EvidenceDirection.NEGATIVE,
                    "/api/a-share/special-data/limit-break-pool", retrieved_at=retrieved_at),
    ]
    if full.break_rate_pct is not None:
        facts.append(
            _plain_fact("E-SENTIMENT-04", "sentiment", full.break_rate_pct, "%", "全A",
                        query_date, request_ids, "炸板率", EvidenceDirection.CONTEXT,
                        "/api/a-share/special-data", retrieved_at=retrieved_at)
        )
    selected_label = "所选指数成分情绪不可用"
    uncertainties: list[EvidenceItem] = []
    if selected is None:
        uncertainties.append(
            _uncertainty(
                "E-SENTIMENT-U01", "sentiment", "缺少所选指数A股成分范围。",
                subject_name, query_date,
            )
        )
    else:
        facts.extend(
            [
                _plain_fact("E-SENTIMENT-05", "sentiment", selected.limit_up_count, "只",
                            subject_name, query_date, request_ids, "成分股涨停家数",
                            EvidenceDirection.POSITIVE, "/api/a-share/special-data",
                            retrieved_at=retrieved_at),
                _plain_fact("E-SENTIMENT-06", "sentiment", selected.limit_down_count, "只",
                            subject_name, query_date, request_ids, "成分股跌停家数",
                            EvidenceDirection.NEGATIVE, "/api/a-share/special-data",
                            retrieved_at=retrieved_at),
                _plain_fact("E-SENTIMENT-07", "sentiment", selected.limit_break_count, "只",
                            subject_name, query_date, request_ids, "成分股炸板家数",
                            EvidenceDirection.NEGATIVE, "/api/a-share/special-data",
                            retrieved_at=retrieved_at),
            ]
        )
        selected_label = _sentiment_label(selected.limit_up_count, selected.limit_down_count,
                                          selected.break_rate_pct)
    full_label = _sentiment_label(
        full.limit_up_count, full.limit_down_count, full.break_rate_pct
    )
    if selected is None:
        direction = _sentiment_direction(
            full.limit_up_count, full.limit_down_count, full.break_rate_pct
        )
        label = f"{subject_name}成分情绪不可用；全A{full_label}（背景）"
    else:
        direction = _sentiment_direction(
            selected.limit_up_count,
            selected.limit_down_count,
            selected.break_rate_pct,
        )
        label = f"{subject_name}{selected_label}；全A{full_label}（背景）"
    summary = (
        f"全A涨停{full.limit_up_count}只、跌停{full.limit_down_count}只、"
        f"炸板{full.limit_break_count}只，情绪{full_label}，仅作市场背景；"
        + (
            f"{subject_name}成分情绪{selected_label}，作为本次情绪方向投票。"
            if selected is not None
            else f"{subject_name}成分情绪不可用，本次暂以全A情绪代替。"
        )
    )
    inference = _inference(
        "E-SENTIMENT-I01", "sentiment", direction, subject_name,
        query_date, summary, label,
    )
    return _available_dimension(
        "sentiment", label, direction, summary, tuple(facts), inference,
        tuple(uncertainties),
    )


def evaluate_valuation_evidence(
    valuations: list[ValuationSnapshot] | None,
    metadata: tuple[ResponseMetadata, ...],
    universe_count: int | None,
    sampled_count: int | None,
    scope: str,
    *,
    effective_date: str | None = None,
    error: str | None = None,
) -> DimensionEvidence:
    if error or valuations is None or universe_count is None or sampled_count is None:
        return _unavailable_dimension(
            "valuation", "估值", scope, error or "估值输入不完整"
        )
    pe_values = [item.pe_ttm for item in valuations if item.pe_ttm is not None and item.pe_ttm > 0]
    pb_values = [item.pb_mrq for item in valuations if item.pb_mrq is not None and item.pb_mrq > 0]
    retrieved_at = _latest_retrieval_time(metadata)
    request_ids = _request_ids(metadata)
    endpoint = "/api/a-share/valuations/snapshot"
    facts: list[EvidenceItem] = []
    if pe_values:
        facts.append(
            _plain_fact("E-VALUATION-01", "valuation", median(pe_values), "倍", scope,
                        effective_date, request_ids, "正值PE(TTM)中位数",
                        EvidenceDirection.CONTEXT, endpoint, retrieved_at=retrieved_at)
        )
    if pb_values:
        facts.append(
            _plain_fact("E-VALUATION-02", "valuation", median(pb_values), "倍", scope,
                        effective_date, request_ids, "正值PB(MRQ)中位数",
                        EvidenceDirection.CONTEXT, endpoint, retrieved_at=retrieved_at)
        )
    pe_rate = len(pe_values) / sampled_count * 100 if sampled_count else 0
    pb_rate = len(pb_values) / sampled_count * 100 if sampled_count else 0
    facts.extend(
        [
            _plain_fact("E-VALUATION-03", "valuation", pe_rate, "%", scope,
                        effective_date, request_ids, "PE有效率", EvidenceDirection.CONTEXT, endpoint,
                        retrieved_at=retrieved_at),
            _plain_fact("E-VALUATION-04", "valuation", pb_rate, "%", scope,
                        effective_date, request_ids, "PB有效率", EvidenceDirection.CONTEXT, endpoint,
                        retrieved_at=retrieved_at),
            _plain_fact("E-VALUATION-05", "valuation",
                        sampled_count / universe_count * 100 if universe_count else 0,
                        "%", scope, effective_date, request_ids, "成分股抽样覆盖率",
                        EvidenceDirection.CONTEXT, endpoint, retrieved_at=retrieved_at),
        ]
    )
    label = "当前估值截面可用" if pe_values or pb_values else "有效估值不足"
    summary = (
        f"{scope}当前估值截面：PE有效率{pe_rate:.1f}%，PB有效率{pb_rate:.1f}%；"
        "缺少历史估值分位，不能据此判断高估或低估。"
    )
    inference = _inference(
        "E-VALUATION-I01", "valuation", EvidenceDirection.CONTEXT,
        scope, effective_date, summary, label,
    )
    uncertainties = (
        _uncertainty(
            "E-VALUATION-U01", "valuation",
            "接口不提供历史估值分位，当前截面不能单独支持高估或低估判断。",
            scope, effective_date,
        ),
    )
    if sampled_count < universe_count:
        uncertainties += (
            _uncertainty(
                "E-VALUATION-U02", "valuation",
                f"使用{sampled_count}只样本代表{universe_count}只A股成分。",
                scope, effective_date,
            ),
        )
    uncertainties += (
        _uncertainty(
            "E-VALUATION-U03", "valuation",
            "估值快照接口未单独返回行情归属日，当前按交易日历最近已完成交易日归属。",
            scope, effective_date,
        ),
    )
    return _available_dimension(
        "valuation", label, EvidenceDirection.CONTEXT, summary,
        tuple(facts), inference, uncertainties,
    )


def _available_dimension(
    dimension: str,
    label: str,
    direction: EvidenceDirection,
    summary: str,
    facts: tuple[EvidenceItem, ...],
    inference: EvidenceItem,
    uncertainties: tuple[EvidenceItem, ...],
) -> DimensionEvidence:
    assessment = DimensionAssessment(
        dimension=dimension,
        label=label,
        direction=direction,
        summary=summary,
        available=True,
        fact_evidence_ids=tuple(item.id for item in facts),
        inference_evidence_id=inference.id,
        uncertainty_evidence_ids=tuple(item.id for item in uncertainties),
    )
    return DimensionEvidence(assessment, facts + (inference,) + uncertainties)


def _unavailable_dimension(
    dimension: str, label: str, scope: str, error: str
) -> DimensionEvidence:
    item = _uncertainty(
        f"E-{dimension.upper()}-U01", dimension,
        f"{label}不可用：{error}", scope, None,
    )
    assessment = DimensionAssessment(
        dimension=dimension,
        label="不可用",
        direction=EvidenceDirection.UNKNOWN,
        summary=item.description,
        available=False,
        fact_evidence_ids=(),
        inference_evidence_id=None,
        uncertainty_evidence_ids=(item.id,),
    )
    return DimensionEvidence(assessment, (item,))


def _fact(
    evidence_id: str,
    dimension: str,
    value: float | int,
    unit: str,
    scope: str,
    effective_date: str,
    metadata: ResponseMetadata,
    label: str,
    direction: EvidenceDirection,
    endpoint: str,
) -> EvidenceItem:
    return _plain_fact(
        evidence_id, dimension, value, unit, scope, effective_date,
        (metadata.request_id,), label, direction, endpoint,
        retrieved_at=metadata.retrieval_time.isoformat(),
    )


def _plain_fact(
    evidence_id: str,
    dimension: str,
    value: float | int,
    unit: str,
    scope: str,
    effective_date: str | None,
    request_ids: tuple[str, ...],
    label: str,
    direction: EvidenceDirection,
    endpoint: str,
    *,
    retrieved_at: str | None = None,
) -> EvidenceItem:
    formatted = f"{value:+.2f}" if isinstance(value, float) else str(value)
    return EvidenceItem(
        id=evidence_id,
        dimension=dimension,
        evidence_type=EvidenceType.FACT,
        direction=direction,
        description=f"{label}为{formatted}{unit}。",
        raw_value=value,
        unit=unit,
        scope=scope,
        effective_date=effective_date,
        source="扶摇金融数据API",
        endpoint=endpoint,
        request_ids=tuple(value for value in request_ids if value),
        retrieved_at=retrieved_at,
    )


def _inference(
    evidence_id: str,
    dimension: str,
    direction: EvidenceDirection,
    scope: str,
    effective_date: str | None,
    description: str,
    label: str,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        dimension=dimension,
        evidence_type=EvidenceType.INFERENCE,
        direction=direction,
        description=description,
        raw_value=label,
        unit=None,
        scope=scope,
        effective_date=effective_date,
        source="本地确定性规则",
        endpoint=None,
        request_ids=(),
    )


def _uncertainty(
    evidence_id: str,
    dimension: str,
    description: str,
    scope: str,
    effective_date: str | None,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        dimension=dimension,
        evidence_type=EvidenceType.UNCERTAINTY,
        direction=EvidenceDirection.UNKNOWN,
        description=description,
        raw_value=None,
        unit=None,
        scope=scope,
        effective_date=effective_date,
        source="本地数据质量检查",
        endpoint=None,
        request_ids=(),
    )


def _sign_direction(value: float) -> EvidenceDirection:
    if value > 0:
        return EvidenceDirection.POSITIVE
    if value < 0:
        return EvidenceDirection.NEGATIVE
    return EvidenceDirection.NEUTRAL


def _tone_direction(tone: str) -> EvidenceDirection:
    if tone == "positive":
        return EvidenceDirection.POSITIVE
    if tone == "negative":
        return EvidenceDirection.NEGATIVE
    return EvidenceDirection.NEUTRAL


def _breadth_label(score: float) -> tuple[str, EvidenceDirection]:
    if score >= 20:
        return "明显扩张", EvidenceDirection.POSITIVE
    if score >= 5:
        return "温和扩张", EvidenceDirection.POSITIVE
    if score > -5:
        return "基本均衡", EvidenceDirection.NEUTRAL
    if score > -20:
        return "温和收缩", EvidenceDirection.NEGATIVE
    return "明显收缩", EvidenceDirection.NEGATIVE


def _liquidity_label(change: float) -> str:
    magnitude = abs(change)
    if magnitude < 5:
        return "基本持平"
    if magnitude < 20:
        return "温和放大" if change > 0 else "温和收缩"
    return "明显放大" if change > 0 else "明显收缩"


def _sentiment_label(
    limit_up: int, limit_down: int, break_rate: float | None
) -> str:
    if limit_up == 0 and limit_down == 0:
        return "中性"
    if limit_up >= max(3, limit_down * 1.5) and (
        break_rate is None or break_rate < 30
    ):
        return "偏活跃"
    if limit_down >= max(3, limit_up * 1.5):
        return "偏弱"
    return "分化"


def _sentiment_direction(
    limit_up: int, limit_down: int, break_rate: float | None
) -> EvidenceDirection:
    label = _sentiment_label(limit_up, limit_down, break_rate)
    if label == "偏活跃":
        return EvidenceDirection.POSITIVE
    if label == "偏弱":
        return EvidenceDirection.NEGATIVE
    return EvidenceDirection.NEUTRAL


def _simple_returns(bars: list[PriceBar]) -> tuple[float, float]:
    ordered = sorted(bars, key=lambda item: item.date_ms)
    if len(ordered) < 61:
        raise ValueError("宽基相对表现至少需要61根日K")
    latest = ordered[-1].close_price
    return (
        (latest / ordered[-21].close_price - 1) * 100,
        (latest / ordered[-61].close_price - 1) * 100,
    )


def _latest_metadata_date(metadata: Iterable[ResponseMetadata]) -> str | None:
    values = tuple(metadata)
    return max(item.timestamp for item in values).date().isoformat() if values else None


def _request_ids(metadata: Iterable[ResponseMetadata]) -> tuple[str, ...]:
    return tuple(item.request_id for item in metadata if item.request_id)


def _latest_retrieval_time(metadata: Iterable[ResponseMetadata]) -> str | None:
    values = tuple(metadata)
    return max(item.retrieval_time for item in values).isoformat() if values else None


def _trading_day_span(
    effective_dates: list[str], trading_dates: tuple[str, ...]
) -> int | None:
    if len(effective_dates) <= 1:
        return 0
    ordered = sorted(set(trading_dates))
    positions = {value: index for index, value in enumerate(ordered)}
    if not ordered or any(value not in positions for value in effective_dates):
        return None
    indexes = [positions[value] for value in effective_dates]
    return max(indexes) - min(indexes)
