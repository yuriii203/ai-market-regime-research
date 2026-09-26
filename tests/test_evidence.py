from __future__ import annotations

from datetime import date, timedelta

from src.analysis.evidence import (
    EvidenceDirection,
    EvidenceType,
    build_evidence_bundle,
    evaluate_sentiment_evidence,
)
from src.clients.fuyao import (
    DataResult,
    PoolItem,
    PoolResult,
    PriceBar,
    ResponseMetadata,
    StockSnapshot,
    ValuationSnapshot,
    shanghai_date_to_ms,
)
from src.indicators.trend import calculate_trend_indicators


def _metadata(request_id: str, day: date) -> ResponseMetadata:
    return ResponseMetadata(
        request_id=request_id,
        timestamp_ms=shanghai_date_to_ms(day),
        code=0,
        message="success",
    )


def _bars(start: date, daily_change: float) -> list[PriceBar]:
    return [
        PriceBar(
            date_ms=shanghai_date_to_ms(start + timedelta(days=index)),
            open_price=100 + index * daily_change,
            high_price=101 + index * daily_change,
            low_price=99 + index * daily_change,
            close_price=100 + index * daily_change,
            volume=1000,
            turnover=1_000_000 + index * 1000,
        )
        for index in range(70)
    ]


def _pool(
    day: date,
    request_id: str,
    items: list[PoolItem],
) -> PoolResult:
    return PoolResult(
        metadata=_metadata(request_id, day),
        total=len(items),
        items=items,
        request_ids=(request_id,),
    )


def _complete_bundle():
    start = date(2026, 1, 1)
    trend_bars = _bars(start, -0.1)
    as_of = trend_bars[-1].trading_date
    indicators = calculate_trend_indicators(trend_bars)
    style_histories = {
        "大盘": DataResult(_metadata("style-large", as_of), _bars(start, -0.05)),
        "小盘": DataResult(_metadata("style-small", as_of), _bars(start, -0.2)),
        "成长": DataResult(_metadata("style-growth", as_of), _bars(start, -0.3)),
    }
    snapshots = [
        StockSnapshot("600000.SH", 10, 1.0, 100),
        StockSnapshot("600001.SH", 10, -2.0, 100),
        StockSnapshot("600002.SH", 10, -1.0, 100),
        StockSnapshot("600003.SH", 10, None, 100),
    ]
    valuations = [
        ValuationSnapshot("600000.SH", "甲", 10, 1),
        ValuationSnapshot("600001.SH", "乙", 20, 2),
        ValuationSnapshot("600002.SH", "丙", -5, None),
    ]
    pools = (
        _pool(
            as_of,
            "up",
            [
                PoolItem("600000.SH", "甲", continue_day_cnt=2),
                PoolItem("000001.SZ", "非成分", continue_day_cnt=1),
            ],
        ),
        _pool(as_of, "down", [PoolItem("600001.SH", "乙")]),
        _pool(as_of, "break", [PoolItem("600002.SH", "丙", open_times=2)]),
    )
    return build_evidence_bundle(
        subject_symbol="000300.SH",
        subject_name="沪深300",
        indicators=indicators,
        trend_metadata=_metadata("trend", as_of),
        style_histories=style_histories,
        breadth_snapshots=snapshots,
        breadth_metadata=(_metadata("breadth", as_of + timedelta(days=1)),),
        breadth_universe_count=300,
        breadth_sampled_count=4,
        sentiment_pools=pools,
        sentiment_query_date=as_of.isoformat(),
        sentiment_universe_codes={"600000.SH", "600001.SH", "600002.SH"},
        valuations=valuations,
        valuation_metadata=(_metadata("valuation", as_of + timedelta(days=1)),),
        valuation_universe_count=300,
        valuation_sampled_count=3,
        breadth_effective_date=(as_of + timedelta(days=1)).isoformat(),
        valuation_effective_date=(as_of + timedelta(days=1)).isoformat(),
        trading_dates=(
            as_of.isoformat(),
            (as_of + timedelta(days=1)).isoformat(),
        ),
    )


def test_bundle_is_deterministic_and_has_stable_unique_evidence_ids() -> None:
    first = _complete_bundle()
    second = _complete_bundle()

    assert first.to_json() == second.to_json()
    assert [item.dimension for item in first.dimensions] == [
        "trend",
        "breadth",
        "style",
        "liquidity",
        "sentiment",
        "valuation",
    ]
    ids = [item.id for item in first.evidence]
    assert len(ids) == len(set(ids))
    assert all(item.scope for item in first.evidence)
    assert all(
        item.request_ids and item.effective_date and item.endpoint
        for item in first.evidence
        if item.evidence_type == EvidenceType.FACT
    )
    assert all(
        item.retrieved_at
        for item in first.evidence
        if item.evidence_type == EvidenceType.FACT
    )


def test_missing_dimension_becomes_explicit_uncertainty() -> None:
    bundle = _complete_bundle()
    rebuilt = build_evidence_bundle(
        subject_symbol=bundle.subject_symbol,
        subject_name=bundle.subject_name,
        indicators=calculate_trend_indicators(_bars(date(2026, 1, 1), -0.1)),
        trend_metadata=_metadata("trend", date(2026, 3, 11)),
        errors={"breadth": "接口超时"},
    )

    breadth = next(item for item in rebuilt.dimensions if item.dimension == "breadth")
    uncertainty = next(
        item for item in rebuilt.evidence if item.id == "E-BREADTH-U01"
    )
    assert breadth.available is False
    assert "breadth" in rebuilt.missing_dimensions
    assert uncertainty.evidence_type == EvidenceType.UNCERTAINTY
    assert "接口超时" in uncertainty.description


def test_context_dimensions_are_not_forced_into_market_direction() -> None:
    bundle = _complete_bundle()
    by_dimension = {item.dimension: item for item in bundle.dimensions}

    assert by_dimension["liquidity"].direction == EvidenceDirection.CONTEXT
    assert by_dimension["style"].direction == EvidenceDirection.CONTEXT
    assert by_dimension["valuation"].direction == EvidenceDirection.CONTEXT
    assert "不能据此判断高估或低估" in by_dimension["valuation"].summary


def test_cross_date_inputs_are_recorded_as_time_conflict() -> None:
    bundle = _complete_bundle()

    assert bundle.time_conflicts
    assert "相差1个交易日" in bundle.time_conflicts[0]


def test_selected_index_sentiment_votes_while_full_market_is_context() -> None:
    day = date(2026, 3, 11)
    up_items = [
        PoolItem(f"6000{index:02d}.SH", f"上涨{index}") for index in range(10)
    ]
    selected_down = [
        PoolItem(f"6010{index:02d}.SH", f"下跌{index}") for index in range(3)
    ]
    result = evaluate_sentiment_evidence(
        (
            _pool(day, "up", up_items),
            _pool(day, "down", selected_down),
            _pool(day, "break", []),
        ),
        day.isoformat(),
        {item.thscode for item in selected_down},
        "沪深300",
    )

    assert result.assessment.direction == EvidenceDirection.NEGATIVE
    assert result.assessment.label.startswith("沪深300偏弱")
    assert "全A偏活跃（背景）" in result.assessment.label
    assert "作为本次情绪方向投票" in result.assessment.summary
