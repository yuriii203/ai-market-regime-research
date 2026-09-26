from __future__ import annotations

from datetime import date

from src.clients.fuyao import (
    PoolItem,
    PoolResult,
    PriceBar,
    ResponseMetadata,
    StockSnapshot,
    ValuationSnapshot,
)
from src.indicators.market_dimensions import (
    calculate_market_dimensions,
    calculate_market_sentiment,
)
from src.indicators.trend import calculate_trend_indicators


def _bars(multiplier: float) -> list[PriceBar]:
    return [
        PriceBar(
            date_ms=1_700_000_000_000 + day * 86_400_000,
            open_price=100 + day * multiplier,
            high_price=101 + day * multiplier,
            low_price=99 + day * multiplier,
            close_price=100 + day * multiplier,
            volume=1000,
            turnover=1_000_000 + day * 1000,
        )
        for day in range(70)
    ]


def _pool(total: int, consecutive: int | None = None) -> PoolResult:
    metadata = ResponseMetadata("request", 1_700_000_000_000, 0, "success")
    items = (
        [PoolItem("000001.SZ", "示例", continue_day_cnt=consecutive)]
        if consecutive is not None
        else []
    )
    return PoolResult(metadata=metadata, total=total, items=items)


def test_market_dimensions_use_explicit_and_auditable_formulas() -> None:
    histories = {
        "大盘": _bars(0.2),
        "小盘": _bars(0.5),
        "成长": _bars(-0.1),
    }
    selected = calculate_trend_indicators(histories["大盘"])
    snapshots = [
        StockSnapshot("000001.SZ", 10, 1.0, 100),
        StockSnapshot("000002.SZ", 10, -2.0, 100),
        StockSnapshot("000003.SZ", 10, 0.0, 100),
        StockSnapshot("000004.SZ", 10, 3.0, 100),
    ]
    valuations = [
        ValuationSnapshot("000001.SZ", "甲", 10, 1),
        ValuationSnapshot("000002.SZ", "乙", 20, 2),
        ValuationSnapshot("000003.SZ", "丙", -5, None),
    ]

    result = calculate_market_dimensions(
        style_histories=histories,
        selected_indicators=selected,
        snapshots=snapshots,
        valuations=valuations,
        limit_up=_pool(8, 4),
        limit_down=_pool(2),
        limit_break=_pool(2),
        universe_count=10,
    )

    assert result.style.leader == "小盘"
    assert result.style.laggard == "成长"
    assert result.breadth.advancing == 2
    assert result.breadth.declining == 1
    assert result.breadth.flat == 1
    assert result.breadth.breadth_score_pct == 25
    assert result.sentiment.break_rate_pct == 20
    assert result.sentiment.max_consecutive_limit_up == 4
    assert result.valuation.median_pe_ttm == 15
    assert result.valuation.median_pb_mrq == 1.5
    assert result.valuation.coverage_pct == 30


def test_sentiment_can_be_filtered_to_selected_index_constituents() -> None:
    metadata = ResponseMetadata("request", 1_700_000_000_000, 0, "success")
    limit_up = PoolResult(
        metadata,
        3,
        [
            PoolItem("000001.SZ", "甲", continue_day_cnt=2),
            PoolItem("600000.SH", "乙", continue_day_cnt=4),
            PoolItem("300001.SZ", "丙", continue_day_cnt=1),
        ],
    )
    limit_down = PoolResult(
        metadata, 1, [PoolItem("000002.SZ", "丁")]
    )
    limit_break = PoolResult(
        metadata, 1, [PoolItem("600001.SH", "戊", open_times=2)]
    )

    full = calculate_market_sentiment(limit_up, limit_down, limit_break)
    selected = calculate_market_sentiment(
        limit_up,
        limit_down,
        limit_break,
        universe_codes={"600000.SH", "600001.SH"},
    )

    assert full.limit_up_count == 3
    assert selected.limit_up_count == 1
    assert selected.limit_down_count == 0
    assert selected.limit_break_count == 1
    assert selected.max_consecutive_limit_up == 4
    assert selected.break_rate_pct == 50
