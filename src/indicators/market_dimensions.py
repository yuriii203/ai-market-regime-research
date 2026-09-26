from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from src.clients.fuyao import PoolResult, PriceBar, StockSnapshot, ValuationSnapshot
from src.indicators.trend import TrendIndicators, calculate_trend_indicators


@dataclass(frozen=True)
class StyleRow:
    name: str
    return_20d_pct: float
    return_60d_pct: float


@dataclass(frozen=True)
class StyleRotation:
    rows: tuple[StyleRow, ...]
    leader: str
    laggard: str
    spread_20d_pct_points: float


@dataclass(frozen=True)
class MarketBreadth:
    advancing: int
    declining: int
    flat: int
    valid_count: int
    universe_count: int
    breadth_score_pct: float

    @property
    def coverage_pct(self) -> float:
        if self.universe_count == 0:
            return 0.0
        return self.valid_count / self.universe_count * 100


@dataclass(frozen=True)
class Liquidity:
    average_turnover_20d: float | None
    change_5d_vs_20d_pct: float | None


@dataclass(frozen=True)
class MarketSentiment:
    limit_up_count: int
    limit_down_count: int
    limit_break_count: int
    max_consecutive_limit_up: int | None
    break_rate_pct: float | None


@dataclass(frozen=True)
class MarketValuation:
    median_pe_ttm: float | None
    median_pb_mrq: float | None
    pe_valid_count: int
    pb_valid_count: int
    sample_count: int
    universe_count: int

    @property
    def coverage_pct(self) -> float:
        if self.universe_count == 0:
            return 0.0
        return self.sample_count / self.universe_count * 100


@dataclass(frozen=True)
class MarketDimensions:
    style: StyleRotation
    breadth: MarketBreadth
    liquidity: Liquidity
    sentiment: MarketSentiment
    valuation: MarketValuation


def calculate_market_dimensions(
    *,
    style_histories: dict[str, list[PriceBar]],
    selected_indicators: TrendIndicators,
    snapshots: list[StockSnapshot],
    valuations: list[ValuationSnapshot],
    limit_up: PoolResult,
    limit_down: PoolResult,
    limit_break: PoolResult,
    universe_count: int,
) -> MarketDimensions:
    return MarketDimensions(
        style=calculate_style_rotation(style_histories),
        breadth=calculate_market_breadth(snapshots, universe_count),
        liquidity=calculate_liquidity(selected_indicators),
        sentiment=calculate_market_sentiment(
            limit_up, limit_down, limit_break
        ),
        valuation=calculate_market_valuation(valuations, universe_count),
    )


def calculate_style_rotation(
    style_histories: dict[str, list[PriceBar]],
) -> StyleRotation:
    style_rows = tuple(
        StyleRow(
            name=name,
            return_20d_pct=indicators.return_20d_pct,
            return_60d_pct=indicators.return_60d_pct,
        )
        for name, history in style_histories.items()
        for indicators in [calculate_trend_indicators(history)]
    )
    if not style_rows:
        raise ValueError("风格轮动至少需要一个指数序列")
    ranked = sorted(style_rows, key=lambda row: row.return_20d_pct, reverse=True)
    return StyleRotation(
        rows=tuple(ranked),
        leader=ranked[0].name,
        laggard=ranked[-1].name,
        spread_20d_pct_points=(
            ranked[0].return_20d_pct - ranked[-1].return_20d_pct
        ),
    )



def calculate_market_breadth(
    snapshots: list[StockSnapshot], universe_count: int
) -> MarketBreadth:
    breadth_values = [
        item.price_change_ratio_pct
        for item in snapshots
        if item.price_change_ratio_pct is not None
    ]
    advancing = sum(value > 0 for value in breadth_values)
    declining = sum(value < 0 for value in breadth_values)
    flat = len(breadth_values) - advancing - declining
    breadth_score = (
        (advancing - declining) / len(breadth_values) * 100
        if breadth_values
        else 0.0
    )
    return MarketBreadth(
        advancing=advancing,
        declining=declining,
        flat=flat,
        valid_count=len(breadth_values),
        universe_count=universe_count,
        breadth_score_pct=breadth_score,
    )



def calculate_liquidity(selected_indicators: TrendIndicators) -> Liquidity:
    return Liquidity(
        average_turnover_20d=selected_indicators.turnover_20d_avg,
        change_5d_vs_20d_pct=selected_indicators.turnover_5d_vs_20d_pct,
    )


def calculate_market_sentiment(
    limit_up: PoolResult,
    limit_down: PoolResult,
    limit_break: PoolResult,
    *,
    universe_codes: set[str] | None = None,
) -> MarketSentiment:
    if universe_codes is None:
        up_items = limit_up.items
        down_items = limit_down.items
        break_items = limit_break.items
        up_count = limit_up.total
        down_count = limit_down.total
        break_count = limit_break.total
    else:
        up_items = [item for item in limit_up.items if item.thscode in universe_codes]
        down_items = [
            item for item in limit_down.items if item.thscode in universe_codes
        ]
        break_items = [
            item for item in limit_break.items if item.thscode in universe_codes
        ]
        up_count = len(up_items)
        down_count = len(down_items)
        break_count = len(break_items)

    consecutive_values = [
        item.continue_day_cnt
        for item in up_items
        if item.continue_day_cnt is not None
    ]
    attempted_limit_up = up_count + break_count
    return MarketSentiment(
        limit_up_count=up_count,
        limit_down_count=down_count,
        limit_break_count=break_count,
        max_consecutive_limit_up=(
            max(consecutive_values) if consecutive_values else None
        ),
        break_rate_pct=(
            break_count / attempted_limit_up * 100
            if attempted_limit_up > 0
            else None
        ),
    )



def calculate_market_valuation(
    valuations: list[ValuationSnapshot], universe_count: int
) -> MarketValuation:
    pe_values = [
        item.pe_ttm
        for item in valuations
        if item.pe_ttm is not None and item.pe_ttm > 0
    ]
    pb_values = [
        item.pb_mrq
        for item in valuations
        if item.pb_mrq is not None and item.pb_mrq > 0
    ]
    return MarketValuation(
        median_pe_ttm=median(pe_values) if pe_values else None,
        median_pb_mrq=median(pb_values) if pb_values else None,
        pe_valid_count=len(pe_values),
        pb_valid_count=len(pb_values),
        sample_count=len(valuations),
        universe_count=universe_count,
    )
