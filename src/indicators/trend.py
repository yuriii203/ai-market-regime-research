from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import fmean
from typing import Sequence

from src.clients.fuyao import PriceBar


class IndicatorDataError(ValueError):
    """The supplied market data cannot support a reliable calculation."""


@dataclass(frozen=True)
class TrendIndicators:
    as_of: date
    close_price: float
    return_20d_pct: float
    return_60d_pct: float
    ma20: float
    ma60: float
    close_vs_ma20_pct: float
    close_vs_ma60_pct: float
    max_drawdown_60d_pct: float
    turnover_5d_avg: float | None
    turnover_20d_avg: float | None
    turnover_5d_vs_20d_pct: float | None
    warnings: tuple[str, ...]


def calculate_trend_indicators(bars: Sequence[PriceBar]) -> TrendIndicators:
    """Calculate deterministic trend and liquidity indicators.

    Return calculations need 61 closes: the latest close and the close 60
    trading intervals earlier. Moving averages use the latest 20/60 bars.
    """
    ordered = sorted(bars, key=lambda bar: bar.date_ms)
    if len(ordered) < 61:
        raise IndicatorDataError(
            f"至少需要61根日K才能计算60日收益率，当前只有{len(ordered)}根"
        )
    if len({bar.date_ms for bar in ordered}) != len(ordered):
        raise IndicatorDataError("日K数据包含重复交易日")
    if any(bar.close_price <= 0 for bar in ordered):
        raise IndicatorDataError("收盘价包含非正数，无法计算收益率和回撤")

    latest = ordered[-1]
    closes_20 = [bar.close_price for bar in ordered[-20:]]
    closes_60 = [bar.close_price for bar in ordered[-60:]]
    ma20 = fmean(closes_20)
    ma60 = fmean(closes_60)

    turnover_5d_avg: float | None = None
    turnover_20d_avg: float | None = None
    turnover_change_pct: float | None = None
    warnings: list[str] = []
    turnover_window = [bar.turnover for bar in ordered[-20:]]
    if any(value is None for value in turnover_window):
        warnings.append("近20日成交额存在缺失，成交额变化指标不可用")
    else:
        turnover_values = [float(value) for value in turnover_window if value is not None]
        if any(value < 0 for value in turnover_values):
            warnings.append("近20日成交额包含负数，成交额变化指标不可用")
        else:
            turnover_20d_avg = fmean(turnover_values)
            turnover_5d_avg = fmean(turnover_values[-5:])
            if turnover_20d_avg == 0:
                warnings.append("近20日平均成交额为0，成交额变化指标不可用")
            else:
                turnover_change_pct = (
                    turnover_5d_avg / turnover_20d_avg - 1
                ) * 100

    return TrendIndicators(
        as_of=latest.trading_date,
        close_price=latest.close_price,
        return_20d_pct=_period_return_pct(latest.close_price, ordered[-21].close_price),
        return_60d_pct=_period_return_pct(latest.close_price, ordered[-61].close_price),
        ma20=ma20,
        ma60=ma60,
        close_vs_ma20_pct=_period_return_pct(latest.close_price, ma20),
        close_vs_ma60_pct=_period_return_pct(latest.close_price, ma60),
        max_drawdown_60d_pct=calculate_max_drawdown_pct(closes_60),
        turnover_5d_avg=turnover_5d_avg,
        turnover_20d_avg=turnover_20d_avg,
        turnover_5d_vs_20d_pct=turnover_change_pct,
        warnings=tuple(warnings),
    )


def calculate_max_drawdown_pct(prices: Sequence[float]) -> float:
    """Return maximum peak-to-subsequent-trough loss as a positive percent."""
    if not prices:
        raise IndicatorDataError("最大回撤计算至少需要一个价格")
    if any(price <= 0 for price in prices):
        raise IndicatorDataError("最大回撤价格序列包含非正数")

    peak = float(prices[0])
    worst_drawdown = 0.0
    for price in prices:
        peak = max(peak, float(price))
        drawdown = float(price) / peak - 1
        worst_drawdown = min(worst_drawdown, drawdown)
    return abs(worst_drawdown) * 100


def _period_return_pct(latest: float, previous: float) -> float:
    return (latest / previous - 1) * 100

