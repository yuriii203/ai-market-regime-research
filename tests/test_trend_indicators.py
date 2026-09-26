from __future__ import annotations

from datetime import date

import pytest

from src.clients.fuyao import PriceBar, shanghai_date_to_ms
from src.indicators.trend import (
    IndicatorDataError,
    calculate_max_drawdown_pct,
    calculate_trend_indicators,
)


DAY_MS = 86_400_000


def make_bars(
    closes: list[float], turnovers: list[float | None] | None = None
) -> list[PriceBar]:
    if turnovers is None:
        turnovers = [1_000.0] * len(closes)
    start_ms = shanghai_date_to_ms(date(2026, 1, 1))
    return [
        PriceBar(
            date_ms=start_ms + index * DAY_MS,
            open_price=close,
            high_price=close,
            low_price=close,
            close_price=close,
            volume=100.0,
            turnover=turnovers[index],
        )
        for index, close in enumerate(closes)
    ]


def test_calculates_returns_moving_averages_and_zero_drawdown() -> None:
    closes = [100.0 + index for index in range(61)]
    result = calculate_trend_indicators(make_bars(closes))

    assert result.return_20d_pct == pytest.approx((160 / 140 - 1) * 100)
    assert result.return_60d_pct == pytest.approx(60.0)
    assert result.ma20 == pytest.approx(150.5)
    assert result.ma60 == pytest.approx(130.5)
    assert result.close_vs_ma20_pct == pytest.approx((160 / 150.5 - 1) * 100)
    assert result.close_vs_ma60_pct == pytest.approx((160 / 130.5 - 1) * 100)
    assert result.max_drawdown_60d_pct == pytest.approx(0.0)


def test_max_drawdown_uses_peak_before_trough() -> None:
    assert calculate_max_drawdown_pct([100.0, 120.0, 90.0, 110.0]) == pytest.approx(
        25.0
    )


def test_turnover_change_compares_five_day_and_twenty_day_averages() -> None:
    closes = [100.0] * 61
    turnovers = [1_000.0] * 56 + [2_000.0] * 5

    result = calculate_trend_indicators(make_bars(closes, turnovers))

    assert result.turnover_5d_avg == pytest.approx(2_000.0)
    assert result.turnover_20d_avg == pytest.approx(1_250.0)
    assert result.turnover_5d_vs_20d_pct == pytest.approx(60.0)
    assert result.warnings == ()


def test_missing_turnover_is_explicitly_marked_unavailable() -> None:
    closes = [100.0] * 61
    turnovers: list[float | None] = [1_000.0] * 61
    turnovers[-3] = None

    result = calculate_trend_indicators(make_bars(closes, turnovers))

    assert result.turnover_5d_avg is None
    assert result.turnover_20d_avg is None
    assert result.turnover_5d_vs_20d_pct is None
    assert "成交额存在缺失" in result.warnings[0]


def test_bars_are_sorted_before_calculation() -> None:
    bars = make_bars([100.0 + index for index in range(61)])

    result = calculate_trend_indicators(list(reversed(bars)))

    assert result.close_price == 160.0


def test_insufficient_bars_fail_loudly() -> None:
    with pytest.raises(IndicatorDataError, match="至少需要61根"):
        calculate_trend_indicators(make_bars([100.0] * 60))


def test_duplicate_trading_dates_fail_loudly() -> None:
    bars = make_bars([100.0] * 61)
    bars[-1] = PriceBar(
        date_ms=bars[-2].date_ms,
        open_price=100.0,
        high_price=100.0,
        low_price=100.0,
        close_price=100.0,
        volume=100.0,
        turnover=1_000.0,
    )

    with pytest.raises(IndicatorDataError, match="重复交易日"):
        calculate_trend_indicators(bars)


def test_non_positive_close_fails_loudly() -> None:
    closes = [100.0] * 60 + [0.0]

    with pytest.raises(IndicatorDataError, match="非正数"):
        calculate_trend_indicators(make_bars(closes))

