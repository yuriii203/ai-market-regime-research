from datetime import date

import pytest

from src.analysis.trend_state import evaluate_trend_state
from src.indicators.trend import TrendIndicators


def indicators_with(
    *,
    return_20d: float,
    return_60d: float,
    close_vs_ma20: float,
    close_vs_ma60: float,
) -> TrendIndicators:
    return TrendIndicators(
        as_of=date(2026, 9, 24),
        close_price=4_500.0,
        return_20d_pct=return_20d,
        return_60d_pct=return_60d,
        ma20=4_450.0,
        ma60=4_400.0,
        close_vs_ma20_pct=close_vs_ma20,
        close_vs_ma60_pct=close_vs_ma60,
        max_drawdown_60d_pct=8.0,
        turnover_5d_avg=100.0,
        turnover_20d_avg=100.0,
        turnover_5d_vs_20d_pct=0.0,
        warnings=(),
    )


@pytest.mark.parametrize(
    ("values", "label", "score"),
    [
        ((1.0, 1.0, 1.0, 1.0), "趋势偏强", 4),
        ((1.0, 1.0, 1.0, -1.0), "震荡偏强", 2),
        ((1.0, 1.0, -1.0, -1.0), "方向分化", 0),
        ((1.0, -1.0, -1.0, -1.0), "震荡偏弱", -2),
        ((-1.0, -1.0, -1.0, -1.0), "趋势偏弱", -4),
    ],
)
def test_state_mapping_is_deterministic(
    values: tuple[float, float, float, float], label: str, score: int
) -> None:
    state = evaluate_trend_state(
        indicators_with(
            return_20d=values[0],
            return_60d=values[1],
            close_vs_ma20=values[2],
            close_vs_ma60=values[3],
        )
    )

    assert state.label == label
    assert state.score == score


def test_negative_state_explains_improvement_conditions() -> None:
    state = evaluate_trend_state(
        indicators_with(
            return_20d=-1.0,
            return_60d=-2.0,
            close_vs_ma20=-1.0,
            close_vs_ma60=-2.0,
        )
    )

    assert len(state.contradicting_evidence) == 4
    assert any("重新站上MA20" in item for item in state.switch_conditions)

