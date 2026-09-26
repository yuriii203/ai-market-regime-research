from __future__ import annotations

from dataclasses import dataclass

from src.indicators.trend import TrendIndicators


@dataclass(frozen=True)
class TrendState:
    label: str
    score: int
    tone: str
    summary: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    risk_context: tuple[str, ...]
    switch_conditions: tuple[str, ...]


def evaluate_trend_state(indicators: TrendIndicators) -> TrendState:
    """Map four transparent directional signals to a trend-only state."""
    signals = (
        (indicators.return_20d_pct >= 0, f"20日收益率 {indicators.return_20d_pct:+.2f}%"),
        (indicators.return_60d_pct >= 0, f"60日收益率 {indicators.return_60d_pct:+.2f}%"),
        (
            indicators.close_vs_ma20_pct >= 0,
            f"收盘价相对MA20 {indicators.close_vs_ma20_pct:+.2f}%",
        ),
        (
            indicators.close_vs_ma60_pct >= 0,
            f"收盘价相对MA60 {indicators.close_vs_ma60_pct:+.2f}%",
        ),
    )
    score = sum(1 if positive else -1 for positive, _ in signals)
    supporting = tuple(text for positive, text in signals if positive)
    contradicting = tuple(text for positive, text in signals if not positive)

    if score == 4:
        label, tone = "趋势偏强", "positive"
        summary = "短中期收益与均线位置均为正，趋势证据方向一致。"
    elif score == 2:
        label, tone = "震荡偏强", "positive"
        summary = "多数趋势证据偏正，但仍有一项信号尚未确认。"
    elif score == 0:
        label, tone = "方向分化", "neutral"
        summary = "正负趋势证据数量相同，暂不支持单一方向结论。"
    elif score == -2:
        label, tone = "震荡偏弱", "negative"
        summary = "多数趋势证据偏负，但仍有一项信号保持正向。"
    else:
        label, tone = "趋势偏弱", "negative"
        summary = "短中期收益与均线位置均为负，趋势证据方向一致。"

    risk_context: list[str] = [
        f"近60日最大回撤 {indicators.max_drawdown_60d_pct:.2f}%"
    ]
    if indicators.turnover_5d_vs_20d_pct is None:
        risk_context.append("成交额变化不可用")
    elif indicators.turnover_5d_vs_20d_pct >= 20:
        risk_context.append(
            f"近5日平均成交额较20日均值放大 {indicators.turnover_5d_vs_20d_pct:.2f}%"
        )
    elif indicators.turnover_5d_vs_20d_pct <= -20:
        risk_context.append(
            f"近5日平均成交额较20日均值收缩 {abs(indicators.turnover_5d_vs_20d_pct):.2f}%"
        )
    else:
        risk_context.append(
            f"近5日平均成交额相对20日均值变化 {indicators.turnover_5d_vs_20d_pct:+.2f}%"
        )

    switch_conditions = _switch_conditions(signals)
    return TrendState(
        label=label,
        score=score,
        tone=tone,
        summary=summary,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        risk_context=tuple(risk_context),
        switch_conditions=switch_conditions,
    )


def _switch_conditions(
    signals: tuple[tuple[bool, str], ...],
) -> tuple[str, ...]:
    positive_count = sum(positive for positive, _ in signals)
    if positive_count >= 3:
        return (
            "若20日收益率转负且收盘价跌破MA20，短期状态将转弱。",
            "若60日收益率转负且收盘价跌破MA60，中期状态将转弱。",
        )
    if positive_count <= 1:
        return (
            "若20日收益率转正且收盘价重新站上MA20，短期状态将改善。",
            "若60日收益率转正且收盘价重新站上MA60，中期状态将改善。",
        )
    return (
        "若正向信号增加至3项以上，状态将转为偏强。",
        "若负向信号增加至3项以上，状态将转为偏弱。",
    )

