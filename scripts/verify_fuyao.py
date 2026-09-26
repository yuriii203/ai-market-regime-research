from __future__ import annotations

from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.analysis.trend_state import evaluate_trend_state
from src.clients.fuyao import FuyaoClient, SHANGHAI_TZ, shanghai_date_to_ms
from src.indicators.trend import calculate_trend_indicators


def main() -> None:
    load_dotenv()
    client = FuyaoClient.from_env()

    calendar = client.get_trading_days()
    end_date = datetime.now(SHANGHAI_TZ).date()
    start_date = end_date - timedelta(days=200)
    history = client.get_index_history(
        "000300.SH",
        start_ms=shanghai_date_to_ms(start_date),
        end_ms=shanghai_date_to_ms(end_date),
    )
    indicators = calculate_trend_indicators(history.items)
    state = evaluate_trend_state(indicators)

    print("CALENDAR_STATUS=OK")
    print(f"CALENDAR_ITEMS={len(calendar.items)}")
    print(f"CALENDAR_RANGE={calendar.items[0].date}..{calendar.items[-1].date}")
    print(f"CALENDAR_TIMESTAMP={calendar.metadata.timestamp.isoformat()}")
    print(f"CALENDAR_REQUEST_ID={calendar.metadata.request_id}")
    print("HISTORY_STATUS=OK")
    print(f"HISTORY_SYMBOL=000300.SH")
    print(f"HISTORY_ITEMS={len(history.items)}")
    print(
        "HISTORY_RANGE="
        f"{history.items[0].trading_date.isoformat()}.."
        f"{history.items[-1].trading_date.isoformat()}"
    )
    print(f"HISTORY_TIMESTAMP={history.metadata.timestamp.isoformat()}")
    print(f"HISTORY_REQUEST_ID={history.metadata.request_id}")
    print("INDICATORS_STATUS=OK")
    print(f"RETURN_20D_PCT={indicators.return_20d_pct:.6f}")
    print(f"RETURN_60D_PCT={indicators.return_60d_pct:.6f}")
    print(f"MA20={indicators.ma20:.6f}")
    print(f"MA60={indicators.ma60:.6f}")
    print(f"MAX_DRAWDOWN_60D_PCT={indicators.max_drawdown_60d_pct:.6f}")
    print(
        "TURNOVER_5D_VS_20D_PCT="
        f"{indicators.turnover_5d_vs_20d_pct:.6f}"
        if indicators.turnover_5d_vs_20d_pct is not None
        else "TURNOVER_5D_VS_20D_PCT=UNAVAILABLE"
    )
    print(f"TREND_STATE={state.label}")
    print(f"TREND_SCORE={state.score}")
    print(f"DATA_DATES_MATCH={history.items[-1].trading_date.strftime('%Y%m%d') == calendar.items[-1].date}")


if __name__ == "__main__":
    main()
