from __future__ import annotations

from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.analysis.trend_state import evaluate_trend_state
from src.clients.fuyao import FuyaoClient, SHANGHAI_TZ, shanghai_date_to_ms
from src.indicators.trend import calculate_trend_indicators


INDEXES = ("000001.SH", "000300.SH", "000852.SH", "399006.SZ")


def main() -> None:
    load_dotenv()
    client = FuyaoClient.from_env()
    end_date = datetime.now(SHANGHAI_TZ).date()
    start_date = end_date - timedelta(days=200)
    start_ms = shanghai_date_to_ms(start_date)
    end_ms = shanghai_date_to_ms(end_date)

    for symbol in INDEXES:
        history = client.get_index_history(
            symbol, start_ms=start_ms, end_ms=end_ms
        )
        indicators = calculate_trend_indicators(history.items)
        state = evaluate_trend_state(indicators)
        print(
            f"SYMBOL={symbol} STATUS=OK ITEMS={len(history.items)} "
            f"LAST_DATE={history.items[-1].trading_date.isoformat()} "
            f"STATE_SCORE={state.score} REQUEST_ID_PRESENT={bool(history.metadata.request_id)}"
        )


if __name__ == "__main__":
    main()

