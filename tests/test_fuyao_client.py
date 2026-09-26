from __future__ import annotations

from typing import Any

import pytest
import requests

from src.clients.fuyao import (
    FuyaoBusinessError,
    FuyaoClient,
    FuyaoEmptyDataError,
    FuyaoTimeoutError,
    is_a_share_thscode,
)


class FakeResponse:
    def __init__(
        self, payload: Any, *, status_code: int = 200, json_error: bool = False
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def json(self) -> Any:
        if self.json_error:
            raise ValueError("invalid json")
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.response = response
        self.error = error
        self.last_call: dict[str, Any] | None = None

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.last_call = {"url": url, **kwargs}
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def make_client(session: FakeSession) -> FuyaoClient:
    return FuyaoClient("test-key", session=session)  # type: ignore[arg-type]


def test_trading_days_preserves_metadata_and_items() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 0,
                "message": "success",
                "request_id": "calendar-request",
                "data": {
                    "timestamp": 1748275200000,
                    "item": [
                        {"date_ms": 1747929600000, "date": "20250523"},
                        {"date_ms": 1748188800000, "date": "20250526"},
                    ],
                },
            }
        )
    )

    result = make_client(session).get_trading_days()

    assert result.metadata.request_id == "calendar-request"
    assert result.metadata.timestamp_ms == 1748275200000
    assert result.metadata.received_at is not None
    assert result.metadata.retrieval_time == result.metadata.received_at
    assert [item.date for item in result.items] == ["20250523", "20250526"]
    assert session.last_call is not None
    assert session.last_call["url"].endswith("/api/a-share/calendar/trading-days")


def test_index_history_parses_price_bars_and_parameters() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 0,
                "message": "success",
                "request_id": "history-request",
                "data": {
                    "timestamp": 1747584000000,
                    "adjust": None,
                    "item": [
                        {
                            "date_ms": 1716134400000,
                            "open_price": 3108.22,
                            "high_price": 3125.74,
                            "low_price": 3101.15,
                            "close_price": 3120.68,
                            "volume": 281000000,
                            "turnover": 360000000000,
                        }
                    ],
                },
            }
        )
    )

    result = make_client(session).get_index_history(
        "000300.sh", start_ms=1716105600000, end_ms=1747641600000
    )

    assert result.metadata.request_id == "history-request"
    assert result.items[0].close_price == 3120.68
    assert session.last_call is not None
    assert session.last_call["params"]["thscode"] == "000300.SH"
    assert session.last_call["params"]["interval"] == "1d"


def test_timeout_becomes_specific_error() -> None:
    session = FakeSession(error=requests.Timeout("too slow"))

    with pytest.raises(FuyaoTimeoutError, match="请求超时"):
        make_client(session).get_trading_days()


def test_empty_items_are_not_silently_accepted() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 0,
                "message": "success",
                "request_id": "empty-request",
                "data": {"timestamp": 1748275200000, "item": []},
            }
        )
    )

    with pytest.raises(FuyaoEmptyDataError) as captured:
        make_client(session).get_trading_days()

    assert captured.value.request_id == "empty-request"


def test_business_error_keeps_code_and_request_id() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 2003,
                "message": "capability denied",
                "request_id": "denied-request",
                "data": None,
            }
        )
    )

    with pytest.raises(FuyaoBusinessError) as captured:
        make_client(session).get_trading_days()

    assert captured.value.code == 2003
    assert captured.value.request_id == "denied-request"
    assert "权限不足" in str(captured.value)


def test_http_429_is_normalized_to_rate_limit_error() -> None:
    session = FakeSession(FakeResponse({}, status_code=429))

    with pytest.raises(FuyaoBusinessError) as captured:
        make_client(session).get_trading_days()

    assert captured.value.code == 4001


def test_stock_snapshot_parses_breadth_fields() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 0,
                "message": "success",
                "request_id": "snapshot-request",
                "data": {
                    "timestamp": 1748275200000,
                    "total": 1,
                    "item": [
                        {
                            "thscode": "600519.SH",
                            "last_price": 1500,
                            "price_change_ratio_pct": 1.25,
                            "turnover": 123000000,
                        }
                    ],
                },
            }
        )
    )

    result = make_client(session).get_stock_snapshots(["600519.sh"])

    assert result.items[0].price_change_ratio_pct == 1.25
    assert session.last_call is not None
    assert session.last_call["params"]["thscodes"] == "600519.SH"


def test_empty_limit_pool_is_valid_sentiment_fact() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "code": 0,
                "message": "success",
                "request_id": "pool-request",
                "data": {
                    "timestamp": 1748275200000,
                    "pagination": {"total": 0, "pages": 0, "size": 200, "page": 1},
                    "item": [],
                },
            }
        )
    )

    result = make_client(session).get_limit_pool("up", date_ms=1748275200000)

    assert result.total == 0
    assert result.items == []


def test_snapshot_batch_limit_fails_before_network() -> None:
    session = FakeSession(FakeResponse({}))

    with pytest.raises(ValueError, match="最多100"):
        make_client(session).get_stock_snapshots(
            [f"{value:06d}.SZ" for value in range(101)]
        )

    assert session.last_call is None


@pytest.mark.parametrize(
    ("thscode", "expected"),
    [
        ("600000.SH", True),
        ("688001.SH", True),
        ("900925.SH", False),
        ("000001.SZ", True),
        ("300001.SZ", True),
        ("200001.SZ", False),
        ("430001.BJ", True),
        ("000300.SH", False),
    ],
)
def test_a_share_code_filter_excludes_b_shares_and_indices(
    thscode: str, expected: bool
) -> None:
    assert is_a_share_thscode(thscode) is expected


@pytest.mark.parametrize(
    ("thscode", "interval", "start_ms", "end_ms"),
    [
        ("000300.SH,000001.SH", "1d", 1, 2),
        ("000300.SH", "1m", 1, 2),
        ("000300.SH", "1d", 2, 1),
    ],
)
def test_invalid_history_arguments_fail_before_network(
    thscode: str, interval: str, start_ms: int, end_ms: int
) -> None:
    session = FakeSession(FakeResponse({}))

    with pytest.raises(ValueError):
        make_client(session).get_index_history(
            thscode, interval=interval, start_ms=start_ms, end_ms=end_ms
        )

    assert session.last_call is None
