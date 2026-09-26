from __future__ import annotations

import json

import pytest
import requests

from src.clients.ifind_mcp import (
    IfindBusinessError,
    IfindEmptyDataError,
    IfindMcpClient,
    IfindMcpResult,
    IfindProtocolError,
    IfindTimeoutError,
    IfindTrace,
    SlidingWindowRateLimiter,
    TradingDateRange,
    absolute_trading_window,
    validate_indicator_date_range,
)


class FakeResponse:
    def __init__(self, data, *, status_code=200, headers=None):
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = "" if data is None else json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _rpc_result(business_data, *, code=1):
    envelope = {"code": code, "msg": "success", "subCode": None, "subMsg": None, "data": business_data}
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": json.dumps(envelope, ensure_ascii=False)}]},
    }


def _client(tool_response):
    session = FakeSession([
        FakeResponse({"jsonrpc": "2.0", "id": 1, "result": {}}, headers={"Mcp-Session-Id": "session-1"}),
        FakeResponse(None),
        tool_response,
    ])
    return IfindMcpClient("secret", session=session), session


def test_client_decodes_nested_news_json_and_creates_local_trace() -> None:
    nested = json.dumps([{"资讯标题": "测试新闻", "日期": "2026-09-25"}], ensure_ascii=False)
    client, _ = _client(FakeResponse(_rpc_result({"data": nested})))

    result = client.call_tool("news", "search_news", {"query": "A股", "size": 1})

    assert result.data["data"][0]["资讯标题"] == "测试新闻"
    assert result.trace.trace_id.startswith("ifind-")
    assert result.trace.provider_request_id is None
    assert len(result.trace.response_hash) == 64


def test_business_error_is_not_silently_accepted() -> None:
    client, _ = _client(FakeResponse(_rpc_result(None, code=0)))

    with pytest.raises(IfindBusinessError):
        client.call_tool("news", "search_news", {"query": "A股"})


def test_empty_business_data_is_not_silently_accepted() -> None:
    client, _ = _client(FakeResponse(_rpc_result({})))

    with pytest.raises(IfindEmptyDataError):
        client.call_tool("news", "search_news", {"query": "A股"})


def test_timeout_is_normalized() -> None:
    client = IfindMcpClient("secret", session=FakeSession([requests.Timeout()]))

    with pytest.raises(IfindTimeoutError):
        client.call_tool("news", "search_news", {"query": "A股"})


def test_absolute_trading_window_uses_dates_not_natural_days() -> None:
    result = absolute_trading_window(
        ("20260918", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"),
        as_of="2026-09-24",
        count=5,
    )

    assert result == TradingDateRange("2026-09-18", "2026-09-24")


def test_indicator_date_range_rejects_parser_drift() -> None:
    result = IfindMcpResult(
        IfindTrace("trace", "index", "sector_data", "now", 3, "hash"),
        {"indicators_params": {"区间涨跌幅": {"起始交易日期": "20260923", "截止交易日期": "20260924"}}},
        {},
        {},
    )

    with pytest.raises(IfindProtocolError, match="日期口径"):
        validate_indicator_date_range(
            result, TradingDateRange("2026-09-18", "2026-09-24")
        )


def test_indicator_date_range_accepts_exact_absolute_dates() -> None:
    result = IfindMcpResult(
        IfindTrace("trace", "index", "sector_data", "now", 3, "hash"),
        {"indicators_params": {"区间涨跌幅": {"起始交易日期": "20260918", "截止交易日期": "20260924"}}},
        {},
        {},
    )

    validate_indicator_date_range(
        result, TradingDateRange("2026-09-18", "2026-09-24")
    )


def test_rate_limiter_waits_before_third_call() -> None:
    current = [0.0]
    sleeps = []

    def clock():
        return current[0]

    def sleeper(seconds):
        sleeps.append(seconds)
        current[0] += seconds

    limiter = SlidingWindowRateLimiter(2, 1.0, clock=clock, sleeper=sleeper)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [1.0]
