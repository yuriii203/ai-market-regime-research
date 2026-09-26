from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import requests


BASE_URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers"
SERVER_PATHS = {
    "stock": "hexin-ifind-ds-stock-mcp",
    "edb": "hexin-ifind-ds-edb-mcp",
    "news": "hexin-ifind-ds-news-mcp",
    "global_stock": "hexin-ifind-ds-global-stock-mcp",
    "index": "hexin-ifind-ds-index-mcp",
    "future": "hexin-ifind-ds-futures-mcp",
}
ALLOWED_TOOLS = {
    "news": {"search_notice", "search_news", "search_trending_news"},
    "edb": {"search_edb", "get_edb_data"},
    "index": {"index_data", "sector_data", "index_highfreq_quotes"},
    "stock": {
        "get_stock_summary", "search_stocks", "get_stock_performance",
        "get_stock_info", "get_stock_shareholders", "get_stock_financials",
        "get_risk_indicators", "get_stock_events", "get_esg_data",
        "stock_highfreq_quotes",
    },
    "global_stock": {
        "search_global_stocks", "global_stock_profile", "global_stock_quotes",
        "global_stock_financial", "global_stock_events",
    },
    "future": {"future_profile", "future_quotes"},
}
BLOCKED_KEYS = {"__proto__", "prototype", "constructor"}


class IfindMcpError(RuntimeError):
    """Base error safe to surface as an iFinD research-step failure."""


class IfindConfigurationError(IfindMcpError):
    pass


class IfindTimeoutError(IfindMcpError):
    pass


class IfindTransportError(IfindMcpError):
    pass


class IfindProtocolError(IfindMcpError):
    pass


class IfindBusinessError(IfindMcpError):
    def __init__(self, code: object, message: str, sub_code: object = None) -> None:
        self.code = code
        self.message = message
        self.sub_code = sub_code
        suffix = f"，subCode={sub_code}" if sub_code not in (None, "") else ""
        super().__init__(f"iFinD业务错误 code={code}{suffix}：{message}")


class IfindEmptyDataError(IfindMcpError):
    pass


@dataclass(frozen=True)
class IfindTrace:
    trace_id: str
    server_type: str
    tool_name: str
    called_at: str
    jsonrpc_id: int
    response_hash: str
    provider_request_id: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_id_type": "local",
            "server_type": self.server_type,
            "tool_name": self.tool_name,
            "called_at": self.called_at,
            "jsonrpc_id": self.jsonrpc_id,
            "response_hash": self.response_hash,
            "provider_request_id": self.provider_request_id,
        }


@dataclass(frozen=True)
class IfindMcpResult:
    trace: IfindTrace
    data: Any
    business_envelope: dict[str, Any]
    raw_rpc: dict[str, Any]


@dataclass(frozen=True)
class TradingDateRange:
    start: str
    end: str

    @property
    def start_compact(self) -> str:
        return self.start.replace("-", "")

    @property
    def end_compact(self) -> str:
        return self.end.replace("-", "")


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_calls: int = 2,
        period_seconds: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls <= 0 or period_seconds <= 0:
            raise ValueError("限流参数必须大于0")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait_seconds = self.period_seconds - (now - self._calls[0])
            self.sleeper(max(0.001, wait_seconds))


class IfindMcpClient:
    def __init__(
        self,
        auth_token: str,
        *,
        base_url: str = BASE_URL,
        timeout: tuple[float, float] = (5.0, 60.0),
        session: requests.Session | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        if not auth_token.strip():
            raise IfindConfigurationError("缺少iFinD MCP密钥")
        self.auth_token = auth_token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        self._session_ids: dict[str, str] = {}
        self._next_ids: dict[str, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "IfindMcpClient":
        token = (
            os.getenv("IFIND_MCP_AUTH_TOKEN", "").strip()
            or os.getenv("IFIND_API_KEY", "").strip()
        )
        if not token:
            config_value = os.getenv("IFIND_MCP_CONFIG", "").strip()
            config_path = (
                Path(config_value)
                if config_value
                else Path.home() / ".codex" / "skills" / "ifind-finance-data" / "mcp_config.json"
            )
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                token = str(config.get("auth_token", "")).strip()
            except (OSError, json.JSONDecodeError, AttributeError):
                token = ""
        if not token:
            raise IfindConfigurationError(
                "缺少IFIND_MCP_AUTH_TOKEN/IFIND_API_KEY，且未找到有效mcp_config.json"
            )
        return cls(token)

    def call_tool(
        self,
        server_type: str,
        tool_name: str,
        params: dict[str, Any],
    ) -> IfindMcpResult:
        self._validate_call(server_type, tool_name, params)
        self._initialize(server_type)
        self.rate_limiter.acquire()
        rpc_id = self._next_id(server_type)
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
        }
        raw_rpc, _ = self._post(server_type, payload)
        if not isinstance(raw_rpc, dict):
            raise IfindProtocolError("iFinD MCP返回内容不是JSON对象")
        if "error" in raw_rpc:
            error = raw_rpc["error"]
            raise IfindBusinessError(
                error.get("code") if isinstance(error, dict) else "rpc",
                error.get("message", str(error)) if isinstance(error, dict) else str(error),
            )
        envelope = _extract_business_envelope(raw_rpc)
        if envelope.get("code") != 1:
            raise IfindBusinessError(
                envelope.get("code"),
                str(envelope.get("subMsg") or envelope.get("msg") or "未知错误"),
                envelope.get("subCode"),
            )
        data = _decode_json_strings(envelope.get("data"))
        if _is_empty(data):
            raise IfindEmptyDataError(f"{server_type}.{tool_name}返回空数据")
        raw_text = json.dumps(raw_rpc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        trace = IfindTrace(
            trace_id=f"ifind-{uuid.uuid4().hex}",
            server_type=server_type,
            tool_name=tool_name,
            called_at=datetime.now().astimezone().isoformat(),
            jsonrpc_id=rpc_id,
            response_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        )
        return IfindMcpResult(trace, data, envelope, raw_rpc)

    def _initialize(self, server_type: str) -> None:
        if server_type in self._session_ids:
            return
        rpc_id = self._next_id(server_type)
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "market-regime-research", "version": "0.2.0"},
            },
        }
        _, headers = self._post(server_type, payload, include_session=False)
        session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        if not session_id:
            raise IfindProtocolError("iFinD初始化成功但未返回Mcp-Session-Id")
        self._session_ids[server_type] = session_id
        self._post(
            server_type,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    def _post(
        self,
        server_type: str,
        payload: dict[str, Any],
        *,
        include_session: bool = True,
    ) -> tuple[Any, dict[str, str]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": self.auth_token,
        }
        if include_session and server_type in self._session_ids:
            headers["Mcp-Session-Id"] = self._session_ids[server_type]
        try:
            response = self.session.post(
                f"{self.base_url}/{SERVER_PATHS[server_type]}",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise IfindTimeoutError(f"iFinD请求超时：{server_type}") from exc
        except requests.RequestException as exc:
            raise IfindTransportError(f"iFinD连接失败：{exc}") from exc
        if response.status_code >= 400:
            raise IfindTransportError(
                f"iFinD HTTP错误 {response.status_code}：{server_type}"
            )
        text = getattr(response, "text", "") or ""
        if not text.strip():
            return None, dict(response.headers)
        try:
            return response.json(), dict(response.headers)
        except (ValueError, json.JSONDecodeError) as exc:
            raise IfindProtocolError("iFinD返回了非JSON响应") from exc

    def _next_id(self, server_type: str) -> int:
        with self._lock:
            value = self._next_ids.get(server_type, 0) + 1
            self._next_ids[server_type] = value
            return value

    @staticmethod
    def _validate_call(
        server_type: str, tool_name: str, params: dict[str, Any]
    ) -> None:
        if server_type not in SERVER_PATHS:
            raise ValueError(f"未知iFinD服务：{server_type}")
        if tool_name not in ALLOWED_TOOLS.get(server_type, set()):
            raise ValueError(f"工具不属于服务{server_type}：{tool_name}")
        _validate_params(params)


def absolute_trading_window(
    trading_dates: list[str] | tuple[str, ...],
    *,
    as_of: str | date,
    count: int,
) -> TradingDateRange:
    if count <= 0:
        raise ValueError("交易日窗口必须大于0")
    as_of_date = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    normalized = sorted(
        {
            _parse_date(value)
            for value in trading_dates
            if _parse_date(value) <= as_of_date
        }
    )
    if len(normalized) < count:
        raise ValueError(f"交易日历不足：需要{count}日，只有{len(normalized)}日")
    selected = normalized[-count:]
    return TradingDateRange(selected[0].isoformat(), selected[-1].isoformat())


def validate_indicator_date_range(
    result: IfindMcpResult,
    expected: TradingDateRange,
) -> None:
    parameter_sets = _find_named_values(result.data, "indicators_params")
    ranges: list[tuple[str, str]] = []
    for parameter_set in parameter_sets:
        if not isinstance(parameter_set, dict):
            continue
        for value in parameter_set.values():
            if not isinstance(value, dict):
                continue
            start = value.get("起始交易日期")
            end = value.get("截止交易日期")
            if start and end:
                ranges.append((str(start).replace("-", ""), str(end).replace("-", "")))
    if not ranges:
        raise IfindProtocolError("iFinD响应缺少可校验的绝对交易日起止参数")
    expected_pair = (expected.start_compact, expected.end_compact)
    if any(item != expected_pair for item in ranges):
        raise IfindProtocolError(
            f"iFinD日期口径与计划不一致：期望{expected_pair}，实际{ranges}"
        )


def _extract_business_envelope(raw_rpc: dict[str, Any]) -> dict[str, Any]:
    content = raw_rpc.get("result", {}).get("content")
    if not isinstance(content, list):
        raise IfindProtocolError("iFinD MCP响应缺少result.content")
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(envelope, dict) and "code" in envelope:
            return envelope
    raise IfindProtocolError("iFinD MCP响应没有可解析的业务JSON")


def _decode_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        clean = value.strip()
        if clean.startswith(("{", "[")):
            try:
                return _decode_json_strings(json.loads(clean))
            except json.JSONDecodeError:
                return value
        return value
    if isinstance(value, list):
        return [_decode_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_json_strings(item) for key, item in value.items()}
    return value


def _find_named_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                found.append(item)
            found.extend(_find_named_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_named_values(item, key))
    return found


def _validate_params(params: dict[str, Any]) -> None:
    if not isinstance(params, dict):
        raise TypeError("iFinD参数必须是JSON对象")

    def walk(value: Any) -> None:
        if value is None or isinstance(value, (str, int, bool)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("iFinD参数包含无效数字")
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if isinstance(value, dict):
            for name, item in value.items():
                if name in BLOCKED_KEYS:
                    raise TypeError("iFinD参数包含禁止字段")
                walk(item)
            return
        raise TypeError("iFinD参数包含不支持的类型")

    walk(params)
    json.dumps(params, allow_nan=False)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _parse_date(value: str) -> date:
    clean = value.strip()
    if len(clean) == 8 and clean.isdigit():
        clean = f"{clean[:4]}-{clean[4:6]}-{clean[6:]}"
    return date.fromisoformat(clean)

