from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Generic, TypeVar
from zoneinfo import ZoneInfo

import requests


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_BASE_URL = "https://fuyao.aicubes.cn"
ERROR_LABELS = {
    1001: "缺少必填参数",
    1002: "参数格式错误",
    1003: "参数取值越界",
    1004: "参数冲突",
    2001: "未认证",
    2003: "权限不足",
    3001: "标的不存在",
    3002: "数据未就绪",
    3004: "标的类型不支持该能力",
    4001: "频率超限",
    5001: "服务内部错误",
    5002: "上游服务超时",
    5003: "数据源不可用",
}

T = TypeVar("T")


class FuyaoError(RuntimeError):
    """Base exception for errors safe to surface as a data-source failure."""


class FuyaoConfigurationError(FuyaoError):
    """The local Fuyao client configuration is incomplete."""


class FuyaoTimeoutError(FuyaoError):
    """The request exceeded its configured timeout."""


class FuyaoTransportError(FuyaoError):
    """The remote service could not be reached or returned invalid HTTP."""


class FuyaoResponseError(FuyaoError):
    """The payload did not follow the documented ApiResponse contract."""


class FuyaoEmptyDataError(FuyaoError):
    """The request succeeded but did not contain usable business records."""

    def __init__(self, endpoint: str, request_id: str | None) -> None:
        self.endpoint = endpoint
        self.request_id = request_id
        suffix = f"，request_id={request_id}" if request_id else ""
        super().__init__(f"扶摇接口返回空数据：{endpoint}{suffix}")


class FuyaoBusinessError(FuyaoError):
    """Fuyao returned a non-zero business code in a valid envelope."""

    def __init__(self, code: int, message: str, request_id: str | None) -> None:
        self.code = code
        self.message = message
        self.request_id = request_id
        label = ERROR_LABELS.get(code, "未知业务错误")
        suffix = f"，request_id={request_id}" if request_id else ""
        super().__init__(f"扶摇业务错误 {code}（{label}）：{message}{suffix}")


@dataclass(frozen=True)
class ResponseMetadata:
    request_id: str
    timestamp_ms: int
    code: int
    message: str
    received_at: datetime | None = None

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=SHANGHAI_TZ)

    @property
    def retrieval_time(self) -> datetime:
        """Local time when this client received the response envelope."""
        return self.received_at or self.timestamp


@dataclass(frozen=True)
class DataResult(Generic[T]):
    metadata: ResponseMetadata
    items: list[T]


@dataclass(frozen=True)
class TradingDay:
    date_ms: int
    date: str


@dataclass(frozen=True)
class PriceBar:
    date_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float | None
    turnover: float | None

    @property
    def trading_date(self) -> date:
        return datetime.fromtimestamp(
            self.date_ms / 1000, tz=SHANGHAI_TZ
        ).date()


@dataclass(frozen=True)
class IndexConstituent:
    thscode: str
    ticker: str
    name: str


@dataclass(frozen=True)
class StockSnapshot:
    thscode: str
    last_price: float | None
    price_change_ratio_pct: float | None
    turnover: float | None


@dataclass(frozen=True)
class ValuationSnapshot:
    thscode: str
    name: str | None
    pe_ttm: float | None
    pb_mrq: float | None


@dataclass(frozen=True)
class PoolItem:
    thscode: str
    name: str
    continue_day_cnt: int | None = None
    open_times: int | None = None


@dataclass(frozen=True)
class PoolResult:
    metadata: ResponseMetadata
    total: int
    items: list[PoolItem]
    request_ids: tuple[str, ...] = ()


def shanghai_date_to_ms(value: date) -> int:
    """Convert an Asia/Shanghai natural date to Unix milliseconds."""
    dt = datetime.combine(value, time.min, tzinfo=SHANGHAI_TZ)
    return int(dt.timestamp() * 1000)


def is_a_share_thscode(value: str) -> bool:
    """Return whether a code belongs to the A-share universes served here.

    Shanghai B shares use the 900 prefix and Shenzhen B shares use 200;
    broad exchange indices can include them even though A-share endpoints reject them.
    """
    clean = value.strip().upper()
    if "." not in clean:
        return False
    ticker, suffix = clean.rsplit(".", 1)
    if len(ticker) != 6 or not ticker.isdigit():
        return False
    if suffix == "SH":
        return ticker.startswith("6")
    if suffix == "SZ":
        return ticker.startswith(("0", "3"))
    if suffix == "BJ":
        return True
    return False


class FuyaoClient:
    """Small, deterministic REST client for the Fuyao market-data API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: tuple[float, float] = (5.0, 20.0),
        session: requests.Session | None = None,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise FuyaoConfigurationError("缺少 FUYAO_API_KEY")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "X-api-key": clean_key,
                "Accept": "application/json",
                "User-Agent": "market-regime-research/0.1",
            }
        )

    @classmethod
    def from_env(cls) -> "FuyaoClient":
        return cls(os.getenv("FUYAO_API_KEY", ""))

    def get_trading_days(self) -> DataResult[TradingDay]:
        endpoint = "/api/a-share/calendar/trading-days"
        metadata, items = self._get_items(endpoint)
        parsed = [
            TradingDay(date_ms=int(item["date_ms"]), date=str(item["date"]))
            for item in items
        ]
        return DataResult(metadata=metadata, items=parsed)

    def get_index_history(
        self,
        thscode: str,
        *,
        start_ms: int,
        end_ms: int,
        interval: str = "1d",
    ) -> DataResult[PriceBar]:
        clean_code = thscode.strip().upper()
        if not clean_code or "," in clean_code:
            raise ValueError("thscode 必须是单个完整指数代码，例如 000300.SH")
        if interval != "1d":
            raise ValueError("指数历史K线当前仅支持 interval='1d'")
        if start_ms >= end_ms:
            raise ValueError("start_ms 必须早于 end_ms")

        endpoint = "/api/a-share-index/prices/historical"
        metadata, items = self._get_items(
            endpoint,
            params={
                "thscode": clean_code,
                "interval": interval,
                "start": start_ms,
                "end": end_ms,
            },
        )
        parsed = [
            PriceBar(
                date_ms=int(item["date_ms"]),
                open_price=float(item["open_price"]),
                high_price=float(item["high_price"]),
                low_price=float(item["low_price"]),
                close_price=float(item["close_price"]),
                volume=_optional_float(item.get("volume")),
                turnover=_optional_float(item.get("turnover")),
            )
            for item in items
        ]
        return DataResult(metadata=metadata, items=parsed)

    def get_index_constituents(
        self, thscode: str
    ) -> DataResult[IndexConstituent]:
        clean_code = _clean_single_thscode(thscode, "指数代码")
        endpoint = "/api/a-share-index/constituents/ths-stock-list"
        metadata, items = self._get_items(
            endpoint, params={"thscode": clean_code}
        )
        parsed = [
            IndexConstituent(
                thscode=str(item["thscode"]),
                ticker=str(item["ticker"]),
                name=str(item["name"]),
            )
            for item in items
        ]
        return DataResult(metadata=metadata, items=parsed)

    def get_stock_snapshots(
        self, thscodes: list[str]
    ) -> DataResult[StockSnapshot]:
        clean_codes = _clean_thscode_batch(thscodes)
        endpoint = "/api/a-share/prices/snapshot"
        metadata, items = self._get_items(
            endpoint, params={"thscodes": ",".join(clean_codes)}
        )
        parsed = [
            StockSnapshot(
                thscode=str(item["thscode"]),
                last_price=_optional_float(item.get("last_price")),
                price_change_ratio_pct=_optional_float(
                    item.get("price_change_ratio_pct")
                ),
                turnover=_optional_float(item.get("turnover")),
            )
            for item in items
        ]
        return DataResult(metadata=metadata, items=parsed)

    def get_valuations(
        self, thscodes: list[str]
    ) -> DataResult[ValuationSnapshot]:
        clean_codes = _clean_thscode_batch(thscodes)
        endpoint = "/api/a-share/valuations/snapshot"
        metadata, items = self._get_items(
            endpoint, params={"thscodes": ",".join(clean_codes)}
        )
        parsed = [
            ValuationSnapshot(
                thscode=str(item["thscode"]),
                name=None if item.get("name") is None else str(item["name"]),
                pe_ttm=_optional_float(item.get("pe_ttm")),
                pb_mrq=_optional_float(item.get("pb_mrq")),
            )
            for item in items
        ]
        return DataResult(metadata=metadata, items=parsed)

    def get_limit_pool(
        self, pool: str, *, date_ms: int
    ) -> PoolResult:
        definitions = {
            "up": ("limit-up-pool", "continue_day_cnt", "desc"),
            "down": ("limit-down-pool", "last_limit_time", "desc"),
            "break": ("limit-break-pool", "open_times", "desc"),
        }
        if pool not in definitions:
            raise ValueError("pool 必须是 up、down 或 break")
        path, sort_field, sort_dir = definitions[pool]
        endpoint = f"/api/a-share/special-data/{path}"
        request_ids: list[str] = []
        parsed: list[PoolItem] = []
        metadata: ResponseMetadata | None = None
        total = 0
        pages = 1
        page = 1
        while page <= pages:
            page_metadata, data = self._get_data(
                endpoint,
                params={
                    "date_ms": int(date_ms),
                    "page": page,
                    "size": 200,
                    "sort_field": sort_field,
                    "sort_dir": sort_dir,
                },
            )
            if metadata is None:
                metadata = page_metadata
            request_ids.append(page_metadata.request_id)
            pagination = data.get("pagination")
            items = data.get("item")
            if not isinstance(pagination, dict) or not isinstance(items, list):
                raise FuyaoResponseError(
                    f"扶摇成功响应缺少 pagination 或 item：{endpoint}"
                )
            total = int(pagination.get("total", len(items)))
            pages = max(1, int(pagination.get("pages", 1)))
            parsed.extend(
                PoolItem(
                    thscode=str(item["thscode"]),
                    name=str(item["name"]),
                    continue_day_cnt=_optional_int(item.get("continue_day_cnt")),
                    open_times=_optional_int(item.get("open_times")),
                )
                for item in items
                if isinstance(item, dict)
            )
            page += 1
        if metadata is None:
            raise FuyaoResponseError(f"扶摇分页响应为空：{endpoint}")
        return PoolResult(
            metadata=metadata,
            total=total,
            items=parsed,
            request_ids=tuple(request_ids),
        )

    def _get_items(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> tuple[ResponseMetadata, list[dict[str, Any]]]:
        metadata, data = self._get_data(endpoint, params=params)
        items = data.get("item")
        if not isinstance(items, list):
            raise FuyaoResponseError(f"扶摇成功响应缺少 item 数组：{endpoint}")
        if not items:
            raise FuyaoEmptyDataError(endpoint, metadata.request_id or None)
        if not all(isinstance(item, dict) for item in items):
            raise FuyaoResponseError(f"扶摇 item 包含非对象元素：{endpoint}")
        return metadata, items

    def _get_data(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> tuple[ResponseMetadata, dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            raise FuyaoTimeoutError(
                f"扶摇接口请求超时：{endpoint}，请稍后重试"
            ) from exc
        except requests.RequestException as exc:
            raise FuyaoTransportError(
                f"无法连接扶摇接口：{endpoint}（{exc.__class__.__name__}）"
            ) from exc

        if response.status_code == 429:
            raise FuyaoBusinessError(4001, "HTTP 429 请求频率超限", None)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise FuyaoTransportError(
                f"扶摇接口返回 HTTP {response.status_code}：{endpoint}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise FuyaoResponseError(
                f"扶摇接口返回了非 JSON 内容：{endpoint}"
            ) from exc
        if not isinstance(payload, dict):
            raise FuyaoResponseError(f"扶摇响应不是 JSON 对象：{endpoint}")

        request_id_value = payload.get("request_id")
        request_id = "" if request_id_value is None else str(request_id_value)
        try:
            code = int(payload["code"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FuyaoResponseError(f"扶摇响应缺少有效 code：{endpoint}") from exc
        message = str(payload.get("message", ""))
        if code != 0:
            raise FuyaoBusinessError(code, message, request_id or None)

        data = payload.get("data")
        if not isinstance(data, dict):
            raise FuyaoResponseError(f"扶摇成功响应缺少 data 对象：{endpoint}")
        try:
            timestamp_ms = int(data["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FuyaoResponseError(
                f"扶摇成功响应缺少有效 timestamp：{endpoint}"
            ) from exc
        metadata = ResponseMetadata(
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            code=code,
            message=message,
            received_at=datetime.now(SHANGHAI_TZ),
        )
        return metadata, data


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _clean_single_thscode(value: str, label: str) -> str:
    clean_code = value.strip().upper()
    if not clean_code or "," in clean_code:
        raise ValueError(f"{label}必须是单个完整 thscode")
    return clean_code


def _clean_thscode_batch(values: list[str]) -> list[str]:
    clean_codes = list(dict.fromkeys(value.strip().upper() for value in values))
    if not clean_codes or any(not code or "," in code for code in clean_codes):
        raise ValueError("thscodes 必须是非空的完整代码列表")
    if len(clean_codes) > 100:
        raise ValueError("单次批量查询最多100个 thscode")
    return clean_codes
