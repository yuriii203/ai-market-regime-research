from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class DailyQuotaExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class DailyQuotaSnapshot:
    date: str
    limit: int
    used: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class DailyResearchQuota:
    """Process-shared, file-backed daily quota for public research buttons."""

    def __init__(
        self,
        limit: int,
        state_path: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("每日调用额度必须大于0")
        self.limit = limit
        self.state_path = state_path
        self._now = now or (lambda: datetime.now(SHANGHAI_TZ))
        self._lock = threading.Lock()

    def snapshot(self) -> DailyQuotaSnapshot:
        with self._lock:
            return self._load_current()

    def reserve(self, units: int) -> DailyQuotaSnapshot:
        if units < 0:
            raise ValueError("预留调用次数不能为负数")
        with self._lock:
            current = self._load_current()
            if current.used + units > current.limit:
                raise DailyQuotaExceededError(
                    f"今日继续研究额度不足：剩余{current.remaining}次，本次需要{units}次。"
                    "额度将在上海时区次日自动重置。"
                )
            updated = DailyQuotaSnapshot(
                date=current.date,
                limit=current.limit,
                used=current.used + units,
            )
            self._save(updated)
            return updated

    def _load_current(self) -> DailyQuotaSnapshot:
        today = self._now().astimezone(SHANGHAI_TZ).date().isoformat()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            stored_date = str(payload.get("date", ""))
            used = int(payload.get("used", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stored_date, used = "", 0
        if stored_date != today:
            used = 0
        return DailyQuotaSnapshot(today, self.limit, max(0, used))

    def _save(self, snapshot: DailyQuotaSnapshot) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"date": snapshot.date, "used": snapshot.used},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)


@lru_cache(maxsize=1)
def get_research_quota() -> DailyResearchQuota:
    raw_limit = os.getenv("CONTINUE_RESEARCH_DAILY_CALL_LIMIT", "60").strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError("CONTINUE_RESEARCH_DAILY_CALL_LIMIT必须是整数") from exc
    state_path = Path(
        os.getenv(
            "CONTINUE_RESEARCH_QUOTA_FILE",
            ".runtime/continue_research_quota.json",
        )
    )
    return DailyResearchQuota(limit, state_path)
