from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.research.quota import DailyQuotaExceededError, DailyResearchQuota


SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_daily_quota_reserves_calls_and_persists_between_instances(tmp_path) -> None:
    state_path = tmp_path / "quota.json"
    now = lambda: datetime(2026, 9, 26, 10, tzinfo=SHANGHAI)
    first = DailyResearchQuota(5, state_path, now=now)

    snapshot = first.reserve(3)
    second = DailyResearchQuota(5, state_path, now=now)

    assert snapshot.remaining == 2
    assert second.snapshot().used == 3
    with pytest.raises(DailyQuotaExceededError, match="剩余2次"):
        second.reserve(3)


def test_daily_quota_resets_on_next_shanghai_day(tmp_path) -> None:
    current = [datetime(2026, 9, 26, 23, tzinfo=SHANGHAI)]
    quota = DailyResearchQuota(5, tmp_path / "quota.json", now=lambda: current[0])
    quota.reserve(5)

    current[0] = datetime(2026, 9, 27, 0, 1, tzinfo=SHANGHAI)

    assert quota.snapshot().used == 0
    assert quota.snapshot().remaining == 5
