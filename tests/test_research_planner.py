from __future__ import annotations

from src.research.models import ResearchQuestion
from src.research.planner import build_research_plan


TRADING_DATES = (
    "2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02",
    "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08",
    "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14",
    "2026-09-15", "2026-09-16", "2026-09-17",
    "2026-09-18", "2026-09-21", "2026-09-22",
    "2026-09-23", "2026-09-24", "2026-09-25",
)


def _plan(question: ResearchQuestion):
    return build_research_plan(
        question,
        subject_symbol="000300.SH",
        subject_name="沪深300",
        as_of="2026-09-24",
        trading_dates=TRADING_DATES,
    )


def test_all_five_questions_have_deterministic_routes() -> None:
    plans = [_plan(question) for question in ResearchQuestion]

    assert len(plans) == 5
    assert all(plan.steps for plan in plans)
    assert all(plan.to_json() == plan.to_json() for plan in plans)


def test_macro_plan_searches_before_fetching_data() -> None:
    plan = _plan(ResearchQuestion.BIGGEST_RISK)

    assert plan.steps[0].tool_name == "search_edb"
    assert plan.steps[1].tool_name == "get_edb_data"
    assert plan.steps[1].depends_on == ("RS-01",)
    assert [item.tool_name for item in plan.steps[:4]] == [
        "search_edb", "get_edb_data", "get_edb_data", "get_edb_data"
    ]
    assert "中债国债" in plan.steps[1].params["query"]
    assert "美国:国债" in plan.steps[2].params["query"]
    assert "美元兑人民币" in plan.steps[3].params["query"]


def test_remote_queries_use_explicit_dates_and_subject_limit() -> None:
    industry = _plan(ResearchQuestion.INDUSTRY_SUPPORT)

    assert len(industry.steps) == 2
    assert all("20260918至20260924" in step.params["query"] for step in industry.steps)
    assert all(step.params["query"].count("、") == 4 for step in industry.steps)


def test_event_plan_uses_bounded_time_window() -> None:
    plan = _plan(ResearchQuestion.STATE_CHANGING_EVENTS)
    params = plan.steps[0].params

    assert params["time_start"] == "2026-09-10"
    assert params["time_end"] == "2026-09-24"
    assert params["size"] == 5


def test_large_small_plan_uses_twenty_trading_days_and_explicit_proxies() -> None:
    plan = _plan(ResearchQuestion.LARGE_VS_SMALL)
    query = plan.steps[0].params["query"]

    assert "沪深300、中证1000" in query
    assert "20260828至20260924" in query
    assert "上证指数" not in query
