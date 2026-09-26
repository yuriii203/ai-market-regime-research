from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.fuyao import FuyaoClient, SHANGHAI_TZ, shanghai_date_to_ms
from src.clients.ifind_mcp import IfindMcpClient
from src.research.models import ResearchQuestion
from src.research.planner import build_research_plan
from src.research.services import (
    research_events,
    research_historical_analogs,
    research_industries,
    research_risk_variables,
)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    fuyao = FuyaoClient.from_env()
    ifind = IfindMcpClient.from_env()
    calendar = fuyao.get_trading_days()
    end_date = datetime.now(SHANGHAI_TZ).date()
    history = fuyao.get_index_history(
        "000300.SH",
        start_ms=shanghai_date_to_ms(end_date - timedelta(days=320)),
        end_ms=shanghai_date_to_ms(end_date),
    )
    as_of = history.items[-1].trading_date.isoformat()
    trading_dates = tuple(_normalize_date(item.date) for item in calendar.items)

    def plan(question: ResearchQuestion):
        return build_research_plan(
            question,
            subject_symbol="000300.SH",
            subject_name="沪深300",
            as_of=as_of,
            trading_dates=trading_dates,
        )

    if "--risk-only" in sys.argv:
        risks = research_risk_variables(plan(ResearchQuestion.BIGGEST_RISK), ifind)
        _print_result("RISKS", risks)
        return

    events = research_events(plan(ResearchQuestion.STATE_CHANGING_EVENTS), ifind)
    industries = research_industries(plan(ResearchQuestion.INDUSTRY_SUPPORT), ifind)
    risks = research_risk_variables(plan(ResearchQuestion.BIGGEST_RISK), ifind)
    history_result = research_historical_analogs(
        plan(ResearchQuestion.HISTORICAL_ANALOG), history.items
    )
    for name, result in (
        ("EVENTS", events),
        ("INDUSTRIES", industries),
        ("RISKS", risks),
        ("HISTORY", history_result),
    ):
        _print_result(name, result)


def _print_result(name, result) -> None:
    traces = sum(bool(item.local_trace_id) for item in result.evidence)
    print(
        f"{name}_STATUS={result.status} EVIDENCE={len(result.evidence)} "
        f"TRACES={traces} FAILURES={len(result.failed_steps)}"
    )
    print(f"{name}_CONCLUSION={result.conclusion}")


def _normalize_date(value: str) -> str:
    clean = value.strip()
    if len(clean) == 8 and clean.isdigit():
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:]}"
    return clean


if __name__ == "__main__":
    main()
