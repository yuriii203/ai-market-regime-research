from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.clients.fuyao import PriceBar, shanghai_date_to_ms
from src.clients.ifind_mcp import IfindMcpResult, IfindTrace
from src.indicators.market_dimensions import MarketBreadth, StyleRotation, StyleRow
from src.research.models import ResearchQuestion
from src.research.planner import build_research_plan
from src.research.services import (
    research_events,
    research_historical_analogs,
    research_industries,
    research_large_vs_small,
    research_market_supplement,
    research_risk_variables,
    validate_research_compliance,
)


TRADING_DATES = (
    "2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02",
    "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08",
    "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14",
    "2026-09-15", "2026-09-16", "2026-09-17",
    "2026-09-18", "2026-09-21", "2026-09-22",
    "2026-09-23", "2026-09-24", "2026-09-25",
)


class StubClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def call_tool(self, server_type, tool_name, params):
        self.calls.append((server_type, tool_name, params))
        return self.results.pop(0)


def _result(server, tool, data, number=1):
    return IfindMcpResult(
        IfindTrace(f"ifind-{number}", server, tool, "2026-09-25T10:00:00+08:00", number, "a" * 64),
        data,
        {"code": 1, "data": data},
        {},
    )


def _plan(question):
    return build_research_plan(
        question,
        subject_symbol="000300.SH",
        subject_name="沪深300",
        as_of="2026-09-24",
        trading_dates=TRADING_DATES,
    )


def _sector_answer(rows):
    body = "\n".join(f"|{index}|{name}|{value}|" for index, (name, value) in enumerate(rows, 1))
    return (
        "|证券代码|证券简称|成份区间涨跌幅(总市值加权平均)（单位：%）|\n"
        "|---|---|---|\n" + body
    )


def test_event_research_returns_sources_and_dimensions_without_prediction() -> None:
    client = StubClient([
        _result("news", "search_news", {"data": [{
            "资讯标题": "央行召开货币政策例会",
            "资讯内容": "保持流动性充裕并支持科技创新",
            "日期": "2026-09-24",
            "URL": "https://example.test/news/1",
        }]})
    ])

    result = research_events(_plan(ResearchQuestion.STATE_CHANGING_EVENTS), client)

    assert result.status == "complete"
    assert result.evidence[0].raw_value["url"].startswith("https://")
    assert "流动性" in result.evidence[0].raw_value["related_dimensions"]
    assert "不直接推导市场涨跌" in result.conclusion


def test_empty_news_result_degrades_to_failed_research_without_exception() -> None:
    client = StubClient([_result("news", "search_news", {"data": []})])

    result = research_events(
        _plan(ResearchQuestion.STATE_CHANGING_EVENTS), client
    )

    assert result.status == "failed"
    assert result.failed_steps == ("RS-01",)
    assert "没有取得可用事件记录" in result.conclusion
    assert client.calls[0][:2] == ("news", "search_news")


def test_event_research_filters_duplicates_out_of_window_and_low_relevance() -> None:
    client = StubClient([_result("news", "search_news", {"data": [
        {
            "资讯标题": "央行召开货币政策例会释放流动性信号",
            "资讯内容": "央行讨论货币政策与流动性。",
            "日期": "2026-09-24 10:00:00",
            "URL": "https://example.test/policy?id=1",
            "来源": "测试媒体",
        },
        {
            "资讯标题": "央行召开货币政策例会释放流动性信号",
            "资讯内容": "重复记录",
            "日期": "2026-09-24",
            "URL": "https://example.test/policy?id=2",
        },
        {
            "资讯标题": "央行召开货币政策例会释放流动性信号！",
            "资讯内容": "标题近似重复",
            "日期": "2026-09-24",
            "URL": "https://other.test/policy",
        },
        {
            "资讯标题": "国务院发布资本市场改革政策",
            "资讯内容": "超出计划窗口",
            "日期": "2026-08-01",
            "URL": "https://example.test/old",
        },
        {
            "资讯标题": "沪深300ETF配置关注与投资观点",
            "资讯内容": "产品推广内容",
            "日期": "2026-09-23",
            "URL": "https://example.test/etf",
        },
    ]})])

    result = research_events(
        _plan(ResearchQuestion.STATE_CHANGING_EVENTS), client
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].effective_date == "2026-09-24"
    assert result.evidence[0].raw_value["event_category"] == "货币/利率"
    assert result.evidence[0].raw_value["evidence_quality"] == "高"
    assert any("重复" in item for item in result.uncertainties)
    assert any("时间窗" in item for item in result.uncertainties)
    assert any("相关性偏低" in item for item in result.uncertainties)
    assert result.executed_steps == ("RS-01",)


def test_industry_research_ranks_ten_samples_and_disclaims_contribution() -> None:
    params = {"区间": {"起始交易日期": "20260918", "截止交易日期": "20260924"}}
    first = _result("index", "sector_data", {
        "answer": _sector_answer([("电子", 3.0), ("银行", -1.0), ("医药生物", 2.0), ("食品饮料", -2.0), ("计算机", 1.0)]),
        "indicators_params": params,
    })
    second = _result("index", "sector_data", {
        "answer": _sector_answer([("非银金融", 0.5), ("电力设备", 4.0), ("汽车", -3.0), ("有色金属", 1.5), ("机械设备", -0.5)]),
        "indicators_params": params,
    }, 2)

    result = research_industries(_plan(ResearchQuestion.INDUSTRY_SUPPORT), StubClient([first, second]))

    assert len(result.evidence) == 10
    assert "电力设备" in result.conclusion
    assert "汽车" in result.conclusion
    assert "不代表指数贡献" in result.conclusion
    assert "完整31行业" in result.uncertainties[0]


def test_risk_research_ranks_by_change_and_freshness_not_model_opinion() -> None:
    search = _result("edb", "search_edb", {"data": "候选指标"})
    china_yield = _result("edb", "get_edb_data", {"datas": [{"data": {
        "data": [["2026-09-18", 1.68], ["2026-09-24", 1.69]],
        "columns": ["日期", "中债国债到期收益率:10年"],
        "attrs": {"中债国债到期收益率:10年": {"unit": "%"}},
    }}]}, 2)
    us_yield = _result("edb", "get_edb_data", {"datas": [{"data": {
        "data": [["2026-09-18", 3.80], ["2026-09-24", 3.85]],
        "columns": ["日期", "美国:国债收益率:10年"],
        "attrs": {"美国:国债收益率:10年": {"unit": "%"}},
    }}]}, 3)
    currency = _result("edb", "get_edb_data", {"datas": [{"data": {
        "data": [["2026-09-18", 7.10], ["2026-09-24", 7.12]],
        "columns": ["日期", "美元兑人民币"],
        "attrs": {"美元兑人民币": {"unit": ""}},
    }}]}, 4)
    global_index = _result("index", "index_data", {"answer": (
        "|证券代码|证券简称|日期|收盘价|涨跌幅|\n|---|---|---|---|---|\n"
        "|SPX.GI|标普500|20260918|100|0.1|\n|SPX.GI|标普500|20260924|105|1.0|"
    )}, 5)
    futures = _result("future", "future_quotes", {"answer": (
        "|证券代码|证券简称|日期|涨跌幅(收盘价)|收盘价|\n|---|---|---|---|---|\n"
        "|SCZL.INE|原油主连|20260918|0.1|100|\n|SCZL.INE|原油主连|20260924|2.0|120|"
    )}, 6)

    result = research_risk_variables(
        _plan(ResearchQuestion.BIGGEST_RISK),
        StubClient([search, china_yield, us_yield, currency, global_index, futures]),
    )

    assert "原油主连" in result.conclusion
    assert result.evidence[0].raw_value["risk_score"] == 20.0
    assert "不代表对A股存在确定因果关系" in result.uncertainties[0]


def test_risk_research_ignores_non_trading_placeholder_rows() -> None:
    results = [
        _result("edb", "search_edb", {"data": "候选指标"}),
        _result("edb", "get_edb_data", {"datas": []}, 2),
        _result("edb", "get_edb_data", {"datas": []}, 3),
        _result("edb", "get_edb_data", {"datas": []}, 4),
        _result("index", "index_data", {"answer": (
            "|证券代码|证券简称|日期|收盘价|涨跌幅|\n|---|---|---|---|---|\n"
            "|SPX.GI|标普500|20260918|100|0.1|\n"
            "|SPX.GI|标普500|20260919|100||\n"
            "|SPX.GI|标普500|20260920|100|null|\n"
            "|SPX.GI|标普500|20260924|110|1.0|"
        )}, 5),
        _result("future", "future_quotes", {"answer": "无可用记录"}, 6),
    ]

    result = research_risk_variables(
        _plan(ResearchQuestion.BIGGEST_RISK), StubClient(results)
    )

    sp500 = next(item for item in result.evidence if "标普500" in item.statement)
    assert sp500.raw_value["change"] == pytest.approx(10.0)


def test_market_supplement_keeps_fuyao_breadth_primary_when_ifind_width_is_null() -> None:
    highfreq = _result("index", "index_highfreq_quotes", {"tables": [[
        ["证券代码", "time", "上涨家数", "下跌家数", "成交额"],
        ["000300.SH", "2026-09-24 16:00:00", "null", "null", "375000000000"],
    ]]})
    funds = _result("stock", "search_stocks", {"answer": (
        "|股票代码|股票简称|主力资金流向|\n|---|---|---|\n|600900.SH|长江电力|403092023|"
    )}, 2)
    breadth = MarketBreadth(55, 239, 4, 298, 300, -61.7)

    result = research_market_supplement(
        _plan(ResearchQuestion.LARGE_VS_SMALL), StubClient([highfreq, funds]), breadth
    )

    assert result.evidence[0].source == "扶摇成分股快照"
    assert "仍以扶摇" in "".join(result.uncertainties)
    assert "-184只" in result.conclusion


def test_historical_analog_returns_three_separated_dates_with_disclaimer() -> None:
    start = date(2026, 1, 1)
    bars = []
    for index in range(180):
        day = start + timedelta(days=index)
        close = 100 + index * 0.08 + math.sin(index / 8) * 4
        bars.append(PriceBar(shanghai_date_to_ms(day), close, close, close, close, 1, 1000))

    result = research_historical_analogs(
        _plan(ResearchQuestion.HISTORICAL_ANALOG), bars
    )

    assert result.status == "complete"
    assert len(result.evidence) == 3
    assert "不代表历史后续走势会重复" in result.conclusion


def test_large_small_research_uses_local_style_as_primary_explanation() -> None:
    remote = _result("index", "index_data", {"answer": (
        "|证券代码|证券简称|日期|收盘价|涨跌幅|\n|---|---|---|---|---|\n"
        "|000300.SH|沪深300|20260924|4439|-1.7|"
    )})
    style = StyleRotation(
        rows=(
            StyleRow("上证指数", -1.0, -2.0),
            StyleRow("沪深300", -2.0, -5.0),
            StyleRow("中证1000", -4.0, -8.0),
        ),
        leader="上证指数",
        laggard="中证1000",
        spread_20d_pct_points=3.0,
    )

    result = research_large_vs_small(
        _plan(ResearchQuestion.LARGE_VS_SMALL), StubClient([remote]), style
    )

    assert "问题前提成立" in result.conclusion
    assert "沪深300" in result.conclusion
    assert "中证1000" in result.conclusion
    assert any(item.source == "扶摇历史K线/本地计算" for item in result.evidence)
    assert any("风格代理" in item for item in result.uncertainties)


def test_large_small_research_rejects_false_premise() -> None:
    remote = _result("index", "index_data", {"answer": (
        "|证券代码|证券简称|日期|收盘价|涨跌幅|\n|---|---|---|---|---|\n"
        "|000300.SH|沪深300|20260924|4439|-1.7|"
    )})
    style = StyleRotation(
        rows=(
            StyleRow("沪深300", -4.13, -7.75),
            StyleRow("中证1000", -2.11, -12.23),
        ),
        leader="中证1000",
        laggard="沪深300",
        spread_20d_pct_points=2.02,
    )

    result = research_large_vs_small(
        _plan(ResearchQuestion.LARGE_VS_SMALL), StubClient([remote]), style
    )

    assert "问题前提不成立" in result.conclusion
    assert "相对落后2.02个百分点" in result.conclusion


def test_compliance_validator_rejects_prediction_and_trading_advice() -> None:
    base = research_events(
        _plan(ResearchQuestion.STATE_CHANGING_EVENTS),
        StubClient([_result("news", "search_news", {"data": []})]),
    )

    with pytest.raises(ValueError, match="投资建议"):
        validate_research_compliance(replace(base, conclusion="建议买入并保证收益"))
