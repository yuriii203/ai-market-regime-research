from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from src.clients.fuyao import PriceBar
from src.clients.ifind_mcp import (
    IfindMcpClient,
    IfindMcpError,
    IfindMcpResult,
    TradingDateRange,
    validate_indicator_date_range,
)
from src.indicators.market_dimensions import MarketBreadth, StyleRotation
from src.indicators.trend import calculate_trend_indicators
from src.research.models import (
    ResearchEvidence,
    ResearchPlan,
    ResearchResult,
    ResearchStep,
)


EVENT_DIMENSION_KEYWORDS = {
    "流动性": ("货币政策", "降准", "降息", "利率", "流动性", "融资"),
    "估值": ("利率", "国债", "风险溢价", "监管"),
    "风格轮动": ("科技", "医药", "消费", "地产", "新能源", "金融", "行业"),
    "市场情绪": ("风险", "波动", "制裁", "冲突", "调查", "退市"),
    "市场宽度": ("小微", "民营", "中小企业", "普惠"),
}
EVENT_HIGH_RELEVANCE_KEYWORDS = (
    "国务院", "央行", "证监会", "金融监管", "财政部", "发改委", "货币政策",
    "财政政策", "降准", "降息", "加息", "关税", "制裁", "美联储", "汇率",
    "国债收益率", "GDP", "CPI", "PPI", "PMI",
)
EVENT_MEDIUM_RELEVANCE_KEYWORDS = (
    "监管", "流动性", "融资", "回购", "增持", "退市", "地产", "地缘冲突",
    "风险事件", "行业政策", "资本市场改革",
)
EVENT_LOW_RELEVANCE_KEYWORDS = (
    "ETF", "投资观点", "配置关注", "布局", "收复", "反弹", "盘点",
)
RESEARCH_PROHIBITED_PHRASES = (
    "建议买入", "建议卖出", "可以买入", "可以卖出", "建议加仓", "建议减仓",
    "保证收益", "必涨", "必跌", "上涨概率为",
)


def research_events(
    plan: ResearchPlan,
    client: IfindMcpClient,
) -> ResearchResult:
    evidence: list[ResearchEvidence] = []
    uncertainties: list[str] = []
    failures: list[str] = []
    executed: list[str] = []
    rejected = {"duplicate": 0, "date": 0, "relevance": 0}
    for step in _mcp_steps(plan, tool_names={"search_news"}):
        executed.append(step.id)
        try:
            result = client.call_tool(step.server_type or "", step.tool_name, step.params)
            records = _news_records(result.data)
            if not records:
                raise ValueError("新闻结果没有可用记录")
            start_date = date.fromisoformat(str(step.params["time_start"]))
            end_date = date.fromisoformat(str(step.params["time_end"]))
            seen_urls: set[str] = set()
            seen_titles: list[str] = []
            for record in records:
                title = str(record.get("资讯标题") or record.get("标题") or "").strip()
                published = _normalize_news_date(
                    record.get("日期") or record.get("发布时间")
                )
                if not title or published is None:
                    rejected["date"] += 1
                    continue
                published_date = date.fromisoformat(published)
                if not start_date <= published_date <= end_date:
                    rejected["date"] += 1
                    continue
                summary = str(record.get("资讯内容") or record.get("内容") or "").strip()
                url = str(record.get("URL") or record.get("url") or "").strip()
                canonical_url = _canonical_news_url(url)
                fingerprint = _news_title_fingerprint(title)
                if (
                    canonical_url and canonical_url in seen_urls
                ) or any(_news_titles_similar(fingerprint, prior) for prior in seen_titles):
                    rejected["duplicate"] += 1
                    continue
                category, quality = _event_category_and_quality(title + " " + summary)
                if quality == "低":
                    rejected["relevance"] += 1
                    continue
                if canonical_url:
                    seen_urls.add(canonical_url)
                seen_titles.append(fingerprint)
                dimensions = _event_dimensions(title + " " + summary)
                publisher = str(
                    record.get("资讯来源") or record.get("来源") or record.get("媒体") or ""
                ).strip()
                evidence.append(
                    _remote_evidence(
                        f"RE-EVENT-{len(evidence) + 1:02d}",
                        result,
                        f"{title}；可能关联维度：{'、'.join(dimensions)}。",
                        effective_date=published or None,
                        raw_value={
                            "title": title,
                            "summary": summary[:500],
                            "url": url,
                            "publisher": publisher,
                            "event_category": category,
                            "evidence_quality": quality,
                            "related_dimensions": dimensions,
                        },
                    )
                )
        except (IfindMcpError, ValueError) as exc:
            failures.append(step.id)
            uncertainties.append(f"{step.title}失败：{exc}")
    if rejected["duplicate"]:
        uncertainties.append(f"已剔除{rejected['duplicate']}条URL或标题高度重复的新闻。")
    if rejected["date"]:
        uncertainties.append(f"已剔除{rejected['date']}条缺少日期或超出计划时间窗的新闻。")
    if rejected["relevance"]:
        uncertainties.append(f"已剔除{rejected['relevance']}条相关性偏低的行情评论或产品推广内容。")
    conclusion = (
        f"检索到{len(evidence)}条近期事件线索；它们仅用于识别可能变化的维度，"
        "不直接推导市场涨跌。"
        if evidence
        else "没有取得可用事件记录，不能据此判断哪些事件可能改变当前状态。"
    )
    return _result(plan, conclusion, evidence, uncertainties, failures, executed)


def research_industries(
    plan: ResearchPlan,
    client: IfindMcpClient,
) -> ResearchResult:
    rows: list[tuple[str, float, IfindMcpResult]] = []
    uncertainties: list[str] = [
        "仅覆盖10个代表性申万一级行业，不是完整31行业排名，也不是指数贡献度。"
    ]
    failures: list[str] = []
    executed: list[str] = []
    expected = _plan_date_range(plan)
    for step in _mcp_steps(plan):
        executed.append(step.id)
        try:
            result = client.call_tool(step.server_type or "", step.tool_name, step.params)
            validate_indicator_date_range(result, expected)
            for row in _markdown_rows(result.data):
                name = _security_name(row)
                value = _first_numeric(row, includes=("区间涨跌幅",))
                if name and value is not None:
                    rows.append((name.split("->")[-1], value, result))
        except (IfindMcpError, ValueError) as exc:
            failures.append(step.id)
            uncertainties.append(f"{step.title}失败：{exc}")
    unique: dict[str, tuple[float, IfindMcpResult]] = {}
    for name, value, result in rows:
        unique[name] = (value, result)
    ranked = sorted(unique.items(), key=lambda item: item[1][0], reverse=True)
    evidence = [
        _remote_evidence(
            f"RE-INDUSTRY-{index + 1:02d}",
            result,
            f"样本行业{name}区间涨跌幅为{value:+.2f}%。",
            effective_date=expected.end,
            raw_value=value,
            unit="%",
        )
        for index, (name, (value, result)) in enumerate(ranked)
    ]
    if ranked:
        conclusion = (
            f"在{len(ranked)}个样本行业中，{ranked[0][0]}相对领先"
            f"（{ranked[0][1][0]:+.2f}%），{ranked[-1][0]}相对落后"
            f"（{ranked[-1][1][0]:+.2f}%）。这是样本表现，不代表指数贡献。"
        )
    else:
        conclusion = "没有取得可比较的样本行业数据。"
    return _result(plan, conclusion, evidence, uncertainties, failures, executed)


def research_risk_variables(
    plan: ResearchPlan,
    client: IfindMcpClient,
) -> ResearchResult:
    variables: list[dict[str, Any]] = []
    uncertainties: list[str] = [
        "风险排序比较的是近期变化幅度和数据新鲜度，不代表对A股存在确定因果关系。"
    ]
    failures: list[str] = []
    executed: list[str] = []
    for step in _mcp_steps(plan):
        executed.append(step.id)
        try:
            result = client.call_tool(step.server_type or "", step.tool_name, step.params)
            if step.tool_name == "search_edb":
                continue
            variables.extend(_risk_series(result, plan.as_of))
        except (IfindMcpError, ValueError) as exc:
            failures.append(step.id)
            uncertainties.append(f"{step.title}失败：{exc}")
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in variables:
        current = deduplicated.get(item["name"])
        if current is None or item["latest_date"] > current["latest_date"]:
            deduplicated[item["name"]] = item
    ranked = sorted(
        deduplicated.values(), key=lambda item: item["risk_score"], reverse=True
    )
    evidence = [
        _remote_evidence(
            f"RE-RISK-{index + 1:02d}",
            item["result"],
            item["statement"],
            effective_date=item["latest_date"],
            raw_value={
                "change": item["change"],
                "change_unit": item["change_unit"],
                "freshness_days": item["freshness_days"],
                "risk_score": round(item["risk_score"], 4),
            },
            unit=item["change_unit"],
        )
        for index, item in enumerate(ranked)
    ]
    if ranked:
        leader = ranked[0]
        conclusion = (
            f"按近期绝对变化和数据新鲜度排序，当前变化最显著的候选风险变量是"
            f"{leader['name']}（{leader['statement']}）。这只是风险监控优先级，不是涨跌预测。"
        )
    else:
        conclusion = "没有取得足够的宏观、跨市场或商品序列，无法排序风险变量。"
    return _result(plan, conclusion, evidence, uncertainties, failures, executed)


def research_large_vs_small(
    plan: ResearchPlan,
    client: IfindMcpClient,
    style: StyleRotation,
) -> ResearchResult:
    evidence: list[ResearchEvidence] = []
    uncertainties: list[str] = [
        "沪深300和中证1000只是大盘、小盘风格代理，不能代表全部大盘股和小盘股。",
        "两只指数的行业结构不同，相对收益差不能单独解释为市值风格的因果结果。",
    ]
    failures: list[str] = []
    executed: list[str] = ["RS-02"]
    remote_rows: list[dict[str, str]] = []
    for step in _mcp_steps(plan, tool_names={"index_data"}):
        executed.append(step.id)
        try:
            result = client.call_tool(step.server_type or "", step.tool_name, step.params)
            remote_rows.extend(_markdown_rows(result.data))
            evidence.append(
                _remote_evidence(
                    "RE-STYLE-REMOTE-01",
                    result,
                    "iFinD返回沪深300与中证1000同一20个交易日区间的日频表现。",
                    effective_date=plan.as_of,
                    raw_value=remote_rows,
                )
            )
        except IfindMcpError as exc:
            failures.append(step.id)
            uncertainties.append(f"{step.title}失败：{exc}")
    for index, row in enumerate(style.rows):
        evidence.append(
            ResearchEvidence(
                id=f"RE-STYLE-LOCAL-{index + 1:02d}",
                evidence_type="fact",
                statement=(
                    f"{row.name}近20日收益率{row.return_20d_pct:+.2f}%，"
                    f"近60日收益率{row.return_60d_pct:+.2f}%。"
                ),
                source="扶摇历史K线/本地计算",
                effective_date=plan.as_of,
                retrieved_at=plan.as_of,
                local_trace_id=None,
                response_hash=None,
                raw_value={"return_20d_pct": row.return_20d_pct, "return_60d_pct": row.return_60d_pct},
                unit="%",
            )
        )
    rows_by_name = {row.name: row for row in style.rows}
    large = rows_by_name.get("沪深300")
    small = rows_by_name.get("中证1000")
    if large is None or small is None:
        failures.append("RS-02")
        conclusion = "缺少沪深300或中证1000的20日收益率，无法验证大盘强于小盘这一前提。"
    else:
        spread = large.return_20d_pct - small.return_20d_pct
        if spread > 0:
            conclusion = (
                f"问题前提成立：近20日沪深300收益率{large.return_20d_pct:+.2f}%，"
                f"高于中证1000的{small.return_20d_pct:+.2f}%，"
                f"大盘代理相对领先{spread:.2f}个百分点。"
            )
        else:
            conclusion = (
                f"问题前提不成立：近20日沪深300收益率{large.return_20d_pct:+.2f}%，"
                f"未高于中证1000的{small.return_20d_pct:+.2f}%，"
                f"大盘代理相对落后{abs(spread):.2f}个百分点。"
            )
        conclusion += " iFinD同一20个交易日数据仅用于交叉核对，不将相对表现解释为因果。"
    return _result(plan, conclusion, evidence, uncertainties, failures, executed)


def research_market_supplement(
    plan: ResearchPlan,
    client: IfindMcpClient,
    breadth: MarketBreadth,
) -> ResearchResult:
    evidence = [
        ResearchEvidence(
            id="RE-BREADTH-PRIMARY-01",
            evidence_type="fact",
            statement=(
                f"扶摇成分股宽度：上涨{breadth.advancing}只、下跌{breadth.declining}只、"
                f"平盘{breadth.flat}只，有效覆盖{breadth.coverage_pct:.1f}%。"
            ),
            source="扶摇成分股快照",
            effective_date=plan.as_of,
            retrieved_at=plan.as_of,
            local_trace_id=None,
            response_hash=None,
            raw_value={"advancing": breadth.advancing, "declining": breadth.declining},
        )
    ]
    uncertainties: list[str] = []
    failures: list[str] = []
    executed: list[str] = []
    calls = _mcp_steps(
        plan, tool_names={"index_highfreq_quotes", "search_stocks"}
    )
    for step in calls:
        executed.append(step.id)
        try:
            result = client.call_tool(step.server_type or "", step.tool_name, step.params)
            if step.tool_name == "index_highfreq_quotes":
                table = _highfreq_rows(result.data)
                if table:
                    row = table[0]
                    turnover = _number(row.get("成交额"))
                    rise = _number(row.get("上涨家数"))
                    fall = _number(row.get("下跌家数"))
                    evidence.append(
                        _remote_evidence(
                            "RE-LIQUIDITY-SUP-01", result,
                            f"iFinD指数成交额为{turnover:.0f}。" if turnover is not None else "iFinD未返回可用成交额。",
                            effective_date=str(row.get("time", ""))[:10] or None,
                            raw_value={"turnover": turnover, "rise_count": rise, "fall_count": fall},
                        )
                    )
                    if rise is None or fall is None:
                        uncertainties.append("iFinD未返回所选指数上涨/下跌家数，宽度仍以扶摇成分股快照为准。")
                    else:
                        uncertainties.append("iFinD上涨/下跌家数口径尚未确认，不参与所选指数宽度投票。")
            else:
                rows = _markdown_rows(result.data)
                evidence.append(
                    _remote_evidence(
                        "RE-FLOW-SUP-01", result,
                        f"iFinD返回{len(rows)}只主力资金净流入靠前的A股样本，仅反映资金活跃标的。",
                        effective_date=plan.as_of,
                        raw_value=rows[:5],
                    )
                )
        except IfindMcpError as exc:
            failures.append(step.id)
            uncertainties.append(f"{step.title}失败：{exc}")
    conclusion = (
        f"市场宽度主证据仍采用扶摇成分股快照，净宽度为"
        f"{breadth.advancing - breadth.declining:+d}只；iFinD成交额和资金排名仅作补充。"
    )
    return _result(plan, conclusion, evidence, uncertainties, failures, executed)


def research_historical_analogs(
    plan: ResearchPlan,
    history: list[PriceBar],
) -> ResearchResult:
    ordered = sorted(history, key=lambda item: item.date_ms)
    if len(ordered) < 100:
        return _result(
            plan,
            "历史样本不足，无法可靠寻找相似阶段。",
            [],
            ["至少需要100根日K，以计算当前特征并避开相邻重叠区间。"],
            ["RS-01"],
            ["RS-01"],
        )
    target = calculate_trend_indicators(ordered)
    candidates: list[tuple[float, int, Any]] = []
    for end_index in range(60, len(ordered) - 20):
        indicators = calculate_trend_indicators(ordered[: end_index + 1])
        distance = _feature_distance(target, indicators)
        candidates.append((distance, end_index, indicators))
    selected: list[tuple[float, int, Any]] = []
    for candidate in sorted(candidates):
        if all(abs(candidate[1] - existing[1]) >= 10 for existing in selected):
            selected.append(candidate)
        if len(selected) == 3:
            break
    evidence = []
    for index, (distance, _, indicators) in enumerate(selected):
        similarities, differences = _feature_comparison(target, indicators)
        evidence.append(
            ResearchEvidence(
                id=f"RE-HISTORY-{index + 1:02d}",
                evidence_type="inference",
                statement=(
                    f"{indicators.as_of.isoformat()}与当前状态相似度"
                    f"{100 / (1 + distance):.1f}分；相似点：{'、'.join(similarities) or '无明显项'}；"
                    f"差异：{'、'.join(differences) or '差异较小'}。"
                ),
                source="扶摇历史K线/本地相似度计算",
                effective_date=indicators.as_of.isoformat(),
                retrieved_at=plan.as_of,
                local_trace_id=None,
                response_hash=None,
                raw_value={"distance": distance, "similarity": 100 / (1 + distance)},
            )
        )
    conclusion = (
        f"找到{len(evidence)}个历史相似日期。相似度仅描述当时的趋势特征，"
        "不代表历史后续走势会重复。"
    )
    return _result(
        plan,
        conclusion,
        evidence,
        ["历史相似度不包含事件、政策和完整宏观环境。"],
        [] if evidence else ["RS-01"],
        ["RS-01"],
    )


def validate_research_compliance(result: ResearchResult) -> None:
    text = "\n".join(
        [result.conclusion, *result.uncertainties]
        + [item.statement for item in result.evidence]
    )
    matched = [phrase for phrase in RESEARCH_PROHIBITED_PHRASES if phrase in text]
    if matched:
        raise ValueError("继续研究结果包含预测或投资建议用语：" + "、".join(matched))


def _risk_series(result: IfindMcpResult, as_of: str) -> list[dict[str, Any]]:
    series = _structured_series(result.data)
    if not series:
        series = _markdown_series(result.data)
    output: list[dict[str, Any]] = []
    reference = date.fromisoformat(as_of)
    for name, points, unit in series:
        usable = sorted({day: value for day, value in points}.items())
        if len(usable) < 2 or usable[0][1] == 0:
            continue
        latest_date = usable[-1][0]
        first, latest = usable[0][1], usable[-1][1]
        freshness_days = max(0, (reference - date.fromisoformat(latest_date)).days)
        freshness = 1.0 if freshness_days <= 1 else 0.75 if freshness_days <= 3 else 0.25 if freshness_days <= 10 else 0.0
        if "收益率" in name or unit == "%" and "国债" in name:
            change = (latest - first) * 100
            change_unit = "bp"
            magnitude = abs(change) / 10
        else:
            change = (latest / first - 1) * 100
            change_unit = "%"
            magnitude = abs(change)
        output.append(
            {
                "name": name,
                "change": change,
                "change_unit": change_unit,
                "latest_date": latest_date,
                "freshness_days": freshness_days,
                "risk_score": magnitude * freshness,
                "statement": f"{name}区间变化{change:+.2f}{change_unit}，数据截至{latest_date}",
                "result": result,
            }
        )
    return output


def _structured_series(data: Any) -> list[tuple[str, list[tuple[str, float]], str]]:
    output = []
    datas = data.get("datas") if isinstance(data, dict) else None
    if not isinstance(datas, list):
        return output
    for block in datas:
        table = block.get("data", {}).get("data") if isinstance(block, dict) else None
        columns = block.get("data", {}).get("columns") if isinstance(block, dict) else None
        attrs = block.get("data", {}).get("attrs", {}) if isinstance(block, dict) else {}
        if not isinstance(table, list) or not isinstance(columns, list) or len(columns) < 2:
            continue
        for column_index, name in enumerate(columns[1:], start=1):
            points = []
            for row in table:
                if isinstance(row, list) and len(row) > column_index:
                    numeric = _number(row[column_index])
                    if numeric is not None:
                        points.append((_date_text(row[0]), numeric))
            unit = str(attrs.get(name, {}).get("unit", "")) if isinstance(attrs, dict) else ""
            if points:
                output.append((str(name), points, unit))
    return output


def _markdown_series(data: Any) -> list[tuple[str, list[tuple[str, float]], str]]:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    units: dict[str, str] = {}
    for row in _markdown_rows(data):
        name = _security_name(row)
        day = _row_date(row)
        if not name or not day:
            continue
        change_fields = [
            value for key, value in row.items() if "涨跌幅" in key
        ]
        if change_fields and all(_number(value) is None for value in change_fields):
            # iFinD may repeat the previous close on weekends/non-trading days.
            continue
        value_key = next(
            (
                key
                for key in row
                if key == "收盘价"
                or key.startswith("收盘价（")
                or key.startswith("收盘价(")
                or key == "最新价"
            ),
            next(
                (
                    key
                    for key in row
                    if key not in {"证券代码", "证券简称", "日期", "涨跌幅", "涨跌幅(收盘价)"}
                    and "涨跌幅" not in key
                    and _number(row[key]) is not None
                ),
                None,
            ),
        )
        if value_key is None:
            continue
        value = _number(row[value_key])
        if value is None:
            continue
        groups[name].append((day, value))
        units[name] = "%" if "收益率" in name else ""
    return [(name, points, units[name]) for name, points in groups.items()]


def _markdown_rows(data: Any) -> list[dict[str, str]]:
    answers = _find_key_values(data, "answer")
    rows: list[dict[str, str]] = []
    for answer in answers:
        if not isinstance(answer, str):
            continue
        lines = answer.splitlines()
        index = 0
        while index + 1 < len(lines):
            if lines[index].strip().startswith("|") and _is_separator(lines[index + 1]):
                header = _cells(lines[index])
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    values = _cells(lines[index])
                    if len(values) == len(header):
                        rows.append(dict(zip(header, values)))
                    index += 1
                continue
            index += 1
    return rows


def _highfreq_rows(data: Any) -> list[dict[str, Any]]:
    tables = data.get("tables") if isinstance(data, dict) else None
    if not isinstance(tables, list):
        return []
    rows = []
    for table in tables:
        if not isinstance(table, list) or len(table) < 2:
            continue
        header = table[0]
        if not isinstance(header, list):
            continue
        rows.extend(dict(zip(header, row)) for row in table[1:] if isinstance(row, list))
    return rows


def _news_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and "备注" not in item]
    if isinstance(data, dict):
        for key in ("data", "items", "records"):
            value = data.get(key)
            records = _news_records(value)
            if records:
                return records
    return []


def _normalize_news_date(value: Any) -> str | None:
    text = str(value or "").strip()
    compact = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", text)
    separated = re.search(r"(?<!\d)(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    matched = compact or separated
    if matched is None:
        return None
    try:
        return date(
            int(matched.group(1)), int(matched.group(2)), int(matched.group(3))
        ).isoformat()
    except ValueError:
        return None


def _canonical_news_url(value: str) -> str:
    if not value.startswith(("https://", "http://")):
        return ""
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def _news_title_fingerprint(title: str) -> str:
    text = re.sub(r"\b\d{6}\b", "", title.upper())
    return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", text)


def _news_titles_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 12 and shorter in longer:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.82


def _event_category_and_quality(text: str) -> tuple[str, str]:
    high_matches = [word for word in EVENT_HIGH_RELEVANCE_KEYWORDS if word in text]
    medium_matches = [word for word in EVENT_MEDIUM_RELEVANCE_KEYWORDS if word in text]
    low_matches = [word for word in EVENT_LOW_RELEVANCE_KEYWORDS if word in text]
    if high_matches:
        if any(word in text for word in ("国务院", "证监会", "金融监管", "财政部", "发改委", "监管")):
            category = "政策/监管"
        elif any(word in text for word in ("央行", "货币政策", "降准", "降息", "加息", "国债收益率")):
            category = "货币/利率"
        else:
            category = "宏观/跨市场"
        return category, "高"
    if medium_matches and not (low_matches and len(low_matches) >= len(medium_matches)):
        return "市场机制/公司行为", "中"
    return "普通市场信息", "低"


def _event_dimensions(text: str) -> tuple[str, ...]:
    matched = tuple(
        dimension
        for dimension, keywords in EVENT_DIMENSION_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    )
    return matched or ("重要事件",)


def _feature_distance(current: Any, candidate: Any) -> float:
    differences = (
        (current.return_20d_pct - candidate.return_20d_pct) / 10,
        (current.return_60d_pct - candidate.return_60d_pct) / 20,
        (current.close_vs_ma20_pct - candidate.close_vs_ma20_pct) / 5,
        (current.close_vs_ma60_pct - candidate.close_vs_ma60_pct) / 10,
        (current.max_drawdown_60d_pct - candidate.max_drawdown_60d_pct) / 15,
    )
    return math.sqrt(sum(value * value for value in differences))


def _feature_comparison(current: Any, candidate: Any) -> tuple[list[str], list[str]]:
    features = (
        ("20日收益", current.return_20d_pct, candidate.return_20d_pct, 3.0),
        ("60日收益", current.return_60d_pct, candidate.return_60d_pct, 5.0),
        ("MA20位置", current.close_vs_ma20_pct, candidate.close_vs_ma20_pct, 2.0),
        ("MA60位置", current.close_vs_ma60_pct, candidate.close_vs_ma60_pct, 3.0),
        ("最大回撤", current.max_drawdown_60d_pct, candidate.max_drawdown_60d_pct, 4.0),
    )
    similarities, differences = [], []
    for name, now, then, tolerance in features:
        target = similarities if abs(now - then) <= tolerance else differences
        target.append(f"{name}{then:+.1f}%")
    return similarities, differences


def _remote_evidence(
    evidence_id: str,
    result: IfindMcpResult,
    statement: str,
    *,
    effective_date: str | None,
    raw_value: Any,
    unit: str | None = None,
) -> ResearchEvidence:
    return ResearchEvidence(
        id=evidence_id,
        evidence_type="fact",
        statement=statement,
        source=f"iFinD MCP/{result.trace.server_type}.{result.trace.tool_name}",
        effective_date=effective_date,
        retrieved_at=result.trace.called_at,
        local_trace_id=result.trace.trace_id,
        response_hash=result.trace.response_hash,
        raw_value=raw_value,
        unit=unit,
    )


def _result(
    plan: ResearchPlan,
    conclusion: str,
    evidence: Iterable[ResearchEvidence],
    uncertainties: Iterable[str],
    failed_steps: Iterable[str],
    executed_steps: Iterable[str] = (),
) -> ResearchResult:
    evidence_tuple = tuple(evidence)
    failed_tuple = tuple(failed_steps)
    status = "failed" if not evidence_tuple else "partial" if failed_tuple else "complete"
    return ResearchResult(
        plan=plan,
        status=status,
        conclusion=conclusion,
        evidence=evidence_tuple,
        uncertainties=tuple(uncertainties),
        failed_steps=failed_tuple,
        executed_steps=tuple(dict.fromkeys(executed_steps)),
    )


def _mcp_steps(
    plan: ResearchPlan,
    *,
    tool_names: set[str] | None = None,
) -> tuple[ResearchStep, ...]:
    return tuple(
        item
        for item in plan.steps
        if item.execution_kind == "mcp"
        and (tool_names is None or item.tool_name in tool_names)
    )


def _plan_date_range(plan: ResearchPlan) -> TradingDateRange:
    for step in plan.steps:
        query = str(step.params.get("query", ""))
        matched = re.search(r"(\d{8})至(\d{8})", query)
        if matched:
            start, end = matched.groups()
            return TradingDateRange(
                f"{start[:4]}-{start[4:6]}-{start[6:]}",
                f"{end[:4]}-{end[4:6]}-{end[6:]}",
            )
    raise ValueError("研究计划缺少绝对日期区间")


def _find_key_values(value: Any, key: str) -> list[Any]:
    found = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                found.append(item)
            found.extend(_find_key_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_key_values(item, key))
    return found


def _is_separator(line: str) -> bool:
    clean = line.strip().replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
    return clean == "" and "-" in line


def _cells(line: str) -> list[str]:
    return [item.strip().replace("\t", "") for item in line.strip().strip("|").split("|")]


def _security_name(row: dict[str, str]) -> str:
    return str(row.get("证券简称") or row.get("指标名称") or "").strip()


def _row_date(row: dict[str, str]) -> str | None:
    value = str(row.get("日期") or "").strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value or None


def _date_text(value: Any) -> str:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _first_numeric(row: dict[str, str], *, includes: tuple[str, ...]) -> float | None:
    for key, value in row.items():
        if any(token in key for token in includes):
            parsed = _number(value)
            if parsed is not None:
                return parsed
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.lower() in {"", "null", "none", "--", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None
