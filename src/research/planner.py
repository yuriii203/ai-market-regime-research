from __future__ import annotations

from datetime import date, timedelta

from src.clients.ifind_mcp import absolute_trading_window
from src.research.models import ResearchPlan, ResearchQuestion, ResearchStep


INDUSTRY_BATCHES = (
    ("电子", "银行", "医药生物", "食品饮料", "计算机"),
    ("非银金融", "电力设备", "汽车", "有色金属", "机械设备"),
)


def build_research_plan(
    question: ResearchQuestion,
    *,
    subject_symbol: str,
    subject_name: str,
    as_of: str,
    trading_dates: tuple[str, ...],
) -> ResearchPlan:
    window = absolute_trading_window(
        trading_dates,
        as_of=as_of,
        count=20 if question == ResearchQuestion.LARGE_VS_SMALL else 5,
    )
    steps = _steps_for(question, subject_name, as_of, window.start_compact, window.end_compact)
    return ResearchPlan(
        schema_version="1.0",
        question=question,
        subject_symbol=subject_symbol,
        subject_name=subject_name,
        as_of=as_of,
        rationale=_rationale(question),
        steps=steps,
        stop_conditions=(
            "关键工具失败时停止生成确定性结论，并保留已取得证据。",
            "返回日期与计划日期不一致时拒绝使用该步数据。",
            "证据不足时输出不确定信息，不用常识补造事实。",
        ),
    )


def _steps_for(
    question: ResearchQuestion,
    subject_name: str,
    as_of: str,
    start: str,
    end: str,
) -> tuple[ResearchStep, ...]:
    if question == ResearchQuestion.LARGE_VS_SMALL:
        return (
            _mcp(
                "RS-01", "比较宽基指数", "index", "index_data",
                {"query": f"沪深300、中证1000在{start}至{end}的每日收盘点位和涨跌幅"},
                "大盘代理与小盘代理近20个交易日的同期表现",
            ),
            _local("RS-02", "结合现有风格轮动", "local.style_context", "扶摇风格指数与当前状态"),
            _mcp(
                "RS-03", "补充指数成交额与可用宽度", "index", "index_highfreq_quotes",
                {
                    "symbols": "000300.SH,000852.SH,399006.SZ",
                    "indicators": "最新价,涨跌幅,上涨家数,下跌家数,成交额",
                    "data_mode": "real_time",
                },
                "指数成交额及口径可确认的宽度字段",
            ),
            _mcp(
                "RS-04", "补充A股资金活跃样本", "stock", "search_stocks",
                {"query": f"{end}主力资金净流入排名前5的A股，返回代码、简称和主力资金净流入"},
                "主力资金净流入靠前的5只A股样本",
            ),
        )
    if question == ResearchQuestion.INDUSTRY_SUPPORT:
        return tuple(
            _mcp(
                f"RS-{index + 1:02d}", f"查询行业样本第{index + 1}批", "index", "sector_data",
                {"query": f"{'、'.join(batch)}这5个申万一级行业在{start}至{end}的区间涨跌幅"},
                "样本行业区间表现",
            )
            for index, batch in enumerate(INDUSTRY_BATCHES)
        )
    if question == ResearchQuestion.BIGGEST_RISK:
        return (
            _mcp(
                "RS-01", "搜索宏观风险指标", "edb", "search_edb",
                {"query": "人民币汇率、中国和美国十年期国债收益率等A股市场风险变量"},
                "候选指标ID、频率、单位与来源",
            ),
            _mcp(
                "RS-02", "获取中国十年期国债收益率", "edb", "get_edb_data",
                {"query": f"中债国债到期收益率:10年（{start}-{end}）"},
                "中国十年期国债收益率近期变化", ("RS-01",),
            ),
            _mcp(
                "RS-03", "获取美国十年期国债收益率", "edb", "get_edb_data",
                {"query": f"美国:国债收益率:10年（{start}-{end}）"},
                "美国十年期国债收益率近期变化", ("RS-01",),
            ),
            _mcp(
                "RS-04", "获取美元兑人民币", "edb", "get_edb_data",
                {"query": f"美元兑人民币（{start}-{end}）"},
                "美元兑人民币近期变化", ("RS-01",),
            ),
            _mcp(
                "RS-05", "获取海外市场表现", "index", "index_data",
                {"query": f"恒生指数、标普500指数、纳斯达克综合指数在{start}至{end}的每日收盘点位和涨跌幅"},
                "跨市场指数近期变化",
            ),
            _mcp(
                "RS-06", "获取大宗商品表现", "future", "future_quotes",
                {"query": f"原油主力合约、沪铜主力合约、沪金主力合约在{start}至{end}的收盘价和涨跌幅"},
                "大宗商品风险变量近期变化",
            ),
        )
    if question == ResearchQuestion.HISTORICAL_ANALOG:
        return (
            _local(
                "RS-01", "计算历史状态相似度", "local.historical_similarity",
                "扶摇历史K线中的相似日期、相似点与差异",
            ),
        )
    if question == ResearchQuestion.STATE_CHANGING_EVENTS:
        end_date = date.fromisoformat(as_of)
        start_date = end_date - timedelta(days=14)
        return (
            _mcp(
                "RS-01", "检索近期政策与财经事件", "news", "search_news",
                {
                    "query": f"与A股及{subject_name}当前市场状态相关的重要政策、宏观与财经新闻",
                    "time_start": start_date.isoformat(),
                    "time_end": end_date.isoformat(),
                    "size": 5,
                },
                "标题、相关片段、发布日期和原始URL",
            ),
        )
    raise ValueError(f"不支持的研究问题：{question}")


def _mcp(
    step_id: str,
    title: str,
    server_type: str,
    tool_name: str,
    params: dict,
    expected_output: str,
    depends_on: tuple[str, ...] = (),
) -> ResearchStep:
    return ResearchStep(
        step_id, title, "mcp", server_type, tool_name, params,
        expected_output, depends_on,
    )


def _local(step_id: str, title: str, tool_name: str, expected_output: str) -> ResearchStep:
    return ResearchStep(
        step_id, title, "local", None, tool_name, {}, expected_output, (),
    )


def _rationale(question: ResearchQuestion) -> str:
    return {
        ResearchQuestion.LARGE_VS_SMALL: (
            "先用沪深300作为大盘代理、中证1000作为小盘代理验证问题前提，"
            "再用同一20个交易日区间和现有风格证据解释差异；前提不成立时直接说明。"
        ),
        ResearchQuestion.INDUSTRY_SUPPORT: "分批查询代表性行业，在本地排序；没有权重时只描述领先或落后。",
        ResearchQuestion.BIGGEST_RISK: "先搜索宏观指标，再比较宏观、海外市场与商品变量的数据新鲜度和变化。",
        ResearchQuestion.HISTORICAL_ANALOG: "只用历史行情计算相似度，不假设历史结果会重复。",
        ResearchQuestion.STATE_CHANGING_EVENTS: "检索近期事件并保留原始URL，只识别可能影响的维度，不预测涨跌。",
    }[question]
