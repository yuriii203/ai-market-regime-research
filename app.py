from __future__ import annotations

import os
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analysis.trend_state import TrendState, evaluate_trend_state
from src.analysis.evidence import (
    DimensionAssessment,
    EvidenceBundle,
    EvidenceItem,
    build_evidence_bundle,
)
from src.analysis.deepseek_explainer import (
    MarketExplanation,
    build_local_fallback,
    explain_market_assessment,
)
from src.analysis.market_assessment import MarketAssessment, build_market_assessment
from src.clients.deepseek import DeepSeekClient, DeepSeekConfigurationError
from src.clients.ifind_mcp import IfindMcpClient, IfindMcpError
from src.clients.fuyao import (
    DataResult,
    FuyaoClient,
    FuyaoError,
    IndexConstituent,
    PoolResult,
    PriceBar,
    SHANGHAI_TZ,
    StockSnapshot,
    TradingDay,
    ValuationSnapshot,
    is_a_share_thscode,
    shanghai_date_to_ms,
)
from src.indicators.market_dimensions import (
    MarketSentiment,
    calculate_liquidity,
    calculate_market_breadth,
    calculate_market_sentiment,
    calculate_market_valuation,
    calculate_style_rotation,
)
from src.indicators.trend import (
    IndicatorDataError,
    TrendIndicators,
    calculate_trend_indicators,
)
from src.research.models import ResearchPlan, ResearchQuestion, ResearchResult
from src.research.planner import build_research_plan
from src.research.quota import DailyQuotaExceededError, get_research_quota
from src.research.services import (
    research_events,
    research_historical_analogs,
    research_industries,
    research_large_vs_small,
    research_market_supplement,
    research_risk_variables,
    validate_research_compliance,
)


load_dotenv()

INDEX_OPTIONS = {
    "000001.SH": "上证指数",
    "000300.SH": "沪深300",
    "000852.SH": "中证1000",
    "399006.SZ": "创业板指",
}


@dataclass(frozen=True)
class ConstituentUniverse:
    source: DataResult[IndexConstituent]
    eligible_items: tuple[IndexConstituent, ...]
    excluded_non_a_count: int


@dataclass(frozen=True)
class BreadthMarketData:
    universe: ConstituentUniverse
    results: tuple[DataResult[StockSnapshot], ...]
    sampled_count: int


@dataclass(frozen=True)
class ValuationMarketData:
    universe: ConstituentUniverse
    results: tuple[DataResult[ValuationSnapshot], ...]
    sampled_count: int


@dataclass(frozen=True)
class SentimentMarketData:
    limit_up: PoolResult
    limit_down: PoolResult
    limit_break: PoolResult
    query_date: str

st.set_page_config(
    page_title="AI 驱动的股票市场大势研判",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
    .hero-kicker {color: #D84A3A; font-weight: 700; letter-spacing: .08em; font-size: .82rem; line-height: 1.5;}
    .state-card {border-radius: 18px; padding: 1.35rem 1.5rem; border: 1px solid; margin: .5rem 0 1.2rem;}
    /* A股配色：红涨、绿跌。 */
    .state-positive {background: #FFF3F1; border-color: #E7B3AC;}
    .state-neutral {background: #FFF9EC; border-color: #E8D19C;}
    .state-negative {background: #F0F9F4; border-color: #A8D8BC;}
    .state-positive .hero-kicker {color: #D84A3A;}
    .state-negative .hero-kicker {color: #16845B;}
    .state-neutral .hero-kicker {color: #9A6700;}
    .state-label {font-size: 1.75rem; font-weight: 750; color: #172033;}
    .state-summary {color: #475467; margin-top: .35rem;}
    .state-note {font-size: .82rem; color: #7B8495; margin-top: .75rem;}
    .assessment-meta {display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .8rem; color: #667085; font-size: .82rem;}
    .assessment-tag {background: rgba(255,255,255,.72); border: 1px solid rgba(102,112,133,.2); border-radius: 999px; padding: .2rem .65rem;}
    .dimension-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .55rem; margin: .45rem 0 1rem;}
    .dimension-item {background: #F8FAFC; border: 1px solid #E8ECF2; border-radius: 10px; padding: .65rem .8rem;}
    .dimension-name {font-size: .76rem; color: #7B8495; margin-bottom: .15rem;}
    .dimension-value {font-size: .96rem; color: #344054; font-weight: 650;}
    .trend-detail {border-left: 3px solid #CBD5E1; padding: .15rem 0 .15rem .9rem; margin: .35rem 0 1rem;}
    .trend-detail-title {font-size: 1.05rem; font-weight: 700; color: #344054;}
    .trend-detail-copy {font-size: .9rem; color: #667085; margin-top: .25rem;}
    @media (max-width: 800px) {.dimension-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
    div[data-testid="stMetric"] {background: white; border: 1px solid #E5E9F0; padding: 1rem; border-radius: 14px;}
    div[data-testid="stMetricLabel"] {color: #667085;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_fuyao_market_data(
    symbol: str,
) -> tuple[DataResult[TradingDay], DataResult[PriceBar]]:
    client = FuyaoClient.from_env()
    calendar = client.get_trading_days()
    end_date = datetime.now(SHANGHAI_TZ).date()
    start_date = end_date - timedelta(days=200)
    history = client.get_index_history(
        symbol,
        start_ms=shanghai_date_to_ms(start_date),
        end_ms=shanghai_date_to_ms(end_date),
    )
    return calendar, history


@st.cache_data(ttl=300, show_spinner=False)
def load_style_market_data() -> dict[str, DataResult[PriceBar]]:
    client = FuyaoClient.from_env()
    end_date = datetime.now(SHANGHAI_TZ).date()
    start_date = end_date - timedelta(days=200)
    start_ms = shanghai_date_to_ms(start_date)
    end_ms = shanghai_date_to_ms(end_date)
    return {
        code: client.get_index_history(code, start_ms=start_ms, end_ms=end_ms)
        for code in INDEX_OPTIONS
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_constituent_universe(symbol: str) -> ConstituentUniverse:
    constituents = FuyaoClient.from_env().get_index_constituents(symbol)
    eligible = tuple(
        item for item in constituents.items if is_a_share_thscode(item.thscode)
    )
    if not eligible:
        raise ValueError(f"{symbol} 没有可用于A股接口的成分股")
    return ConstituentUniverse(
        source=constituents,
        eligible_items=eligible,
        excluded_non_a_count=len(constituents.items) - len(eligible),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_breadth_market_data(symbol: str) -> BreadthMarketData:
    client = FuyaoClient.from_env()
    universe = load_constituent_universe(symbol)
    sampled = _evenly_spaced_sample(list(universe.eligible_items), limit=300)
    code_batches = _constituent_code_batches(sampled)
    results = tuple(client.get_stock_snapshots(batch) for batch in code_batches)
    return BreadthMarketData(
        universe=universe,
        results=results,
        sampled_count=len(sampled),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_valuation_market_data(symbol: str) -> ValuationMarketData:
    client = FuyaoClient.from_env()
    universe = load_constituent_universe(symbol)
    sampled = _evenly_spaced_sample(list(universe.eligible_items), limit=300)
    code_batches = _constituent_code_batches(sampled)
    results = tuple(client.get_valuations(batch) for batch in code_batches)
    return ValuationMarketData(
        universe=universe,
        results=results,
        sampled_count=len(sampled),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment_market_data(latest_trading_day_ms: int) -> SentimentMarketData:
    client = FuyaoClient.from_env()
    query_date = datetime.fromtimestamp(
        latest_trading_day_ms / 1000, tz=SHANGHAI_TZ
    ).date().isoformat()
    return SentimentMarketData(
        limit_up=client.get_limit_pool("up", date_ms=latest_trading_day_ms),
        limit_down=client.get_limit_pool("down", date_ms=latest_trading_day_ms),
        limit_break=client.get_limit_pool("break", date_ms=latest_trading_day_ms),
        query_date=query_date,
    )


def _constituent_code_batches(
    items: list[IndexConstituent],
) -> list[list[str]]:
    return [
        [item.thscode for item in items[start : start + 100]]
        for start in range(0, len(items), 100)
    ]


def _evenly_spaced_sample(
    items: list[IndexConstituent], *, limit: int
) -> list[IndexConstituent]:
    if limit <= 0:
        raise ValueError("样本上限必须大于0")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    indexes = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in indexes]


def build_current_evidence_bundle(
    *,
    symbol: str,
    index_name: str,
    indicators: TrendIndicators,
    history: DataResult[PriceBar],
    calendar: DataResult[TradingDay],
    sources: dict[str, object],
    errors: dict[str, str],
) -> EvidenceBundle:
    style_source = sources.get("style")
    breadth_source = sources.get("breadth")
    sentiment_source = sources.get("sentiment")
    valuation_source = sources.get("valuation")
    universe = sources.get("universe")

    style_histories = (
        {
            INDEX_OPTIONS[code]: result
            for code, result in style_source.items()
        }
        if style_source is not None
        else None
    )
    breadth_snapshots = (
        [item for result in breadth_source.results for item in result.items]
        if breadth_source is not None
        else None
    )
    valuations = (
        [item for result in valuation_source.results for item in result.items]
        if valuation_source is not None
        else None
    )
    sentiment_pools = (
        (
            sentiment_source.limit_up,
            sentiment_source.limit_down,
            sentiment_source.limit_break,
        )
        if sentiment_source is not None
        else None
    )
    evidence_errors = {
        dimension: message
        for dimension, message in errors.items()
        if dimension in {"style", "breadth", "sentiment", "valuation"}
    }
    trading_dates = tuple(
        _normalize_calendar_date(item.date) for item in calendar.items
    )
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
    completed_dates = [value for value in trading_dates if value <= today]
    latest_completed_trading_date = (
        max(completed_dates) if completed_dates else indicators.as_of.isoformat()
    )
    return build_evidence_bundle(
        subject_symbol=symbol,
        subject_name=index_name,
        indicators=indicators,
        trend_metadata=history.metadata,
        style_histories=style_histories,
        breadth_snapshots=breadth_snapshots,
        breadth_metadata=(
            tuple(result.metadata for result in breadth_source.results)
            if breadth_source is not None
            else ()
        ),
        breadth_universe_count=(
            len(breadth_source.universe.eligible_items)
            if breadth_source is not None
            else None
        ),
        breadth_sampled_count=(
            breadth_source.sampled_count if breadth_source is not None else None
        ),
        sentiment_pools=sentiment_pools,
        sentiment_query_date=(
            sentiment_source.query_date if sentiment_source is not None else None
        ),
        sentiment_universe_codes=(
            {item.thscode for item in universe.eligible_items}
            if universe is not None
            else None
        ),
        valuations=valuations,
        valuation_metadata=(
            tuple(result.metadata for result in valuation_source.results)
            if valuation_source is not None
            else ()
        ),
        valuation_universe_count=(
            len(valuation_source.universe.eligible_items)
            if valuation_source is not None
            else None
        ),
        valuation_sampled_count=(
            valuation_source.sampled_count if valuation_source is not None else None
        ),
        breadth_effective_date=latest_completed_trading_date,
        valuation_effective_date=latest_completed_trading_date,
        trading_dates=trading_dates,
        errors=evidence_errors,
    )


def _normalize_calendar_date(value: str) -> str:
    clean = value.strip()
    if len(clean) == 8 and clean.isdigit():
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:]}"
    return clean


def render_trend_dimension_summary(
    state: TrendState,
    indicators: TrendIndicators,
    index_name: str,
) -> None:
    relation20 = "高于" if indicators.close_vs_ma20_pct >= 0 else "低于"
    relation60 = "高于" if indicators.close_vs_ma60_pct >= 0 else "低于"
    short_label = state.label.removeprefix("趋势")
    st.markdown(
        f"""
<div class="trend-detail">
  <div class="trend-detail-title">趋势维度：{escape(short_label)}</div>
  <div class="trend-detail-copy">{escape(state.summary)}</div>
  <div class="trend-detail-copy">近20日 {indicators.return_20d_pct:+.2f}%，近60日 {indicators.return_60d_pct:+.2f}%；最新收盘价{relation20}MA20、{relation60}MA60。</div>
  <div class="state-note">{escape(index_name)}数据截至 {indicators.as_of.isoformat()}。趋势是综合研判的一个输入维度，不是第二个综合结论。</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_rule_assessment(
    assessment: MarketAssessment, bundle: EvidenceBundle
) -> None:
    """Render the deterministic 6D-6F result without requiring an LLM."""
    st.subheader("程序综合研判")
    st.caption(
        "以下结论由本地确定性规则直接生成，不依赖DeepSeek；"
        "事实、程序推断与不确定信息分别展示。"
    )
    tone = (
        "positive"
        if "偏强" in assessment.state_label
        else "negative"
        if "偏弱" in assessment.state_label
        else "neutral"
    )
    conflict = escape(assessment.primary_conflict)
    st.markdown(
        f"""
<div class="state-card state-{tone}">
  <div class="hero-kicker">确定性综合状态 · {escape(assessment.subject_name)}（{escape(assessment.subject_symbol)}）</div>
  <div class="state-label">{escape(assessment.state_label)}</div>
  <div class="state-summary">{escape(assessment.conclusion)}</div>
  <div class="state-summary"><strong>主要矛盾：</strong>{conflict}</div>
  <div class="assessment-meta">
    <span class="assessment-tag">数据基准 {escape(assessment.assessment_date)}</span>
    <span class="assessment-tag">评估日期 {escape(assessment.evaluation_date)}</span>
    <span class="assessment-tag">规则版本 {escape(assessment.schema_version)}</span>
  </div>
  <div class="state-note">{escape(assessment.compliance_note)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    dimension_items = "".join(
        (
            '<div class="dimension-item">'
            f'<div class="dimension-name">{escape(_dimension_display_name(item.dimension))}</div>'
            f'<div class="dimension-value">{escape(_dimension_short_label(item))}</div>'
            "</div>"
        )
        for item in bundle.dimensions
    )
    st.markdown("**六个构成维度**")
    st.markdown(
        f'<div class="dimension-grid">{dimension_items}</div>',
        unsafe_allow_html=True,
    )
    st.caption("六个维度共同构成上方综合结论；单个维度不是第二个综合结论。")

    confidence_col, completeness_col, time_col = st.columns(3)
    confidence_col.metric(
        "综合置信度（证据质量）",
        f"{assessment.confidence.score:.1f}分",
        delta=f"{assessment.confidence.level}置信度",
        delta_color="off",
        help=assessment.confidence.meaning,
    )
    available_count = sum(item.available for item in bundle.dimensions)
    completeness_col.metric(
        "已返回维度",
        f"{available_count}/{len(bundle.dimensions)}",
        help=(
            "表示趋势、市场宽度、风格、流动性、市场情绪和估值是否已返回；"
            "不等于全量覆盖，具体质量请看置信度。"
        ),
    )
    time_col.metric(
        "数据时点",
        "存在交易日差异" if bundle.time_conflicts else "交易日一致",
        help="比较数据归属交易日；接口获取时间单独记录，不参与该判断。",
    )
    st.caption(assessment.confidence.meaning)

    conclusion_tab, evidence_tab, confidence_tab, switch_tab = st.tabs(
        ["结论与主要矛盾", "证据与不确定性", "置信度明细", "切换条件与时点"]
    )
    evidence_by_id = {item.id: item for item in bundle.evidence}

    with conclusion_tab:
        st.markdown("**综合结论**")
        st.write(assessment.conclusion)
        st.markdown("**主要矛盾**")
        st.write(assessment.primary_conflict)
        st.markdown("**六维状态摘要**")
        for dimension, summary in zip(bundle.dimensions, assessment.dimension_summaries):
            availability = "可用" if dimension.available else "不可用"
            st.write(f"• {_dimension_display_name(dimension.dimension)}（{availability}）：{summary}")

    with evidence_tab:
        support_col, counter_col, uncertainty_col = st.columns(3)
        with support_col:
            st.markdown("**支持综合结论**")
            _render_evidence_references(
                assessment.supporting_evidence_ids,
                evidence_by_id,
                "没有与综合方向一致的核心证据。",
            )
        with counter_col:
            st.markdown("**反向证据**")
            _render_evidence_references(
                assessment.counter_evidence_ids,
                evidence_by_id,
                "当前没有方向相反的核心证据。",
            )
        with uncertainty_col:
            st.markdown("**不确定信息**")
            _render_evidence_references(
                assessment.uncertainty_evidence_ids,
                evidence_by_id,
                "当前没有额外不确定项。",
            )
        with st.expander("查看完整证据台账与追踪信息"):
            ledger = pd.DataFrame(
                [
                    {
                        "证据编号": item.id,
                        "类别": _evidence_type_name(item.evidence_type.value),
                        "维度": _dimension_display_name(item.dimension),
                        "方向": _direction_display_name(item.direction.value),
                        "内容": item.description,
                        "范围": item.scope,
                        "有效日期": item.effective_date or "未提供",
                        "接口获取时间": item.retrieved_at or "未提供",
                        "来源": item.source,
                        "接口": item.endpoint or "本地规则",
                        "request_id": "、".join(item.request_ids) or "不适用",
                    }
                    for item in bundle.evidence
                ]
            )
            st.dataframe(ledger, hide_index=True, width="stretch")

    with confidence_tab:
        st.progress(min(1.0, max(0.0, assessment.confidence.score / 100)))
        component_rows = pd.DataFrame(
            [
                {
                    "评分项": item.label,
                    "满分": item.max_points,
                    "得分": item.awarded_points,
                    "扣分": round(item.max_points - item.awarded_points, 1),
                    "依据": item.reason,
                }
                for item in assessment.confidence.components
            ]
        )
        st.dataframe(component_rows, hide_index=True, width="stretch")
        st.markdown("**扣分原因**")
        if assessment.confidence.deduction_reasons:
            for reason in assessment.confidence.deduction_reasons:
                st.write(f"• {reason}")
        else:
            st.success("本次没有置信度扣分。")
        st.info("置信度只表示数据和证据质量，不是上涨概率，也不是历史胜率。")

    with switch_tab:
        st.markdown("**跨维度状态切换条件**")
        for condition in assessment.switch_conditions:
            st.write(f"• {condition.id}：{condition.description}")
            st.caption(
                "关联证据："
                + ("、".join(condition.related_evidence_ids) or "当前相关证据不可用")
            )
        st.markdown("**数据时点检查**")
        if bundle.time_conflicts:
            for warning in assessment.data_time_warnings:
                st.warning(warning)
        else:
            st.success(
                "各维度事实数据归属于同一交易日；接口获取时间已在证据台账中单独记录。"
            )


def _render_evidence_references(
    evidence_ids: tuple[str, ...],
    evidence_by_id: dict[str, EvidenceItem],
    empty_message: str,
) -> None:
    if not evidence_ids:
        st.caption(empty_message)
        return
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            st.warning(f"{evidence_id}：证据记录缺失")
            continue
        st.markdown(f"`{item.id}` · {_evidence_type_name(item.evidence_type.value)}")
        st.write(item.description)
        st.caption(
            f"{_dimension_display_name(item.dimension)} · {item.scope} · "
            f"{item.effective_date or '时点未提供'}"
        )


def _dimension_display_name(value: str) -> str:
    return {
        "trend": "趋势",
        "breadth": "市场宽度",
        "style": "风格轮动",
        "liquidity": "流动性",
        "sentiment": "市场情绪",
        "valuation": "估值",
    }.get(value, value)


def _dimension_short_label(item: DimensionAssessment) -> str:
    label = str(item.label)
    if item.dimension == "trend":
        return label.removeprefix("趋势")
    return label


def _evidence_type_name(value: str) -> str:
    return {"fact": "事实", "inference": "程序推断", "uncertainty": "不确定信息"}.get(
        value, value
    )


def _direction_display_name(value: str) -> str:
    return {
        "positive": "正向",
        "negative": "负向",
        "neutral": "中性",
        "context": "背景信息",
        "unknown": "未知",
    }.get(value, value)


AI_EVIDENCE_REFERENCE_PATTERN = re.compile(r"\[(E-[A-Z]+-(?:I|U)?\d+)\]")


def _humanize_ai_citations(text: str, labels: dict[str, str]) -> str:
    return AI_EVIDENCE_REFERENCE_PATTERN.sub(
        lambda matched: f"〔{labels.get(matched.group(1), '证据')}〕",
        text,
    )


def _render_ai_citation_ledger(
    explanation: MarketExplanation,
    bundle: EvidenceBundle,
) -> dict[str, str]:
    labels = {
        evidence_id: f"证据{index}"
        for index, evidence_id in enumerate(explanation.cited_evidence_ids, start=1)
    }
    evidence_by_id = {item.id: item for item in bundle.evidence}
    with st.expander("查看本次AI引用的证据"):
        rows = []
        for evidence_id, label in labels.items():
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            rows.append(
                {
                    "正文标记": label,
                    "说明": item.description,
                    "维度": _dimension_display_name(item.dimension),
                    "类型": _evidence_type_name(item.evidence_type.value),
                    "有效日期": item.effective_date or "未提供",
                    "内部证据编号": item.id,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("本次没有可展示的引用证据。")
    return labels


def render_ai_explanation(
    assessment: MarketAssessment, bundle: EvidenceBundle
) -> None:
    st.subheader("AI解释")
    st.caption(
        "DeepSeek只解释程序已经计算的结论和证据，不重新计算指标；"
        "输出会经过证据编号、数字和合规校验。"
    )
    assessment_key = hashlib.sha256(
        assessment.to_json().encode("utf-8")
    ).hexdigest()
    if st.button("生成 / 刷新 DeepSeek 解读", key="generate_deepseek_explanation"):
        try:
            client = DeepSeekClient.from_env()
        except DeepSeekConfigurationError as exc:
            explanation = build_local_fallback(
                assessment, bundle, reason=str(exc)
            )
        else:
            with st.spinner("DeepSeek正在解释结构化研判……"):
                explanation = explain_market_assessment(
                    assessment, bundle, client
                )
        st.session_state["deepseek_explanation_v1"] = {
            "assessment_key": assessment_key,
            "explanation": explanation,
        }

    cached = st.session_state.get("deepseek_explanation_v1")
    if not isinstance(cached, dict) or cached.get("assessment_key") != assessment_key:
        st.info("点击按钮后生成受约束的AI解读；不调用时不产生模型费用。")
        return
    explanation = cached.get("explanation")
    if not isinstance(explanation, MarketExplanation):
        st.warning("AI解释缓存格式无效，请重新生成。")
        return
    if explanation.source == "local_fallback":
        st.warning(
            "AI解释未通过证据追溯校验或接口暂时不可用。"
            "上方程序综合研判仍然有效，本页不展示重复的本地回退正文。"
        )
        with st.expander("查看AI调用与校验技术详情"):
            st.write(f"最终原因：{explanation.fallback_reason or '未提供'}")
            _render_ai_attempts(explanation)
        return
    else:
        st.caption(
            f"解释来源：DeepSeek · 模型：{explanation.model} · "
            f"request_id：{explanation.request_id or '未返回'}"
        )
    citation_labels = {
        evidence_id: f"证据{index}"
        for index, evidence_id in enumerate(explanation.cited_evidence_ids, start=1)
    }
    st.markdown(f"**{explanation.state_label}**")
    st.write(_humanize_ai_citations(explanation.overview, citation_labels))
    st.markdown("**主要矛盾**")
    st.write(_humanize_ai_citations(explanation.primary_conflict, citation_labels))
    st.markdown("**置信度说明**")
    st.write(_humanize_ai_citations(explanation.confidence, citation_labels))
    st.markdown("**不确定信息**")
    st.write(_humanize_ai_citations(explanation.uncertainty, citation_labels))
    st.markdown("**状态切换条件**")
    for condition in explanation.switch_conditions:
        st.write(
            f"{condition.condition_id}："
            f"{_humanize_ai_citations(condition.explanation, citation_labels)}"
        )
    _render_ai_citation_ledger(explanation, bundle)
    st.caption(assessment.compliance_note)
    with st.expander("查看AI调用与校验记录"):
        _render_ai_attempts(explanation)


def _render_ai_attempts(explanation: MarketExplanation) -> None:
    if not explanation.attempts:
        st.caption("没有可用的调用记录。")
        return
    rows = pd.DataFrame(
        [
            {
                "尝试": item.attempt_number,
                "状态": {
                    "accepted": "通过",
                    "rejected": "校验拒绝",
                    "transport_error": "接口失败",
                }.get(item.status, item.status),
                "模型": item.model or "未返回",
                "request_id": item.request_id or "未返回",
                "详情": item.detail,
            }
            for item in explanation.attempts
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")


def make_chart_data(history: DataResult[PriceBar]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "交易日": [bar.trading_date for bar in history.items],
            "收盘点位": [bar.close_price for bar in history.items],
        }
    )
    frame["交易日"] = pd.to_datetime(frame["交易日"])
    frame = frame.set_index("交易日")
    frame["MA20"] = frame["收盘点位"].rolling(20).mean()
    frame["MA60"] = frame["收盘点位"].rolling(60).mean()
    return frame


def make_index_chart(chart_data: pd.DataFrame, index_name: str) -> alt.Chart:
    display_data = chart_data.rename(
        columns={"MA20": "20日均线", "MA60": "60日均线"}
    )
    long_data = (
        display_data.reset_index()
        .melt(id_vars="交易日", var_name="序列", value_name="点位")
        .dropna()
    )
    start = chart_data.index.min().strftime("%Y-%m-%d")
    end = chart_data.index.max().strftime("%Y-%m-%d")
    return (
        alt.Chart(long_data)
        .mark_line()
        .encode(
            x=alt.X(
                "交易日:T",
                title=None,
                scale=alt.Scale(domain=[start, end]),
                axis=alt.Axis(format="%m-%d", labelAngle=0, tickCount=8),
            ),
            y=alt.Y(
                "点位:Q",
                title=f"{index_name}点位",
                scale=alt.Scale(zero=False, nice=True),
            ),
            color=alt.Color(
                "序列:N",
                title=None,
                scale=alt.Scale(
                    domain=["收盘点位", "20日均线", "60日均线"],
                    range=["#D84A3A", "#2474D2", "#74A9E6"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            strokeWidth=alt.condition(
                alt.datum.序列 == "收盘点位", alt.value(2.8), alt.value(1.8)
            ),
            tooltip=[
                alt.Tooltip("交易日:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("序列:N", title="指标"),
                alt.Tooltip("点位:Q", title="点位", format=",.2f"),
            ],
        )
        .properties(height=420)
    )


def describe_turnover_change(value: float | None) -> str:
    if value is None:
        return "数据不可用"
    magnitude = abs(value)
    if magnitude < 5:
        return "基本持平"
    if magnitude < 20:
        return "温和放大" if value > 0 else "温和收缩"
    return "明显放大" if value > 0 else "明显收缩"


def render_metric_panel(indicators: TrendIndicators) -> None:
    st.subheader("趋势指标")
    close_col, return20_col, return60_col, drawdown_col = st.columns(4)
    close_col.metric(
        "最新收盘点位",
        f"{indicators.close_price:,.2f}",
        help=f"{indicators.as_of.isoformat()} 的收盘点位。",
    )
    return20_col.metric(
        "20日收益率（约1个月）",
        f"{indicators.return_20d_pct:+.2f}%",
        help="最新收盘价相对20个交易日前的变化。",
    )
    return60_col.metric(
        "60日收益率（约1季度）",
        f"{indicators.return_60d_pct:+.2f}%",
        help="最新收盘价相对60个交易日前的变化。",
    )
    drawdown_col.metric(
        "近60日最大回撤",
        f"{indicators.max_drawdown_60d_pct:.2f}%",
        help="最近60个交易日从阶段高点到后续低点的最大跌幅，显示为正的损失幅度。",
    )

    turnover_col, ma20_col, ma60_col = st.columns(3)
    turnover_text = (
        "不可用"
        if indicators.turnover_5d_vs_20d_pct is None
        else f"{indicators.turnover_5d_vs_20d_pct:+.2f}%"
    )
    turnover_col.metric(
        "近5日成交额较20日均值",
        turnover_text,
        delta=describe_turnover_change(indicators.turnover_5d_vs_20d_pct),
        delta_color="off",
        help="近5日平均成交额相对近20日平均成交额的变化。该指标描述交易活跃度，不直接代表涨跌方向。",
    )
    ma20_col.metric(
        "MA20（近1月平均点位）",
        f"{indicators.ma20:,.2f}",
        delta=f"{indicators.close_vs_ma20_pct:+.2f}%",
        delta_color="inverse",
        help="采用A股配色：下方绿色负值表示最新收盘价低于MA20；红色正值表示高于MA20。",
    )
    ma60_col.metric(
        "MA60（近1季度平均点位）",
        f"{indicators.ma60:,.2f}",
        delta=f"{indicators.close_vs_ma60_pct:+.2f}%",
        delta_color="inverse",
        help="采用A股配色：下方绿色负值表示最新收盘价低于MA60；红色正值表示高于MA60。",
    )
    st.caption(
        f"最大回撤 {indicators.max_drawdown_60d_pct:.2f}% 表示近60日从阶段高点最多下跌约"
        f" {indicators.max_drawdown_60d_pct:.2f}%；成交额变化只反映活跃度。"
    )


def render_evidence_panel(state: TrendState) -> None:
    st.subheader("趋势单维度依据")
    st.caption("这一部分只解释指数趋势，是上方综合研判的一个组成维度。")
    positive_col, negative_col, switch_col = st.columns(3)
    with positive_col:
        st.markdown("**支持偏强的证据**")
        if state.supporting_evidence:
            for evidence in state.supporting_evidence:
                st.write(f"• {evidence}")
        else:
            st.caption("当前没有支持偏强的趋势证据")
    with negative_col:
        st.markdown("**支持偏弱的证据**")
        if state.contradicting_evidence:
            for evidence in state.contradicting_evidence:
                st.write(f"• {evidence}")
        else:
            st.caption("当前没有支持偏弱的趋势证据")
    with switch_col:
        st.markdown("**可能改变判断的条件**")
        for condition in state.switch_conditions:
            st.write(f"• {condition}")


def render_market_dimensions(
    sources: dict[str, object],
    errors: dict[str, str],
    indicators: TrendIndicators,
    index_name: str,
) -> None:
    st.subheader("扩展市场维度")
    st.caption(
        "各维度独立加载、独立标注范围与时间，暂不合成为涨跌预测。"
        "历史日K、最新快照和估值可能具有不同可用时点。"
    )
    style_tab, breadth_tab, liquidity_tab, sentiment_tab, valuation_tab = st.tabs(
        ["风格/宽基", "市场宽度", "流动性", "市场情绪", "估值"]
    )

    with style_tab:
        if "style" in errors or "style" not in sources:
            st.warning(f"宽基相对表现不可用：{errors.get('style', '请重新生成市场研判')}")
        else:
            style_sources = sources["style"]
            style = calculate_style_rotation(
                {
                    INDEX_OPTIONS[code]: result.items
                    for code, result in style_sources.items()
                }
            )
            leader_col, laggard_col, spread_col = st.columns(3)
            leader_col.metric("20日相对领先（可能仍为负）", style.leader)
            laggard_col.metric("20日相对落后", style.laggard)
            spread_col.metric(
                "领先—落后差距", f"{style.spread_20d_pct_points:.2f}个百分点"
            )
            style_frame = pd.DataFrame(
                [
                    {
                        "宽基/风格代理": row.name,
                        "20日收益率": f"{row.return_20d_pct:+.2f}%",
                        "60日收益率": f"{row.return_60d_pct:+.2f}%",
                    }
                    for row in style.rows
                ]
            )
            st.dataframe(style_frame, hide_index=True, width="stretch")
            style_dates = sorted(
                {result.items[-1].trading_date.isoformat() for result in style_sources.values()}
            )
            style_retrieved_at = max(
                result.metadata.retrieval_time for result in style_sources.values()
            )
            st.caption(
                "范围：四个宽基指数；数据截至："
                + "、".join(style_dates)
                + f"；接口获取时间：{style_retrieved_at:%Y-%m-%d %H:%M:%S}。"
                + "领先仅表示过去20日相对表现较好，不等于上涨。"
            )
            with st.expander("查看风格/宽基追踪信息"):
                st.code(
                    "\n".join(
                        f"{code}_request_id={result.metadata.request_id}"
                        for code, result in style_sources.items()
                    ),
                    language=None,
                )

    with breadth_tab:
        if "breadth" in errors or "breadth" not in sources:
            st.warning(
                f"{index_name}市场宽度不可用："
                f"{errors.get('breadth', '请重新生成市场研判')}"
            )
        else:
            breadth_sources = sources["breadth"]
            snapshots = [
                item for result in breadth_sources.results for item in result.items
            ]
            breadth = calculate_market_breadth(
                snapshots, len(breadth_sources.universe.eligible_items)
            )
            up_col, down_col, flat_col, score_col = st.columns(4)
            up_col.metric("上涨家数（红涨）", breadth.advancing)
            down_col.metric("下跌家数（绿跌）", breadth.declining)
            flat_col.metric("平盘家数", breadth.flat)
            score_col.metric(
                "宽度净值",
                f"{breadth.breadth_score_pct:+.1f}%",
                help="（上涨家数－下跌家数）÷ 有效样本数。",
            )
            breadth_retrieved_at = max(
                result.metadata.retrieval_time for result in breadth_sources.results
            )
            universe = breadth_sources.universe
            st.caption(
                f"范围：{index_name}的A股成分；数据归属交易日：{indicators.as_of.isoformat()}；"
                f"接口获取时间：{breadth_retrieved_at:%Y-%m-%d %H:%M:%S}。"
                f"原始成分 {len(universe.source.items)} 只，排除非A股 {universe.excluded_non_a_count} 只；"
                f"抽样 {breadth_sources.sampled_count} 只，有效 {breadth.valid_count} 只，"
                f"相对A股成分覆盖率 {breadth.coverage_pct:.1f}%。"
            )
            with st.expander("查看市场宽度追踪信息"):
                st.code(
                    "\n".join(
                        [f"constituents_request_id={universe.source.metadata.request_id}"]
                        + [
                            f"snapshot_request_id={result.metadata.request_id}"
                            for result in breadth_sources.results
                        ]
                    ),
                    language=None,
                )

    with liquidity_tab:
        liquidity = calculate_liquidity(indicators)
        average = liquidity.average_turnover_20d
        change = liquidity.change_5d_vs_20d_pct
        average_text = "不可用" if average is None else f"{average / 1e8:,.2f}亿元"
        change_text = "不可用" if change is None else f"{change:+.2f}%"
        avg_col, change_col = st.columns(2)
        avg_col.metric("指数近20日平均成交额", average_text)
        change_col.metric(
            "近5日较20日均值", change_text, delta=describe_turnover_change(change), delta_color="off"
        )
        st.caption(
            f"范围：{index_name}指数；数据截至：{indicators.as_of.isoformat()}。"
            "成交额衡量交易活跃度。放量可能发生在上涨或下跌中，"
            "因此不使用红绿涨跌色，也不单独推导方向。"
        )

    with sentiment_tab:
        if "sentiment" in errors or "sentiment" not in sources:
            st.warning(f"市场情绪不可用：{errors.get('sentiment', '请重新生成市场研判')}")
        else:
            sentiment_sources = sources["sentiment"]
            full_sentiment = calculate_market_sentiment(
                sentiment_sources.limit_up,
                sentiment_sources.limit_down,
                sentiment_sources.limit_break,
            )
            sentiment_retrieved_at = max(
                sentiment_sources.limit_up.metadata.retrieval_time,
                sentiment_sources.limit_down.metadata.retrieval_time,
                sentiment_sources.limit_break.metadata.retrieval_time,
            )
            st.markdown("**全A市场情绪**")
            _render_sentiment_metrics(full_sentiment)
            st.caption(
                f"范围：全A；交易日：{sentiment_sources.query_date}。"
                f"接口获取时间：{sentiment_retrieved_at:%Y-%m-%d %H:%M:%S}。"
                "该组数据与下拉框所选指数无关，仅作市场背景。"
            )

            if "universe" in errors or "universe" not in sources:
                st.warning(
                    f"无法计算{index_name}成分股情绪："
                    f"{errors.get('universe', '请重新生成市场研判')}"
                )
            else:
                universe = sources["universe"]
                selected_sentiment = calculate_market_sentiment(
                    sentiment_sources.limit_up,
                    sentiment_sources.limit_down,
                    sentiment_sources.limit_break,
                    universe_codes={item.thscode for item in universe.eligible_items},
                )
                st.markdown(f"**{index_name}的A股成分情绪**")
                _render_sentiment_metrics(selected_sentiment)
                st.caption(
                    f"范围：{len(universe.eligible_items)} 只A股成分；"
                    f"已排除 {universe.excluded_non_a_count} 只非A股；"
                    f"交易日：{sentiment_sources.query_date}。"
                    f"接口获取时间：{sentiment_retrieved_at:%Y-%m-%d %H:%M:%S}。"
                )
            with st.expander("查看市场情绪追踪信息"):
                st.code(
                    "\n".join(
                        f"{name}_request_id={request_id}"
                        for name, pool in (
                            ("limit_up", sentiment_sources.limit_up),
                            ("limit_down", sentiment_sources.limit_down),
                            ("limit_break", sentiment_sources.limit_break),
                        )
                        for request_id in (pool.request_ids or (pool.metadata.request_id,))
                    ),
                    language=None,
                )

    with valuation_tab:
        if "valuation" in errors or "valuation" not in sources:
            st.warning(
                f"{index_name}估值不可用："
                f"{errors.get('valuation', '请重新生成市场研判')}"
            )
        else:
            valuation_sources = sources["valuation"]
            valuations = [
                item for result in valuation_sources.results for item in result.items
            ]
            valuation = calculate_market_valuation(
                valuations, len(valuation_sources.universe.eligible_items)
            )
            pe_col, pb_col, coverage_col = st.columns(3)
            pe = valuation.median_pe_ttm
            pb = valuation.median_pb_mrq
            pe_col.metric("正值PE(TTM)中位数", "不可用" if pe is None else f"{pe:.2f}倍")
            pb_col.metric("正值PB(MRQ)中位数", "不可用" if pb is None else f"{pb:.2f}倍")
            coverage_col.metric("估值记录覆盖率", f"{valuation.coverage_pct:.1f}%")
            valuation_retrieved_at = max(
                result.metadata.retrieval_time for result in valuation_sources.results
            )
            universe = valuation_sources.universe
            pe_rate = valuation.pe_valid_count / valuation.sample_count * 100
            pb_rate = valuation.pb_valid_count / valuation.sample_count * 100
            st.caption(
                f"范围：{index_name}的A股成分；数据归属交易日：{indicators.as_of.isoformat()}；"
                f"接口获取时间：{valuation_retrieved_at:%Y-%m-%d %H:%M:%S}。"
                f"原始成分 {len(universe.source.items)} 只，排除非A股 {universe.excluded_non_a_count} 只；"
                f"抽样 {valuation_sources.sampled_count} 只。PE有效率 {pe_rate:.1f}%，"
                f"PB有效率 {pb_rate:.1f}%；负值与空值不参与中位数。"
            )
            st.caption("接口不提供历史估值分位，因此不能仅凭该截面判断高估或低估。")
            with st.expander("查看估值追踪信息"):
                st.code(
                    "\n".join(
                        [f"constituents_request_id={universe.source.metadata.request_id}"]
                        + [
                            f"valuation_request_id={result.metadata.request_id}"
                            for result in valuation_sources.results
                        ]
                    ),
                    language=None,
                )


def _render_sentiment_metrics(sentiment: MarketSentiment) -> None:
    up_col, down_col, break_col, streak_col = st.columns(4)
    up_col.metric("涨停家数", sentiment.limit_up_count)
    down_col.metric("跌停家数", sentiment.limit_down_count)
    break_col.metric("炸板家数", sentiment.limit_break_count)
    streak = sentiment.max_consecutive_limit_up
    streak_col.metric("最高连板", "无" if streak is None else f"{streak}板")
    break_rate = sentiment.break_rate_pct
    st.caption(
        "炸板率="
        + ("不可用" if break_rate is None else f"{break_rate:.1f}%")
        + "（炸板家数 ÷〔涨停家数＋炸板家数〕）。"
    )


def render_details(
    calendar: DataResult[TradingDay],
    history: DataResult[PriceBar],
    indicators: TrendIndicators,
    chart_data: pd.DataFrame,
    index_name: str,
    symbol: str,
) -> None:
    source_tab, formula_tab, quality_tab, raw_tab = st.tabs(
        ["数据时间与来源", "计算公式", "数据异常检查", "最近20日数据"]
    )
    latest_bar = history.items[-1]
    latest_calendar_day = calendar.items[-1]

    with source_tab:
        st.markdown("**数据来源：扶摇金融数据 API**")
        source_rows = pd.DataFrame(
            [
                {
                    "数据": "A股交易日历",
                    "接口": "/api/a-share/calendar/trading-days",
                    "数据截至": latest_calendar_day.date,
                    "接口获取时间": calendar.metadata.retrieval_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "追踪ID": f"{calendar.metadata.request_id[:8]}…",
                },
                {
                    "数据": f"{index_name}历史日K",
                    "接口": "/api/a-share-index/prices/historical",
                    "数据截至": latest_bar.trading_date.isoformat(),
                    "接口获取时间": history.metadata.retrieval_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "追踪ID": f"{history.metadata.request_id[:8]}…",
                },
            ]
        )
        st.dataframe(source_rows, hide_index=True, width="stretch")
        st.caption(f"标的：{index_name}（{symbol}）；周期：日线（1d）；时区：Asia/Shanghai。")
        with st.expander("查看完整技术追踪信息"):
            st.code(
                "\n".join(
                    [
                        f"calendar_request_id={calendar.metadata.request_id}",
                        f"history_request_id={history.metadata.request_id}",
                        f"calendar_retrieved_at={calendar.metadata.retrieval_time.isoformat()}",
                        f"history_retrieved_at={history.metadata.retrieval_time.isoformat()}",
                        f"calendar_source_timestamp={calendar.metadata.timestamp.isoformat()}",
                        f"history_source_timestamp={history.metadata.timestamp.isoformat()}",
                    ]
                ),
                language=None,
            )

    with formula_tab:
        formulas = pd.DataFrame(
            [
                ["20日收益率（约1个月）", "最新收盘价 ÷ 20个交易日前收盘价 - 1"],
                ["60日收益率（约1季度）", "最新收盘价 ÷ 60个交易日前收盘价 - 1"],
                ["MA20（近1月平均点位）", "最近20根日K收盘价的算术平均值"],
                ["MA60（近1季度平均点位）", "最近60根日K收盘价的算术平均值"],
                ["近60日最大回撤", "最近60日内，阶段高点至后续低点的最大跌幅"],
                ["近5日成交额较20日均值", "近5日平均成交额 ÷ 近20日平均成交额 - 1"],
            ],
            columns=["指标", "计算公式/口径"],
        )
        st.dataframe(formulas, hide_index=True, width="stretch")
        st.caption(
            "收益率和成交额变化以百分比展示；最大回撤显示为正的损失幅度。"
            "成交额变化绝对值小于5%标为基本持平，5%—20%标为温和变化，20%以上标为明显变化。"
        )

    with quality_tab:
        anomalies: list[str] = list(indicators.warnings)
        calendar_latest_date = datetime.strptime(
            latest_calendar_day.date, "%Y%m%d"
        ).date()
        if latest_bar.trading_date != calendar_latest_date:
            anomalies.append(
                "最新K线交易日与交易日历最新日期不一致："
                f"K线={latest_bar.trading_date.isoformat()}，"
                f"日历={latest_calendar_day.date}"
            )
        if history.metadata.request_id == "" or calendar.metadata.request_id == "":
            anomalies.append("至少一个接口缺少request_id，证据追溯不完整")

        if anomalies:
            st.warning("检测到数据异常，本页未静默忽略：")
            for anomaly in anomalies:
                st.write(f"• {anomaly}")
        else:
            st.success("检查通过：数据非空、日期一致、request_id完整、指标输入有效。")

        st.markdown("**风险背景**")
        state = evaluate_trend_state(indicators)
        for context in state.risk_context:
            st.write(f"• {context}")

    with raw_tab:
        st.dataframe(chart_data.tail(20), width="stretch")


def execute_continue_research(
    question: ResearchQuestion,
    *,
    symbol: str,
    index_name: str,
    calendar: DataResult[TradingDay],
    history: DataResult[PriceBar],
    sources: dict[str, object],
) -> tuple[ResearchPlan, tuple[ResearchResult, ...]]:
    """Execute one fixed research route without changing the stage-six objects."""
    as_of = history.items[-1].trading_date.isoformat()
    trading_dates = tuple(_normalize_calendar_date(item.date) for item in calendar.items)
    plan = build_research_plan(
        question,
        subject_symbol=symbol,
        subject_name=index_name,
        as_of=as_of,
        trading_dates=trading_dates,
    )
    remote_call_count = sum(
        step.execution_kind == "mcp" for step in plan.steps
    )
    if remote_call_count:
        get_research_quota().reserve(remote_call_count)
    if question == ResearchQuestion.HISTORICAL_ANALOG:
        results = (research_historical_analogs(plan, list(history.items)),)
    else:
        client = IfindMcpClient.from_env()
        if question == ResearchQuestion.STATE_CHANGING_EVENTS:
            results = (research_events(plan, client),)
        elif question == ResearchQuestion.INDUSTRY_SUPPORT:
            results = (research_industries(plan, client),)
        elif question == ResearchQuestion.BIGGEST_RISK:
            results = (research_risk_variables(plan, client),)
        elif question == ResearchQuestion.LARGE_VS_SMALL:
            style_source = sources.get("style")
            if not isinstance(style_source, dict) or not style_source:
                raise ValueError("现有风格数据不可用，无法解释大盘与小盘差异")
            style = calculate_style_rotation(
                {
                    INDEX_OPTIONS[code]: list(result.items)
                    for code, result in style_source.items()
                }
            )
            primary = research_large_vs_small(plan, client, style)
            results_list = [primary]
            breadth_source = sources.get("breadth")
            if isinstance(breadth_source, BreadthMarketData):
                snapshots = [
                    item
                    for batch in breadth_source.results
                    for item in batch.items
                ]
                breadth = calculate_market_breadth(
                    snapshots, len(breadth_source.universe.eligible_items)
                )
                results_list.append(
                    research_market_supplement(plan, client, breadth)
                )
            results = tuple(results_list)
        else:
            raise ValueError(f"不支持的研究问题：{question.value}")
    for result in results:
        validate_research_compliance(result)
    return plan, results


def _research_raw_value(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= 500 else rendered[:497] + "..."


def _research_evidence_quality(item: object) -> str:
    raw_value = getattr(item, "raw_value", None)
    if isinstance(raw_value, dict) and raw_value.get("evidence_quality"):
        return str(raw_value["evidence_quality"])
    if getattr(item, "effective_date", None) and (
        getattr(item, "local_trace_id", None) or "本地计算" in getattr(item, "source", "")
    ):
        return "高"
    if getattr(item, "effective_date", None):
        return "中"
    return "低"


def _research_step_status(
    step_id: str,
    results: tuple[ResearchResult, ...],
) -> str:
    failed = {item for result in results for item in result.failed_steps}
    executed = {item for result in results for item in result.executed_steps}
    if step_id in failed:
        return "失败"
    if step_id in executed:
        return "完成"
    return "未执行"


def render_continue_research(
    *,
    symbol: str,
    index_name: str,
    calendar: DataResult[TradingDay],
    history: DataResult[PriceBar],
    sources: dict[str, object],
) -> None:
    st.subheader("继续研究")
    st.caption(
        "Agent按固定问题制定取数计划并选择iFinD MCP工具；数据获取、排序和历史相似度均由程序执行。"
        "单项研究失败不会改变上方阶段六综合研判，DeepSeek不参与工具选择或计算。"
    )
    try:
        quota = get_research_quota().snapshot()
        st.caption(
            f"每日iFinD继续研究额度：已用 {quota.used}/{quota.limit} 次，"
            f"剩余 {quota.remaining} 次（上海时区每日重置）。"
        )
    except ValueError as exc:
        st.warning(f"每日额度配置无效：{exc}")
    questions = tuple(ResearchQuestion)
    columns = st.columns(3)
    selected_question: ResearchQuestion | None = None
    for index, question in enumerate(questions):
        if columns[index % 3].button(
            question.value,
            key=f"continue_research_{question.name.lower()}",
            width="stretch",
        ):
            selected_question = question

    if selected_question is not None:
        with st.spinner(f"正在研究：{selected_question.value}"):
            try:
                plan, results = execute_continue_research(
                    selected_question,
                    symbol=symbol,
                    index_name=index_name,
                    calendar=calendar,
                    history=history,
                    sources=sources,
                )
            except (IfindMcpError, DailyQuotaExceededError, ValueError) as exc:
                st.session_state["continue_research_v1"] = {
                    "symbol": symbol,
                    "question": selected_question,
                    "error": str(exc),
                }
            else:
                st.session_state["continue_research_v1"] = {
                    "symbol": symbol,
                    "question": selected_question,
                    "plan": plan,
                    "results": results,
                }

    cached = st.session_state.get("continue_research_v1")
    if not isinstance(cached, dict) or cached.get("symbol") != symbol:
        return
    question = cached["question"]
    st.markdown(f"### 研究问题：{question.value}")
    if "error" in cached:
        st.warning(f"本次继续研究未完成：{cached['error']}")
        st.caption("该错误仅保存在继续研究区域；阶段六综合结论、证据与置信度均未改变。")
        return

    plan: ResearchPlan = cached["plan"]
    results: tuple[ResearchResult, ...] = cached["results"]
    st.markdown("**Agent制定的取数计划**")
    st.write(plan.rationale)
    for step in plan.steps:
        status = _research_step_status(step.id, results)
        icon = {"完成": "✅", "失败": "❌", "未执行": "⏸️"}[status]
        tool = (
            f"{step.server_type}.{step.tool_name}"
            if step.server_type
            else step.tool_name
        )
        with st.expander(
            f"{icon} {step.id} · {step.title} · {status}",
            expanded=status == "失败",
        ):
            detail_col1, detail_col2 = st.columns(2)
            detail_col1.write(
                f"**执行方式：**{'iFinD MCP' if step.execution_kind == 'mcp' else '本地程序'}"
            )
            detail_col2.write(f"**选择的工具：**`{tool}`")
            st.write(f"**预期输出：**{step.expected_output}")
            if step.params:
                st.markdown("**调用参数**")
                st.json(step.params, expanded=False)
    st.caption(
        "状态口径：点击前为等待执行，页面旋转提示表示执行中；完成后按实际记录显示"
        "“完成、未执行或失败”。若只有部分步骤成功，程序归纳显示“部分完成”。"
    )
    st.caption("停止条件：" + "；".join(plan.stop_conditions))

    st.markdown("**程序归纳**")
    for result in results:
        status_label = {"complete": "完成", "partial": "部分完成", "failed": "失败"}.get(
            result.status, result.status
        )
        st.write(f"• [{status_label}] {result.conclusion}")

    evidence_rows = []
    trace_rows = []
    event_items = []
    for result in results:
        for item in result.evidence:
            quality = _research_evidence_quality(item)
            evidence_rows.append(
                {
                    "证据编号": item.id,
                    "结构化证据": item.statement,
                    "数据有效日": item.effective_date or "未提供",
                    "来源": item.source,
                    "证据质量": quality,
                }
            )
            trace_rows.append(
                {
                    "证据编号": item.id,
                    "类型": item.evidence_type,
                    "原始数值": _research_raw_value(item.raw_value),
                    "单位": item.unit or "—",
                    "接口获取时间": item.retrieved_at,
                    "本地trace_id": item.local_trace_id or "本地计算",
                    "响应哈希": item.response_hash or "不适用",
                }
            )
            if isinstance(item.raw_value, dict):
                url = str(item.raw_value.get("url") or "").strip()
                if item.id.startswith("RE-EVENT-"):
                    event_items.append((item, quality, url))

    if event_items:
        st.markdown("**重要事件线索**")
        for item, quality, url in event_items:
            raw = item.raw_value
            with st.container(border=True):
                st.markdown(f"**{raw.get('title', item.statement)}**")
                metadata = [
                    f"日期：{item.effective_date or '未提供'}",
                    f"类别：{raw.get('event_category', '未分类')}",
                    f"证据质量：{quality}",
                    f"关联维度：{'、'.join(raw.get('related_dimensions', ())) or '未识别'}",
                ]
                if raw.get("publisher"):
                    metadata.insert(1, f"来源：{raw['publisher']}")
                st.caption(" · ".join(metadata))
                if raw.get("summary"):
                    st.write(raw["summary"])
                if url.startswith(("https://", "http://")):
                    st.link_button("查看原始新闻", url)

    st.markdown("**结构化证据**")
    if evidence_rows:
        st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, width="stretch")
    else:
        st.info("本次没有取得可用的结构化证据。")
    with st.expander("技术追踪详情（原始数值、trace_id与响应哈希）"):
        if trace_rows:
            st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")
        else:
            st.write("本次没有技术追踪记录。")

    uncertainties = tuple(
        dict.fromkeys(item for result in results for item in result.uncertainties)
    )
    st.markdown("**不确定性**")
    if uncertainties:
        for item in uncertainties:
            st.write(f"• {item}")
    else:
        st.write("• 本次没有额外不确定项；仍需遵守页面底部合规边界。")
    st.caption(
        f"计划基准交易日：{plan.as_of} · 计划版本：{plan.schema_version} · "
        "本区域未调用DeepSeek。"
    )


st.caption("宽基指数趋势体检 · MARKET REGIME RESEARCH")
st.title("AI 驱动的股票市场大势研判")
st.caption(
    "使用可追溯证据描述宽基指数趋势，并从风格、宽度、流动性、情绪与估值补充市场截面。"
)

with st.sidebar:
    st.header("研究设置")
    symbol = st.selectbox(
        "研究对象",
        options=list(INDEX_OPTIONS),
        index=1,
        format_func=lambda code: f"{INDEX_OPTIONS[code]}（{code}）",
    )
    st.caption("可比较大盘、小盘与成长风格；顶部状态仍是所选指数的趋势判断。")
    run_analysis = st.button("生成市场研判", type="primary", width="stretch")

    st.divider()
    st.markdown("**服务配置**")
    for variable, label in (
        ("FUYAO_API_KEY", "扶摇 API"),
        ("IFIND_API_KEY", "iFinD MCP"),
        ("DEEPSEEK_API_KEY", "DeepSeek API"),
    ):
        configured = bool(os.getenv(variable, "").strip())
        st.write(f"{'✅' if configured else '⚪'} {label}")
    st.caption("只检查密钥是否存在，不展示密钥内容。")

if run_analysis:
    st.session_state.pop("deepseek_explanation_v1", None)
    st.session_state.pop("continue_research_v1", None)
    try:
        with st.spinner("正在获取趋势行情并校验证据……"):
            calendar_data, history_data = load_fuyao_market_data(symbol)
            st.session_state["market_analysis_v1"] = {
                "symbol": symbol,
                "index_name": INDEX_OPTIONS[symbol],
                "calendar": calendar_data,
                "history": history_data,
            }
            st.session_state.pop("market_error", None)
    except FuyaoError as exc:
        st.session_state.pop("market_analysis_v1", None)
        st.session_state.pop("dimension_sources", None)
        st.session_state.pop("dimension_errors", None)
        st.session_state.pop("evidence_bundle_v1", None)
        st.session_state.pop("market_assessment_v1", None)
        st.session_state["market_error"] = str(exc)
    else:
        dimension_sources: dict[str, object] = {}
        dimension_errors: dict[str, str] = {}
        dimension_loaders = (
            ("universe", lambda: load_constituent_universe(symbol)),
            ("style", load_style_market_data),
            ("breadth", lambda: load_breadth_market_data(symbol)),
            ("valuation", lambda: load_valuation_market_data(symbol)),
            (
                "sentiment",
                lambda: load_sentiment_market_data(
                    history_data.items[-1].date_ms
                ),
            ),
        )
        with st.spinner("正在独立加载风格、宽度、情绪与估值证据……"):
            for dimension_name, loader in dimension_loaders:
                try:
                    dimension_sources[dimension_name] = loader()
                except (FuyaoError, ValueError) as exc:
                    dimension_errors[dimension_name] = str(exc)
        st.session_state["dimension_sources"] = dimension_sources
        st.session_state["dimension_errors"] = dimension_errors

if "market_error" in st.session_state:
    st.error(f"数据链路异常：{st.session_state['market_error']}")
    st.warning("由于关键数据不可用，本次不生成市场状态。请检查配置或稍后重试。")

if "market_analysis_v1" not in st.session_state:
    st.info("点击左侧“生成市场研判”，加载扶摇真实数据并生成市场状态。")
    st.markdown("### 将展示")
    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.markdown("**状态**  \n趋势偏强 / 分化 / 趋势偏弱")
    preview_col2.markdown("**证据**  \n趋势、风格、宽度、流动性、情绪、估值")
    preview_col3.markdown("**追溯**  \n数据时间、口径、来源、request_id")
else:
    analysis_data = st.session_state["market_analysis_v1"]
    selected_symbol = analysis_data["symbol"]
    selected_index_name = analysis_data["index_name"]
    calendar_result = analysis_data["calendar"]
    history_result = analysis_data["history"]
    try:
        indicators = calculate_trend_indicators(history_result.items)
    except IndicatorDataError as exc:
        st.session_state.pop("evidence_bundle_v1", None)
        st.session_state.pop("market_assessment_v1", None)
        st.error(f"指标计算异常：{exc}")
        st.warning("由于指标输入不完整，本次不生成市场状态。")
    else:
        state = evaluate_trend_state(indicators)
        chart_data = make_chart_data(history_result)
        try:
            evidence_bundle = build_current_evidence_bundle(
                symbol=selected_symbol,
                index_name=selected_index_name,
                indicators=indicators,
                history=history_result,
                calendar=calendar_result,
                sources=st.session_state.get("dimension_sources", {}),
                errors=st.session_state.get("dimension_errors", {}),
            )
            st.session_state["evidence_bundle_v1"] = evidence_bundle
            market_assessment = build_market_assessment(
                evidence_bundle,
                evaluation_date=datetime.now(SHANGHAI_TZ).date(),
                trading_dates=tuple(
                    _normalize_calendar_date(item.date)
                    for item in calendar_result.items
                ),
            )
            st.session_state["market_assessment_v1"] = market_assessment
            st.session_state.pop("evidence_bundle_error", None)
        except ValueError as exc:
            st.session_state.pop("evidence_bundle_v1", None)
            st.session_state.pop("market_assessment_v1", None)
            st.session_state["evidence_bundle_error"] = str(exc)

        if (
            "evidence_bundle_v1" in st.session_state
            and "market_assessment_v1" in st.session_state
        ):
            render_rule_assessment(
                st.session_state["market_assessment_v1"],
                st.session_state["evidence_bundle_v1"],
            )
            with st.expander("AI解释（可选，默认折叠）", expanded=False):
                render_ai_explanation(
                    st.session_state["market_assessment_v1"],
                    st.session_state["evidence_bundle_v1"],
                )
        st.subheader("指数趋势与市场维度明细")
        render_trend_dimension_summary(state, indicators, selected_index_name)
        render_metric_panel(indicators)

        st.subheader(f"{selected_index_name}指数走势")
        st.caption("纵轴按真实数据范围缩放；横轴止于最新交易日。悬停可查看中文指标名和两位小数。")
        st.altair_chart(
            make_index_chart(chart_data, selected_index_name), width="stretch"
        )

        render_market_dimensions(
            st.session_state.get("dimension_sources", {}),
            st.session_state.get("dimension_errors", {}),
            indicators,
            selected_index_name,
        )

        render_evidence_panel(state)
        render_details(
            calendar_result,
            history_result,
            indicators,
            chart_data,
            selected_index_name,
            selected_symbol,
        )

        render_continue_research(
            symbol=selected_symbol,
            index_name=selected_index_name,
            calendar=calendar_result,
            history=history_result,
            sources=st.session_state.get("dimension_sources", {}),
        )

        st.divider()
        st.caption(
            "合规说明：本页面描述已发生的市场数据，不预测确定性涨跌，"
            "不提供收益承诺、买卖建议或仓位建议。"
        )
