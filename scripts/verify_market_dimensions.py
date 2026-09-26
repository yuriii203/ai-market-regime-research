from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from src.research.models import ResearchQuestion


INDEXES = ("000001.SH", "000300.SH", "000852.SH", "399006.SZ")


def main() -> None:
    app_path = PROJECT_ROOT / "app.py"
    for symbol in INDEXES:
        app = AppTest.from_file(app_path, default_timeout=60).run()
        app.selectbox[0].set_value(symbol).run()
        app.button[0].click().run(timeout=60)
        sentiment_values = [
            metric.value
            for metric in app.metric
            if metric.label in {"涨停家数", "跌停家数", "炸板家数"}
        ]
        bundle = app.session_state["evidence_bundle_v1"]
        assessment = app.session_state["market_assessment_v1"]
        evidence_ids = [item.id for item in bundle.evidence]
        subheaders = [item.value for item in app.subheader]
        metric_labels = {item.label for item in app.metric}
        frames = [item.value for item in app.dataframe]
        markdown_text = "\n".join(str(item.value) for item in app.markdown)
        button_labels = {item.label for item in app.button}
        print(
            f"SYMBOL={symbol} EXCEPTIONS={len(app.exception)} "
            f"WARNINGS={len(app.warning)} METRICS={len(app.metric)} "
            f"SENTIMENT={sentiment_values} DIMENSIONS={len(bundle.dimensions)} "
            f"EVIDENCE={len(evidence_ids)} MISSING={bundle.missing_dimensions} "
            f"TIME_CONFLICTS={len(bundle.time_conflicts)} "
            f"STATE={assessment.state_label} CONFIDENCE={assessment.confidence.score}"
        )
        if app.exception:
            raise RuntimeError(str(app.exception[0].value))
        if "程序综合研判" not in subheaders or "AI解释" not in subheaders:
            raise RuntimeError(f"{symbol}: rule or AI section is missing")
        if subheaders.index("程序综合研判") > subheaders.index("AI解释"):
            raise RuntimeError(f"{symbol}: AI section appears before deterministic result")
        if markdown_text.count('<div class="state-label">') != 1:
            raise RuntimeError(f"{symbol}: more than one primary state card")
        if "当前趋势状态" in markdown_text:
            raise RuntimeError(f"{symbol}: legacy trend hero card still exists")
        if "趋势是综合研判的一个输入维度，不是第二个综合结论" not in markdown_text:
            raise RuntimeError(f"{symbol}: trend hierarchy explanation is missing")
        if 'class="dimension-grid"' not in markdown_text:
            raise RuntimeError(f"{symbol}: six-dimension summary is missing")
        expected_research_buttons = {item.value for item in ResearchQuestion}
        if expected_research_buttons - button_labels:
            raise RuntimeError(
                f"{symbol}: continue-research buttons are missing "
                f"{expected_research_buttons - button_labels}"
            )
        if {
            "综合置信度（证据质量）",
            "已返回维度",
            "数据时点",
        } - metric_labels:
            raise RuntimeError(f"{symbol}: deterministic summary metrics are missing")
        if not any("证据编号" in frame.columns for frame in frames):
            raise RuntimeError(f"{symbol}: evidence ledger is missing")
        if not any("评分项" in frame.columns for frame in frames):
            raise RuntimeError(f"{symbol}: confidence breakdown is missing")
        unexpected_warnings = [
            item.value
            for item in app.warning
            if "各维度有效日期" not in item.value
        ]
        if unexpected_warnings:
            raise RuntimeError(f"{symbol}: unexpected warnings {unexpected_warnings}")
        if len(bundle.dimensions) != 6:
            raise RuntimeError(f"{symbol}: expected 6 dimensions")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise RuntimeError(f"{symbol}: duplicate evidence IDs")
        if bundle.to_json() != bundle.to_json():
            raise RuntimeError(f"{symbol}: unstable evidence JSON")
        referenced_ids = set(assessment.supporting_evidence_ids)
        referenced_ids.update(assessment.counter_evidence_ids)
        referenced_ids.update(assessment.uncertainty_evidence_ids)
        for condition in assessment.switch_conditions:
            referenced_ids.update(condition.related_evidence_ids)
        if not referenced_ids <= set(evidence_ids):
            raise RuntimeError(f"{symbol}: assessment references unknown evidence")
        if assessment.to_json() != assessment.to_json():
            raise RuntimeError(f"{symbol}: unstable assessment JSON")


if __name__ == "__main__":
    main()
