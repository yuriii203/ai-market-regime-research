from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from src.research.models import ResearchQuestion


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def main() -> None:
    app_path = PROJECT_ROOT / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    _button(app, "生成市场研判").click().run(timeout=120)
    if app.exception:
        raise RuntimeError(f"阶段六初始化失败：{app.exception[0].value}")

    original_assessment = app.session_state["market_assessment_v1"].to_json()
    original_bundle = app.session_state["evidence_bundle_v1"].to_json()
    for question in ResearchQuestion:
        _button(app, question.value).click().run(timeout=180)
        if app.exception:
            raise RuntimeError(
                f"{question.value} 页面异常：{app.exception[0].value}"
            )
        cached = app.session_state["continue_research_v1"]
        if cached.get("error"):
            raise RuntimeError(f"{question.value} 调用失败：{cached['error']}")
        if cached["question"] != question:
            raise RuntimeError(f"{question.value} 缓存问题不一致")
        results = cached["results"]
        evidence_count = sum(len(item.evidence) for item in results)
        trace_count = sum(
            evidence.local_trace_id is not None
            for result in results
            for evidence in result.evidence
        )
        if not results or not all(result.conclusion for result in results):
            raise RuntimeError(f"{question.value} 缺少程序归纳")
        if question != ResearchQuestion.HISTORICAL_ANALOG and trace_count == 0:
            raise RuntimeError(f"{question.value} 缺少iFinD本地trace_id")
        if app.session_state["market_assessment_v1"].to_json() != original_assessment:
            raise RuntimeError(f"{question.value} 改写了阶段六综合研判")
        if app.session_state["evidence_bundle_v1"].to_json() != original_bundle:
            raise RuntimeError(f"{question.value} 改写了阶段六证据包")
        print(
            f"QUESTION={question.value} "
            f"STATUS={','.join(item.status for item in results)} "
            f"EVIDENCE={evidence_count} TRACES={trace_count}"
        )


if __name__ == "__main__":
    main()
