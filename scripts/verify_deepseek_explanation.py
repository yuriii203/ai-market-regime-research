from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.deepseek_explainer import explain_market_assessment
from src.clients.deepseek import DeepSeekClient


def main() -> None:
    project_root = PROJECT_ROOT
    load_dotenv(project_root / ".env")
    app = AppTest.from_file(project_root / "app.py", default_timeout=60).run()
    app.button[0].click().run(timeout=60)
    if app.exception:
        raise RuntimeError(str(app.exception[0].value))
    if not any("DeepSeek" in item.label for item in app.button):
        raise RuntimeError("DeepSeek explanation button is missing")
    assessment = app.session_state["market_assessment_v1"]
    bundle = app.session_state["evidence_bundle_v1"]
    result = explain_market_assessment(
        assessment, bundle, DeepSeekClient.from_env()
    )
    print(
        f"SOURCE={result.source} MODEL={result.model} "
        f"REQUEST_ID_PRESENT={bool(result.request_id)} "
        f"CITATIONS={len(result.cited_evidence_ids)} "
        f"ATTEMPTS={len(result.attempts)} "
        f"FALLBACK={result.fallback_reason or ''}"
    )
    if result.source != "deepseek":
        raise RuntimeError("DeepSeek validation failed")


if __name__ == "__main__":
    main()
