from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_without_calling_external_services() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=10)

    assert not app.exception
    assert any("AI 驱动的股票市场大势研判" in item.value for item in app.title)
    assert any("MARKET REGIME RESEARCH" in item.value for item in app.caption)
    assert any(button.label == "生成市场研判" for button in app.button)
    assert len(app.selectbox[0].options) == 4
    assert all(
        code in " ".join(app.selectbox[0].options)
        for code in ("000001.SH", "000300.SH", "000852.SH", "399006.SZ")
    )
