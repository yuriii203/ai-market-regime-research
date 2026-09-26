from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=40).run()
    app.button[0].click().run(timeout=40)

    print(f"PAGE_EXCEPTIONS={len(app.exception)}")
    print(f"PAGE_METRICS={len(app.metric)}")
    print(f"PAGE_TABS={len(app.tabs)}")
    print(f"PAGE_SELECTBOX_OPTIONS={len(app.selectbox[0].options)}")
    if app.exception:
        raise RuntimeError(str(app.exception[0].value))


if __name__ == "__main__":
    main()

