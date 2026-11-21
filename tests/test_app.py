"""Smoke test: the Streamlit app renders in demo mode without exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_in_demo_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no data/ here, so the app must fall back to synthetic records
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "People analytics explorer"
    assert any("Demo mode" in w.value for w in at.warning)
    at.selectbox[0].set_value("Gender").run()  # survival by gender
    assert not at.exception
