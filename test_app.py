"""Streamlit 앱이 오류 없이 뜨는지 확인하는 스모크 테스트."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_premium_metric_matches_book_example():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.slider[0].set_value(2)  # 보험 기간 2년
    at.slider[1].set_value(3.0)
    at.run()
    assert not at.exception
    assert any("1,19" in m.value for m in at.metric)
