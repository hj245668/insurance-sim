"""Streamlit 앱이 오류 없이 뜨는지, 교재 예제와 맞는지 확인하는 스모크 테스트."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DEMO = "교재 예제 표 (60~63세)"


def find(items, label):
    return next(x for x in items if x.label == label)


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_young_age_premium_is_positive():
    at = AppTest.from_file(APP, default_timeout=60).run()
    find(at.slider, "가입 나이").set_value(20)
    at.run()
    assert not at.exception
    assert any(m.label == "연납 순보험료" and m.value != "0원" for m in at.metric)


def test_premium_matches_book_example_with_demo_table():
    at = AppTest.from_file(APP, default_timeout=60).run()
    find(at.radio, "생명표").set_value(DEMO).run()
    find(at.slider, "보험 기간(년)").set_value(2)
    find(at.slider, "이자율(%)").set_value(3.0)
    at.run()
    assert not at.exception
    assert any("1,19" in m.value for m in at.metric)
