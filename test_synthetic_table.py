"""가상 Gompertz-Makeham 생명표와 CSV 로더 검증."""
import math

import pytest

from insurance_sim.life import (
    gompertz_makeham_table, load_life_table_csv, net_level_premium,
)

T = gompertz_makeham_table()


def test_range_and_radix():
    assert T.start_age == 20 and T.max_age == 100
    assert T.l(20) == 100000


def test_one_year_mortality_matches_formula():
    a, b, c = 0.0004, 0.00002, 1.1
    expected = 1 - math.exp(-(a + b * c**40 * (c - 1) / math.log(c)))
    assert T.q(40) == pytest.approx(expected)


def test_mortality_increases_with_age():
    assert T.q(25) < T.q(40) < T.q(60) < T.q(80)


def test_premium_increases_with_issue_age():
    p30 = net_level_premium(T, 30, 20, 0.03, 1e8)
    p50 = net_level_premium(T, 50, 20, 0.03, 1e8)
    assert 0 < p30 < p50


def test_endowment_costs_more_than_term():
    assert net_level_premium(T, 30, 20, 0.03, 1e8, "endowment") > net_level_premium(
        T, 30, 20, 0.03, 1e8, "term"
    )


def test_csv_roundtrip(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("age,lx\n30,1000\n31,990\n32,975\n", encoding="utf-8")
    t = load_life_table_csv(f)
    assert t.start_age == 30 and t.q(30) == pytest.approx(0.01)


def test_csv_requires_consecutive_ages(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("age,lx\n30,1000\n32,990\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_life_table_csv(f)
