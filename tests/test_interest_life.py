"""교재 1~4장 예제를 코드로 검증한다."""
import pytest

from insurance_sim.interest import (
    annuity_due, annuity_immediate, present_value,
)
from insurance_sim.life import (
    DEMO_TABLE, LifeTable, annuity_due_apv, constant_hazard_survival,
    endowment_apv, gompertz_hazard, net_level_premium, prospective_reserve,
    term_insurance_apv,
)

T = DEMO_TABLE


# ---- 1장 이자론 ----
def test_ex1_present_value():
    assert present_value(1000, 0.05, 10) == pytest.approx(613.9, abs=0.05)


def test_ex2_annuities():
    assert annuity_immediate(0.05, 5) == pytest.approx(4.3295, abs=1e-4)
    assert annuity_due(0.05, 5) == pytest.approx(4.5460, abs=1e-4)


def test_annuity_due_equals_growth_of_immediate():
    assert annuity_due(0.04, 8) == pytest.approx(1.04 * annuity_immediate(0.04, 8))


def test_zero_interest_annuity():
    assert annuity_due(0.0, 7) == 7 and annuity_immediate(0.0, 7) == 7


# ---- 2장 생존모델 ----
def test_ex3_survival_probabilities():
    assert T.q(60) == pytest.approx(0.01111, abs=1e-5)
    assert T.p(60, 2) == pytest.approx(0.97556, abs=1e-5)
    assert T.deferred_q(60, 2) == pytest.approx(0.01556, abs=1e-5)


def test_death_probabilities_sum_to_total():
    total = sum(T.deferred_q(60, k) for k in range(3))
    assert total == pytest.approx(T.q(60, 3))


def test_ex4_constant_hazard():
    assert constant_hazard_survival(0.02, 1) == pytest.approx(0.9802, abs=1e-4)
    assert constant_hazard_survival(0.02, 10) == pytest.approx(0.8187, abs=1e-4)


def test_gompertz_hazard_increases_with_age():
    assert gompertz_hazard(90) > gompertz_hazard(60) > gompertz_hazard(30)


def test_lifetable_validation():
    with pytest.raises(ValueError):
        LifeTable(60, (100, 120))
    with pytest.raises(ValueError):
        T.l(70)


# ---- 3장 보험료 ----
def test_ex5_one_year_term():
    assert term_insurance_apv(T, 60, 1, 0.03, 1e8) == pytest.approx(1_078_750, abs=5)


def test_ex6_two_year_term_and_premium():
    assert term_insurance_apv(T, 60, 2, 0.03, 1e8) == pytest.approx(2_335_540, abs=10)
    assert annuity_due_apv(T, 60, 2, 0.03) == pytest.approx(1.960087, abs=1e-5)
    assert net_level_premium(T, 60, 2, 0.03, 1e8) == pytest.approx(1_191_550, abs=10)


def test_equivalence_principle():
    p = net_level_premium(T, 60, 3, 0.03, 1e8)
    assert p * annuity_due_apv(T, 60, 3, 0.03) == pytest.approx(
        term_insurance_apv(T, 60, 3, 0.03, 1e8)
    )


def test_endowment_identity_A_equals_1_minus_d_adue():
    i = 0.03
    d = i / (1 + i)
    for n in (1, 2, 3):
        assert endowment_apv(T, 60, n, i) == pytest.approx(
            1 - d * annuity_due_apv(T, 60, n, i)
        )


def test_premium_increases_with_mortality():
    safer = LifeTable(60, (90000, 89500, 89000, 88500))
    assert net_level_premium(T, 60, 2, 0.03) > net_level_premium(safer, 60, 2, 0.03)


# ---- 4장 준비금 ----
def test_ex7_reserve():
    assert prospective_reserve(T, 60, 2, 0.03, t=1, benefit=1e8) == pytest.approx(
        117_490, abs=10
    )


def test_reserve_zero_at_issue():
    assert prospective_reserve(T, 60, 3, 0.03, t=0, benefit=1e8) == pytest.approx(0, abs=1e-6)


def test_reserve_bounds():
    with pytest.raises(ValueError):
        prospective_reserve(T, 60, 2, 0.03, t=2)
