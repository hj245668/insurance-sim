"""교재 5장 예제 검증과 몬테카를로 대조 테스트."""
import math

import pytest

from insurance_sim.nonlife import (
    compound_poisson_moments, gross_premium, lundberg_exponential,
    normal_premium, pure_premium,
)
from insurance_sim.simulation import (
    empirical_quantile, simulate_aggregate_loss, simulate_ruin_probability,
)


def test_ex8_premiums():
    pure = pure_premium(0.1, 2_000_000)
    assert pure == pytest.approx(200_000)
    assert gross_premium(pure, 0.2) == pytest.approx(250_000)


def test_expense_ratio_validation():
    with pytest.raises(ValueError):
        gross_premium(1.0, 1.0)


def test_ex9_moments_and_normal_premium():
    mean, var = compound_poisson_moments(0.1, 200, 100, 1000)  # 만원 단위
    assert mean == pytest.approx(20_000)
    assert math.sqrt(var) == pytest.approx(2236.07, abs=0.01)
    assert normal_premium(0.1, 200, 100, 1000, 0.99) == pytest.approx(25.2, abs=0.01)


def test_ex10_lundberg():
    assert lundberg_exponential(2000, 0.2, 200) == pytest.approx(0.1574, abs=1e-4)


def test_ruin_decreases_with_capital_and_loading():
    assert lundberg_exponential(4000, 0.2, 200) < lundberg_exponential(2000, 0.2, 200)
    assert lundberg_exponential(2000, 0.4, 200) < lundberg_exponential(2000, 0.2, 200)


def test_mc_aggregate_loss_matches_theory():
    losses = simulate_aggregate_loss(1000, 0.1, 200, 100, n_trials=20000, seed=1)
    mean, var = compound_poisson_moments(0.1, 200, 100, 1000)
    assert losses.mean() == pytest.approx(mean, rel=0.01)
    assert losses.std(ddof=1) == pytest.approx(math.sqrt(var), rel=0.05)


def test_mc_quantile_close_to_normal_approx_but_skewed_right():
    losses = simulate_aggregate_loss(1000, 0.1, 200, 100, n_trials=20000, seed=2)
    q99 = empirical_quantile(losses, 0.99)
    normal_q99 = normal_premium(0.1, 200, 100, 1000, 0.99) * 1000
    assert q99 == pytest.approx(normal_q99, rel=0.03)
    assert q99 > normal_q99 * 0.99  # 오른쪽 꼬리가 길어 정규근사보다 작지 않다


def test_mc_ruin_matches_lundberg():
    est = simulate_ruin_probability(2000, 0.2, 200, n_paths=10000, n_claims=400, seed=3)
    assert est == pytest.approx(lundberg_exponential(2000, 0.2, 200), abs=0.015)
