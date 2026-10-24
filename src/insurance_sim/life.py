"""생존모델과 생명보험 (교재 2~4장)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .interest import discount_factor


@dataclass(frozen=True)
class LifeTable:
    """start_age 세부터 1세 간격의 생존자 수 l_x."""

    start_age: int
    lx: tuple

    def __post_init__(self):
        if len(self.lx) < 2:
            raise ValueError("lx needs at least two ages")
        if any(b > a for a, b in zip(self.lx, self.lx[1:])):
            raise ValueError("lx must be non-increasing")

    @property
    def max_age(self) -> int:
        return self.start_age + len(self.lx) - 1

    def l(self, age: int) -> float:
        if not self.start_age <= age <= self.max_age:
            raise ValueError(f"age {age} outside table range")
        return self.lx[age - self.start_age]

    def p(self, age: int, t: int = 1) -> float:
        """t년 생존확률 tp_x."""
        return self.l(age + t) / self.l(age)

    def q(self, age: int, t: int = 1) -> float:
        """t년 내 사망확률 tq_x."""
        return 1.0 - self.p(age, t)

    def deferred_q(self, age: int, k: int, t: int = 1) -> float:
        """k년 뒤부터 t년 사이 사망확률 k|t q_x."""
        return (self.l(age + k) - self.l(age + k + t)) / self.l(age)


# 교재의 연습용 가상 생명표 (실제 통계 아님)
DEMO_TABLE = LifeTable(60, (90000, 89000, 87800, 86400))


def constant_hazard_survival(mu: float, t: float) -> float:
    """위험률이 일정할 때 S(t) = exp(-mu t)."""
    return math.exp(-mu * t)


def gompertz_hazard(x: float, b: float = 0.00002, c: float = 1.1) -> float:
    """가상의 Gompertz 위험률 mu_x = B c^x."""
    return b * c**x


def term_insurance_apv(table, age, n, i, benefit=1.0):
    """n년 정기보험 일시납순보험료 (사망 연말 지급)."""
    return benefit * sum(
        discount_factor(i, k + 1) * table.deferred_q(age, k) for k in range(n)
    )


def pure_endowment_apv(table, age, n, i, benefit=1.0):
    return benefit * discount_factor(i, n) * table.p(age, n)


def endowment_apv(table, age, n, i, benefit=1.0):
    return term_insurance_apv(table, age, n, i, benefit) + pure_endowment_apv(
        table, age, n, i, benefit
    )


def annuity_due_apv(table, age, n, i):
    """n년 유기 생명연금(기초) ä_{x:n}."""
    return sum(discount_factor(i, k) * table.p(age, k) for k in range(n))


def _apv(kind, table, age, n, i, benefit):
    if kind == "term":
        return term_insurance_apv(table, age, n, i, benefit)
    if kind == "endowment":
        return endowment_apv(table, age, n, i, benefit)
    raise ValueError("kind must be 'term' or 'endowment'")


def net_level_premium(table, age, n, i, benefit=1.0, kind="term"):
    """등가원리: P * ä = 보험금의 기대현가."""
    return _apv(kind, table, age, n, i, benefit) / annuity_due_apv(table, age, n, i)


def prospective_reserve(table, age, n, i, t, benefit=1.0, kind="term"):
    """t년 경과 시점(보험료 납입 직전)의 장래법 책임준비금."""
    if not 0 <= t < n:
        raise ValueError("t must satisfy 0 <= t < n")
    p = net_level_premium(table, age, n, i, benefit, kind)
    return _apv(kind, table, age + t, n - t, i, benefit) - p * annuity_due_apv(
        table, age + t, n - t, i
    )
