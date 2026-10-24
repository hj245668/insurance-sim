"""이자론 기본 함수 (교재 1장)."""


def discount_factor(i: float, t: float = 1.0) -> float:
    """현가계수 v^t = (1+i)^(-t)."""
    return (1.0 + i) ** (-t)


def present_value(amount: float, i: float, n: float) -> float:
    """n년 뒤 금액의 현재가치."""
    return amount * discount_factor(i, n)


def annuity_immediate(i: float, n: int) -> float:
    """기말 연금 a_n = (1 - v^n) / i."""
    if i == 0:
        return float(n)
    return (1.0 - discount_factor(i, n)) / i


def annuity_due(i: float, n: int) -> float:
    """기초 연금 ä_n = (1 - v^n) / d,  d = i / (1+i)."""
    if i == 0:
        return float(n)
    d = i / (1.0 + i)
    return (1.0 - discount_factor(i, n)) / d
