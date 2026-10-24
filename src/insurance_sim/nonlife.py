"""손해보험: 빈도·심도, 복합포아송, 파산확률 (교재 5장)."""
import math
from statistics import NormalDist


def pure_premium(freq: float, severity_mean: float) -> float:
    return freq * severity_mean


def gross_premium(pure: float, expense_ratio: float) -> float:
    if not 0 <= expense_ratio < 1:
        raise ValueError("expense_ratio must be in [0, 1)")
    return pure / (1.0 - expense_ratio)


def compound_poisson_moments(lam, sev_mean, sev_sd, n_policies=1):
    """복합포아송 총손해의 (평균, 분산). Var = lam * E[X^2]."""
    ex2 = sev_sd**2 + sev_mean**2
    return lam * n_policies * sev_mean, lam * n_policies * ex2


def normal_premium(lam, sev_mean, sev_sd, n_policies, confidence=0.99):
    """정규근사로 confidence 수준을 감당하는 계약당 보험료."""
    mean, var = compound_poisson_moments(lam, sev_mean, sev_sd, n_policies)
    z = NormalDist().inv_cdf(confidence)
    return (mean + z * math.sqrt(var)) / n_policies


def lundberg_exponential(u: float, theta: float, mu: float) -> float:
    """지수 심도의 파산확률 psi(u) = exp(-theta u / ((1+theta) mu)) / (1+theta)."""
    return math.exp(-theta * u / ((1.0 + theta) * mu)) / (1.0 + theta)
