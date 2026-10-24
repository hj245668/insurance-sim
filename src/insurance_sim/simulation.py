"""몬테카를로 시뮬레이션."""
import numpy as np


def simulate_aggregate_loss(n_policies, lam, sev_mean, sev_sd, n_trials=20000, seed=0):
    """감마 심도의 복합포아송 총손해를 n_trials번 뽑는다.

    사고 N건의 감마 합은 감마(N*shape, scale)이므로 벡터로 한 번에 뽑는다.
    """
    rng = np.random.default_rng(seed)
    shape = (sev_mean / sev_sd) ** 2
    scale = sev_sd**2 / sev_mean
    n = rng.poisson(lam * n_policies, n_trials)
    losses = np.zeros(n_trials)
    m = n > 0
    losses[m] = rng.gamma(shape * n[m], scale)
    return losses


def empirical_quantile(losses, q=0.99):
    return float(np.quantile(losses, q))


def simulate_ruin_probability(u, theta, mu, n_paths=10000, n_claims=400, seed=0):
    """고전 위험모형(lambda=1, 지수 심도)에서 파산확률 추정."""
    rng = np.random.default_rng(seed)
    c = (1.0 + theta) * mu
    waits = rng.exponential(1.0, (n_paths, n_claims))
    claims = rng.exponential(mu, (n_paths, n_claims))
    surplus = u + np.cumsum(c * waits - claims, axis=1)
    return float(np.mean(surplus.min(axis=1) < 0))
