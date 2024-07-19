from typing import Tuple, Sequence, Union
import numpy as np
import scipy.stats as stat


def ilogit(x: Union[np.ndarray, float]) -> np.ndarray:
    return np.exp(x) / (1 + np.exp(x))


def development(
    N: int,
    M: int,
    alpha: Sequence[float],
    omega: float,
    beta: float,
    gamma: Tuple[float, float],
    lambd: Tuple[float, float],
    tau: int = 6,
    init: Tuple[float, float] = (-1, 0.2),
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    years = list(range(N))
    lags = list(range(M))
    years_lags = [(year, lag) for year in years for lag in lags]

    y = np.empty(shape=(N, M))
    mu = np.empty(shape=(N, M))
    sigma2 = np.empty(shape=(N, M))
    ll = np.empty(shape=(N, M))

    for i, j in years_lags:
        lag = j + 1
        if not j:
            y[i, j] = rng.lognormal(*init)
            ll[i, j] = 0.0
            mu[i, j] = 0
            sigma2[i, j] = 0
        else:
            lagged_y = y[i, j - 1]
            if lag <= tau:
                mu[i, j] = lagged_y * alpha[j - 1]
                sigma2[i, j] = np.exp(gamma[0] + gamma[1] * lag + np.log(lagged_y))
            else:
                mu[i, j] = lagged_y * omega**beta**lag
                sigma2[i, j] = np.exp(lambd[0] + lambd[1] * lag + np.log(lagged_y))
            try:
                distr = stat.lognorm(scale=mu[i, j], s=np.sqrt(sigma2[i, j]))
                y[i, j] = distr.rvs(random_state=rng)
                ll[i, j] = distr.logpdf(y[i, j])
            except ValueError:
                y[i, j] = np.nan
                ll[i, j] = np.nan

    return y, ll
