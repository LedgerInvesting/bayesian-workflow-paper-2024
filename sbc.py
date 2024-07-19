from typing import Dict, Union
import cmdstanpy as csp
import scipy.stats as stat
import numpy as np
import json
from tqdm import tqdm
from simulate import ilogit, development

from plot import plot_ranks

SEED = 1234

RNG = np.random.default_rng(SEED)

RESULTS = "results"

DEV = csp.CmdStanModel(stan_file="stan/development-sbc.stan")

TAU = 5
RHO = [6, 10]

P = 1000
SAMPLES = 1000
STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": SAMPLES,
    "parallel_chains": 4,
    "inits": 0,
    "seed": SEED,
    "show_progress": False,
}


def stan_data(y: np.ndarray) -> Dict[str, Union[int, np.ndarray]]:
    N, M = y.shape
    index = np.array([[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(y)])
    train_i, test_i = (
        np.concatenate([i[:n] for i, n in zip(index, range(N, 0, -1))]),
        np.concatenate(
            [
                i[-n:] if n else np.array([], dtype=int)
                for i, n in zip(index, range(0, N))
            ]
        ),
    )
    ii, jj = (
        np.array([[i + 1] * len(yy) for i, yy in enumerate(y)]),
        np.array([list(range(1, len(yy) + 1)) for yy in y]),
    )
    return {
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "tau": TAU,
        "rho": RHO,
        "ii": np.concatenate([ii.flatten()[train_i], ii.flatten()[test_i]]),
        "jj": np.concatenate([jj.flatten()[train_i], jj.flatten()[test_i]]),
        "B": np.concatenate([index.flatten()[train_i], index.flatten()[test_i]]),
        "y": y.flatten(),
        "prior_only": 0,
    }


def generate_data(P: int = P):
    N = 10
    M = 10
    alpha_star_rng = stat.norm(
        [0] * (TAU - 1),
        [1] * (TAU - 1),  # [1 / (j + 1) for j in range(TAU - 1)],
    )
    omega_star_rng = stat.halfnorm(0, 1)
    beta_star_rng = stat.norm(0, 1)
    gamma_rng = stat.norm((-3, -1), (0.25, 0.1))
    lambda_rng = stat.norm((-3, -1), (0.25, 0.1))
    data = []
    while len(data) < P:
        pars = {
            "alpha": np.exp(alpha_star_rng.rvs(random_state=RNG)),
            "omega": np.exp(omega_star_rng.rvs(random_state=RNG)),
            "beta": ilogit(beta_star_rng.rvs(random_state=RNG)),
            "gamma": gamma_rng.rvs(random_state=RNG),
            "lambd": lambda_rng.rvs(random_state=RNG),
        }
        y, ll = development(
            N=N,
            M=M,
            **pars,
            tau=TAU,
            init=(-1, 0.2),
            seed=None,
        )
        if np.isnan(y).any():
            continue
        else:
            data.append(((y, ll), pars))
    return data


def fit_development(data):
    fits = []
    for (y, ll), pars in tqdm(data):
        fits.append(
            (
                DEV.sample(
                    data=stan_data(y),
                    **STAN_CONFIG,
                ),
                y,
                ll,
                pars,
            )
        )
    return fits


def rank(fits):
    ranks = {}
    N, M = 10, 10
    index = np.array([i for i in range(N * M)]).reshape((N, M))
    train_index = np.array(
        [idx for i, n in zip(index, range(10, 0, -1)) for idx in i[:n]]
    )
    thin = 10
    for fit, y, ll, pars in fits:
        draws_raw = fit.stan_variables()
        draws = {k: v[::thin] for k, v in draws_raw.items()}
        keys = (k for k in draws if k in pars or k == "lambda")
        for par in keys:
            py_par = "lambd" if par == "lambda" else par
            if par in ranks:
                ranks[par].append(sum(pars[py_par] > draws[par]))
            else:
                ranks[par] = [sum(pars[py_par] > draws[par])]
        if "log likelihood" in ranks:
            ranks["log likelihood"].append(
                sum(
                    ll.flatten()[train_index].sum()
                    > draws["log_lik"][:, train_index].sum(axis=1)
                )
            )
        else:
            ranks["log likelihood"] = [
                sum(ll.flatten().sum() > draws["log_lik"][:, train_index].sum(axis=1))
            ]
        tilde_y = "tilde{y}[{1,10}]"
        if tilde_y in ranks:
            ranks[tilde_y].append(sum(y[0][-1] > draws["y_tilde"][:, 9]))
        else:
            ranks[tilde_y] = [sum(y[0][-1] > draws["y_tilde"][:, 9])]
    ranks["L"] = max([np.max(rank) for rank in ranks.values()])
    json.dump(
        {k: np.asarray(v).tolist() for k, v in ranks.items()},
        open(RESULTS + "/ranks.json", "w"),
    )
    return {k: np.asarray(v).tolist() for k, v in ranks.items()}


def main():
    data = generate_data(P)
    fits = fit_development(data)
    ranks = rank(fits)
    plot_ranks(ranks)


if __name__ == "__main__":
    main()
