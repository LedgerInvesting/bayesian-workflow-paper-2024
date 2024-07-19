from typing import List, Dict, Union, Tuple
from scipy.special import logsumexp

import cmdstanpy as csp
import json
import numpy as np
import bayesblend as bb

from plot import plot_scores, plot_percentiles, plot_predictions

SEED = 1234
SCALER = 1e4
DATA = {
    "PP": "data/pp.json",
    "WC": "data/wc.json",
    "CA": "data/ca.json",
    "OO": "data/oo.json",
}
MODEL_NAMES = {
    "rw": "Random Walk",
    "rev": "Mean Reversion",
}
PARAMS = {
    "PP": dict(tau=4, rho=[5, 10]),
    "WC": dict(tau=6, rho=[4, 10]),
    "CA": dict(tau=4, rho=[5, 10]),
    "OO": dict(tau=6, rho=[4, 10]),
}
development = csp.CmdStanModel(stan_file="stan/development.stan")
forecast_rw = csp.CmdStanModel(stan_file="stan/forecast-random-walk.stan")
forecast_rev = csp.CmdStanModel(stan_file="stan/forecast-reversion.stan")

SAMPLES = 2500

STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": SAMPLES,
    "parallel_chains": 4,
    "chains": 4,
    "inits": 0,
    "seed": SEED,
    "show_progress": True,
    "adapt_delta": 0.95,
}

RESULTS = "results"


def load_data(lob: str) -> List[Dict[str, List[Union[float, int]]]]:
    return json.load(open(DATA[lob], "r"))


def prep_development_data(data) -> Dict[str, int | float | np.ndarray]:
    common_data = []
    for d in data.values():
        d = np.array(d)[..., 0]
        N, M = d.shape
        index = np.array(
            [[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(d)]
        )
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
            np.array([[i + 1] * len(yy) for i, yy in enumerate(d)]),
            np.array([list(range(1, len(yy) + 1)) for yy in d]),
        )
        common_data.append(
            {
                "T": len(train_i),
                "T_prime": len(test_i),
                "M": M,
                "ii": np.concatenate(
                    [ii.flatten()[train_i], ii.flatten()[test_i]]
                ).tolist(),
                "jj": np.concatenate(
                    [jj.flatten()[train_i], jj.flatten()[test_i]]
                ).tolist(),
                "B": np.concatenate(
                    [index.flatten()[train_i], index.flatten()[test_i]]
                ).tolist(),
                "prior_only": 0,
            }
        )
    if not all(common_data[0] == common_data[i] for i in range(len(common_data))):
        raise ValueError("Indices do not match.")
    else:
        common_data = common_data[0]
        y = [np.array(d)[..., 0].flatten() / SCALER for d in data.values()]
        premium = [np.array(d)[..., 2].flatten() / SCALER for d in data.values()]
        max_pred = [max(d.flatten()) * 100 for d in y]
    return {"y": y, "N": len(y), "premium": premium, "MAX_PRED": max_pred} | common_data


def prep_forecast_data(fit_dev, stan_data_dev):
    y_tilde = fit_dev.stan_variable("y_tilde")
    M, N, _ = y_tilde.shape
    y = np.array(stan_data_dev["y"]).reshape(N, 10, 10)[:, :, -1]
    premium = np.array(stan_data_dev["premium"]).reshape(N, 10, 10)[:, :, -1]
    ulr = y_tilde.reshape(M, N, 10, 10)[:, :, :9, -1] / premium[:, :9]
    ulr_mean = np.mean(ulr, axis=0)
    ulr_std = np.std(ulr, axis=0)
    T = y.shape[1]
    return {
        "N": N,
        "T": T,
        "T_train": 9,
        "y": y,
        "ulr_mean": ulr_mean,
        "ulr_std": ulr_std,
        "premium": premium,
        "prior_only": 0,
    }


def fit_development(data, prior_only=0):
    fit = development.sample(
        data=data | {"prior_only": prior_only},
        **STAN_CONFIG,
    )
    return fit


def fit_forecast_rw(data, prior_only=0):
    fit = forecast_rw.sample(
        data=data | {"prior_only": prior_only},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    return fit


def fit_forecast_rev(data, prior_only=0):
    fit = forecast_rev.sample(
        data=data | {"prior_only": prior_only},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    return fit


def fit_stacking(fits) -> bb.Draws:
    stack = bb.MleStacking.from_cmdstanpy(
        fits,
        log_lik_name="log_lik_train",
        post_pred_name="y_pred_train",
    )
    test_draws = {
        name: bb.Draws.from_cmdstanpy(
            fit,
            log_lik_name="log_lik_test",
            post_pred_name="y_pred_test",
        )
        for name, fit in fits.items()
    }
    stack.fit()
    stack_pred = stack.predict(test_draws)
    return stack_pred


def elpd(fit):
    return logsumexp(fit.log_lik_test[:, :, 0], axis=0) - np.log(
        fit.log_lik_test.shape[0]
    )


def squared_error(data, fit):
    return ((fit.y_pred_test[:, :, -1].mean(axis=0) - data["y"][:, -1]) ** 2).flatten()


def percentile(data, fit):
    return np.mean(fit.y_pred_test[:, :, -1] <= data["y"][:, -1], axis=0).flatten()


def percentile_right_edge(data, fit):
    M, N, _ = fit.y_tilde.shape
    y_tilde = fit.y_tilde.reshape((M, N, 10, 10))
    y = np.array(data["y"]).reshape((N, 10, 10))
    return np.mean(y_tilde[:, :, 1:8, -1] <= y[:, 1:8, -1], axis=0).flatten()


def score_model(data, fit, lob, name) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    elpds = elpd(fit)
    ses = squared_error(data, fit)
    percentiles = percentile(
        data,
        fit,
    )
    with open(RESULTS + f"/{name}-elpds-{lob}.json", "w") as f:
        json.dump(elpds.tolist(), f)
    with open(RESULTS + f"/{name}-ses-{lob}.json", "w") as f:
        json.dump(ses.tolist(), f)
    with open(RESULTS + f"/{name}-percentiles-{lob}.json", "w") as f:
        json.dump(percentiles.tolist(), f)
    return elpds, ses, percentiles


def score_stacking(data, pred, lob):
    y_test = data["y"][:, -1]
    percentiles = np.mean(pred.post_pred[:, :, -1] <= y_test, axis=0)
    ses = (pred.post_pred[:, :, -1].mean(axis=0) - data["y"][:, -1]) ** 2
    with open(RESULTS + f"/stack-elpds-{lob}.json", "w") as f:
        json.dump(pred.lpd.tolist(), f)
    with open(RESULTS + f"/stack-ses-{lob}.json", "w") as f:
        json.dump(ses.tolist(), f)
    with open(RESULTS + f"/stack-percentiles-{lob}.json", "w") as f:
        json.dump(percentiles.tolist(), f)
    return pred.lpd, ses, percentiles


def prior_predict_pipeline(stan_data_for, models, lob):
    fits = {name: model(stan_data_for, prior_only=1) for name, model in models.items()}
    plot_predictions(stan_data_for, fits, lob, prior_only=1)


def main() -> None:
    scores = {}
    dev_fits = {}
    for_fits = {}
    for lob in list(DATA):
        data = load_data(lob)
        stan_data_dev = prep_development_data(data) | PARAMS[lob]
        fit_dev = fit_development(stan_data_dev)
        stan_data_for = prep_forecast_data(fit_dev, stan_data_dev)
        models = {"rw": fit_forecast_rw, "rev": fit_forecast_rev}
        prior_predict_pipeline(stan_data_for, models, lob)
        fits = {name: model(stan_data_for) for name, model in models.items()}
        plot_predictions(stan_data_for, fits, lob, prior_only=0)
        score = {
            name: score_model(stan_data_for, fit, lob, name)
            for name, fit in fits.items()
        }
        fit_stack = fit_stacking(fits)
        score["stack"] = score_stacking(stan_data_for, fit_stack, lob)
        scores[lob] = score
        dev_fits[lob] = fit_dev
        for_fits[lob] = fits
    return scores, dev_fits, for_fits


if __name__ == "__main__":
    scores, dev_fits, for_fits = main()
    plot_scores(scores)
    plot_percentiles(scores)
