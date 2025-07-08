import dill

from typing import List, Dict, Union, Tuple
from scipy.special import logsumexp

import cmdstanpy as csp
import json
import numpy as np
import bayesblend as bb

from plot import plot_scores, plot_percentiles, plot_predictions, plot_cashflows

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
    "rw (diffuse)": "Random Walk (Diffuse)",
    "rev (diffuse)": "Mean Reversion (Diffuse)",
    "rw (inform)": "Random Walk (Informed)",
    "rev (inform)": "Mean Reversion (Informed)",
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

SAMPLES = 1250

STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": 500,
    "parallel_chains": 8,
    "chains": 8,
    "inits": 0,
    "seed": SEED,
    "show_progress": True,
    "adapt_delta": 0.95,
    "max_treedepth": 11,
}

RESULTS = "results"


def load_data(lob: str) -> List[Dict[str, List[Union[float, int]]]]:
    return json.load(open(DATA[lob], "r"))


# NOTE: this function assumes all program-level triangles have the same dimension
def prep_development_data(
    data, pad: bool = False, max_dev: int | None = None
) -> Dict[str, int | float | np.ndarray]:
    def _array_from_ragged_list(data, pad=pad):
        if pad:
            max_dev_across_ay = (
                max(len(ay) for ay in data) if max_dev is None else max_dev
            )
            d = [ay + [[-999, -999]] * (max_dev_across_ay - len(ay)) for ay in data]
            return np.array(d)
        return np.array(data)

    common_data = []
    for d in data.values():
        d = _array_from_ragged_list(d)[..., 0]
        N, M = d.shape
        index = np.array(
            [[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(d)]
        )
        train_i = np.concatenate([i[:n] for i, n in zip(index, range(N, 0, -1))])
        test_i = np.delete(index.flatten(), train_i)
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
        # paid loss
        y = [
            _array_from_ragged_list(d)[..., 0].flatten().astype(float)
            for d in data.values()
        ]
        premium = [
            _array_from_ragged_list(d)[..., 1].flatten().astype(float)
            for d in data.values()
        ]
        # scale data, but leave -999 padding unchanged if they exist
        for i, y_program in enumerate(y):
            not_padding_mask = (
                y_program != -999 if pad else np.ones_like(y_program, dtype=bool)
            )
            y[i][not_padding_mask] /= SCALER
            premium[i][not_padding_mask] /= SCALER
        max_pred = [np.round(max(d.flatten()) * 100, 3) for d in y]
    return {
        "y": y,
        "N": len(y),
        "premium": premium,
        "MAX_PRED": max_pred,
        "MAX_OBS_DEV": N,
    } | common_data


def prep_forecast_data(
    fit_dev, stan_data_dev, n_ay=10, n_dev=10, n_leave_out=1, n_year_ahead=0
):
    y_tilde = fit_dev.stan_variable("y_tilde")
    M, N, _ = y_tilde.shape
    y = np.array(stan_data_dev["y"]).reshape(N, n_ay, n_dev)[:, :, -1]
    premium = np.array(stan_data_dev["premium"]).reshape(N, n_ay, n_dev)[:, :, 0]
    # loss ratio conversion
    n_train = premium.shape[1] - n_leave_out
    ulr = y_tilde.reshape(M, N, n_ay, n_dev)[:, :, :n_train, -1] / premium[:, :n_train]
    ulr_mean = np.nanmean(ulr, axis=0)
    ulr_std = np.nanstd(ulr, axis=0)
    T = y.shape[1]
    # if all y (test data) are padding, use the predicted values instead. These are only used
    # for setting the prior on the "true" ULR mean/SD.
    if (y == -999).all():
        y = y_tilde.reshape(M, N, n_ay, n_dev)[:, :, :n_train, -1].mean(axis=0)
    return {
        "N": N,
        "T": T,
        "T_train": n_train,
        "y": y,
        "ulr_mean": ulr_mean,
        "ulr_std": ulr_std,
        "premium": premium,
        "prior_only": 0,
        "n_year_ahead": n_year_ahead,
    }


def prep_cashflows_data(fit_dev, fit_stack, n_programs=6, use_monotonic=True):
    # select from programs with monotonically decreasing ATAs for illustration
    mean_log_ata = np.nanmean(fit_dev.stan_variable("log_ata"), axis=0)
    monotonic_programs = [
        i
        for i in range(mean_log_ata.shape[0])
        if use_monotonic or np.all(np.diff(mean_log_ata[i]) <= 0)
    ]
    program_idx = np.random.choice(
        monotonic_programs, min(n_programs, len(monotonic_programs)), replace=False
    )
    # ATAs and variance parameters from development model
    log_ata = fit_dev.stan_variable("log_ata")[:, program_idx, :]
    # ultimate loss for future AY from forecast model
    y_ult = fit_stack.post_pred[:, program_idx, -1]
    return {"atas": np.exp(log_ata), "y_ult": y_ult}, program_idx


def fit_development(data, prior_only=0, prior_scale=1):
    fit = development.sample(
        data=data | {"prior_only": prior_only, "prior_scale": prior_scale},
        **STAN_CONFIG,
    )
    return fit


def fit_forecast_rw(data, prior_only=0, prior_scale=1):
    fit = forecast_rw.sample(
        data=data | {"prior_only": prior_only, "prior_scale": prior_scale},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    return fit


def fit_forecast_rev(data, prior_only=0, prior_scale=1):
    fit = forecast_rev.sample(
        data=data | {"prior_only": prior_only, "prior_scale": prior_scale},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    return fit


def fit_stacking(fits, lob: str | None = None) -> bb.Draws:
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
    if lob is not None:
        with open(f"{RESULTS}/stacking-{lob}.pkl", "wb") as f:
            dill.dump(stack, f)
    return stack_pred


def elpd(fit):
    return logsumexp(fit.log_lik_test[:, :, 0], axis=0) - np.log(
        fit.log_lik_test.shape[0]
    )


def squared_error(data, fit):
    return ((fit.y_pred_test[:, :, -1].mean(axis=0) - data["y"][:, -1]) ** 2).flatten()


def percentile(data, fit):
    return np.mean(fit.y_pred_test[:, :, -1] <= data["y"][:, -1], axis=0).flatten()


def simulate_cashflows(y_ult: np.ndarray, atas: np.ndarray):
    cashflows = np.empty((atas.shape[0], atas.shape[1], atas.shape[2] + 1))
    cashflows[:, :, -1] = y_ult
    for program in range(atas.shape[1]):
        for lag in range(cashflows.shape[2] - 1, 0, -1):
            cashflows[:, program, lag - 1] = (
                cashflows[:, program, lag] / atas[:, program, lag - 1]
            )
    return cashflows


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


def score_stacking(data, pred, lob, suffix=""):
    y_test = data["y"][:, -1]
    percentiles = np.nanmean(pred.post_pred[:, :, -1] <= y_test, axis=0)
    ses = (np.nanmean(pred.post_pred[:, :, -1], axis=0) - data["y"][:, -1]) ** 2
    with open(RESULTS + f"/stack-elpds-{lob}{suffix}.json", "w") as f:
        json.dump(pred.lpd.tolist(), f)
    with open(RESULTS + f"/stack-ses-{lob}{suffix}.json", "w") as f:
        json.dump(ses.tolist(), f)
    with open(RESULTS + f"/stack-percentiles-{lob}{suffix}.json", "w") as f:
        json.dump(percentiles.tolist(), f)
    return pred.lpd, ses, percentiles


def prior_predict_pipeline(stan_data_for, models, lob, prior_scale=1, suffix=""):
    fits = {
        name: model(stan_data_for, prior_only=1, prior_scale=prior_scale)
        for name, model in models.items()
    }
    plot_predictions(stan_data_for, fits, lob, prior_only=1, suffix=suffix)


def save_forecast_parameters(fits, lob):
    def summarize_params(fit, params):
        extracted = {}
        for param in params:
            if param == "gamma":
                extracted["gamma_mu"] = (
                    fit.stan_variable("gamma").mean(axis=(0, 1)).tolist()
                )
                extracted["gamma_sigma"] = (
                    fit.stan_variable("gamma").std(axis=(0, 1)).tolist()
                )
            else:
                extracted[param] = fit.stan_variable(param).mean()
        return extracted

    rw_params = [
        "eta_init_mu",
        "eta_init_sigma",
        "epsilon_mu",
        "epsilon_sigma",
        "gamma",
    ]
    rev_params = rw_params + [
        "mu_mu",
        "mu_sigma",
        "phi_mu",
        "phi_sigma",
    ]

    rw_summary = summarize_params(fits["rw"], rw_params)
    rev_summary = summarize_params(fits["rev"], rev_params)

    with open(f"priors/forecast-rw-{lob}.json", "w") as f:
        json.dump(rw_summary, f)
    with open(f"priors/forecast-rev-{lob}.json", "w") as f:
        json.dump(rev_summary, f)


def main() -> None:
    scores = {}
    scores_diffuse = {}
    scores_inform = {}
    dev_fits = {}
    for_fits = {}
    cash = {}
    for lob in list(DATA):
        data = load_data(lob)
        # development
        stan_data_dev = prep_development_data(data) | PARAMS[lob]
        fit_dev = fit_development(stan_data_dev)
        fit_dev_diffuse = fit_development(stan_data_dev, prior_scale=2.0)
        fit_dev_inform = fit_development(stan_data_dev, prior_scale=0.5)
        # forecast + prior and posterior predictions
        stan_data_for = prep_forecast_data(fit_dev, stan_data_dev)
        stan_data_for_diffuse = prep_forecast_data(fit_dev_diffuse, stan_data_dev)
        stan_data_for_inform = prep_forecast_data(fit_dev_inform, stan_data_dev)
        models = {"rw": fit_forecast_rw, "rev": fit_forecast_rev}
        prior_predict_pipeline(stan_data_for, models, lob)
        prior_predict_pipeline(
            stan_data_for_diffuse, models, lob, prior_scale=2, suffix="-diffuse"
        )
        prior_predict_pipeline(
            stan_data_for_inform, models, lob, prior_scale=0.5, suffix="-inform"
        )
        fits = {name: model(stan_data_for) for name, model in models.items()}
        # save forecast parameters to use as priors
        save_forecast_parameters(fits, lob)
        fits_diffuse = {
            f"{name} (diffuse)": model(stan_data_for_diffuse, prior_scale=2)
            for name, model in models.items()
        }
        fits_inform = {
            f"{name} (inform)": model(stan_data_for_inform, prior_scale=0.5)
            for name, model in models.items()
        }
        plot_predictions(stan_data_for, fits, lob, prior_only=0)
        plot_predictions(
            stan_data_for_diffuse, fits_diffuse, lob, prior_only=0, suffix="-diffuse"
        )
        plot_predictions(
            stan_data_for_inform, fits_inform, lob, prior_only=0, suffix="-inform"
        )
        # score models then stack
        score = {
            name: score_model(stan_data_for, fit, lob, name)
            for name, fit in fits.items()
        }
        score_diffuse = {
            name: score_model(stan_data_for_diffuse, fit, lob, name)
            for name, fit in fits_diffuse.items()
        }
        score_inform = {
            name: score_model(stan_data_for_inform, fit, lob, name)
            for name, fit in fits_inform.items()
        }
        pred_stack = fit_stacking(fits, lob=lob)
        pred_stack_diffuse = fit_stacking(fits_diffuse)
        pred_stack_inform = fit_stacking(fits_inform)
        score["stack"] = score_stacking(stan_data_for, pred_stack, lob)
        score_diffuse["stack (diffuse)"] = score_stacking(
            stan_data_for_diffuse, pred_stack_diffuse, lob, suffix="-diffuse"
        )
        score_inform["stack (inform)"] = score_stacking(
            stan_data_for_inform, pred_stack_inform, lob, suffix="-inform"
        )
        # cashflows
        stan_data_cash, cashflow_program_idx = prep_cashflows_data(fit_dev, pred_stack)
        cashflows = simulate_cashflows(**stan_data_cash)
        premiums = stan_data_for["premium"][cashflow_program_idx, -1]
        # save objects
        scores[lob] = score
        scores_diffuse[lob] = score_diffuse
        scores_inform[lob] = score_inform
        dev_fits[lob] = fit_dev
        for_fits[lob] = fits
        cash[lob] = (cashflows, premiums, cashflow_program_idx)
    return (
        scores,
        scores_diffuse,
        scores_inform,
        dev_fits,
        for_fits,
        cash,
    )


if __name__ == "__main__":
    (scores, scores_diffuse, scores_inform, dev_fits, for_fits, cash) = main()
    plot_scores(scores)
    plot_scores(scores_diffuse, suffix="-diffuse")
    plot_scores(scores_inform, suffix="-inform")
    plot_percentiles(scores)
    for lob, cashflows in cash.items():
        plot_cashflows(*cashflows, lob=lob)
