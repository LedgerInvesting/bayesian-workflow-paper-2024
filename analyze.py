import json
import dill

import cmdstanpy as csp
import bayesblend as bb

from backtest import (
    prep_development_data,
    prep_forecast_data,
    prep_cashflows_data,
    simulate_cashflows,
)

from plot import (
    plot_development,
    plot_predictions,
    plot_cashflows,
)

# Configuration
DATA = {"OO": "data/balona-2020.json"}

PRIORS = {
    "rw": "priors/forecast-rw-OO.json",
    "rev": "priors/forecast-rev-OO.json",
}

# stacking model fitted to out-of-sample data per backtest.py
STACKING_MODEL = "results/stacking-OO.pkl"

# body and tail training cutoffs
PARAMS = {"OO": dict(tau=6, rho=[4, 15])}

# how many years out to make predictions with tail model
MAX_DEV = 30

STAN_CONFIG = {
    "iter_sampling": 1250,
    "iter_warmup": 500,
    "parallel_chains": 8,
    "chains": 8,
    "inits": 0,
    "seed": 1234,
    "show_progress": True,
    "adapt_delta": 0.99,
    "max_treedepth": 11,
}

MODELS = {
    "development": csp.CmdStanModel(stan_file="stan/development.stan"),
    "forecast_rw": csp.CmdStanModel(stan_file="stan/forecast-random-walk-nonhier.stan"),
    "forecast_rev": csp.CmdStanModel(stan_file="stan/forecast-reversion-nonhier.stan"),
}

for model, prior_path in PRIORS.items():
    with open(prior_path, "r") as f:
        PRIORS[model] = json.load(f)
        PRIORS[model] = {
            param.replace("_mu", "__loc").replace("_sigma", "__scale"): value
            for param, value in PRIORS[model].items()
        }


def fit_development_model(data):
    return MODELS["development"].sample(
        data=data | {"prior_only": 0, "prior_scale": 1, **PARAMS["OO"]},
        **STAN_CONFIG,
    )


def fit_forecasting_models(data):
    fit_rw = MODELS["forecast_rw"].sample(
        data=data | {"prior_only": 0, **PRIORS["rw"]},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    fit_rev = MODELS["forecast_rev"].sample(
        data=data | {"prior_only": 0, **PRIORS["rev"]},
        **STAN_CONFIG | {"adapt_delta": 0.99},
    )
    return {"rw": fit_rw, "rev": fit_rev}


def blend_predictions(fits):
    with open(STACKING_MODEL, "rb") as f:
        stack = dill.load(f)
    draws = {
        name: bb.Draws.from_cmdstanpy(fit, post_pred_name="y_pred")
        for name, fit in fits.items()
    }
    return stack.predict(draws)


def main():
    # Load data
    with open(DATA["OO"], "r") as f:
        data = json.load(f)
    stan_data_dev = prep_development_data(data, pad=True, max_dev=MAX_DEV)

    # Fit development model
    fit_dev = fit_development_model(stan_data_dev)

    # Prepare forecast data
    stan_data_for = prep_forecast_data(
        fit_dev, stan_data_dev, n_ay=18, n_dev=MAX_DEV, n_leave_out=0, n_year_ahead=1
    )

    # Fit forecasting models
    fits = fit_forecasting_models(stan_data_for)

    # Blend predictions using stacking
    stack_pred = blend_predictions(fits)

    # Apply cashflow model
    stan_data_cash, cashflow_program_idx = prep_cashflows_data(
        fit_dev, stack_pred, 1, False
    )
    cashflows = simulate_cashflows(**stan_data_cash)
    premiums = stan_data_for["premium"][cashflow_program_idx, -1]

    # Plot results
    plot_development(stan_data_dev, title="Observed")
    plot_development(
        stan_data_dev,
        title="Predicted",
        post_pred=fit_dev.stan_variable("y_tilde"),
    )
    plot_predictions(
        stan_data_for,
        fits,
        "OO",
        prior_only=0,
        n_idx=cashflow_program_idx,
        suffix="-balona-2020",
        y_pred_train="y_pred",
        y_pred_test=None,
    )
    plot_cashflows(cashflows, premiums, cashflow_program_idx, lob="OO-balona-2020")


if __name__ == "__main__":
    main()
