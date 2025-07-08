# Ledger's Bayesian workflow scripts

This repository contains the scripts used to reproduce the analyses in "A Bayesian workflow for securitizing casualty insurance risk" by Haines,
Goold, \& Shoun (2024).

The Bayesian models themselves are implemented in Stan scripts located in the `bayesian-workflow-paper-2024/stan/` directory.

## Install requirements

All analyses were run with `Python 3.11`. Once `Python 3.11` is installed locally, we recommend the
following steps:

1. navigate to your local `bayesian-workflow-paper-2024/` directory
2. initialize a virtual environment: `python3.11 venv env`
3. activate the environment: `source env/bin/activate`
4. install requirements: `pip install -r requirements/requirements.txt`

## Reproduce analyses

Analyses can then be reproduced by running the following scripts in order:

1. download and pre-process the data: `python -m pull`
2. run simulation-based calibration: `python -m sbc`
3. run backtests along with prior and posterior predictive checks: `python -m backtest`
4. run Balona (2020) working example: `python -m analyze`

Once analyses are reproduced, figures are located in the `bayesian-workflow-paper-2024/figures/` directory.
