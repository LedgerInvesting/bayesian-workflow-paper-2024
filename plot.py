from typing import List, Dict, Tuple
import numpy as np
import scipy.stats as stat
import re
import matplotlib.pyplot as plt

plt.style.use("publication.mplstyle")

SCALER = 1e4

LETTERS = list(map(chr, range(65, 91)))

MODEL_NAMES = ["Random Walk", "Mean Reversion", "Stacked"]
MODELS = {"rw": "Random Walk", "rev": "Mean Reversion"}

LOB_COLORS = {
    "PP": "#332288",
    "WC": "#44aa99",
    "CA": "#88ccee",
    "OO": "#ddcc77",
}

PATH = "figures/"

DataType = List[List[int]]


def _flatten_ranks(rank: np.ndarray):
    if isinstance(rank, (list, np.ndarray)):
        return {i: _flatten_ranks(r) for i, r in enumerate(rank)}
    else:
        return rank


def flat_ranks(ranks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    flattened_ranks = {}
    for par, rank in ranks.items():
        if par in ["L"]:
            continue
        flat = _flatten_ranks(rank)
        for _, f in flat.items():
            if isinstance(f, dict):
                for idx, r in f.items():
                    name = f"{par}[{idx}]"
                    if name in flattened_ranks:
                        flattened_ranks[name] = np.append(flattened_ranks[name], r)
                    else:
                        flattened_ranks[name] = np.array([r])
            else:
                if par in flattened_ranks:
                    flattened_ranks[par] = np.append(flattened_ranks[par], f)
                else:
                    flattened_ranks[par] = np.array([f])
    return {k: v for k, v in flattened_ranks.items()}


def plot_ranks(ranks: Dict[str, np.ndarray]) -> None:
    flattened_ranks = flat_ranks(ranks)
    R = 3
    C = 4
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, sharex=True, sharey=True)
    keys = list(flattened_ranks)
    L = int(ranks["L"])
    M = len(flattened_ranks[keys[0]])
    bins = int(min(L + 1, max(np.floor(M / 10), 5)))
    lower = stat.binom(M, np.floor((L + 1) / bins) / L).ppf(0.005)
    upper = stat.binom(M, np.ceil((L + 1) / bins) / L).ppf(0.995)
    mean = stat.binom(M, 1 / bins).ppf(0.5)
    hist_pars = {"bins": bins, "color": "white", "edgecolor": "black"}
    greeks = ["alpha", "gamma", "omega", "beta", "lambda"]
    nudge = int(L * 0.1)
    uniform = [
        (-nudge, lower),
        (0, mean),
        (-nudge, upper),
        (L + nudge, upper),
        (L, mean),
        (L + nudge, lower),
    ]
    for i, (r, c) in enumerate(rc):
        if i >= len(flattened_ranks):
            break
        key = keys[i]
        clean_key = key.replace("_", "").replace("[", "_").replace("]", "")
        clean_key = (
            re.sub(r"[0-9]", lambda i: str(int(i.group(0)) + 1), clean_key)
            if "tilde" not in clean_key
            else "$y(1,10)$"
        )
        rank = flattened_ranks[key]
        ax[r, c].fill(
            *zip(*uniform), color="lightgray", edgecolor="skyblue", lw=2, alpha=0.5
        )
        ax[r, c].hist(rank, **hist_pars)
        if any(greek in key for greek in greeks):
            ax[r, c].set_title(rf"$\{clean_key}$")
        else:
            ax[r, c].set_title(clean_key)
        if r == (R - 1):
            ax[r, c].set_xlabel("Rank statistics")
        if not c:
            ax[r, c].set_ylabel("Frequency")
    plt.savefig(PATH + "/ranks.png")
    plt.close()


def plot_scores(
    scores: Dict[
        str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]
) -> None:
    R = len(scores)
    C = 2
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C)
    for i, (r, c) in enumerate(rc):
        lob = list(scores)[r]
        score_raw = np.array(list(scores[lob].values()))[:, c, :]
        score = np.sqrt(score_raw) * SCALER / 1e3 if c == 1 else score_raw
        if c == 1:
            ordered = sorted((s, i) for i, s in enumerate(score.mean(axis=1)))
            diffs = np.array(
                [
                    score[min(ordered)[1], :] - score[m, :]
                    for m in [i for _, i in ordered]
                ]
            )
            diffs_mu = diffs.mean(axis=1)
            diffs_se = np.sqrt(diffs.var(axis=1, ddof=1) / score.shape[-1])
        else:
            ordered = sorted(
                [(s, i) for i, s in enumerate(score.sum(axis=1))],
                reverse=True,
            )
            diffs = np.array(
                [
                    score[max(ordered)[1], :] - score[m, :]
                    for m in [i for _, i in ordered]
                ]
            )
            diffs_mu = diffs.sum(axis=1)
            diffs_se = np.sqrt(diffs.var(axis=1, ddof=1) * score.shape[-1])
        names = [MODEL_NAMES[i] for _, i in ordered]
        lowers, uppers = diffs_mu - diffs_se * 2, diffs_mu + diffs_se * 2
        marker = "^" if c else "o"
        errors = [
            (abs(low - mu), abs(mu - high))
            for mu, low, high in zip(diffs_mu, lowers, uppers)
        ]
        for i, (name, mu, error) in enumerate(zip(names, diffs_mu, errors)):
            ax[r, c].errorbar(
                mu,
                i,
                xerr=error[0],
                color=LOB_COLORS[lob],
                ecolor="black",
                elinewidth=2,
                fmt=marker,
                alpha=0.3 if mu == 0.0 else 1,
                label=lob.upper() if i == 1 else None,
            )
            if not i:
                ax[r, c].text(
                    0,
                    -0.3,
                    ordered[0][0].round(1),
                    fontsize=12,
                    ha="center",
                    va="center",
                )
        ax[r, c].set_yticks(range(score.shape[0]))
        ax[r, c].set_yticklabels(names)
        ax[r, c].axvline(0, ls=":", color="gray")
        ax[r, c].set_ylim(ax[r, c].get_ylim()[::-1])
        if r == (R - 1):
            ax[r, c].set_xlabel("ELPD difference" * (1 - c) + c * "RMSE difference")
        if not c:
            ax[r, c].text(
                -0.1,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )
    labels_handles = {
        label + str(i % 2): handle
        for i, ax in enumerate(fig.axes)
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(
        [h for l, h in labels_handles.items() if l[-1] == "0"],
        [l[:-1] for l, h in labels_handles.items() if l[-1] == "0"],
        title="ELPD difference +/- 2 SE",
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.32, 1.1),
    )
    fig.legend(
        [h for l, h in labels_handles.items() if l[-1] == "1"],
        [l[:-1] for l, h in labels_handles.items() if l[-1] == "1"],
        title="RMSE +/- 2 SE difference ($1000s)",
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.775, 1.1),
    )
    plt.savefig(PATH + "/scores.png")
    plt.close()


def plot_percentiles(
    scores: Dict[
        str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]
):
    percentiles = {
        lob: {model: metrics[2] for model, metrics in scores[lob].items()}
        for lob in scores
    }
    R = len(percentiles)
    C = len(next(iter(percentiles.values())))
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, sharex=True, sharey="row")
    bins = 30
    pad = 5
    L = 100
    M = next(iter(next(iter(percentiles.values())).values())).size
    bins = int(min(L + 1, max(np.floor(M / 10), 5)))
    lower = stat.binom(M, 1 / bins).ppf(0.005)
    upper = stat.binom(M, 1 / bins).ppf(0.995)
    mean = stat.binom(M, 1 / bins).ppf(0.5)
    nudge = L * 0.1
    uniform = [
        (-nudge, lower),
        (0, mean),
        (-nudge, upper),
        (L + nudge, upper),
        (L, mean),
        (L + nudge, lower),
    ]
    hist_pars = dict(alpha=0.6, bins=bins, edgecolor="black")
    for i, (r, c) in enumerate(rc):
        lob = list(percentiles)[r]
        model = list(percentiles[lob].keys())[c]
        p = percentiles[lob][model] * 100
        ax[r, c].fill(
            *zip(*uniform), color="lightgray", edgecolor="skyblue", lw=2, alpha=0.5
        )
        ax[r, c].hist(
            p.flatten(), color=LOB_COLORS[lob], **hist_pars, label=lob.upper()
        )
        if not c:
            ax[r, c].set_ylabel("Frequency")
            ax[r, c].text(
                -0.125,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )
        if r == (R - 1):
            ax[r, c].set_xlabel("Percentile")
        if not r:
            ax[r, c].set_title(MODEL_NAMES[c])
    labels_handles = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.5, 1.075),
    )
    plt.savefig(PATH + "/calibration.png")
    plt.close()


# def plot_percentiles_development(percentiles: Dict[str, np.ndarray]):
#     percentiles = {
#         lob: {model: metrics[2] for model, metrics in scores[lob].items()}
#         for lob in scores
#     }
#     R = len(percentiles)
#     C = len(next(iter(percentiles.values())))
#     rc = [(r, c) for r in range(R) for c in range(C)]
#     fig, ax = plt.subplots(R, C, sharex=True, sharey="row")
#     bins = 30
#     pad = 5
#     L = 100
#     M = next(iter(next(iter(percentiles.values())).values())).size
#     bins = int(min(L + 1, max(np.floor(M / 10), 5)))
#     lower = stat.binom(M, 1 / bins).ppf(0.005)
#     upper = stat.binom(M, 1 / bins).ppf(0.995)
#     mean = stat.binom(M, 1 / bins).ppf(0.5)
#     nudge = L * 0.1
#     uniform = [
#         (-nudge, lower),
#         (0, mean),
#         (-nudge, upper),
#         (L + nudge, upper),
#         (L, mean),
#         (L + nudge, lower),
#     ]
#     hist_pars = dict(alpha=0.6, bins=bins, edgecolor="black")
#     for i, (r, c) in enumerate(rc):
#         lob = list(percentiles)[r]
#         model = list(percentiles[lob].keys())[c]
#         p = percentiles[lob][model] * 100
#         ax[r, c].fill(
#             *zip(*uniform), color="lightgray", edgecolor="skyblue", lw=2, alpha=0.5
#         )
#         ax[r, c].hist(
#             p.flatten(), color=LOB_COLORS[lob], **hist_pars, label=lob.upper()
#         )
#         if not c:
#             ax[r, c].set_ylabel("Frequency")
#             ax[r, c].text(
#                 -0.125,
#                 1.1,
#                 LETTERS[r],
#                 transform=ax[r, c].transAxes,
#                 fontsize=20,
#                 fontweight="bold",
#             )
#         if r == (R - 1):
#             ax[r, c].set_xlabel("Percentile")
#         if not r:
#             ax[r, c].set_title(MODEL_NAMES[c])
#     labels_handles = {
#         label: handle
#         for ax in fig.axes
#         for handle, label in zip(*ax.get_legend_handles_labels())
#     }
#     fig.legend(
#         labels_handles.values(),
#         labels_handles.keys(),
#         loc="upper center",
#         ncol=len(LOB_COLORS),
#         bbox_to_anchor=(0.5, 1.075),
#     )
#     plt.savefig(PATH + "/calibration.png")
#     plt.close()


def plot_predictions(data, fits, lob, prior_only, samples=30):
    R, C = (3, 2)
    S, N = next(iter(fits.values())).y_pred_train.shape[:2]
    n_idx = [6, 35, 46]
    s_idx = np.random.choice(range(S), samples, replace=False)
    y_test = data["y"][n_idx, :]
    y_pred = {
        model: np.concatenate([fit.y_pred_train, fit.y_pred_test], axis=2)[s_idx, :, :][
            :, n_idx, :
        ]
        for model, fit in fits.items()
    }
    models = list(y_pred)
    y_preds = list(y_pred.values())
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C)
    x = [i + 1 for i in range(y_test.shape[1])]
    for i, (r, c) in enumerate(rc):
        ax[r, c].scatter(
            x=x,
            y=y_test[r, :],
            color="gray",
            edgecolor="black",
            label="True Value" if i == 0 else None,
            zorder=1,
        )
        for j, y in enumerate(y_preds[c][:, r, :]):
            ax[r, c].plot(
                x,
                y,
                color="#c40e1d",
                alpha=0.5,
                lw=3,
                label=(
                    f"{'Prior' if prior_only else 'Posterior'} Predicted Value"
                    if (i == 0 and j == 0)
                    else None
                ),
                zorder=1,
            )
        if r == 0:
            ax[r, c].set_title(f"{MODELS[models[c]]}")
        if r == 2:
            ax[r, c].set_xlabel("Accident Year Index")
        if (r, c) == (1, 0):
            ax[r, c].set_ylabel("Ultimate Loss ($10,000s)")
        ax[r, c].grid(True)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        PATH + ("/prior" if prior_only else "/posterior") + f"-predictions-{lob}.png"
    )
    plt.close()
