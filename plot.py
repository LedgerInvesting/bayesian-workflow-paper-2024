from typing import List, Dict, Tuple
import numpy as np
import scipy.stats as stat
import re
import matplotlib.pyplot as plt

plt.style.use("publication.mplstyle")

SCALER = 1e4

LETTERS = list(map(chr, range(65, 91)))

MODEL_NAMES = ["Random Walk", "Mean Reversion", "Stacked"]
MODELS = {
    "rw": "Random Walk",
    "rev": "Mean Reversion",
    "rw (diffuse)": "Random Walk (Diffuse)",
    "rev (diffuse)": "Mean Reversion (Diffuse)",
    "rw (inform)": "Random Walk (Informed)",
    "rev (inform)": "Mean Reversion (Informed)",
}

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
    ],
    suffix="",
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
    plt.gcf().set_size_inches(12, 8)
    plt.savefig(PATH + f"/scores{suffix}.png")
    plt.close()


def plot_percentiles(
    scores: Dict[
        str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ],
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
    plt.gcf().set_size_inches(12, 8)
    plt.savefig(PATH + "/calibration.png")
    plt.close()


def plot_development(data, n_ay=18, n_dev=30, program_idx=0, title="", post_pred=None):
    program_losses = data["y"][program_idx].reshape(n_ay, n_dev)
    program_premium = data["premium"][program_idx].reshape(n_ay, n_dev)
    program_losses[np.where(program_losses == -999)] = np.nan
    program_premium[np.where(program_premium == -999)] = np.nan
    program_lr = program_losses / program_premium

    if post_pred is None:
        # Original single plot code
        fig, ax = plt.subplots()
        norm = plt.Normalize(1, n_ay)
        colors = plt.cm.RdBu(norm(np.arange(1, n_ay + 1)))

        for i, year in enumerate(np.arange(n_ay) + 1):
            ax.plot(np.arange(n_dev) + 1, program_lr[i, :], color=colors[i])

        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Accident Year Index")
        cbar.ax.tick_params(labelsize=12)

        ax.set_xlabel("Development Year Index", fontsize=14)
        ax.set_ylabel("Paid Loss Ratio", fontsize=15)
        ax.set_title(f"Loss Development ({title.title()})", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    else:
        # Create subplots for each accident year
        n_cols = 3
        n_rows = (n_ay + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        # Calculate posterior statistics and mask observed values
        pred_means = np.mean(post_pred, axis=0)[program_idx].reshape(n_ay, n_dev)
        pred_50 = np.percentile(post_pred, [25, 75], axis=0)[:, program_idx, :].reshape(
            2, n_ay, n_dev
        )
        pred_95 = np.percentile(post_pred, [2.5, 97.5], axis=0)[
            :, program_idx, :
        ].reshape(2, n_ay, n_dev)

        # Mask out predictions where we have observed data
        pred_means[np.where(~np.isnan(program_losses))] = np.nan
        pred_50[
            :,
            np.where(~np.isnan(program_losses))[0],
            np.where(~np.isnan(program_losses))[1],
        ] = np.nan
        pred_95[
            :,
            np.where(~np.isnan(program_losses))[0],
            np.where(~np.isnan(program_losses))[1],
        ] = np.nan

        # Convert to loss ratios
        pred_lr_means = pred_means / program_premium[:, 0].reshape(-1, 1)
        pred_lr_50 = pred_50 / program_premium[:, 0].reshape(-1, 1)
        pred_lr_95 = pred_95 / program_premium[:, 0].reshape(-1, 1)

        for i in range(n_ay):
            ax = axes[i]

            # Plot observed values
            ax.plot(
                np.arange(n_dev) + 1,
                program_lr[i, :],
                color="black",
                label="Observed",
                linewidth=2,
            )

            observed_mask = ~np.isnan(program_lr[i, :])
            ax.scatter(
                np.arange(n_dev)[observed_mask] + 1,
                program_lr[i, observed_mask],
                color="black",
                s=30,
                zorder=5,
                label="_nolegend_",
            )

            # Plot posterior predictions with intervals
            ax.plot(
                np.arange(n_dev) + 1,
                pred_lr_means[i, :],
                color="#c40e1d",
                label="Posterior Mean",
                linewidth=2,
            )
            ax.fill_between(
                np.arange(n_dev) + 1,
                pred_lr_50[0, i, :],
                pred_lr_50[1, i, :],
                color="#c40e1d",
                alpha=0.3,
                label="50% CI",
            )
            ax.fill_between(
                np.arange(n_dev) + 1,
                pred_lr_95[0, i, :],
                pred_lr_95[1, i, :],
                color="#c40e1d",
                alpha=0.1,
                label="95% CI",
            )
            # title
            ax.set_title(f"AY {i+1}")
            ax.grid(True)
            # Add y-axis label only for middle subplot on left side
            middle_row = n_rows // 2
            if i == middle_row * n_cols:  # Middle-left subplot
                ax.set_ylabel("    Paid Loss Ratio")

            # Add x-axis label only for middle subplot on bottom row
            bottom_middle = n_ay - (n_ay % n_cols) - 2  # Index of bottom middle subplot
            if i == bottom_middle:  # Bottom-middle subplot
                ax.set_xlabel("Development Year")

        # Remove empty subplots
        for i in range(n_ay, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(
            f"Loss Development by Accident Year ({title.title()})", fontsize=24
        )

    if post_pred is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4
        )
        plt.gcf().set_size_inches(12, 12)
    else:
        plt.gcf().set_size_inches(5, 4)
    file_name = f"{PATH}/development-curves-{title.lower()}.png"
    plt.savefig(file_name)
    plt.close()


def plot_predictions(
    data,
    fits,
    lob,
    prior_only,
    suffix="",
    n_idx=[6, 35, 46],
    y_pred_train="y_pred_train",
    y_pred_test="y_pred_test",
):
    R, C = (len(n_idx), 2)
    S, N = next(iter(fits.values())).stan_variable(y_pred_train).shape[:2]
    has_test_preds = y_pred_test is not None
    y_test = data["y"][n_idx, :] / data["premium"][n_idx, :]
    y_pred = {
        model: (
            fit.stan_variable(y_pred_train)
            if not has_test_preds
            else np.concatenate(
                [fit.stan_variable(y_pred_train), fit.stan_variable(y_pred_test)],
                axis=2,
            )
        )[:, n_idx, :]
        for model, fit in fits.items()
    }
    models = list(y_pred)
    y_preds = list(y_pred.values())
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C)
    ax = np.atleast_2d(ax)
    x = [i + 1 for i in range(y_preds[0].shape[2])]

    for i, (r, c) in enumerate(rc):
        if has_test_preds:
            ax[r, c].scatter(
                x=x,
                y=y_test[r, :],
                color="gray",
                edgecolor="black",
                label="True Value" if i == 0 else None,
                zorder=5,
            )

        # Calculate mean and quantiles for current predictions
        premium = data["premium"][n_idx[r], :]
        y = y_preds[c][:, r, :]

        # Extend premium if needed
        missing_premium = y.shape[1] - premium.shape[0]
        if missing_premium > 0:
            premium = np.append(premium, [premium[-1]] * missing_premium)

        # Calculate normalized predictions
        y_normalized = y / premium

        # Calculate statistics
        mean_pred = np.nanpercentile(y_normalized, 50, axis=0)
        ci_50 = np.nanpercentile(y_normalized, [25, 75], axis=0)
        ci_95 = np.nanpercentile(y_normalized, [2.5, 97.5], axis=0)

        # Get latest ULR statistics
        latest_mean = np.nanmean(y_normalized[:, -1])
        latest_sd = np.nanstd(y_normalized[:, -1])

        # Add text box with latest ULR stats
        stats_text = f"Latest AY ULR\nMean: {latest_mean:.2f}\nSD: {latest_sd:.2f}"
        ax[r, c].text(
            0.95,
            0.9 if has_test_preds else 0.2,
            stats_text,
            fontsize=None if has_test_preds else 15,
            transform=ax[r, c].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
        )

        # Plot mean and confidence intervals
        ax[r, c].plot(
            x,
            mean_pred,
            color="#c40e1d",
            label=(
                f"{'Prior' if prior_only else 'Posterior'} Median" if (i == 0) else None
            ),
            zorder=2,
            linewidth=2,
        )
        ax[r, c].fill_between(
            x,
            ci_50[0],
            ci_50[1],
            color="#c40e1d",
            alpha=0.3,
            label="50% CI" if (i == 0) else None,
            zorder=1,
        )
        ax[r, c].fill_between(
            x,
            ci_95[0],
            ci_95[1],
            color="#c40e1d",
            alpha=0.1,
            label="95% CI" if (i == 0) else None,
            zorder=0,
        )

        if r == 0:
            ax[r, c].set_title(f"{MODELS[models[c]]}")
        if len(n_idx) == 1:
            ax[r, c].set_xlabel("Accident Year Index")
            ax[r, c].set_ylabel("Ultimate Loss Ratio")
        else:
            if r == 2:
                ax[r, c].set_xlabel("Accident Year Index")
            if (r, c) == (1, 0):
                ax[r, c].set_ylabel("Ultimate Loss Ratio")
        # ax[r, c].grid(True)
        ax[r, c].set_xticks(x)
        ax[r, c].set_xticklabels(x)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    if has_test_preds:
        fig.legend(
            handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05)
        )
        plt.gcf().set_size_inches(13, len(n_idx) * 4)
    else:
        fig.legend(
            handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12)
        )
        plt.gcf().set_size_inches(15, len(n_idx) * 6)
    plt.savefig(
        PATH
        + ("/prior" if prior_only else "/posterior")
        + f"-predictions-{lob}{suffix}.png"
    )
    plt.close()


def plot_cashflows(
    cashflows: np.ndarray, premiums: np.ndarray, program_idx: List[int], lob: str
):
    cashflows = cashflows * SCALER / 1e3
    premiums = premiums * SCALER / 1e3
    timepoints = np.arange(cashflows.shape[2]) + 1
    if len(program_idx) == 1:
        R, C = 1, 1
    else:
        R, C = 3, 2
    fig, axs = plt.subplots(R, C, figsize=(15, 10), sharex=True)
    axs = np.atleast_2d(axs)
    for i, idx in enumerate(program_idx):
        r, c = divmod(i, C)
        ax = axs[r, c]
        mean_cashflows = np.nanmean(cashflows[:, i, :], axis=0)
        ci_50 = np.nanpercentile(cashflows[:, i, :], [25, 75], axis=0)
        ci_95 = np.nanpercentile(cashflows[:, i, :], [2.5, 97.5], axis=0)
        ci_99 = np.nanpercentile(cashflows[:, i, :], [0.5, 99.5], axis=0)
        ax.plot(timepoints, mean_cashflows, color="#c40e1d", label="Mean")
        ax.fill_between(
            timepoints, ci_50[0], ci_50[1], color="#c40e1d", alpha=0.3, label="50% CI"
        )
        ax.fill_between(
            timepoints, ci_95[0], ci_95[1], color="#c40e1d", alpha=0.2, label="95% CI"
        )
        ax.fill_between(
            timepoints, ci_99[0], ci_99[1], color="#c40e1d", alpha=0.1, label="99% CI"
        )
        ax.axhline(
            y=premiums[i], color="#332288", linestyle="--", label="Premium", linewidth=2
        )
        # mean + SD of ultimate LR
        mean_ulr = np.nanmean(cashflows[:, i, -1] / premiums[i])
        sd_ulr = np.nanstd(cashflows[:, i, -1] / premiums[i])
        ax.text(
            0.95,
            0.05,
            f"ULR Mean: {mean_ulr:.2f}\nSD: {sd_ulr:.2f}",
            fontsize=20,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            color="black",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )
        ax.set_xticks(timepoints)
        ax.set_xticklabels(timepoints)
        if len(program_idx) == 1:
            ax.set_xlabel("Development Year Index")
            ax.set_ylabel("Paid loss ($1000s)")
            ax.tick_params(axis="both", which="major")
        else:
            if r == 2:
                ax.set_xlabel("Development Year Index")
            if r == 1 and c == 0:
                ax.set_ylabel("Paid loss ($1000s)")
            ax.set_title(f"Program {idx}")
    for i in range(len(program_idx), R * C):
        fig.delaxes(axs.flatten()[i])
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f"Underwriting Cashflows ({lob.upper()})", fontsize=30)
    if len(program_idx) == 1:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5
        )
        plt.gcf().set_size_inches(11, 6)
    else:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5
        )
        plt.gcf().set_size_inches(14, 16)
    plt.savefig(PATH + f"/cashflows-{lob}.png")
    plt.close()
