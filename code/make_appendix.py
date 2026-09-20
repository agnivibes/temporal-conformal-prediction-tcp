import threadpin

threadpin.pin()
"""Build tables and figures from saved forecasts."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--sensitivity", type=Path, required=True)
ap.add_argument("--outdir", type=Path, required=True)
args = ap.parse_args()
out = args.outdir
table_dir = out / "tables"
models = [
    "TCP",
    "TCP-RM",
    "CQR",
    "ACI-fixed",
    "QR-base",
    "GARCH-t",
    "FHS",
    "GARCH-EVT",
    "RiskMetrics",
    "Hist",
    "Reversed-sign-control",
    "Fixed-coefficient-control",
]
labels = {
    "ACI-fixed": "ACI (clipped)",
    "Hist": "Historical simulation",
    "Reversed-sign-control": "Reversed-sign control",
    "Fixed-coefficient-control": "Fixed-coefficient control",
}
assets = ["SP500", "BTC-USD", "Gold"]
display = {"SP500": "S&P 500", "BTC-USD": "Bitcoin", "Gold": "Gold"}
texnames = {"SP500": "S\\&P 500", "BTC-USD": "Bitcoin", "Gold": "Gold"}
filetags = {"SP500": "sp500", "BTC-USD": "bitcoin", "Gold": "gold"}
iv = pd.read_csv(args.results / "ALL_interval.csv")
var = pd.read_csv(args.results / "ALL_var.csv")
sens = pd.read_csv(args.sensitivity)
dmt = pd.read_csv(table_dir / "checked_dm.csv")
crisis = pd.read_csv(table_dir / "checked_crisis.csv")
raw = {p.stem[4:]: np.load(p) for p in sorted(args.results.glob("raw_*.npz"))}
assert len(raw) == 13


def pval(p):
    return "$<10^{-4}$" if p < 0.0001 else f"{p:.4f}"


def bernoulli_ll(x, n, p):
    return xlogy(x, p) + xlogy(n - x, 1 - p)


def backtests(y, q, p):
    h = (y < q).astype(int)
    n = len(h)
    x = h.sum()
    uc = max(0.0, 2 * (bernoulli_ll(x, n, x / n) - bernoulli_ll(x, n, p)))
    counts = np.bincount(2 * h[:-1] + h[1:], minlength=4)
    n00, n01, n10, n11 = counts
    n0, n1 = (n00 + n01, n10 + n11)
    assert n0 > 0 and n1 > 0
    q0, q1 = (n01 / n0, n11 / n1)
    qm = (n01 + n11) / (n0 + n1)
    ind = max(
        0.0,
        2
        * (
            bernoulli_ll(n01, n0, q0)
            + bernoulli_ll(n11, n1, q1)
            - bernoulli_ll(n01 + n11, n0 + n1, qm)
        ),
    )
    centered = h - p
    X = np.column_stack(
        [np.ones(n - 4)] + [centered[4 - j : n - j] for j in range(1, 5)] + [q[4:]]
    )
    fitted = X @ np.linalg.lstsq(X, centered[4:], rcond=None)[0]
    dq = fitted @ fitted / (p * (1 - p))
    return dict(
        exceed=int(x),
        p_uc=chi2.sf(uc, 1),
        p_ind=chi2.sf(ind, 1),
        p_cc=chi2.sf(uc + ind, 2),
        p_DQ=chi2.sf(dq, 6),
    )


max_p_error = 0.0
rechecked_tail = []
for _, row in var.iterrows():
    z = raw[row.series]
    model = "CQR-1s" if row.model == "CQR-1s-at-effective-level" else row.model
    key = f"var__{model}__{row.tau:g}"
    q = z[key]
    assert np.isfinite(q).all() and np.isfinite(z["y"]).all()
    computed = backtests(z["y"], q, float(row.p_target))
    assert computed["exceed"] == int(row.exceed)
    for col in ["p_uc", "p_ind", "p_cc", "p_DQ"]:
        max_p_error = max(max_p_error, abs(float(row[col]) - computed[col]))
    rechecked_tail.append(
        dict(
            series=row.series,
            model=row.model,
            tau=row.tau,
            p_target=row.p_target,
            **computed,
        )
    )
assert max_p_error < 1e-10
pd.DataFrame(rechecked_tail).to_csv(
    table_dir / "recomputed_tail_tests.csv", index=False
)
iv["absolute_coverage_error"] = (iv.coverage - 0.95).abs()
iv["winkler_rank"] = iv.groupby("series").winkler.rank(method="average", ascending=True)
summary = (
    iv.groupby("model")
    .agg(
        coverage_mad=("absolute_coverage_error", "mean"),
        mean_winkler_rank=("winkler_rank", "mean"),
    )
    .reindex(models)
)
summary["tcp_better"] = 0
summary["no_difference"] = 0
summary["competitor_better"] = 0
for model in models[1:]:
    d = dmt[dmt.model == model]
    assert len(d) == 13
    summary.loc[model, "tcp_better"] = int(((d.difference < 0) & (d.p < 0.05)).sum())
    summary.loc[model, "no_difference"] = int((d.p >= 0.05).sum())
    summary.loc[model, "competitor_better"] = int(
        ((d.difference > 0) & (d.p < 0.05)).sum()
    )
summary.to_csv(table_dir / "all_series_summary.csv")
tables = {}
lines = [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\small",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\caption{Interval results across thirteen series. Coverage MAD is the mean absolute difference between each series' coverage and 0.95. Interval-score ranks are calculated within each series across all twelve rows, with rank one best, and then averaged. The last three columns count two-sided Diebold--Mariano comparisons of TCP against the listed method at the unadjusted 5\\% level. ND denotes no significant difference.}",
    "\\label{tab:all-series-summary}",
    "\\begin{tabular}{lrrrrr}",
    "\\toprule",
    "Method & Coverage MAD & Mean rank & TCP better & ND & Other better \\\\",
    "\\midrule",
]
for model, row in summary.iterrows():
    if model == "Reversed-sign-control":
        lines.append("\\midrule")
    counts = (
        "-- & -- & --"
        if model == "TCP"
        else f"{int(row.tcp_better)} & {int(row.no_difference)} & {int(row.competitor_better)}"
    )
    lines.append(
        f"{labels.get(model, model)} & {row.coverage_mad:.6f} & {row.mean_winkler_rank:.3f} & {counts} "
        + "\\\\"
    )
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["SERIES_TABLE"] = "\n".join(lines)
for asset in ["BTC-USD", "Gold"]:
    tag = filetags[asset]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{"
        + display[asset]
        + " interval performance from February to April 2020. All methods use the same 55 recorded observations and a nominal 95\\% coverage target. Width and the Winkler interval score are in percentage points.}",
        "\\label{tab:covwidth-"
        + ("btc" if asset == "BTC-USD" else "gold")
        + "-crisis}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Coverage & Mean width & Winkler score \\\\",
        "\\midrule",
    ]
    for model in models[:10]:
        row = crisis[(crisis.series == asset) & (crisis.model == model)].iloc[0]
        lines.append(
            f"{labels.get(model, model)} & {row.coverage:.4f} & {row.width:.4f} & {row.winkler:.4f} "
            + "\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tables["CRISIS_" + tag.upper()] = "\n".join(lines)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{"
        + display[asset]
        + " lower-tail backtests on 1,448 observations. Here $p$ is the reference exception probability. A dagger marks a two-sided interval endpoint used as a 2.5\\% diagnostic. The two CQR-1s rows use the same forecasts, constructed at requested level $\\tau=0.025$, with different reference probabilities. Test abbreviations follow Table~\\ref{tab:backtests-sp500}.}",
        "\\label{tab:backtests-" + tag + "}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Forecast & $p$ & Exceptions & $p_{\\rm UC}$ & $p_{\\rm IND}$ & $p_{\\rm CC}$ & $p_{\\rm DQ}$ \\\\",
        "\\midrule",
    ]
    for model in models + ["CQR-1s", "CQR-1s-at-effective-level"]:
        row = var[
            (var.series == asset) & (var.model == model) & (var.tau == 0.025)
        ].iloc[0]
        if model == "CQR-1s":
            lines.append("\\midrule")
        lab = labels.get(model, model)
        if model in ["TCP", "TCP-RM", "CQR", "ACI-fixed", "Reversed-sign-control"]:
            lab += "$^\\dagger$"
        p = "$1/61$" if model == "CQR-1s-at-effective-level" else "0.025"
        if model == "CQR-1s-at-effective-level":
            lab = "CQR-1s"
        values = " & ".join((pval(row[c]) for c in ["p_uc", "p_ind", "p_cc", "p_DQ"]))
        lines.append(f"{lab} & {p} & {int(row.exceed)} & {values} " + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tables["TAIL_" + tag.upper()] = "\n".join(lines)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{"
        + display[asset]
        + " sensitivity to total window $w$, calibration size $m$ and initial step size $\\gamma_0$. A dash denotes TCP without the online offset; the other rows use TCP-RM. The training size is $w-m$. Width and the Winkler score are in percentage points. Sample size $n$ varies with $w$.}",
        "\\label{tab:sensitivity-" + tag + "}",
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "$w$ & $m$ & $\\gamma_0$ & $n$ & Coverage & Mean width & Winkler score \\\\",
        "\\midrule",
    ]
    for j, ((w, m), group) in enumerate(
        sens[sens.series == asset].groupby(["w", "cal"], sort=True)
    ):
        if j:
            lines.append("\\midrule")
        for _, row in group.iterrows():
            g = "--" if row.method == "TCP" else f"{row.gamma_0:.3f}"
            lines.append(
                f"{w} & {m} & {g} & {int(row.n)} & {row.coverage:.4f} & {row.width:.4f} & {row.winkler:.4f} "
                + "\\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tables["SENSITIVITY_" + tag.upper()] = "\n".join(lines)
lines = [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\small",
    "\\caption{Numbers of series with test $p$-values strictly greater than 0.05, out of thirteen series. The final column requires UC, CC and DQ all to exceed 0.05. A dagger marks a two-sided endpoint diagnostic. The two CQR-1s rows evaluate the same forecasts at different reference probabilities. These are descriptive test counts, not a joint test or proof of calibration.}",
    "\\label{tab:tail-test-counts}",
    "\\begin{tabular}{lrrrrr}",
    "\\toprule",
    "Forecast & Reference $p$ & UC & CC & DQ & All three \\\\",
    "\\midrule",
]
tail_counts = []
for model in models + ["CQR-1s", "CQR-1s-at-effective-level"]:
    rows = var[(var.model == model) & (var.tau == 0.025)]
    assert len(rows) == 13
    passed = rows[["p_uc", "p_cc", "p_DQ"]].gt(0.05)
    counts = [int(passed[col].sum()) for col in passed.columns] + [
        int(passed.all(axis=1).sum())
    ]
    lab = labels.get(model, model)
    if model in ["TCP", "TCP-RM", "CQR", "ACI-fixed", "Reversed-sign-control"]:
        lab += "$^\\dagger$"
    if model == "CQR-1s":
        lines.append("\\midrule")
    p = "$1/61$" if model == "CQR-1s-at-effective-level" else "0.025"
    if model == "CQR-1s-at-effective-level":
        lab = "CQR-1s"
    lines.append(f"{lab} & {p} & " + " & ".join(map(str, counts)) + " \\\\")
    tail_counts.append(
        dict(model=model, uc=counts[0], cc=counts[1], dq=counts[2], all_three=counts[3])
    )
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["TAIL_COUNTS"] = "\n".join(lines)
pd.DataFrame(tail_counts).to_csv(table_dir / "tail_test_counts.csv", index=False)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }
)
for asset in ["BTC-USD", "Gold"]:
    z = raw[asset]
    dates = pd.DatetimeIndex(z["date"])
    mask = (dates >= "2020-02-01") & (dates <= "2020-04-30")
    assert mask.sum() == 55
    d, y = (dates[mask], z["y"][mask])
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.7), sharex=True, sharey=True)
    for ax, model in zip(
        axes.flat, ["TCP", "ACI-fixed", "QR-base", "GARCH-t", "FHS", "Hist"]
    ):
        lo, hi = (z["lo__" + model][mask], z["hi__" + model][mask])
        miss = (y < lo) | (y > hi)
        ax.fill_between(d, lo, hi, color="#4676a6", alpha=0.22, lw=0)
        ax.plot(d, y, color="#222222", lw=0.8)
        ax.scatter(d[miss], y[miss], s=15, c="#ba3030", zorder=5)
        ax.axhline(0, color="#777777", lw=0.55, ls=":")
        ax.set_title(
            f"{labels.get(model, model)}\nCoverage {100 * (1 - miss.mean()):.1f}%; width {np.mean(hi - lo):.2f}"
        )
        ax.set_xlim(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(axis="y", alpha=0.13)
    for ax in axes[:, 0]:
        ax.set_ylabel("Return (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("2020")
    fig.legend(
        handles=[
            Patch(facecolor="#4676a6", alpha=0.22, label="Nominal 95% interval"),
            Line2D([0], [0], color="#222222", lw=1, label="Observed return"),
            Line2D(
                [0],
                [0],
                color="#ba3030",
                marker="o",
                lw=0,
                markersize=4,
                label="Interval miss",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.3)
    fig.savefig(
        out / "figures" / f"tcp_crisis_{filetags[asset]}.png", bbox_inches="tight"
    )
    plt.close(fig)
trace_checks = []
for asset, z in raw.items():
    dates = pd.DatetimeIndex(z["date"])
    y = z["y"]
    n = len(y)
    c = z["hi__TCP"] - z["hi__QR-base"]
    b = z["hi__TCP-RM"] - z["hi__TCP"]
    c_error = np.max(np.abs(c - (z["lo__QR-base"] - z["lo__TCP"])))
    b_error = np.max(np.abs(b - (z["lo__TCP"] - z["lo__TCP-RM"])))
    hit_tcp = (z["lo__TCP"] <= y) & (y <= z["hi__TCP"])
    hit_rm = (z["lo__TCP-RM"] <= y) & (y <= z["hi__TCP-RM"])
    gamma = 0.01 / (1 + 0.1 * np.arange(1, n + 1)) ** 0.7
    b_next = np.maximum(b[:-1] + gamma[:-1] * (~hit_rm[:-1] - 0.05), -c[:-1])
    recurrence_error = np.max(np.abs(b[1:] - b_next))
    assert max(c_error, b_error, recurrence_error) < 1e-10 and b[0] == 0
    budgets = np.r_[0, np.cumsum(gamma[:-1])]
    assert np.all(b >= -0.05 * budgets - 1e-10) and np.all(b <= 0.95 * budgets + 1e-10)
    trace_checks.append(
        dict(
            series=asset,
            c_endpoint_error=float(c_error),
            b_endpoint_error=float(b_error),
            recurrence_error=float(recurrence_error),
            negative_effective_expansion=int((c + b < -1e-10).sum()),
            max_abs_b=float(np.max(np.abs(b))),
        )
    )
    if asset not in assets:
        continue
    rolling_tcp = (
        pd.Series(hit_tcp.astype(float)).rolling(30, min_periods=30).mean().to_numpy()
    )
    rolling_rm = (
        pd.Series(hit_rm.astype(float)).rolling(30, min_periods=30).mean().to_numpy()
    )
    pd.DataFrame(
        {
            "date": dates,
            "y": y,
            "c": c,
            "b": b,
            "effective_expansion": c + b,
            "rolling30_TCP": rolling_tcp,
            "rolling30_TCP_RM": rolling_rm,
        }
    ).to_csv(table_dir / f"trace_values_{filetags[asset]}.csv", index=False)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 8.3),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2, 1.2]},
    )
    axes[0].fill_between(
        dates, z["lo__TCP"], z["hi__TCP"], color="#4676a6", alpha=0.2, lw=0
    )
    for side in ["lo__", "hi__"]:
        axes[0].plot(dates, z[side + "TCP"], color="#30669a", lw=0.65)
        axes[0].plot(
            dates, z[side + "TCP-RM"], color="#c76d1a", lw=0.65, ls=(0, (5, 3))
        )
    axes[0].plot(dates, y, color="#222222", lw=0.45)
    axes[0].set_title(display[asset], loc="left", fontweight="bold")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#30669a", lw=1, label="TCP"),
            Line2D([0], [0], color="#c76d1a", lw=1, ls="--", label="TCP-RM"),
            Line2D([0], [0], color="#222222", lw=1, label="Observed return"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=7.5,
        ncol=3,
    )
    axes[1].plot(dates, 100 * rolling_tcp, color="#30669a", lw=0.95, label="TCP")
    axes[1].plot(
        dates,
        100 * rolling_rm,
        color="#c76d1a",
        lw=0.95,
        ls=(0, (5, 3)),
        label="TCP-RM",
    )
    axes[1].axhline(95, color="#555555", lw=0.7, ls=":", label="95% target")
    axes[1].set_ylabel("Rolling coverage\n(%, 30 observations)")
    axes[1].legend(loc="lower right", frameon=True, fontsize=7.5, ncol=3)
    axes[1].set_ylim(max(0, 100 * np.nanmin([rolling_tcp, rolling_rm]) - 5), 102)
    axes[2].plot(dates, c, color="#30669a", lw=0.9, label="$c_t$")
    axes[2].plot(
        dates, c + b, color="#c76d1a", lw=0.9, ls=(0, (5, 3)), label="$c_t+b_t$"
    )
    axes[2].axhline(0, color="#555555", lw=0.6, ls=":")
    axes[2].set_ylabel("Expansion\n(percentage points)")
    axes[2].legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
    axes[3].plot(dates, b, color="#6b4c9a", lw=1.0)
    axes[3].axhline(0, color="#555555", lw=0.6, ls=":")
    axes[3].set_ylabel("Offset $b_t$" + "\n(percentage points)")
    axes[3].set_xlabel("Forecast date")
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.set_xlim(dates[0], dates[-1])
        ax.grid(axis="y", alpha=0.13)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(
        out / "figures" / f"tcp_rm_trace_{filetags[asset]}.png", bbox_inches="tight"
    )
    plt.close(fig)
for key, table in tables.items():
    (table_dir / f"{key.lower()}.tex").write_text(table + "\n")
pd.DataFrame(trace_checks).to_csv(table_dir / "trace_checks.csv", index=False)
facts = dict(
    tail_rows_recomputed=len(rechecked_tail),
    max_tail_p_value_difference=float(max_p_error),
    trace_recurrences_checked=13,
    max_trace_recurrence_difference=max((x["recurrence_error"] for x in trace_checks)),
    step_sum_1448=float(gamma.sum()),
    post_1448_update_bounds=[float(-0.05 * gamma.sum()), float(0.95 * gamma.sum())],
    appendix_figures=5,
    appendix_tables=len(tables),
)
(table_dir / "appendix_checks.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts))
