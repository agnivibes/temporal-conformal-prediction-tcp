import threadpin

threadpin.pin()
"""Build tables and figures from saved forecasts."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--sensitivity", type=Path, required=True)
parser.add_argument("--outdir", type=Path, required=True)
args = parser.parse_args()
out = args.outdir
out.mkdir(parents=True, exist_ok=True)
(out / "tables").mkdir(exist_ok=True)
(out / "figures").mkdir(exist_ok=True)
assets = ["SP500", "BTC-USD", "Gold"]
names = {"SP500": "S\\&P 500", "BTC-USD": "Bitcoin", "Gold": "Gold"}
display = {"SP500": "S&P 500", "BTC-USD": "Bitcoin", "Gold": "Gold"}
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
iv = pd.read_csv(args.results / "ALL_interval.csv")
var = pd.read_csv(args.results / "ALL_var.csv")
sens = pd.read_csv(args.sensitivity)
raw = {p.stem[4:]: np.load(p) for p in sorted(args.results.glob("raw_*.npz"))}
assert len(raw) == 13


def metrics(y, lo, hi):
    coverage = np.mean((lo <= y) & (y <= hi))
    width = np.mean(hi - lo)
    scores = hi - lo + 40 * np.maximum(lo - y, 0) + 40 * np.maximum(y - hi, 0)
    return (coverage, width, scores)


def dm(a, b):
    d = a - b
    n = len(d)
    dc = d - d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    variance = np.dot(dc, dc) / n
    for j in range(1, lag + 1):
        variance += 2 * (1 - j / (lag + 1)) * np.dot(dc[j:], dc[:-j]) / n
    stat = d.mean() / np.sqrt(max(variance, 1e-16) / n)
    return (float(d.mean()), float(2 * norm.sf(abs(stat))))


def pval(p):
    return "$<10^{-4}$" if p < 0.0001 else f"{p:.4f}"


dm_rows, crisis_rows = ([], [])
comparison_count = 0
negative_rm = 0
negative_rm_series = 0
max_offset = 0.0
max_summary_error = 0.0
for asset, z in raw.items():
    y = z["y"]
    dates = pd.DatetimeIndex(z["date"])
    assert len(y) == 1448
    assert set((k[4:] for k in z.files if k.startswith("lo__"))) == set(models)
    scores = {}
    for model in models:
        lo, hi = (z["lo__" + model], z["hi__" + model])
        assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo <= hi)
        coverage, width, score = metrics(y, lo, hi)
        row = iv[(iv.series == asset) & (iv.model == model)].iloc[0]
        err = np.max(
            np.abs(
                np.array([coverage, width, score.mean()])
                - row[["coverage", "width", "winkler"]].to_numpy(float)
            )
        )
        max_summary_error = max(max_summary_error, float(err))
        scores[model] = score
        if asset in assets:
            m = (dates >= "2020-02-01") & (dates <= "2020-04-30")
            cov, wid, sc = metrics(y[m], lo[m], hi[m])
            assert m.sum() == 55
            crisis_rows.append(
                dict(
                    series=asset,
                    model=model,
                    n=int(m.sum()),
                    coverage=cov,
                    width=wid,
                    winkler=sc.mean(),
                )
            )
    for model in models[1:]:
        difference, p = dm(scores["TCP"], scores[model])
        dm_rows.append(dict(series=asset, model=model, difference=difference, p=p))
    gap = z["hi__TCP"] - z["hi__CQR"]
    assert np.all(gap >= -1e-10)
    comparison_count += int(np.sum(np.abs(gap) > 1e-10))
    neg = int(np.sum(z["hi__TCP-RM"] - z["hi__QR-base"] < -1e-10))
    negative_rm += neg
    negative_rm_series += int(neg > 0)
    max_offset = max(max_offset, float(np.max(np.abs(z["hi__TCP-RM"] - z["hi__TCP"]))))
assert max_summary_error < 1e-10
crisis = pd.DataFrame(crisis_rows)
dmt = pd.DataFrame(dm_rows)
vol = dmt[dmt.model.isin(["GARCH-t", "FHS", "GARCH-EVT", "RiskMetrics"])]
tally = [
    int(((vol.difference > 0) & (vol.p < 0.05)).sum()),
    int((vol.p >= 0.05).sum()),
    int(((vol.difference < 0) & (vol.p < 0.05)).sum()),
]
main_sens = sens[(sens.w == 252) & (sens.cal == 60) & (sens.gamma_0 == 0.01)]
for _, row in main_sens.iterrows():
    ref = iv[(iv.series == row.series) & (iv.model == "TCP-RM")].iloc[0]
    assert np.allclose(
        row[["coverage", "width", "winkler"]].to_numpy(float),
        ref[["coverage", "width", "winkler"]].to_numpy(float),
        atol=1e-10,
        rtol=0,
    )
tables = {}
lines = [
    "\\begin{table}[p]",
    "\\centering",
    "\\small",
    "\\caption{Interval forecasts for the three main assets. All rows use the same 1,448 forecast dates and a 95\\% coverage target. Width and the Winkler interval score are in percentage points. The last two rows for each asset are diagnostic controls. Lower interval scores are preferred.}",
    "\\label{tab:results}",
    "\\begin{tabular}{lrrr}",
    "\\toprule",
    "Method & Coverage & Mean width & Winkler score \\\\",
    "\\midrule",
]
for ai, asset in enumerate(assets):
    if ai:
        lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{" + names[asset] + "}} \\\\")
    for model in models:
        if model == "Reversed-sign-control":
            lines.append("\\addlinespace[3pt]")
        row = iv[(iv.series == asset) & (iv.model == model)].iloc[0]
        lines.append(
            f"{labels.get(model, model)} & {row.coverage:.4f} & {row.width:.4f} & {row.winkler:.4f} "
            + "\\\\"
        )
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["MAIN_TABLE"] = "\n".join(lines)
lines = [
    "\\begin{table}[tbp]",
    "\\centering",
    "\\small",
    "\\caption{S\\&P 500 interval performance from February to April 2020. Each method is evaluated on the same 55 recorded observations. The coverage target is 95\\%. Width and the Winkler score are in percentage points.}",
    "\\label{tab:covwidth-sp500-crisis}",
    "\\begin{tabular}{lrrr}",
    "\\toprule",
    "Method & Coverage & Mean width & Winkler score \\\\",
    "\\midrule",
]
for model in models[:10]:
    row = crisis[(crisis.series == "SP500") & (crisis.model == model)].iloc[0]
    lines.append(
        f"{labels.get(model, model)} & {row.coverage:.4f} & {row.width:.4f} & {row.winkler:.4f} "
        + "\\\\"
    )
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["CRISIS_TABLE"] = "\n".join(lines)
lines = [
    "\\begin{table}[tbp]",
    "\\centering",
    "\\small",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\caption{S\\&P 500 lower-tail backtests on 1,448 observations. $p$ is the reference exception probability; the remaining $p$-columns contain test $p$-values. A dagger marks a two-sided interval endpoint tested as a 2.5\\% diagnostic. The two CQR-1s rows use the same forecasts, constructed at requested level $\\tau=0.025$, and test them at the requested and attainable reference levels. UC, IND, CC and DQ denote unconditional coverage, independence, conditional coverage and dynamic quantile tests.}",
    "\\label{tab:backtests-sp500}",
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    "Forecast & $p$ & Exceptions & $p_{\\rm UC}$ & $p_{\\rm IND}$ & $p_{\\rm CC}$ & $p_{\\rm DQ}$ \\\\",
    "\\midrule",
]
for model in models:
    row = var[(var.series == "SP500") & (var.model == model) & (var.tau == 0.025)].iloc[
        0
    ]
    lab = labels.get(model, model)
    if model in ["TCP", "TCP-RM", "CQR", "ACI-fixed", "Reversed-sign-control"]:
        lab += "$^\\dagger$"
    vals = " & ".join((pval(row[key]) for key in ["p_uc", "p_ind", "p_cc", "p_DQ"]))
    lines.append(f"{lab} & 0.025 & {int(row.exceed)} & {vals} " + "\\\\")
lines.append("\\midrule")
for model, level in [("CQR-1s", "0.025"), ("CQR-1s-at-effective-level", "$1/61$")]:
    row = var[(var.series == "SP500") & (var.model == model) & (var.tau == 0.025)].iloc[
        0
    ]
    vals = " & ".join((pval(row[key]) for key in ["p_uc", "p_ind", "p_cc", "p_DQ"]))
    lines.append(f"CQR-1s & {level} & {int(row.exceed)} & {vals} " + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["TAIL_TABLE"] = "\n".join(lines)
lines = [
    "\\begin{table}[p]",
    "\\centering",
    "\\small",
    "\\caption{S\\&P 500 sensitivity to the total window $w$, calibration size $m$ and initial step size $\\gamma_0$. A dash denotes TCP without the online offset. All other rows use TCP-RM. The training size is $w-m$. Sample length $n$ changes with $w$. Width and the Winkler score are in percentage points.}",
    "\\label{tab:sensitivity-sp500}",
    "\\begin{tabular}{rrrrrrr}",
    "\\toprule",
    "$w$ & $m$ & $\\gamma_0$ & $n$ & Coverage & Mean width & Winkler score \\\\",
    "\\midrule",
]
for gi, ((w, cal), group) in enumerate(
    sens[sens.series == "SP500"].groupby(["w", "cal"], sort=True)
):
    if gi:
        lines.append("\\midrule")
    for _, row in group.iterrows():
        g = "--" if row.method == "TCP" else f"{row.gamma_0:.3f}"
        lines.append(
            f"{w} & {cal} & {g} & {int(row.n)} & {row.coverage:.4f} & {row.width:.4f} & {row.winkler:.4f} "
            + "\\\\"
        )
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables["SENSITIVITY_TABLE"] = "\n".join(lines)
for key, table in tables.items():
    (out / "tables" / f"{key.lower()}.tex").write_text(table + "\n")
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
z = raw["SP500"]
dates = pd.DatetimeIndex(z["date"])
mask = (dates >= "2020-02-01") & (dates <= "2020-04-30")
d, y = (dates[mask], z["y"][mask])
fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.7), sharex=True, sharey=True)
for ax, model in zip(
    axes.flat, ["TCP", "ACI-fixed", "QR-base", "GARCH-t", "FHS", "Hist"]
):
    lo, hi = (z["lo__" + model][mask], z["hi__" + model][mask])
    miss = (y < lo) | (y > hi)
    ax.fill_between(d, lo, hi, color="#4676a6", alpha=0.22, lw=0)
    ax.plot(d, y, color="#222222", linewidth=0.8)
    ax.scatter(d[miss], y[miss], s=15, c="#ba3030", zorder=5)
    ax.axhline(0, color="#777777", linewidth=0.55, linestyle=":")
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
    out / "figures" / "tcp_crisis_sp500.png",
    bbox_inches="tight",
    metadata={"Description": "Prediction intervals from the saved local forecasts."},
)
plt.close(fig)
fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
for ax, asset in zip(axes, assets):
    z = raw[asset]
    dates = pd.DatetimeIndex(z["date"])
    mask = (dates >= "2020-02-01") & (dates <= "2020-04-30")
    d, y = (dates[mask], z["y"][mask])
    ax.fill_between(
        d, z["lo__TCP"][mask], z["hi__TCP"][mask], color="#4676a6", alpha=0.2, lw=0
    )
    for side in ["lo__", "hi__"]:
        ax.plot(d, z[side + "TCP"][mask], color="#30669a", linewidth=1.05)
        ax.plot(
            d,
            z[side + "TCP-RM"][mask],
            color="#c76d1a",
            linewidth=1.0,
            linestyle=(0, (5, 3)),
        )
    ax.plot(d, y, color="#222222", linewidth=0.8)
    ax.axhline(0, color="#777777", linewidth=0.5, linestyle=":")
    ax.set_title(display[asset], loc="left")
    ax.set_ylabel("Return (%)")
    ax.grid(axis="y", alpha=0.13)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"))
axes[-1].set_xlabel("2020")
fig.legend(
    handles=[
        Line2D([0], [0], color="#30669a", lw=1.2, label="TCP endpoints"),
        Line2D(
            [0], [0], color="#c76d1a", lw=1.2, ls=(0, (5, 3)), label="TCP-RM endpoints"
        ),
        Line2D([0], [0], color="#222222", lw=1, label="Observed return"),
    ],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8.5,
    bbox_to_anchor=(0.5, 0.003),
)
fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1)
fig.savefig(
    out / "figures" / "tcp_rm_crisis_comparison.png",
    bbox_inches="tight",
    metadata={
        "Description": "TCP and TCP-RM intervals from the saved local forecasts."
    },
)
plt.close(fig)
mad = (
    iv.assign(error=np.abs(iv.coverage - 0.95)).groupby("model").error.mean().to_dict()
)
tail = var[np.isclose(var.p_target, 0.025)].copy()
tail["all3"] = (tail[["p_uc", "p_cc", "p_DQ"]] > 0.05).all(axis=1)
eff = var[(var.model == "CQR-1s-at-effective-level") & (var.tau == 0.025)]
facts = dict(
    n_assets=len(raw),
    n_each=1448,
    max_interval_summary_difference=max_summary_error,
    dm_volatility_tally=tally,
    coverage_mad=mad,
    tcp_cqr_differences=comparison_count,
    rm_negative_radii=negative_rm,
    rm_negative_radius_assets=negative_rm_series,
    max_absolute_rm_offset=max_offset,
    all_three_pass=tail.groupby("model").all3.sum().to_dict(),
    effective_level_cqr_pass={
        c: int((eff[c] > 0.05).sum()) for c in ["p_uc", "p_cc", "p_DQ"]
    },
    sensitivity_source=args.sensitivity.name,
    sensitivity_rows=int(len(sens)),
    sensitivity_main_config_matches=True,
)
(out / "tables" / "checked_facts.json").write_text(json.dumps(facts, indent=2))
dmt.to_csv(out / "tables" / "checked_dm.csv", index=False)
crisis.to_csv(out / "tables" / "checked_crisis.csv", index=False)
iv.to_csv(out / "tables" / "interval_source.csv", index=False)
var.to_csv(out / "tables" / "tail_source.csv", index=False)
sens.to_csv(out / "tables" / "sensitivity_source.csv", index=False)
inputs = [
    args.results / "ALL_interval.csv",
    args.results / "ALL_var.csv",
    args.sensitivity,
]
inputs += sorted(args.results.glob("raw_*.npz"))
manifest = [
    dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs
]
(out / "tables" / "input_hashes.json").write_text(json.dumps(manifest, indent=2))
print(
    json.dumps(
        {
            "main_tables_checked": True,
            "dm_tally": tally,
            "figures": 2,
            "max_summary_error": max_summary_error,
            "sensitivity_main_config_matches": True,
        }
    )
)
