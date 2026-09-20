"""Fit all study methods on common forecast dates."""

import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import threadpin as _tp

_tp.pin()
import sys, os, json, time, argparse
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import genpareto

sys.path.insert(0, os.path.dirname(__file__))
import core, evaluation as ev
import warnings

warnings.filterwarnings("ignore")
ALPHA = 0.05
TAUS_LO = (0.01, 0.025, 0.05)
TAUS_ALL = (0.01, 0.025, 0.05, 0.975)
METHODS = (
    "TCP",
    "TCP-RM",
    "CQR",
    "ACI-fixed",
    "Reversed-sign-control",
    "QR-base",
    "GARCH-t",
    "FHS",
    "GARCH-EVT",
    "RiskMetrics",
    "Hist",
    "Fixed-coefficient-control",
)


def classical_block(r, dates, start, garch_fit_window, refit_every, z_window):
    import garch_family as gf

    G, F, E, diag = gf.run(
        r,
        dates,
        start,
        taus_lo=TAUS_LO,
        tau_hi=0.975,
        fit_window=garch_fit_window,
        refit_every=refit_every,
        z_window=z_window,
    )
    n = len(r)
    idx = np.arange(start, n)
    d = pd.DatetimeIndex(dates)[idx]
    out = {"GARCH-t": G, "FHS": F, "GARCH-EVT": E}
    v = np.empty(n + 1)
    v[0] = np.var(r[:50])
    for t in range(n):
        v[t + 1] = 0.94 * v[t] + 0.06 * r[t] ** 2
    s = np.sqrt(v[:n])[idx]
    w = {"date": d, "y": r[idx], "q0.975": s * stats.norm.ppf(0.975)}
    for tau in TAUS_LO:
        z0 = stats.norm.ppf(tau)
        w[f"q{tau}"] = s * z0
        w[f"es{tau}"] = s * (-stats.norm.pdf(z0) / tau)
    out["RiskMetrics"] = w
    h = {"date": d, "y": r[idx], "q0.975": np.empty(len(idx))}
    for tau in TAUS_LO:
        h[f"q{tau}"] = np.empty(len(idx))
        h[f"es{tau}"] = np.empty(len(idx))
    for j, t in enumerate(idx):
        wnd = r[t - 252 : t]
        for tau in TAUS_LO:
            q = np.quantile(wnd, tau)
            tl = wnd[wnd <= q]
            h[f"q{tau}"][j] = q
            h[f"es{tau}"][j] = tl.mean() if len(tl) else q
        h["q0.975"][j] = np.quantile(wnd, 0.975)
    out["Hist"] = h
    vol = np.zeros(n)
    vol[0] = np.std(r[:50])
    for t in range(1, n):
        vol[t] = np.sqrt(1e-06 + 0.05 * r[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
    k = {"date": d, "y": r[idx], "q0.975": stats.norm.ppf(0.975) * vol[idx]}
    for tau in TAUS_LO:
        z0 = stats.norm.ppf(tau)
        k[f"q{tau}"] = vol[idx] * z0
        k[f"es{tau}"] = vol[idx] * (-stats.norm.pdf(z0) / tau)
    out["Fixed-coefficient-control"] = k
    return (out, diag)


def run_asset(
    name,
    r,
    dates,
    w=252,
    cal=60,
    learner="lgbm",
    seed=0,
    garch_fit_window=750,
    refit_every=25,
    z_window=500,
    outdir=".",
    slim=False,
):
    t0 = time.time()
    pos, X, y = core.build_features(r)
    start_raw = int(pos[w])
    cache = core.base_qr_cache(
        r, dates, taus=(0.025, 0.975), window=w, cal=cal, learner=learner, seed=seed
    )
    cache_lo = {}
    for tau in TAUS_LO:
        if tau == 0.025:
            cache_lo[tau] = {
                "date": cache["date"],
                "y": cache["y"],
                "q_cal": cache["q_cal"][:, [0], :],
                "y_cal": cache["y_cal"],
                "q_test": cache["q_test"][:, [0]],
            }
        elif not slim:
            c = core.base_qr_cache(
                r, dates, taus=(tau,), window=w, cal=cal, learner=learner, seed=seed
            )
            cache_lo[tau] = c
    print(f"  [{name}] base caches done {time.time() - t0:.0f}s", flush=True)
    methods = {}
    lo, hi, tr = core.wrap_tcp(cache, ALPHA, rm=False)
    methods["TCP"] = (lo, hi, tr)
    lo, hi, tr = core.wrap_tcp(cache, ALPHA, rm=True)
    methods["TCP-RM"] = (lo, hi, tr)
    lo, hi, tr = core.wrap_tcp(cache, ALPHA, rm=False, truncate=False)
    methods["CQR"] = (lo, hi, tr)
    lo, hi, tr = core.wrap_aci(cache, ALPHA, eta=0.05)
    methods["ACI-fixed"] = (lo, hi, tr)
    lo, hi, tr = core.wrap_aci(cache, ALPHA, eta=0.05, reverse_sign=True)
    methods["Reversed-sign-control"] = (lo, hi, tr)
    methods["QR-base"] = (cache["q_test"][:, 0], cache["q_test"][:, 1], {})
    print(f"  [{name}] conformal done {time.time() - t0:.0f}s", flush=True)
    cls, gdiag = classical_block(
        r, dates, start_raw, garch_fit_window, refit_every, z_window
    )
    print(f"  [{name}] classical done {time.time() - t0:.0f}s", flush=True)
    dref = pd.DatetimeIndex(cache["date"])
    yref = cache["y"]
    for k, v in cls.items():
        assert (
            len(v["date"]) == len(dref) and (pd.DatetimeIndex(v["date"]) == dref).all()
        ), f"timestamp mismatch: {k}"
        assert np.allclose(v["y"], yref), f"outcome mismatch: {k}"
    n = len(dref)
    rows, iv_store, var_store = ([], {}, {})
    for k, (lo_, hi_, _) in methods.items():
        iv_store[k] = (np.asarray(lo_), np.asarray(hi_))
    for k, v in cls.items():
        iv_store[k] = (v["q0.025"], v["q0.975"])
    eff_levels = {}
    for tau in TAUS_LO:
        if tau not in cache_lo:
            continue
        c = cache_lo[tau]
        try:
            q1s, eff = core.wrap_cqr_onesided(c, tau)
            var_store["CQR-1s", tau] = q1s
            eff_levels[tau] = eff
        except core.ConformalLevelUnattainable as e:
            print(
                f"  [{name}] one-sided conformal VaR at tau={tau} SKIPPED: {e}",
                flush=True,
            )
            eff_levels[tau] = None
        var_store["QR-base", tau] = c["q_test"][:, 0]
    pd.DataFrame(
        [
            {
                "series": name,
                "tau": k,
                "effective_guaranteed_miscoverage": v,
                "cal_size": cal,
                "attainable_floor": core.attainable_level(cal),
            }
            for k, v in eff_levels.items()
        ]
    ).to_csv(f"{outdir}/conformal_levels_{name}.csv", index=False)
    for k, v in cls.items():
        for tau in TAUS_LO:
            var_store[k, tau] = v[f"q{tau}"]
    for k in ["TCP", "TCP-RM", "ACI-fixed", "Reversed-sign-control", "CQR"]:
        if k in iv_store:
            var_store[k, 0.025] = iv_store[k][0]
    missing = sorted(set(METHODS) - set(iv_store))
    assert not missing, f"method(s) missing from the runner: {missing}"
    assert set(iv_store) == set(
        METHODS
    ), f"runner produced an unexpected method set: {sorted(set(iv_store) ^ set(METHODS))}"
    res_iv, res_var, losses = ([], [], {})
    for k, (lo_, hi_) in iv_store.items():
        met, cov, wink = ev.interval_metrics(yref, lo_, hi_, ALPHA)
        met.update({"series": name, "model": k})
        res_iv.append(met)
        losses[k] = wink
    for (k, tau), q in var_store.items():
        esv = cls[k][f"es{tau}"] if k in cls else None
        bt, exc = ev.var_backtests(yref, q, tau, esv)
        bt.update({"series": name, "model": k, "tau": tau})
        res_var.append(bt)
    pd.DataFrame(res_iv).to_csv(f"{outdir}/iv_{name}.csv", index=False)
    pd.DataFrame(res_var).to_csv(f"{outdir}/var_{name}.csv", index=False)
    gdiag.to_csv(f"{outdir}/garchdiag_{name}.csv", index=False)
    np.savez_compressed(
        f"{outdir}/raw_{name}.npz",
        date=dref.values.astype("datetime64[ns]"),
        y=yref,
        **{f"lo__{k}": v[0] for k, v in iv_store.items()},
        **{f"hi__{k}": v[1] for k, v in iv_store.items()},
        **{f"var__{k}__{tau}": np.asarray(v) for (k, tau), v in var_store.items()},
        **{
            f"es__{k}__{tau}": np.asarray(cls[k][f"es{tau}"])
            for k in cls
            for tau in TAUS_LO
        },
    )
    print(f"  [{name}] DONE n={n} {time.time() - t0:.0f}s", flush=True)
    return (pd.DataFrame(res_iv), pd.DataFrame(res_var))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--assets",
        default="SP500,FTSE,N225,DAX,BTC-USD,ETH-USD,LTC-USD,Gold,WTI,NatGas,EURUSD=X,GBPUSD=X,JPYUSD=X",
    )
    ap.add_argument("--outdir", default="results")
    ap.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "financial_returns.csv"
        ),
    )
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--cal", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slim", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.data, index_col=0, parse_dates=True)
    df.rename(
        columns={
            "^GSPC": "SP500",
            "GC=F": "Gold",
            "CL=F": "WTI",
            "NG=F": "NatGas",
            "('VIX', '^VIX')": "VIX",
            "^FTSE": "FTSE",
            "^N225": "N225",
            "^GDAXI": "DAX",
        },
        inplace=True,
    )
    IV, VR = ([], [])
    for s in a.assets.split(","):
        ser = df[s].dropna()
        i, v = run_asset(
            s,
            ser.values,
            ser.index,
            w=a.window,
            cal=a.cal,
            seed=a.seed,
            outdir=a.outdir,
            slim=a.slim,
        )
        IV.append(i)
        VR.append(v)
    tag = "_breadth" if a.slim else ""
    pd.concat(IV).to_csv(f"{a.outdir}/ALL_interval{tag}.csv", index=False)
    pd.concat(VR).to_csv(f"{a.outdir}/ALL_var{tag}.csv", index=False)
    print("saved combined tables")
