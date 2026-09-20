"""GARCH-t, filtered historical simulation and GARCH-EVT forecasts."""

import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import genpareto
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def _filter(r, mu, om, a, b, s2_0):
    n = len(r)
    s2 = np.empty(n + 1)
    s2[0] = s2_0
    for t in range(n):
        s2[t + 1] = om + a * (r[t] - mu) ** 2 + b * s2[t]
    return s2


def _degenerate(p_):
    mu, om, a, b, nu = p_
    if not np.isfinite([mu, om, a, b, nu]).all():
        return True
    return (
        om <= 1e-10 or a < 1e-06 or b < 1e-06 or (a + b > 1.0 + 1e-06) or (nu <= 2.05)
    )


def _robust_fit(y, prev, strict=False):
    v = float(np.var(y))
    starts = [
        None,
        np.array([float(np.mean(y)), 0.05 * v, 0.08, 0.9, 6.0]),
        np.array([0.0, 0.2 * v, 0.15, 0.75, 5.0]),
        np.array([float(np.median(y)), 0.02 * v, 0.05, 0.93, 8.0]),
    ]
    if prev is not None:
        starts.insert(
            1,
            np.array(
                [
                    prev[0],
                    max(prev[1], 1e-08),
                    min(max(prev[2], 0.02), 0.3),
                    min(max(prev[3], 0.5), 0.95),
                    max(prev[4], 3.0),
                ]
            ),
        )
    am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    best = bad = None
    bll = badll = -np.inf
    tried = 0
    bestflag = badflag = -1
    beststart = badstart = -1
    for sv in starts:
        tried += 1
        try:
            kw = dict(disp="off", show_warning=False, options={"maxiter": 800})
            if sv is not None:
                kw["starting_values"] = sv
            res = am.fit(**kw)
            pr = res.params
            cand = (
                float(pr["mu"]),
                float(pr["omega"]),
                float(pr["alpha[1]"]),
                float(pr["beta[1]"]),
                float(pr["nu"]),
            )
            ll = float(res.loglikelihood)
            flag = int(getattr(res, "convergence_flag", -1))
            if not np.isfinite(ll):
                continue
            if _degenerate(cand):
                if ll > badll and np.isfinite(cand).all() and (cand[1] > 0):
                    bad, badll, badflag, badstart = (cand, ll, flag, tried)
                continue
            if ll > bll:
                best, bll, bestflag, beststart = (cand, ll, flag, tried)
        except Exception:
            continue
    if best is not None:
        return (best, bll, tried, "accepted", bestflag, beststart)
    if strict:
        return (None, -np.inf, tried, "rejected", -1, -1)
    if bad is not None:
        return (bad, badll, tried, "fallback", badflag, badstart)
    return (None, -np.inf, tried, "rejected", -1, -1)


def run(
    r,
    dates,
    start,
    taus_lo=(0.01, 0.025, 0.05),
    tau_hi=0.975,
    fit_window=750,
    refit_every=25,
    z_window=500,
    evt_thr=0.1,
    refit_anchor=None,
    strict=False,
):
    r = np.asarray(r, float)
    n = len(r)
    idx = np.arange(start, n)
    d = pd.DatetimeIndex(dates)[idx]
    keys = [f"q{t}" for t in taus_lo] + [f"es{t}" for t in taus_lo] + [f"q{tau_hi}"]
    G = {"date": d, "y": r[idx], "sigma": np.full(len(idx), np.nan)}
    F = {"date": d, "y": r[idx]}
    E = {"date": d, "y": r[idx]}
    for k in keys:
        G[k] = np.full(len(idx), np.nan)
        F[k] = np.full(len(idx), np.nan)
        E[k] = np.full(len(idx), np.nan)
    diag = []
    params = None
    fit_lo = None
    carried = 0
    for j, t in enumerate(idx):
        anchor = start if refit_anchor is None else refit_anchor
        if params is None or (t - anchor) % refit_every == 0:
            lo = max(0, t - fit_window)
            new, ll, nstart, status, optflag, which_start = _robust_fit(
                r[lo:t], params, strict=strict
            )
            degen = 0 if status == "accepted" else 1
            if new is not None:
                params = new
                fit_lo = lo
                carried = 0
                diag.append(
                    {
                        "t": t,
                        "date": str(d[j].date()),
                        "mu": params[0],
                        "omega": params[1],
                        "alpha": params[2],
                        "beta": params[3],
                        "nu": params[4],
                        "persist": params[2] + params[3],
                        "optimizer_converged": int(optflag == 0),
                        "optimizer_flag": optflag,
                        "start_used": which_start,
                        "n_fit": t - lo,
                        "loglik": ll,
                        "n_starts_tried": nstart,
                        "degenerate": degen,
                        "status": status,
                        "carried_forward": 0,
                        "near_unit_persistence": int(params[2] + params[3] >= 0.999),
                    }
                )
            else:
                carried = 1
                diag.append(
                    {
                        "t": t,
                        "date": str(d[j].date()),
                        "optimizer_converged": 0,
                        "optimizer_flag": optflag,
                        "start_used": which_start,
                        "n_fit": t - lo,
                        "degenerate": 1,
                        "n_starts_tried": nstart,
                        "status": status,
                        "carried_forward": int(params is not None),
                        "no_fit_available": int(params is None),
                        "persist": (
                            params[2] + params[3] if params is not None else np.nan
                        ),
                        "near_unit_persistence": (
                            int(params[2] + params[3] >= 0.999)
                            if params is not None
                            else 0
                        ),
                    }
                )
                if params is None:
                    continue
        mu, om, a, b, nu = params
        s2 = _filter(r[fit_lo:t], mu, om, a, b, np.var(r[fit_lo : min(fit_lo + 60, t)]))
        sig_t = np.sqrt(s2[-1])
        G["sigma"][j] = sig_t
        z = (r[fit_lo:t] - mu) / np.sqrt(s2[:-1])
        z = z[np.isfinite(z)]
        if len(z) > z_window:
            z = z[-z_window:]
        sc = np.sqrt(nu / (nu - 2.0))
        for tau in taus_lo:
            tq = stats.t.ppf(tau, nu)
            G[f"q{tau}"][j] = mu + sig_t * tq / sc
            G[f"es{tau}"][j] = (
                mu
                + sig_t
                * (-(stats.t.pdf(tq, nu) / tau) * ((nu + tq**2) / (nu - 1.0)))
                / sc
            )
        G[f"q{tau_hi}"][j] = mu + sig_t * stats.t.ppf(tau_hi, nu) / sc
        if len(z) < 100:
            continue
        for tau in taus_lo:
            qz = np.quantile(z, tau)
            tl = z[z <= qz]
            F[f"q{tau}"][j] = mu + sig_t * qz
            F[f"es{tau}"][j] = mu + sig_t * (tl.mean() if len(tl) else qz)
        F[f"q{tau_hi}"][j] = mu + sig_t * np.quantile(z, tau_hi)
        E[f"q{tau_hi}"][j] = F[f"q{tau_hi}"][j]
        L = -z
        u = np.quantile(L, 1 - evt_thr)
        exc = L[L > u] - u
        if len(exc) < 25:
            continue
        try:
            xi, _, bta = genpareto.fit(exc, floc=0.0)
        except Exception:
            continue
        if not np.isfinite(xi) or xi >= 1.0 or bta <= 0:
            continue
        Nu = len(exc) / len(L)
        for tau in taus_lo:
            pu = tau
            if pu >= Nu:
                E[f"q{tau}"][j] = F[f"q{tau}"][j]
                E[f"es{tau}"][j] = F[f"es{tau}"][j]
                continue
            vq = (
                u + bta / xi * ((pu / Nu) ** (-xi) - 1)
                if abs(xi) > 1e-08
                else u - bta * np.log(pu / Nu)
            )
            esL = vq / (1 - xi) + (bta - xi * u) / (1 - xi)
            E[f"q{tau}"][j] = mu - sig_t * vq
            E[f"es{tau}"][j] = mu - sig_t * esL
    return (G, F, E, pd.DataFrame(diag))
