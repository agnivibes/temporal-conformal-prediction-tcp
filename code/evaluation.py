"""Interval scores, lower-tail backtests and paired score comparisons."""

import numpy as np, pandas as pd, math
from scipy.stats import chi2, norm


def interval_metrics(y, lo, hi, alpha=0.05):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    if not ok.all():
        y, lo, hi = (y[ok], lo[ok], hi[ok])
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    winkler = (
        width + 2 / alpha * np.maximum(0, lo - y) + 2 / alpha * np.maximum(0, y - hi)
    )
    return (
        {
            "coverage": float(cov.mean()),
            "width": float(width.mean()),
            "winkler": float(winkler.mean()),
            "n": int(len(y)),
            "n_missing": int((~ok).sum()),
        },
        cov.astype(int),
        winkler,
    )


def kupiec(exc, p):
    exc = np.asarray(exc, dtype=int)
    T = exc.size
    x = int(exc.sum())
    if T == 0:
        return (np.nan, np.nan, x, T)
    if x == 0:
        LR = -2 * (T * math.log(1 - p))
    elif x == T:
        LR = -2 * (T * math.log(p))
    else:
        pi = x / T
        LR = -2 * ((T - x) * math.log((1 - p) / (1 - pi)) + x * math.log(p / pi))
    LR = max(LR, 0.0)
    return (LR, float(1 - chi2.cdf(LR, 1)), x, T)


def christoffersen_ind(exc):
    z = np.asarray(exc, dtype=int)
    if len(z) < 2:
        return (np.nan, np.nan)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(z)):
        a, b = (z[i - 1], z[i])
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    n0, n1 = (n00 + n01, n10 + n11)
    if n0 == 0 or n1 == 0:
        return (np.nan, np.nan)
    p01, p11 = (n01 / n0, n11 / n1)
    p = (n01 + n11) / (n0 + n1)

    def ll(n_, q):
        return 0.0 if n_ == 0 else n_ * math.log(max(q, 1e-300))

    L1 = ll(n01, p01) + ll(n00, 1 - p01) + ll(n11, p11) + ll(n10, 1 - p11)
    L0 = ll(n01 + n11, p) + ll(n00 + n10, 1 - p)
    LR = max(-2 * (L0 - L1), 0.0)
    return (LR, float(1 - chi2.cdf(LR, 1)))


def cc_test(exc, p):
    lu, _, x, T = kupiec(exc, p)
    li, _ = christoffersen_ind(exc)
    if not np.isfinite(lu) or not np.isfinite(li):
        return (np.nan, np.nan)
    LR = lu + li
    return (LR, float(1 - chi2.cdf(LR, 2)))


def dq_test(exc, var, p, lags=4):
    exc = np.asarray(exc, dtype=float)
    var = np.asarray(var, dtype=float)
    h = exc - p
    T = len(h)
    if T <= lags + 3:
        return (np.nan, np.nan)
    Z = [np.ones(T - lags)]
    for L in range(1, lags + 1):
        Z.append(h[lags - L : T - L])
    Z.append(var[lags:])
    Z = np.column_stack(Z)
    hh = h[lags:]
    try:
        b = np.linalg.lstsq(Z, hh, rcond=None)[0]
        stat = float(b @ (Z.T @ Z) @ b / (p * (1 - p)))
    except Exception:
        return (np.nan, np.nan)
    return (stat, float(1 - chi2.cdf(stat, Z.shape[1])))


def var_backtests(y, var_q, p, es=None):
    y = np.asarray(y)
    v = np.asarray(var_q)
    ok = np.isfinite(v) & np.isfinite(y)
    y, v = (y[ok], v[ok])
    exc = (y < v).astype(int)
    lu, pu, x, T = kupiec(exc, p)
    li, pi_ = christoffersen_ind(exc)
    lc, pc = cc_test(exc, p)
    dq, pdq = dq_test(exc, v, p)
    out = {
        "p_target": p,
        "exceed": x,
        "n": T,
        "rate": x / max(T, 1),
        "LR_uc": lu,
        "p_uc": pu,
        "LR_ind": li,
        "p_ind": pi_,
        "LR_cc": lc,
        "p_cc": pc,
        "DQ": dq,
        "p_DQ": pdq,
        "pinball": float(np.mean(pinball_loss(y, v, p))),
    }
    if es is not None:
        e = np.asarray(es)[ok]
        out["FZ0"] = float(np.nanmean(fz0_loss(y, v, e, p)))
        out["ES_ratio"] = _es_ratio(y, v, e)
    return (out, exc)


def _es_ratio(y, v, e):
    m = y < v
    if m.sum() < 5:
        return np.nan
    return float(np.mean(y[m]) / np.mean(e[m]))


def pinball_loss(y, q, tau):
    d = np.asarray(y) - np.asarray(q)
    return np.where(d >= 0, tau * d, (tau - 1) * d)


def fz0_loss(y, v, e, alpha):
    y, v, e = map(np.asarray, (y, v, e))
    e = np.where(e >= -1e-12, np.nan, e)
    ind = (y <= v).astype(float)
    return -(1.0 / (alpha * e)) * ind * (v - y) + v / e + np.log(-e) - 1.0


def dm_test(l1, l2, hac_lags=None):
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 20:
        return (np.nan, np.nan, np.nan)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    dbar = d.mean()
    dc = d - dbar
    g0 = np.dot(dc, dc) / T
    s = g0
    for L in range(1, hac_lags + 1):
        gl = np.dot(dc[L:], dc[:-L]) / T
        s += 2 * (1 - L / (hac_lags + 1)) * gl
    s = max(s, 1e-16)
    stat = dbar / np.sqrt(s / T)
    return (float(dbar), float(stat), float(2 * (1 - norm.cdf(abs(stat)))))
