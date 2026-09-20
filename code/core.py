"""Features, rolling quantile forecasts and conformal calibration."""

import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import chi2, genpareto
import warnings

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb

    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def build_features(returns, n_lags=5):
    """Build predictors from returns strictly before each forecast date."""
    s = pd.Series(np.asarray(returns, dtype=float))
    df = pd.DataFrame({"return": s})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["return"].shift(lag)
    df["vol20"] = df["return"].shift(1).rolling(20).std()
    df["ret_sq"] = df["return"].shift(1) ** 2
    df["sign1"] = np.sign(df["return"].shift(1))
    df = df.dropna()
    return (
        df.index.to_numpy(),
        df.drop(columns="return").to_numpy(),
        df["return"].to_numpy(),
    )


def base_qr_cache(
    returns,
    dates,
    taus=(0.025, 0.975),
    window=252,
    cal=60,
    n_lags=5,
    learner="lgbm",
    seed=0,
):
    """Fit on the training slice; predict the calibration slice and next observation."""
    pos, X, y = build_features(returns, n_lags=n_lags)
    dates = pd.DatetimeIndex(dates)
    fdates = dates[pos]
    n = len(y)
    w = int(window)
    m = int(cal)
    out = {"date": [], "y": [], "q_cal": [], "y_cal": [], "q_test": []}
    for t in range(w, n):
        Xtr, ytr = (X[t - w : t - m], y[t - w : t - m])
        Xcal, ycal = (X[t - m : t], y[t - m : t])
        qc = np.empty((len(taus), m))
        qt = np.empty(len(taus))
        for j, tau in enumerate(taus):
            mdl = _make_learner(learner, tau, seed)
            mdl.fit(Xtr, ytr)
            qc[j] = mdl.predict(Xcal)
            qt[j] = mdl.predict(X[t].reshape(1, -1))[0]
        out["date"].append(fdates[t])
        out["y"].append(y[t])
        out["q_cal"].append(qc)
        out["y_cal"].append(ycal)
        out["q_test"].append(qt)
    return {
        "date": pd.DatetimeIndex(out["date"]),
        "y": np.array(out["y"]),
        "q_cal": np.array(out["q_cal"]),
        "y_cal": np.array(out["y_cal"]),
        "q_test": np.array(out["q_test"]),
        "taus": np.array(taus),
    }


def _make_learner(kind, tau, seed):
    if kind != "lgbm":
        raise ValueError("The study uses the LightGBM quantile learner.")
    if not HAVE_LGB:
        raise ImportError(
            "Install the packages in requirements.txt before fitting forecasts."
        )
    return lgb.LGBMRegressor(
        objective="quantile", alpha=tau, random_state=seed, verbose=-1, n_jobs=1
    )


class ConformalLevelUnattainable(ValueError):
    pass


def attainable_level(m):
    """Smallest finite split-conformal miscoverage bound under exchangeability."""
    return 1.0 / (m + 1.0)


def _conf_q(scores, level, strict=True):
    """Select the finite-sample conformal order statistic; raise if it is unavailable."""
    m = len(scores)
    k = int(np.ceil((m + 1) * level))
    if k > m:
        if strict:
            raise ConformalLevelUnattainable(
                f"level={level} needs order statistic {k} of m={m}; the split-conformal quantile is +inf here. Smallest attainable miscoverage is 1/(m+1)={attainable_level(m):.6f}. Increase the calibration window."
            )
        return np.inf
    k = max(k, 1)
    return np.sort(scores)[k - 1]


def wrap_tcp(
    cache, alpha=0.05, rm=False, gamma_0=0.01, lam=0.1, beta=0.7, truncate=True
):
    """Apply CQR, zero-truncated CQR (TCP), or TCP with an online offset."""
    n = len(cache["y"])
    lo = np.empty(n)
    hi = np.empty(n)
    Crm = 0.0
    Csplit_tr = np.empty(n)
    Crm_tr = np.empty(n)
    for i in range(n):
        ql_c, qu_c = (cache["q_cal"][i][0], cache["q_cal"][i][1])
        yc = cache["y_cal"][i]
        s = np.maximum(ql_c - yc, yc - qu_c)
        if truncate:
            s = np.maximum(s, 0.0)
        C = _conf_q(s, 1 - alpha)
        Ceff = C + (Crm if rm else 0.0)
        lo[i] = cache["q_test"][i][0] - Ceff
        hi[i] = cache["q_test"][i][1] + Ceff
        Csplit_tr[i] = C
        Crm_tr[i] = Crm if rm else 0.0
        if rm:
            miss = 0.0 if lo[i] <= cache["y"][i] <= hi[i] else 1.0
            g = gamma_0 / (1.0 + lam * (i + 1)) ** beta
            Crm = max(Crm + g * (miss - alpha), -C)
    return (lo, hi, {"C_split": Csplit_tr, "C_rm": Crm_tr})


def wrap_aci(
    cache,
    alpha=0.05,
    eta=0.05,
    reverse_sign=False,
    alpha_min=0.0001,
    alpha_max=0.3,
    truncate=True,
):
    """Apply clipped ACI; reverse_sign selects the diagnostic control."""
    n = len(cache["y"])
    lo = np.empty(n)
    hi = np.empty(n)
    at = np.empty(n)
    a = alpha
    for i in range(n):
        ql_c, qu_c = (cache["q_cal"][i][0], cache["q_cal"][i][1])
        yc = cache["y_cal"][i]
        s = np.maximum(ql_c - yc, yc - qu_c)
        if truncate:
            s = np.maximum(s, 0.0)
        a_eff = float(np.clip(a, alpha_min, alpha_max))
        thr = np.quantile(s, 1.0 - a_eff, method="higher")
        lo[i] = cache["q_test"][i][0] - thr
        hi[i] = cache["q_test"][i][1] + thr
        at[i] = a_eff
        err = 0.0 if lo[i] <= cache["y"][i] <= hi[i] else 1.0
        step = err - alpha if reverse_sign else alpha - err
        a = float(np.clip(a + eta * step, alpha_min, alpha_max))
    return (lo, hi, {"alpha_t": at})


def wrap_cqr_onesided(cache_lo, tau, strict=True):
    """Return lower endpoints and the exchangeable finite-sample reference level."""
    n = len(cache_lo["y"])
    l = np.empty(n)
    m = len(cache_lo["y_cal"][0])
    k = int(np.ceil((m + 1) * (1 - tau)))
    if k > m:
        if strict:
            raise ConformalLevelUnattainable(
                f"one-sided tau={tau} needs order statistic {k} of m={m}; not attainable. Finest attainable miscoverage at m={m} is {attainable_level(m):.6f}."
            )
        return (np.full(n, -np.inf), 0.0)
    eff = 1.0 - k / (m + 1.0)
    for i in range(n):
        s = cache_lo["q_cal"][i][0] - cache_lo["y_cal"][i]
        l[i] = cache_lo["q_test"][i][0] - _conf_q(s, 1 - tau, strict=strict)
    return (l, eff)
