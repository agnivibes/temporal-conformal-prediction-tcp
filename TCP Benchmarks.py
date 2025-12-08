# code1_benchmarks.py
# TCP vs Benchmarks (QR-rolling, GARCH, Historical Simulation) + TCP vs TCP-RM + ACI baseline

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# tqdm with safe fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# Prefer LightGBM; fallback to sklearn
try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM for quantile regression.")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGB = False
    print("LightGBM not found. Falling back to scikit-learn's GradientBoostingRegressor.")

# ==============================
# Helpers
# ==============================

def interval_score(alpha, low, up, y):
    low = np.asarray(low); up = np.asarray(up); y = np.asarray(y)
    over  = np.maximum(0.0, low - y)
    under = np.maximum(0.0, y - up)
    return (up - low) + (2.0/alpha)*over + (2.0/alpha)*under

def adaptation_speed(covered_series, alpha=0.05, window=30, eps=0.01):
    if len(covered_series) < window:
        return np.nan
    roll = pd.Series(covered_series).rolling(window).mean().to_numpy()
    lo, hi = 1 - alpha - eps, 1 - alpha + eps
    inside = (roll >= lo) & (roll <= hi)
    if inside.any():
        depart = np.argmax(~inside)
        if depart == 0 and inside[0]:
            if (~inside).any():
                depart = np.argmax(~inside)
            else:
                return 0
    else:
        depart = 0
    for k in range(depart + 1, len(roll)):
        if lo <= roll[k] <= hi:
            return k - depart
    return np.nan

def build_features(returns, n_lags=5):
    s = pd.Series(returns).astype(float)
    df = pd.DataFrame({'return': s})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['return'].shift(lag)

    # AFTER (no leakage; only info available before t)
    df['vol20']  = df['return'].shift(1).rolling(20).std()
    df['ret_sq'] = df['return'].shift(1) ** 2
    df['sign1']  = np.sign(df['return'].shift(1))
    df = df.dropna()
    X = df.drop(columns='return').values
    y = df['return'].values
    return df, X, y

# ---------- Backtests ----------
from math import log
from scipy.stats import chi2

def kupiec_test(miss, alpha):
    """Unconditional coverage LR_uc."""
    miss = np.asarray(miss)
    miss = miss[np.isfinite(miss)]
    T = miss.size
    if T == 0:
        return np.nan, np.nan, 0, 0
    x = miss.sum()
    if x == 0 or x == T:
        pi_hat = max(min(x / max(T,1), 1 - 1e-8), 1e-8)
    else:
        pi_hat = x / T
    LR_uc = -2 * ( (T - x) * log((1 - alpha)/(1 - pi_hat) + 1e-12) + x * log(alpha/(pi_hat + 1e-12) + 1e-12) )
    p = 1 - chi2.cdf(LR_uc, df=1)
    return LR_uc, p, int(x), int(T)

def christoffersen_independence(miss):
    """Independence LR_ind from 2x2 transition counts."""
    z = np.asarray(miss, dtype=int)
    z = z[np.isfinite(z)]
    if len(z) < 2: return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(z)):
        a, b = z[i-1], z[i]
        if   a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else:                   n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / (n0 + n1)

    import math
    def ll(nxy, px):
        return 0 if nxy == 0 else nxy * math.log(max(px,1e-12))

    L_alt  = ll(n01, p01) + ll(n00, 1 - p01) + ll(n11, p11) + ll(n10, 1 - p11)
    L_null = ll(n01 + n11, p) + ll(n00 + n10, 1 - p)
    LR_ind = -2 * (L_null - L_alt)
    pval = 1 - chi2.cdf(LR_ind, df=1)
    return LR_ind, pval

def christoffersen_conditional(miss, alpha):
    """Conditional coverage LR_cc = LR_uc + LR_ind (df=2)."""
    LR_uc, _, _, _ = kupiec_test(miss, alpha)
    LR_ind, _ = christoffersen_independence(miss)
    if np.isnan(LR_uc) or np.isnan(LR_ind):
        return np.nan, np.nan
    LR_cc = LR_uc + LR_ind
    p = 1 - chi2.cdf(LR_cc, df=2)
    return LR_cc, p

# ==============================
# TCP / TCP-RM
# ==============================

class TemporalConformalPredictor:
    def __init__(self, window_size=252, cal_size=60, alpha=0.05,
                 gamma_0=0.01, lambda_param=0.1, beta=0.7, n_lags=5,
                 model_type=None, rm_update=False):
        assert window_size > cal_size >= 10
        self.window_size = int(window_size)
        self.cal_size = int(cal_size)
        self.alpha = float(alpha)
        self.gamma_0 = float(gamma_0)
        self.lambda_param = float(lambda_param)
        self.beta = float(beta)
        self.n_lags = int(n_lags)
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self.rm_update = bool(rm_update)
        self._init_models()
        self.C_offset = 0.0
        self.cover_errs, self.coverages, self.intervals = [], [], []
        self.trace = {'t': [], 'y': [], 'ql': [], 'qu': [], 'C_split': [], 'C_rm': [], 'C_eff': [],
                      'low': [], 'up': [], 'covered': []}
        self.feature_index = None

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, random_state=0, verbose=-1)
            self.up_model  = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, random_state=0, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, random_state=0)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha/2, random_state=0)

    def _lr(self, t):
        return self.gamma_0 / (1.0 + self.lambda_param * t) ** self.beta

    def get_trace_df(self, index=None):
        df = pd.DataFrame(self.trace)
        if index is not None and len(df) > 0:
            df['date'] = index[df['t'].values]
            df = df[['date'] + [c for c in df.columns if c != 'date']]
        return df

    def fit(self, returns):
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        self.feature_index = feat_df.index
        w, m = self.window_size, self.cal_size
        w_tr = w - m
        start_idx = w
        n = len(y)
        self.cover_errs.clear(); self.coverages.clear(); self.intervals.clear(); self.C_offset = 0.0
        if n <= start_idx:
            self.intervals = np.zeros((0, 2)); self.coverages = np.array([]); return

        for t in range(start_idx, n):
            tr_lo, tr_hi = t - w, t - m
            cal_lo = t - m
            X_tr, y_tr = X[tr_lo:tr_hi], y[tr_lo:tr_hi]
            X_cal, y_cal = X[cal_lo:t], y[cal_lo:t]
            if len(y_tr) < max(20, w_tr // 4):
                continue

            self.low_model.fit(X_tr, y_tr)
            self.up_model.fit(X_tr, y_tr)

            ql_cal = self.low_model.predict(X_cal)
            qu_cal = self.up_model.predict(X_cal)
            s_cal = np.maximum.reduce([ql_cal - y_cal, y_cal - qu_cal, np.zeros_like(y_cal)])

            m_eff = len(s_cal)
            if m_eff == 0:
                continue
            s_sorted = np.sort(s_cal)
            k = int(np.ceil((m_eff + 1) * (1 - self.alpha)))
            k = min(max(k, 1), m_eff)
            C_split = s_sorted[k - 1]

            ql = float(self.low_model.predict(X[t].reshape(1, -1))[0])
            qu = float(self.up_model.predict(X[t].reshape(1, -1))[0])
            C_rm = float(self.C_offset) if self.rm_update else 0.0
            C_eff = C_split + C_rm
            low_i, up_i = ql - C_eff, qu + C_eff
            self.intervals.append([low_i, up_i])

            yt = y[t]
            covered = (yt >= low_i) and (yt <= up_i)
            self.coverages.append(covered)
            err = (0 if covered else 1) - self.alpha
            self.cover_errs.append(err)

            if self.rm_update:
                lr = self._lr(len(self.cover_errs))
                self.C_offset += lr * err
                self.C_offset = max(self.C_offset, -C_split)

            tr = self.trace
            tr['t'].append(t); tr['y'].append(yt); tr['ql'].append(ql); tr['qu'].append(qu)
            tr['C_split'].append(C_split); tr['C_rm'].append(C_rm); tr['C_eff'].append(C_eff)
            tr['low'].append(low_i); tr['up'].append(up_i); tr['covered'].append(covered)

        self.intervals = np.array(self.intervals)
        self.coverages = np.array(self.coverages, dtype=bool)

    def metrics(self):
        if len(self.coverages) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        widths = self.intervals[:, 1] - self.intervals[:, 0]
        return {
            'coverage_rate': float(np.mean(self.coverages)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(self.coverages))
        }

# ==============================
# Baselines (rolling QR, GARCH, Hist, QR-Linear)
# ==============================

class QuantileRegressionRolling:
    """Out-of-sample (walk-forward) Quantile Regression (tree-based)."""
    def __init__(self, alpha=0.05, n_lags=5, train_window=192, model_type=None):
        self.alpha = float(alpha)
        self.n_lags = int(n_lags)
        self.train_window = int(train_window)
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self._init_models()
        self.intervals = None
        self.y_true = None

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, random_state=0, verbose=-1)
            self.up_model  = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, random_state=0, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, random_state=0)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha/2, random_state=0)

    def fit(self, returns):
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        n = len(y); w = self.train_window
        if n <= w:
            self.intervals = np.zeros((0, 2)); self.y_true = np.array([]); return
        iv, y_out = [], []
        for t in range(w, n):
            Xtr, Ytr = X[t-w:t], y[t-w:t]
            self.low_model.fit(Xtr, Ytr)
            self.up_model.fit(Xtr, Ytr)
            ql = float(self.low_model.predict(X[t].reshape(1, -1))[0])
            qu = float(self.up_model.predict(X[t].reshape(1, -1))[0])
            iv.append([ql, qu]); y_out.append(y[t])
        self.intervals = np.array(iv)
        self.y_true = np.array(y_out)

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths  = iv[:, 1] - iv[:, 0]
        return {
            'coverage_rate': float(np.mean(covered)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(covered))
        }

class QuantileRegressionRollingLinear:
    """Optional linear QR baseline using sklearn.linear_model.QuantileRegressor."""
    def __init__(self, alpha=0.05, n_lags=5, train_window=192):
        self.alpha = float(alpha)
        self.n_lags = int(n_lags)
        self.train_window = int(train_window)
        self.low_model = None
        self.up_model = None
        self.intervals = None
        self.y_true = None
        self._ok = self._init_models()

    def _init_models(self):
        try:
            from sklearn.linear_model import QuantileRegressor
            self.low_model = QuantileRegressor(quantile=self.alpha/2, alpha=0.0, solver="highs")
            self.up_model  = QuantileRegressor(quantile=1 - self.alpha/2, alpha=0.0, solver="highs")
            return True
        except Exception as e:
            print(f"(QR-Linear disabled) {e}")
            return False

    def fit(self, returns):
        if not self._ok:
            self.intervals = np.zeros((0, 2)); self.y_true = np.array([]); return
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        n = len(y); w = self.train_window
        if n <= w:
            self.intervals = np.zeros((0, 2)); self.y_true = np.array([]); return
        iv, y_out = [], []
        for t in range(w, n):
            Xtr, Ytr = X[t-w:t], y[t-w:t]
            self.low_model.fit(Xtr, Ytr)
            self.up_model.fit(Xtr, Ytr)
            ql = float(self.low_model.predict(X[t].reshape(1, -1))[0])
            qu = float(self.up_model.predict(X[t].reshape(1, -1))[0])
            iv.append([ql, qu]); y_out.append(y[t])
        self.intervals = np.array(iv); self.y_true = np.array(y_out)

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths  = iv[:, 1] - iv[:, 0]
        return {'coverage_rate': float(np.mean(covered)),
                'avg_interval_width': float(np.mean(widths)),
                'n_predictions': int(len(covered))}

class GARCHModel:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.intervals = None
        self.y_true = None

    def fit(self, returns):
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n < 60:
            self.intervals = np.zeros((0, 2)); self.y_true = np.array([]); return
        vol = np.zeros(n)
        vol[0] = np.std(r[:50]) if n > 50 else max(np.std(r), 1e-6)
        for t in range(1, n):
            vol[t] = np.sqrt(1e-6 + 0.05*r[t-1]**2 + 0.9*vol[t-1]**2)
        z = stats.norm.ppf(1 - self.alpha/2.0)
        self.intervals = np.array([[-z*vol[t], z*vol[t]] for t in range(50, n)])
        self.y_true = r[50:]

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths  = iv[:, 1] - iv[:, 0]
        return {'coverage_rate': float(np.mean(covered)),
                'avg_interval_width': float(np.mean(widths)),
                'n_predictions': int(len(covered))}

class HistoricalSimulation:
    def __init__(self, window_size=252, alpha=0.05):
        self.window_size = int(window_size)
        self.alpha = float(alpha)
        self.intervals = None
        self.y_true = None

    def fit(self, returns):
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n <= self.window_size:
            self.intervals = np.zeros((0, 2)); self.y_true = np.array([]); return
        lows, ups = [], []
        for t in range(self.window_size, n):
            w = r[t-self.window_size:t]
            lows.append(np.percentile(w, 100*self.alpha/2.0))
            ups.append(np.percentile(w, 100*(1 - self.alpha/2.0)))
        self.intervals = np.column_stack([lows, ups])
        self.y_true = r[self.window_size:]

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths  = iv[:, 1] - iv[:, 0]
        return {'coverage_rate': float(np.mean(covered)),
                'avg_interval_width': float(np.mean(widths)),
                'n_predictions': int(len(covered))}

# ==============================
# ACI baseline: adapts target miscoverage alpha_t
# ==============================

class ACIBaseline:
    """
    ACI on top of rolling QR forecasts, using the same (train, cal) split as TCP:
      - Train on [t-w, t-m), calibrate on [t-m, t)
      - Threshold = quantile_{1 - alpha_t}(scores on calibration window)
      - Update alpha_t via Robbins–Monro using current miss
    """
    def __init__(self, window_size=252, cal_size=60, alpha=0.05, eta=0.05, n_lags=5, model_type=None,
                 alpha_min=1e-4, alpha_max=0.30):
        assert window_size > cal_size >= 10
        self.window_size = int(window_size)
        self.cal_size = int(cal_size)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.n_lags = int(n_lags)
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self._init_models()
        self.trace = {'t': [], 'y': [], 'ql': [], 'qu': [], 'alpha_t': [], 'thr': [],
                      'low': [], 'up': [], 'covered': []}
        self.intervals = None
        self.coverages = None
        self.feature_index = None

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=0.5*self.alpha, random_state=0, verbose=-1)
            self.up_model  = lgb.LGBMRegressor(objective='quantile', alpha=1-0.5*self.alpha, random_state=0, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=0.5*self.alpha, random_state=0)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=1-0.5*self.alpha, random_state=0)

    def get_trace_df(self, index=None):
        df = pd.DataFrame(self.trace)
        if index is not None and len(df) > 0:
            df['date'] = index[df['t'].values]
            df = df[['date'] + [c for c in df.columns if c != 'date']]
        return df

    @staticmethod
    def _score(yv, lo, hi):
        return np.maximum(lo - yv, np.maximum(0.0, yv - hi))

    def fit(self, returns):
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        self.feature_index = feat_df.index
        w, m = self.window_size, self.cal_size
        start_idx = w
        n = len(y)
        alpha_t = self.alpha
        intervals = []
        covered_list = []

        if n <= start_idx:
            self.intervals = np.zeros((0, 2)); self.coverages = np.array([]); return

        for t in range(start_idx, n):
            tr_lo, tr_hi = t - w, t - m
            cal_lo = t - m
            X_tr, y_tr = X[tr_lo:tr_hi], y[tr_lo:tr_hi]
            X_cal, y_cal = X[cal_lo:t], y[cal_lo:t]
            if len(y_tr) < max(20, (w-m)//4):
                continue

            self.low_model.fit(X_tr, y_tr)
            self.up_model.fit(X_tr, y_tr)
            ql_cal = self.low_model.predict(X_cal)
            qu_cal = self.up_model.predict(X_cal)

            s = self._score(y_cal, ql_cal, qu_cal)
            if len(s) == 0:
                continue
            thr = np.quantile(s, 1.0 - alpha_t, method="higher")

            ql = float(self.low_model.predict(X[t].reshape(1, -1))[0])
            qu = float(self.up_model.predict(X[t].reshape(1, -1))[0])
            low_i, up_i = ql - thr, qu + thr
            intervals.append([low_i, up_i])

            yt = y[t]
            covered = (yt >= low_i) and (yt <= up_i)
            covered_list.append(covered)

            miss = 1 - int(covered)
            alpha_t = float(np.clip(alpha_t + self.eta * (miss - self.alpha), self.alpha_min, self.alpha_max))

            tr = self.trace
            tr['t'].append(t); tr['y'].append(yt); tr['ql'].append(ql); tr['qu'].append(qu)
            tr['alpha_t'].append(alpha_t); tr['thr'].append(thr)
            tr['low'].append(low_i); tr['up'].append(up_i); tr['covered'].append(covered)

        self.intervals = np.asarray(intervals)
        self.coverages = np.asarray(covered_list, dtype=bool)

    def metrics(self):
        if self.coverages is None or len(self.coverages) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        widths = self.intervals[:, 1] - self.intervals[:, 0]
        return {
            'coverage_rate': float(np.mean(self.coverages)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(self.coverages))
        }

# ==============================
# Sensitivity
# ==============================

def run_sensitivity_analysis(returns_data, series_name, rm_update=False):
    method_tag = 'TCP-RM' if rm_update else 'TCP'
    print(f"\n--- Sensitivity: {series_name} [{method_tag}] ---")
    window_sizes = [100, 252, 500]
    cal_sizes = [40]
    results = []
    if rm_update:
        gammas = [0.005, 0.01, 0.05]
        grid = [(w, c, g) for w in window_sizes for c in cal_sizes for g in gammas]
    else:
        grid = [(w, c, None) for w in window_sizes for c in cal_sizes]
    for (w, c, g) in tqdm(grid, desc=f"Sensitivity ({series_name}, {method_tag})"):
        tcp = TemporalConformalPredictor(window_size=w, cal_size=c,
                                         gamma_0=(g if g is not None else 0.01),
                                         rm_update=rm_update)
        tcp.fit(returns_data)
        met = tcp.metrics()
        row = {'series': series_name, 'method': method_tag,
               'window_size': w, 'cal_size': c,
               'coverage': met['coverage_rate'],
               'avg_interval_width': met['avg_interval_width'],
               'n_predictions': met['n_predictions']}
        if rm_update:
            row['gamma_0'] = g
        results.append(row)
    df = pd.DataFrame(results)
    out = f'sensitivity_results_{series_name}_{method_tag}.csv'
    df.to_csv(out, index=False, float_format='%.4f')
    print(f"Saved {out}")
    return df

# ==============================
# Optional plotting 
# ==============================

def plot_tcp_rm_traces(series, tcp_df, tcrm_df, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(Plot skipped) {e}"); return
    import pandas as pd

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tcrm_df['date'], tcrm_df['y'], lw=0.8, label='Return')
    ax1.plot(tcp_df['date'], tcp_df['low'], lw=0.8, label='TCP low')
    ax1.plot(tcp_df['date'], tcp_df['up'],  lw=0.8, label='TCP up')
    ax1.plot(tcrm_df['date'], tcrm_df['low'], lw=0.8, label='TCP-RM low', alpha=0.9)
    ax1.plot(tcrm_df['date'], tcrm_df['up'],  lw=0.8, label='TCP-RM up',  alpha=0.9)
    ax1.set_title(f'{series}: returns and 95% intervals')
    ax1.legend(loc='upper right', ncol=2, fontsize=9)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    cov_tcp  = pd.Series(tcp_df['covered']).rolling(30).mean()
    cov_tcrm = pd.Series(tcrm_df['covered']).rolling(30).mean()
    ax2.plot(tcp_df['date'], cov_tcp,  lw=1.0, label='TCP roll cov')
    ax2.plot(tcrm_df['date'], cov_tcrm, lw=1.0, label='TCP-RM roll cov')
    ax2.axhline(0.95, ls='--', lw=0.8, color='k'); ax2.set_ylim(0.7, 1.0)
    ax2.set_title('30-day rolling coverage'); ax2.legend(loc='lower right', fontsize=9)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(tcp_df['date'],  tcp_df['C_split'], lw=1.0, label='C_split (TCP)')
    ax3.plot(tcrm_df['date'], tcrm_df['C_split'], lw=1.0, label='C_split (TCP-RM)', alpha=0.8)
    ax3.plot(tcrm_df['date'], tcrm_df['C_rm'],    lw=1.0, label='C_RM (offset)')
    ax3.plot(tcrm_df['date'], tcrm_df['C_eff'],   lw=1.2, label='C_eff = split + RM')
    ax3.set_title('Threshold evolution'); ax3.legend(loc='upper right', ncol=2, fontsize=9)

    plt.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
    print(f"Saved plot: {out_png}")

# ==============================
# Backtest helper (NEW)
# ==============================

def compute_backtests_for_intervals(y_true, L, U, alpha, model_name, series_name):
    if len(y_true) == 0 or len(L) == 0:
        return {'series': series_name, 'model': model_name,
                'LR_uc': np.nan, 'p_uc': np.nan,
                'LR_ind': np.nan, 'p_ind': np.nan,
                'LR_cc': np.nan, 'p_cc': np.nan,
                'exceedances': 0, 'n': 0}
    miss = ((y_true < L) | (y_true > U)).astype(int)
    LR_uc, p_uc, x, T = kupiec_test(miss, alpha)
    LR_ind, p_ind = christoffersen_independence(miss)
    LR_cc, p_cc = christoffersen_conditional(miss, alpha)
    return {'series': series_name, 'model': model_name,
            'LR_uc': LR_uc, 'p_uc': p_uc,
            'LR_ind': LR_ind, 'p_ind': p_ind,
            'LR_cc': LR_cc, 'p_cc': p_cc,
            'exceedances': x, 'n': T}

# ==============================
# Main
# ==============================

if __name__ == "__main__":
    try:
        df = pd.read_csv('financial_returns.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'financial_returns.csv' not found."); raise SystemExit

    df.rename(columns={'^GSPC': 'SP500', 'GC=F': 'Gold'}, inplace=True)
    print("Available series:", df.columns.tolist())

    series_list = ['SP500', 'BTC-USD', 'Gold']
    all_results = []
    backtest_rows = []

    for series in series_list:
        if series not in df.columns:
            print(f"\nWarning: Series '{series}' not found. Skipping."); continue

        print(f"\n{'='*22} Processing: {series} {'='*22}")
        rets = df[series].dropna().values

        print("\nRunning main benchmarks...")
        models = {
            'TCP':       TemporalConformalPredictor(rm_update=False),
            'TCP-RM':    TemporalConformalPredictor(rm_update=True),
            'ACI':       ACIBaseline(window_size=252, cal_size=60, eta=0.05),  # NEW
            'QR':        QuantileRegressionRolling(train_window=252-60),
            'QR-Linear': QuantileRegressionRollingLinear(train_window=252-60),  # NEW optional
            'GARCH':     GARCHModel(),
            'Hist':      HistoricalSimulation()
        }

        # Fit and collect metrics
        for name, model in tqdm(models.items(), desc=f"Benchmarking ({series})"):
            model.fit(rets)
            met = model.metrics(); met['model'] = name; met['series'] = series
            all_results.append(met)

            # Save detailed traces for TCP/TCP-RM/ACI (needed for coverage plots and backtests)
            if name in ('TCP', 'TCP-RM', 'ACI'):
                feat_df, _, _ = build_features(df[series].dropna().values, n_lags=5)
                aligned_dates = df[series].dropna().index[feat_df.index]

                if name in ('TCP', 'TCP-RM'):
                    tr = models[name].get_trace_df(index=aligned_dates)
                    tr['interval_score'] = interval_score(0.05, tr['low'].values, tr['up'].values, tr['y'].values)
                else:  # ACI
                    tr = models[name].get_trace_df(index=aligned_dates)
                    tr['interval_score'] = interval_score(0.05, tr['low'].values, tr['up'].values, tr['y'].values)

                path = f'trace_{series}_{name}.csv'
                tr.to_csv(path, index=False, float_format='%.6f')
                print(f"Saved trace: {path}")

        # Summaries & plots for TCP vs TCP-RM (unchanged)
        try:
            tcp_df  = pd.read_csv(f'trace_{series}_TCP.csv',    parse_dates=['date'])
            tcrm_df = pd.read_csv(f'trace_{series}_TCP-RM.csv', parse_dates=['date'])
        except FileNotFoundError:
            print("Missing traces for TCP/TCP-RM; skipping summary."); 
        else:
            rows = []
            for mname, tr in [('TCP', tcp_df), ('TCP-RM', tcrm_df)]:
                cov = tr['covered'].mean()
                width = (tr['up'] - tr['low']).mean()
                iscore = tr['interval_score'].mean()
                adapt = adaptation_speed(tr['covered'].values, alpha=0.05, window=30, eps=0.01)
                rows.append({'series': series, 'method': mname, 'coverage': cov,
                             'avg_interval_width': width, 'interval_score': iscore,
                             'adaptation_days': adapt})
            pd.DataFrame(rows).to_csv(f'tcp_vs_rm_summary_{series}.csv', index=False, float_format='%.4f')
            print(f"Saved tcp_vs_rm_summary_{series}.csv")
            plot_tcp_rm_traces(series, tcp_df, tcrm_df, out_png=f'tcp_rm_traces_{series}.png')

        # ---------- Backtests for all methods (NEW) ----------
        alpha = 0.05

        # TCP
        try:
            tr = pd.read_csv(f'trace_{series}_TCP.csv', parse_dates=['date'])
            bt = compute_backtests_for_intervals(tr['y'].values, tr['low'].values, tr['up'].values, alpha, 'TCP', series)
            backtest_rows.append(bt)
        except FileNotFoundError:
            pass

        # TCP-RM
        try:
            tr = pd.read_csv(f'trace_{series}_TCP-RM.csv', parse_dates=['date'])
            bt = compute_backtests_for_intervals(tr['y'].values, tr['low'].values, tr['up'].values, alpha, 'TCP-RM', series)
            backtest_rows.append(bt)
        except FileNotFoundError:
            pass

        # ACI
        try:
            tr = pd.read_csv(f'trace_{series}_ACI.csv', parse_dates=['date'])
            bt = compute_backtests_for_intervals(tr['y'].values, tr['low'].values, tr['up'].values, alpha, 'ACI', series)
            backtest_rows.append(bt)
        except FileNotFoundError:
            pass

        # QR (tree)
        if models['QR'].y_true is not None and len(models['QR'].y_true) > 0:
            yq = models['QR'].y_true
            Lq = models['QR'].intervals[:,0]; Uq = models['QR'].intervals[:,1]
            bt = compute_backtests_for_intervals(yq, Lq, Uq, alpha, 'QR', series)
            backtest_rows.append(bt)

        # QR-Linear
        if models['QR-Linear'].y_true is not None and len(models['QR-Linear'].y_true) > 0:
            yq = models['QR-Linear'].y_true
            Lq = models['QR-Linear'].intervals[:,0]; Uq = models['QR-Linear'].intervals[:,1]
            bt = compute_backtests_for_intervals(yq, Lq, Uq, alpha, 'QR-Linear', series)
            backtest_rows.append(bt)

        # GARCH
        if models['GARCH'].y_true is not None and len(models['GARCH'].y_true) > 0:
            yg = models['GARCH'].y_true
            Lg = models['GARCH'].intervals[:,0]; Ug = models['GARCH'].intervals[:,1]
            bt = compute_backtests_for_intervals(yg, Lg, Ug, alpha, 'GARCH', series)
            backtest_rows.append(bt)

        # Historical Simulation
        if models['Hist'].y_true is not None and len(models['Hist'].y_true) > 0:
            yh = models['Hist'].y_true
            Lh = models['Hist'].intervals[:,0]; Uh = models['Hist'].intervals[:,1]
            bt = compute_backtests_for_intervals(yh, Lh, Uh, alpha, 'Hist', series)
            backtest_rows.append(bt)

        # Sensitivity (unchanged)
        run_sensitivity_analysis(rets, series, rm_update=False)
        run_sensitivity_analysis(rets, series, rm_update=True)

        # Save backtests for this series
        if backtest_rows:
            pd.DataFrame(backtest_rows).to_csv(f'backtests_{series}.csv', index=False, float_format='%.6f')
            print(f"Saved backtests_{series}.csv")

    if all_results:
        df_out = pd.DataFrame(all_results)[['series','model','coverage_rate','avg_interval_width','n_predictions']]
        df_out.to_csv('model_results_main.csv', index=False, float_format='%.4f')
        print("\n" + "="*50 + "\nBenchmark results saved to model_results_main.csv")
        print(df_out.round(3))
    else:
        print("\nNo results generated. Check input data and series list.")
