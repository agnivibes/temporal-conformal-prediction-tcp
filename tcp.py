# TCP vs Benchmarks (QR, GARCH, Historical Simulation) + TCP vs TCP-RM comparison
# ------------------------------------------------------------------------------
# Outputs per series:
#   - trace_{SERIES}_TCP.csv
#   - trace_{SERIES}_TCP-RM.csv
#   - tcp_vs_rm_summary_{SERIES}.csv
#   - tcp_rm_traces_{SERIES}.png   (optional if matplotlib available)
#   - sensitivity_results_{SERIES}_{METHOD}.csv
#   - model_results_main.csv
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Try LightGBM for performance, otherwise fallback to sklearn
try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM for quantile regression.")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGB = False
    print("LightGBM not found. Falling back to scikit-learn's GradientBoostingRegressor.")


# ==============================
# Utility Metrics / Helpers
# ==============================

def interval_score(alpha, low, up, y):
    """Winkler-style proper interval score (lower is better)."""
    low = np.asarray(low); up = np.asarray(up); y = np.asarray(y)
    over = np.maximum(0.0, low - y)
    under = np.maximum(0.0, y - up)
    return (up - low) + (2.0 / alpha) * over + (2.0 / alpha) * under

def adaptation_speed(covered_series, alpha=0.05, window=30, eps=0.01):
    """
    Steps until the rolling coverage (window mean) re-enters [1-alpha-eps, 1-alpha+eps]
    after first departure. Returns np.nan if not applicable.
    """
    if len(covered_series) < window:
        return np.nan
    roll = pd.Series(covered_series).rolling(window).mean().to_numpy()
    target_lo, target_hi = 1 - alpha - eps, 1 - alpha + eps

    # Find first departure
    inside = (roll >= target_lo) & (roll <= target_hi)
    if inside.any():
        depart = np.argmax(~inside)  # first False
        if depart == 0 and inside[0]:
            if (~inside).any():
                depart = np.argmax(~inside)
            else:
                return 0
    else:
        depart = 0

    # From departure forward, first return
    for k in range(depart + 1, len(roll)):
        if target_lo <= roll[k] <= target_hi:
            return k - depart
    return np.nan


# ==============================
# Feature Construction (shared)
# ==============================

def build_features(returns, n_lags=5):
    """
    Build lag/vol features on a return series (1D np.array or pd.Series).
    Returns (feature_df, y), both already dropna-aligned.
    """
    s = pd.Series(returns).astype(float)
    df = pd.DataFrame({'return': s})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['return'].shift(lag)
    df['vol20'] = df['return'].rolling(20).std()
    df['ret_sq'] = df['return'] ** 2
    df['sign1'] = np.sign(df['return'].shift(1))
    df = df.dropna()
    X = df.drop(columns='return').values
    y = df['return'].values
    return df, X, y


# ==============================
# TCP / TCP-RM (split-conformal)
# ==============================

class TemporalConformalPredictor:
    """
    Temporal Conformal Prediction with split-conformal threshold (C_split) and
    optional Robbins–Monro online offset (C_rm). Two-sided, symmetric correction.

    Effective threshold: C_eff = C_split + C_rm (C_rm == 0 if rm_update=False).
    """
    def __init__(self,
                 window_size=252,
                 cal_size=60,
                 alpha=0.05,
                 gamma_0=0.01,
                 lambda_param=0.1,
                 beta=0.7,
                 n_lags=5,
                 model_type=None,
                 rm_update=False):
        assert window_size > cal_size >= 10, "Require window_size > cal_size >= 10."
        self.window_size = int(window_size)
        self.cal_size = int(cal_size)
        self.alpha = float(alpha)
        self.gamma_0 = float(gamma_0)
        self.lambda_param = float(lambda_param)
        self.beta = float(beta)
        self.n_lags = int(n_lags)
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self.rm_update = bool(rm_update)

        # models
        self._init_models()

        # state
        self.C_offset = 0.0  # C_rm
        self.cover_errs = []
        a = []  # noqa
        self.coverages = []
        self.intervals = []

        # tracing (filled in fit)
        self.trace = {
            't': [], 'y': [], 'ql': [], 'qu': [],
            'C_split': [], 'C_rm': [], 'C_eff': [],
            'low': [], 'up': [], 'covered': []
        }
        self.feature_index = None  # to map to dates

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, verbose=-1)
            self.up_model = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha / 2)
            self.up_model = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha / 2)

    def _lr(self, t):
        """Decaying learning rate at step t (1-based)."""
        return self.gamma_0 / (1.0 + self.lambda_param * t) ** self.beta

    def get_trace_df(self, index=None):
        df = pd.DataFrame(self.trace)
        if index is not None and len(index) > 0:
            df['date'] = index[df['t'].values]
            cols = ['date'] + [c for c in df.columns if c != 'date']
            df = df[cols]
        return df

    def fit(self, returns):
        """
        Sequential fit over time with rolling window split:
        For each step t:
          - Train on last w_tr points
          - Calibrate on last cal_size points
          - Compute split-conformal threshold C_split
          - Predict for time t (the current feature row)
          - Apply RM offset if enabled
          - Update RM offset with e_t
        """
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        self.feature_index = feat_df.index

        w = self.window_size
        m = self.cal_size
        w_tr = w - m
        assert w_tr > 0

        start_idx = w
        n = len(y)
        if n <= start_idx:
            self.intervals = np.zeros((0, 2))
            self.coverages = np.array([])
            return

        self.cover_errs.clear()
        self.coverages.clear()
        self.intervals.clear()
        self.C_offset = 0.0

        for t in range(start_idx, n):
            # Window slices [t-w, t-1]
            train_lo = t - w
            train_hi = t - m
            cal_lo = t - m
            # cal_hi = t

            X_train = X[train_lo:train_hi]
            y_train = y[train_lo:train_hi]
            X_cal = X[cal_lo:t]
            y_cal = y[cal_lo:t]

            if len(y_train) < max(20, w_tr // 4):
                continue

            # Fit quantile forecasters on train slice
            self.low_model.fit(X_train, y_train)
            self.up_model.fit(X_train, y_train)

            # Predict quantiles on calibration slice
            ql_cal = self.low_model.predict(X_cal)
            qu_cal = self.up_model.predict(X_cal)
            s_cal = np.maximum.reduce([ql_cal - y_cal, y_cal - qu_cal, np.zeros_like(y_cal)])

            # Split-conformal threshold
            m_eff = len(s_cal)
            if m_eff == 0:
                continue
            s_sorted = np.sort(s_cal)
            k = int(np.ceil((m_eff + 1) * (1 - self.alpha)))
            k = min(max(k, 1), m_eff)
            C_split = s_sorted[k - 1]

            # Predict for current t
            X_curr = X[t].reshape(1, -1)
            ql = float(self.low_model.predict(X_curr)[0])
            qu = float(self.up_model.predict(X_curr)[0])

            # Effective threshold
            C_rm = float(self.C_offset) if self.rm_update else 0.0
            C_eff = C_split + C_rm

            low_i = ql - C_eff
            up_i = qu + C_eff
            self.intervals.append([low_i, up_i])

            yt = y[t]
            covered = (yt >= low_i) and (yt <= up_i)
            self.coverages.append(covered)

            # miscoverage indicator - alpha
            err = (0 if covered else 1) - self.alpha
            self.cover_errs.append(err)

            # RM update
            if self.rm_update:
                lr = self._lr(len(self.cover_errs))  # 1-based
                self.C_offset += lr * err
                # keep offset >= -C_split so C_eff >= 0
                self.C_offset = max(self.C_offset, -C_split)

            # Trace
            self.trace['t'].append(t)
            self.trace['y'].append(yt)
            self.trace['ql'].append(ql)
            self.trace['qu'].append(qu)
            self.trace['C_split'].append(C_split)
            self.trace['C_rm'].append(C_rm)
            self.trace['C_eff'].append(C_eff)
            self.trace['low'].append(low_i)
            self.trace['up'].append(up_i)
            self.trace['covered'].append(covered)

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
# Baselines
# ==============================

class QuantileRegressionBaseline:
    """Static quantile regression (no conformal, no adaptation)."""
    def __init__(self, alpha=0.05, n_lags=5, model_type=None):
        self.alpha = alpha
        self.n_lags = n_lags
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self._init_models()
        self.intervals = None
        self.y_true = None

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, verbose=-1)
            self.up_model = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha / 2)
            self.up_model = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha / 2)

    def fit(self, returns):
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        if len(y) < 50:
            self.intervals = np.zeros((0, 2))
            self.y_true = np.array([])
            return
        self.low_model.fit(X, y)
        self.up_model.fit(X, y)
        low_preds = self.low_model.predict(X)
        up_preds = self.up_model.predict(X)
        self.intervals = np.vstack([low_preds, up_preds]).T
        self.y_true = y

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths = iv[:, 1] - iv[:, 0]
        return {
            'coverage_rate': float(np.mean(covered)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(covered))
        }


class GARCHModel:
    """Simple GARCH(1,1)-style volatility path with Gaussian intervals."""
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.intervals = None
        self.y_true = None

    def fit(self, returns):
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n < 60:
            self.intervals = np.zeros((0, 2))
            self.y_true = np.array([])
            return
        vol = np.zeros(n)
        vol[0] = np.std(r[:50]) if n > 50 else max(np.std(r), 1e-6)
        for t in range(1, n):
            vol[t] = np.sqrt(1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
        z = stats.norm.ppf(1 - self.alpha / 2.0)
        self.intervals = np.array([[-z * vol[t], z * vol[t]] for t in range(50, n)])
        self.y_true = r[50:]

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths = iv[:, 1] - iv[:, 0]
        return {
            'coverage_rate': float(np.mean(covered)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(covered))
        }


class HistoricalSimulation:
    """Rolling empirical quantiles (non-parametric)."""
    def __init__(self, window_size=252, alpha=0.05):
        self.window_size = int(window_size)
        self.alpha = float(alpha)
        self.intervals = None
        self.y_true = None

    def fit(self, returns):
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n <= self.window_size:
            self.intervals = np.zeros((0, 2))
            self.y_true = np.array([])
            return
        lows, ups = [], []
        for t in range(self.window_size, n):
            w = r[t - self.window_size: t]
            lows.append(np.percentile(w, 100 * self.alpha / 2.0))
            ups.append(np.percentile(w, 100 * (1 - self.alpha / 2.0)))
        self.intervals = np.column_stack([lows, ups])
        self.y_true = r[self.window_size:]

    def metrics(self):
        if self.y_true is None or len(self.y_true) == 0:
            return {'coverage_rate': 0.0, 'avg_interval_width': 0.0, 'n_predictions': 0}
        iv = self.intervals
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        widths = iv[:, 1] - iv[:, 0]
        return {
            'coverage_rate': float(np.mean(covered)),
            'avg_interval_width': float(np.mean(widths)),
            'n_predictions': int(len(covered))
        }


# ==============================
# Sensitivity Analysis
# ==============================

def run_sensitivity_analysis(returns_data, series_name, rm_update=False):
    """
    Runs TCP or TCP-RM sensitivity.
      - TCP (rm_update=False): sweep window_size only (100, 252, 500)
      - TCP-RM (rm_update=True): sweep window_size x gamma_0
    Saves sensitivity_results_{series}_{method}.csv
    """
    method_tag = 'TCP-RM' if rm_update else 'TCP'
    print(f"\n--- Running Sensitivity Analysis for {series_name} [{method_tag}] ---")

    window_sizes = [100, 252, 500]
    cal_sizes = [40]  # fixed calibration slice to match paper
    results = []

    if rm_update:
        gammas = [0.005, 0.01, 0.05]
        param_grid = [(w, c, g) for w in window_sizes for c in cal_sizes for g in gammas]
    else:
        # TCP: no gamma sweep
        param_grid = [(w, c, None) for w in window_sizes for c in cal_sizes]

    for (w, c, g) in tqdm(param_grid, desc=f"Sensitivity Test ({series_name}, {method_tag})"):
        tcp = TemporalConformalPredictor(window_size=w, cal_size=c,
                                         gamma_0=(g if g is not None else 0.01),
                                         rm_update=rm_update)
        tcp.fit(returns_data)
        met = tcp.metrics()
        row = {
            'series': series_name,
            'method': method_tag,
            'window_size': w,
            'cal_size': c,
            'coverage': met['coverage_rate'],
            'avg_interval_width': met['avg_interval_width'],
            'n_predictions': met['n_predictions']
        }
        if rm_update:
            row['gamma_0'] = g
        results.append(row)

    results_df = pd.DataFrame(results)
    filename = f'sensitivity_results_{series_name}_{method_tag}.csv'
    results_df.to_csv(filename, index=False, float_format='%.4f')
    print(f"Saved sensitivity results for {series_name} [{method_tag}] to {filename}")
    return results_df


# ==============================
# Optional Plotting
# ==============================

def plot_tcp_rm_traces(series, tcp_df, tcrm_df, out_png):
    """
    Creates a 3-panel figure:
      (1) returns + intervals
      (2) 30-day rolling coverage
      (3) thresholds (C_split, C_rm, C_eff)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(Plot skipped: matplotlib not available) {e}")
        return

    fig = plt.figure(figsize=(12, 9))

    # Top: returns + intervals (use TCP-RM's period as reference)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tcrm_df['date'], tcrm_df['y'], lw=0.8, label='Return')
    ax1.plot(tcp_df['date'], tcp_df['low'], lw=0.8, label='TCP low')
    ax1.plot(tcp_df['date'], tcp_df['up'], lw=0.8, label='TCP up')
    ax1.plot(tcrm_df['date'], tcrm_df['low'], lw=0.8, label='TCP-RM low', alpha=0.9)
    ax1.plot(tcrm_df['date'], tcrm_df['up'], lw=0.8, label='TCP-RM up', alpha=0.9)
    ax1.set_title(f'{series}: returns and 95% intervals')
    ax1.legend(loc='upper right', ncol=2, fontsize=9)

    # Middle: 30d rolling coverage
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    cov_tcp = pd.Series(tcp_df['covered']).rolling(30).mean()
    cov_tcrm = pd.Series(tcrm_df['covered']).rolling(30).mean()
    ax2.plot(tcp_df['date'], cov_tcp, lw=1.0, label='TCP roll cov')
    ax2.plot(tcrm_df['date'], cov_tcrm, lw=1.0, label='TCP-RM roll cov')
    ax2.axhline(0.95, ls='--', lw=0.8, color='k')
    ax2.set_ylim(0.7, 1.0)
    ax2.set_title('30-day rolling coverage')
    ax2.legend(loc='lower right', fontsize=9)

    # Bottom: thresholds
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(tcp_df['date'], tcp_df['C_split'], lw=1.0, label='C_split (TCP)')
    ax3.plot(tcrm_df['date'], tcrm_df['C_split'], lw=1.0, label='C_split (TCP-RM)', alpha=0.8)
    ax3.plot(tcrm_df['date'], tcrm_df['C_rm'], lw=1.0, label='C_RM (offset)')
    ax3.plot(tcrm_df['date'], tcrm_df['C_eff'], lw=1.2, label='C_eff = split + RM')
    ax3.set_title('Threshold evolution')
    ax3.legend(loc='upper right', ncol=2, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    # 1) Load data
    try:
        df = pd.read_csv('financial_returns.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'financial_returns.csv' not found. Place it in the working directory.")
        raise SystemExit

    # Standardize column names if needed
    df.rename(columns={'^GSPC': 'SP500', 'GC=F': 'Gold'}, inplace=True)
    print("Available series:", df.columns.tolist())

    series_list = ['SP500', 'BTC-USD', 'Gold']
    all_results = []

    for series in series_list:
        if series not in df.columns:
            print(f"\nWarning: Series '{series}' not found. Skipping.")
            continue

        print(f"\n{'=' * 22} Processing: {series} {'=' * 22}")
        rets = df[series].dropna().values

        # 2) Benchmarks (including TCP and TCP-RM)
        print("\nRunning main benchmark models...")
        models = {
            'TCP': TemporalConformalPredictor(rm_update=False),          # split only
            'TCP-RM': TemporalConformalPredictor(rm_update=True),       # split + RM offset
            'QR': QuantileRegressionBaseline(),
            'GARCH': GARCHModel(),
            'Hist': HistoricalSimulation()
        }

        for name, model in tqdm(models.items(), desc=f"Benchmarking ({series})"):
            model.fit(rets)
            met = model.metrics()
            met['model'] = name
            met['series'] = series
            all_results.append(met)

            # Save traces for TCP / TCP-RM and compute interval scores + adaptation
            if name in ('TCP', 'TCP-RM'):
                feat_df, _, _ = build_features(df[series].dropna().values, n_lags=5)
                feat_index = feat_df.index
                aligned_dates = df[series].dropna().index[feat_index]

                tr = model.get_trace_df(index=aligned_dates)
                tr['interval_score'] = interval_score(0.05, tr['low'].values, tr['up'].values, tr['y'].values)
                trace_path = f'trace_{series}_{name}.csv'
                tr.to_csv(trace_path, index=False, float_format='%.6f')
                print(f"Saved trace: {trace_path}")

        # 3) TCP vs TCP-RM summary table
        try:
            tcp_df = pd.read_csv(f'trace_{series}_TCP.csv', parse_dates=['date'])
            tcrm_df = pd.read_csv(f'trace_{series}_TCP-RM.csv', parse_dates=['date'])
        except FileNotFoundError:
            print("Trace files not found for TCP/TCP-RM; skipping comparison summary for this series.")
            continue

        summary_rows = []
        for mname, tr in [('TCP', tcp_df), ('TCP-RM', tcrm_df)]:
            cov = tr['covered'].mean()
            width = (tr['up'] - tr['low']).mean()
            iscore = tr['interval_score'].mean()
            adapt = adaptation_speed(tr['covered'].values, alpha=0.05, window=30, eps=0.01)
            summary_rows.append({
                'series': series, 'method': mname,
                'coverage': cov, 'avg_interval_width': width,
                'interval_score': iscore, 'adaptation_days': adapt
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = f'tcp_vs_rm_summary_{series}.csv'
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"Saved TCP vs TCP-RM summary: {summary_path}")

        # 4) Optional figure
        plot_tcp_rm_traces(series, tcp_df, tcrm_df, out_png=f'tcp_rm_traces_{series}.png')

        # 5) Sensitivity analysis
        # TCP sensitivity: run for ALL THREE series
        run_sensitivity_analysis(rets, series, rm_update=False)   # TCP (w only)

        # TCP-RM sensitivity: ONLY for SP500 and BTC-USD (as requested)
        run_sensitivity_analysis(rets, series, rm_update=True)    # TCP-RM (w x gamma)

    # 6) Consolidate & save benchmark results
    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = ['series', 'model', 'coverage_rate', 'avg_interval_width', 'n_predictions']
        results_df = results_df[cols]
        results_df.to_csv('model_results_main.csv', index=False, float_format='%.4f')
        print(f"\n{'=' * 50}\nBenchmark results saved to model_results_main.csv")
        print(results_df.round(3))
    else:
        print("\nNo results were generated. Please check the input data and series list.")
