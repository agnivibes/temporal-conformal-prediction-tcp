
# code2_visualizations.py
# Visualizations (TCP, TCP-RM, QR-rolling, GARCH, Hist, ACI)

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# tqdm with safe fallback (no hard dependency)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM for quantile regression.")
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
    USE_LGB = False
    print("LightGBM not found. Falling back to scikit-learn's GradientBoostingRegressor.")

# -----------------------
# Utilities
# -----------------------

def ensure_dirs():
    os.makedirs('figures', exist_ok=True)

def build_features(returns, n_lags=5):
    s = pd.Series(returns).astype(float)
    df = pd.DataFrame({'return': s})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['return'].shift(lag)

    # no leakage; only info available before t
    df['vol20']  = df['return'].shift(1).rolling(20).std()   # r_{t-1},...,r_{t-20}
    df['ret_sq'] = df['return'].shift(1) ** 2                # r_{t-1}^2
    df['sign1']  = np.sign(df['return'].shift(1))

    df = df.dropna()
    X = df.drop(columns='return').values
    y = df['return'].values
    return df, X, y

def window_metrics(returns_series, iv_df, start_date, end_date):
    if iv_df is None or len(iv_df) == 0:
        return np.nan, np.nan, 0
    df = pd.DataFrame({'y': returns_series}).join(iv_df, how='inner')
    win = df.loc[start_date:end_date]
    if len(win) == 0:
        return np.nan, np.nan, 0
    cov = ((win['y'] >= win['lower']) & (win['y'] <= win['upper'])).mean()
    width = (win['upper'] - win['lower']).mean()
    return float(cov), float(width), int(len(win))

# -----------------------
# Models (match benchmarks)
# -----------------------

class TemporalConformalPredictor:
    def __init__(self, window_size=252, cal_size=60, alpha=0.05,
                 gamma_0=0.01, lambda_param=0.1, beta=0.7, n_lags=5,
                 model_type=None, rm_update=False):
        assert window_size > cal_size >= 10
        self.window_size = int(window_size); self.cal_size = int(cal_size)
        self.alpha = float(alpha); self.gamma_0 = float(gamma_0)
        self.lambda_param = float(lambda_param); self.beta = float(beta)
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
            self.low_model = lgb.LGBMRegressor(
                objective='quantile', alpha=self.alpha / 2, random_state=0, verbose=-1
            )
            self.up_model = lgb.LGBMRegressor(
                objective='quantile', alpha=1 - self.alpha / 2, random_state=0, verbose=-1
            )
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
            Xtr, Ytr = X[tr_lo:tr_hi], y[tr_lo:tr_hi]
            Xcal, Ycal = X[cal_lo:t], y[cal_lo:t]
            if len(Ytr) < max(20, w_tr // 4): continue

            self.low_model.fit(Xtr, Ytr)
            self.up_model.fit(Xtr, Ytr)

            ql_cal = self.low_model.predict(Xcal)
            qu_cal = self.up_model.predict(Xcal)
            s_cal = np.maximum.reduce([ql_cal - Ycal, Ycal - qu_cal, np.zeros_like(Ycal)])
            m_eff = len(s_cal)
            if m_eff == 0: continue
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

        self.intervals = np.array(self.intervals); self.coverages = np.array(self.coverages, dtype=bool)

    def intervals_df(self, date_index):
        if len(self.intervals) == 0:
            return pd.DataFrame(columns=['lower','upper'])
        aligned_dates = pd.Index(date_index)[self.feature_index]
        tpos = np.array(self.trace['t'], dtype=int)
        pred_dates = aligned_dates[tpos]
        return pd.DataFrame(self.intervals, columns=['lower','upper'], index=pred_dates)

class QuantileRegressionRolling:
    def __init__(self, alpha=0.05, n_lags=5, train_window=192, model_type=None):
        self.alpha = float(alpha); self.n_lags = int(n_lags)
        self.train_window = int(train_window)
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self._init_models()
        self.intervals = None
        self.feature_index = None
        self.pred_positions = None

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(
                objective='quantile', alpha=self.alpha / 2, random_state=0, verbose=-1
            )
            self.up_model = lgb.LGBMRegressor(
                objective='quantile', alpha=1 - self.alpha / 2, random_state=0, verbose=-1
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, random_state=0)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, random_state=0)
            self.up_model.set_params(alpha=1 - self.alpha/2)

    def fit(self, returns):
        feat_df, X, y = build_features(returns, n_lags=self.n_lags)
        self.feature_index = feat_df.index
        n = len(y); w = self.train_window
        if n <= w:
            self.intervals = np.zeros((0, 2)); self.pred_positions = []
            return
        iv = []
        for t in range(w, n):
            Xtr, Ytr = X[t-w:t], y[t-w:t]
            self.low_model.fit(Xtr, Ytr); self.up_model.fit(Xtr, Ytr)
            ql = float(self.low_model.predict(X[t].reshape(1, -1))[0])
            qu = float(self.up_model.predict(X[t].reshape(1, -1))[0])
            iv.append([ql, qu])
        self.intervals = np.array(iv)
        self.pred_positions = list(range(w, n))

    def intervals_df(self, date_index):
        if self.intervals is None or len(self.intervals) == 0:
            return pd.DataFrame(columns=['lower','upper'])
        aligned = pd.Index(date_index)[self.feature_index]
        pred_dates = aligned[self.pred_positions]
        return pd.DataFrame(self.intervals, columns=['lower','upper'], index=pred_dates)

class GARCHModel:
    def __init__(self, alpha=0.05):
        self.alpha = alpha; self.intervals = None; self.start_idx = 50
    def fit(self, returns):
        r = np.asarray(returns, dtype=float); n = len(r)
        if n < 60: self.intervals = np.zeros((0,2)); return
        vol = np.zeros(n); vol[0] = np.std(r[:50]) if n > 50 else max(np.std(r), 1e-6)
        for t in range(1, n):
            vol[t] = np.sqrt(1e-6 + 0.05*r[t-1]**2 + 0.9*vol[t-1]**2)
        z = stats.norm.ppf(1 - self.alpha/2.0)
        self.intervals = np.array([[-z*vol[t], z*vol[t]] for t in range(self.start_idx, n)])
    def intervals_df(self, date_index):
        if self.intervals is None or len(self.intervals) == 0:
            return pd.DataFrame(columns=['lower','upper'])
        return pd.DataFrame(self.intervals, columns=['lower','upper'], index=pd.Index(date_index)[self.start_idx:])

class HistoricalSimulation:
    def __init__(self, window_size=252, alpha=0.05):
        self.window_size = int(window_size); self.alpha = float(alpha); self.intervals=None
    def fit(self, returns):
        r = np.asarray(returns, dtype=float); n = len(r)
        if n <= self.window_size: self.intervals = np.zeros((0,2)); return
        lows, ups = [], []
        for t in range(self.window_size, n):
            w = r[t-self.window_size:t]
            lows.append(np.percentile(w, 100*self.alpha/2.0))
            ups.append(np.percentile(w, 100*(1 - self.alpha/2.0)))
        self.intervals = np.column_stack([lows, ups])
    def intervals_df(self, date_index):
        if self.intervals is None or len(self.intervals) == 0:
            return pd.DataFrame(columns=['lower','upper'])
        return pd.DataFrame(self.intervals, columns=['lower','upper'], index=pd.Index(date_index)[self.window_size:])

# -----------------------
# ACI baseline (MATCHES BENCHMARKS)
# -----------------------

class ACIBaseline:
    """
    ACI on top of rolling QR forecasts, using the same (train, cal) split as TCP:
      - Train on [t-w, t-m), calibrate on [t-m, t)
      - Threshold = quantile_{1 - alpha_t}(scores on calibration window)
      - Update alpha_t via Robbins–Monro using current miss

    This is the same behavior used in code1_benchmarks.py, with an added
    intervals_df(...) method for visualization.
    """
    def __init__(self, window_size=252, cal_size=60, alpha=0.05, eta=0.05,
                 n_lags=5, model_type=None, alpha_min=1e-4, alpha_max=0.30):
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
            self.low_model = lgb.LGBMRegressor(objective='quantile',
                                               alpha=0.5*self.alpha,
                                               random_state=0,
                                               verbose=-1)
            self.up_model  = lgb.LGBMRegressor(objective='quantile',
                                               alpha=1-0.5*self.alpha,
                                               random_state=0,
                                               verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile',
                                                       alpha=0.5*self.alpha,
                                                       random_state=0)
            self.up_model  = GradientBoostingRegressor(loss='quantile',
                                                       alpha=1-0.5*self.alpha,
                                                       random_state=0)

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
            alpha_t = float(np.clip(alpha_t + self.eta * (miss - self.alpha),
                                    self.alpha_min, self.alpha_max))

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

    def intervals_df(self, date_index):
        """
        New: map intervals to calendar dates (for visualization),
        mirroring TemporalConformalPredictor.intervals_df.
        """
        if self.intervals is None or len(self.intervals) == 0:
            return pd.DataFrame(columns=['lower', 'upper'])
        aligned_dates = pd.Index(date_index)[self.feature_index]
        tpos = np.array(self.trace['t'], dtype=int)
        pred_dates = aligned_dates[tpos]
        return pd.DataFrame(self.intervals, columns=['lower', 'upper'], index=pred_dates)

# -----------------------
# Plotting
# -----------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'figure.dpi': 140,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 12
})

def _panel(ax, title, returns_series, df_iv, start_date, end_date):
    plot_df = pd.DataFrame({'return': returns_series}).join(df_iv, how='inner')
    viz = plot_df.loc[start_date:end_date]
    if len(viz) == 0:
        ax.set_title(f"{title} (no data)"); ax.axhline(0, color='grey', ls='--', lw=0.8); return
    covered = (viz['return'] >= viz['lower']) & (viz['return'] <= viz['upper'])
    misses = viz[~covered]
    ax.fill_between(viz.index, viz['lower'], viz['upper'], alpha=0.35, label='95% Interval')
    ax.plot(viz.index, viz['return'], lw=0.9, color='black', label='Return')
    if len(misses) > 0:
        ax.scatter(misses.index, misses['return'], s=16, color='red', zorder=5)
    cov, width, n = window_metrics(returns_series, df_iv, start_date, end_date)
    ax.set_title(f"{title}\nCOVID window: cov={cov:.3f}, width={width:.3f}, n={n}")
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax.tick_params(axis='x', rotation=45)

def plot_all_models_grid(series_name, returns_series, model_dfs, start_date, end_date, outfile):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    order = [
        ('TCP (Adaptive)', 'TCP'),
        ('TCP-RM (Adaptive+RM)', 'TCP-RM'),
        ('QR (Rolling)', 'QR'),
        ('GARCH (Parametric)', 'GARCH'),
        ('Historical Sim', 'Hist'),
        ('ACI (Adaptive α)', 'ACI'),
    ]
    for ax_idx, (title, key) in enumerate(order):
        _panel(axes[ax_idx], title, returns_series, model_dfs[key], start_date, end_date)
    fig.suptitle(f'{series_name}: Model Intervals During COVID-19 Crash (Feb–Apr 2020)', y=0.98)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
    fig.text(0.5, 0.01, 'Date', ha='center'); fig.text(0.01, 0.5, 'Daily Log Return', va='center', rotation='vertical')
    fig.savefig(outfile, bbox_inches='tight'); plt.close(fig); print(f"Saved: {outfile}")

def plot_tcp_vs_tcrm_overlay(series_name, returns_series, iv_tcp, iv_tcrm, start_date, end_date, outfile):
    plt.figure(figsize=(12, 4.2))
    plot_df = pd.DataFrame({'return': returns_series})
    tcp_df  = plot_df.join(iv_tcp,  how='inner').rename(columns={'lower':'tcp_low','upper':'tcp_up'})
    tcrm_df = plot_df.join(iv_tcrm, how='inner').rename(columns={'lower':'tcrm_low','upper':'tcrm_up'})
    viz_tcp  = tcp_df.loc[start_date:end_date]
    viz_tcrm = tcrm_df.loc[start_date:end_date]
    viz_ret  = plot_df.loc[start_date:end_date]
    if len(viz_tcp) > 0:
        plt.fill_between(viz_tcp.index, viz_tcp['tcp_low'], viz_tcp['tcp_up'], alpha=0.28, label='TCP 95% Interval')
    if len(viz_tcrm) > 0:
        plt.plot(viz_tcrm.index, viz_tcrm['tcrm_low'], lw=1.0, label='TCP-RM low')
        plt.plot(viz_tcrm.index, viz_tcrm['tcrm_up'],  lw=1.0, label='TCP-RM up')
    if len(viz_ret) > 0:
        plt.plot(viz_ret.index, viz_ret['return'], lw=0.9, color='black', label='Return')
    plt.axhline(0, color='grey', ls='--', lw=0.8)
    plt.title(f'{series_name}: TCP vs TCP-RM during COVID (Feb–Apr 2020)')
    plt.legend(loc='upper right', ncol=2, fontsize=9); plt.xticks(rotation=45)
    ax = plt.gca(); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.ylabel('Daily Log Return'); plt.tight_layout(); plt.savefig(outfile, bbox_inches='tight'); plt.close()
    print(f"Saved: {outfile}")

# -----------------------
# Orchestration
# -----------------------

def run_visuals_for_series(series_name, rets, dates, crash_start, crash_end):
    tcp   = TemporalConformalPredictor(rm_update=False)
    tcrm  = TemporalConformalPredictor(rm_update=True)
    qr    = QuantileRegressionRolling(train_window=252-60)
    garch = GARCHModel()
    hist  = HistoricalSimulation()
    # IMPORTANT: ACI baseline now matches code1_benchmarks.py behavior
    aci   = ACIBaseline(window_size=252, cal_size=60, alpha=0.05, eta=0.05)

    print(f"\nFitting models for {series_name}…")
    tcp.fit(rets); tcrm.fit(rets); qr.fit(rets); garch.fit(rets); hist.fit(rets); aci.fit(rets)

    returns_series = pd.Series(rets, index=pd.Index(dates), name='return')
    iv_tcp   = tcp.intervals_df(pd.Index(dates))
    iv_tcrm  = tcrm.intervals_df(pd.Index(dates))
    iv_qr    = qr.intervals_df(pd.Index(dates))
    iv_garch = garch.intervals_df(pd.Index(dates))
    iv_hist  = hist.intervals_df(pd.Index(dates))
    iv_aci   = aci.intervals_df(pd.Index(dates))

    model_dfs = {'TCP': iv_tcp, 'TCP-RM': iv_tcrm, 'QR': iv_qr, 'GARCH': iv_garch, 'Hist': iv_hist, 'ACI': iv_aci}

    if series_name.upper() in ['SP500', '^GSPC']:
        grid_out = 'figures/all_models_visualizationSP.png'; overlay_out = 'figures/tcp_vs_tcrm_SP500.png'
    elif series_name.upper() in ['BTC-USD', 'BTC', 'BTCUSD']:
        grid_out = 'figures/all_models_visualizationBTC.png'; overlay_out = 'figures/tcp_vs_tcrm_BTC-USD.png'
    elif series_name.title() == 'Gold' or series_name.upper() == 'GC=F':
        grid_out = 'figures/all_models_visualizationG.png'; overlay_out = 'figures/tcp_vs_tcrm_Gold.png'
    else:
        grid_out = f'figures/all_models_visualization_{series_name}.png'; overlay_out = f'figures/tcp_vs_tcrm_{series_name}.png'

    plot_all_models_grid(series_name, returns_series, model_dfs, crash_start, crash_end, grid_out)
    plot_tcp_vs_tcrm_overlay(series_name, returns_series, iv_tcp, iv_tcrm, crash_start, crash_end, overlay_out)

    # Save COVID-window metrics to CSV (now includes ACI)
    rows = []
    for key, iv in [('TCP', iv_tcp), ('TCP-RM', iv_tcrm), ('QR(rolling)', iv_qr),
                    ('GARCH', iv_garch), ('Hist', iv_hist), ('ACI', iv_aci)]:
        cov, w, n = window_metrics(returns_series, iv, crash_start, crash_end)
        rows.append({'model': key, 'cov': cov, 'width': w, 'n': n})
    pd.DataFrame(rows).to_csv(f'figures/covid_window_metrics_{series_name}.csv',
                              index=False, float_format='%.4f')

if __name__ == '__main__':
    ensure_dirs()
    try:
        df = pd.read_csv('financial_returns.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        raise SystemExit("Error: 'financial_returns.csv' not found.")
    df.rename(columns={'^GSPC': 'SP500', 'GC=F': 'Gold'}, inplace=True)

    series_order = ['SP500', 'BTC-USD', 'Gold']
    crash_start, crash_end = '2020-02-01', '2020-04-30'

    for series in series_order:
        if series not in df.columns:
            print(f"Warning: Series '{series}' not found. Skipping."); continue
        ser = df[series].dropna()
        run_visuals_for_series(series, ser.values, ser.index, crash_start, crash_end)

    print("\nAll visualizations saved in ./figures/")
