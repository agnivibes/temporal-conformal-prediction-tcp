#TCP vs Benchmarks (QR, GARCH, Historical Simulation)
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try LightGBM, otherwise fallback to sklearn
try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGB = False

from sklearn.linear_model import QuantileRegressor

# --- 1. Model classes ---

class TemporalConformalPredictor:
    def __init__(self, window_size=252, alpha=0.05,
                 gamma_0=0.01, lambda_param=0.1, beta=0.7,
                 model_type=None, n_lags=5):
        self.window_size, self.alpha = window_size, alpha
        self.gamma_0, self.lambda_param, self.beta = gamma_0, lambda_param, beta
        self.n_lags = n_lags
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self.con_thr_low = 0.0
        self.con_thr_up  = 0.0
        self.cover_errs = []
        self.intervals  = []
        self.coverages  = []
        self._init_models()

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha/2)
            self.up_model  = lgb.LGBMRegressor(objective='quantile', alpha=1-self.alpha/2)
        elif self.model_type == 'sklearn':
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=1-self.alpha/2)
        else:
            self.low_model = QuantileRegressor(quantile=self.alpha/2, solver='highs')
            self.up_model  = QuantileRegressor(quantile=1-self.alpha/2, solver='highs')

    def _lr(self, t):
        return self.gamma_0 / (1 + self.lambda_param * t) ** self.beta

    def fit(self, returns):
        df = pd.DataFrame({'return': returns})
        # feature engineering
        for lag in range(1, self.n_lags+1):
            df[f'lag_{lag}'] = df['return'].shift(lag)
        df['vol20']  = df['return'].rolling(20).std()
        df['ret_sq'] = df['return']**2
        df['sign1']  = df['return'].shift(1).apply(np.sign).fillna(0)
        df = df.dropna()

        X = df.drop(columns='return').values
        y = df['return'].values
        start = max(self.window_size, self.n_lags+20)

        for t in range(start, len(y)):
            X_train, y_train = X[t-self.window_size:t], y[t-self.window_size:t]
            if len(y_train) < 20:
                continue
            self.low_model.fit(X_train, y_train)
            self.up_model.fit(X_train, y_train)
            X_curr = X[t].reshape(1, -1)
            low_p = self.low_model.predict(X_curr)[0]
            up_p  = self.up_model.predict(X_curr)[0]
            low_i = low_p - self.con_thr_low
            up_i  = up_p  + self.con_thr_up
            self.intervals.append([low_i, up_i])
            covered = (y[t] >= low_i) and (y[t] <= up_i)
            err = (0 if covered else 1) - self.alpha
            self.cover_errs.append(err)
            self.coverages.append(covered)
            lr = self._lr(len(self.cover_errs))
            if y[t] < low_i:
                self.con_thr_low += lr * abs(err)
            elif y[t] > up_i:
                self.con_thr_up  += lr * abs(err)
            else:
                self.con_thr_low = max(0, self.con_thr_low - lr*0.1)
                self.con_thr_up  = max(0, self.con_thr_up  - lr*0.1)

        self.intervals = np.array(self.intervals)
        self.coverages = np.array(self.coverages)

    def metrics(self):
        cov_rate = np.mean(self.coverages)
        widths   = self.intervals[:,1] - self.intervals[:,0]
        return {
            'coverage_rate': cov_rate,
            'avg_interval_width': np.mean(widths),
            'n_predictions': len(self.coverages)
        }


class QuantileRegressionBaseline:
    """Plain ML quantile-regression baseline without conformal adjustment."""
    def __init__(self, alpha=0.05, model_type=None):
        self.alpha = alpha
        # choose model
        if model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=alpha/2)
            self.up_model  = lgb.LGBMRegressor(objective='quantile', alpha=1-alpha/2)
        elif model_type == 'sklearn' or not USE_LGB:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=alpha/2)
            self.up_model  = GradientBoostingRegressor(loss='quantile', alpha=1-alpha/2)
        else:
            self.low_model = QuantileRegressor(quantile=alpha/2, solver='highs')
            self.up_model  = QuantileRegressor(quantile=1-alpha/2, solver='highs')
        self.intervals = []

    def fit(self, returns):
        df = pd.DataFrame({'return': returns})
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['return'].shift(lag)
        df['vol20']  = df['return'].rolling(20).std()
        df['ret_sq'] = df['return']**2
        df['sign1']  = df['return'].shift(1).apply(np.sign).fillna(0)
        df = df.dropna()

        X = df.drop(columns='return').values
        y = df['return'].values

        # fit on full sample
        self.low_model.fit(X, y)
        self.up_model.fit(X, y)

        # produce one-step predictions
        for i in range(len(X)):
            x_i = X[i].reshape(1, -1)
            l = self.low_model.predict(x_i)[0]
            u = self.up_model.predict(x_i)[0]
            self.intervals.append([l, u])
        self.intervals = np.array(self.intervals)

    def metrics(self, returns):
        n0 = len(returns) - len(self.intervals)
        r = np.array(returns)[n0:]
        iv = self.intervals
        covered = [(r[i]>=iv[i,0] and r[i]<=iv[i,1]) for i in range(len(iv))]
        widths = iv[:,1] - iv[:,0]
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(widths),
            'n_predictions': len(covered)
        }


class GARCHModel:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, returns):
        r = np.array(returns); n = len(r)
        vol = np.zeros(n); vol[0] = np.std(r[:50]) if n>50 else np.std(r)
        for t in range(1, n):
            vol[t] = np.sqrt(1e-6 + 0.05*r[t-1]**2 + 0.9*vol[t-1]**2)
        self.intervals = [
            [stats.norm.ppf(self.alpha/2, 0, vol[t]),
             stats.norm.ppf(1-self.alpha/2, 0, vol[t])]
            for t in range(50, n)
        ]

    def metrics(self, returns):
        r = np.array(returns)[50:]
        iv = np.array(self.intervals)
        covered = [(r[i]>=iv[i,0] and r[i]<=iv[i,1]) for i in range(len(iv))]
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(iv[:,1] - iv[:,0]),
            'n_predictions': len(covered)
        }


class HistoricalSimulation:
    def __init__(self, window_size=252, alpha=0.05):
        self.window_size, self.alpha = window_size, alpha

    def fit(self, returns):
        r = np.array(returns)
        self.intervals = [
            [np.percentile(r[t-self.window_size:t], 100*self.alpha/2),
             np.percentile(r[t-self.window_size:t], 100*(1-self.alpha/2))]
            for t in range(self.window_size, len(r))
        ]

    def metrics(self, returns):
        r = np.array(returns)[self.window_size:]
        iv = np.array(self.intervals)
        covered = [(r[i]>=iv[i,0] and r[i]<=iv[i,1]) for i in range(len(iv))]
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(iv[:,1] - iv[:,0]),
            'n_predictions': len(covered)
        }


# --- 2. Load, rename, and run multiple series ---
df = pd.read_csv('financial_returns.csv', index_col=0, parse_dates=True)

# rename to friendly labels
df.rename(columns={
    '^GSPC':   'SP500',
    '^FTSE':   'FTSE100',
    '^N225':   'Nikkei225',
    '^GDAXI':  'DAX',
    "('VIX', '^VIX')": 'VIX'
}, inplace=True)

print("Available series:", df.columns.tolist())

series_list = ['SP500', 'BTC-USD', 'GC=F']

for series in series_list:
    rets = df[series].values

    tcp = TemporalConformalPredictor()
    qr  = QuantileRegressionBaseline()
    garch = GARCHModel()
    hist  = HistoricalSimulation()

    tcp.fit(rets)
    qr.fit(rets)
    garch.fit(rets)
    hist.fit(rets)

    results = {
        'TCP':  tcp.metrics(),
        'QR':   qr.metrics(rets),
        'GARCH':garch.metrics(rets),
        'Hist': hist.metrics(rets)
    }

    print(f"\n=== Results for {series} ===")
    print(f"{'Model':<10}{'Coverage':>12}{'Width':>12}{'N':>8}")
    for name, m in results.items():
        print(f"{name:<10}{m['coverage_rate']:12.3f}"
              f"{m['avg_interval_width']:12.3f}{m['n_predictions']:8d}")
records = []
for series in series_list:
    rets = df[series].values

    tcp = TemporalConformalPredictor();    tcp.fit(rets);    mt_tcp = tcp.metrics()
    qr  = QuantileRegressionBaseline();    qr.fit(rets);     mt_qr  = qr.metrics(rets)
    garch = GARCHModel();                  garch.fit(rets);  mt_garch = garch.metrics(rets)
    hist  = HistoricalSimulation();        hist.fit(rets);   mt_hist  = hist.metrics(rets)

    for name, mt in [('TCP',mt_tcp),('QR',mt_qr),('GARCH',mt_garch),('Hist',mt_hist)]:
        records.append({
            'series': series,
            'model': name,
            'coverage': mt['coverage_rate'],
            'avg_interval_width': mt['avg_interval_width'],
            'n_predictions': mt['n_predictions']
        })

# create DataFrame and save
results_df = pd.DataFrame(records)
results_df.to_csv('model_results.csv', index=False)
print("Saved all results to model_results.csv")
