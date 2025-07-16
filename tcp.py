# TCP vs Benchmarks (QR, GARCH, Historical Simulation)


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


# --- 1. Model Classes ---

class TemporalConformalPredictor:
    """
    Implements the Temporal Conformal Prediction (TCP) framework.
    Combines a quantile regressor with an online adaptive calibration layer.
    """

    def __init__(self, window_size=252, alpha=0.05,
                 gamma_0=0.01, lambda_param=0.1, beta=0.7,
                 model_type=None, n_lags=5):
        self.window_size, self.alpha = window_size, alpha
        self.gamma_0, self.lambda_param, self.beta = gamma_0, lambda_param, beta
        self.n_lags = n_lags
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self.con_thr_low = 0.0
        self.con_thr_up = 0.0
        self.cover_errs = []
        self.intervals = []
        self.coverages = []
        self._init_models()

    def _init_models(self):
        """Initializes the underlying quantile regression models."""
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, verbose=-1)
            self.up_model = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, verbose=-1)
        else:  # Fallback to scikit-learn
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha / 2)
            self.up_model = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha / 2)

    def _lr(self, t):
        """Calculates the decaying learning rate for a given time step t."""
        return self.gamma_0 / (1 + self.lambda_param * t) ** self.beta

    def _create_features(self, returns):
        """Constructs the feature matrix from the raw return series."""
        df = pd.DataFrame({'return': returns})
        for lag in range(1, self.n_lags + 1):
            df[f'lag_{lag}'] = df['return'].shift(lag)
        df['vol20'] = df['return'].rolling(20).std()
        df['ret_sq'] = df['return'] ** 2
        df['sign1'] = df['return'].shift(1).apply(np.sign).fillna(0)
        df = df.dropna().reset_index(drop=True)
        return df

    def fit(self, returns):
        """Fits the TCP model in an online, sequential fashion."""
        feature_df = self._create_features(returns)
        X = feature_df.drop(columns='return').values
        y = feature_df['return'].values

        # Start after the initial warmup period
        start_idx = self.window_size

        for t in range(start_idx, len(y)):
            # Define the training window for this step
            X_train, y_train = X[t - self.window_size:t], y[t - self.window_size:t]

            if len(y_train) < 20: continue  # Ensure enough data for vol calculation

            # Retrain the quantile models on the current window
            self.low_model.fit(X_train, y_train)
            self.up_model.fit(X_train, y_train)

            # Predict quantiles for the current time step
            X_curr = X[t].reshape(1, -1)
            low_p = self.low_model.predict(X_curr)[0]
            up_p = self.up_model.predict(X_curr)[0]

            # Apply the conformal correction
            low_i = low_p - self.con_thr_low
            up_i = up_p + self.con_thr_up
            self.intervals.append([low_i, up_i])

            # Check for coverage and calculate error
            covered = (y[t] >= low_i) and (y[t] <= up_i)
            self.coverages.append(covered)

            err = (0 if covered else 1) - self.alpha
            self.cover_errs.append(err)

            # Update conformal thresholds using the modified Robbins-Monro scheme
            lr = self._lr(len(self.cover_errs))
            if y[t] < low_i:  # Lower bound breached
                self.con_thr_low += lr * abs(err)
            elif y[t] > up_i:  # Upper bound breached
                self.con_thr_up += lr * abs(err)
            else:  # No breach, apply small decay to prevent intervals from growing too wide
                self.con_thr_low = max(0, self.con_thr_low - lr * 0.1)
                self.con_thr_up = max(0, self.con_thr_up - lr * 0.1)

        self.intervals = np.array(self.intervals)
        self.coverages = np.array(self.coverages)

    def metrics(self):
        """Calculates and returns key performance metrics."""
        if len(self.coverages) == 0:
            return {'coverage_rate': 0, 'avg_interval_width': 0, 'n_predictions': 0}
        cov_rate = np.mean(self.coverages)
        widths = self.intervals[:, 1] - self.intervals[:, 0]
        return {
            'coverage_rate': cov_rate,
            'avg_interval_width': np.mean(widths),
            'n_predictions': len(self.coverages)
        }


class QuantileRegressionBaseline:
    """Plain ML quantile-regression baseline without conformal adjustment."""

    def __init__(self, alpha=0.05, model_type=None):
        self.alpha = alpha
        self.model_type = model_type or ('lightgbm' if USE_LGB else 'sklearn')
        self._init_models()
        self.intervals = []

    def _init_models(self):
        if self.model_type == 'lightgbm' and USE_LGB:
            self.low_model = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha / 2, verbose=-1)
            self.up_model = lgb.LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.low_model = GradientBoostingRegressor(loss='quantile', alpha=self.alpha / 2)
            self.up_model = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha / 2)

    def _create_features(self, returns):
        df = pd.DataFrame({'return': returns})
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['return'].shift(lag)
        df['vol20'] = df['return'].rolling(20).std()
        df['ret_sq'] = df['return'] ** 2
        df['sign1'] = df['return'].shift(1).apply(np.sign).fillna(0)
        df = df.dropna().reset_index(drop=True)
        return df

    def fit(self, returns):
        feature_df = self._create_features(returns)
        X = feature_df.drop(columns='return').values
        y = feature_df['return'].values

        # Fit on the full sample (static, non-adaptive)
        self.low_model.fit(X, y)
        self.up_model.fit(X, y)

        # Predict intervals for the entire dataset
        low_preds = self.low_model.predict(X)
        up_preds = self.up_model.predict(X)
        self.intervals = np.vstack([low_preds, up_preds]).T
        self.y_true = y

    def metrics(self):
        covered = (self.y_true >= self.intervals[:, 0]) & (self.y_true <= self.intervals[:, 1])
        widths = self.intervals[:, 1] - self.intervals[:, 0]
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(widths),
            'n_predictions': len(covered)
        }


class GARCHModel:
    """Simple GARCH(1,1) model for volatility forecasting."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.intervals = []
        self.y_true = []

    def fit(self, returns):
        r = np.array(returns)
        n = len(r)
        vol = np.zeros(n)

        # Initialize volatility with a 50-day rolling standard deviation
        vol[0] = np.std(r[:50]) if n > 50 else np.std(r)

        # GARCH(1,1) update rule
        for t in range(1, n):
            vol[t] = np.sqrt(1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)

        # Create intervals based on Gaussian assumption
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        self.intervals = np.array([[-z_score * vol[t], z_score * vol[t]] for t in range(50, n)])
        self.y_true = r[50:]

    def metrics(self):
        iv = np.array(self.intervals)
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(iv[:, 1] - iv[:, 0]),
            'n_predictions': len(covered)
        }


class HistoricalSimulation:
    """Non-parametric baseline using rolling empirical quantiles."""

    def __init__(self, window_size=252, alpha=0.05):
        self.window_size, self.alpha = window_size, alpha
        self.intervals = []
        self.y_true = []

    def fit(self, returns):
        r = np.array(returns)
        self.intervals = [
            [np.percentile(r[t - self.window_size:t], 100 * self.alpha / 2),
             np.percentile(r[t - self.window_size:t], 100 * (1 - self.alpha / 2))]
            for t in range(self.window_size, len(r))
        ]
        self.y_true = r[self.window_size:]

    def metrics(self):
        iv = np.array(self.intervals)
        covered = (self.y_true >= iv[:, 0]) & (self.y_true <= iv[:, 1])
        return {
            'coverage_rate': np.mean(covered),
            'avg_interval_width': np.mean(iv[:, 1] - iv[:, 0]),
            'n_predictions': len(covered)
        }


# --- NEW: Function for Sensitivity Analysis ---
def run_sensitivity_analysis(returns_data, series_name):
    """
    Runs TCP with varying hyperparameters to test for robustness.
    This is a critical step for a strong AISTATS submission.
    """
    print(f"\n--- Running Sensitivity Analysis for {series_name} ---")

    # Define hyperparameter grids to test
    window_sizes = [100, 252, 500]
    gammas = [0.005, 0.01, 0.05]

    results = []

    # Create a grid of all parameter combinations
    param_grid = [(w, g) for w in window_sizes for g in gammas]

    # Use tqdm for a nice progress bar during the potentially long run
    for w, g in tqdm(param_grid, desc=f"Sensitivity Test ({series_name})"):
        tcp = TemporalConformalPredictor(window_size=w, gamma_0=g)
        tcp.fit(returns_data)
        metrics = tcp.metrics()

        results.append({
            'series': series_name,
            'window_size': w,
            'gamma_0': g,
            'coverage': metrics['coverage_rate'],
            'avg_interval_width': metrics['avg_interval_width']
        })

    results_df = pd.DataFrame(results)

    # Save results to a CSV 
    filename = f'sensitivity_results_{series_name}.csv'
    results_df.to_csv(filename, index=False, float_format='%.4f')
    print(f"\nSaved sensitivity results for {series_name} to {filename}")

    return results_df


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and prepare data
    try:
        df = pd.read_csv('financial_returns.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'financial_returns.csv' not found. Please ensure the data file is in the correct directory.")
        exit()

    # Rename columns for cleaner labels
    df.rename(columns={
        '^GSPC': 'SP500',
        'GC=F': 'Gold',
    }, inplace=True)

    print("Available series:", df.columns.tolist())

    series_list = ['SP500', 'BTC-USD', 'Gold']
    all_results = []

    # 2. Run main benchmarks and sensitivity analysis for each series
    for series in series_list:
        if series not in df.columns:
            print(f"\nWarning: Series '{series}' not found in the data. Skipping.")
            continue

        print(f"\n{'=' * 20} Processing: {series} {'=' * 20}")
        rets = df[series].dropna().values

        # --- Run Main Benchmarks ---
        print("\nRunning main benchmark models...")
        models = {
            'TCP': TemporalConformalPredictor(),
            'QR': QuantileRegressionBaseline(),
            'GARCH': GARCHModel(),
            'Hist': HistoricalSimulation()
        }

        for name, model in tqdm(models.items(), desc=f"Benchmarking ({series})"):
            model.fit(rets)
            metrics = model.metrics()
            metrics['model'] = name
            metrics['series'] = series
            all_results.append(metrics)

        # --- Run Sensitivity Analysis for TCP ---
        run_sensitivity_analysis(rets, series)

    # 3. Consolidate and save main results
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Reorder columns for clarity
        cols = ['series', 'model', 'coverage_rate', 'avg_interval_width', 'n_predictions']
        results_df = results_df[cols]

        results_df.to_csv('model_results_main.csv', index=False, float_format='%.4f')
        print(f"\n{'=' * 50}\nBenchmark results saved to model_results_main.csv")
        print(results_df.round(3))
    else:
        print("\nNo results were generated. Please check the input data and series list.")

