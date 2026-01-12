"""
Macro-Driven Strategic Asset Allocation System - V3
Forward Return Prediction Model (12-Month Horizon)

Target: US Equities / Bonds / Gold
Methodology: Annualized Forward Returns ~ Macro State Features
Estimation: Robust Huber Regression + Stability Selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import t
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')

# NBER Recession Dates (approximate for FRED-MD plotting)
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

TRANSFORMATION_LABELS = {
    1: "Level",
    2: "Δ",
    3: "Δ²",
    4: "log",
    5: "Δlog",
    6: "Δ²log",
    7: "Δpct"
}

# Streamlit Setup moved to main()

# CSS moved to main()


# ============================================================================
# DATA PIPELINE
# ============================================================================

def compute_forward_returns(prices: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    Compute annualized forward returns for each asset.
    """
    log_prices = np.log(prices)
    forward_log_return = log_prices.shift(-horizon_months) - log_prices
    annualized_return = forward_log_return / (horizon_months / 12)
    return annualized_return


def prepare_macro_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform macro variables into current-state features.
    No forward-looking information used.
    """
    features = pd.DataFrame(index=macro_data.index)
    
    for col in macro_data.columns:
        series = macro_data[col]
        
        # Level
        features[f'{col}_level'] = series
        
        # Moving averages
        features[f'{col}_MA12'] = series.rolling(12).mean()
        features[f'{col}_MA60'] = series.rolling(60).mean()
        
        # Z-score vs 5Y average
        ma60 = series.rolling(60).mean()
        std60 = series.rolling(60).std()
        features[f'{col}_zscore'] = (series - ma60) / std60
        
        # Percentile rank (rolling 10Y window)
        features[f'{col}_pctl'] = series.rolling(120).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
        )
        
        # Momentum / slope
        ma12 = series.rolling(12).mean()
        std = series.rolling(60).std()
        features[f'{col}_slope12'] = (ma12 - ma12.shift(12)) / std
        features[f'{col}_slope60'] = (ma60 - ma60.shift(60)) / std
    
    return features.dropna()


# ============================================================================

@st.cache_data(ttl=3600)
def get_series_descriptions(file_path: str = 'FRED-MD_updated_appendix.csv') -> dict:
    """Load series descriptions from appendix."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        # Create mapping from fred ID to description
        mapping = dict(zip(df['fred'], df['description']))
        # Add some manual mappings for derived variables if any
        mapping['SPREAD'] = '10Y Treasury - Fed Funds Spread'
        mapping['BAA_AAA'] = 'Baa - Aaa Corporate Bond Spread'
        mapping['CAPACITY'] = 'Capacity Utilization: Manufacturing'
        return mapping
    except Exception as e:
        st.warning(f"Could not load series descriptions: {e}")
        return {}


@st.cache_data(ttl=3600)
def load_fred_md_data(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data for specified macro variables."""
    try:
        df_raw = pd.read_csv(file_path)
        df = df_raw.iloc[1:].copy()
        df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True).dt.tz_localize(None)
        df = df.set_index('sasdate')
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Core Macro Variables from Spec
        mapping = {
            'PAYEMS': 'PAYEMS',     # Labor
            'UNRATE': 'UNRATE',     # Labor
            'INDPRO': 'INDPRO',     # Output
            'CUMFNS': 'CAPACITY',   # Output
            'CPIAUCSL': 'CPI',      # Prices
            'WPSFD49207': 'PPI',    # Prices
            'PCEPI': 'PCE',         # Prices
            'FEDFUNDS': 'FEDFUNDS', # Rates
            'GS10': 'GS10',         # Rates
            'HOUST': 'HOUST',       # Housing
            'M2SL': 'M2'            # Money
        }
        
        data = pd.DataFrame(index=df.index)
        for fred_col, target_col in mapping.items():
            if fred_col in df.columns:
                data[target_col] = df[fred_col]
        
        # Derived Financial Variables
        if 'GS10' in data.columns and 'FEDFUNDS' in data.columns:
            data['SPREAD'] = data['GS10'] - data['FEDFUNDS']
        
        if 'BAA' in df.columns and 'AAA' in df.columns:
            data['BAA_AAA'] = df['BAA'] - df['AAA']
            
        # Apply log transformations where appropriate (Spec says keep core variables, 
        # but apply transformations in prepare_macro_features. 
        # However, level variables like INDPRO/CPI/etc usually need log-level or log-diff.
        # Spec says {VAR}_level = X_t. I'll stick to logs for index-like variables as per previous logic
        # but the spec doesn't explicitly say log. I'll log transform them here to keep consistency
        # with standard macro modelling when using levels of prices/indices.)
        log_vars = ['PAYEMS', 'INDPRO', 'CPI', 'PPI', 'PCE', 'HOUST', 'M2']
        for col in log_vars:
            if col in data.columns:
                data[col] = np.log(data[col].replace(0, np.nan))
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data
        
    except Exception as e:
        st.error(f"Error loading FRED-MD data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_asset_data(start_date: str = '1960-01-01') -> pd.DataFrame:
    """Load long history asset prices."""
    
    # EQUITIES from FRED-MD (S&P 500)
    equity_prices = pd.Series(dtype=float)
    try:
        df_macro_raw = pd.read_csv('2025-11-MD.csv').iloc[1:]
        df_macro_raw['sasdate'] = pd.to_datetime(df_macro_raw['sasdate'], utc=True).dt.tz_localize(None)
        df_macro_raw.set_index('sasdate', inplace=True)
        
        if 'S&P 500' in df_macro_raw.columns:
            equity_prices = pd.to_numeric(df_macro_raw['S&P 500'], errors='coerce').dropna()
    except Exception as e:
        st.warning(f"Equity data error: {e}")

    # GOLD - using PPI for Gold (WPU1022) as long-term proxy
    gold_prices = pd.Series(dtype=float)
    try:
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        gold_ppi = gold_ppi.resample('MS').last()
        gold_prices = gold_ppi['WPU1022'].dropna()
    except Exception as e:
        st.warning(f"Gold data error: {e}")

    # BONDS - synthetic total return from GS10
    bond_prices = pd.Series(dtype=float)
    try:
        gs10 = web.DataReader('GS10', 'fred', start_date)
        gs10.index = pd.to_datetime(gs10.index, utc=True).tz_localize(None)
        yields = gs10['GS10'] / 100
        duration = 7.5
        # Calculate monthly total returns
        carry = yields.shift(1) / 12
        price_change = -duration * (yields - yields.shift(1))
        synth_ret = carry + price_change
        # Convert returns to index
        bond_prices = 100 * (1 + synth_ret.fillna(0)).cumprod()
    except Exception as e:
        st.warning(f"Bond data error: {e}")

    df_prices = pd.DataFrame({
        'EQUITY': equity_prices,
        'GOLD': gold_prices,
        'BONDS': bond_prices
    }).dropna()
    
    return df_prices


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class Winsorizer:
    """
    Caps features at a specific Z-score threshold.
    Aligned with benchmarking_engine.py
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.means_ = X_df.mean()
        self.stds_ = X_df.std()
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            # Handle possible division by zero or NaN std
            std = self.stds_[col] if self.stds_[col] > 1e-9 else 1e-9
            z_score = (X_df[col] - self.means_[col]) / std
            X_df[col] = X_df[col].mask(z_score > self.threshold, self.means_[col] + self.threshold * std)
            X_df[col] = X_df[col].mask(z_score < -self.threshold, self.means_[col] - self.threshold * std)
        return X_df

# ============================================================================
# ESTIMATION LOGIC
# ============================================================================

def estimate_with_hac(y: pd.Series, X: pd.DataFrame, lag: int = 11) -> dict:
    """
    OLS estimation with Newey-West HAC standard errors.
    """
    # Add constant for OLS
    import statsmodels.api as sm
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_stats': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'resid': model.resid
    }


def estimate_robust(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Robust regression (Huber) that is less sensitive to outliers (like 2008 or 2020).
    Replaces standard OLS for stability.
    """
    from sklearn.linear_model import HuberRegressor
    
    # 1. Robust Scaling (Winsorization)
    # Cap features at roughly 3 standard deviations (1% and 99% quantiles) to prevent explosions
    X_clipped = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    
    # 2. Fit Huber Regressor
    # epsilon=1.35 is standard for 95% efficiency on normal data
    model = HuberRegressor(epsilon=1.35, max_iter=200)
    model.fit(X_clipped, y)
    
    return {
        'model': model,
        'coefficients': pd.Series(model.coef_, index=X.columns),
        'intercept': model.intercept_,
        'fitted_values': model.predict(X_clipped)
    }


@st.cache_data(show_spinner=False)
def select_features_elastic_net(y: pd.Series, X: pd.DataFrame, 
                                 n_iterations: int = 50,
                                 sample_fraction: float = 0.7,
                                 threshold: float = 0.7,
                                 l1_ratio: float = 0.5) -> tuple:
    """
    Implement Stability Selection via bootstrapping.
    """
    from sklearn.linear_model import ElasticNet
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    n_features = X.shape[1]
    selection_counts = pd.Series(0, index=X.columns)
    
    # 1. Bootstrapping Loop
    for _ in range(n_iterations):
        # Subsample indices
        sample_size = int(len(y) * sample_fraction)
        indices = np.random.choice(len(y), size=sample_size, replace=False)
        
        y_sample = y.iloc[indices]
        X_sample = X_scaled.iloc[indices]
        
        # Fit a simple ElasticNet (faster than CV for each bootstrap)
        # We use a small alpha for selection
        model = ElasticNet(alpha=0.01, l1_ratio=l1_ratio, max_iter=5000)
        model.fit(X_sample, y_sample)
        
        # Record non-zero coefficients
        selection_counts[model.coef_ != 0] += 1
        
    # 2. Calculate Probabilities
    selection_probs = selection_counts / n_iterations
    
    # 3. Apply Threshold
    selected = selection_probs[selection_probs > threshold].index.tolist()
    
    # 4. Fallback Logic: ensure at least one feature
    if not selected and not selection_probs.empty:
        selected = [selection_probs.idxmax()]
    
    return selected, selection_probs


def run_walk_forward_backtest(y: pd.Series, X: pd.DataFrame, 
                              min_train_months: int = 240, 
                              horizon_months: int = 12,
                              rebalance_freq: int = 12,
                              asset_class: str = 'EQUITY') -> tuple:
    """
    Recursive walk-forward (expanding window) backtest.
    Returns: (oos_results_df, selection_history_df)
    """
    import statsmodels.api as sm
    from scipy.stats import t
    
    results = []
    selection_history = []
    dates = X.index
    
    start_idx = min_train_months + horizon_months
    
    if start_idx >= len(dates):
        return pd.DataFrame(columns=['predicted_return', 'lower_ci', 'upper_ci']), pd.DataFrame()

    for i in range(start_idx, len(dates), rebalance_freq):
        current_date = dates[i]
        
        train_limit = current_date - pd.DateOffset(months=horizon_months)
        train_mask = X.index <= train_limit
        
        X_train_all = X[train_mask]
        y_train_all = y[train_mask].dropna()
        X_train = X_train_all.loc[y_train_all.index]
        
        if len(y_train_all) < 60:
            continue
            
        # Determine prediction window
        end_window = min(i + rebalance_freq, len(dates))
        predict_idx = dates[i : end_window]
            
        if asset_class == 'EQUITY':
            # EQUITY: Random Forest on ALL features (mirrors benchmark)
            # Benchmark start_idx=240, step=12, horizon=12
            win = Winsorizer(threshold=3.0)
            X_train_clean = win.fit_transform(X_train)
            X_live_clean = win.transform(X.loc[predict_idx])
            
            model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_train_clean, y_train_all)
            
            # Record selection (in RF, we can use importance as proxy)
            importance = pd.Series(model.feature_importances_, index=X.columns)
            selection_history.append({**importance.to_dict(), 'date': current_date})
            
            raw_preds = model.predict(X_live_clean)
            pred_vals = pd.Series(np.clip(raw_preds, -0.30, 0.30), index=predict_idx)
            se = np.std(y_train_all - model.predict(X_train_clean))
            n_features = X_train_clean.shape[1]

        elif asset_class == 'BONDS':
            # BONDS: ElasticNetCV on ALL features (mirrors benchmark)
            win = Winsorizer(threshold=3.0)
            X_train_clean = win.fit_transform(X_train)
            X_live_clean = win.transform(X.loc[predict_idx])
            
            model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=5000)
            model.fit(X_train_clean, y_train_all)
            
            # Record selection
            coefs = pd.Series(model.coef_, index=X.columns)
            selection_history.append({**coefs.to_dict(), 'date': current_date})
            
            raw_preds = model.predict(X_live_clean)
            pred_vals = pd.Series(np.clip(raw_preds, -0.30, 0.30), index=predict_idx)
            se = np.std(y_train_all - model.predict(X_train_clean))
            n_features = X_train_clean.shape[1]

        else: # GOLD
            # GOLD: Simple OLS on ALL features (mirrors benchmark)
            win = Winsorizer(threshold=3.0)
            X_train_clean = win.fit_transform(X_train)
            X_live_clean = win.transform(X.loc[predict_idx])
            
            model = LinearRegression()
            model.fit(X_train_clean, y_train_all)
            
            # Record selection
            coefs = pd.Series(model.coef_, index=X.columns)
            selection_history.append({**coefs.to_dict(), 'date': current_date})
            
            raw_preds = model.predict(X_live_clean)
            pred_vals = pd.Series(np.clip(raw_preds, -0.30, 0.30), index=predict_idx)
            se = np.std(y_train_all - model.predict(X_train_clean))
            n_features = X_train_clean.shape[1]
        
        t_crit = t.ppf(0.95, df=len(y_train_all)-n_features-1) if n_features > 0 else 1.66
        
        for date in predict_idx:
            val = pred_vals.loc[date]
            results.append({
                'date': date,
                'predicted_return': val,
                'lower_ci': val - (t_crit * se),
                'upper_ci': val + (t_crit * se)
            })
            
    if not results:
        return pd.DataFrame(columns=['predicted_return', 'lower_ci', 'upper_ci']), pd.DataFrame()
        
    oos_df = pd.DataFrame(results).set_index('date')
    selection_df = pd.DataFrame(selection_history).set_index('date')
    
    return oos_df, selection_df


@st.cache_data(show_spinner=False)
def cached_walk_forward(y, X, min_train_months=240, horizon_months=12, rebalance_freq=12, asset_class='EQUITY'):
    return run_walk_forward_backtest(y, X, min_train_months, horizon_months, rebalance_freq, asset_class)




def time_series_cv(y: pd.Series, X: pd.DataFrame, n_splits: int = 5):
    """
    Time-series cross-validation for model selection.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=60)  # 60-month gap to avoid leakage
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if len(X_train) < 60: continue
        
        # Fit and evaluate
        model = ElasticNet(l1_ratio=0.5, alpha=0.01)
        # Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        scores.append(score)
    
    return (np.mean(scores), np.std(scores)) if scores else (0.0, 0.0)


@st.cache_data(show_spinner=False)
def stability_analysis(y: pd.Series, X: pd.DataFrame, 
                       horizon_months: int = 12,
                       window_years: int = 25,
                       step_years: int = 5) -> list:
    """
    Rolling window estimation for stability assessment.
    """
    window_months = window_years * 12
    step_months = step_years * 12
    
    results = []
    
    for start in range(0, len(y) - window_months, step_months):
        end = start + window_months
        
        y_window = y.iloc[start:end]
        X_window = X.iloc[start:end]
        
        # Drop NaN (forward returns missing at end)
        valid = y_window.notna()
        y_valid = y_window[valid]
        X_valid = X_window[valid]
        
        if len(y_valid) < 120:  # Minimum 10 years of valid data
            continue
        
        # Estimate using Stability Selection
        selected_features, selection_probs = select_features_elastic_net(y_valid, X_valid)
        
        # Fit OLS on selected features to get coefficients for stability analysis
        if selected_features:
            hac_res = estimate_with_hac(y_valid, X_valid[selected_features], lag=horizon_months//2)
            coef_dict = hac_res['coefficients'].to_dict()
        else:
            coef_dict = {col: 0.0 for col in X.columns}
        
        results.append({
            'start_date': y.index[start],
            'end_date': y.index[end - 1],
            'selected_features': selected_features,
            'coefficients': coef_dict,
            'n_selected': len(selected_features)
        })
    
    return results


def compute_stability_metrics(stability_results: list, feature_names: list) -> pd.DataFrame:
    """
    Compute stability metrics for each feature across estimation windows.
    """
    n_windows = len(stability_results)
    if n_windows == 0:
        return pd.DataFrame(columns=['feature', 'persistence', 'sign_consistency', 'magnitude_stability', 'mean_coefficient'])
        
    metrics = []
    for feat in feature_names:
        coefs = []
        for result in stability_results:
            if feat in result['coefficients']:
                coefs.append(result['coefficients'][feat])
        
        coefs = np.array(coefs)
        non_zero = coefs[coefs != 0]
        
        persistence = len(non_zero) / n_windows
        
        if len(non_zero) > 1:
            sign_consistency = max(
                (non_zero > 0).sum() / len(non_zero),
                (non_zero < 0).sum() / len(non_zero)
            )
            cv = np.std(non_zero) / np.abs(np.mean(non_zero)) if np.mean(non_zero) != 0 else 999
            magnitude_stability = 1 / (1 + cv)
            mean_coef = np.mean(non_zero)
        elif len(non_zero) == 1:
            sign_consistency = 1.0
            magnitude_stability = 1.0
            mean_coef = non_zero[0]
        else:
            sign_consistency = 0.0
            magnitude_stability = 0.0
            mean_coef = 0.0
        
        metrics.append({
            'feature': feat,
            'persistence': persistence,
            'sign_consistency': sign_consistency,
            'magnitude_stability': magnitude_stability,
            'mean_coefficient': mean_coef
        })
    
    return pd.DataFrame(metrics)


# ============================================================================
# SIGNALS & ALLOCATION
# ============================================================================

def compute_expected_returns(macro_features_current: pd.Series,
                              stable_coefficients: pd.Series,
                              intercept: float) -> float:
    """
    Compute expected annualized return for an asset.
    """
    # Align features
    common_features = stable_coefficients.index.intersection(macro_features_current.index)
    
    expected_return = intercept + (
        macro_features_current[common_features] * stable_coefficients[common_features]
    ).sum()
    
    return expected_return


def compute_confidence_interval(expected_return: float,
                                 prediction_std_error: float,
                                 confidence: float = 0.90) -> tuple:
    """
    Compute confidence interval for expected return.
    """
    # Use t-distribution with large df (approximates normal)
    t_crit = t.ppf((1 + confidence) / 2, df=100)
    
    lower = expected_return - t_crit * prediction_std_error
    upper = expected_return + t_crit * prediction_std_error
    
    return lower, upper


def compute_driver_attribution(macro_features_current: pd.Series,
                                stable_coefficients: pd.Series,
                                feature_means: pd.Series) -> pd.DataFrame:
    """
    Attribute expected return to each driver.
    """
    attributions = []
    
    for feat in stable_coefficients.index:
        if feat not in macro_features_current.index:
            continue
        
        current_val = macro_features_current[feat]
        mean_val = feature_means.get(feat, 0)
        coef = stable_coefficients[feat]
        
        # Contribution = coef × (current - mean)
        contribution = coef * (current_val - mean_val)
        
        # Direction
        if contribution > 0.005:
            direction = 'TAILWIND'
        elif contribution < -0.005:
            direction = 'HEADWIND'
        else:
            direction = 'NEUTRAL'
        
        attributions.append({
            'feature': feat,
            'coefficient': coef,
            'current_value': current_val,
            'historical_mean': mean_val,
            'deviation': current_val - mean_val,
            'contribution': contribution,
            'direction': direction
        })
    
    return pd.DataFrame(attributions).sort_values('contribution', key=abs, ascending=False)


def evaluate_regime(macro_data: pd.DataFrame, alert_threshold: float = 2.0) -> tuple:
    """
    Simplified regime detection for risk management.
    """
    recent = macro_data.tail(60)
    indicators = {}
    
    # Credit stress
    if 'BAA_AAA' in recent.columns:
        spread_z = (recent['BAA_AAA'].iloc[-1] - recent['BAA_AAA'].mean()) / recent['BAA_AAA'].std()
        indicators['credit'] = spread_z
    else:
        indicators['credit'] = 0
    
    # Yield curve
    if 'SPREAD' in recent.columns:
        curve_z = -(recent['SPREAD'].iloc[-1] - recent['SPREAD'].mean()) / recent['SPREAD'].std()
        indicators['curve'] = curve_z
    else:
        indicators['curve'] = 0
    
    stress_score = 0.5 * indicators['credit'] + 0.5 * indicators['curve']
    
    if stress_score > alert_threshold:
        status = "ALERT"
    elif stress_score > alert_threshold * 0.6:
        status = "WARNING"
    else:
        status = "CALM"
        
    return status, stress_score, indicators


def compute_allocation(expected_returns: dict,
                       confidence_intervals: dict,
                       regime_status: str,
                       risk_free_rate: float = 0.04) -> dict:
    """
    Compute target allocation based on expected returns.
    """
    # Base weights
    base = {'EQUITY': 0.60, 'BONDS': 0.30, 'GOLD': 0.10}
    min_w = {'EQUITY': 0.20, 'BONDS': 0.20, 'GOLD': 0.05}
    max_w = {'EQUITY': 0.80, 'BONDS': 0.50, 'GOLD': 0.25}
    
    # Regime adjustment
    if regime_status == 'ALERT':
        regime_mult = {'EQUITY': 0.5, 'BONDS': 1.2, 'GOLD': 1.5}
    elif regime_status == 'WARNING':
        regime_mult = {'EQUITY': 0.75, 'BONDS': 1.1, 'GOLD': 1.25}
    else:
        regime_mult = {'EQUITY': 1.0, 'BONDS': 1.0, 'GOLD': 1.0}
    
    # Expected return adjustment
    weights = {}
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        exp_ret = expected_returns.get(asset, 0.05)
        excess_return = exp_ret - risk_free_rate
        
        # Scale: +1% excess return → +5% weight adjustment
        return_adj = 1.0 + (excess_return * 5)
        
        # Combined adjustment
        adj_weight = base[asset] * regime_mult[asset] * return_adj
        weights[asset] = np.clip(adj_weight, min_w[asset], max_w[asset])
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights



@st.cache_data(ttl=3600)
def load_full_fred_md_raw(file_path: str = '2025-11-MD.csv') -> tuple[pd.DataFrame, pd.Series]:
    """Load the complete raw FRED-MD dataset and its transformation codes."""
    try:
        df_raw = pd.read_csv(file_path)
        transform_codes = df_raw.iloc[0]
        df = df_raw.iloc[1:].copy()
        df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True).dt.tz_localize(None)
        df = df.set_index('sasdate')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, transform_codes
    except Exception as e:
        st.error(f"Error loading full FRED-MD raw data: {e}")
        return pd.DataFrame(), pd.Series()


@st.cache_data(ttl=3600)
def load_fred_appendix(file_path: str = 'FRED-MD_updated_appendix.csv') -> pd.DataFrame:
    """Load FRED-MD appendix for series names and groupings."""
    try:
        # Try different encodings for robustness
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if 'fred' in df.columns:
                    # Normalize index to uppercase for robust matching
                    df['fred'] = df['fred'].str.upper()
                    df = df.set_index('fred')
                return df
            except UnicodeDecodeError:
                continue
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading FRED appendix: {e}")
        return pd.DataFrame()


def apply_transformation(series: pd.Series, tcode: int) -> pd.Series:
    """Apply McCracken & Ng (2016) transformations."""
    if tcode == 1:
        return series
    elif tcode == 2:
        return series.diff()
    elif tcode == 3:
        return series.diff().diff()
    elif tcode == 4:
        return np.log(series.replace(0, np.nan)).ffill()
    elif tcode == 5:
        return np.log(series.replace(0, np.nan)).diff().fillna(0)
    elif tcode == 6:
        return np.log(series.replace(0, np.nan)).diff().diff().fillna(0)
    elif tcode == 7:
        return series.pct_change().diff()
    else:
        return series


def get_transformation_label(tcode: int) -> str:
    """Get human-readable label for transformation code."""
    labels = {
        1: "Level (no change)",
        2: "Change: x(t) - x(t-1)",
        3: "Double Change",
        4: "Log: log(x(t))",
        5: "Log Change",
        6: "Double Log Change",
        7: "Change in % Change"
    }
    return labels.get(int(tcode), "Unknown")


def plot_fred_series(data: pd.Series, title: str, subtitle: str, is_transformed: bool = False) -> go.Figure:
    """Create a detailed plot for a FRED-MD series with stats and recession bands."""
    theme = create_theme()
    color = '#ff4757' if is_transformed else '#4da6ff'
    
    # Calculate statistics
    stats = {
        'Mean': data.mean(),
        'Std': data.std(),
        'Q1': data.quantile(0.25),
        'Med': data.median(),
        'Q3': data.quantile(0.75)
    }
    stats_str = ", ".join([f"{k}: {v:.2e}" if abs(v) < 0.01 and v != 0 else f"{k}: {v:.2f}" for k, v in stats.items()])
    
    fig = go.Figure()
    
    # Add Recession Bands
    for start, end in NBER_RECESSIONS:
        if pd.to_datetime(start) <= data.index[-1] and pd.to_datetime(end) >= data.index[0]:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="#ffffff", opacity=0.07,
                layer="below", line_width=0
            )
            
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        line=dict(color=color, width=1.3),
        hovertemplate='%{x|%b %Y}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 10px; color: #888;'>{subtitle}</span><br><span style='font-size: 9px; color: #555;'>{stats_str}</span>",
            font=dict(family='IBM Plex Mono', size=13, color='#e8e8e8'),
            x=0.05, y=0.92
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=75, b=40),
        height=280,
        xaxis=dict(**theme['xaxis']),
        yaxis=dict(**theme['yaxis'])
    )
    
    return fig


def create_theme():
    return {
        'paper_bgcolor': '#0a0a0a',
        'plot_bgcolor': '#111111',
        'font': {'family': 'IBM Plex Mono', 'color': '#888888', 'size': 11},
        'xaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a'},
        'yaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a'}
    }


def plot_allocation(weights: dict) -> go.Figure:
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.65,
        marker=dict(colors=[colors[k] for k in weights.keys()], line=dict(color='#0a0a0a', width=2)),
        textinfo='label+percent',
        textfont=dict(family='IBM Plex Mono', size=11, color='#e8e8e8')
    )])
    
    theme = create_theme()
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        annotations=[dict(text='<b>TARGET</b>', x=0.5, y=0.5,
                         font=dict(family='IBM Plex Mono', size=12, color='#888'), showarrow=False)]
    )
    return fig


def plot_assets(prices: pd.DataFrame) -> go.Figure:
    theme = create_theme()
    normalized = prices / prices.iloc[0] * 100
    
    fig = go.Figure()
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col], name=col,
            mode='lines', line=dict(color=colors.get(col, '#888'), width=1.5)
        ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=280,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', title='Indexed (100)')
    )
    return fig


def plot_ect(ect: pd.Series) -> go.Figure:
    theme = create_theme()
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#2a2a2a")
    
    std = ect.std()
    fig.add_hrect(y0=-std, y1=std, fillcolor="rgba(77, 166, 255, 0.05)", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=ect.index, y=ect.values, mode='lines',
        line=dict(color='#4da6ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(77, 166, 255, 0.1)'
    ))
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=10, b=40),
        height=150,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', title='ECT')
    )
    return fig


def plot_feature_heatmap(selection_history: pd.DataFrame, descriptions: dict) -> go.Figure:
    """
    Visualize stability selection probabilities over time.
    """
    if selection_history.empty:
        return go.Figure()
        
    theme = create_theme()
    
    # Transpose so variables are on Y-axis and time on X-axis
    df_plot = selection_history.T
    
    # Sort Y-axis by average probability to put most stable on top
    avg_probs = df_plot.mean(axis=1).sort_values(ascending=True)
    df_plot = df_plot.loc[avg_probs.index]
    
    # Create labels with descriptions
    y_labels = [f"{col} ({descriptions.get(col.split('_')[0], 'Macro Variable')})" for col in df_plot.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=df_plot.values,
        x=df_plot.columns,
        y=y_labels,
        colorscale=[[0, '#0a0a0a'], [0.5, 'rgba(77, 166, 255, 0.2)'], [1, '#4da6ff']],
        showscale=True,
        colorbar=dict(title=dict(text='Selection Prob', side='right'), thickness=15, len=0.7)
    ))
    
    fig.update_layout(
        title=dict(text='STABILITY SELECTION: FEATURE PERSISTENCE OVER TIME', font=dict(size=12, color='#888')),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        xaxis=dict(title='Estimation Date', gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', tickfont=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=40),
        height=500
    )
    
    return fig


def plot_stability_boxplot(results: dict, asset: str, descriptions: dict = None) -> go.Figure:
    """Show coefficient distribution across windows."""
    theme = create_theme()
    
    if asset not in results or 'all_coefficients' not in results[asset]:
        return go.Figure()
    
    coef_df = results[asset]['all_coefficients']
    stable_feats = results[asset].get('stable_features', [])[:8]
    
    if not stable_feats:
        return go.Figure()
    
    fig = go.Figure()
    colors = ['#ff6b35', '#4da6ff', '#ffd700', '#00d26a', '#ff4757', '#9b59b6', '#1abc9c', '#e74c3c']
    
    for i, feat in enumerate(stable_feats):
        if feat in coef_df.columns:
            base_var = feat.split('_')[0]
            desc = descriptions.get(base_var, base_var) if descriptions else base_var
            
            fig.add_trace(go.Box(
                y=coef_df[feat], 
                name=feat[:12],
                marker_color=colors[i % len(colors)],
                boxmean=True,
                hovertext=desc,
                hoverinfo='y+text+name'
            ))
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        showlegend=False,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#1a1a1a', title='Coefficient')
    )
    return fig


def plot_trend_bars(trend_df: pd.DataFrame, variables: list, descriptions: dict = None) -> go.Figure:
    """Plot 5Y trend comparison."""
    theme = create_theme()
    
    filtered = trend_df[trend_df['Variable'].isin(variables)]
    if filtered.empty:
        return go.Figure()
    
    colors = ['#00d26a' if x > 0.05 else '#ff4757' if x < -0.05 else '#888888' 
              for x in filtered['Slope_5Y']]
    
    hovers = []
    for var in filtered['Variable']:
        desc = descriptions.get(var, var) if descriptions else var
        hovers.append(desc)
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered['Variable'],
        y=filtered['Slope_5Y'],
        marker_color=colors,
        text=[f"{x:.2f}" for x in filtered['Slope_5Y']],
        textposition='outside',
        hovertext=hovers,
        hoverinfo='y+text+x'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#2a2a2a")
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='#1a1a1a', title='5Y Slope (Annualized)')
    )
    return fig


def plot_driver_vs_asset(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, 
                         feat_name: str, asset: str, descriptions: dict = None) -> go.Figure:
    """Plot dual-axis comparison of macro driver and forward asset return using raw values."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
        
    # Align data
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
        
    macro_vals = combined[feat_name]
    asset_vals = combined[asset]
    
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    title_text = f"{feat_name} ({desc}) vs {asset} Forward Return"
    
    fig = go.Figure()
    
    # Macro series (left axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=macro_vals, name=feat_name,
        mode='lines', line=dict(color='#00d26a', width=1.5),
        hovertemplate="<b>" + feat_name + "</b>: %{y:.4f}<extra></extra>"
    ))
    
    # Asset returns (right axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=asset_vals, name=asset,
        mode='lines', line=dict(color='#4da6ff', width=1.5),
        yaxis='y2',
        hovertemplate="<b>" + asset + " Return</b>: %{y:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title_text, 
                  font=dict(family='IBM Plex Mono', size=11, color='#888')),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=50, t=40, b=40),
        height=350,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#1a1a1a',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dash',
            spikethickness=1,
            spikecolor='#555'
        ),
        yaxis=dict(gridcolor='#1a1a1a', side='left', title=f'Macro: {feat_name}'),
        yaxis2=dict(gridcolor='#1a1a1a', overlaying='y', side='right', title='Forward Return (Annualized)', tickformat='.0%')
    )
    return fig


def plot_driver_scatter(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, 
                        feat_name: str, asset: str, descriptions: dict = None) -> go.Figure:
    """Scatter plot of Driver vs Asset Return, colored by decade."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
        
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
        
    combined['Decade'] = (combined.index.year // 10 * 10).astype(str) + "s"
    
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    
    fig = px.scatter(
        combined, x=feat_name, y=asset, color='Decade',
        trendline="ols",
        title=f"Correlation Density: {feat_name} ({desc}) vs {asset}",
        labels={feat_name: f"{feat_name}", asset: f"{asset} Fwd Return"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        xaxis=dict(gridcolor='#1a1a1a', title=feat_name),
        yaxis=dict(gridcolor='#1a1a1a', title=f"{asset} Return", tickformat='.1%')
    )
    return fig


def plot_rolling_correlation(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, 
                             feat_name: str, asset: str, window: int = 60) -> go.Figure:
    """Plot 60-month rolling correlation between driver and asset."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
        
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if len(combined) < window:
        return go.Figure()
        
    rolling_corr = combined[feat_name].rolling(window).corr(combined[asset])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr,
        mode='lines', line=dict(color='#ff6b35', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)',
        name='Rolling Correlation'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#444")
    
    fig.update_layout(
        title=f"Rolling {window}M Correlation: {feat_name} vs {asset}",
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', title='Correlation', range=[-1, 1])
    )
    return fig


def plot_quintile_analysis(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, 
                           feat_name: str, asset: str, horizon_months: int = 12) -> go.Figure:
    """Group asset returns into quintiles based on driver values."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
        
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
        
    combined['Quintile'] = pd.qcut(combined[feat_name], 5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
    quintile_avg = combined.groupby('Quintile')[asset].mean().reset_index()
    
    colors = ['#ff4757', '#ffa502', '#ced6e0', '#2ed573', '#1e90ff']
    
    fig = px.bar(
        quintile_avg, x='Quintile', y=asset,
        title=f"Quintile Analysis: {asset} {horizon_months}M Return by {feat_name} Bucket",
        color='Quintile',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=350,
        showlegend=False,
        xaxis=dict(gridcolor='#1a1a1a', title=f"{feat_name} Quintiles"),
        yaxis=dict(gridcolor='#1a1a1a', title=f'Avg {horizon_months}M Forward Return', tickformat='.1%')
    )
    return fig


def plot_combined_driver_analysis(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, 
                                  feat_name: str, asset: str, descriptions: dict = None, 
                                  window: int = 60, horizon_months: int = 12) -> go.Figure:
    """Combined chart with shared X-axis: Top (Driver vs Asset), Bottom (Rolling Correlation)."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
        
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
        
    macro_vals = combined[feat_name]
    asset_vals = combined[asset]
    rolling_corr = macro_vals.rolling(window).corr(asset_vals)
    
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # 1. Top Panel: Dual Axis Comparison
    # Macro series (left axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=macro_vals, name=feat_name,
        mode='lines', line=dict(color='#00d26a', width=1.5),
        hovertemplate="<b>" + feat_name + "</b>: %{y:.4f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)
    
    # Asset returns (right axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=asset_vals, name=asset,
        mode='lines', line=dict(color='#4da6ff', width=1.5),
        hovertemplate="<b>" + asset + f" {horizon_months}M Return</b>: %{{y:.2%}}<extra></extra>"
    ), row=1, col=1, secondary_y=True)
    
    # 2. Bottom Panel: Rolling Correlation
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr,
        mode='lines', line=dict(color='#ff6b35', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)',
        name=f'{window}M Rolling Correlation',
        hovertemplate="<b>Correlation</b>: %{y:.2f}<extra></extra>"
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#444", row=2, col=1)
    
    # Layout updates
    fig.update_layout(
        title=dict(text=f"{feat_name} ({desc}) Analysis", 
                  font=dict(family='IBM Plex Mono', size=12, color='#888')),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=50, t=60, b=40),
        height=550,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified'
    )
    
    # Axis styling
    fig.update_xaxes(gridcolor='#1a1a1a', row=1, col=1)
    fig.update_xaxes(gridcolor='#1a1a1a', row=2, col=1)
    fig.update_yaxes(title_text=f"Macro: {feat_name}", gridcolor='#1a1a1a', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{horizon_months}M Fwd Return", gridcolor='#1a1a1a', tickformat='.0%', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=f"{window}M Correlation", gridcolor='#1a1a1a', range=[-1, 1], row=2, col=1)
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def plot_variable_survival(stability_results_map: dict, asset: str, descriptions: dict = None) -> go.Figure:
    """
    Charts how often each variable was selected in the Walk-Forward/Rolling tests.
    This proves which variables are "Real" and which were just noise.
    """
    theme = create_theme()
    
    if asset not in stability_results_map:
        return go.Figure().update_layout(title="No stability data available", **theme)
    
    # Extract coefficients history
    coef_df = stability_results_map[asset].get('all_coefficients', pd.DataFrame())
    
    if coef_df.empty:
        return go.Figure().update_layout(title="No selection history available", **theme)
    
    # Count non-zero occurrences for each feature
    # We ignore 'const' if present
    feature_cols = [c for c in coef_df.columns if c != 'const']
    counts = (coef_df[feature_cols].fillna(0) != 0).sum().sort_values(ascending=True)
    
    # Filter only those that survived at least once
    counts = counts[counts > 0]
    
    if counts.empty:
        return go.Figure().update_layout(title="No drivers survived the stability test", **theme)
    
    # Prepare labels with descriptions if available
    labels = []
    for feat in counts.index:
        desc = descriptions.get(feat.split('_')[0], feat) if descriptions else feat
        labels.append(f"<b>{feat}</b><br><span style='font-size:9px; color:#666;'>{desc}</span>")
    
    # Create the chart
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=labels,
        orientation='h',
        marker=dict(
            color=counts.values,
            colorscale='Oranges',
            line=dict(color='#0a0a0a', width=1)
        ),
        text=counts.values,
        textposition='auto',
    ))
    
    # Create the chart with combined layout settings
    layout_args = theme.copy()
    layout_args.update({
        'title': dict(
            text=f"VARIABLE SURVIVAL LEADERBOARD - {asset}",
            font=dict(size=14, color='#ff6b35')
        ),
        'xaxis_title': "Number of Windows Selected",
        'margin': dict(l=20, r=20, t=60, b=40),
        'height': 400 + (len(counts) * 15),
    })
    
    # Update yaxis from theme with showgrid=False
    if 'yaxis' in layout_args:
        layout_args['yaxis'] = {**layout_args['yaxis'], 'showgrid': False}
    else:
        layout_args['yaxis'] = dict(showgrid=False)
        
    fig.update_layout(**layout_args)
    
    return fig


@st.cache_data(show_spinner=False)
def run_collective_analysis(y, X, l1_ratio, min_persistence, estimation_window_years, horizon_months):
    """
    Runs the full analysis for all assets in one go and caches the results.
    This prevents mid-rerun layout shifts that can reset st.tabs.
    """
    expected_returns = {}
    confidence_intervals = {}
    driver_attributions = {}
    stability_results_map = {}
    model_stats = {}
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        y_asset = y[asset]
        
        # Stability Analysis (Rolling Windows)
        stab_results = stability_analysis(y_asset, X, horizon_months=horizon_months, window_years=estimation_window_years)
        metrics = compute_stability_metrics(stab_results, X.columns)
        
        # Stability Selection (Full Sample Bootstrapping)
        stable_features, selection_probs = select_features_elastic_net(
            y_asset.dropna(), 
            X.loc[y_asset.dropna().index],
            threshold=min_persistence,
            l1_ratio=l1_ratio
        )
        
        # Final Asset-Specific Estimation
        y_valid = y_asset.dropna()
        X_valid = X.loc[y_valid.index]
        
        # RESTRICT TO STABLE FEATURES ONLY
        if stable_features:
            X_valid = X_valid[stable_features]
            X_current = X.tail(1)[stable_features]
        else:
            # Fallback to all if somehow empty (though selection logic avoids this)
            X_current = X.tail(1)

        win = Winsorizer(threshold=3.0)
        X_valid_clean = win.fit_transform(X_valid)
        X_current_clean = win.transform(X_current)
        
        if asset == 'EQUITY':
            model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_valid_clean, y_valid)
            exp_ret = model.predict(X_current_clean)[0]
            prediction_se = np.std(y_valid - model.predict(X_valid_clean))
            hac_results = {
                'model': model,
                'coefficients': pd.Series(0, index=X_valid.columns.tolist() + ['const']),
                'intercept': 0,
                'importance': pd.Series(model.feature_importances_, index=X_valid.columns)
            }
            beta = hac_results['importance']
            selected_features = X_valid.columns.tolist()
        elif asset == 'BONDS':
            model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=5000)
            model.fit(X_valid_clean, y_valid)
            exp_ret = model.predict(X_current_clean)[0]
            prediction_se = np.std(y_valid - model.predict(X_valid_clean))
            hac_results = {
                'model': model,
                'coefficients': pd.Series(model.coef_, index=X_valid.columns),
                'intercept': model.intercept_
            }
            beta = pd.Series(model.coef_, index=X_valid.columns)
            selected_features = beta[beta != 0].index.tolist()
        else: # GOLD
            model = LinearRegression()
            model.fit(X_valid_clean, y_valid)
            exp_ret = model.predict(X_current_clean)[0]
            prediction_se = np.std(y_valid - model.predict(X_valid_clean))
            hac_results = {
                'model': model,
                'coefficients': pd.Series(model.coef_, index=X_valid.columns),
                'intercept': model.intercept_
            }
            beta = pd.Series(model.coef_, index=X_valid.columns)
            selected_features = beta[beta != 0].index.tolist()

        # CI calculation
        t_stat = 1.645 # 90% approx
        expected_returns[asset] = exp_ret
        confidence_intervals[asset] = [exp_ret - t_stat * prediction_se, exp_ret + t_stat * prediction_se]
        
        # Attribution
        attr_data = []
        for feat in selected_features:
            val = X_current_clean[feat].iloc[0]
            coef = beta[feat]
            impact = val * coef
            attr_data.append({
                'feature': feat,
                'Impact': impact,
                'State': val,
                'Weight': coef,
                'Signal': 'BULLISH' if impact > 0.005 else 'BEARISH' if impact < -0.005 else 'NEUTRAL',
                'Link': metrics[feat]['correlation'] if feat in metrics else 0
            })
        driver_attributions[asset] = pd.DataFrame(attr_data).sort_values('Impact', ascending=False)
        stability_results_map[asset] = {
            'metrics': metrics,
            'stable_features': stable_features,
            'hac_results': hac_results,
            'all_coefficients': pd.DataFrame([res['coefficients'] for res in stab_results])
        }
        model_stats[asset] = hac_results

    return expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats


def plot_backtest(actual_returns: pd.Series, 
                  predicted_returns: pd.Series,
                  confidence_lower: pd.Series,
                  confidence_upper: pd.Series,
                  confidence_level: float = 0.90) -> go.Figure:
    """
    Plot predicted vs actual forward returns with bottom-anchored minimal hover labels.
    """
    fig = go.Figure()
    
    # Calculate CI margin for hover
    ci_margin = (confidence_upper - confidence_lower) / 2
    
    # 1. VISIBLE TRACES (Hover Disabled)
    # Confidence band
    fig.add_trace(go.Scatter(
        x=predicted_returns.index,
        y=confidence_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=predicted_returns.index,
        y=confidence_lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(77, 166, 255, 0.2)',
        name=f'{int(confidence_level*100)}% CI',
        hoverinfo='skip'
    ))
    
    # Predicted
    fig.add_trace(go.Scatter(
        x=predicted_returns.index,
        y=predicted_returns,
        mode='lines',
        line=dict(color='#4da6ff', width=2),
        name='Predicted',
        hoverinfo='skip'
    ))
    
    # Actual
    fig.add_trace(go.Scatter(
        x=actual_returns.index,
        y=actual_returns,
        mode='lines',
        line=dict(color='#ff6b35', width=2),
        name='Actual',
        hoverinfo='skip'
    ))
    
    # 2. GHOST HOVER TRACES (Anchored at Bottom)
    # Using a secondary hidden y-axis [0, 1] to pin labels to the bottom (y=0.05)
    hover_y = 0.05
    
    # Pred Hover
    fig.add_trace(go.Scatter(
        x=predicted_returns.index,
        y=[hover_y] * len(predicted_returns),
        yaxis='y2',
        name='Pred',
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hovertemplate="<b>Pred</b>: %{customdata:.1%}<extra></extra>",
        customdata=predicted_returns
    ))
    
    # Act Hover
    fig.add_trace(go.Scatter(
        x=actual_returns.index,
        y=[hover_y] * len(actual_returns),
        yaxis='y2',
        name='Act',
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hovertemplate="<b>Act</b>: %{customdata:.1%}<extra></extra>",
        customdata=actual_returns
    ))
    
    theme = create_theme()
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=350,
        hovermode='x',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.6)',
            font=dict(family='IBM Plex Mono', size=11)
        ),
        xaxis=dict(
            gridcolor='#1a1a1a',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dash',
            spikethickness=1,
            spikecolor='#555'
        ),
        yaxis=dict(gridcolor='#1a1a1a', title='Annualized Return'),
        yaxis2=dict(
            range=[0, 1],
            overlaying='y',
            visible=False,
            fixedrange=True
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
    )
    
    return fig


def construct_model_summary(asset: str, model_stats: dict) -> str:
    """
    Constructs a readable summary of the model (Equation or Feature Importance).
    """
    if asset not in model_stats:
        return "Model details not available."
    
    m_info = model_stats[asset]
    
    if asset == 'EQUITY':
        # Random Forest: Show Feature Importance
        importance = m_info.get('importance', pd.Series())
        if importance.empty:
            return "Non-linear Ensemble (Random Forest). Variable sensitivities are dynamic."
        
        top_5 = importance.sort_values(ascending=False).head(5)
        summary = "**Architecture: Random Forest (Non-Linear Ensemble)**\n\n"
        summary += "Top Predictive Drivers (Feature Importance):\n"
        for feat, imp in top_5.items():
            summary += f"- {feat}: `{imp:.4f}`\n"
        return summary
    
    else:
        # Linear Models (BONDS, GOLD): Show Equation
        intercept = m_info.get('intercept', 0)
        coefs = m_info.get('coefficients', pd.Series())
        
        if coefs.empty:
            return "Linear Model. Coefficients not available."
        
        # Filter significant coefficients (|coef| > 1e-6)
        sig_coefs = coefs[coefs.abs() > 1e-6]
        if 'const' in sig_coefs:
            sig_coefs = sig_coefs.drop('const')
            
        arch_name = "ElasticNet (Regularized Linear)" if asset == 'BONDS' else "Simple OLS (Linear Regression)"
        equation = f"**Architecture: {arch_name}**\n\n"
        equation += f"Predicted Return = `{intercept:.4f}`"
        
        for feat, val in sig_coefs.items():
            sign = "+" if val >= 0 else "-"
            equation += f" {sign} (`{abs(val):.4f}` * {feat})"
            
        return equation


def generate_narrative(expected_returns: dict,
                       driver_attributions: dict,
                       regime_status: str) -> str:
    """
    Generate human-readable summary of the analysis.
    """
    narratives = []
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        if asset not in expected_returns or asset not in driver_attributions:
            continue
            
        exp_ret = expected_returns[asset]
        attr = driver_attributions[asset]
        
        tailwinds = attr[attr['Impact'] > 0.005].sort_values('Impact', ascending=False).head(2)
        headwinds = attr[attr['Impact'] < -0.005].sort_values('Impact', ascending=True).head(2)
        
        # Extract feature names (remove transformation suffixes for display)
        tailwind_list = [f.split('_')[0] for f in tailwinds['feature'].tolist()]
        headwind_list = [f.split('_')[0] for f in headwinds['feature'].tolist()]
        
        tailwind_str = ', '.join(tailwind_list) or 'none'
        headwind_str = ', '.join(headwind_list) or 'none'
        
        narratives.append(
            f"**{asset}** ({exp_ret:.1%} expected): "
            f"Tailwinds from {tailwind_str}; headwinds from {headwind_str}."
        )
    
    regime_note = {
        'CALM': 'Regime is stable, no defensive adjustment needed.',
        'WARNING': 'Elevated stress detected, modest defensive tilt applied.',
        'ALERT': 'High stress regime, significant defensive positioning.'
    }.get(regime_status, 'Regime status unknown.')
    
    return '\n\n'.join(narratives) + f'\n\n{regime_note}'


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Macro-Driven Strategic Asset Allocation System",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar Configuration (Moved up for header visibility)
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid #2a2a2a; margin-bottom: 1rem;">
            <span style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #ff6b35;">CONFIG</span>
        </div>
        """, unsafe_allow_html=True)
        
        horizon_months = st.slider("Horizon (Months)", 3, 36, 12, help="Forward return horizon")
        l1_ratio = st.slider("L1 Ratio", 0.1, 0.9, 0.5, 0.1, help="Elastic Net mixing parameter")
        min_persistence = st.slider("Min Persistence", 0.3, 0.9, 0.6, 0.1, help="Feature selection threshold")
        confidence_level = st.slider("Confidence Level", 0.80, 0.95, 0.90, 0.05)
        estimation_window_years = st.slider("Estimation Window (Years)", 15, 35, 25)
        
        alert_threshold = st.slider("Alert Threshold", 1.0, 3.0, 2.0, 0.25)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100
    
    
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-tertiary: #1a1a1a;
        --border-color: #2a2a2a;
        --text-primary: #e8e8e8;
        --text-secondary: #888888;
        --text-muted: #555555;
        --accent-orange: #ff6b35;
        --accent-green: #00d26a;
        --accent-red: #ff4757;
        --accent-blue: #4da6ff;
        --accent-gold: #ffd700;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    .header-container {
        background: linear-gradient(180deg, #111111 0%, #0a0a0a 100%);
        border-bottom: 1px solid var(--border-color);
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-orange);
        letter-spacing: 0.5px;
        margin: 0;
    }
    
    .header-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    .panel-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-card {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0.25rem;
    }
    
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.warning { color: var(--accent-orange); }
    
    .data-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        width: 100%;
        border-collapse: collapse;
    }
    
    .data-table th {
        background-color: var(--bg-tertiary);
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        text-align: left;
    }
    
    .data-table td {
        color: var(--text-primary);
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton > button {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-orange);
        border-color: var(--accent-orange);
        color: #000;
    }
    
    .stTabs [data-baseweb="tab-list"] { background-color: var(--bg-secondary); gap: 0; }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-primary);
        color: var(--accent-orange);
        border-bottom-color: var(--bg-primary);
    }
    
    .debug-box {
        background: #1a1a1a;
        border: 1px solid #333;
        padding: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #888;
        margin: 0.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="header-container">
        <p class="header-title">◈ MACRO-DRIVEN STRATEGIC ASSET ALLOCATION SYSTEM</p>
        <p class="header-subtitle">FORWARD RETURN PREDICTION · {horizon_months}-MONTH HORIZON</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    macro_data = load_fred_md_data()
    asset_prices = load_asset_data()
    descriptions = get_series_descriptions()
    
    if macro_data.empty or asset_prices.empty:
        st.error("Failed to load required data.")
        return
    
    common_idx = macro_data.index.intersection(asset_prices.index)
    macro_data = macro_data.loc[common_idx]
    asset_prices = asset_prices.loc[common_idx]
    
    # 1. Feature Preparation
    macro_features = prepare_macro_features(macro_data)
    y_forward = compute_forward_returns(asset_prices, horizon_months=horizon_months)
    
    # Align again because of lags and forward shifts
    valid_idx = macro_features.index.intersection(y_forward.index)
    X = macro_features.loc[valid_idx]
    y = y_forward.loc[valid_idx]
    
    # 2. Main Analysis Loop (Synchronized & Cached)
    with st.spinner("Synchronizing Macro Models..."):
        expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats = run_collective_analysis(
            y, X, l1_ratio, min_persistence, estimation_window_years, horizon_months
        )
            
    # 3. Regime and Allocation
    regime_status, stress_score, stress_indicators = evaluate_regime(macro_data, alert_threshold=alert_threshold)
    target_weights = compute_allocation(expected_returns, confidence_intervals, regime_status, risk_free_rate=risk_free_rate)
    
    # 4. Dashboard Implementation
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Regime", regime_status)
    with m2:
        st.metric("Stress Score", f"{stress_score:.2f}")
    with m3:
        st.metric("Risk-Free Rate", f"{risk_free_rate:.1%}")
    with m4:
        st.metric("Horizon", f"{horizon_months}m")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ALLOCATION", "STABLE DRIVERS", "SERIES", "BACKTEST", "DIAGNOSTICS"])
    
    with tab1:
        st.markdown(f'<div class="panel-header">EXPECTED {horizon_months}M RETURNS & STRATEGIC POSITIONING</div>', unsafe_allow_html=True)
        
        # summary_panel
        summary_data = []
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            exp = expected_returns[asset]
            ci = confidence_intervals[asset]
            # Historical avg (approx)
            avg_ret = y[asset].mean()
            diff = exp - avg_ret
            rec = "OVERWEIGHT" if diff > 0.01 else "UNDERWEIGHT" if diff < -0.01 else "NEUTRAL"
            
            summary_data.append({
                'Asset': asset,
                'Expected Return': f"{exp:.1%}",
                f'{int(confidence_level*100)}% CI': f"[{ci[0]:.1%}, {ci[1]:.1%}]",
                'vs Historical': f"{diff:+.1%}",
                'Recommendation': rec,
                'Target Weight': f"{target_weights[asset]:.0%}"
            })
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, width='stretch')
        
        # Allocation Charts
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.plotly_chart(plot_allocation(target_weights), width='stretch')
        with col_c2:
            st.markdown('<div class="panel-header">STRATEGIC RATIONALE</div>', unsafe_allow_html=True)
            narrative = generate_narrative(expected_returns, driver_attributions, regime_status)
            st.markdown(f"""
            <div style="background:#111; border:1px solid #2a2a2a; padding:1rem; border-radius:2px; font-size:0.85rem; color:#ccc;">
                {narrative}
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.markdown(f'<div class="panel-header">STABLE MACRO DRIVERS (PERSISTENCE > {int(min_persistence*100)}%)</div>', unsafe_allow_html=True)
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            with st.expander(f"Drivers & Equation for {asset}", expanded=(asset=='EQUITY')):
                # Display Equation / Summary
                st.markdown(construct_model_summary(asset, model_stats))
                st.divider()
                
                attr = driver_attributions[asset]
                # Filter for stable features only
                stable_feats = stability_results_map[asset].get('stable_features', [])
                attr = attr[attr['feature'].isin(stable_feats)].copy()
                # Add absolute impact for sorting if needed, but we keep Impact column
                attr['AbsWeight'] = attr['Weight'].abs()
                attr = attr.sort_values('AbsWeight', ascending=False)

                selection = st.dataframe(
                    attr[['feature', 'Signal', 'Impact', 'Weight', 'State', 'Link']], 
                    hide_index=True, 
                    width='stretch',
                    on_select='rerun',
                    selection_mode='single-row',
                    key=f"df_selection_{asset}",
                    column_config={
                        'feature': st.column_config.TextColumn("Driver", width="medium"),
                        'Signal': st.column_config.TextColumn("Current Signal", width="small"),
                        'Impact': st.column_config.NumberColumn("Impact", format="%.4f", help="Total contribution (Weight * State)"),
                        'Weight': st.column_config.NumberColumn("Weight", format="%.3f", help="Predictive impact of the driver"),
                        'State': st.column_config.NumberColumn("State (Z)", format="%.2f", help="Current Z-score of the macro driver"),
                        'Link': st.column_config.NumberColumn("Link (Corr)", format="%.2f", help="Historical correlation with asset returns")
                    }
                )
                
                selected_rows = selection.get('selection', {}).get('rows', [])
                if selected_rows:
                    row_idx = selected_rows[0]
                    selected_feat = attr.iloc[row_idx]['feature']
                    
                    st.plotly_chart(
                        plot_combined_driver_analysis(macro_features, y_forward, selected_feat, asset, descriptions, horizon_months=horizon_months),
                        width='stretch'
                    )

                    # NEW: Deep Dive Analysis Row 2 (Correlation & Quintiles side-by-side)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plot_driver_scatter(macro_features, y_forward, selected_feat, asset, descriptions), width='stretch')
                    with c2:
                        st.plotly_chart(plot_quintile_analysis(macro_features, y_forward, selected_feat, asset, horizon_months=horizon_months), width='stretch')
                
                # Stability Boxplot
                st.plotly_chart(plot_stability_boxplot(stability_results_map, asset, descriptions), width='stretch')

                # Step 3: Survival Leaderboard
                st.plotly_chart(plot_variable_survival(stability_results_map, asset, descriptions), width='stretch')

    with tab3:
        # Asset and Display Mode Selection on a single row
        sel_col1, sel_col2, sel_col3, sel_col4, sel_col5, sel_col6 = st.columns([1.5, 1, 1, 1, 2, 3])
        with sel_col1:
            st.markdown("**Impact on:**")
        with sel_col2:
            f_equity = st.checkbox("Equity", value=True, key="check_equity")
        with sel_col3:
            f_bonds = st.checkbox("Bonds", value=False, key="check_bonds")
        with sel_col4:
            f_gold = st.checkbox("Gold", value=False, key="check_gold")
        
        with sel_col5:
            st.markdown("**Display Mode:**")
        with sel_col6:
            display_mode = st.radio("Display Mode", ["Raw", "Transformed", "Normalized"], horizontal=True, key="series_display_mode", label_visibility="collapsed")

        if display_mode == "Normalized":
            st.info("💡 **Normalized Mode**: Displays point-in-time Z-scores using an expanding window (no look-ahead bias), with values clipped at ±3.0 standard deviations for optimal readability.")


        # Load raw data and appendix
        df_full, transform_codes = load_full_fred_md_raw()
        appendix = load_fred_appendix()
        
        active_series = {} # fred_md_symbol -> max_abs_importance (Weight)
        
        # Robust mapping from model core names back to FRED-MD raw symbols
        name_to_fred = {
            'PAYEMS': 'PAYEMS',
            'UNRATE': 'UNRATE',
            'INDPRO': 'INDPRO',
            'CAPACITY': 'CUMFNS',
            'CPI': 'CPIAUCSL',
            'PPI': 'WPSFD49207',
            'PCE': 'PCEPI',
            'FEDFUNDS': 'FEDFUNDS',
            'GS10': 'GS10',
            'HOUST': 'HOUST',
            'M2': 'M2SL'
        }
        # Reverse mapping for display
        fred_to_name = {v: k for k, v in name_to_fred.items()}
        fred_to_name.update({'BAA': 'BAA', 'AAA': 'AAA'})

        # Longest names first to ensure correct prefix matching
        target_names = sorted(list(name_to_fred.keys()) + ['SPREAD', 'BAA_AAA'], key=len, reverse=True)
        
        # If any specific asset is checked, we identify the top impactful drivers
        selected_assets = [a for a in ['EQUITY', 'BONDS', 'GOLD'] if (f_equity if a=='EQUITY' else f_bonds if a=='BONDS' else f_gold)]

        if selected_assets:
            for asset in selected_assets:
                attr = driver_attributions[asset].copy()
                # We use ALL features from attributions to match the bullet points (which include all top imports)
                attr['AbsWeight'] = attr['Weight'].abs()
                
                # Map each feature back to one or more FRED-MD series
                for _, row in attr.iterrows():
                    feat = row['feature']
                    weight = row['AbsWeight']
                    
                    # Identify base model name (e.g., HOUST_MA60 -> HOUST)
                    base_name = None
                    for tn in target_names:
                        if feat == tn or feat.startswith(tn + "_"):
                            base_name = tn
                            break
                    
                    if not base_name: 
                        base_name = feat.split('_')[0]

                    # Map to FRED-MD symbols
                    fred_cols = []
                    if base_name == 'SPREAD': fred_cols = ['GS10', 'FEDFUNDS']
                    elif base_name == 'BAA_AAA': fred_cols = ['BAA', 'AAA']
                    else: fred_cols = [name_to_fred.get(base_name, base_name)]
                    
                    for col in fred_cols:
                        if col in df_full.columns:
                            # Maintain the maximum weight encountered for this series
                            if col not in active_series or weight > active_series[col]:
                                active_series[col] = weight
            
            # Sort by absolute weight (decreasing) and take top 15
            sorted_by_importance = sorted(active_series.items(), key=lambda x: x[1], reverse=True)
            sorted_series = [s[0] for s in sorted_by_importance[:15]]
        else:
            # If no asset selected, show all alphabetically
            sorted_series = sorted(list(df_full.columns))

        if not df_full.empty and sorted_series:
            num_series = len(sorted_series)
            
            # Create subplots with ABSOLUTE zero spacing and no external titles
            fig = make_subplots(
                rows=num_series, 
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0 # True zero spacing for a contiguous vertical area
            )
            
            for i, col in enumerate(sorted_series):
                row = i + 1
                tcode = 1
                if col in transform_codes.index:
                    try: tcode = int(transform_codes[col])
                    except: pass
                
                raw_data = df_full[col].dropna()
                if display_mode == "Raw":
                    data = raw_data
                else:
                    data = apply_transformation(raw_data, tcode).dropna()
                    if display_mode == "Normalized" and not data.empty:
                        # Expanding window z-score (point-in-time vision)
                        # We use a 12-month minimum window to establish a stable mean/std
                        means = data.expanding(min_periods=12).mean()
                        stds = data.expanding(min_periods=12).std()
                        data = (data - means) / stds
                        
                        # Winsorization (Clip at +/- 3.0 for readability)
                        data = data.clip(-3.0, 3.0)
                
                if data.empty: continue
                
                # Get description for hover
                desc = col
                if col.upper() in appendix.index:
                    series_info = appendix.loc[col.upper()]
                    desc_str = series_info['description'] if isinstance(series_info, pd.Series) else series_info.iloc[0]['description']
                    desc = f"{col}: {desc_str}"
                
                # Use display name if available (e.g. PPI instead of WPSFD49207)
                display_name = fred_to_name.get(col, col)
                
                # Append transformation to name if in Transformed/Normalized mode
                if display_mode in ["Transformed", "Normalized"]:
                    label = TRANSFORMATION_LABELS.get(tcode, "Unknown")
                    display_name = f"{display_name} ({label})"
                    if display_mode == "Normalized":
                        display_name += " [Z]"

                color = '#4da6ff' if display_mode == "Raw" else '#ff4757' if display_mode == "Transformed" else '#00d26a'

                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data.values,
                        mode='lines',
                        name=display_name,
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'<b>{display_name}</b><br>%{{x|%b %Y}}<br>Value: %{{y:.4f}}<extra>{desc}</extra>'
                    ),
                    row=row, col=1
                )
                
                # Get plain description for annotation
                plain_desc = ""
                if col.upper() in appendix.index:
                    series_info = appendix.loc[col.upper()]
                    plain_desc = series_info['description'] if isinstance(series_info, pd.Series) else series_info.iloc[0]['description']

                # Add Internal Title (Annotation) with Description
                annotation_text = f"<b>{display_name}</b>"
                if plain_desc:
                    annotation_text += f": {plain_desc}"

                fig.add_annotation(
                    text=annotation_text,
                    xref=f"x{row if row > 1 else ''} domain", yref=f"y{row if row > 1 else ''} domain",
                    x=0.01, y=0.95,
                    showarrow=False,
                    font=dict(color='#ff6b35', size=11, family='IBM Plex Mono'),
                    bgcolor='rgba(0,0,0,0.6)',
                    bordercolor='#2a2a2a',
                    borderwidth=1,
                    align='left'
                )

                # Add Recession Bands to each subplot
                for start, end in NBER_RECESSIONS:
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="#ffffff", opacity=0.07,
                        layer="below", line_width=0,
                        row=row, col=1
                    )
            
            theme = create_theme()
            # Synchronize axes styling with high-visibility cross-subplot spikelines
            fig.update_xaxes(
                **theme['xaxis'],
                showspikes=True,
                spikemode='across', # Force the line across subplot gaps
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='rgba(255,255,255,0.8)', # Prominent white line
                spikethickness=1,
                showticklabels=False
            )
            # Only show ticks on the bottom-most plot
            fig.update_xaxes(showticklabels=True, row=num_series, col=1)
            
            fig.update_yaxes(**theme['yaxis'])

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=10, b=40),
                height=150 * num_series, # High density
                showlegend=False,
                hovermode='x', # Better for individual subplot spikeline triggers in zero-gap
                hoverdistance=-1,
                spikedistance=-1
            )
            
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            if len(active_series) > 15:
                st.warning(f"Displaying {len(active_series)} series. Use asset filters to focus on key drivers.")

    with tab4:
        asset_to_plot = st.selectbox("Select Asset", ['EQUITY', 'BONDS', 'GOLD'], key="backtest_asset_select")
        
        model_display_names = {
            'EQUITY': 'Random Forest (Non-Linear Ensemble)',
            'BONDS': 'ElasticNetCV (Regularized Linear)',
            'GOLD': 'Simple OLS (Linear Regression)'
        }
        st.info(f"Target Architecture: **{model_display_names.get(asset_to_plot)}** | Training Window: **240 Months**")
        
        # Run Walk-Forward Backtest (Cached)
        with st.spinner(f"Running Walk-Forward Backtest for {asset_to_plot}..."):
            oos_results, selection_history = cached_walk_forward(
                y[asset_to_plot], 
                X, 
                min_train_months=240, 
                horizon_months=horizon_months, 
                rebalance_freq=12,
                asset_class=asset_to_plot
            )
        
        if not oos_results.empty:
            # Align actual returns with OOS predictions
            actual_oos = y[asset_to_plot].loc[oos_results.index]
            
            # Plot
            fig_backtest = plot_backtest(
                actual_returns=actual_oos, 
                predicted_returns=oos_results['predicted_return'], 
                confidence_lower=oos_results['lower_ci'], 
                confidence_upper=oos_results['upper_ci'],
                confidence_level=confidence_level
            )
            
            # Visual Cue: Update line style for OOS Prediction
            for trace in fig_backtest.data:
                if trace.name == 'Predicted':
                    trace.name = 'OOS Prediction (Walk-Forward)'
                    trace.line.color = '#4da6ff'
            
            st.plotly_chart(fig_backtest, width='stretch')
            
            # Stats
            corr = actual_oos.corr(oos_results['predicted_return'])
            rmse = np.sqrt(((actual_oos - oos_results['predicted_return'])**2).mean())
            st.markdown(f"**OOS Correlation:** {corr:.2f} | **OOS RMSE:** {rmse:.2%}")
        else:
            st.warning("Insufficient data for Walk-Forward Backtest.")

    with tab5:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Regime Indicators**")
            st.dataframe(pd.DataFrame([{'Indicator': k, 'Value': v} for k,v in stress_indicators.items()]), hide_index=True)
        with col_d2:
            st.markdown("**Best-in-Class Models**")
            model_info = {
                'EQUITY': 'Random Forest Regressor (Non-Linear Ensemble)',
                'BONDS': 'ElasticNetCV (L1/L2 Regularized Linear)',
                'GOLD': 'Simple OLS (Ordinary Least Squares)'
            }
            for asset in ['EQUITY', 'BONDS', 'GOLD']:
                m_type = model_info.get(asset, "Unknown")
                n_feats = len(stability_results_map[asset]['stable_features'])
                st.markdown(f"**{asset}**")
                st.markdown(f"- Architecture: `{m_type}`")
                st.markdown(f"- Features: `{n_feats}` macro drivers utilized")
        
        st.divider()
        st.markdown("**Feature Selection Persistence**")
        asset_heatmap = st.selectbox("Select Asset for Heatmap", ['EQUITY', 'BONDS', 'GOLD'], key='heatmap_asset')
        
        with st.spinner(f"Loading selection history for {asset_heatmap}..."):
            _, selection_df = cached_walk_forward(
                y[asset_heatmap], 
                X, 
                min_train_months=240, 
                horizon_months=horizon_months, 
                rebalance_freq=12,
                asset_class=asset_heatmap
            )
            
        if not selection_df.empty:
            st.plotly_chart(plot_feature_heatmap(selection_df, descriptions), width='stretch')
        else:
            st.info("No selection history available for this asset.")
        
        if st.button("Export Results Summary"):
            summary_df = pd.DataFrame(summary_data)
            st.download_button("Download CSV", summary_df.to_csv(index=False), "expected_returns.csv", "text/csv")


if __name__ == "__main__":
    main()