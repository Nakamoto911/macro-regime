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
from backtester import StrategyBacktester
from opus import load_fred_md_data, load_asset_data, prepare_macro_features, compute_forward_returns
import os

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

# Redundant functions now imported from opus.py


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


# Redundant functions now imported from opus.py


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
    
    # Pre-clean: Drop columns that are all NaN or have any NaN in the provided sample
    X = X.dropna(axis=1)
    
    if X.empty:
        return [], pd.Series(dtype=float)

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
                              asset_class: str = 'EQUITY',
                              selection_threshold: float = 0.6,
                              l1_ratio: float = 0.5) -> tuple:
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
        # Drop columns in training set that contain any NaNs (schema evolution)
        X_train = X_train_all.loc[y_train_all.index].dropna(axis=1)
        
        if len(y_train_all) < 60:
            continue
            
        # Determine prediction window
        end_window = min(i + rebalance_freq, len(dates))
        predict_idx = dates[i : end_window]
        
        # -----------------------------------------------------------
        # 1. Point-in-Time Feature Selection
        # -----------------------------------------------------------
        # Use only training data available at this time step
        stable_feats, _ = select_features_elastic_net(
            y_train_all, X_train, 
            threshold=selection_threshold, 
            l1_ratio=l1_ratio,
            n_iterations=30 # Reduced for performance during backtest
        )
        
        # Fallback if no features selected
        if not stable_feats:
            stable_feats = X_train.columns.tolist()
            
        # -----------------------------------------------------------
        # 2. Filter & Winsorize
        # -----------------------------------------------------------
        X_train_sel = X_train[stable_feats]
        # Ensure we only select columns that exist in live data (should be all)
        X_live_sel = X.loc[predict_idx][stable_feats]

        win = Winsorizer(threshold=3.0)
        X_train_clean = win.fit_transform(X_train_sel)
        X_live_clean = win.transform(X_live_sel)
        
        # Record Selection (Binary 1/0 for heatmap)
        current_selection = pd.Series(0, index=X.columns)
        current_selection[stable_feats] = 1
        selection_history.append({**current_selection.to_dict(), 'date': current_date})

        # -----------------------------------------------------------
        # 3. Model Estimation & Prediction
        # -----------------------------------------------------------
        if asset_class == 'EQUITY':
            model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_train_clean, y_train_all)
            raw_preds = model.predict(X_live_clean)
            
        elif asset_class == 'BONDS':
            # Use fixed alphas or simple CV to save time, or full CV if acceptable
            model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=5000)
            model.fit(X_train_clean, y_train_all)
            raw_preds = model.predict(X_live_clean)
            
        else: # GOLD
            model = LinearRegression()
            model.fit(X_train_clean, y_train_all)
            raw_preds = model.predict(X_live_clean)

        # Clip predictions to realistic bounds
        pred_vals = pd.Series(np.clip(raw_preds, -0.30, 0.30), index=predict_idx)
        
        # -----------------------------------------------------------
        # 4. HAC Adjustment for Overlapping Data (Horizon = 12)
        # -----------------------------------------------------------
        # 1. Calculate raw in-sample residuals
        in_sample_preds = model.predict(X_train_clean)
        residuals = y_train_all - in_sample_preds

        # 2. Apply HAC Adjustment (Horizon = 12)
        # Since we have 11/12 overlap, we inflate the standard error by sqrt(12)
        # to correct for the underestimated variance due to autocorrelation.
        hac_adjustment = np.sqrt(horizon_months) 
        se_raw = np.std(residuals)
        se_adjusted = se_raw * hac_adjustment

        # 3. Adjust Degrees of Freedom for Effective Sample Size
        # We effectively have 1 independent observation per year, not per month.
        n_effective = len(y_train_all) / horizon_months
        n_features_used = X_train_clean.shape[1]
        df_effective = max(1, int(n_effective - n_features_used - 1))

        # 4. Calculate Critical Value with adjusted DF
        t_crit = t.ppf(0.95, df=df_effective) 

        # 5. Generate Bands using Adjusted SE
        for date in predict_idx:
            val = pred_vals.loc[date]
            results.append({
                'date': date,
                'predicted_return': val,
                # Use se_adjusted here instead of raw se
                'lower_ci': val - (t_crit * se_adjusted),
                'upper_ci': val + (t_crit * se_adjusted)
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
        valid_mask = y_window.notna()
        y_valid = y_window[valid_mask]
        # Drop columns that have NaNs in this specific window slice
        X_valid = X_window[valid_mask].dropna(axis=1)
        
        if len(y_valid) < 120 or X_valid.empty:  # Minimum 10 years of valid data
            continue
        
        # Estimate using Stability Selection
        selected_features, selection_probs = select_features_elastic_net(y_valid, X_valid)
        
        # Fit OLS on selected features to get coefficients for stability analysis
        if selected_features:
            hac_res = estimate_with_hac(y_valid, X_valid[selected_features], lag=horizon_months//2)
            coef_dict = hac_res['coefficients'].to_dict()
        else:
            coef_dict = {col: 0.0 for col in X_valid.columns}
            
        # Record window correlations for all features (univariate)
        window_corrs = X_valid.corrwith(y_valid)
        
        results.append({
            'start_date': y.index[start],
            'end_date': y.index[end - 1],
            'selected_features': selected_features,
            'coefficients': coef_dict,
            'correlations': window_corrs.to_dict(),
            'n_selected': len(selected_features)
        })
    
    return results


def compute_stability_metrics(stability_results: list, feature_names: list) -> pd.DataFrame:
    """
    Compute stability metrics for each feature across estimation windows.
    """
    n_windows = len(stability_results)
    if n_windows == 0:
        return pd.DataFrame(columns=['feature', 'persistence', 'sign_consistency', 'magnitude_stability', 'mean_coefficient', 'correlation'])
        
    metrics = []
    for feat in feature_names:
        coefs = []
        corrs = []
        for result in stability_results:
            if feat in result.get('coefficients', {}):
                coefs.append(result['coefficients'][feat])
            if feat in result.get('correlations', {}):
                corrs.append(result['correlations'][feat])
        
        coefs = np.array(coefs)
        non_zero = coefs[coefs != 0]
        
        persistence = len(non_zero) / n_windows
        
        # Historical Link (Avg Correlation)
        avg_corr = np.mean(corrs) if corrs else 0.0
        
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
            'mean_coefficient': mean_coef,
            'correlation': avg_corr
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


def get_aggregated_predictions(y, X, horizon_months=12):
    """
    Call cached_walk_forward for EQUITY, BONDS, GOLD and merge results with progress feedback.
    """
    results_preds = {}
    results_lower = {}
    
    assets = ['EQUITY', 'BONDS', 'GOLD']
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, asset in enumerate(assets):
        status_text.markdown(f"**Step {i+1}/3:** Generating OOS Predictions for `{asset}`...")
        progress_bar.progress((i) / len(assets))
        
        oos_results, _ = cached_walk_forward(
            y[asset], 
            X, 
            min_train_months=240, 
            horizon_months=horizon_months, 
            rebalance_freq=12,
            asset_class=asset
        )
        results_preds[asset] = oos_results['predicted_return']
        results_lower[asset] = oos_results['lower_ci']
        
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()
    
    preds_df = pd.DataFrame(results_preds)
    lower_ci_df = pd.DataFrame(results_lower)
    return preds_df, lower_ci_df


def get_historical_stress(macro_data):
    """
    Calculate historical Stress Score based on rolling 60m Z-scores.
    """
    history = pd.DataFrame(index=macro_data.index)
    
    # 1. Calculate BAA_AAA spread Z-Score (rolling 60m)
    if 'BAA_AAA' in macro_data.columns:
        rolled_mean = macro_data['BAA_AAA'].rolling(window=60).mean()
        rolled_std = macro_data['BAA_AAA'].rolling(window=60).std()
        history['credit'] = (macro_data['BAA_AAA'] - rolled_mean) / rolled_std
    else:
        history['credit'] = 0
        
    # 2. Calculate Yield Curve (10Y-FedFunds) Z-Score (rolling 60m)
    if 'SPREAD' in macro_data.columns:
        rolled_mean = macro_data['SPREAD'].rolling(window=60).mean()
        rolled_std = macro_data['SPREAD'].rolling(window=60).std()
        history['curve'] = -(macro_data['SPREAD'] - rolled_mean) / rolled_std
    else:
        history['curve'] = 0
        
    stress_score = 0.5 * history['credit'].fillna(0) + 0.5 * history['curve'].fillna(0)
    return stress_score


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
                fillcolor=theme['recession_color'], opacity=0.07,
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
            text=f"<b>{title}</b><br><span style='font-size: 10px; color: {theme['text_secondary']};'>{subtitle}</span><br><span style='font-size: 9px; color: {theme['text_muted']};'>{stats_str}</span>",
            font=dict(family='IBM Plex Mono', size=13, color=theme['font']['color']),
            x=0.05, y=0.92
        ),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=75, b=40),
        height=280,
        xaxis=dict(**theme['xaxis']),
        yaxis=dict(**theme['yaxis'])
    )
    
    return fig


def create_theme():
    theme_name = st.session_state.get('theme', 'dark')
    if theme_name == 'light':
        return {
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#ffffff',
            'gridcolor': '#f0f0f0',
            'linecolor': '#e0e0e0',
            'label_color': '#444444',
            'font': {'family': 'IBM Plex Mono', 'color': '#1a1a1a', 'size': 11},
            'xaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#e0e0e0', 'tickcolor': '#555', 'tickfont': {'color': '#666'}},
            'yaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#e0e0e0', 'tickcolor': '#555', 'tickfont': {'color': '#666'}},
            'recession_color': '#cccccc',
            'border_color': '#dee2e6',
            'text_secondary': '#444444',
            'text_muted': '#666666'
        }
    else:
        return {
            'paper_bgcolor': '#0a0a0a',
            'plot_bgcolor': '#111111',
            'gridcolor': '#1a1a1a',
            'linecolor': '#2a2a2a',
            'label_color': '#888888',
            'font': {'family': 'IBM Plex Mono', 'color': '#e8e8e8', 'size': 11},
            'xaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a', 'tickcolor': '#888', 'tickfont': {'color': '#888'}},
            'yaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a', 'tickcolor': '#888', 'tickfont': {'color': '#888'}},
            'recession_color': '#ffffff',
            'border_color': '#2a2a2a',
            'text_secondary': '#888888',
            'text_muted': '#555555'
        }


def plot_allocation(weights: dict) -> go.Figure:
    theme = create_theme()
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.65,
        marker=dict(colors=[colors[k] for k in weights.keys()], line=dict(color=theme['paper_bgcolor'], width=2)),
        textinfo='label+percent',
        textfont=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])
    )])
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        annotations=[dict(text='<b>TARGET</b>', x=0.5, y=0.5,
                         font=dict(family='IBM Plex Mono', size=12, color=theme['label_color']), showarrow=False)]
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
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=280,
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Indexed (100)', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color']))
    )
    return fig


def plot_ect(ect: pd.Series) -> go.Figure:
    theme = create_theme()
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'])
    
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
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='ECT', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color']))
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
        colorscale=[[0, theme['plot_bgcolor']], [0.5, 'rgba(77, 166, 255, 0.2)'], [1, '#4da6ff']],
        showscale=True,
        colorbar=dict(title=dict(text='Selection Prob', side='right', font=dict(color=theme['label_color'])), thickness=15, len=0.7, tickfont=dict(color=theme['label_color']))
    ))
    
    fig.update_layout(
        title=dict(text='STABILITY SELECTION: FEATURE PERSISTENCE OVER TIME', font=dict(size=12, color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        xaxis=dict(title=dict(text='Estimation Date', font=dict(color=theme['label_color'])), gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(size=9, color=theme['label_color'])),
        margin=dict(l=10, r=10, t=40, b=40),
        height=500,
        font=theme['font']
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
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Coefficient', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        font=theme['font']
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
    
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'])
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        xaxis=dict(tickangle=-45, tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='5Y Slope (Annualized)', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
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
                  font=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=50, t=40, b=40),
        height=350,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])),
        hovermode='x unified',
        xaxis=dict(
            gridcolor=theme['gridcolor'],
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dash',
            spikethickness=1,
            spikecolor=theme['text_muted'],
            tickfont=dict(color=theme['label_color'])
        ),
        yaxis=dict(gridcolor=theme['gridcolor'], side='left', title=dict(text=f'Macro: {feat_name}', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis2=dict(gridcolor=theme['gridcolor'], overlaying='y', side='right', title=dict(text='Forward Return (Annualized)', font=dict(color=theme['label_color'])), tickformat='.0%', tickfont=dict(color=theme['label_color']))
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
        title=dict(font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])),
        xaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=feat_name, font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f"{asset} Return", font=dict(color=theme['label_color'])), tickformat='.1%', tickfont=dict(color=theme['label_color']))
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
    
    fig.update_layout(
        title=dict(text=f"Rolling {window}M Correlation: {feat_name} vs {asset}", font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Correlation', font=dict(color=theme['label_color'])), range=[-1, 1], tickfont=dict(color=theme['label_color']))
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
        title=dict(font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=350,
        showlegend=False,
        xaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f"{feat_name} Quintiles", font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f'Avg {horizon_months}M Forward Return', font=dict(color=theme['label_color'])), tickformat='.1%', tickfont=dict(color=theme['label_color']))
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
    
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'], row=2, col=1)
    
    # Layout updates
    fig.update_layout(
        title=dict(text=f"{feat_name} ({desc}) Analysis", 
                  font=dict(family='IBM Plex Mono', size=12, color=theme['label_color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=50, t=60, b=40),
        height=550,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])),
        hovermode='x unified'
    )
    
    # Axis styling
    fig.update_xaxes(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), row=1, col=1)
    fig.update_xaxes(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), row=2, col=1)
    fig.update_yaxes(title_text=f"Macro: {feat_name}", gridcolor=theme['gridcolor'], row=1, col=1, secondary_y=False, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    fig.update_yaxes(title_text=f"{horizon_months}M Fwd Return", gridcolor=theme['gridcolor'], tickformat='.0%', row=1, col=1, secondary_y=True, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    fig.update_yaxes(title_text=f"{window}M Correlation", gridcolor=theme['gridcolor'], range=[-1, 1], row=2, col=1, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    
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
    # For Tab 5 Heatmap, we now prefer the PIT selection history.
    # However, this function is generic. We need to check what is passed.
    
    # If using the new backtest_selection_map, it's a DataFrame directly.
    # If using stability_results_map, it's inside 'all_coefficients'.
    
    if isinstance(stability_results_map.get(asset), pd.DataFrame):
       coef_df = stability_results_map[asset]
    else:
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
        labels.append(f"<b>{feat}</b><br><span style='font-size:9px; color:{theme['text_muted']};'>{desc}</span>")
    
    # Create the chart
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=labels,
        orientation='h',
        marker=dict(
            color=counts.values,
            colorscale='Oranges',
            line=dict(color=theme['paper_bgcolor'], width=1)
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


@st.cache_data(show_spinner="Synchronizing Live Macro Models...")
def get_live_model_signals(y, X, l1_ratio, min_persistence, estimation_window_years, horizon_months):
    """
    Computes the CURRENT 'Live' mode signals based on full history.
    Used for Tabs 1 (Allocation), 2 (Stable Drivers), and 3 (Series).
    """
    expected_returns = {}
    confidence_intervals = {}
    driver_attributions = {}
    stability_results_map = {}
    model_stats = {}
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        y_asset = y[asset]
        
        # Stability Analysis (Rolling Windows) - STILL USED for "Link" metric in Tab 2/Allocation
        stab_results = stability_analysis(y_asset, X, horizon_months=horizon_months, window_years=estimation_window_years)
        metrics = compute_stability_metrics(stab_results, X.columns)
        
        # Stability Selection (Full Sample Bootstrapping) - Defines "Today's Drivers"
        stable_features, selection_probs = select_features_elastic_net(
            y_asset.dropna(), 
            X.loc[y_asset.dropna().index],
            threshold=min_persistence,
            l1_ratio=l1_ratio
        )
        
        # Final Asset-Specific Estimation
        y_valid = y_asset.dropna()
        # Drop columns in X that have NaNs in the valid y range
        X_valid_all = X.loc[y_valid.index].dropna(axis=1)
        
        # RESTRICT TO STABLE FEATURES ONLY
        if stable_features:
            # Intersection in case some stable features were dropped due to NaNs in y_valid range
            final_feats = [f for f in stable_features if f in X_valid_all.columns]
        else:
            final_feats = X_valid_all.columns.tolist()
            
        if not final_feats:
            final_feats = X_valid_all.columns.tolist()
            # Absolute fallback if somehow X_valid_all has zero cols (impossible with current PIT)
            final_feats = X.columns.tolist()[:5] 
            
        X_valid = X_valid_all[final_feats]
        # Robust Current State Prediction: Take last row, fill NaNs with 0 (neutral) 
        # as a failsafe against late-arriving macro data.
        X_current = X.tail(1)[final_feats].infer_objects(copy=False).fillna(0)

        win = Winsorizer(threshold=3.0)
        X_valid_clean = win.fit_transform(X_valid)
        X_current_clean = win.transform(X_current).infer_objects(copy=False).fillna(0)
        
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
            # Safe access to metrics DataFrame
            if feat in metrics.index:
                feat_corr = metrics.loc[feat, 'correlation']
            else:
                feat_corr = 0
                
            attr_data.append({
                'feature': feat,
                'Impact': impact,
                'State': val,
                'Weight': coef,
                'Signal': 'BULLISH' if impact > 0.005 else 'BEARISH' if impact < -0.005 else 'NEUTRAL',
                'Link': feat_corr
            })
        driver_attributions[asset] = pd.DataFrame(attr_data).sort_values('Impact', ascending=False)
        stability_results_map[asset] = {
            'metrics': metrics,
            'stable_features': stable_features,
            'hac_results': hac_results,
            # We still keep 'all_coefficients' derived from Rolling Stability for Tab 2 context if needed,
            # but Tab 5 will no longer use this.
            'all_coefficients': pd.DataFrame([res['coefficients'] for res in stab_results])
        }
        model_stats[asset] = hac_results

    return expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats


@st.cache_data(show_spinner="Running Walk-Forward Simulator (This may take a minute)...")
def get_historical_backtest(y, X, min_train_months, horizon_months, rebalance_freq, selection_threshold, l1_ratio):
    """
    Wrapper for the PIT Backtester to run it for all assets and cache the result.
    This drives Tabs 4 (Prediction), 5 (Diagnostics), and 6 (Backtest).
    """
    results = {}
    heatmaps = {}
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        oos_df, sel_df = run_walk_forward_backtest(
            y[asset], X, 
            min_train_months=min_train_months, 
            horizon_months=horizon_months, 
            rebalance_freq=rebalance_freq, 
            asset_class=asset,
            selection_threshold=selection_threshold,
            l1_ratio=l1_ratio
        )
        results[asset] = oos_df
        heatmaps[asset] = sel_df
        
    return results, heatmaps


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
            bgcolor='rgba(0,0,0,0.6)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)',
            font=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])
        ),
        xaxis=dict(
            gridcolor=theme['gridcolor'],
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dash',
            spikethickness=1,
            spikecolor=theme['text_muted'],
            tickfont=dict(color=theme['label_color'])
        ),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Annualized Return', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
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
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
        
    # Sidebar Configuration (Moved up for header visibility)
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid var(--border-color); margin-bottom: 1rem;">
            <span style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #ff6b35;">CONFIG & THEME</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme Toggle
        theme_col1, theme_col2 = st.columns([1, 1])
        with theme_col1:
            if st.button("🌙 Dark", use_container_width=True, 
                         type="primary" if st.session_state.theme == 'dark' else "secondary"):
                st.session_state.theme = 'dark'
                st.rerun()
        with theme_col2:
            if st.button("☀ Light", use_container_width=True,
                         type="primary" if st.session_state.theme == 'light' else "secondary"):
                st.session_state.theme = 'light'
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        horizon_months = st.slider("Horizon (Months)", 3, 36, 12, help="Forward return horizon")
        l1_ratio = st.slider("L1 Ratio", 0.1, 0.9, 0.5, 0.1, help="Elastic Net mixing parameter")
        min_persistence = st.slider("Min Persistence", 0.3, 0.9, 0.6, 0.1, help="Feature selection threshold")
        confidence_level = st.slider("Confidence Level", 0.80, 0.95, 0.90, 0.05)
        estimation_window_years = st.slider("Estimation Window (Years)", 15, 35, 25)
        
        alert_threshold = st.slider("Alert Threshold", 1.0, 3.0, 2.0, 0.25)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100
    
    
    # Theme variables
    if st.session_state.theme == 'light':
        bg_primary = "#ffffff"
        bg_secondary = "#f8f9fa"
        bg_tertiary = "#f1f3f5"
        border_color = "#dee2e6"
        text_primary = "#1a1a1a"
        text_secondary = "#4a4a4a"
        text_muted = "#666666"
        header_gradient = "linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%)"
        plot_grid = "#f5f5f5"
    else:
        bg_primary = "#0a0a0a"
        bg_secondary = "#111111"
        bg_tertiary = "#1a1a1a"
        border_color = "#2a2a2a"
        text_primary = "#e8e8e8"
        text_secondary = "#888888"
        text_muted = "#555555"
        header_gradient = "linear-gradient(180deg, #111111 0%, #0a0a0a 100%)"
        plot_grid = "#1a1a1a"

    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-tertiary: {bg_tertiary};
        --border-color: {border_color};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --header-gradient: {header_gradient};
        --accent-orange: #ff6b35;
        --accent-green: #00d26a;
        --accent-red: #ff4757;
        --accent-blue: #4da6ff;
        --accent-gold: #ffd700;
    }}
    
    .stApp {{
        background-color: var(--bg-primary);
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    
    .main .block-container {{
        padding: 1rem 2rem;
        max-width: 100%;
    }}
    
    .header-container {{
        background: var(--header-gradient);
        border-bottom: 1px solid var(--border-color);
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }}
    
    .header-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-orange);
        letter-spacing: 0.5px;
        margin: 0;
    }}
    
    .header-subtitle {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }}
    
    .panel-header {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }}
    
    .metric-card {{
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 0.75rem;
        text-align: center;
    }}
    
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    
    .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0.25rem;
    }}
    
    .metric-value.positive {{ color: var(--accent-green); }}
    .metric-value.negative {{ color: var(--accent-red); }}
    .metric-value.warning {{ color: var(--accent-orange); }}
    
    .data-table {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        width: 100%;
        border-collapse: collapse;
    }}
    
    .data-table th {{
        background-color: var(--bg-tertiary);
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        text-align: left;
    }}
    
    .data-table td {{
        color: var(--text-primary);
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }}
    
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .stButton > button {{
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background-color: var(--accent-orange);
        border-color: var(--accent-orange);
        color: #000;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{ background-color: var(--bg-secondary); gap: 0; }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 0 20px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--bg-primary);
        color: var(--accent-orange);
        border-bottom-color: var(--bg-primary);
    }}
    
    .debug-box {{
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        padding: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin: 0.5rem 0;
        border-radius: 2px;
    }}

    /* Streamlit overrides for light mode readability */
    label, p, span {{
        color: var(--text-primary) !important;
    }}
    .stSlider label, .stNumberInput label, .stSelectbox label {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    input {{
        color: var(--text-primary) !important;
        background-color: var(--bg-tertiary) !important;
    }}
    .stMarkdown div p {{
        color: var(--text-secondary) !important;
    }}
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="header-container">
        <p class="header-title">◈ MACRO-DRIVEN STRATEGIC ASSET ALLOCATION SYSTEM</p>
        <p class="header-subtitle">FORWARD RETURN PREDICTION · {horizon_months}-MONTH HORIZON</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    asset_prices = load_asset_data()
    descriptions = get_series_descriptions()
    
    # 1. LIVE MODEL DATA (LATEST VINTAGE)
    # Always use the latest available vintage for "Nowcast" and current allocation
    macro_data_current = load_fred_md_data()
    if macro_data_current.empty or asset_prices.empty:
        st.error("Failed to load required data.")
        return
    
    # Generate features fresh from the latest vintage
    X_current = prepare_macro_features(macro_data_current)
    y_forward = compute_forward_returns(asset_prices, horizon_months=horizon_months)
    
    # Align for Live Model
    valid_idx = X_current.index.intersection(y_forward.index)
    X_live = X_current.loc[valid_idx]
    y_live = y_forward.loc[valid_idx]

    # 2. BACKTEST DATA (POINT-IN-TIME)
    # Prefer PIT matrix for historical simulation to avoid revision bias
    PIT_FILE = 'PIT_Macro_Features.csv'
    if os.path.exists(PIT_FILE):
        X_pit = pd.read_csv(PIT_FILE, index_col=0, parse_dates=True)
        # st.toast(f"Backtest Engine: Using PIT Matrix ({len(X_pit)} rows)", icon="⏳")
    else:
        st.warning("⚠️ PIT Matrix not found. Backtest will use latest vintage (contains look-ahead bias).")
        X_pit = X_current.copy()
        
    # Align for Backtest
    # Note: y_forward is the same (asset prices don't get revised)
    valid_idx_pit = X_pit.index.intersection(y_forward.index)
    X_backtest = X_pit.loc[valid_idx_pit]
    y_backtest = y_forward.loc[valid_idx_pit]
    
    # 3. Execution
    
    # A. Live Model (for current Allocation & Narrative) - Uses Latest Vintage
    expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats = get_live_model_signals(
        y_live, X_live, l1_ratio, min_persistence, estimation_window_years, horizon_months
    )
    
    # B. Prediction Model Validation (Revised Data Simulator) - Uses Full Revised History Features
    # Note: Diagnostic tab and Prediction tab use these results.
    prediction_results, prediction_selection = get_historical_backtest(
        y_live, X_live, 
        min_train_months=240, 
        horizon_months=horizon_months, 
        rebalance_freq=12,
        selection_threshold=min_persistence,
        l1_ratio=l1_ratio
    )
    
    # C. Historic Backtest (Unbiased PIT Simulator) - Uses PIT Features
    # Note: 'Backtest' tab and Simulation Engine use these results.
    backtest_results, backtest_selection = get_historical_backtest(
        y_backtest, X_backtest, 
        min_train_months=240, 
        horizon_months=horizon_months, 
        rebalance_freq=12,
        selection_threshold=min_persistence,
        l1_ratio=l1_ratio
    )
            
    # 4. Regime and Allocation
    regime_status, stress_score, stress_indicators = evaluate_regime(macro_data_current, alert_threshold=alert_threshold)
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ALLOCATION", "STABLE DRIVERS", "SERIES", "PREDICTION", "DIAGNOSTIC", "BACKTEST", "README"
    ])
    with tab1:
        st.markdown(f'<div class="panel-header">EXPECTED {horizon_months}M RETURNS & STRATEGIC POSITIONING</div>', unsafe_allow_html=True)
        
        # summary_panel
        summary_data = []
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            exp = expected_returns[asset]
            ci = confidence_intervals[asset]
            # Historical avg (approx)
            avg_ret = y_live[asset].mean()
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
            <div style="background:var(--bg-secondary); border:1px solid var(--border-color); padding:1rem; border-radius:2px; font-size:0.85rem; color:var(--text-secondary);">
                {narrative}
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.markdown(f'<div class="panel-header">STABLE MACRO DRIVERS (PERSISTENCE > {int(min_persistence*100)}%)</div>', unsafe_allow_html=True)
        st.info("💡 **Note**: Drivers identified based on full history (2000-Present). The Equations below are 'best fit' retrospective descriptions of the current regime.")
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
                        plot_combined_driver_analysis(X_live, y_forward, selected_feat, asset, descriptions, horizon_months=horizon_months),
                        width='stretch'
                    )

                    # NEW: Deep Dive Analysis Row 2 (Correlation & Quintiles side-by-side)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plot_driver_scatter(X_live, y_forward, selected_feat, asset, descriptions), width='stretch')
                    with c2:
                        st.plotly_chart(plot_quintile_analysis(X_live, y_forward, selected_feat, asset, horizon_months=horizon_months), width='stretch')
                

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
            theme = create_theme()
            has_assets = len(selected_assets) > 0
            num_series = len(sorted_series) + (1 if has_assets else 0)
            
            # Create subplots with ABSOLUTE zero spacing and no external titles
            fig = make_subplots(
                rows=num_series, 
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0 # True zero spacing for a contiguous vertical area
            )
            
            start_row = 1
            if has_assets:
                start_row = 2
                # Add asset return trace(s) at row 1
                for asset in selected_assets:
                    asset_data = y_live[asset].dropna()
                    
                    # Use unique colors for assets to distinguish from macro
                    asset_color = '#ffffff' # White for high contrast
                    if asset == 'EQUITY': asset_color = '#ffd700' # Gold
                    elif asset == 'BONDS': asset_color = '#00ffff' # Cyan
                    elif asset == 'GOLD': asset_color = '#ffffff' # White
                    
                    fig.add_trace(
                        go.Scatter(
                            x=asset_data.index, 
                            y=asset_data.values,
                            mode='lines',
                            name=f"{asset} Ret",
                            line=dict(color=asset_color, width=2),
                            hovertemplate=f'<b>{asset} 12M Forward Return</b><br>%{{x|%b %Y}}<br>Return: %{{y:.2%}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                fig.add_annotation(
                    text=f"<b>ASSET 12M FORWARD RETURNS</b>",
                    xref="x domain", yref="y domain",
                    x=0.01, y=0.95,
                    showarrow=False,
                    font=dict(color='#ff6b35', size=11, family='IBM Plex Mono'),
                    bgcolor='rgba(0,0,0,0.6)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)',
                    bordercolor=theme['border_color'],
                    borderwidth=1,
                    align='left'
                )
                
                # Add Recession Bands to row 1
                for start, end in NBER_RECESSIONS:
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=theme['recession_color'], opacity=0.07,
                        layer="below", line_width=0,
                        row=1, col=1
                    )

            for i, col in enumerate(sorted_series):
                row = i + start_row
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
                    bgcolor='rgba(0,0,0,0.6)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)',
                    bordercolor=theme['border_color'],
                    borderwidth=1,
                    align='left'
                )

                # Add Recession Bands to each subplot
                for start, end in NBER_RECESSIONS:
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=theme['recession_color'], opacity=0.07,
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
                spikecolor=theme['recession_color'], # Dynamic spike color
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
        st.info(f"Target Architecture: **{model_display_names.get(asset_to_plot)}** | Training Window: **240 Months** | Data Variant: **Revised Macro Data**")
        
        # Display Prediction results automatically (Revised Data)
        # Use prediction results (Revised Data) for this tab
        oos_results = prediction_results.get(asset_to_plot, pd.DataFrame())
        
        if not oos_results.empty:
            # Align actual returns (Revised) with OOS predictions
            actual_oos = y_live[asset_to_plot].loc[oos_results.index]
            
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
            st.warning("Insufficient data for Walk-Forward Model Validation.")

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
        st.markdown("**Model Stability & Feature Selection Persistence**")
        
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            with st.expander(f"Diagnostics for {asset}", expanded=(asset == 'EQUITY')):
                if st.button(f"📊 VIEW {asset} DIAGNOSTICS", key=f"btn_diag_{asset}"):
                    selection_df = prediction_selection.get(asset, pd.DataFrame())
                        
                    if not selection_df.empty:
                        st.markdown("### Feature Selection Persistence")
                        st.plotly_chart(plot_feature_heatmap(selection_df, descriptions), width='stretch', key=f"heatmap_{asset}")
                        
                        st.divider()
                        st.markdown("### Stability Analysis")
                        # Stability Boxplot
                        st.plotly_chart(plot_stability_boxplot(stability_results_map, asset, descriptions), width='stretch', key=f"boxplot_{asset}")
    
                        # Variable Survival Leaderboard
                        st.plotly_chart(plot_variable_survival(stability_results_map, asset, descriptions), width='stretch', key=f"survival_{asset}")
                    else:
                        st.info(f"No diagnostic data available for {asset}.")
                else:
                    st.info(f"Click the button to load stability and persistence diagnostics for {asset}.")
        
        if st.button("Export Results Summary"):
            summary_df = pd.DataFrame(summary_data)
            st.download_button("Download CSV", summary_df.to_csv(index=False), "expected_returns.csv", "text/csv")

    with tab6:
        st.markdown('<div class="panel-header">STRATEGY LAB: MACRO-DRIVEN BACKTESTER</div>', unsafe_allow_html=True)
        
        # Zone A: Configuration
        strategy_type = st.radio("Select Strategy", ["Max Return", "Min Volatility", "Min Drawdown", "Min Loss"], horizontal=True)

        with st.expander("📖 Detailed Strategy Methodologies (Click to expand)", expanded=False):
            st.markdown("""
            ### 🚀 Max Return (Tactical Momentum)
            *   **Core Logic**: This strategy identifies the top-performing assets based on predicted returns.
            *   **Decision**: It selects up to the **'Top N Assets'** that have a predicted return higher than the **Risk-Free Rate**.
            *   **Allocation**: 
                *   **Equal**: Splits the **'Max Asset Weight'** evenly across all selected Top-N assets.
                *   **Proportional**: Splits the **'Max Asset Weight'** based on the relative strength of their predicted returns.
            *   **Goal**: Maximize returns through momentum while allowing for diversification across the strongest forecasted assets.

            ### 📉 Min Volatility (Quantitative Optimization)
            *   **Core Logic**: Uses **Global Minimum Variance (GMV)** optimization. It ignores return predictions and focuses entirely on the co-movement and risk of assets.
            *   **Decision**: It utilizes a **Rolling Covariance Matrix** (based on your 'Covariance Lookback') to calculate the weights that minimize the combined portfolio variance.
            *   **Allocation**: Solver-based weights (summing to 100%) constrained between 0% and 100%. 
            *   **Goal**: Create the smoothest possible equity curve by exploiting diversification and avoiding volatile assets.

            ### 🛡️ Min Drawdown (Regime-Aware Defense)
            *   **Core Logic**: A hybrid approach that switches behavior based on the **Macro Stress Score** (Aggregate Z-Score of instability indicators).
            *   **Decision**: 
                *   **Normal Mode**: Active when Stress Score < 'Alert Threshold'. It defaults to a balanced **60/30/10** allocation.
                *   **Defensive Mode**: Triggered when Stress Score > 'Alert Threshold'. It enforces a strict **'Defensive Equity Cap'** (e.g., max 10% stocks) and moves the rest into Bonds and a specified **'Cash Floor'**.
            *   **Goal**: Protect capital during major market crashes and systemic instabilities.

            ### 💎 Min Loss (Safety-First Confidence)
            *   **Core Logic**: Uses the **Lower 95% Confidence Interval** of the predictions. This strategy requires 'high conviction'.
            *   **Decision**: Instead of looking at what *might* happen (Mean), it looks at the *worst-case* (Lower CI). It only invests in an asset if its **Lower CI is > 'Confidence Threshold'**.
            *   **Allocation**: 
                *   If multiple assets satisfy the condition, they are ranked by either **'Lower CI'** (Safety-First) or **'Expected Return'** (Profit-Optimized).
                *   It selects the top ranked assets (up to **'Top N Assets'**) and allocates the **'Max Combined Weight'** according to the chosen **'Weighting Scheme'**. 
                *   If no asset qualifies, it moves **100% to Cash**.
            *   **Goal**: Minimize the probability of negative returns by only taking bets when the model is statistically very confident in a positive outcome, while allowing for diversification among safe bets.
            """)

        with st.form("backtest_config_form"):
            st.markdown("**1. Passive Benchmark Configuration**")
            bc1, bc2, bc3 = st.columns(3)
            bw_eq = bc1.slider("Benchmark: Equity %", 0, 100, 60, key="bench_eq")
            bw_bond = bc2.slider("Benchmark: Bonds %", 0, 100, 40, key="bench_bond")
            bw_gold = bc3.slider("Benchmark: Gold %", 0, 100, 0, key="bench_gold")
            
            st.divider()
            st.markdown("**2. Simulation Window**")
            # Calculate available OOS range (approximate starting point)
            all_dates = asset_prices.index
            min_train = 240
            if len(all_dates) > min_train:
                oos_start_date = all_dates[min_train]
                oos_end_date = all_dates[-1]
                
                # Create a date range selector
                start_str = oos_start_date.strftime('%Y-%m')
                end_str = oos_end_date.strftime('%Y-%m')
                
                # Fallback to requested defaults if available in options
                default_start = "2000-01" if "2000-01" in [d.strftime('%Y-%m') for d in all_dates[min_train:]] else start_str
                default_end = end_str # Use latest available date by default

                selected_range = st.select_slider(
                    "Select Simulation Period",
                    options=[d.strftime('%Y-%m') for d in all_dates[min_train:]],
                    value=(default_start, default_end)
                )
                sim_start, sim_end = selected_range
            else:
                st.warning("Insufficient data for simulation window selection.")
                sim_start, sim_end = None, None

            st.divider()
            st.markdown("**3. Asset Allocation Constraints**")
            st.info("💡 Set the minimum and maximum percentage allowed for each asset class across the entire simulation.")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                st.markdown("**EQUITY**")
                min_eq = st.slider("Min Eq %", 0, 100, 0, key="min_eq")
                max_eq = st.slider("Max Eq %", 0, 100, 100, key="max_eq")
            with ac2:
                st.markdown("**BONDS**")
                min_bond = st.slider("Min Bond %", 0, 100, 0, key="min_bond")
                max_bond = st.slider("Max Bond %", 0, 100, 100, key="max_bond")
            with ac3:
                st.markdown("**GOLD**")
                min_gold = st.slider("Min Gold %", 0, 100, 0, key="min_gold")
                max_gold = st.slider("Max Gold %", 0, 100, 15, key="max_gold")

            st.divider()
            st.markdown(f"**4. {strategy_type} Parameters**")
            col_lab1, col_lab2 = st.columns(2)
            with col_lab1:
                initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
                trading_cost = st.slider("Trading Cost (bps)", 0, 50, 30)
            with col_lab2:
                rebalance_freq = st.selectbox("Rebalance Frequency", [1, 3, 12], index=1, format_func=lambda x: f"Every {x} Month(s)")
            
            # Dynamic Parameters (Rendered inside form)
            params = {}
            if strategy_type == "Min Drawdown":
                st.divider()
                sc1, sc2, sc3 = st.columns(3)
                params['alert_threshold'] = sc1.slider("Alert Threshold", 1.0, 3.0, 2.0)
                params['defensive_equity_cap'] = sc2.slider("Defensive Equity Cap", 0.0, 0.4, 0.2)
                params['defensive_cash_floor'] = sc3.slider("Defensive Cash Floor", 0.3, 0.8, 0.5)
            elif strategy_type == "Max Return":
                st.divider()
                mc1, mc2, mc3 = st.columns(3)
                params['max_weight'] = mc1.slider("Max Combined Weight", 0.5, 1.0, 1.0)
                params['top_n'] = mc2.slider("Top N Assets", 1, 3, 3)
                params['weighting_scheme'] = mc3.selectbox("Weighting Scheme", ["Equal", "Proportional"], index=1)
                params['risk_free_rate'] = risk_free_rate
            elif strategy_type == "Min Volatility":
                st.divider()
                params['cov_lookback'] = st.slider("Covariance Lookback (Months)", 24, 120, 60)
            elif strategy_type == "Min Loss":
                st.divider()
                lc1, lc2, lc3 = st.columns(3)
                params['confidence_threshold'] = lc1.slider("Confidence Threshold", -0.05, 0.05, 0.0, step=0.005, format="%.3f")
                params['rank_by'] = lc2.selectbox("Rank Qualified Assets By", ["Lower CI", "Expected Return"])
                params['max_weight'] = lc3.slider("Max Combined Weight", 0.1, 1.0, 1.0)
                
                lc4, lc5 = st.columns(2)
                params['top_n'] = lc4.slider("Top N Assets", 1, 3, 3)
                params['weighting_scheme'] = lc5.selectbox("Weighting Scheme", ["Equal", "Proportional"], index=1)
            
            st.divider()
            submitted = st.form_submit_button("🚀 RUN STRATEGY SIMULATION", width='stretch', type="primary")

        # Run Backtest
        if submitted:
            # Package all parameters
            min_weights = {
                'EQUITY': min_eq / 100.0,
                'BONDS': min_bond / 100.0,
                'GOLD': min_gold / 100.0
            }
            max_weights = {
                'EQUITY': max_eq / 100.0,
                'BONDS': max_bond / 100.0,
                'GOLD': max_gold / 100.0
            }

            full_params = {
                'initial_capital': initial_capital,
                'trading_cost_bps': trading_cost,
                'rebalance_freq': rebalance_freq,
                'risk_free_rate': risk_free_rate,
                'min_weights': min_weights,
                'max_weights': max_weights,
                **params
            }
            
            b_total_w = bw_eq + bw_bond + bw_gold
            benchmark_weights = {
                'EQUITY': bw_eq / b_total_w if b_total_w > 0 else 0,
                'BONDS': bw_bond / b_total_w if b_total_w > 0 else 0,
                'GOLD': bw_gold / b_total_w if b_total_w > 0 else 0
            }

            with st.spinner("Initializing strategy engine..."):
                # Prepare inputs
                # Prepare inputs from Unbiased Simulator Results
                # preds_df, lower_ci_df = get_aggregated_predictions(y, X, horizon_months=horizon_months)
                preds_df = pd.DataFrame({k: v['predicted_return'] for k, v in backtest_results.items()}).dropna()
                lower_ci_df = pd.DataFrame({k: v['lower_ci'] for k, v in backtest_results.items()}).dropna()
                hist_stress = get_historical_stress(macro_data_current)
                
                # Filter by selected date range
                if sim_start and sim_end:
                    preds_df = preds_df.loc[sim_start:sim_end]
                    lower_ci_df = lower_ci_df.loc[sim_start:sim_end]
                    hist_stress = hist_stress.loc[sim_start:sim_end]
                    
                # Initialize Backtester
                lab_bt = StrategyBacktester(asset_prices, preds_df, lower_ci_df, hist_stress)
                
                # Run
                lab_results = lab_bt.run_strategy(strategy_type, **full_params)
                benchmark_results = lab_bt.run_strategy("Buy & Hold", weights_dict=benchmark_weights, initial_capital=initial_capital, trading_cost_bps=trading_cost, rebalance_freq=rebalance_freq)
                
                # Store in session state to persist after rerun (if any widgets change)
                st.session_state.lab_results = lab_results
                st.session_state.benchmark_results = benchmark_results
                st.session_state.lab_stress = hist_stress
                st.session_state.lab_freq = rebalance_freq
                st.session_state.lab_initial_capital = initial_capital
                st.session_state.lab_strategy_type = strategy_type
                st.session_state.lab_benchmark_weights = benchmark_weights
                st.session_state.lab_max_weights = max_weights
                st.session_state.lab_horizon_months = horizon_months
                st.session_state.lab_full_params = full_params
                # Initialize reset count if not present
                if 'lab_reset_count' not in st.session_state:
                    st.session_state.lab_reset_count = 0

        if "lab_results" in st.session_state:
            lab_results = st.session_state.lab_results
            benchmark_results = st.session_state.benchmark_results
            hist_stress = st.session_state.lab_stress
            lab_freq = st.session_state.get('lab_freq', 1)
            initial_capital = st.session_state.lab_initial_capital 
            strategy_type = st.session_state.get('lab_strategy_type', strategy_type)
            benchmark_weights = st.session_state.get('lab_benchmark_weights', {})
            max_weights = st.session_state.get('lab_max_weights', {})
            horizon_months = st.session_state.get('lab_horizon_months', 12)
            full_params = st.session_state.get('lab_full_params', {})

            # --- SHARED UI THEME & LAYOUT ---
            theme = create_theme()
            layout_args = {
                'height': 800,
                'showlegend': True,
                'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=theme['font']['color'])),
                'margin': dict(l=50, r=50, t=60, b=50),
                'hovermode': 'x unified',
                'paper_bgcolor': theme['paper_bgcolor'],
                'plot_bgcolor': theme['plot_bgcolor'],
                'font': theme['font']
            }


            # --- TABS FOR RESULTS ---
            # 1. Selection detection is now handled directly in the chart render block for immediacy.
            # (We removed the hoisted loop to avoid confusing states).
            

            # Logic: We use a fixed number of tabs to avoid UI glitches with dynamic counts.
            # But we respect the "priority" by naming and contents.
            # If drilldown is NOT active, we show only one tab as requested: "remove the historical performance tab".
            # Actually, to make it work reliably with the "click" trigger, dynamic tabs should work.
            # Let's try one more robust way: use st.tabs conditionally but with a fixed length if possible? No.
            
            drilldown_active = st.session_state.get('lab_drilldown_active', False)
            
            # Use appropriate labels: Rolling always first, Drill-Down second if active
            if drilldown_active:
                tab_labels = ["🔄 ROLLING ANALYSIS", "📈 HISTORICAL PERFORMANCE (DRILL-DOWN)"]
            else:
                tab_labels = ["🔄 ROLLING ANALYSIS"]
            
            results_tabs = st.tabs(tab_labels)
            
            # Assignment based on order
            res_tab_roll = results_tabs[0]
            res_tab_hist = results_tabs[1] if drilldown_active else None

            if res_tab_hist:
                with res_tab_hist:
                    # Filter results for the drill-down window
                    d_start = st.session_state.lab_drilldown_start
                    d_end = d_start + pd.DateOffset(months=st.session_state.lab_drilldown_duration * 12)
                    
                    # Create filtered copies for plotting
                    lab_results_disp = {
                        'equity_curve': lab_results['equity_curve'].loc[d_start:d_end],
                        'weights': lab_results['weights'].loc[d_start:d_end],
                        'metrics': lab_results['metrics'] # We'll re-calculate below
                    }
                    bench_results_disp = {
                        'equity_curve': benchmark_results['equity_curve'].loc[d_start:d_end],
                        'metrics': benchmark_results['metrics']
                    }
                    hist_stress_disp = hist_stress.loc[d_start:d_end]
                    
                    # Re-base equity curves to initial_capital for the drill-down window
                    if not lab_results_disp['equity_curve'].empty:
                        base_val = lab_results_disp['equity_curve'].iloc[0]
                        lab_results_disp['equity_curve'] = (lab_results_disp['equity_curve'] / base_val) * initial_capital
                    if not bench_results_disp['equity_curve'].empty:
                        base_val = bench_results_disp['equity_curve'].iloc[0]
                        bench_results_disp['equity_curve'] = (bench_results_disp['equity_curve'] / base_val) * initial_capital

                    # Re-calculate metrics for this specific window
                    tmp_bt = StrategyBacktester(asset_prices, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
                    lab_results_disp['metrics'] = tmp_bt._calculate_metrics(lab_results_disp['equity_curve'], lab_results_disp['weights'])
                    bench_results_disp['metrics'] = tmp_bt._calculate_metrics(bench_results_disp['equity_curve'], pd.DataFrame())

                    st.caption(f"Showing performance for **{st.session_state.lab_drilldown_duration} Year window** starting **{d_start.strftime('%b %Y')}**")
                    if st.button("🔄 Reset to Full Analysis", key="reset_drilldown"):
                        st.session_state.lab_drilldown_active = False
                        st.session_state.lab_reset_count = st.session_state.get('lab_reset_count', 0) + 1
                        st.rerun()

                    # Zone B: Visualization (Unified Stacked Chart)
                    fig_lab = make_subplots(
                        rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.02,
                        row_heights=[0.35, 0.45, 0.2],
                        specs=[[{"secondary_y": True}], [{}], [{}]]
                    )

                    # 1. Asset Allocation (Row 1)
                    weights_df = lab_results_disp['weights']
                    colors_map = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700', 'CASH': '#444'}
                    for asset in ['EQUITY', 'BONDS', 'GOLD', 'CASH']:
                        if asset in weights_df.columns:
                            fig_lab.add_trace(go.Scatter(
                                x=weights_df.index, y=weights_df[asset],
                                name=asset, stackgroup='one',
                                line=dict(color=colors_map[asset], width=0.5),
                                fillcolor=colors_map[asset].replace('0.5', '0.3'),
                                legendgroup='assets'
                            ), row=1, col=1)
                    
                    # Overlay Stress Score
                    fig_lab.add_trace(go.Scatter(
                        x=hist_stress_disp.index, y=hist_stress_disp.values / hist_stress_disp.max() if not hist_stress_disp.empty and hist_stress_disp.max() > 0 else hist_stress_disp,
                        name='Stress Score (Norm)', line=dict(color='rgba(255,255,255,0.2)', dash='dot'),
                        legendgroup='regime'
                    ), row=1, col=1, secondary_y=True)

                    # 2. Cumulative Returns (Row 2)
                    fig_lab.add_trace(go.Scatter(
                        x=lab_results_disp['equity_curve'].index, y=lab_results_disp['equity_curve'].values, 
                        name='Strategy NAV', line=dict(color='#ff6b35', width=2),
                        legendgroup='nav'
                    ), row=2, col=1)
                    fig_lab.add_trace(go.Scatter(
                        x=bench_results_disp['equity_curve'].index, y=bench_results_disp['equity_curve'].values, 
                        name='Benchmark NAV', line=dict(color='#888', dash='dash'),
                        legendgroup='nav'
                    ), row=2, col=1)

                    # 3. Drawdown Profile (Row 3)
                    def get_drawdown(curve):
                        return (curve / curve.expanding().max()) - 1
                    
                    fig_lab.add_trace(go.Scatter(
                        x=lab_results_disp['equity_curve'].index, y=get_drawdown(lab_results_disp['equity_curve']), 
                        name='Strategy DD', fill='tozeroy', line=dict(color='#ff4757'),
                        legendgroup='dd'
                    ), row=3, col=1)
                    fig_lab.add_trace(go.Scatter(
                        x=bench_results_disp['equity_curve'].index, y=get_drawdown(bench_results_disp['equity_curve']), 
                        name='Benchmark DD', line=dict(color='#888'),
                        legendgroup='dd'
                    ), row=3, col=1)

                    # Layout Updates
                    use_log = st.checkbox("Log Scale (NAV Chart)", key="lab_log_scale_drill")
                    fig_lab.update_layout(**layout_args)
                    fig_lab.update_yaxes(title_text="Allocation", range=[0, 1], tickformat='.0%', row=1, col=1, gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
                    fig_lab.update_yaxes(range=[0, 1.2], showgrid=False, showticklabels=False, row=1, col=1, secondary_y=True)
                    fig_lab.update_yaxes(title_text="NAV ($)", type="log" if use_log else "linear", row=2, col=1, gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
                    fig_lab.update_yaxes(title_text="Drawdown", tickformat='.0%', row=3, col=1, gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
                    fig_lab.update_xaxes(**theme['xaxis'], row=1, col=1)
                    fig_lab.update_xaxes(**theme['xaxis'], row=2, col=1)
                    fig_lab.update_xaxes(**theme['xaxis'], row=3, col=1)

                    st.plotly_chart(fig_lab, width='stretch')

                    # Metrics Table
                    st.divider()
                    st.markdown("**Performance Metrics Summary (Selected Window)**")
                    metrics_comp = pd.DataFrame({
                        'Strategy': lab_results_disp['metrics'],
                        'Benchmark': bench_results_disp['metrics']
                    }).T
                    for col in ['CAGR', 'Volatility', 'Max Drawdown']:
                        if col in metrics_comp.columns: metrics_comp[col] = metrics_comp[col].apply(lambda x: f"{x:.2%}")
                    for col in ['Sharpe', 'Sortino', 'Calmar']:
                        if col in metrics_comp.columns: metrics_comp[col] = metrics_comp[col].apply(lambda x: f"{x:.2f}")
                    if 'Turnover' in metrics_comp.columns: metrics_comp['Turnover'] = metrics_comp['Turnover'].apply(lambda x: f"{x:.2f}x")
                    st.table(metrics_comp)

            # --- ROLLING ANALYSIS TAB CONTENT ---
            with res_tab_roll:
                # Rolling Analysis configuration
                col_roll1, col_roll2 = st.columns([1, 2])
                with col_roll1:
                    holding_duration = st.selectbox(
                        "Holding Duration", 
                        options=[1, 2, 3, 5, 10], 
                        index=3, 
                        format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}",
                        key="roll_duration"
                    )
                
                # Calculations
                window = holding_duration * 12
                
                def get_rolling_metrics(equity_curve, window_size, risk_free_rate=0.04):
                    # Returns
                    returns = equity_curve.pct_change().dropna()
                    
                    # Rolling CAGR
                    # (End / Start) ^ (1/Years) - 1
                    # We look forward: if I buy at T, what is my return at T+window?
                    rolling_cagr = (equity_curve.shift(-window_size) / equity_curve) ** (12 / window_size) - 1
                    rolling_cagr = rolling_cagr.dropna()
                    
                    # Rolling Volatility (using periodic returns within the window)
                    def calc_vol(x):
                        rets = x.pct_change().dropna()
                        return rets.std() * np.sqrt(12)
                    
                    # This is slow for large datasets, but usually okay for monthly backtests
                    rolling_vol = equity_curve.rolling(window=window_size+1).apply(calc_vol).shift(-window_size).dropna()
                    
                    # Rolling Sharpe
                    rolling_sharpe = (rolling_cagr - risk_free_rate) / rolling_vol
                    
                    # Rolling Sortino
                    def calc_sortino(x):
                        rets = x.pct_change().dropna()
                        # Sortino relative to RF
                        rf_monthly = risk_free_rate / 12.0
                        downside_rets = np.minimum(0, rets - rf_monthly)
                        downside_std = np.sqrt(np.mean(downside_rets**2)) * np.sqrt(12)
                        cagr = (x.iloc[-1] / x.iloc[0]) ** (12 / (len(x)-1)) - 1
                        return (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0

                    rolling_sortino = equity_curve.rolling(window=window_size+1).apply(calc_sortino).shift(-window_size).dropna()
                    
                    # Rolling Max Drawdown
                    def calc_mdd(x):
                        dd = (x / x.cummax()) - 1
                        return dd.min()
                    
                    rolling_mdd = equity_curve.rolling(window=window_size+1).apply(calc_mdd).shift(-window_size).dropna()

                    return {
                        'cagr': rolling_cagr,
                        'vol': rolling_vol,
                        'sharpe': rolling_sharpe,
                        'sortino': rolling_sortino,
                        'mdd': rolling_mdd
                    }

                with st.spinner(f"Calculating rolling metrics for {holding_duration}Y window..."):
                    strat_roll = get_rolling_metrics(lab_results['equity_curve'], window)
                    bench_roll = get_rolling_metrics(benchmark_results['equity_curve'], window)
                    
                    # Intersect dates to ensure alignment
                    common_roll_idx = strat_roll['cagr'].index.intersection(bench_roll['cagr'].index)
                    if common_roll_idx.empty:
                        st.warning("Insufficient data for the selected holding duration.")
                    else:
                        for k in strat_roll: strat_roll[k] = strat_roll[k].loc[common_roll_idx]
                        for k in bench_roll: bench_roll[k] = bench_roll[k].loc[common_roll_idx]
                        
                        # Layout: Charts on left, Table on right
                        col_charts, col_table = st.columns([2, 1])
                        
                        with col_charts:
                            # 1. Rolling Return Chart
                            fig_roll_ret = go.Figure()
                            fig_roll_ret.add_trace(go.Scatter(
                                x=strat_roll['cagr'].index, y=strat_roll['cagr'],
                                name='Strategy', 
                                mode='lines+markers',
                                line=dict(color='#ff6b35', width=2),
                                marker=dict(size=12, opacity=0.01), # Large invisible hit area for single-click
                                hovertemplate='%{y:.1%}<br>%{x|%b %Y}<extra>Strategy</extra>'
                            ))
                            fig_roll_ret.add_trace(go.Scatter(
                                x=bench_roll['cagr'].index, y=bench_roll['cagr'],
                                name='Benchmark', 
                                mode='lines',
                                line=dict(color='#888', dash='dash'),
                                hovertemplate='%{y:.1%}<br>%{x|%b %Y}<extra>Benchmark</extra>'
                            ))
                            fig_roll_ret.update_layout(
                                title=dict(
                                    text=f"Rolling {holding_duration}Y Return (CAGR)",
                                    font=dict(color=theme['font']['color'])
                                ),
                                **{k:v for k,v in layout_args.items() if k not in ['height', 'margin', 'hovermode']},
                                height=400,
                                margin=dict(l=50, r=20, t=50, b=20),
                                hovermode='closest', # Closest hover is much better for click detection than 'x unified'
                                dragmode='pan',
                                clickmode='event+select'
                            )
                            fig_roll_ret.update_yaxes(tickformat='.1%', gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
                            # Add vertical spikeline
                            fig_roll_ret.update_xaxes(
                                **theme['xaxis'],
                                showspikes=True,
                                spikethickness=1,
                                spikedash='dot',
                                spikemode='across',
                                spikesnap='cursor'
                            )
                            # Enable interaction with rotated key
                            reset_idx = st.session_state.get('lab_reset_count', 0)
                            sel_ret = st.plotly_chart(
                                fig_roll_ret, 
                                width='stretch', 
                                on_select="rerun", 
                                selection_mode=["points", "box"],
                                key=f"roll_ret_chart_{holding_duration}_{reset_idx}",
                                config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoom2d', 'pan2d']}
                            )
                            # DEBUG: st.toast(f"Ret Chart: {str(sel_ret)[:50]}...")
                            
                            # Immediate detection logic
                            def process_selection(sel_data, dur):
                                if not sel_data or not isinstance(sel_data, dict): return False
                                # Plotly click events inside on_select can put data in 'selection' or directly in 'points'
                                pts = sel_data.get("selection", {}).get("points", []) or sel_data.get("points", [])
                                if not pts and "selection" in sel_data:
                                    # Handle case where selection might be a list directly
                                    if isinstance(sel_data["selection"], list):
                                        pts = sel_data["selection"]
                                
                                if not pts: return False
                                try:
                                    new_start = pd.to_datetime(pts[0]['x'])
                                    if not st.session_state.get('lab_drilldown_active') or st.session_state.get('lab_drilldown_start') != new_start:
                                        st.session_state.lab_drilldown_active = True
                                        st.session_state.lab_drilldown_start = new_start
                                        st.session_state.lab_drilldown_duration = dur
                                        st.rerun()
                                        return True
                                except Exception:
                                    return False
                                return False

                            process_selection(sel_ret, holding_duration)
                            
                            # 2. Rolling Volatility Chart
                            fig_roll_vol = go.Figure()
                            fig_roll_vol.add_trace(go.Scatter(
                                x=strat_roll['vol'].index, y=strat_roll['vol'],
                                name='Strategy', 
                                mode='lines+markers',
                                line=dict(color='#ff6b35', width=2),
                                marker=dict(size=12, opacity=0.01),
                                hovertemplate='%{y:.1%}<br>%{x|%b %Y}<extra>Strategy</extra>'
                            ))
                            fig_roll_vol.add_trace(go.Scatter(
                                x=bench_roll['vol'].index, y=bench_roll['vol'],
                                name='Benchmark', 
                                mode='lines',
                                line=dict(color='#888', dash='dash'),
                                hovertemplate='%{y:.1%}<br>%{x|%b %Y}<extra>Benchmark</extra>'
                            ))
                            fig_roll_vol.update_layout(
                                title=dict(
                                    text=f"Rolling {holding_duration}Y Volatility",
                                    font=dict(color=theme['font']['color'])
                                ),
                                **{k:v for k,v in layout_args.items() if k not in ['height', 'margin', 'hovermode']},
                                height=350,
                                margin=dict(l=50, r=20, t=50, b=20),
                                hovermode='closest',
                                dragmode='pan',
                                clickmode='event+select'
                            )
                            fig_roll_vol.update_yaxes(tickformat='.1%', gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
                            fig_roll_vol.update_xaxes(
                                **theme['xaxis'],
                                showspikes=True,
                                spikethickness=1,
                                spikedash='dot',
                                spikemode='across',
                                spikesnap='cursor'
                            )
                            sel_vol = st.plotly_chart(
                                fig_roll_vol, 
                                width='stretch', 
                                on_select="rerun", 
                                selection_mode=["points", "box"],
                                key=f"roll_vol_chart_{holding_duration}_{reset_idx}",
                                config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoom2d', 'pan2d']}
                            )
                            process_selection(sel_vol, holding_duration)

                        with col_table:
                            st.markdown(f"**Key Metrics ({holding_duration}Y Rolling)**")
                            
                            def summarize_roll(roll_data):
                                cagr = roll_data['cagr']
                                avg_cagr = cagr.mean()
                                avg_vol = roll_data['vol'].mean()
                                
                                # Ratio of Averages for Sharpe
                                agg_sharpe = (avg_cagr - risk_free_rate) / avg_vol if avg_vol > 0 else 0
                                
                                return pd.Series({
                                    'Average Return': f"{avg_cagr:.2%}",
                                    '$10,000 Becomes': f"${10000 * (1 + avg_cagr)**holding_duration:,.0f}",
                                    'Min Return': f"{cagr.min():.2%}",
                                    '25th Percentile': f"{cagr.quantile(0.25):.2%}",
                                    'Median Return': f"{cagr.median():.2%}",
                                    '75th Percentile': f"{cagr.quantile(0.75):.2%}",
                                    'Max Return': f"{cagr.max():.2%}",
                                    'Volatility': f"{avg_vol:.2%}",
                                    'Sharpe Ratio': f"{agg_sharpe:.2f}",
                                    'Sortino Ratio': f"{roll_data['sortino'].median():.2f}",
                                    'Positive Periods': f"{(cagr > 0).mean():.1%}",
                                    'Max Drawdown': f"{roll_data['mdd'].min():.2%}"
                                })

                            m_strat = summarize_roll(strat_roll)
                            m_bench = summarize_roll(bench_roll)
                            
                            # Add Avg Rebalancing Cost
                            # The full backtest cost is known. For rolling windows, we approximate or use the average rebalance cost per step.
                            # The user asked for "rebalancing cost (Avg)".
                            total_rebal_cost = lab_results['metrics'].get('Rebalancing Cost', 0)
                            n_months = len(lab_results['equity_curve']) - 1
                            avg_monthly_rebal = total_rebal_cost / n_months if n_months > 0 else 0
                            avg_rebal_window = avg_monthly_rebal * window
                            
                            m_strat['Rebalancing Cost (Avg)'] = f"${avg_rebal_window:,.0f}"
                            m_bench['Rebalancing Cost (Avg)'] = "$0" # Benchmark is buy & hold 
                            
                            metrics_df = pd.DataFrame({
                                'Metric': m_strat.index,
                                'Strategy': m_strat.values,
                                'Benchmark': m_bench.values
                            })
                            
                            # Custom HTML Table with Row-Level Tooltips
                            tooltips = {
                                'Average Return': 'Geometric average (CAGR) of all rolling periods.',
                                '$10,000 Becomes': 'Projected value of $10,000 after one holding period based on Avg CAGR.',
                                'Min Return': 'The lowest CAGR observed across all rolling windows.',
                                '25th Percentile': '25% of the rolling periods had a return lower than this value.',
                                'Median Return': 'The middle value of all returns; 50% were higher and 50% were lower.',
                                '75th Percentile': '75% of the rolling periods had a return lower than this value.',
                                'Max Return': 'The highest CAGR observed across all rolling windows.',
                                'Volatility': 'Average annualized standard deviation of returns during the periods.',
                                'Sharpe Ratio': 'Average risk-adjusted return (Excess Return / Volatility).',
                                'Sortino Ratio': 'Downside-risk-adjusted return relative to Risk-Free Rate. Table shows the MEDIAN of rolling Sortino ratios to prevent outlier artifacts.',
                                'Positive Periods': 'The percentage of rolling windows that ended with a positive return.',
                                'Max Drawdown': 'The maximum peak-to-trough decline observed across all rolling windows.',
                                'Rebalancing Cost (Avg)': 'The average estimated cost of trading during the rolling period.'
                            }

                            # Style block
                            style_html = """
<style>
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-family: inherit;
    background: var(--bg-secondary);
    border-radius: 8px;
    overflow: visible;
    color: var(--text-primary);
    margin-top: 10px;
    border: 1px solid var(--border-color);
}
.metrics-table th {
    text-align: left;
    padding: 12px 15px;
    background: var(--bg-tertiary);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-secondary);
    border-bottom: 2px solid var(--border-color);
}
.metrics-table td {
    padding: 10px 15px;
    border-bottom: 1px solid var(--border-color);
    font-size: 0.85rem;
}
.metrics-table tr:hover td {
    background: var(--bg-tertiary);
}
.tooltip-container {
    position: relative;
    cursor: help;
    border-bottom: 1px dotted var(--accent-blue);
    color: var(--accent-blue);
}
.tooltip-container .tooltip-text {
    visibility: hidden;
    width: 200px;
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
    text-align: left;
    border-radius: 6px;
    padding: 8px 12px;
    position: absolute;
    z-index: 999;
    bottom: 125%;
    left: 0;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.75rem;
    font-weight: normal;
    line-height: 1.4;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border-color);
    pointer-events: none;
}
.tooltip-container:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
.val-strat { color: var(--accent-orange); font-weight: 600; }
.val-bench { color: var(--text-secondary); }
</style>
"""
                            
                            rows_html = ""
                            for _, m_row in metrics_df.iterrows():
                                m_name = m_row['Metric']
                                tip = tooltips.get(m_name, "")
                                rows_html += f'<tr><td><span class="tooltip-container">{m_name}<span class="tooltip-text">{tip}</span></span></td><td class="val-strat">{m_row["Strategy"]}</td><td class="val-bench">{m_row["Benchmark"]}</td></tr>'
                            
                            table_html = f"""
<table class="metrics-table">
    <thead>
        <tr>
            <th>Metric</th>
            <th>Strategy</th>
            <th>Benchmark</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>
"""
                            st.markdown(style_html + table_html, unsafe_allow_html=True)
                            
                            # Period summary info
                            if not common_roll_idx.empty:
                                start_dt = common_roll_idx[0].strftime('%b %Y')
                                last_start_dt = common_roll_idx[-1]
                                end_dt = last_start_dt.strftime('%b %Y')
                                
                                # The true analysis end date is the end of the last window
                                analysis_end_dt = (last_start_dt + pd.DateOffset(months=window)).strftime('%b %Y')
                                
                                n_per = len(common_roll_idx)
                                st.markdown(f"""
                                <div style="margin-top: 15px; padding: 12px; border-radius: 8px; background: var(--bg-tertiary); border: 1px solid var(--border-color); font-size: 0.85rem; color: var(--text-secondary);">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">
                                        <span>📅 <b style="color:var(--text-primary);">Start Dates:</b> {start_dt} — {end_dt}</span>
                                        <span>🏁 <b style="color:var(--text-primary);">Analysis End:</b> {analysis_end_dt}</span>
                                    </div>
                                    <div style="text-align: center; color: var(--accent-blue); font-weight: 600;">
                                        📊 Rolling Periods: {n_per}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        # --- STRATEGY & BENCHMARK DETAIL CONTAINER ---
                        st.divider()
                        
                        # Prepare data for the summary
                        all_drivers = []
                        for asset_type in ['EQUITY', 'BONDS', 'GOLD']:
                            drivers = stability_results_map[asset_type]['stable_features']
                            all_drivers.extend(drivers)
                        unique_drivers = sorted(list(set(all_drivers)))
                        drivers_str = ", ".join(unique_drivers)

                        strat_desc_map = {
                            "Max Return": "Tactical momentum approach identifying top performing assets based on predicted returns.",
                            "Min Volatility": "Global Minimum Variance (GMV) optimization focusing on risk and co-movement.",
                            "Min Drawdown": "Regime-aware defense switching based on Macro Stress Score (Aggregate Z-Score).",
                            "Min Loss": "Safety-first approach using Lower 95% Confidence Intervals for high-conviction signals."
                        }
                        
                        model_info = {
                            'EQUITY': 'Random Forest Regressor (Non-Linear Ensemble)',
                            'BONDS': 'ElasticNetCV (L1/L2 Regularized Linear)',
                            'GOLD': 'Simple OLS (Ordinary Least Squares)'
                        }

                        # Render Strategy & Benchmark Detail Summary
                        analysis_summary_html = f"""<div style="padding: 24px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; margin-top: 30px; font-family: 'Inter', sans-serif;">
<h2 style="margin-top: 0; color: var(--accent-blue); font-size: 1.25rem; border-bottom: 2px solid var(--border-color); padding-bottom: 12px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">📋 <span style="letter-spacing: 0.5px;">SYSTEM ARCHITECTURE & METHODOLOGY</span></h2>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
<div style="display: flex; flex-direction: column; gap: 20px;">
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid var(--accent-orange);">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">🚀 Strategy Optimization: {strategy_type}</h3>
<p style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; margin: 0;">
The optimization core utilizes the <b>{strategy_type}</b> engine. {strat_desc_map.get(strategy_type, "")} 
Dynamic allocation is enforced through a <b>constrained optimization</b> framework ensuring weights sum to 100% (Cash inclusive) 
while respecting asset-level bounds (e.g., Equity cap at {max_weights['EQUITY']:.0%}).
</p>
</div>
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid var(--accent-blue);">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">🧬 Heterogeneous Model Architectures</h3>
<div style="display: flex; flex-direction: column; gap: 8px;">
<div style="font-size: 0.8rem; color: var(--text-secondary);">
<b style="color: var(--text-primary);">Equity Core:</b> {model_info['EQUITY']}. Capable of mapping non-linear state interactions across the 126-variable feature space.
</div>
<div style="font-size: 0.8rem; color: var(--text-secondary);">
<b style="color: var(--text-primary);">Fixed Income:</b> {model_info['BONDS']}. Employs L1/L2 regularization via Coordinate Descent for optimal persistence in sparse macro environments.
</div>
<div style="font-size: 0.8rem; color: var(--text-secondary);">
<b style="color: var(--text-primary);">Commodities (Gold):</b> {model_info['GOLD']}. Leverages cointegration-ready parsimony for long-horizon stability.
</div>
</div>
</div>
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">📡 Feature Engineering & Signal Processing</h3>
<p style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; margin: 0;">
Raw macro data undergoes <b>Stationarity Transformations</b> (level to Δlog) based on FRED-MD specifications. Features are سپس normalized via an <b>Expanding-Window Z-Score</b> pipeline to prevent look-ahead bias. Robustness is ensured through <b>Z-Score Winsorization</b> at a 3.0σ threshold, mitigating the impact of exogenous shocks (e.g., GFC, 2020).
</p>
</div>
</div>
<div style="display: flex; flex-direction: column; gap: 20px;">
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">⚖️ Parameters & Operational Constraints</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.82rem; color: var(--text-secondary);">
<div><b style="color: var(--text-primary);">Rebal. Frequency:</b> {lab_freq} Month(s)</div>
<div><b style="color: var(--text-primary);">Trading Friction:</b> {full_params.get('trading_cost_bps', 0)} bps</div>
<div><b style="color: var(--text-primary);">Horizon:</b> {horizon_months}M Forward Return</div>
<div><b style="color: var(--text-primary);">Simulation Span:</b> {len(lab_results['equity_curve'])} Months</div>
<div style="grid-column: span 2; padding-top: 5px; border-top: 1px solid var(--border-color);">
<b style="color: var(--text-primary);">Active Macro Drivers:</b> <code style="font-size: 0.75rem; background: rgba(0,0,0,0.2); color: var(--accent-blue); padding: 2px 4px;">{drivers_str[:120]}...</code>
</div>
</div>
</div>
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid #f43f5e;">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">📊 Benchmark: Buy & Hold Configuration</h3>
<div style="display: flex; justify-content: space-around; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 6px; text-align: center;">
<div><div style="font-size: 0.75rem; color: var(--text-secondary);">EQUITY (SPY)</div><div style="font-size: 1.1rem; font-weight: 700; color: #ff6b35;">{benchmark_weights['EQUITY']:.0%}</div></div>
<div><div style="font-size: 0.75rem; color: var(--text-secondary);">FIXED INC. (AGG)</div><div style="font-size: 1.1rem; font-weight: 700; color: #4da6ff;">{benchmark_weights['BONDS']:.0%}</div></div>
<div><div style="font-size: 0.75rem; color: var(--text-secondary);">GOLD (GLD)</div><div style="font-size: 1.1rem; font-weight: 700; color: #ffd700;">{benchmark_weights['GOLD']:.0%}</div></div>
</div>
</div>
<div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; border-left: 4px solid #eab308;">
<h3 style="font-size: 0.95rem; color: var(--text-primary); margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">🛡️ Backtest Protocol (Walk-Forward)</h3>
<p style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; margin: 0;">
Validation employs a <b>Recursive Out-of-Sample (OOS) Walk-Forward</b> methodology. Each period's prediction is generated from a model trained solely on historical data up to that point (Expanding Window), maintaining a minimum <b>240-month training anchor</b> to ensure convergence of state coefficients. Execution assumes an initial capital of <b>${initial_capital:,.0f}</b>.
</p>
</div>
</div>
</div>
</div>"""
                        st.markdown(analysis_summary_html, unsafe_allow_html=True)

        else:
            st.info("💡 Select a strategy and click the button above to start the out-of-sample simulation. The process may take up to 60 seconds on the first run.")

    with tab7:
        st.markdown('<div class="panel-header">SYSTEM SPECIFICATIONS & DOCUMENTATION</div>', unsafe_allow_html=True)
        try:
            with open("specs.md", "r") as f:
                readme_content = f.read()
            st.markdown(readme_content)
        except Exception as e:
            st.error(f"Error loading specs.md: {e}")


if __name__ == "__main__":
    main()