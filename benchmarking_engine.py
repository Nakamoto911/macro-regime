import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
from xgboost import XGBRegressor

# Attempt to import tensorflow for LSTM, but keep it optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class KerasLSTMRegressor(BaseEstimator, RegressorMixin):
    """
    Shallow LSTM for small-sample macro data.
    """
    def __init__(self, units=16, dropout=0.2, epochs=50, batch_size=32):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(self.units, input_shape=input_shape),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y):
        if not TF_AVAILABLE:
            return self
            
        # Reshape X for LSTM: (samples, timesteps, features)
        # Here we treat each observation as a single timestep sequence for simplicity
        # or we could use a sliding window, but standard tabular -> LSTM often uses (N, 1, F)
        if hasattr(X, 'values'):
            X = X.values

        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        self.model = self._build_model((1, X.shape[1]))
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        if not TF_AVAILABLE or self.model is None:
            return np.zeros(len(X))
            
        if hasattr(X, 'values'):
            X = X.values

        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_reshaped).flatten()

from data_utils import load_fred_md_data, load_asset_data, prepare_macro_features, compute_forward_returns

# Suppress convergence warnings for cleaner output during benchmark
warnings.filterwarnings("ignore")

# ==========================================
# 1. Preprocessing & Transformers
# ==========================================

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Caps features at a specific Z-score threshold.
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

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.means_ is None or self.stds_ is None:
            return X_df
        
        lower = self.means_ - self.threshold * self.stds_
        upper = self.means_ + self.threshold * self.stds_
        
        # We use axis=1 to ensure alignment on column names
        return X_df.clip(lower=lower, upper=upper, axis=1)


class FactorStripper(BaseEstimator, TransformerMixin):
    """
    Orthogonalizes features against common macro drivers (Inflation, Growth, etc.)
    Output format: {Feature}_resid_{Driver}
    """
    def __init__(self, drivers=['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS']):
        self.drivers = drivers
        self.models_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        available_drivers = [d for d in self.drivers if d in X_df.columns]
        
        for driver in available_drivers:
            # We need to handle NaNs for each driver-feature pair independently
            self.models_[driver] = {}
            driver_data = X_df[driver]
            
            for col in X_df.columns:
                if col == driver:
                    continue
                
                # Regress col on driver, but ONLY where both have data
                pair_df = X_df[[col, driver]].dropna()
                if len(pair_df) < 24: # Need minimum history to fit a relationship
                    continue
                    
                lr = LinearRegression()
                lr.fit(pair_df[[driver]].values, pair_df[col].values)
                self.models_[driver][col] = lr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        new_cols = {}
        
        for driver, models in self.models_.items():
            if driver not in X_df.columns:
                continue
            
            driver_series = X_df[driver]
            # Only predict where driver has data
            valid_driver_mask = driver_series.notna()
            if not valid_driver_mask.any():
                continue
                
            driver_input = driver_series.loc[valid_driver_mask].values.reshape(-1, 1)
            
            for col, model in models.items():
                if col not in X_df.columns:
                    continue
                
                # Initialize with NaNs
                resid = np.full(len(X_df), np.nan)
                
                # Predict only for valid driver dates
                preds = model.predict(driver_input)
                
                # Calculate residue: Actual - Predicted
                # Note: Actual might still be NaN at some of these dates, which is fine
                resid[valid_driver_mask] = X_df[col].loc[valid_driver_mask].values - preds
                
                new_cols[f"{col}_resid_{driver}"] = resid
                
        if new_cols:
            resids_df = pd.DataFrame(new_cols, index=X_df.index)
            combined = pd.concat([X_df, resids_df], axis=1)
            # Deduplicate columns (keep first)
            return combined.loc[:, ~combined.columns.duplicated()]
        return X_df


class PointInTimeFactorStripper(BaseEstimator, TransformerMixin):
    """
    Orthogonalizes features against macro drivers using ONLY historically available data.
    Implements expanding window estimation for each time point.
    """
    
    def __init__(self, drivers=['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS'], 
                 min_history: int = 60,
                 update_frequency: int = 12):
        """
        Args:
            drivers: List of driver variable names
            min_history: Minimum observations before fitting (default: 5 years)
            update_frequency: Re-estimate coefficients every N periods (efficiency)
        """
        self.drivers = drivers
        self.min_history = min_history
        self.update_frequency = update_frequency
        self.coefficient_history_ = {}  # date -> {driver -> {feature -> coef}}
        
    def fit_transform_pit(self, X: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
        """
        Point-in-time fit and transform in a single pass.
        """
        X = X.sort_index()
        dates = X.index
        n = len(dates)
        
        available_drivers = [d for d in self.drivers if d in X.columns]
        if not available_drivers:
            return X
        
        # Pre-allocate result arrays for efficiency
        feature_cols = [c for c in X.columns if c not in available_drivers]
        resid_col_names = [f"{col}_resid_{drv}" 
                          for col in feature_cols 
                          for drv in available_drivers]
        
        result_data = np.full((n, len(resid_col_names)), np.nan)
        
        current_coefficients = {}  # driver -> {feature -> (coef, intercept)}
        last_update_idx = -self.update_frequency - 1 # Force update on first valid point
        
        # Pre-extract values for fast access
        X_values = X.values
        driver_indices = [X.columns.get_loc(d) for d in available_drivers]
        feature_indices = [X.columns.get_loc(f) for f in feature_cols]
        
        for t_idx in range(self.min_history, n):
            # Check if we need to update coefficients
            if t_idx - last_update_idx >= self.update_frequency:
                if t_idx % 60 == 0: # Log every 5 years in terminal
                    print(f"DEBUG: PIT Orthogonalization Progress: {t_idx}/{n} steps")
                # Fit on data [0, t_idx - 1] (strictly historical)
                X_train = X.iloc[:t_idx]
                current_coefficients = self._fit_coefficients(X_train, available_drivers)
                last_update_idx = t_idx
                
                # Store for diagnostics/reproducibility
                self.coefficient_history_[dates[t_idx]] = current_coefficients.copy()
            
            # Apply current coefficients to transform X_t using pre-extracted values
            col_idx = 0
            for f_idx in feature_indices:
                feat_name = feature_cols[feature_indices.index(f_idx)]
                actual_val = X_values[t_idx, f_idx]
                
                for d_idx in driver_indices:
                    drv_name = available_drivers[driver_indices.index(d_idx)]
                    if drv_name in current_coefficients and feat_name in current_coefficients[drv_name]:
                        coef, intercept = current_coefficients[drv_name][feat_name]
                        driver_val = X_values[t_idx, d_idx]
                        
                        if not np.isnan(driver_val) and not np.isnan(actual_val):
                            predicted = coef * driver_val + intercept
                            result_data[t_idx, col_idx] = actual_val - predicted
                    col_idx += 1
            
            # Progress reporting
            if progress_cb and t_idx % 12 == 0:
                progress_cb(t_idx / n, dates[t_idx])
        
        # Construct result DataFrame
        resid_df = pd.DataFrame(result_data, index=dates, columns=resid_col_names)
        
        # Combine with original (keeping original columns)
        combined = pd.concat([X, resid_df], axis=1)
        return combined.loc[:, ~combined.columns.duplicated()]
    
    def _fit_coefficients(self, X_train: pd.DataFrame, drivers: list) -> dict:
        """Fit orthogonalization coefficients on training data."""
        coefficients = {}
        
        for driver in drivers:
            coefficients[driver] = {}
            driver_data = X_train[driver].dropna()
            
            if len(driver_data) < 24:
                continue
            
            for col in X_train.columns:
                if col == driver or col in drivers:
                    continue
                
                # Pairwise complete observations
                pair = X_train[[driver, col]].dropna()
                if len(pair) < 24:
                    continue
                
                # Simple linear regression
                x = pair[driver].values
                y = pair[col].values
                
                # Closed-form OLS
                x_mean = x.mean()
                y_mean = y.mean()
                numer = np.sum((x - x_mean) * (y - y_mean))
                denom = np.sum((x - x_mean)**2)
                coef = numer / (denom + 1e-10)
                intercept = y_mean - coef * x_mean
                
                coefficients[driver][col] = (coef, intercept)
        
        return coefficients
    
    def transform(self, X):
        """Standard transform method required by Sklearn (uses last fitted state)."""
        # This is for the 'live' prediction where we use the most recent coefficients
        # In PIT backtest, we should use fit_transform_pit
        X_df = pd.DataFrame(X)
        if not self.coefficient_history_:
            return X_df
            
        last_date = sorted(self.coefficient_history_.keys())[-1]
        current_coefficients = self.coefficient_history_[last_date]
        
        new_cols = {}
        for driver, features in current_coefficients.items():
            if driver not in X_df.columns: continue
            for col, (coef, intercept) in features.items():
                if col not in X_df.columns: continue
                new_cols[f"{col}_resid_{driver}"] = X_df[col] - (coef * X_df[driver] + intercept)
        
        if new_cols:
            resids_df = pd.DataFrame(new_cols, index=X_df.index)
            combined = pd.concat([X_df, resids_df], axis=1)
            return combined.loc[:, ~combined.columns.duplicated()]
        return X_df

    def get_coefficient_stability(self) -> pd.DataFrame:
        """Analyze how orthogonalization coefficients evolve over time."""
        if not self.coefficient_history_:
            return pd.DataFrame()
        
        records = []
        for date, drivers in self.coefficient_history_.items():
            for driver, features in drivers.items():
                for feature, (coef, _) in features.items():
                    records.append({
                        'date': date,
                        'driver': driver,
                        'feature': feature,
                        'coefficient': coef
                    })
        
        return pd.DataFrame(records)

def select_features_elastic_net(y: pd.Series, X: pd.DataFrame, 
                                 n_iterations: int = 1, # Default reduced to 1 for speed
                                 sample_fraction: float = 0.9, # Higher fraction for single run
                                 threshold: float = 0.0, # Threshold irrelevant if n_iter=1 (coef!=0)
                                 l1_ratio: float = 0.5,
                                 st_progress=None) -> tuple:
    """
    Feature selection. 
    If n_iterations=1 (Fast Mode), uses standard ElasticNet (SelectFromModel).
    If n_iterations>1 (Stable Mode), uses Bootstrapped Stability Selection.
    """
    
    # Pre-clean
    X = X.dropna(axis=1)
    if X.empty:
        return [], pd.Series(dtype=float)

    # 1. Univariate Screening (Keep top 100 to protect Solver)
    if X.shape[1] > 100:
        corrs = X.corrwith(y).abs().fillna(0)
        top_cols = corrs.sort_values(ascending=False).head(100).index.tolist()
        X_screened = X[top_cols]
    else:
        X_screened = X

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_screened), columns=X_screened.columns, index=X_screened.index)
    
    # --- FAST PATH: Single Iteration (Standard Lasso/Net) ---
    if n_iterations == 1:
        # Standard fit on full data
        model = ElasticNet(alpha=0.01, l1_ratio=l1_ratio, max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_scaled, y)
        
        # Select features with non-zero coefficients
        selected = X_screened.columns[model.coef_ != 0].tolist()
        
        # Fallback if nothing selected
        if not selected:
            # Pick max coef
            best_idx = np.argmax(np.abs(model.coef_))
            selected = [X_screened.columns[best_idx]]
            
        # Return mock probabilities (1.0 for selected)
        probs = pd.Series(0, index=X.columns)
        probs[selected] = 1.0
        return selected, probs

    # --- STABLE PATH: Bootstrap (Slower, for Diagnostics) ---
    selection_counts = pd.Series(0, index=X_screened.columns)
    
    for i in range(n_iterations):
        if st_progress and i % 5 == 0: # Reduce UI updates
            st_progress.write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Stability Bootstrap {i+1}/{n_iterations}...")
            
        indices = np.random.choice(len(y), size=int(len(y) * sample_fraction), replace=False)
        y_sample = y.iloc[indices]
        X_sample = X_scaled.iloc[indices]
        
        model = ElasticNet(alpha=0.01, l1_ratio=l1_ratio, max_iter=1000, tol=1e-3)
        model.fit(X_sample, y_sample)
        selection_counts[model.coef_ != 0] += 1
        
    selection_probs = selection_counts / n_iterations
    selected = selection_probs[selection_probs > threshold].index.tolist()
    
    if not selected and not selection_probs.empty:
        selected = [selection_probs.idxmax()]
            
    return selected, selection_probs

# ==========================================
# 2. Custom Model: Regime Switching
# ==========================================

class RegimeSwitchingModel(BaseEstimator, RegressorMixin):
    """
    Switches between two models based on a regime indicator (e.g., Inflation Moving Average).
    """
    def __init__(self, model_low, model_high, regime_col_idx=0, threshold=0.04):
        self.model_low = model_low
        self.model_high = model_high
        self.regime_col_idx = regime_col_idx
        self.threshold = threshold
        self.trained_low = None
        self.trained_high = None

    def fit(self, X, y):
        regime_vals = X[:, self.regime_col_idx]
        
        mask_high = regime_vals > self.threshold
        mask_low = ~mask_high
        
        self.trained_low = clone(self.model_low)
        self.trained_high = clone(self.model_high)
        
        if mask_low.any():
            self.trained_low.fit(X[mask_low], y[mask_low])
        if mask_high.any():
            self.trained_high.fit(X[mask_high], y[mask_high])
            
        return self

    def predict(self, X):
        regime_vals = X[:, self.regime_col_idx]
        preds = np.zeros(len(X))
        
        mask_high = regime_vals > self.threshold
        mask_low = ~mask_high
        
        if mask_low.any():
            preds[mask_low] = self.trained_low.predict(X[mask_low])
        if mask_high.any():
            preds[mask_high] = self.trained_high.predict(X[mask_high])
            
        return preds

# ==========================================
# 3. Validation Logic (Purged Walk-Forward)
# ==========================================

def run_benchmarking_engine(X: pd.DataFrame, y: pd.Series, start_idx: int, step: int = 12, horizon: int = 12):
    """
    Strict Purged Walk-Forward Validation (V2.0 Pipeline).
    """
    from data_utils import MacroFeatureExpander
    
    models = {
        "Baseline (Mean)": DummyRegressor(strategy='mean'),
        "Simple OLS": LinearRegression(),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    }
    
    if TF_AVAILABLE:
        models["LSTM"] = KerasLSTMRegressor(epochs=20, batch_size=32)
    
    results = {name: [] for name in models.keys()}
    actuals = []
    
    # Pre-select potential drivers for FactorStripper (Big 4)
    big4 = ['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS']
    
    for i in range(start_idx, len(X) - 1, step):
        # 1. Define Training Set (Purged)
        purge_idx = i - horizon
        if purge_idx <= 24: # Minimum training size
            continue
            
        train_X_raw = X.iloc[:purge_idx]
        train_y_raw = y.iloc[:purge_idx].dropna()
        # Align train_X to y
        train_X_raw = train_X_raw.loc[train_y_raw.index]
        
        test_X_raw = X.iloc[i : min(i + step, len(X))]
        test_y = y.iloc[i : min(i + step, len(X))]
        
        if test_y.empty:
            break
            
        # --- NEW V2.0 PIPELINE ---
        
        # A. Orthogonalization (Fit on purged training)
        stripper = FactorStripper(drivers=big4)
        stripper.fit(train_X_raw)
        X_train_ortho = stripper.transform(train_X_raw)
        X_test_ortho = stripper.transform(test_X_raw)
        
        # B. Systematic Expansion
        expander = MacroFeatureExpander()
        X_train_expanded = expander.transform(X_train_ortho)
        X_test_expanded = expander.transform(X_test_ortho)
        
        # Synchronize indices after expansion drops
        common_train_idx = X_train_expanded.index.intersection(train_y_raw.index)
        X_train_prep = X_train_expanded.loc[common_train_idx]
        y_train_prep = train_y_raw.loc[common_train_idx]
        
        X_test_prep = X_test_expanded # No need to drop test yet, we'll align preds
        
        # C. Stability Selection
        selected_feats, _ = select_features_elastic_net(
            y_train_prep, X_train_prep,
            threshold=0.6,
            l1_ratio=0.5
        )
        if not selected_feats:
            selected_feats = X_train_prep.columns[:10].tolist() # Fallback
            
        X_train_sel = X_train_prep[selected_feats]
        X_test_sel = X_test_prep[selected_feats]
        
        # D. Winsorization
        win = Winsorizer(threshold=3.0)
        X_train_final = win.fit_transform(X_train_sel)
        X_test_final = win.transform(X_test_sel)
        
        # --- END V2.0 PIPELINE ---
        
        actual_vals = test_y.loc[test_y.index.intersection(X_test_expanded.index)].values.flatten()
        actuals.extend(actual_vals)
        
        # Fit & Predict Each Model
        for name, model in models.items():
            m = clone(model)
            m.fit(X_train_final, y_train_prep.values.ravel())
            
            # Re-align test_X_final to valid indices
            valid_test_idx = X_test_expanded.index.intersection(test_y.index)
            if valid_test_idx.empty:
                continue
            
            # Find integer positions for valid_test_idx in X_test_expanded
            # Since X_test_final is a numpy array from transform, we need to index it correctly
            test_positions = [X_test_expanded.index.get_loc(idx) for idx in valid_test_idx]
            X_test_input = X_test_final.iloc[test_positions]
            
            preds = m.predict(X_test_input)
            preds = np.clip(preds, -0.30, 0.30)
            results[name].extend(preds)
            
    # Calculate Metrics
    summary = []
    for name, preds in results.items():
        preds = np.array(preds)
        valid_actuals = np.array(actuals[:len(preds)])
        
        if len(preds) == 0:
            continue
            
        # IC (Information Coefficient)
        ic, _ = spearmanr(valid_actuals, preds)
        # RMSE
        rmse = np.sqrt(mean_squared_error(valid_actuals, preds))
        # Hit Rate
        hit_rate = (np.sign(valid_actuals) == np.sign(preds)).mean()
        
        summary.append({
            "Model": name,
            "OOS IC (Corr)": ic,
            "OOS Hit Rate": hit_rate,
            "OOS RMSE": rmse
        })
        
    df_summary = pd.DataFrame(summary).sort_values("OOS IC (Corr)", ascending=False)
    return df_summary, models
            
    # Calculate Metrics
    summary = []
    for name, preds in results.items():
        preds = np.array(preds)
        valid_actuals = np.array(actuals[:len(preds)])
        
        if len(preds) == 0:
            continue
            
        mse = mean_squared_error(valid_actuals, preds)
        rmse = np.sqrt(mse)
        ic, _ = spearmanr(valid_actuals, preds)
        
        summary.append({
            "Model": name,
            "OOS IC (Corr)": ic,
            "OOS RMSE": rmse
        })
        
    df_summary = pd.DataFrame(summary).sort_values("OOS IC (Corr)", ascending=False)
    return df_summary, models

# ==========================================
# 4. Output Formatter
# ==========================================

def display_current_regime_model(name, model, X, y, asset_name):
    """
    Fits model on the FULL dataset using the V2.0 pipeline and displays its parameters.
    """
    from data_utils import MacroFeatureExpander
    print(f"\n>>> FINAL MODEL DETAILS: {asset_name} ({name})")
    
    # 1. Pipeline
    big4 = ['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS']
    stripper = FactorStripper(drivers=big4)
    stripper.fit(X)
    X_ortho = stripper.transform(X)
    
    expander = MacroFeatureExpander()
    X_expanded = expander.transform(X_ortho)
    
    common_idx = X_expanded.index.intersection(y.index)
    X_prep = X_expanded.loc[common_idx]
    y_prep = y.loc[common_idx]
    
    # Selection
    selected_feats, _ = select_features_elastic_net(y_prep, X_prep)
    X_sel = X_prep[selected_feats]
    
    # Winsorize
    win = Winsorizer(threshold=3.0)
    X_final = win.fit_transform(X_sel)
    
    # Fit
    m = clone(model)
    m.fit(X_final, y_prep.values.ravel())
    
    feature_names = selected_feats
    
    if hasattr(m, 'intercept_') and hasattr(m, 'coef_'):
        intercept = m.intercept_
        coefs = m.coef_
        
        equation = f"Return = {intercept:.4f}"
        sorted_indices = np.argsort(np.abs(coefs))[::-1]
        for idx in sorted_indices[:10]:
            if np.abs(coefs[idx]) > 0.0001:
                equation += f" + ({coefs[idx]:.4f} * {feature_names[idx]})"
        print(f"EQUATION:\n{equation}")
        
    elif hasattr(m, 'feature_importances_'):
        importances = m.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("TOP 10 VARIABLES (FEATURE IMPORTANCE):")
        for idx in sorted_indices[:10]:
            print(f"- {feature_names[idx]}: {importances[idx]:.4f}")
            
    elif name == "LSTM":
        print("Architecture: Shallow LSTM (1 Layer, 16 Units, 20 Epochs)")
        print("Sequence Dependency: T-1 sequence mapping to Forward 12M Return.")

    elif isinstance(m, DummyRegressor):
        print(f"EQUATION: Return = {y_prep.mean():.4f} (Historical Mean)")

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # Load Real Data
    print("Loading Real Data from data_utils.py pipeline...")
    # NOTE: load_fred_md_data now applies stationarity
    macro_data = load_fred_md_data()
    asset_prices = load_asset_data()
    
    if macro_data.empty or asset_prices.empty:
        print("Error: Could not load real data.")
    else:
        # Align data
        common_idx = macro_data.index.intersection(asset_prices.index)
        macro_data = macro_data.loc[common_idx]
        asset_prices = asset_prices.loc[common_idx]
        
        # Prepare Targets (12-month horizon)
        horizon = 12
        y_forward = compute_forward_returns(asset_prices, horizon_months=horizon)
        
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            print(f"\n" + "="*60)
            print(f" ASSET CLASS: {asset}")
            print("="*60)
            
            y_asset = y_forward[asset].dropna()
            X_asset = macro_data.loc[y_asset.index.intersection(macro_data.index)]
            y_asset = y_asset.loc[X_asset.index]
            
            # Start benchmark after matching min length
            benchmark_table, model_templates = run_benchmarking_engine(
                X_asset, y_asset, 
                start_idx=240, 
                step=12, 
                horizon=horizon
            )
            
            print("\nOOS PERFORMANCE LEADERBOARD:")
            print(benchmark_table.to_string(index=False))
            
            if not benchmark_table.empty:
                winner_name = benchmark_table.iloc[0]["Model"]
                winner_model = model_templates[winner_name]
                
                display_current_regime_model(
                    winner_name, 
                    winner_model, 
                    X_asset, 
                    y_asset, 
                    asset
                )
            else:
                print(f"No results for {asset}")
