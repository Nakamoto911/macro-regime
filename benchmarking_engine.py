import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNetCV
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
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        self.model = self._build_model((1, X.shape[1]))
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        if not TF_AVAILABLE or self.model is None:
            return np.zeros(len(X))
            
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

def select_features_elastic_net(y: pd.Series, X: pd.DataFrame, 
                                 n_iterations: int = 20,
                                 sample_fraction: float = 0.7,
                                 threshold: float = 0.6,
                                 l1_ratio: float = 0.5,
                                 st_progress=None) -> tuple:
    """
    Implement Stability Selection via bootstrapping.
    """
    from sklearn.linear_model import ElasticNet
    
    # Pre-clean
    X = X.dropna(axis=1)
    if X.empty:
        return [], pd.Series(dtype=float)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    selection_counts = pd.Series(0, index=X.columns)
    
    progress_container = None
    if st_progress:
        progress_container = st_progress.empty()
    
    for i in range(n_iterations):
        if progress_container:
            progress_container.write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Stability Bootstrap {i+1}/{n_iterations}...")
            
        indices = np.random.choice(len(y), size=int(len(y) * sample_fraction), replace=False)
        y_sample = y.iloc[indices]
        X_sample = X_scaled.iloc[indices]
        
        # Reduced max_iter for faster selection-only fitting
        model = ElasticNet(alpha=0.01, l1_ratio=l1_ratio, max_iter=1000, tol=1e-3)
        model.fit(X_sample, y_sample)
        selection_counts[model.coef_ != 0] += 1
        
    # 2. Calculate Probabilities
    selection_probs = selection_counts / n_iterations
    
    # 3. Apply Threshold
    selected = selection_probs[selection_probs > threshold].index.tolist()
    
    # 4. Fallback Logic: ensure at least one feature
    if not selected and not selection_probs.empty:
        selected = [selection_probs.idxmax()]
    
    # Deduplicate (preserve order where possible)
    final_selected = []
    for f in selected:
        if f not in final_selected:
            final_selected.append(f)
            
    return final_selected, selection_probs

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
            X_test_input = X_test_final[test_positions]
            
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
        
        summary.append({
            "Model": name,
            "OOS IC (Corr)": ic,
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
