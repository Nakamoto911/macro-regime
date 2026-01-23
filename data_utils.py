import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os
import time
import json
from yahooquery import Ticker
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS


# ============================================================
# Precomputed Feature Loading (Optimization Layer)
# ============================================================

PRECOMPUTED_DIR = 'precomputed'


def load_precomputed_features() -> pd.DataFrame:
    """
    Load precomputed PIT-expanded features if available.

    This is the primary optimization entry point. If precomputed data exists,
    loading takes ~3 seconds instead of 40+ minutes of runtime PIT computation.

    Returns:
        DataFrame with precomputed features, or None if not available
    """
    path = f"{PRECOMPUTED_DIR}/pit_expanded_features.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_precomputed_metadata() -> dict:
    """Load metadata about precomputed features."""
    path = f"{PRECOMPUTED_DIR}/metadata.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def is_precomputed_fresh(max_age_days: int = 45) -> bool:
    """
    Check if precomputed data is fresh enough.

    FRED-MD updates monthly, so 45 days is a reasonable threshold.
    """
    metadata = load_precomputed_metadata()
    if metadata is None:
        return False

    from datetime import datetime, timedelta
    try:
        created = datetime.fromisoformat(metadata['created'])
        return (datetime.now() - created) < timedelta(days=max_age_days)
    except (KeyError, ValueError):
        return False


def get_precomputed_date_range() -> tuple:
    """Get the date range covered by precomputed features."""
    metadata = load_precomputed_metadata()
    if metadata is None:
        return None, None

    try:
        start = pd.to_datetime(metadata['date_range'][0])
        end = pd.to_datetime(metadata['date_range'][1])
        return start, end
    except (KeyError, ValueError):
        return None, None

try:
    import streamlit as st
    def cache_data_wrapper(func):
        return st.cache_data(ttl=86400, show_spinner="Fetching Data...") (func)
except ImportError:
    def cache_data_wrapper(func):
        return func

def compute_forward_returns(prices: pd.DataFrame, horizon_months: int = 12, 
                            macro_data: pd.DataFrame = None, 
                            vol_scale: bool = True, 
                            excess_return: bool = True) -> pd.DataFrame:
    """
    Compute Volatility-Scaled Excess Return (VSER) for each asset.
    
    Target: Z = (R - Rf) / Sigma
    - R: Annualized forward return
    - Rf: Risk-free rate (FEDFUNDS) at start of period
    - Sigma: 12-month rolling annualized volatility at start of period
    """
    log_prices = np.log(prices)
    forward_log_return = log_prices.shift(-horizon_months) - log_prices
    annualized_return = forward_log_return / (horizon_months / 12)
    
    target = annualized_return.copy()
    
    if excess_return:
        if macro_data is not None and 'FEDFUNDS' in macro_data.columns:
            # CRITICAL FIX: Divide by 100 to convert percentage (e.g. 5.25) to decimal (0.0525)
            # Align Rf with asset price index
            rf = macro_data['FEDFUNDS'].reindex(prices.index).ffill() / 100.0
            for col in target.columns:
                target[col] = target[col] - rf
        else:
            print("Warning: FEDFUNDS not found for excess return calculation. Using nominal returns.")
            
    if vol_scale:
        monthly_returns = prices.pct_change()
        for col in target.columns:
            # 12-month rolling annualized volatility
            vol = monthly_returns[col].rolling(12).std() * np.sqrt(12)
            target[col] = target[col] / vol
            
    return target


def apply_transformation(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply McCracken & Ng (2016) transformation codes.
    1: Level
    2: First Difference
    3: Second Difference
    4: Log
    5: Log Difference
    6: Second Log Difference
    7: Pct Change Difference
    """
    if tcode == 1:
        return series
    elif tcode == 2:
        return series.diff()
    elif tcode == 3:
        return series.diff().diff()
    elif tcode == 4:
        return np.log(series.replace(0, np.nan))
    elif tcode == 5:
        return np.log(series.replace(0, np.nan)).diff()
    elif tcode == 6:
        return np.log(series.replace(0, np.nan)).diff().diff()
    elif tcode == 7:
        return series.pct_change().diff()
    else:
        return series


class MacroFeatureExpander:
    """
    Expands a stationary macro matrix into a high-dimensional feature space.
    Includes Slopes, Lags, Impulse, Volatility, and Symbolic Ratios.
    """
    def __init__(self, slope_windows=[3, 6, 9, 12, 18, 24], lag_windows=[1, 3, 6]):
        self.slope_windows = slope_windows
        self.lag_windows = lag_windows

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate expanded feature set.
        """
        expanded_list = []
        
        # 1. Base Symbolic Ratios
        if 'M2SL' in X.columns and 'INDPRO' in X.columns:
            expanded_list.append((X['M2SL'] / (X['INDPRO'] + 1e-9)).rename('RATIO_M2_GROWTH'))
        if 'CPIAUCSL' in X.columns and 'FEDFUNDS' in X.columns:
            expanded_list.append((X['CPIAUCSL'] / (X['FEDFUNDS'] + 1e-9)).rename('RATIO_CPI_FEDFUNDS'))
        if 'GS10' in X.columns and 'CPIAUCSL' in X.columns:
            expanded_list.append((X['GS10'] / (X['CPIAUCSL'] + 1e-9)).rename('RATIO_GS10_CPI'))
        if 'UNRATE' in X.columns and 'PAYEMS' in X.columns:
            expanded_list.append((X['UNRATE'] / (X['PAYEMS'] + 1e-9)).rename('RATIO_UNRATE_PAYEMS'))
        if 'INDPRO' in X.columns and 'PAYEMS' in X.columns:
            expanded_list.append((X['INDPRO'] / (X['PAYEMS'] + 1e-9)).rename('RATIO_PROD'))

        for col in X.columns:
            series = X[col]
            expanded_list.append(series.rename(col))
            
            # 2. Slopes (Momentum)
            for w in self.slope_windows:
                expanded_list.append(series.diff(w).rename(f'{col}_slope{w}'))
            
            # 3. Lags
            for l in self.lag_windows:
                expanded_list.append(series.shift(l).rename(f'{col}_lag{l}'))
            
            # 4. Impulse (Acceleration)
            slope3 = series.diff(3)
            expanded_list.append((slope3 - slope3.shift(3)).rename(f'{col}_impulse'))
            
            # 5. Volatility
            expanded_list.append(series.rolling(12).std().rename(f'{col}_vol12'))
            
        # 6. Expanding Window Statistics (Vectorized Big Win)
        # Generates expanding Mean, Min, Max for all base features
        expanding = ExpandingWindowFeatures(
            functions=["mean", "min", "max"],
            variables=X.columns.tolist(),
            missing_values='ignore'
        )
        X_expanding = expanding.fit_transform(X.ffill().fillna(0))
        # ExpandingWindowFeatures appends with '_window_{func}' usually, 
        # or just '_mean', '_min', '_max' depending on version.
        # We filter for these new columns.
        new_expanding_cols = [c for c in X_expanding.columns if c not in X.columns]
        if new_expanding_cols:
            expanded_list.append(X_expanding[new_expanding_cols])
            
        features = pd.concat(expanded_list, axis=1)
        # Deduplicate features (keep first)
        unique_features = features.loc[:, ~features.columns.duplicated()]
        return unique_features


class CointegrationEngine:
    """
    Discovers and generates cointegrated 'Spread' features from Raw Levels.
    Captures long-term equilibrium relationships (Level information) to complement Momentum features.
    """
    def __init__(self, manifest_path=f'{PRECOMPUTED_DIR}/cointegration_manifest.json'):
        self.manifest_path = manifest_path
        self.manifest = {}
        # Dynamic Cointegration - Rolling Window Size
        self.rolling_window = 120 # 10 Years
        
        # Targets: Anything Trending
        # We will dynamically populate this from the data, but keep a core list for reference
        self.targets = ['S&P 500', 'SP500', 'EQUITY', 'CPIAUCSL', 'GS10', 'DGS10', 'UNRATE', 'HOUST', 'AAA', 'BAA'] 
        
        # Anchors: Restricted "Macro Majors" to save compute and focus on structural pairs
        self.anchors = ['M2SL', 'GDP', 'GDPC1', 'INDPRO', 'CPIAUCSL', 'PCEPI'] 
        
    def classify_integration_order(self, df_raw: pd.DataFrame) -> dict:
        """
        Classify variables as Stationary (I(0)) or Trending (I(1)) using ADF test.
        Returns dictionary with 'stationary' and 'trending' lists.
        """
        print("Classifying Integration Order (I(0) vs I(1))...")
        stationary = []
        trending = []
        
        numeric_df = df_raw.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 36: # Min history for reliable ADF
                continue
                
            try:
                # ADF Test
                # Null Hypothesis: Non-Stationary (Unit Root)
                # p < 0.05 => Reject Null => Stationary
                res = adfuller(series, maxlag=12, autolag='AIC')
                p_value = res[1]
                
                if p_value < 0.05:
                    stationary.append(col)
                else:
                    trending.append(col)
            except:
                continue
                
        print(f"  Classification: {len(stationary)} Stationary, {len(trending)} Trending")
        return {"stationary": stationary, "trending": trending}

    def fit(self, df_raw: pd.DataFrame):
        """
        Discover cointegrated pairs from raw data.
        df_raw: DataFrame of Raw Levels (Time Series)
        """
        print("Discovering Cointegration Pairs...")
        # 1. Pre-filter & Clean
        numeric_df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        
        # Logarithmic candidates (Strictly positive)
        # Relaxed check: if 95% of data is > 0, we can use log (shutting out 0s)
        log_candidates = []
        for c in numeric_df.columns:
            series = numeric_df[c].dropna()
            if len(series) > 0 and (series > 0).mean() > 0.95:
                log_candidates.append(c)
        
        print(f"  Log Candidates: {len(log_candidates)} found")
        
        # 1.5 Classify Integration Order
        classification = self.classify_integration_order(df_raw)
        trending_vars = set(classification['trending'])
        stationary_vars = set(classification['stationary'])
        
        print(f"  Skipping Stationary Targets: {list(set(self.targets) & stationary_vars)}")
        
        # Filter Universe: Only I(1) variables can be cointegrated
        # Dynamic: Allow ALL trending variables as potential targets
        all_trending = list(trending_vars)
        valid_targets = [t for t in all_trending if t in numeric_df.columns]
        
        # Anchors restricted to Macro Majors
        valid_anchors = [a for a in self.anchors if a in trending_vars and a in numeric_df.columns]
        
        print(f"  Scanning Cointegration: {len(valid_targets)} Targets vs {len(valid_anchors)} Anchors (Rolling OLS)...")
        
        candidates = []
        
        # 2. Pairwise Search
        # Iterate combinations
        for target in valid_targets:
            for anchor in valid_anchors:
                # Map potential names (e.g. S&P 500 might be SP500)
                # For now assume exact match or presence in columns
                t_col = target if target in numeric_df.columns else None
                a_col = anchor if anchor in numeric_df.columns else None
                
                if not t_col or not a_col or t_col == a_col:
                    continue
                    
                # Setup Series
                Y = numeric_df[t_col]
                X = numeric_df[a_col]
                
                # Apply Log if applicable
                use_log = False
                if t_col in log_candidates and a_col in log_candidates:
                    # Safe Log
                    Y = np.log(Y.replace(0, np.nan))
                    X = np.log(X.replace(0, np.nan))
                    use_log = True
                    
                # Align
                common_idx = Y.index.intersection(X.index)
                if len(common_idx) < 60: # Min history
                    continue
                    
                Y_aligned = Y.loc[common_idx]
                X_aligned = X.loc[common_idx]
                
                # 3. Dynamic Cointegration Test (Rolling OLS)
                # Filter NaNs/Infs
                valid_mask = np.isfinite(Y_aligned) & np.isfinite(X_aligned)
                Y_aligned = Y_aligned[valid_mask]
                X_aligned = X_aligned[valid_mask]
                
                if len(Y_aligned) < self.rolling_window + 24:
                     continue

                try:
                    X_exog = sm.add_constant(X_aligned)
                    
                    # Fit Rolling OLS
                    rols = RollingOLS(Y_aligned, X_exog, window=self.rolling_window)
                    rres = rols.fit()
                    
                    params = rres.params
                    if 'const' in params.columns and a_col in params.columns:
                        alpha_t = params['const']
                        beta_t = params[a_col]
                        
                        # Dynamic Residuals
                        spread = Y_aligned - (alpha_t + beta_t * X_aligned)
                        spread_valid = spread.dropna()
                        
                        if len(spread_valid) < 60:
                            continue
                            
                        # ADF Test on Dynamic Residuals
                        adf_result = adfuller(spread_valid, maxlag=12, autolag='AIC')
                        p_value = adf_result[1]
                        
                        # Threshold (0.05)
                        if p_value < 0.05:
                            pair_name = f"spread_{t_col}_{a_col}"
                            candidate_info = {
                                "name": pair_name,
                                "y": t_col,
                                "x": a_col,
                                "is_log": use_log,
                                "p_value": p_value,
                                "window": self.rolling_window,
                                "type": "dynamic_rolling_ols"
                            }
                            candidates.append(candidate_info)
                            # print(f"  Found Dynamic Cointegration: {pair_name} (p={p_value:.4f})")
                        else:
                            if t_col in ['S&P 500', 'EQUITY', 'SP500', 'CPIAUCSL'] and a_col in ['M2SL', 'GDP']:
                                 print(f"  Rejected Dynamic: {t_col} vs {a_col} (p={p_value:.4f})")
                except Exception as e:
                    # print(f"Error testing {t_col} vs {a_col}: {e}")
                    continue
        
        # Post-Search: Top-N Filtering
        # 1. Separate Priority Candidates (e.g. S&P 500)
        priority_targets = ['S&P 500', 'SP500', 'EQUITY', 'CPIAUCSL']
        priority_candidates = []
        other_candidates = []
        
        for c in candidates:
            if c['y'] in priority_targets:
                priority_candidates.append(c)
            else:
                other_candidates.append(c)
                
        # 2. Sort both lists by p-value
        priority_candidates.sort(key=lambda x: x['p_value'])
        other_candidates.sort(key=lambda x: x['p_value'])
        
        # 3. Fill quotas
        # Force keep up to 10 priority pairs
        max_priority = 10
        final_selection = priority_candidates[:max_priority]
        
        # Fill the rest with best remaining
        slots_remaining = 50 - len(final_selection)
        final_selection.extend(other_candidates[:slots_remaining])
        
        # Sort final selection by p-value for consistency
        final_selection.sort(key=lambda x: x['p_value'])
        
        print(f"  Pruning: Kept {len(final_selection)} pairs (Priority Rescued: {len(priority_candidates[:max_priority])}).")
        if final_selection:
            worst_best_p = final_selection[-1]['p_value']
            print(f"  Cutoff P-Value: {worst_best_p:.6f}")
            
        discovered_pairs = {c['name']: {k:v for k,v in c.items() if k != 'name'} for c in final_selection}
        
        self.manifest = discovered_pairs
        self._save_manifest()
        return self

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Spread features based on manifest.
        """
        if not self.manifest:
            self._load_manifest()
            
        if not self.manifest:
            # If still empty after load, and we have raw data, maybe try to fit?
            # Or just return empty
             # For safety in this specific task flow, we check if we should auto-fit
             if not os.path.exists(self.manifest_path):
                 print("  Manifest not found. Running Discovery (fit)...")
                 self.fit(df_raw)
             else:
                 return pd.DataFrame(index=df_raw.index)

        numeric_df = df_raw.apply(pd.to_numeric, errors='coerce')
        spreads = {}
        
        for name, config in self.manifest.items():
            try:
                y_col = config['y']
                x_col = config['x']
                
                if y_col not in numeric_df.columns or x_col not in numeric_df.columns:
                    continue
                    
                Y = numeric_df[y_col]
                X = numeric_df[x_col]
                
                if config.get('is_log', False):
                    # Safety against <= 0
                    Y = np.log(Y.replace(0, np.nan))
                    X = np.log(X.replace(0, np.nan))
                    
                # Dynamic Transform (Must match fit logic)
                window = config.get('window', 120)
                
                # Align
                common_idx = Y.index.intersection(X.index)
                Y_aligned = Y.loc[common_idx]
                X_aligned = X.loc[common_idx]
                
                # Drop NaNs/Infs for calculation
                valid = np.isfinite(Y_aligned) & np.isfinite(X_aligned)
                Y_calc = Y_aligned[valid]
                X_calc = X_aligned[valid]
                
                if len(Y_calc) < window + 12:
                    continue

                # Recalculate Rolling Parameters
                # NOTE: Ideally we would cache these, but for now we recompute on the fly
                # It's reasonably fast for 10-20 pairs.
                X_exog = sm.add_constant(X_calc)
                rols = RollingOLS(Y_calc, X_exog, window=window)
                rres = rols.fit()
                params = rres.params
                
                # Dynamic Spread
                alpha_t = params['const']
                beta_t = params[x_col]
                
                # Re-index to original DF if needed, but here we just use the calculated index
                spread_series = Y_calc - (alpha_t + beta_t * X_calc)
                
                # Reindex back to full df_raw index to handle gaps (forward fill static parameters? No, leave NaN)
                spread_full = spread_series.reindex(df_raw.index)
                
                # Post-Processing: Z-Score (Rolling 60m) to normalize magnitude
                # Use a long robust window to keep it stationary but scaled
                # Standardization is critical for the model to treat it like other % features
                spread_z = (spread_full - spread_full.rolling(60, min_periods=24).mean()) / (spread_full.rolling(60, min_periods=24).std() + 1e-9)
                
                spreads[name] = spread_z
                
            except Exception as e:
                print(f"Failed to generate {name}: {e}")
                
        return pd.DataFrame(spreads, index=df_raw.index)

    def _save_manifest(self):
        if not os.path.exists(PRECOMPUTED_DIR):
            os.makedirs(PRECOMPUTED_DIR)
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
            
    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)


def prepare_macro_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Feature Expansion to Macro Data.
    Optimization: Splits 'Spread' features (Lightweight) vs 'Macro' features (Heavyweight).
    """
    # Identify spread columns (from Cointegration Engine)
    spread_cols = [c for c in macro_data.columns if c.startswith('spread_')]
    macro_cols = [c for c in macro_data.columns if c not in spread_cols]
    
    # 1. Full Expansion for Standard Macro Columns
    # (Generating Slopes, Volatility, Impulse, Expanding Stats)
    df_macro = macro_data[macro_cols]
    expander = MacroFeatureExpander()
    df_macro_expanded = expander.transform(df_macro)
    
    # 2. Lightweight Expansion for Spread columns
    # (Checking Stationarity is already done, we just need history)
    # No Slopes (Spread is already a diff), No Vol (Spread is Z-scored).
    # Just Lags.
    if spread_cols:
        df_spreads = macro_data[spread_cols]
        spread_features_list = [df_spreads] # Keep base Z-score spreads
        
        for lag in [1, 3]:
             spread_features_list.append(df_spreads.shift(lag).add_suffix(f"_lag{lag}"))
             
        df_spreads_expanded = pd.concat(spread_features_list, axis=1)
        
        # Merge
        return pd.concat([df_macro_expanded, df_spreads_expanded], axis=1)
    
    return df_macro_expanded


@cache_data_wrapper
def load_fred_md_data(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data applying McCracken & Ng transformations."""
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        df_raw = pd.read_csv(file_path)
        
        # 1. Detect if it's a standard FRED-MD (has sasdate column)
        if 'sasdate' in df_raw.columns:
            # Transformation row is the first row (index 0)
            tcodes = df_raw.iloc[0, 1:]
            df = df_raw.iloc[1:].copy()
            
            df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=['sasdate']).set_index('sasdate')
            
            # Apply transformations
            transformed_cols = {}
            for col in df.columns:
                if col in tcodes.index:
                    tcode = pd.to_numeric(tcodes[col], errors='coerce')
                    if not pd.isna(tcode):
                        transformed_cols[col] = apply_transformation(pd.to_numeric(df[col], errors='coerce'), int(tcode))
                    else:
                        transformed_cols[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    transformed_cols[col] = pd.to_numeric(df[col], errors='coerce')
            
            data = pd.DataFrame(transformed_cols, index=df.index)
            
            # Branch B: Cointegration (Levels)
            try:
                # Pass clean numeric raw data to engine
                df_numeric = df.apply(pd.to_numeric, errors='coerce')
                coint_engine = CointegrationEngine()
                levels_df = coint_engine.transform(df_numeric)
                
                if not levels_df.empty:
                    data = pd.concat([data, levels_df], axis=1)
            except Exception as e:
                print(f"Cointegration integration warning: {e}")

        else:
            # Assume it's a PIT matrix (already processed or requires index-based handling)
            df = df_raw.copy()
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'date'})
            
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=[date_col]).set_index(date_col)
            
            # For PIT data, we assume it's already stationary or we don't have tcodes
            # But let's try to map columns if they are raw levels
            data = df.apply(pd.to_numeric, errors='coerce')
        
        # Align to Month End
        data.index = data.index + pd.offsets.MonthEnd(0)
        
        # Ensure Big 4 are present for Orthogonalization later
        # We might need to map them if they have different names
        big_4_mapping = {
            'CPIAUCSL': 'CPIAUCSL',
            'INDPRO': 'INDPRO',
            'M2SL': 'M2SL',
            'FEDFUNDS': 'FEDFUNDS'
        }
        for fred_col, target_col in big_4_mapping.items():
            if fred_col in data.columns and target_col not in data.columns:
                data[target_col] = data[fred_col]
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        return data
        
    except Exception as e:
        print(f"Error loading FRED-MD data: {e}")
        return pd.DataFrame()



@cache_data_wrapper
def load_hybrid_asset_data(start_date: str = '1959-01-01', macro_file: str = '2025-11-MD.csv') -> pd.DataFrame:
    """
    Load hybrid asset data: ETF 'Head' spliced onto Macro Proxy 'Tail'.
    Uses Ratio Splicing to eliminate tracking error/look-ahead bias.
    Independent splicing per asset to ensure maximum history (e.g. 1960 for Equity/Bonds).
    """
    # 1. Fetch Proxy Tails (FRED-MD)
    df_proxies_raw = pd.DataFrame()
    try:
        if not os.path.exists(macro_file):
            macro_file = '2025-11-MD.csv'
            
        if os.path.exists(macro_file):
            df_m = pd.read_csv(macro_file)
            if 'sasdate' in df_m.columns:
                df_m = df_m.iloc[1:] # Skip transform row
                date_col = 'sasdate'
            else:
                date_col = 'Unnamed: 0' if 'Unnamed: 0' in df_m.columns else df_m.columns[0]
            
            df_m['date'] = pd.to_datetime(df_m[date_col], utc=True, errors='coerce').dt.tz_localize(None)
            df_m['date'] = df_m['date'] + pd.offsets.MonthEnd(0)
            df_m = df_m.dropna(subset=['date']).set_index('date')
            
            # EQUITY Proxy (S&P 500)
            sp_col = next((c for c in ['S&P 500', 'SP500', 'S&P_500'] if c in df_m.columns), None)
            if sp_col:
                df_proxies_raw['EQUITY'] = pd.to_numeric(df_m[sp_col], errors='coerce')
                
            # BONDS Proxy (Synthetic from GS10 in FRED-MD)
            gs10_col = next((c for c in ['GS10', 'GS10x', 'GS10_'] if c in df_m.columns), None)
            if gs10_col:
                yields = pd.to_numeric(df_m[gs10_col], errors='coerce') / 100
                duration = 7.5
                carry = yields.shift(1) / 12
                price_change = -duration * (yields - yields.shift(1))
                df_proxies_raw['BONDS_RET'] = (carry + price_change).fillna(0)
                
            # GOLD Proxy Fallback (PPI Metals in FRED-MD)
            if 'PPICMM' in df_m.columns:
                df_proxies_raw['GOLD_PROXY'] = pd.to_numeric(df_m['PPICMM'], errors='coerce')
    except Exception as e:
        print(f"Error loading Local Proxies: {e}")

    # Gold Proxy (Try FRED first, then local fallback)
    try:
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        gold_ppi = gold_ppi.resample('ME').last()
        df_proxies_raw['GOLD'] = gold_ppi['WPU1022']
    except Exception as e:
        print(f"Error fetching Gold Proxy from FRED: {e}")
        if 'GOLD_PROXY' in df_proxies_raw.columns:
            df_proxies_raw['GOLD'] = df_proxies_raw['GOLD_PROXY']

    # 2. Fetch ETF Heads (Yahoo Finance)
    etf_map = {'SPY': 'EQUITY', 'IEF': 'BONDS', 'GLD': 'GOLD'}
    df_etfs = pd.DataFrame()
    retry_count = 3
    for attempt in range(retry_count):
        try:
            t = Ticker(list(etf_map.keys()), asynchronous=False)
            df_etf_raw = t.history(period='max', interval='1d')
            if not df_etf_raw.empty:
                df_etfs = df_etf_raw.reset_index().pivot(index='date', columns='symbol', values='adjclose')
                df_etfs = df_etfs.rename(columns=etf_map)
                df_etfs.index = pd.to_datetime(df_etfs.index, utc=True).tz_localize(None)
                df_etfs = df_etfs.resample('ME').last()
                break
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"Error fetching ETF data after {retry_count} attempts: {e}")
            else:
                time.sleep(1) # Wait before retry

    # 3. Independent Splice Engine (Ratio Splicing)
    spliced_results = {}
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        if asset not in df_etfs.columns or df_etfs[asset].dropna().empty:
            # Fallback for Bonds specifically if GS10 exists but IEF doesn't fetch
            if asset == 'BONDS' and 'BONDS_RET' in df_proxies_raw.columns:
                spliced_results[asset] = (1 + df_proxies_raw['BONDS_RET']).cumprod() * 100
            elif asset in df_proxies_raw.columns:
                spliced_results[asset] = df_proxies_raw[asset]
            continue
            
        head_series = df_etfs[asset].dropna()
        t_splice = head_series.index[0]
        head = head_series.loc[t_splice:]
        
        # Proxy Returns calculation
        if asset == 'BONDS':
            proxy_ret = df_proxies_raw.get('BONDS_RET', pd.Series(dtype=float))
        else:
            proxy_ret = df_proxies_raw[asset].pct_change() if asset in df_proxies_raw.columns else pd.Series(dtype=float)
            
        # Backcast Loop
        current_price = head.iloc[0]
        history = []
        
        # Get proxy returns and filter for the backcast period (strictly before t_splice)
        proxy_ret_backcast = proxy_ret[:t_splice].iloc[:-1].iloc[::-1] # Step backwards from t_splice
        
        for date, ret in proxy_ret_backcast.items():
            if date >= t_splice or pd.isna(ret):
                continue
            prev_price = current_price / (1 + ret)
            history.append((date, prev_price))
            current_price = prev_price
        
        if history:
            tail = pd.DataFrame(history, columns=['date', asset]).set_index('date').sort_index()
            spliced_results[asset] = pd.concat([tail[asset], head])
        else:
            spliced_results[asset] = head

    # Combine independently (No global dropna)
    spliced_data = pd.DataFrame(spliced_results)
    
    # Final Sanitization
    spliced_data = spliced_data.replace([np.inf, -np.inf], np.nan)
    spliced_data = spliced_data.where(spliced_data > 0, np.nan) # Prices must be positive
    
    return spliced_data.sort_index().apply(pd.to_numeric, errors='coerce')


def load_asset_data(start_date: str = '1959-01-01', macro_file: str = '2025-11-MD.csv') -> pd.DataFrame:
    """
    Deprecated: Wrapper for load_hybrid_asset_data to maintain backward compatibility.
    """
    return load_hybrid_asset_data(start_date, macro_file)
