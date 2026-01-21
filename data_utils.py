import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os
import time
from yahooquery import Ticker
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

try:
    import streamlit as st
    def cache_data_wrapper(func):
        return st.cache_data(ttl=86400, show_spinner="Fetching Data...") (func)
except ImportError:
    def cache_data_wrapper(func):
        return func

def compute_forward_returns(prices: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    Compute annualized forward returns for each asset.
    """
    log_prices = np.log(prices)
    forward_log_return = log_prices.shift(-horizon_months) - log_prices
    annualized_return = forward_log_return / (horizon_months / 12)
    return annualized_return


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


def prepare_macro_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy wrapper for MacroFeatureExpander.
    """
    expander = MacroFeatureExpander()
    return expander.transform(macro_data)


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
