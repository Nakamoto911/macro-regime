import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import Ticker
import pandas_datareader.data as web
import os
from datetime import datetime

# Set page config for wide layout
st.set_page_config(page_title="Asset Pricing Validation Dashboard", layout="wide")

# --- DATA LOADING & LOGIC ---

@st.cache_data(ttl=86400)
def get_legacy_data(start_date='1960-01-01'):
    """
    Legacy Loader (Baseline):
    - Equity: S&P 500 (FRED-MD or FRED)
    - Bonds: GS10 Synthetic
    - Gold: WPU1022 (PPI)
    """
    try:
        # 1. Equity: Try local FRED-MD first if available
        equity_price = None
        if os.path.exists('2025-11-MD.csv'):
            df_m = pd.read_csv('2025-11-MD.csv')
            sp_col = next((c for c in ['S&P 500', 'SP500', 'S&P_500'] if c in df_m.columns), None)
            if sp_col:
                if 'sasdate' in df_m.columns:
                    df_m = df_m.iloc[1:] # Skip transform row
                    df_m['date'] = pd.to_datetime(df_m['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
                else:
                    df_m['date'] = pd.to_datetime(df_m.iloc[:,0], utc=True, errors='coerce').dt.tz_localize(None)
                df_m = df_m.set_index('date')
                equity_price = pd.to_numeric(df_m[sp_col], errors='coerce')
        
        if equity_price is None:
            sp500 = web.DataReader('SP500', 'fred', start_date)
            sp500.index = pd.to_datetime(sp500.index, utc=True).tz_localize(None)
            equity_price = sp500['SP500']
            
        # 2. Bonds: GS10 Synthetic
        gs10 = web.DataReader('GS10', 'fred', start_date)
        gs10.index = pd.to_datetime(gs10.index, utc=True).tz_localize(None)
        yields = gs10['GS10'] / 100
        duration = 7.5
        carry = yields.shift(1) / 12
        price_change = -duration * (yields - yields.shift(1))
        bond_ret = (carry + price_change).fillna(0)
        
        # 3. Gold: PPI (WPU1022)
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        
        # Combine and resample to ME
        common_dates = gs10.index.intersection(gold_ppi.index).intersection(equity_price.index)
        df_raw = pd.DataFrame(index=common_dates)
        df_raw['EQUITY_PRICE'] = equity_price
        df_raw['BONDS_RET'] = bond_ret
        df_raw['GOLD_PRICE'] = gold_ppi['WPU1022']
        
        # Resample to Month-End
        df = df_raw.resample('ME').last()
        
        # Create Indices (Normalized to 100 at first valid date)
        legacy_indices = pd.DataFrame(index=df.index)
        
        # Equity Index
        eq = df['EQUITY_PRICE'].dropna()
        legacy_indices['EQUITY'] = (eq / eq.iloc[0]) * 100
        
        # Bond Index (Accumulated from returns)
        # Re-accumulate specifically from the resampled returns
        # Actually, it's better to accumulate daily then resample, or resample then accumulate.
        # GS10 is monthly anyway mostly, but let's be careful.
        legacy_indices['BONDS'] = (1 + df['BONDS_RET']).cumprod() * 100
        
        # Gold Index
        gd = df['GOLD_PRICE'].dropna()
        legacy_indices['GOLD'] = (gd / gd.iloc[0]) * 100
        
        return legacy_indices.dropna()
    except Exception as e:
        st.error(f"Error loading legacy data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_modern_data(legacy_df):
    """
    Modern Loader (Challenger):
    - Fetch ETF heads via yahooquery
    - Splice with legacy returns for backcasting
    """
    asset_map = {
        'SPY': 'EQUITY',
        'IEF': 'BONDS',
        'GLD': 'GOLD'
    }
    
    try:
        # Fetch ETF data
        tickers = Ticker(list(asset_map.keys()), asynchronous=True)
        df_etf_raw = tickers.history(period='max', interval='1d')
        
        if df_etf_raw.empty:
            st.error("Yahoo Finance returned empty data.")
            return pd.DataFrame()
            
        df_etf = df_etf_raw.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        df_etf.index = pd.to_datetime(df_etf.index).tz_localize(None)
        df_etf = df_etf.resample('ME').last()
        
        spliced_data = pd.DataFrame()
        
        for ticker, asset in asset_map.items():
            if ticker not in df_etf.columns:
                continue
                
            head_series = df_etf[ticker].dropna()
            if head_series.empty:
                continue
                
            t_splice = head_series.index[0]
            
            # Calculate % returns of legacy series for backcasting
            legacy_returns = legacy_df[asset].pct_change()
            
            # Tail returns up to and including t_splice (to backcast from t_splice)
            tail_returns = legacy_returns.loc[:t_splice].iloc[::-1]
            
            # Backcast Loop (The "Zipper")
            current_price = head_series.iloc[0]
            history = []
            
            dates = tail_returns.index.tolist()
            for i in range(len(dates) - 1):
                t_curr = dates[i]
                t_prev = dates[i+1]
                ret = tail_returns.loc[t_curr]
                
                if pd.isna(ret):
                    continue
                    
                prev_price = current_price / (1 + ret)
                history.append({'date': t_prev, 'price': prev_price})
                current_price = prev_price
                
            # Merge Tail and Head
            if history:
                tail_df = pd.DataFrame(history).set_index('date').sort_index()
                spliced_asset = pd.concat([tail_df['price'], head_series])
            else:
                spliced_asset = head_series
                
            # Normalize to 100 at the same start date as legacy for comparison
            start_date = legacy_df.index[0]
            if start_date in spliced_asset.index:
                spliced_data[asset] = (spliced_asset / spliced_asset.loc[start_date]) * 100
            else:
                # If spliced index doesn't go back far enough, just rebase to its own start
                spliced_data[asset] = (spliced_asset / spliced_asset.iloc[0]) * 100
                
        return spliced_data.dropna()
        
    except Exception as e:
        st.error(f"Error loading modern data: {e}")
        return pd.DataFrame()

def calculate_cagr(series):
    if series.empty:
        return 0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return 0
    return (end_val / start_val) ** (1/years) - 1

# --- UI LAYOUT ---

st.title("🛡️ Asset Pricing Validation Dashboard")
st.markdown("""
Validate the migration from **Synthetic** (Legacy) to **ETF-based** (Modern) pricing.
- **Legacy:** FRED Proxies (SP500, GS10, PPI Gold).
- **Modern:** Backfilled ETFs (SPY, IEF, GLD) using Ratio Splicing.
""")

legacy_data = get_legacy_data()

if legacy_data.empty:
    st.stop()

modern_data = get_modern_data(legacy_data)

if modern_data.empty:
    st.stop()

# Align indices to common history
common_index = legacy_data.index.intersection(modern_data.index)
legacy_data = legacy_data.loc[common_index]
modern_data = modern_data.loc[common_index]

tabs = st.tabs(["🇺🇸 Equities (SPY)", "📉 Bonds (IEF)", "🏆 Gold (GLD)"])

asset_list = ["EQUITY", "BONDS", "GOLD"]

for i, tab in enumerate(tabs):
    asset = asset_list[i]
    with tab:
        # Metrics Header
        cagr_leg = calculate_cagr(legacy_data[asset])
        cagr_mod = calculate_cagr(modern_data[asset])
        delta = cagr_mod - cagr_leg
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Legacy CAGR", f"{cagr_leg:.2%}")
        col2.metric("Modern CAGR", f"{cagr_mod:.2%}")
        col3.metric("CAGR Delta", f"{delta:+.2%}", delta_color="normal")
        
        # Visualization
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=("Log Price (Rebased to 100)", "Divergence (Modern / Legacy - 1)"))
        
        # Row 1: Log Price
        fig.add_trace(go.Scatter(x=legacy_data.index, y=legacy_data[asset], 
                                name="Legacy (Proxy)", line=dict(color='red', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=modern_data.index, y=modern_data[asset], 
                                name="Modern (ETF)", line=dict(color='green')), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        
        # Row 2: Divergence
        divergence = (modern_data[asset] / legacy_data[asset]) - 1
        fig.add_trace(go.Scatter(x=divergence.index, y=divergence, 
                                name="Divergence", line=dict(color='blue'), fill='tozeroy'), row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, template="plotly_white")
        st.plotly_chart(fig, width='stretch')

st.divider()
st.info("""
**Validation Heuristics:**
1. **Equity:** SPY should drift ABOVE Legacy due to dividends (~1.5-2.0% CAGR delta).
2. **Bonds:** IEF should show convexity (higher peaks in 2008/2020) vs GS10 linear proxy.
3. **Splice Points:** Check for smoothness at SPY (1993), IEF (2002), and GLD (2004).
""")
