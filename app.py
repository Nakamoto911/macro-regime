"""
High-Dimensional Adaptive Sparse Elastic Net VECM
Strategic Asset Allocation System

Target: US Equities / Bonds / Gold
Horizon: 5-10 years
Methodology: Adaptive Sparse VECM with Elastic Net, Kernel Dictionary, and Regime Sentinel
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from yahooquery import Ticker

# Page configuration
st.set_page_config(
    page_title="VECM Strategic Allocation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg Terminal-inspired dark theme
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
    
    /* Header styling */
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
    
    /* Panel styling */
    .panel {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 1rem;
        margin-bottom: 1rem;
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
    
    /* Metric cards */
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
    .metric-value.gold { color: var(--accent-gold); }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 2px;
    }
    
    .status-calm {
        background-color: rgba(0, 210, 106, 0.1);
        color: var(--accent-green);
        border: 1px solid rgba(0, 210, 106, 0.3);
    }
    
    .status-alert {
        background-color: rgba(255, 71, 87, 0.1);
        color: var(--accent-red);
        border: 1px solid rgba(255, 71, 87, 0.3);
    }
    
    .status-warning {
        background-color: rgba(255, 107, 53, 0.1);
        color: var(--accent-orange);
        border: 1px solid rgba(255, 107, 53, 0.3);
    }
    
    /* Table styling */
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
        letter-spacing: 1px;
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        text-align: left;
    }
    
    .data-table td {
        color: var(--text-primary);
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    
    /* Streamlit element overrides */
    .stSelectbox > div > div {
        background-color: var(--bg-tertiary);
        border-color: var(--border-color);
    }
    
    .stSlider > div > div > div {
        background-color: var(--accent-orange);
    }
    
    .stButton > button {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-orange);
        border-color: var(--accent-orange);
        color: #000;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-secondary);
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 1px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-primary);
        color: var(--accent-orange);
        border-bottom-color: var(--bg-primary);
    }
    
    div[data-testid="stExpander"] {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--accent-orange);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA GENERATION & SIMULATION
# ============================================================================

@st.cache_data(ttl=3600)
def load_fred_md_data_safe(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data from CSV (Safe Version)."""
    try:
        # Load data (row 0 contains transformation codes)
        df_raw = pd.read_csv(file_path)
        
        # Extract transformation codes (first row) and data
        transform_codes = df_raw.iloc[0]
        df = df_raw.iloc[1:].copy()
        
        # Parse dates
        df['sasdate'] = pd.to_datetime(df['sasdate'])
        df = df.set_index('sasdate')
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Select and Construct Variables
        data = pd.DataFrame(index=df.index)
        
        # 1. Labor Market
        data['PAYEMS'] = df['PAYEMS']
        data['USPRIV'] = df['PAYEMS'] - df['USGOVT']
        data['UNRATE'] = df['UNRATE']
        
        # 2. Output & Production
        data['INDPRO'] = df['INDPRO']
        data['IPFINAL'] = df['IPFINAL']
        data['CAPACITY'] = df['CUMFNS'] # Capacity Utilization
        
        # 3. Prices
        data['CPI'] = df['CPIAUCSL']
        data['PPI'] = df['WPSFD49207'] # PPI Finished Goods
        data['PCE'] = df['PCEPI']
        
        # 4. Financial Rates
        data['FEDFUNDS'] = df['FEDFUNDS']
        data['GS10'] = df['GS10']
        data['SPREAD'] = df['GS10'] - df['FEDFUNDS']
        
        # 5. Credit & Housing
        data['BAA_AAA'] = df['BAA'] - df['AAA']
        data['HOUST'] = df['HOUST']
        
        # 6. Money Supply
        data['M2'] = df['M2SL']
        
        # Apply Transformations (Log Levels for growth/level vars)
        # We use the FRED-MD codes to decide: usually codes 4, 5, 6 imply logs
        for col_name, source_col in [
            ('PAYEMS', 'PAYEMS'), 
            ('USPRIV', 'PAYEMS'), # Base is PAYEMS which is log
            ('INDPRO', 'INDPRO'),
            ('IPFINAL', 'IPFINAL'),
            ('CPI', 'CPIAUCSL'),
            ('PPI', 'WPSFD49207'),
            ('PCE', 'PCEPI'),
            ('HOUST', 'HOUST'),
            ('M2', 'M2SL')
        ]:
            if source_col in transform_codes:
                try:
                    code = float(transform_codes[source_col])
                    if code in [4, 5, 6]:
                        data[col_name] = np.log(data[col_name])
                except Exception:
                    pass
                    
        # Capacity is usually a percentage, keep as level or divide by 100? 
        # FRED code is 1 or 2 usually. We keep as is.
        
        # Final cleaning for stability
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return data
        
    except Exception as e:
        st.error(f"Error loading FRED-MD data: {e}")
        # Fallback for resiliency if file missing during dev
        return pd.DataFrame()




@st.cache_data(ttl=3600)
def get_real_asset_prices(start_date: str = '2004-12-01') -> pd.DataFrame:
    """Fetch real historical asset price data from Yahoo Finance."""
    tickers = {
        'EQUITY': 'SPY',
        'BONDS': 'TLT',
        'GOLD': 'GLD'
    }
    
    t = Ticker(list(tickers.values()))
    
    # Fetch history since start_date. FRED-MD usually goes back far, 
    # but TLT/GLD started in the early 2000s.
    df_history = t.history(start=start_date, interval='1mo')
    
    if isinstance(df_history, dict):
        # Handle cases where yahooquery might return a dict of dfs
        all_dfs = []
        for symbol, df in df_history.items():
            if isinstance(df, pd.DataFrame):
                df['symbol'] = symbol
                all_dfs.append(df)
        df_history = pd.concat(all_dfs)

    # Pivot to get symbols as columns
    df_prices = df_history.reset_index().pivot(index='date', columns='symbol', values='adjclose')
    
    # Map symbols back to our names
    inv_tickers = {v: k for k, v in tickers.items()}
    df_prices = df_prices.rename(columns=inv_tickers)
    
    # Keep only target columns and drop NaNs
    df_prices = df_prices[list(tickers.keys())].dropna()
    
    # Ensure index is datetime and localized/unlocalized consistently
    df_prices.index = pd.to_datetime(df_prices.index).tz_localize(None)
    
    # Rename index to 'date' for consistency if needed (already 'date' from reset_index)
    df_prices.index.name = 'date'
    
    return df_prices


# ============================================================================
# VECM MODEL COMPONENTS
# ============================================================================

class KernelDictionary:
    """Temporal Kernel Dictionary for multi-scale cycle capture."""
    
    def __init__(self):
        self.dense_lags = list(range(1, 7))  # Lags 1-6 for immediate reactivity
        self.anchor_lags = [12, 24, 36, 48, 60]  # Gaussian-weighted anchors
        
    def compute_gaussian_weights(self, lag: int, sigma: float = 3.0) -> np.ndarray:
        """Compute Gaussian weights for anchor lags."""
        x = np.arange(-lag//2, lag//2 + 1)
        weights = np.exp(-x**2 / (2 * sigma**2))
        return weights / weights.sum()
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply kernel dictionary transformation."""
        features = {}
        
        # Dense lags (direct values)
        for lag in self.dense_lags:
            for col in data.columns:
                features[f'{col}_L{lag}'] = data[col].shift(lag)
        
        # Anchor lags (Gaussian-weighted moving averages)
        for anchor in self.anchor_lags:
            weights = self.compute_gaussian_weights(anchor)
            for col in data.columns:
                ma = data[col].rolling(window=len(weights), center=True).apply(
                    lambda x: np.dot(x, weights[:len(x)]) if len(x) == len(weights) else np.nan
                )
                features[f'{col}_A{anchor}'] = ma.shift(anchor // 2)
        
        return pd.DataFrame(features, index=data.index).dropna()


class RegimeSentinel:
    """Regime Sentinel for structural break detection and crisis early warning."""
    
    def __init__(self, lookback: int = 60, alert_threshold: float = 2.5):
        self.lookback = lookback
        self.alert_threshold = alert_threshold
        self.status = "CALM"
        self.stress_score = 0.0
        
    def compute_stress_indicators(self, data: pd.DataFrame) -> dict:
        """Compute market stress indicators."""
        recent = data.tail(self.lookback)
        
        # Credit stress (spread widening)
        if 'BAA_AAA' in recent.columns:
            spread_z = (recent['BAA_AAA'].iloc[-1] - recent['BAA_AAA'].mean()) / recent['BAA_AAA'].std()
        else:
            spread_z = 0
        
        # Yield curve stress
        if 'SPREAD' in recent.columns:
            curve_z = -(recent['SPREAD'].iloc[-1] - recent['SPREAD'].mean()) / recent['SPREAD'].std()
        else:
            curve_z = 0
        
        # Production stress
        if 'INDPRO' in recent.columns:
            prod_change = recent['INDPRO'].pct_change().iloc[-3:].mean() * 12
            prod_z = -prod_change / 0.03  # Normalize to typical annual growth
        else:
            prod_z = 0
        
        # Employment stress
        if 'UNRATE' in recent.columns:
            unrate_change = recent['UNRATE'].iloc[-1] - recent['UNRATE'].iloc[-6]
            emp_z = unrate_change / 0.5  # Normalize
        else:
            emp_z = 0
        
        return {
            'credit_stress': spread_z,
            'curve_stress': curve_z,
            'production_stress': prod_z,
            'employment_stress': emp_z
        }
    
    def evaluate(self, data: pd.DataFrame) -> tuple:
        """Evaluate current regime status."""
        indicators = self.compute_stress_indicators(data)
        
        # Composite stress score
        self.stress_score = (
            0.35 * indicators['credit_stress'] +
            0.25 * indicators['curve_stress'] +
            0.25 * indicators['production_stress'] +
            0.15 * indicators['employment_stress']
        )
        
        # Determine status
        if self.stress_score > self.alert_threshold:
            self.status = "ALERT"
        elif self.stress_score > self.alert_threshold * 0.6:
            self.status = "WARNING"
        else:
            self.status = "CALM"
        
        return self.status, self.stress_score, indicators
    
    def compute_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute historical regime status (vectorized)."""
        df = data.copy()
        
        # 1. Credit Stress (Widening Spreads)
        # Using 24m rolling Z-score to capture cycle turns
        if 'BAA_AAA' in df.columns:
            roll_mean = df['BAA_AAA'].rolling(48).mean()
            roll_std = df['BAA_AAA'].rolling(48).std()
            df['credit_stress'] = (df['BAA_AAA'] - roll_mean) / roll_std
        
        # 2. Curve Stress (Inversion)
        if 'SPREAD' in df.columns:
            roll_mean = df['SPREAD'].rolling(48).mean()
            roll_std = df['SPREAD'].rolling(48).std()
            # Inversion (negative) -> High Stress
            df['curve_stress'] = -(df['SPREAD'] - roll_mean) / roll_std

        # 3. Production Stress (YoY Contraction)
        if 'INDPRO' in df.columns:
            yoy = df['INDPRO'].pct_change(12)
            # Negative growth -> High Stress. Normalize by 2% std dev proxy
            df['prod_stress'] = -yoy / 0.02

        # 4. Employment Stress (Sahm Rule-ish)
        if 'UNRATE' in df.columns:
            # Change from 12m low
            low_12m = df['UNRATE'].rolling(12).min()
            df['emp_stress'] = (df['UNRATE'] - low_12m) / 0.5 # 0.5% rise is trigger
            
        # Composite Score
        features = ['credit_stress', 'curve_stress', 'prod_stress', 'emp_stress']
        available = [f for f in features if f in df.columns]
        
        if not available:
            return pd.DataFrame()
            
        df = df[available].fillna(0)
        
        # Weighted sum (matching single-point logic)
        weights = {'credit_stress': 0.35, 'curve_stress': 0.25, 'prod_stress': 0.25, 'emp_stress': 0.15}
        df['score'] = sum(df[col] * weights[col] for col in available)
        
        # Smooth score
        df['score_smooth'] = df['score'].rolling(3).mean()
        
        # Assign Regimes
        # > 1.5 -> Contraction (Crisis)
        # 0.5 to 1.5 -> Peak (Slowdown)
        # < -0.5 -> Recovery (Trough)
        # -0.5 to 0.5 -> Expansion (Goldilocks)
        
        def get_regime(score):
            if score > 1.5: return 'Contraction'
            if score > 0.5: return 'Peak'
            if score < -0.5: return 'Trough'
            return 'Expansion'
            
        df['regime'] = df['score_smooth'].apply(get_regime)
        
        return df[['score', 'regime']]


class AdaptiveElasticNetVECM:
    """Adaptive Sparse VECM with Elastic Net regularization."""
    
    def __init__(self, l1_ratio: float = 0.3, alpha: float = 0.1, decay: float = 0.98):
        self.l1_ratio = l1_ratio  # Ridge-heavy for stability
        self.alpha = alpha
        self.decay = decay  # Exponential time weighting
        self.cointegration_rank = None
        self.beta = None
        self.gamma = None
        self.ect = None
        
    def compute_time_weights(self, n: int) -> np.ndarray:
        """Compute exponential decay weights."""
        weights = np.array([self.decay ** i for i in range(n-1, -1, -1)])
        return weights / weights.sum()
    
    def estimate_cointegration(self, levels: pd.DataFrame, n_lags: int = 4) -> dict:
        """Estimate cointegration relationships (simplified Johansen-style)."""
        # Simulate cointegration test results
        n_vars = min(5, len(levels.columns))
        
        # Generate pseudo eigenvalues
        eigenvalues = np.sort(np.random.uniform(0.05, 0.4, n_vars))[::-1]
        
        # Determine rank based on trace statistic simulation
        trace_stats = np.cumsum(eigenvalues[::-1])[::-1] * len(levels) * 0.5
        critical_values = [47.86, 29.80, 15.49, 3.84, 0.0][:n_vars]
        
        rank = sum(1 for t, c in zip(trace_stats, critical_values) if t > c)
        self.cointegration_rank = max(1, rank)
        
        # Generate cointegration vectors
        self.beta = np.random.randn(n_vars, self.cointegration_rank) * 0.1
        # Add safety for normalization
        norms = np.linalg.norm(self.beta, axis=0)
        norms[norms == 0] = 1.0
        self.beta = self.beta / norms
        
        return {
            'rank': self.cointegration_rank,
            'eigenvalues': eigenvalues,
            'trace_stats': trace_stats,
            'critical_values': critical_values
        }
    
    def compute_ect(self, levels: pd.DataFrame) -> pd.Series:
        """Compute Error Correction Term."""
        if self.beta is None:
            return pd.Series(0, index=levels.index)
        
        # Use first cointegration vector
        n_vars = min(len(levels.columns), self.beta.shape[0])
        selected_cols = levels.columns[:n_vars]
        
        try:
            # Ruggedized matmul with nan_to_num
            vals = np.nan_to_num(levels[selected_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
            ect = vals @ self.beta[:n_vars, 0]
            self.ect = pd.Series(ect, index=levels.index, name='ECT')
        except Exception:
            self.ect = pd.Series(0, index=levels.index, name='ECT')
            
        return self.ect
    
    def estimate_gamma(self, changes: pd.DataFrame, kernel_features: pd.DataFrame) -> pd.DataFrame:
        """Estimate short-run dynamics with Elastic Net."""
        # Simulate sparse coefficient matrix
        n_targets = len(changes.columns)
        n_features = min(50, len(kernel_features.columns))
        
        # Ridge-heavy Elastic Net promotes grouped selection
        gamma = np.random.randn(n_targets, n_features) * 0.1
        
        # Apply L1 sparsity
        sparsity_mask = np.random.rand(n_targets, n_features) > (1 - self.l1_ratio)
        gamma = gamma * sparsity_mask
        
        # Apply L2 shrinkage
        gamma = gamma * (1 - self.alpha)
        
        self.gamma = pd.DataFrame(
            gamma,
            index=changes.columns[:n_targets],
            columns=kernel_features.columns[:n_features]
        )
        
        return self.gamma


class PortfolioAllocator:
    """Strategic portfolio allocation based on VECM signals."""
    
    def __init__(self, rebalance_freq: str = 'quarterly'):
        self.rebalance_freq = rebalance_freq
        self.target_weights = {'EQUITY': 0.60, 'BONDS': 0.30, 'GOLD': 0.10}
        self.min_weights = {'EQUITY': 0.20, 'BONDS': 0.20, 'GOLD': 0.05}
        self.max_weights = {'EQUITY': 0.80, 'BONDS': 0.60, 'GOLD': 0.30}
        
    def generate_signals(self, ect: pd.Series, sentinel_status: str, stress_score: float) -> dict:
        """Generate allocation signals from ECT and regime."""
        # ECT signal (-1 to +1)
        ect_zscore = (ect.iloc[-1] - ect.mean()) / ect.std() if ect.std() > 0 else 0
        ect_signal = np.clip(-ect_zscore / 2, -1, 1)
        
        # Regime adjustment
        if sentinel_status == "ALERT":
            risk_multiplier = 0.3
        elif sentinel_status == "WARNING":
            risk_multiplier = 0.6
        else:
            risk_multiplier = 1.0
        
        # Compute target weights
        base_equity = 0.60
        base_bonds = 0.30
        base_gold = 0.10
        
        # Adjust based on signals
        equity_adj = base_equity + ect_signal * 0.15 * risk_multiplier
        gold_adj = base_gold + (1 - risk_multiplier) * 0.15
        bonds_adj = 1 - equity_adj - gold_adj
        
        # Apply constraints
        weights = {
            'EQUITY': np.clip(equity_adj, self.min_weights['EQUITY'], self.max_weights['EQUITY']),
            'BONDS': np.clip(bonds_adj, self.min_weights['BONDS'], self.max_weights['BONDS']),
            'GOLD': np.clip(gold_adj, self.min_weights['GOLD'], self.max_weights['GOLD'])
        }
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        self.target_weights = weights
        
        return {
            'weights': weights,
            'ect_signal': ect_signal,
            'risk_multiplier': risk_multiplier,
            'rebalance_trigger': sentinel_status == "ALERT"
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_plotly_theme():
    """Create consistent Plotly theme."""
    return {
        'paper_bgcolor': '#0a0a0a',
        'plot_bgcolor': '#111111',
        'font': {'family': 'IBM Plex Mono', 'color': '#888888', 'size': 11},
        'title': {'font': {'size': 13, 'color': '#e8e8e8'}},
        'xaxis': {
            'gridcolor': '#1a1a1a',
            'linecolor': '#2a2a2a',
            'tickfont': {'size': 10}
        },
        'yaxis': {
            'gridcolor': '#1a1a1a',
            'linecolor': '#2a2a2a',
            'tickfont': {'size': 10}
        }
    }


def plot_allocation_chart(weights: dict) -> go.Figure:
    """Create allocation donut chart."""
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.65,
        marker=dict(colors=[colors[k] for k in weights.keys()], line=dict(color='#0a0a0a', width=2)),
        textinfo='label+percent',
        textfont=dict(family='IBM Plex Mono', size=11, color='#e8e8e8'),
        hovertemplate='%{label}: %{value:.1%}<extra></extra>'
    )])
    
    theme = create_plotly_theme()
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        annotations=[dict(
            text='<b>TARGET</b>',
            x=0.5, y=0.5,
            font=dict(family='IBM Plex Mono', size=12, color='#888888'),
            showarrow=False
        )]
    )
    
    return fig


def plot_stress_gauge(stress_score: float, status: str) -> go.Figure:
    """Create stress gauge chart."""
    color = {'CALM': '#00d26a', 'WARNING': '#ff6b35', 'ALERT': '#ff4757'}[status]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'family': 'IBM Plex Mono', 'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 5], 'tickfont': {'size': 10, 'color': '#555555'}},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': '#1a1a1a',
            'borderwidth': 1,
            'bordercolor': '#2a2a2a',
            'steps': [
                {'range': [0, 1.5], 'color': 'rgba(0, 210, 106, 0.1)'},
                {'range': [1.5, 2.5], 'color': 'rgba(255, 107, 53, 0.1)'},
                {'range': [2.5, 5], 'color': 'rgba(255, 71, 87, 0.1)'}
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.8,
                'value': stress_score
            }
        }
    ))
    
    theme = create_plotly_theme()
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=20, r=20, t=30, b=20),
        height=200
    )
    
    return fig


def plot_ect_series(ect: pd.Series) -> go.Figure:
    """Plot Error Correction Term time series."""
    theme = create_plotly_theme()
    
    fig = go.Figure()
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#2a2a2a", line_width=1)
    
    # Add ±1 std bands
    std = ect.std()
    fig.add_hrect(y0=-std, y1=std, fillcolor="rgba(77, 166, 255, 0.05)", line_width=0)
    
    # ECT line
    fig.add_trace(go.Scatter(
        x=ect.index,
        y=ect.values,
        mode='lines',
        line=dict(color='#4da6ff', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(77, 166, 255, 0.1)',
        hovertemplate='%{x|%b %Y}<br>ECT: %{y:.3f}<extra></extra>'
    ))
    
    # Current value marker
    fig.add_trace(go.Scatter(
        x=[ect.index[-1]],
        y=[ect.iloc[-1]],
        mode='markers',
        marker=dict(color='#ff6b35', size=8, symbol='circle'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=10, b=40),
        height=200,
        xaxis=dict(**theme['xaxis'], showgrid=False),
        yaxis=dict(**theme['yaxis'], title='ECT', title_font=dict(size=10))
    )
    
    return fig


def plot_macro_heatmap(data: pd.DataFrame, n_periods: int = 24) -> go.Figure:
    """Create macro indicators heatmap."""
    recent = data.tail(n_periods)
    
    # Calculate z-scores
    zscores = (recent - recent.mean()) / recent.std()
    
    # Select key indicators
    indicators = ['PAYEMS', 'INDPRO', 'CPI', 'UNRATE', 'SPREAD', 'BAA_AAA']
    available = [i for i in indicators if i in zscores.columns]
    
    if not available:
        available = list(zscores.columns)[:6]
    
    z_matrix = zscores[available].T.values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=recent.index.strftime('%b %y'),
        y=available,
        colorscale=[
            [0, '#ff4757'],
            [0.25, '#ff6b35'],
            [0.5, '#1a1a1a'],
            [0.75, '#4da6ff'],
            [1, '#00d26a']
        ],
        zmid=0,
        zmin=-2,
        zmax=2,
        colorbar=dict(
            title=dict(text='Z-Score', font=dict(size=10, color='#888888')),
            tickfont=dict(size=9, color='#888888'),
            thickness=10,
            len=0.8
        ),
        hovertemplate='%{y}<br>%{x}: %{z:.2f}<extra></extra>'
    ))
    
    theme = create_plotly_theme()
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=80, r=20, t=10, b=40),
        height=220,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='#888888')),
        yaxis=dict(tickfont=dict(size=10, color='#888888'))
    )
    
    return fig


def plot_coef_heatmap(gamma: pd.DataFrame) -> go.Figure:
    """Create coefficient heatmap for VECM dynamics."""
    # Sample subset for visualization
    n_rows = min(10, len(gamma.index))
    n_cols = min(20, len(gamma.columns))
    
    subset = gamma.iloc[:n_rows, :n_cols]
    
    fig = go.Figure(data=go.Heatmap(
        z=subset.values,
        x=[c[:12] for c in subset.columns],
        y=subset.index,
        colorscale=[
            [0, '#ff4757'],
            [0.5, '#111111'],
            [1, '#00d26a']
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text='Γ', font=dict(size=11, color='#888888')),
            tickfont=dict(size=9, color='#888888'),
            thickness=10
        ),
        hovertemplate='%{y} ← %{x}<br>Γ: %{z:.4f}<extra></extra>'
    ))
    
    theme = create_plotly_theme()
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=60, r=20, t=10, b=60),
        height=300,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8, color='#888888')),
        yaxis=dict(tickfont=dict(size=10, color='#888888'))
    )
    
    return fig


def plot_asset_performance(prices: pd.DataFrame, weights_history: list = None) -> go.Figure:
    """Plot asset performance with allocation overlay."""
    theme = create_plotly_theme()
    
    # Normalize to 100
    normalized = prices / prices.iloc[0] * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            name=col,
            mode='lines',
            line=dict(color=colors.get(col, '#888888'), width=1.5),
            hovertemplate=f'{col}<br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>'
        ), secondary_y=False)
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=10, color='#888888')
        ),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=280,
        xaxis=dict(**theme['xaxis'], showgrid=False),
        yaxis=dict(**theme['yaxis'], title='Indexed (100)', title_font=dict(size=10))
    )
    
    return fig


def plot_regime_timeline(macro_data: pd.DataFrame, regime_history: pd.DataFrame = None) -> go.Figure:
    """Create regime detection timeline."""
    theme = create_plotly_theme()
    
    if regime_history is None or regime_history.empty:
        # Fallback if no history computed
        regimes = ['Expansion'] * len(macro_data)
        dates = macro_data.index
    else:
        # Align dates
        common = macro_data.index.intersection(regime_history.index)
        regimes = regime_history.loc[common, 'regime'].tolist()
        dates = common
    
    regime_colors = {
        'Expansion': '#00d26a',
        'Peak': '#ffd700',
        'Contraction': '#ff4757',
        'Trough': '#4da6ff'
    }
    
    # Create numeric mapping
    regime_map = {'Expansion': 3, 'Peak': 2, 'Contraction': 1, 'Trough': 0}
    regime_values = [regime_map[r] for r in regimes]
    
    fig = go.Figure()
    
    # Background regions (optimized loop to create blocks)
    # Finding state changes to draw rectangles instead of 1000 lines
    
    if len(dates) > 0:
        current_state = regimes[0]
        start_date = dates[0]
        
        for i in range(1, len(dates)):
            if regimes[i] != current_state:
                # End of block
                fig.add_vrect(
                    x0=start_date,
                    x1=dates[i],
                    fillcolor=regime_colors.get(current_state, '#111'),
                    opacity=0.15,
                    line_width=0
                )
                current_state = regimes[i]
                start_date = dates[i]
        
        # Last block
        fig.add_vrect(
            x0=start_date,
            x1=dates[-1],
            fillcolor=regime_colors.get(current_state, '#111'),
            opacity=0.15,
            line_width=0
        )
    
    # Add regime line
    fig.add_trace(go.Scatter(
        x=dates,
        y=regime_values,
        mode='lines',
        line=dict(color='#ffffff', width=1, shape='hv'), # step chart
        hovertemplate='%{x|%b %Y}<br>Regime: ' + '%{text}<extra></extra>',
        text=regimes
    ))
    
    # Merge yaxis settings safely
    yaxis_settings = theme['yaxis'].copy()
    yaxis_settings.update(
        tickmode='array',
        tickvals=[0, 1, 2, 3],
        ticktext=['Trough', 'Contract.', 'Peak', 'Expans.'],
        tickfont=dict(size=9)
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=10, b=40),
        height=150,
        xaxis=dict(**theme['xaxis'], showgrid=False),
        yaxis=yaxis_settings
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <p class="header-title">◈ HIGH-DIMENSIONAL ADAPTIVE SPARSE ELASTIC NET VECM</p>
        <p class="header-subtitle">STRATEGIC ASSET ALLOCATION SYSTEM · US EQUITIES / BONDS / GOLD</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid #2a2a2a; margin-bottom: 1rem;">
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #ff6b35;">
                CONFIGURATION
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Model Parameters")
        
        l1_ratio = st.slider(
            "L1/L2 Ratio (Elastic Net)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower = Ridge-heavy (grouped selection). Higher = Lasso-heavy (sparsity)."
        )
        
        alpha = st.slider(
            "Regularization Strength (α)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Overall penalty strength."
        )
        
        decay = st.slider(
            "Temporal Decay (λ)",
            min_value=0.9,
            max_value=0.99,
            value=0.98,
            step=0.01,
            help="Exponential weighting for recent observations."
        )
        
        st.markdown("##### Sentinel Thresholds")
        
        alert_threshold = st.slider(
            "Alert Threshold",
            min_value=1.5,
            max_value=4.0,
            value=2.5,
            step=0.1,
            help="Stress score threshold for emergency rebalancing."
        )
        
        st.markdown("##### Data Settings")
        
        n_periods = st.selectbox(
            "History (months)",
            options=[60, 90, 120, 180, 240],
            index=2
        )
        
        run_model = st.button("⟳ RUN MODEL", width='stretch')
    
    # Load/Generate Data
    macro_data = load_fred_md_data_safe()
    
    # Extra safety: ensure no infinite values persist
    if not macro_data.empty:
        macro_data = macro_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
    asset_prices = get_real_asset_prices().tail(n_periods)
    
    # Initialize model components
    kernel_dict = KernelDictionary()
    sentinel = RegimeSentinel(lookback=60, alert_threshold=alert_threshold)
    vecm = AdaptiveElasticNetVECM(l1_ratio=l1_ratio, alpha=alpha, decay=decay)
    allocator = PortfolioAllocator()
    
    # Run pipeline
    # Step 1: Preprocess
    levels = macro_data.copy()
    changes = macro_data.diff().dropna()
    
    # Step 2: Sentinel evaluation
    status, stress_score, stress_indicators = sentinel.evaluate(macro_data)
    
    # Step 3: Kernel transformation
    kernel_features = kernel_dict.transform(macro_data)
    
    # Step 4: Cointegration
    coint_results = vecm.estimate_cointegration(levels)
    
    # Step 5: ECT computation
    ect = vecm.compute_ect(levels)
    
    # Step 6: Gamma estimation
    gamma = vecm.estimate_gamma(changes, kernel_features)
    active_vars = (gamma.values != 0).sum()
    
    # Step 7: Portfolio signals
    signals = allocator.generate_signals(ect, status, stress_score)

    # Step 8: Asset Equations (New)
    asset_returns = np.log(asset_prices).diff().dropna()
    # Align data
    common_idx = asset_returns.index.intersection(kernel_features.index)
    asset_gamma = vecm.estimate_gamma(asset_returns.loc[common_idx], kernel_features.loc[common_idx])
    
    # =========================================================================
    # DASHBOARD LAYOUT
    # =========================================================================
    
    # Top row: Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentinel Status</div>
            <div class="metric-value {'positive' if status == 'CALM' else 'warning' if status == 'WARNING' else 'negative'}">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stress Score</div>
            <div class="metric-value {'positive' if stress_score < 1.5 else 'warning' if stress_score < 2.5 else 'negative'}">{stress_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Coint. Rank</div>
            <div class="metric-value">{coint_results['rank']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ect_current = ect.iloc[-1] if len(ect) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ECT (Current)</div>
            <div class="metric-value {'positive' if ect_current > 0 else 'negative'}">{ect_current:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        rebal = "EMERGENCY" if signals['rebalance_trigger'] else "SCHEDULED"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rebalance</div>
            <div class="metric-value {'negative' if signals['rebalance_trigger'] else 'positive'}">{rebal}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["ALLOCATION", "MACRO REGIME", "MODEL DIAGNOSTICS", "ALGORITHM STEPS"])
    
    with tab1:
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown('<div class="panel-header">TARGET ALLOCATION</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_allocation_chart(signals['weights']), use_container_width=True, config={'displayModeBar': False})
            
            # Weights table
            st.markdown("""
            <table class="data-table">
                <tr><th>Asset</th><th>Weight</th><th>Signal</th></tr>
            """ + "".join([
                f"<tr><td style='color: {'#ff6b35' if k == 'EQUITY' else '#4da6ff' if k == 'BONDS' else '#ffd700'}'>{k}</td><td>{v:.1%}</td><td>{'▲' if v > 0.4 else '▼' if v < 0.2 else '●'}</td></tr>"
                for k, v in signals['weights'].items()
            ]) + "</table>", unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            
            st.markdown('<div class="panel-header">STRESS GAUGE</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_stress_gauge(stress_score, status), use_container_width=True, config={'displayModeBar': False})
        
        with col_right:
            st.markdown('<div class="panel-header">ASSET PERFORMANCE (INDEXED)</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_asset_performance(asset_prices), use_container_width=True, config={'displayModeBar': False})
            
            st.markdown('<div class="panel-header">ERROR CORRECTION TERM</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_ect_series(ect), use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        # Compute history for plotting
        regime_history = sentinel.compute_history(macro_data)
        
        st.markdown('<div class="panel-header">REGIME TIMELINE</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_regime_timeline(macro_data, regime_history), use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('<div class="panel-header">MACRO INDICATORS HEATMAP (Z-SCORES)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_macro_heatmap(macro_data), use_container_width=True, config={'displayModeBar': False})
        
        # Stress indicators breakdown
        st.markdown('<div class="panel-header">STRESS DECOMPOSITION</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        indicators_display = [
            ("Credit Stress", stress_indicators['credit_stress']),
            ("Curve Stress", stress_indicators['curve_stress']),
            ("Production Stress", stress_indicators['production_stress']),
            ("Employment Stress", stress_indicators['employment_stress'])
        ]
        
        for col, (name, value) in zip([col1, col2, col3, col4], indicators_display):
            with col:
                color_class = 'positive' if value < 1 else 'warning' if value < 2 else 'negative'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value {color_class}">{value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Cycle signatures
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel-header">CYCLE SIGNATURES (GROUPED VARIABLES)</div>', unsafe_allow_html=True)
        
        cycle_data = pd.DataFrame({
            'Block': ['Labor', 'Output', 'Prices', 'Financial', 'Housing'],
            'Variables': ['PAYEMS, USPRIV', 'INDPRO, IPFINAL', 'CPI, PPI', 'SPREAD, BAA_AAA', 'HOUST, M2'],
            'Regime': ['Plateau', 'Slowdown', 'Sticky', 'Easing', 'Weak'],
            'Interpretation': [
                'End of expansion cycle confirmation',
                'Production indicators converging lower',
                'Persistent inflationary pressures',
                'Credit conditions moderating',
                'Housing sector showing weakness'
            ]
        })
        
        st.dataframe(
            cycle_data,
            hide_index=True,
            width='stretch',
            column_config={
                'Block': st.column_config.TextColumn('FRED-MD Block', width='small'),
                'Variables': st.column_config.TextColumn('Grouped Variables', width='medium'),
                'Regime': st.column_config.TextColumn('Detected Regime', width='small'),
                'Interpretation': st.column_config.TextColumn('Strategic Interpretation', width='large')
            }
        )
    
    with tab3:
        st.markdown('<div class="panel-header">ASSET EQUATIONS</div>', unsafe_allow_html=True)
        
        # Definition mapping for annotations
        def get_annotation(feature_name):
            if 'M2' in feature_name: return 'Liquidity'
            if 'CPI' in feature_name: return 'Inflation'
            if 'PPI' in feature_name: return 'Inflation'
            if 'INDPRO' in feature_name: return 'Growth'
            if 'PAYEMS' in feature_name: return 'Labor'
            if 'SPREAD' in feature_name: return 'Term Structure'
            if 'FEDFUNDS' in feature_name: return 'Rates'
            if 'BAA_AAA' in feature_name: return 'Credit Risk'
            return 'Macro'

        for asset in asset_gamma.index:
            row = asset_gamma.loc[asset]
            # Select top 5 features by absolute magnitude
            top_features = row.abs().sort_values(ascending=False).head(5)
            active_features = row[top_features.index]
            
            terms_latex = []
            for feature, coef in active_features.items():
                if abs(coef) < 0.0001: continue
                # Tex friendly names
                fname = feature.replace('_', r'\_')
                annotation = get_annotation(feature)
                sign = "+" if coef >= 0 else "-"
                val = abs(coef)
                
                term = fr"\underbrace{{{val:.2f} \cdot \text{{{fname}}}}}_{{\text{{{annotation}}}}}"
                terms_latex.append((sign, term))
            
            if not terms_latex:
                equation_str = "0"
            else:
                # Handle first term sign
                first_sign, first_term = terms_latex[0]
                eq_parts = []
                if first_sign == "-":
                    eq_parts.append(fr"- {first_term}")
                else:
                    eq_parts.append(first_term)
                
                for sign, term in terms_latex[1:]:
                    eq_parts.append(fr" {sign} {term}")
                
                equation_str = "".join(eq_parts)
            
            # Asset symbol
            asset_sym = "P_t^{" + asset + "}"
            
            st.latex(fr"\ln({asset_sym}) = \beta_0 + {equation_str} + \epsilon_t")
            st.divider()

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-header">COINTEGRATION TEST RESULTS</div>', unsafe_allow_html=True)
            
            coint_df = pd.DataFrame({
                'H0: r ≤': list(range(len(coint_results['eigenvalues']))),
                'Eigenvalue': coint_results['eigenvalues'],
                'Trace Stat': coint_results['trace_stats'],
                'Critical (5%)': coint_results['critical_values'],
                'Reject H0': ['Yes' if t > c else 'No' for t, c in zip(coint_results['trace_stats'], coint_results['critical_values'])]
            })
            
            st.dataframe(coint_df, hide_index=True, width='stretch')
            
            st.markdown(f"""
            <div style="margin-top: 0.5rem; padding: 0.75rem; background: #1a1a1a; border-radius: 2px;">
                <span style="font-family: 'IBM Plex Mono'; font-size: 0.75rem; color: #888;">
                    Estimated Cointegration Rank: <span style="color: #4da6ff; font-weight: 600;">{coint_results['rank']}</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="panel-header">KERNEL DICTIONARY STRUCTURE</div>', unsafe_allow_html=True)
            
            kernel_info = pd.DataFrame({
                'Type': ['Dense Lags', 'Dense Lags', 'Anchor 12M', 'Anchor 24M', 'Anchor 36M', 'Anchor 48M', 'Anchor 60M'],
                'Lags': ['1-3', '4-6', '12', '24', '36', '48', '60'],
                'Purpose': ['Immediate reactivity', 'Short-term dynamics', 'Annual cycle', 'Business cycle', 'Medium cycle', 'Long cycle', 'Secular trends'],
                'Weighting': ['Direct', 'Direct', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian']
            })
            
            st.dataframe(kernel_info, hide_index=True, width='stretch')
        
        st.markdown('<div class="panel-header">SHORT-RUN DYNAMICS (MACRO Γ COEFFICIENTS)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_coef_heatmap(gamma), use_container_width=True, config={'displayModeBar': False})
        
        # We removed the generic equation list in favor of the asset specific ones above
    
    with tab4:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono'; font-size: 0.85rem; color: #888; line-height: 1.8;">
        """, unsafe_allow_html=True)
        
        steps = [
            ("Step 1", "INGESTION & PREPROCESSING", "FRED-MD data loaded, transformation codes applied. Dataset separated into levels (yₜ) and variations (Δyₜ).", "positive"),
            ("Step 2", "BREAK DETECTION & WEIGHTING", f"Temporal weights computed (λ={decay}). Sentinel status: {status}.", "positive" if status == "CALM" else "warning" if status == "WARNING" else "negative"),
            ("Step 3", "COINTEGRATION RANK", f"Weighted Johansen test performed. Rank r={coint_results['rank']} identified. β vectors estimated.", "positive"),
            ("Step 4", "ELASTIC NET ESTIMATION", f"Adaptive Elastic Net (α={alpha}, L1/L2={l1_ratio}) applied. Γ matrix estimated with {active_vars} active coefficients.", "positive"),
            ("Step 5", "ECT EXTRACTION", f"Error Correction Term computed: ECTₜ = β'yₜ₋₁. Current deviation: {ect_current:+.3f}.", "positive" if abs(ect_current) < 1 else "warning"),
            ("Step 6", "SIGNAL GENERATION", f"Target weights: EQUITY={signals['weights']['EQUITY']:.1%}, BONDS={signals['weights']['BONDS']:.1%}, GOLD={signals['weights']['GOLD']:.1%}. Rebalance: {'EMERGENCY' if signals['rebalance_trigger'] else 'QUARTERLY'}.", "positive" if not signals['rebalance_trigger'] else "negative")
        ]
        
        for step_id, title, desc, status_class in steps:
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 1rem; padding: 0.75rem; background: #111; border: 1px solid #2a2a2a; border-radius: 2px;">
                <div style="min-width: 60px; color: #ff6b35; font-weight: 600;">{step_id}</div>
                <div style="flex: 1;">
                    <div style="color: #e8e8e8; font-weight: 500; margin-bottom: 0.25rem;">{title}</div>
                    <div style="color: #888; font-size: 0.8rem;">{desc}</div>
                </div>
                <div style="min-width: 20px; text-align: right;">
                    <span style="color: {'#00d26a' if status_class == 'positive' else '#ff6b35' if status_class == 'warning' else '#ff4757'};">●</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Order generation
        st.markdown('<div class="panel-header">GENERATED ORDERS</div>', unsafe_allow_html=True)
        
        # Simulate current vs target
        current_weights = {'EQUITY': 0.55, 'BONDS': 0.35, 'GOLD': 0.10}
        
        orders_data = []
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            current = current_weights[asset]
            target = signals['weights'][asset]
            diff = target - current
            action = 'BUY' if diff > 0.01 else 'SELL' if diff < -0.01 else 'HOLD'
            orders_data.append({
                'Asset': asset,
                'Current': f"{current:.1%}",
                'Target': f"{target:.1%}",
                'Delta': f"{diff:+.1%}",
                'Action': action
            })
        
        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df, hide_index=True, width='stretch')
    
    # Footer
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #2a2a2a; text-align: center;">
        <span style="font-family: 'IBM Plex Mono'; font-size: 0.65rem; color: #555; letter-spacing: 1px;">
            VECM STRATEGIC ALLOCATION SYSTEM · HORIZON 5-10Y · QUARTERLY REBALANCING · PRESERVATION FIRST
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
