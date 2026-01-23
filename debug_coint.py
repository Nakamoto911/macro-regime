import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from data_utils import load_fred_md_data, CointegrationEngine

print("Loading Raw Data from CSV...")
df_raw = pd.read_csv('2025-11-MD.csv')
tcodes = df_raw.iloc[0, 1:]
df = df_raw.iloc[1:].copy()
df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
df = df.dropna(subset=['sasdate']).set_index('sasdate')
numeric_df = df.apply(pd.to_numeric, errors='coerce')

targets = ['S&P 500', 'SP500', 'EQUITY', 'CPIAUCSL', 'GS10', 'DGS10']
anchors = ['M2SL', 'GDP', 'INDPRO', 'PPI', 'PCE']

print("\n--- Diagnostic Cointegration Check (Raw Levels) ---")
for t in targets:
    for a in anchors:
        if t in numeric_df.columns and a in numeric_df.columns:
            # Check length/quality
            s_t = numeric_df[t].dropna()
            s_a = numeric_df[a].dropna()
            
            # Log Logic
            if (s_t > 0).mean() > 0.95:
                Y = np.log(numeric_df[t].replace(0, np.nan))
                t_lbl = f"log({t})"
            else:
                Y = numeric_df[t]
                t_lbl = t
                
            if (s_a > 0).mean() > 0.95:
                X = np.log(numeric_df[a].replace(0, np.nan))
                a_lbl = f"log({a})"
            else:
                X = numeric_df[a]
                a_lbl = a
            
            # Align
            common_idx = Y.index.intersection(X.index)
            Y = Y.loc[common_idx]
            X = X.loc[common_idx]
            
            # Drop NaNs/Infs
            valid = np.isfinite(Y) & np.isfinite(X)
            Y = Y[valid]
            X = X[valid]
            
            if len(Y) < 60:
                print(f"{t} vs {a}: Skipped (Insufficient Data)")
                continue

            # OLS
            X_exog = sm.add_constant(X)
            model = sm.OLS(Y, X_exog).fit()
            resid = model.resid
            
            # ADF
            # We use a slightly different config to see sensitivity
            res = adfuller(resid, autolag='AIC')
            p_val = res[1]
            
            print(f"{t} vs {a}: p-value = {p_val:.4f}")
