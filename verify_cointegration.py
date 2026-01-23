import pandas as pd
import json
import os
from statsmodels.tsa.stattools import adfuller

print("=== Checking Manifest ===")
manifest_path = 'precomputed/cointegration_manifest.json'
if os.path.exists(manifest_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        print(json.dumps(manifest, indent=2))
else:
    print("ERROR: Manifest not found!")

print("\n=== Checking Parquet ===")
parquet_path = 'precomputed/pit_expanded_features.parquet'
if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    spread_cols = [c for c in df.columns if 'spread' in c and '_resid' not in c and '_slope' not in c and '_lag' not in c and '_impulse' not in c and '_vol' not in c]
    # Actually spread cols are named 'spread_X_Y'. 
    # But expanded features include 'spread_X_Y_slope...', etc.
    # The base spreads are just 'spread_X_Y'.
    # Let's look for base spreads.
    base_spreads = list(manifest.keys())
    
    print(f"Base Spread Columns found in Parquet: {len([c for c in base_spreads if c in df.columns])}")
    
    # Check specifically for S&P 500
    sp500_spreads = [c for c in base_spreads if 'S&P 500' in c or 'SP500' in c]
    print(f"S&P 500 Spreads Found: {sp500_spreads}")
    
    # Analyze a few key ones
    targets_to_check = sp500_spreads[:3] + [c for c in base_spreads if 'M2SL' in c][:3]
    
    for col in set(targets_to_check):
        if col in df.columns:
            print(f"\n--- Analysis of {col} ---")
            series = df[col].dropna()
            print(series.tail())
            
            # ADF Test
            res = adfuller(series.values)
            print(f"ADF Statistic: {res[0]:.4f}")
            print(f"p-value: {res[1]:.4f}")
            if res[1] < 0.10:
                print(">> STATIONARY (Pass)")
            else:
                print(">> NON-STATIONARY (Fail)")
else:
    print("ERROR: Parquet not found!")
