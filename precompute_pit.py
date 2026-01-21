"""
PIT Matrix Precomputation Script
================================
Run this ONCE when new FRED-MD data arrives, not at runtime.
Saves ~40 minutes of redundant computation.

Usage:
    python precompute_pit.py

Output:
    precomputed/pit_expanded_features.parquet  - Full feature matrix for backtest
    precomputed/pit_orthogonalized.parquet     - Before expansion (for debugging)
    precomputed/orthogonalization_coefficients.parquet - For live inference
    precomputed/metadata.json                  - Cache metadata
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed
import os
import time
import json
from datetime import datetime

# Configuration
FRED_MD_FILE = '2025-11-MD.csv'
OUTPUT_DIR = 'precomputed'
DRIVERS = ['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS']
MIN_HISTORY = 60
SLOPE_WINDOWS = [3, 6, 9, 12, 18, 24]
LAG_WINDOWS = [1, 3, 6]


def apply_transformation(series: pd.Series, tcode: int) -> pd.Series:
    """McCracken & Ng (2016) transformation codes."""
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


def load_and_transform_fred_md(file_path: str) -> pd.DataFrame:
    """Load FRED-MD and apply stationarity transforms."""
    print(f"  Loading from {file_path}...")
    df_raw = pd.read_csv(file_path)

    # First row contains transformation codes
    tcodes = df_raw.iloc[0, 1:]
    df = df_raw.iloc[1:].copy()

    # Parse dates
    df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['sasdate']).set_index('sasdate')
    df.index = df.index + pd.offsets.MonthEnd(0)

    # Apply transformations
    transformed = {}
    for col in df.columns:
        if col in tcodes.index:
            tcode = pd.to_numeric(tcodes[col], errors='coerce')
            if not pd.isna(tcode):
                transformed[col] = apply_transformation(pd.to_numeric(df[col], errors='coerce'), int(tcode))
            else:
                transformed[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            transformed[col] = pd.to_numeric(df[col], errors='coerce')

    result = pd.DataFrame(transformed, index=df.index)
    return result.replace([np.inf, -np.inf], np.nan)


def _orthogonalize_single_feature(col: str, drv: str, X: pd.DataFrame, min_history: int) -> tuple:
    """Orthogonalize one feature against one driver using expanding RollingOLS."""
    if col == drv or col not in X.columns or drv not in X.columns:
        return col, drv, None, None

    X_col = X[col].dropna()
    X_drv = X[drv].dropna()

    # Align on common index
    common_idx = X_col.index.intersection(X_drv.index)
    if len(common_idx) < min_history:
        return col, drv, None, None

    X_col = X_col.loc[common_idx]
    X_drv = X_drv.loc[common_idx]

    try:
        exog = sm.add_constant(X_drv)
        n = len(common_idx)

        rols = RollingOLS(endog=X_col, exog=exog, window=n, min_nobs=min_history, expanding=True)
        rres = rols.fit()

        # Lag params by 1 to avoid look-ahead
        params_pit = rres.params.shift(1)
        predicted = params_pit['const'] + params_pit[drv] * X_drv
        resid = X_col - predicted

        # Store latest coefficients for live prediction
        latest_params = rres.params.iloc[-1].to_dict()

        return col, drv, resid, latest_params
    except Exception as e:
        print(f"  Warning: Failed to orthogonalize {col} vs {drv}: {e}")
        return col, drv, None, None


def compute_pit_orthogonalization(X: pd.DataFrame, drivers: list, min_history: int = 60) -> tuple:
    """
    Compute PIT orthogonalized features using parallel RollingOLS.
    Returns: (orthogonalized_df, coefficient_dict)
    """
    print(f"  Computing PIT orthogonalization for {len(X.columns)} features x {len(drivers)} drivers...")

    available_drivers = [d for d in drivers if d in X.columns]
    if not available_drivers:
        print(f"  Warning: No drivers found in data. Available columns: {list(X.columns[:10])}...")
        return X, {}

    feature_cols = [c for c in X.columns if c not in available_drivers]

    # Build task list
    tasks = [(col, drv) for drv in available_drivers for col in feature_cols]
    total_tasks = len(tasks)
    print(f"  Total orthogonalization tasks: {total_tasks}")

    start = time.time()
    results = Parallel(n_jobs=-1, prefer="threads", verbose=5)(
        delayed(_orthogonalize_single_feature)(col, drv, X, min_history)
        for col, drv in tasks
    )
    elapsed = time.time() - start
    print(f"  Orthogonalization completed in {elapsed:.1f}s ({total_tasks / elapsed:.1f} tasks/sec)")

    # Assemble results
    resid_dict = {}
    coef_dict = {}

    for col, drv, resid, params in results:
        if resid is not None:
            resid_dict[f"{col}_resid_{drv}"] = resid
            if drv not in coef_dict:
                coef_dict[drv] = {}
            coef_dict[drv][col] = params

    resid_df = pd.DataFrame(resid_dict, index=X.index)
    combined = pd.concat([X, resid_df], axis=1)

    print(f"  Generated {len(resid_dict)} residual features")

    return combined.loc[:, ~combined.columns.duplicated()], coef_dict


def expand_features(X: pd.DataFrame, slope_windows=None, lag_windows=None) -> pd.DataFrame:
    """
    Expand features with slopes, lags, impulse, and volatility.
    Uses vectorized operations for efficiency.
    """
    if slope_windows is None:
        slope_windows = SLOPE_WINDOWS
    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    print(f"  Expanding {len(X.columns)} features...")
    start = time.time()

    expanded = [X]

    # Vectorized slope computation (all windows at once)
    for w in slope_windows:
        slopes = X.diff(w)
        slopes.columns = [f"{c}_slope{w}" for c in X.columns]
        expanded.append(slopes)

    # Vectorized lag computation
    for lag in lag_windows:
        lagged = X.shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in X.columns]
        expanded.append(lagged)

    # Impulse (acceleration): 3-month slope change
    slope3 = X.diff(3)
    impulse = slope3 - slope3.shift(3)
    impulse.columns = [f"{c}_impulse" for c in X.columns]
    expanded.append(impulse)

    # Vectorized volatility (12-month rolling std)
    vol12 = X.rolling(12).std()
    vol12.columns = [f"{c}_vol12" for c in X.columns]
    expanded.append(vol12)

    # Symbolic ratios (only if base variables exist)
    if 'M2SL' in X.columns and 'INDPRO' in X.columns:
        ratio = X['M2SL'] / (X['INDPRO'].abs() + 1e-9)
        expanded.append(ratio.rename('RATIO_M2_GROWTH').to_frame())
    if 'CPIAUCSL' in X.columns and 'FEDFUNDS' in X.columns:
        ratio = X['CPIAUCSL'] / (X['FEDFUNDS'].abs() + 1e-9)
        expanded.append(ratio.rename('RATIO_CPI_FEDFUNDS').to_frame())
    if 'GS10' in X.columns and 'CPIAUCSL' in X.columns:
        ratio = X['GS10'] / (X['CPIAUCSL'].abs() + 1e-9)
        expanded.append(ratio.rename('RATIO_GS10_CPI').to_frame())
    if 'UNRATE' in X.columns and 'PAYEMS' in X.columns:
        ratio = X['UNRATE'] / (X['PAYEMS'].abs() + 1e-9)
        expanded.append(ratio.rename('RATIO_UNRATE_PAYEMS').to_frame())
    if 'INDPRO' in X.columns and 'PAYEMS' in X.columns:
        ratio = X['INDPRO'] / (X['PAYEMS'].abs() + 1e-9)
        expanded.append(ratio.rename('RATIO_PROD').to_frame())

    result = pd.concat(expanded, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]

    elapsed = time.time() - start
    print(f"  Expansion completed in {elapsed:.1f}s: {len(result.columns)} features")
    return result


def main():
    """Main precomputation pipeline."""
    print("=" * 70)
    print("PIT MATRIX PRECOMPUTATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check for input file
    if not os.path.exists(FRED_MD_FILE):
        print(f"\nERROR: Input file not found: {FRED_MD_FILE}")
        print("Please ensure the FRED-MD data file exists in the current directory.")
        return

    # Step 1: Load and transform
    print("\n[1/4] Loading FRED-MD data...")
    X_raw = load_and_transform_fred_md(FRED_MD_FILE)
    print(f"  Loaded: {X_raw.shape[0]} observations x {X_raw.shape[1]} variables")
    print(f"  Date range: {X_raw.index[0].strftime('%Y-%m')} to {X_raw.index[-1].strftime('%Y-%m')}")

    # Step 2: PIT Orthogonalization
    print("\n[2/4] Computing PIT orthogonalization...")
    X_ortho, coef_dict = compute_pit_orthogonalization(X_raw, DRIVERS, MIN_HISTORY)
    print(f"  Result: {X_ortho.shape[1]} features (including residuals)")

    # Step 3: Feature expansion
    print("\n[3/4] Expanding feature space...")
    X_expanded = expand_features(X_ortho)

    # Clean infinities and fill NaN with 0 for downstream compatibility
    X_expanded = X_expanded.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Step 4: Save outputs
    print("\n[4/4] Saving precomputed matrices...")

    # Save as float32 to reduce file size
    X_expanded = X_expanded.astype('float32')

    # Parquet is faster to load than CSV
    expanded_path = f"{OUTPUT_DIR}/pit_expanded_features.parquet"
    X_expanded.to_parquet(expanded_path)
    print(f"  Saved: {expanded_path} ({os.path.getsize(expanded_path) / 1024 / 1024:.1f} MB)")

    # Also save coefficients for live inference
    coef_records = []
    for drv, feats in coef_dict.items():
        for feat, params in feats.items():
            record = {'driver': drv, 'feature': feat}
            record.update(params)
            coef_records.append(record)

    if coef_records:
        coef_df = pd.DataFrame(coef_records)
        coef_path = f"{OUTPUT_DIR}/orthogonalization_coefficients.parquet"
        coef_df.to_parquet(coef_path)
        print(f"  Saved: {coef_path}")

    # Save raw orthogonalized (before expansion) for debugging
    ortho_path = f"{OUTPUT_DIR}/pit_orthogonalized.parquet"
    X_ortho.astype('float32').to_parquet(ortho_path)
    print(f"  Saved: {ortho_path}")

    # Metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'fred_md_file': FRED_MD_FILE,
        'drivers': DRIVERS,
        'min_history': MIN_HISTORY,
        'slope_windows': SLOPE_WINDOWS,
        'lag_windows': LAG_WINDOWS,
        'date_range': [X_raw.index[0].isoformat(), X_raw.index[-1].isoformat()],
        'n_raw_features': int(X_raw.shape[1]),
        'n_ortho_features': int(X_ortho.shape[1]),
        'n_expanded_features': int(X_expanded.shape[1]),
        'n_observations': int(len(X_expanded))
    }

    meta_path = f"{OUTPUT_DIR}/metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    print("\n" + "=" * 70)
    print("PRECOMPUTATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"  - pit_expanded_features.parquet ({X_expanded.shape[1]} features x {len(X_expanded)} observations)")
    print(f"  - orthogonalization_coefficients.parquet ({len(coef_records)} coefficient pairs)")
    print(f"  - pit_orthogonalized.parquet ({X_ortho.shape[1]} features)")
    print(f"  - metadata.json")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Verify output: pd.read_parquet('{OUTPUT_DIR}/pit_expanded_features.parquet')")
    print(f"  2. Update app.py to use load_precomputed_features()")
    print(f"  3. Schedule monthly refresh via cron/GitHub Actions")


if __name__ == "__main__":
    main()
