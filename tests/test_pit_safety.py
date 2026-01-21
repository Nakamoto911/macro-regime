import numpy as np
import pandas as pd
from benchmarking_engine import PointInTimeFactorStripper, FactorStripper

def test_no_future_leakage():
    """
    Verify that PIT orthogonalization doesn't use future data.
    """
    np.random.seed(42)
    T = 200
    
    # Create synthetic data with known structure
    dates = pd.date_range('2000-01-01', periods=T, freq='M')
    
    # Driver has regime change at T/2
    # First half: mean 0, Second half: mean 2
    driver = np.concatenate([np.random.randn(T//2), np.random.randn(T//2) + 2])
    
    # Feature is related to driver with coefficient that changes
    coef_early = 0.5
    coef_late = -0.5
    feature = np.concatenate([
        coef_early * driver[:T//2] + np.random.randn(T//2) * 0.1,
        coef_late * driver[T//2:] + np.random.randn(T//2) * 0.1
    ])
    
    X = pd.DataFrame({
        'driver': driver,
        'feature': feature
    }, index=dates)
    
    # PIT transformation (min_history=30, update every month for sensitivity)
    pit_stripper = PointInTimeFactorStripper(drivers=['driver'], min_history=30, update_frequency=1)
    X_pit = pit_stripper.fit_transform_pit(X)
    
    # Global transformation (look-ahead)
    global_stripper = FactorStripper(drivers=['driver'])
    global_stripper.fit(X)
    X_global = global_stripper.transform(X)
    
    # Check residuals at T/2 - 1 (last point of first regime)
    idx_check = T//2 - 1
    pit_resid = X_pit['feature_resid_driver'].iloc[idx_check]
    global_resid = X_global['feature_resid_driver'].iloc[idx_check]
    
    print(f"PIT residual at T/2-1: {pit_resid:.4f}")
    print(f"Global residual at T/2-1: {global_resid:.4f}")
    
    # Global stripper uses average coefficient (approx 0).
    # PIT stripper uses local coefficient (approx 0.5).
    # Since feature = 0.5 * driver + noise, the PIT residual should be near 0.
    # The global residual will be feature - 0*driver = feature = 0.5*driver + noise, which is NOT near 0.
    
    assert abs(pit_resid) < abs(global_resid), "PIT residual should be smaller in local regime"
    
    # Verify coefficients in PIT history
    stability = pit_stripper.get_coefficient_stability()
    coef_at_split = stability[stability['date'] == dates[idx_check]]['coefficient'].values[0]
    print(f"PIT coefficient just before split: {coef_at_split:.4f}")
    assert 0.4 < coef_at_split < 0.6, "PIT coefficient should reflect first regime"

    # Now change FUTURE data and verify PIT results for PAST do not change
    X_modified = X.copy()
    X_modified.iloc[T//2:, 0] += 100 # Change future driver significantly
    X_modified.iloc[T//2:, 1] += 100 # Change future feature significantly
    
    X_pit_mod = pit_stripper.fit_transform_pit(X_modified)
    pit_resid_mod = X_pit_mod['feature_resid_driver'].iloc[idx_check]
    
    print(f"PIT residual (modified future): {pit_resid_mod:.4f}")
    assert abs(pit_resid - pit_resid_mod) < 1e-10, "Past PIT residuals must not be affected by future data"
    
    print("PIT Safety Test Passed!")

if __name__ == "__main__":
    test_no_future_leakage()
