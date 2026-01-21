import numpy as np
import pandas as pd
from inference import HodrickInference
# import pytest

def test_hodrick_coverage_sim():
    """
    Monte Carlo test: Under null of no predictability,
    Hodrick t-stats should reject at nominal rate.
    """
    np.random.seed(42)
    n_simulations = 500 # Reduced for speed
    horizon = 12
    T = 300
    nominal_alpha = 0.05
    
    rejections = 0
    
    for _ in range(n_simulations):
        # Generate null data: returns are unpredictable
        # Monthly returns are i.i.d.
        monthly_returns = np.random.randn(T + horizon) * 0.04
        predictor = np.random.randn(T) # Unrelated predictor
        
        # Create overlapping forward returns
        y_vals = []
        for t in range(T):
            y_vals.append(monthly_returns[t:t+horizon].sum())
        
        y = pd.Series(y_vals)
        X = pd.DataFrame({'predictor': predictor, 'const': 1})
        
        # Fit with Hodrick SE
        estimator = HodrickInference(horizon=horizon)
        estimator.fit(y, X)
        
        # Check if we reject (index 0 is predictor, index 1 is constant)
        if estimator.p_values_[0] < nominal_alpha:
            rejections += 1
    
    rejection_rate = rejections / n_simulations
    print(f"Rejection rate (Hodrick): {rejection_rate:.4f}")
    
    # Should be close to nominal alpha
    # With 500 sims, standard error is sqrt(0.05*0.95/500) approx 0.01
    assert 0.02 < rejection_rate < 0.10, f"Rejection rate {rejection_rate} far from {nominal_alpha}"

def test_hodrick_vs_naive():
    """Compare Hodrick SE with naive SE on a single simulated case with overlap."""
    np.random.seed(42)
    horizon = 12
    T = 200
    
    monthly_returns = np.random.randn(T + horizon) * 0.04
    # AR(1) predictor (persistent)
    predictor = np.zeros(T)
    rho = 0.95
    for t in range(1, T):
        predictor[t] = rho * predictor[t-1] + np.random.randn()
    
    y_vals = [monthly_returns[t:t+horizon].sum() for t in range(T)]
    y = pd.Series(y_vals)
    X = pd.DataFrame({'predictor': predictor, 'const': 1})
    
    estimator = HodrickInference(horizon=horizon)
    estimator.fit(y, X)
    
    # Naive SE (ignoring overlap)
    X_vals = X.values
    XtX_inv = np.linalg.inv(X_vals.T @ X_vals)
    beta = XtX_inv @ X_vals.T @ y.values
    resid = y.values - X_vals @ beta
    sigma2 = np.var(resid)
    naive_se = np.sqrt(np.diag(XtX_inv * sigma2 * T / (T-2)))
    
    print(f"Naive SE (predictor): {naive_se[0]:.6f}")
    print(f"Hodrick SE (predictor): {estimator.hodrick_se_[0]:.6f}")
    
    # Hodrick SE should be significantly LARGER than naive SE
    assert estimator.hodrick_se_[0] > naive_se[0] * 1.5

if __name__ == "__main__":
    test_hodrick_vs_naive()
    test_hodrick_coverage_sim()
