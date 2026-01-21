import numpy as np
import pandas as pd
from feature_selection import NestedTimeSeriesFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def test_nested_cv_prevents_overfitting():
    """
    Compare in-sample vs out-of-sample performance.
    Nested CV should show smaller gap than fixed alpha on noise.
    """
    np.random.seed(42)
    T = 300
    K = 50  # Many features
    
    # Generate null data (no true predictability)
    X = pd.DataFrame(np.random.randn(T, K), columns=[f'f{i}' for i in range(K)])
    y = pd.Series(np.random.randn(T) * 0.1)  # Pure noise
    
    # Split
    train_X, test_X = X.iloc[:200], X.iloc[200:]
    train_y, test_y = y.iloc[:200], y.iloc[200:]
    
    # Method 1: Fixed alpha (legacy)
    fixed_model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    
    fixed_model.fit(train_X_scaled, train_y)
    fixed_in_sample_r2 = fixed_model.score(train_X_scaled, train_y)
    fixed_out_sample_r2 = fixed_model.score(test_X_scaled, test_y)
    
    # Method 2: Nested CV
    selector = NestedTimeSeriesFeatureSelector(min_samples=20)
    selector.fit(train_y, train_X, n_bootstrap=10)
    selected = selector.get_selected_features()
    
    # Refit on selected features with CV parameters
    nested_model = ElasticNet(alpha=selector.best_alpha_, l1_ratio=selector.best_l1_ratio_)
    train_X_sel = scaler.fit_transform(train_X[selected])
    test_X_sel = scaler.transform(test_X[selected])
    
    nested_model.fit(train_X_sel, train_y)
    nested_in_sample_r2 = nested_model.score(train_X_sel, train_y)
    nested_out_sample_r2 = nested_model.score(test_X_sel, test_y)
    
    print(f"Fixed alpha: IS R²={fixed_in_sample_r2:.4f}, OOS R²={fixed_out_sample_r2:.4f}")
    print(f"Nested CV:   IS R²={nested_in_sample_r2:.4f}, OOS R²={nested_out_sample_r2:.4f}")
    
    fixed_gap = fixed_in_sample_r2 - fixed_out_sample_r2
    nested_gap = nested_in_sample_r2 - nested_out_sample_r2
    
    print(f"Fixed Gap: {fixed_gap:.4f}, Nested Gap: {nested_gap:.4f}")
    
    # Under pure noise, the nested CV should select fewer features of noise,
    # and the gap should be smaller (less IS overfitting).
    assert nested_gap < fixed_gap, "Nested CV should reduce overfitting gap"
    print("Selection Test Passed!")

if __name__ == "__main__":
    test_nested_cv_prevents_overfitting()
