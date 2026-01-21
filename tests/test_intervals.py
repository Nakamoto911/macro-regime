import numpy as np
import pandas as pd
from prediction_intervals import BootstrapPredictionInterval, CoverageValidator
from sklearn.linear_model import LinearRegression

def test_interval_coverage():
    """
    Verify that BootstrapPredictionInterval achieves nominal coverage on synthetic data.
    """
    np.random.seed(42)
    T = 200
    n_test = 50
    
    # Generate data: y = 0.5*x + noise
    X = pd.DataFrame(np.random.randn(T + n_test, 1), columns=['x'])
    noise = np.random.randn(T + n_test) * 0.5
    y = 0.5 * X['x'] + noise
    
    X_train, X_test = X.iloc[:T], X.iloc[T:]
    y_train, y_test = y.iloc[:T], y.iloc[T:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Interval generator
    nominal_level = 0.90
    bt = BootstrapPredictionInterval(confidence_level=nominal_level, n_bootstrap=100)
    bt.fit(model, X_train, y_train)
    
    # Coverage validator
    validator = CoverageValidator(nominal_level=nominal_level)
    
    # Predict and record
    point_preds, lower, upper = bt.predict_interval(X_test)
    
    for i in range(n_test):
        validator.record(y_test.iloc[i], lower[i], upper[i])
        
    stats = validator.compute_statistics()
    print(f"Nominal: {nominal_level*100}%")
    print(f"Empirical Coverage: {stats['empirical_coverage']*100:.1f}%")
    print(f"CI: [{stats['coverage_ci_lower']*100:.1f}%, {stats['coverage_ci_upper']*100:.1f}%]")
    
    # Empirical coverage should be consistent with nominal (within 2-3 SE)
    # Expected SE for n=50 is roughly 4%
    assert stats['coverage_ci_lower'] <= nominal_level + 0.1, "Coverage way too high or tool broken"
    assert stats['coverage_ci_upper'] >= nominal_level - 0.1, f"Coverage too low: {stats['empirical_coverage']}"
    
    print("Interval Coverage Test Passed!")

if __name__ == "__main__":
    test_interval_coverage()
