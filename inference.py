import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression

class HodrickInference:
    """
    Implements Hodrick (1992) standard errors for overlapping return regressions.
    
    Reference: Hodrick, R. J. (1992). "Dividend Yields and Expected Stock Returns: 
    Alternative Procedures for Inference and Measurement." Review of Financial Studies.
    """
    
    def __init__(self, horizon: int):
        """
        Args:
            horizon: Forecast horizon in periods (e.g., 12 for 12-month returns)
        """
        self.horizon = horizon
        self.coefficients_ = None
        self.hodrick_se_ = None
        self.t_stats_ = None
        self.p_values_ = None
        self.r_squared_ = None
        self.n_obs_ = None
        self.n_features_ = None
        
    def fit(self, y: pd.Series, X: pd.DataFrame) -> 'HodrickInference':
        """
        Fit OLS regression with Hodrick standard errors.
        
        Args:
            y: Overlapping forward returns (T observations)
            X: Predictor matrix (T x K), should include constant if desired
            
        Returns:
            self with fitted attributes
        """
        # Align and clean data
        common_idx = y.index.intersection(X.index)
        y_vals = y.loc[common_idx].values
        X_vals = X.loc[common_idx].values
        
        T, K = X_vals.shape
        h = self.horizon
        
        self.n_obs_ = T
        self.n_features_ = K
        
        # Step 1: OLS estimation
        # Use pseudo-inverse for better stability if X is near-singular
        XtX = X_vals.T @ X_vals
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
            
        beta = XtX_inv @ X_vals.T @ y_vals
        residuals = y_vals - X_vals @ beta
        
        self.coefficients_ = beta
        self.r_squared_ = 1 - np.var(residuals) / (np.var(y_vals) + 1e-10)
        
        # Step 2: Hodrick variance-covariance matrix
        # V(β) = (X'X)^(-1) * S * (X'X)^(-1) where S accounts for MA(h-1) errors
        
        # Compute S = Σ_{j=-(h-1)}^{h-1} w_j Γ_j where w_j is Bartlett weight (Newey-West)
        # to ensure positive definiteness.
        S = np.zeros((K, K))
        
        for j in range(-(h-1), h):
            weight = 1.0 - np.abs(j) / h
            if j >= 0:
                # Covariance at lag j
                for t in range(j, T):
                    S += weight * np.outer(X_vals[t], X_vals[t-j]) * residuals[t] * residuals[t-j]
            else:
                # Negative lag (symmetric contribution)
                for t in range(-j, T):
                    S += weight * np.outer(X_vals[t], X_vals[t-(-j)]) * residuals[t] * residuals[t-(-j)]
        
        S = S / T
        
        # Hodrick variance-covariance matrix
        var_beta = XtX_inv @ S @ XtX_inv * T
        
        # Step 3: Standard errors, t-stats, p-values
        self.hodrick_se_ = np.sqrt(np.maximum(np.diag(var_beta), 1e-10))
        self.t_stats_ = self.coefficients_ / self.hodrick_se_
        
        # Degrees of freedom adjustment (conservative)
        df = max(1, T - K - h)
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df=df))
        
        return self
    
    def summary(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Return coefficient summary table."""
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(self.n_features_)]
            
        return pd.DataFrame({
            'coefficient': self.coefficients_,
            'hodrick_se': self.hodrick_se_,
            't_stat': self.t_stats_,
            'p_value': self.p_values_
        }, index=feature_names)
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values
        return X_new @ self.coefficients_

class NonOverlappingEstimator:
    """
    Estimate predictive regression using non-overlapping returns.
    Apply monthly for signal generation.
    """
    
    def __init__(self, horizon: int, base_model=None):
        """
        Args:
            horizon: Forecast horizon in months
            base_model: Underlying estimator (default: OLS)
        """
        self.horizon = horizon
        self.base_model = base_model
        self.coefficients_ = None
        self.intercept_ = None
        self.coef_std_ = None
        self.feature_names_ = None
        
    def fit(self, y_overlapping: pd.Series, X: pd.DataFrame) -> 'NonOverlappingEstimator':
        """
        Fit on non-overlapping subset, store for monthly prediction.
        
        Strategy: Sample every `horizon` months starting from multiple offsets,
        then average coefficients across offsets for efficiency.
        """
        coefficients_list = []
        intercepts_list = []
        
        # Use all possible non-overlapping subsets (phase shifts)
        for offset in range(self.horizon):
            # Select every horizon-th observation starting at offset
            indices = list(range(offset, len(y_overlapping), self.horizon))
            
            if len(indices) < 20:  # Minimum sample size
                continue
                
            y_subset = y_overlapping.iloc[indices]
            X_subset = X.iloc[indices]
            
            # Clean alignment
            common = y_subset.index.intersection(X_subset.index)
            y_clean = y_subset.loc[common].dropna()
            X_clean = X_subset.loc[common].loc[y_clean.index]
            
            if len(y_clean) < 10: # More lenient for subsets
                continue
            
            # Fit simple OLS (no overlap = standard inference valid)
            model = LinearRegression()
            model.fit(X_clean.values, y_clean.values)
            
            coefficients_list.append(model.coef_)
            intercepts_list.append(model.intercept_)
        
        if not coefficients_list:
            # Fallback to single OLS if phase shifting fails
            model = LinearRegression()
            common = y_overlapping.index.intersection(X.index)
            y_clean = y_overlapping.loc[common].dropna()
            X_clean = X.loc[y_clean.index]
            model.fit(X_clean.values, y_clean.values)
            self.coefficients_ = model.coef_
            self.intercept_ = model.intercept_
            self.coef_std_ = np.zeros_like(model.coef_)
        else:
            # Average across phase shifts (Britten-Jones et al. efficiency argument)
            self.coefficients_ = np.mean(coefficients_list, axis=0)
            self.intercept_ = np.mean(intercepts_list)
            # Compute standard error across phase shifts as robustness check
            self.coef_std_ = np.std(coefficients_list, axis=0)
            
        self.feature_names_ = X.columns.tolist()
        return self
    
    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """Apply monthly for signal generation."""
        preds = X_new.values @ self.coefficients_ + self.intercept_
        return pd.Series(preds, index=X_new.index)
    
    def coefficient_stability_check(self) -> bool:
        """
        Check if coefficients are stable across phase shifts.
        Returns True if coefficient of variation < 0.5 for most features.
        """
        if self.coef_std_ is None:
            return False
        cv = np.abs(self.coef_std_ / (self.coefficients_ + 1e-10))
        return np.mean(cv < 0.5) > 0.7

def valkanov_scaled_t(t_stat: float, R_squared: float, T: int, horizon: int) -> Tuple[float, float]:
    """
    Compute Valkanov (2003) scaled t-statistic for long-horizon regressions.
    
    Args:
        t_stat: Conventional t-statistic
        R_squared: Regression R-squared
        T: Sample size
        horizon: Forecast horizon
        
    Returns:
        scaled_t: Valkanov scaled t-statistic
        p_value: Approximate p-value from simulation-based critical values
    """
    # Scaling factor
    scaling = np.sqrt(T) / horizon
    scaled_t = t_stat / (scaling + 1e-10)
    
    # Critical values (from Valkanov 2003 Table 1, interpolated)
    # These depend on persistence of predictor - using conservative values
    critical_90 = 3.0
    critical_95 = 3.5
    critical_99 = 4.5
    
    if abs(scaled_t) > critical_99:
        p_value = 0.01
    elif abs(scaled_t) > critical_95:
        p_value = 0.05
    elif abs(scaled_t) > critical_90:
        p_value = 0.10
    else:
        p_value = 0.20  # Not significant
    
    return scaled_t, p_value
