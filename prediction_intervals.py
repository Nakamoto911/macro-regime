import numpy as np
import pandas as pd
from sklearn.base import clone
from typing import Tuple, Optional, Dict
import warnings
from scipy import stats

class BootstrapPredictionInterval:
    """
    Compute prediction intervals via residual bootstrap.
    Accounts for both parameter uncertainty and residual variance.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.90,
                 n_bootstrap: int = 200,
                 block_length: Optional[int] = None):
        """
        Args:
            confidence_level: Nominal coverage (e.g., 0.90 for 90% CI)
            n_bootstrap: Number of bootstrap replications
            block_length: Block length for block bootstrap (handles autocorrelation)
                         If None, uses sqrt(T) as default
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.block_length = block_length
        
        # Fitted attributes
        self.residuals_ = None
        self.bootstrap_predictions_ = None
        self.base_model_ = None
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the base model and store residuals.
        """
        self.base_model_ = model
        self.X_train_ = X_train
        self.y_train_ = y_train
        
        # Compute in-sample residuals
        y_pred_train = model.predict(X_train)
        self.residuals_ = y_train.values.flatten() - y_pred_train.flatten()
        
        # Set block length if not provided
        if self.block_length is None:
            self.block_length = max(1, int(np.sqrt(len(y_train))))
        
        return self
    
    def predict_interval(self, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for new data.
        """
        n_new = len(X_new)
        if isinstance(X_new, pd.DataFrame):
            X_new_vals = X_new.values
        else:
            X_new_vals = X_new
            
        bootstrap_preds = np.zeros((self.n_bootstrap, n_new))
        
        for b in range(self.n_bootstrap):
            # Step 1: Resample residuals (block bootstrap for autocorrelation)
            resampled_residuals = self._block_bootstrap_residuals()
            
            # Step 2: Create pseudo-observations
            y_pseudo = self.base_model_.predict(self.X_train_) + resampled_residuals[:len(self.y_train_)]
            
            # Step 3: Refit model on pseudo-data
            model_b = clone(self.base_model_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_b.fit(self.X_train_, y_pseudo)
                except Exception:
                    # If fit fails, use base model predictions
                    bootstrap_preds[b, :] = self.base_model_.predict(X_new_vals)
                    continue
            
            # Step 4: Predict on new data
            pred_b = model_b.predict(X_new_vals)
            
            # Step 5: Add residual noise (unsampled residual at pred time t)
            noise_idx = np.random.choice(len(self.residuals_), size=n_new, replace=True)
            bootstrap_preds[b, :] = pred_b + self.residuals_[noise_idx]
        
        self.bootstrap_predictions_ = bootstrap_preds
        
        # Compute percentiles
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
        point_pred = self.base_model_.predict(X_new_vals)
        
        return point_pred, lower, upper
    
    def _block_bootstrap_residuals(self) -> np.ndarray:
        """Generate block-bootstrapped residuals."""
        n = len(self.residuals_)
        block_len = self.block_length
        n_blocks = int(np.ceil(n / block_len))
        
        resampled = []
        for _ in range(n_blocks):
            # Random starting point
            start = np.random.randint(0, n - block_len + 1)
            resampled.extend(self.residuals_[start:start + block_len])
        
        return np.array(resampled[:n])

class ConformalPredictionInterval:
    """
    Conformal prediction intervals with guaranteed coverage.
    Distribution-free and handles any model type.
    """
    
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.calibration_scores_ = None
        self.width_ = None
        
    def fit(self, model, X_train: pd.DataFrame, y_train: pd.Series,
            X_calib: pd.DataFrame, y_calib: pd.Series):
        """
        Fit model and compute calibration scores on held-out data.
        """
        self.base_model_ = model
        
        # Fit model on training data
        self.base_model_.fit(X_train, y_train)
        
        # Compute conformity scores on calibration set
        y_pred_calib = self.base_model_.predict(X_calib)
        self.calibration_scores_ = np.abs(y_calib.values.flatten() - y_pred_calib.flatten())
        
        # Compute quantile for coverage guarantee
        n_calib = len(self.calibration_scores_)
        # Quantity s s.t. ceil((n+1)*alpha)/n
        alpha = self.confidence_level
        q_idx = int(np.ceil((n_calib + 1) * alpha))
        if q_idx > n_calib: q_idx = n_calib # Boundary
        
        sorted_scores = np.sort(self.calibration_scores_)
        self.width_ = sorted_scores[q_idx - 1]
        
        return self
    
    def predict_interval(self, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals with guaranteed coverage.
        """
        point_pred = self.base_model_.predict(X_new)
        lower = point_pred - self.width_
        upper = point_pred + self.width_
        
        return point_pred, lower, upper

class CoverageValidator:
    """
    Empirically validate prediction interval coverage.
    """
    
    def __init__(self, nominal_level: float = 0.90):
        self.nominal_level = nominal_level
        self.coverage_history_ = []
        self.interval_widths_ = []
        
    def record(self, actual: float, lower: float, upper: float):
        """Record a single prediction interval outcome."""
        if pd.isna(actual) or pd.isna(lower) or pd.isna(upper):
            return
        covered = (lower <= actual) and (actual <= upper)
        self.coverage_history_.append(covered)
        self.interval_widths_.append(upper - lower)
        
    def compute_statistics(self) -> Dict:
        """Compute coverage statistics."""
        if not self.coverage_history_:
            return {}
        
        coverage = np.array(self.coverage_history_)
        widths = np.array(self.interval_widths_)
        
        # Empirical coverage
        empirical_coverage = coverage.mean()
        
        # Coverage confidence interval (binomial)
        n = len(coverage)
        se = np.sqrt(empirical_coverage * (1 - empirical_coverage + 1e-10) / n)
        ci_lower = empirical_coverage - 1.96 * se
        ci_upper = empirical_coverage + 1.96 * se
        
        # Interval width statistics
        mean_width = widths.mean()
        median_width = np.median(widths)
        
        return {
            'empirical_coverage': empirical_coverage,
            'coverage_ci_lower': ci_lower,
            'coverage_ci_upper': ci_upper,
            'n_observations': n,
            'mean_interval_width': mean_width,
            'median_interval_width': median_width,
            'nominal_level': self.nominal_level
        }
    
    def is_coverage_acceptable(self, tolerance: float = 0.05) -> bool:
        """
        Check if empirical coverage is within tolerance of nominal.
        """
        stats = self.compute_statistics()
        if not stats:
            return False
        
        return abs(stats['empirical_coverage'] - self.nominal_level) <= tolerance
