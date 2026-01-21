import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings

class NestedTimeSeriesFeatureSelector:
    """
    Feature selection with proper nested cross-validation for time series.
    
    Outer loop: Walk-forward for feature stability assessment
    Inner loop: Time-series CV for alpha selection
    """
    
    def __init__(self, 
                 l1_ratios: List[float] = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                 n_alphas: int = 50,
                 inner_cv_splits: int = 5,
                 min_samples: int = 60,
                 selection_threshold: float = 0.5):
        """
        Args:
            l1_ratios: Elastic net mixing parameters to try
            n_alphas: Number of alpha values to try per l1_ratio
            inner_cv_splits: Number of CV folds for alpha selection
            min_samples: Minimum training samples required
            selection_threshold: Fraction of bootstrap iterations for stability
        """
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.inner_cv_splits = inner_cv_splits
        self.min_samples = min_samples
        self.selection_threshold = selection_threshold
        
        # Fitted attributes
        self.selected_features_ = None
        self.best_alpha_ = None
        self.best_l1_ratio_ = None
        self.selection_frequencies_ = None
        self.cv_results_ = None
        
    def fit(self, y: pd.Series, X: pd.DataFrame, 
            n_bootstrap: int = 20,
            sample_fraction: float = 0.8) -> 'NestedTimeSeriesFeatureSelector':
        """
        Select features using stability selection with nested CV.
        """
        # Input validation
        common_idx = y.index.intersection(X.index)
        y_clean = y.loc[common_idx].dropna()
        X_clean = X.loc[y_clean.index]
        
        if len(y_clean) < self.min_samples:
            # Fallback for small samples
            self.selected_features_ = X_clean.columns[:10].tolist()
            return self
        
        # Remove constant/near-constant features
        X_clean = X_clean.loc[:, X_clean.std() > 1e-10]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            index=X_clean.index,
            columns=X_clean.columns
        )
        
        # Track feature selection frequency
        selection_counts = pd.Series(0, index=X_scaled.columns)
        alpha_history = []
        l1_history = []
        
        for i in range(n_bootstrap):
            # Subsample (respecting time order - take contiguous block)
            n_samples = int(len(y_clean) * sample_fraction)
            if len(y_clean) > n_samples:
                start_idx = np.random.randint(0, len(y_clean) - n_samples + 1)
            else:
                start_idx = 0
                n_samples = len(y_clean)
            
            y_sample = y_clean.iloc[start_idx:start_idx + n_samples]
            X_sample = X_scaled.iloc[start_idx:start_idx + n_samples]
            
            # Inner CV for alpha selection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Gap to avoid leakage from overlapping returns
                cv = TimeSeriesSplit(n_splits=min(self.inner_cv_splits, len(y_sample)//20), gap=12)
                
                if len(y_sample) < 30: continue
                
                model = ElasticNetCV(
                    l1_ratio=self.l1_ratios,
                    n_alphas=self.n_alphas,
                    cv=cv,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=i,
                    n_jobs=-1
                )
                
                try:
                    model.fit(X_sample, y_sample)
                except Exception:
                    continue
            
            # Record selected features (non-zero coefficients)
            selected_mask = model.coef_ != 0
            # Ensure index alignment
            selection_counts[X_sample.columns[selected_mask]] += 1
            
            alpha_history.append(model.alpha_)
            l1_history.append(model.l1_ratio_)
        
        # Compute selection frequencies
        self.selection_frequencies_ = selection_counts / (len(alpha_history) if alpha_history else 1)
        
        # Select features that appear in >= threshold fraction of bootstraps
        self.selected_features_ = self.selection_frequencies_[
            self.selection_frequencies_ >= self.selection_threshold
        ].index.tolist()
        
        # Fallback: if nothing selected, take top 5 by frequency
        if not self.selected_features_ and not self.selection_frequencies_.empty:
            self.selected_features_ = self.selection_frequencies_.nlargest(5).index.tolist()
        
        # Record median alpha/l1_ratio for diagnostics
        self.best_alpha_ = np.median(alpha_history) if alpha_history else 0.01
        self.best_l1_ratio_ = np.median(l1_history) if l1_history else 0.5
        
        self.cv_results_ = {
            'alpha_history': alpha_history,
            'l1_history': l1_history,
            'n_bootstrap': n_bootstrap
        }
        
        return self
    
    def get_selected_features(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_features_ or []
    
    def get_selection_report(self) -> pd.DataFrame:
        """Return detailed selection report."""
        if self.selection_frequencies_ is None:
            return pd.DataFrame()
        
        report = pd.DataFrame({
            'feature': self.selection_frequencies_.index,
            'selection_frequency': self.selection_frequencies_.values,
            'selected': [f in self.selected_features_ for f in self.selection_frequencies_.index]
        })
        
        return report.sort_values('selection_frequency', ascending=False)

class AdaptiveFeatureSelector:
    """
    Feature selector that adapts to different asset classes.
    Uses asset-specific priors on expected number of features.
    """
    
    ASSET_PRIORS = {
        'EQUITY': {
            'expected_features': 15,
            'l1_ratios': [0.5, 0.7, 0.9],  # More regularization
            'threshold': 0.4
        },
        'BONDS': {
            'expected_features': 8,
            'l1_ratios': [0.7, 0.9, 0.95],  # Sparse
            'threshold': 0.5
        },
        'GOLD': {
            'expected_features': 5,
            'l1_ratios': [0.9, 0.95, 1.0],  # Very sparse (near Lasso)
            'threshold': 0.6
        }
    }
    
    def __init__(self, asset_class: str):
        priors = self.ASSET_PRIORS.get(asset_class, self.ASSET_PRIORS['EQUITY'])
        
        self.selector = NestedTimeSeriesFeatureSelector(
            l1_ratios=priors['l1_ratios'],
            selection_threshold=priors['threshold']
        )
        self.asset_class = asset_class
        self.expected_features = priors['expected_features']
        
    def fit(self, y: pd.Series, X: pd.DataFrame, **kwargs) -> 'AdaptiveFeatureSelector':
        self.selector.fit(y, X, **kwargs)
        
        # Warn if selected features far from prior expectation
        n_selected = len(self.selector.selected_features_ or [])
        if n_selected > self.expected_features * 2:
            warnings.warn(
                f"{self.asset_class}: Selected {n_selected} features, "
                f"expected ~{self.expected_features}. Consider stronger regularization."
            )
        
        return self
    
    def get_selected_features(self) -> List[str]:
        return self.selector.get_selected_features()
