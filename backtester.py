import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Winsorizer:
    """Caps features at a specific Z-score threshold."""
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X, y=None):
        self.means_ = X.mean()
        self.stds_ = X.std()
        return self
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def transform(self, X):
        if self.means_ is None or self.stds_ is None:
            return X
        return X.clip(lower=self.means_ - self.threshold * self.stds_,
                      upper=self.means_ + self.threshold * self.stds_,
                      axis=1)

class StrategyBacktester:
    def __init__(self, prices_df, predictions_df, lower_ci_df, regime_signals):
        """
        Initialize the backtester with required data.
        Aligns all data to common date index.
        """
        # Align all data
        common_idx = prices_df.index.intersection(predictions_df.index).intersection(lower_ci_df.index).intersection(regime_signals.index)
        if common_idx.empty:
            self.prices = pd.DataFrame()
            self.returns = pd.DataFrame()
            self.common_idx = pd.Index([])
            return

        self.prices = prices_df.loc[common_idx]
        self.predictions = predictions_df.loc[common_idx]
        self.lower_ci = lower_ci_df.loc[common_idx]
        self.regime = regime_signals.loc[common_idx]
        
        # Calculate monthly returns for simulation
        self.returns = self.prices.pct_change().dropna()
        self.common_idx = self.returns.index
        self.assets = ['EQUITY', 'BONDS', 'GOLD']
        
        # Filter other data to match returns index
        self.predictions = self.predictions.loc[self.common_idx]
        self.lower_ci = self.lower_ci.loc[self.common_idx]
        self.regime = self.regime.loc[self.common_idx]

    def _simulate_engine(self, target_weights, trading_cost_bps=10, initial_capital=10000, rebalance_freq=1):
        """
        Vectorized simulation engine.
        target_weights: DataFrame of periodic target weights.
        """
        if self.returns.empty:
            return {
                'equity_curve': pd.Series([initial_capital]),
                'weights': pd.DataFrame(columns=self.assets + ['CASH']),
                'metrics': {}
            }
        
        n = len(self.returns)
        equity_curve = np.zeros(n + 1)
        equity_curve[0] = initial_capital
        
        current_weights = np.zeros(len(self.assets))
        applied_weights_history = []
        total_turnover = 0.0
        total_costs_paid = 0.0
        
        # Cost factor
        cost_pct = trading_cost_bps / 10000.0
        
        returns_arr = self.returns[self.assets].values
        target_weights_arr = target_weights[self.assets].values
        
        for t in range(n):
            # 1. Rebalance? (at start of period t)
            # Weights decided at end of t-1 (or start of t) apply to returns of period t (t+1 in price terms)
            if t % rebalance_freq == 0:
                new_target = target_weights_arr[t]
                # Calculate turnover
                turnover_event = np.sum(np.abs(new_target - current_weights))
                total_turnover += turnover_event
                # Apply cost
                cost_amount = equity_curve[t] * (turnover_event * cost_pct)
                total_costs_paid += cost_amount
                equity_curve[t] -= cost_amount
                current_weights = new_target
            
            # 2. Apply returns
            # r_t is return from end of t-1 to end of t
            period_return = np.sum(current_weights * returns_arr[t])
            # Remaining cash return (assuming 0 for simplicity, or we could add risk_free_rate/12)
            cash_weight = 1.0 - np.sum(current_weights)
            # period_return += cash_weight * (risk_free_rate / 12.0) # If we wanted cash yield
            
            equity_curve[t+1] = equity_curve[t] * (1 + period_return)
            
            # Update current weights due to market movement (drift)
            # After returns, the relative weights change
            denom = (1 + period_return)
            if denom != 0:
                current_weights = (current_weights * (1 + returns_arr[t])) / denom
            
            applied_weights_history.append(list(current_weights) + [cash_weight])

        weights_df = pd.DataFrame(applied_weights_history, index=self.common_idx, columns=self.assets + ['CASH'])
        equity_series = pd.Series(equity_curve, index=[self.prices.index[0]] + list(self.common_idx))
        
        # Annualized turnover
        years = len(self.common_idx) / 12.0
        annualized_turnover = (total_turnover / years) if years > 0 else 0
        
        metrics = self._calculate_metrics(equity_series, weights_df, annualized_turnover)
        metrics['Rebalancing Cost'] = total_costs_paid
        
        return {
            'equity_curve': equity_series,
            'weights': weights_df,
            'metrics': metrics
        }

    def _calculate_metrics(self, equity_curve, weights_df, turnover=0):
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            return {}
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        years = len(returns) / 12.0
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(12)
        sharpe = (cagr - 0.04) / vol if vol > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(12) if not downside_returns.empty else 0
        sortino = (cagr - 0.04) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
        
        return {
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Calmar': calmar,
            'Turnover': turnover
        }

    def run_strategy(self, strategy_type, **kwargs):
        """Entry point for running a strategy."""
        if strategy_type == "Max Return":
            weights = self._calc_weights_max_return(**kwargs)
        elif strategy_type == "Min Volatility":
            weights = self._calc_weights_min_vol(**kwargs)
        elif strategy_type == "Min Drawdown":
            weights = self._calc_weights_min_drawdown(**kwargs)
        elif strategy_type == "Min Loss":
            weights = self._calc_weights_min_loss(**kwargs)
        elif strategy_type == "Buy & Hold":
            weights = self._calc_weights_buy_hold(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
            
        rebalance_freq = kwargs.get('rebalance_freq', 1)
        trading_cost_bps = kwargs.get('trading_cost_bps', 10)
        initial_capital = kwargs.get('initial_capital', 10000)
        
        return self._simulate_engine(weights, trading_cost_bps, initial_capital, rebalance_freq)

    def _calc_weights_max_return(self, max_weight=0.8, risk_free_rate=0.04, top_n=1, weighting_scheme="Equal", **kwargs):
        """
        Allocates to assets with highest predicted returns if > rf.
        top_n: Number of assets to include.
        weighting_scheme: 'Equal' or 'Proportional'.
        """
        rf_monthly = risk_free_rate
        weights = pd.DataFrame(0.0, index=self.common_idx, columns=self.assets)
        
        for date, row in self.predictions.iterrows():
            # Filter assets above risk-free rate
            valid_assets = row[row > rf_monthly].sort_values(ascending=False)
            
            if not valid_assets.empty:
                # Take top N
                selected = valid_assets.head(top_n)
                n_selected = len(selected)
                
                if weighting_scheme == "Equal":
                    # Split max_weight equally among selected
                    for asset in selected.index:
                        weights.at[date, asset] = max_weight / n_selected
                else: # Proportional
                    # Split max_weight based on relative predicted returns
                    total_pred = selected.sum()
                    if total_pred > 0:
                        for asset, val in selected.items():
                            weights.at[date, asset] = max_weight * (val / total_pred)
                    else:
                        # Fallback to equal if sum is 0
                        for asset in selected.index:
                            weights.at[date, asset] = max_weight / n_selected
                            
        return weights

    def _calc_weights_min_vol(self, cov_lookback=60, **kwargs):
        """Global Min Variance strategy."""
        weights = pd.DataFrame(0.0, index=self.common_idx, columns=self.assets)
        
        for i in range(len(self.common_idx)):
            date = self.common_idx[i]
            # Use data up to date - 1 month for covariance
            window = self.returns.iloc[max(0, i-cov_lookback):i]
            if len(window) < 12:
                # Default to equal weight if not enough data
                weights.loc[date] = 1.0 / len(self.assets)
                continue
                
            cov = window[self.assets].cov().values * 12
            
            def obj(w):
                return np.dot(w.T, np.dot(cov, w))
            
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in self.assets]
            res = minimize(obj, [1.0/3]*3, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                weights.loc[date] = res.x
            else:
                weights.loc[date] = 1.0 / len(self.assets)
        return weights

    def _calc_weights_min_drawdown(self, alert_threshold=2.0, defensive_equity_cap=0.20, defensive_cash_floor=0.50, **kwargs):
        """MVO with Regime Switching."""
        weights = pd.DataFrame(0.0, index=self.common_idx, columns=self.assets)
        
        for i in range(len(self.common_idx)):
            date = self.common_idx[i]
            stress = self.regime.loc[date]
            
            # Simple MVO or heuristic
            if stress > alert_threshold:
                # Defensive
                weights.at[date, 'EQUITY'] = min(0.1, defensive_equity_cap)
                weights.at[date, 'BONDS'] = 1.0 - defensive_cash_floor - weights.at[date, 'EQUITY']
                # GOLD remains 0 or small? Heuristic:
                weights.at[date, 'GOLD'] = 0.0
            else:
                # Normal MVO (Simplified to standard weights for now, or could implement full MVO)
                # Let's do a simple risk-parity or static 60/30/10 as "normal"
                weights.loc[date] = [0.6, 0.3, 0.1]
                
        return weights

    def _calc_weights_min_loss(self, confidence_threshold=0.0, rank_by="Lower CI", max_weight=0.8, top_n=3, weighting_scheme="Equal", **kwargs):
        """
        Allocates to assets based on high-conviction safety.
        Invests only if Lower CI > confidence_threshold.
        If multiple qualify, ranks them and applies diversification logic.
        """
        weights = pd.DataFrame(0.0, index=self.common_idx, columns=self.assets)
        
        for date, row_ci in self.lower_ci.iterrows():
            # 1. Identify qualified assets (Lower CI > threshold)
            qualified = row_ci[row_ci > confidence_threshold]
            
            if not qualified.empty:
                # 2. Ranking logic
                if rank_by == "Expected Return":
                    # Rank qualified assets by their predicted mean return
                    pred_row = self.predictions.loc[date]
                    selected = pred_row[qualified.index].sort_values(ascending=False).head(top_n)
                else: # Default: Rank by Lower CI
                    selected = qualified.sort_values(ascending=False).head(top_n)
                
                n_selected = len(selected)
                if n_selected > 0:
                    if weighting_scheme == "Equal":
                        for asset in selected.index:
                            weights.at[date, asset] = max_weight / n_selected
                    else: # Proportional
                        # Split max_weight based on the ranking metric values
                        # If rank_by is Expected Return, use predicted returns
                        # If rank_by is Lower CI, use those values
                        total_score = selected.sum()
                        if total_score > 0:
                            for asset, val in selected.items():
                                weights.at[date, asset] = max_weight * (val / total_score)
                        else:
                            for asset in selected.index:
                                weights.at[date, asset] = max_weight / n_selected
                                
        return weights

    def _calc_weights_buy_hold(self, weights_dict=None, **kwargs):
        """Static weights."""
        if weights_dict is None:
            weights_dict = {'EQUITY': 0.6, 'BONDS': 0.3, 'GOLD': 0.1}
        
        weights = pd.DataFrame(0.0, index=self.common_idx, columns=self.assets)
        for asset, w in weights_dict.items():
            if asset in self.assets:
                weights[asset] = w
        return weights
