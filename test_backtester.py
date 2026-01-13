import pandas as pd
import numpy as np
from backtester import StrategyBacktester

def test_backtester():
    # Create dummy data
    dates = pd.date_range('2010-01-01', periods=120, freq='ME')
    prices = pd.DataFrame({
        'EQUITY': np.cumprod(1 + np.random.normal(0.01, 0.04, 120)),
        'BONDS': np.cumprod(1 + np.random.normal(0.003, 0.01, 120)),
        'GOLD': np.cumprod(1 + np.random.normal(0.005, 0.03, 120))
    }, index=dates)
    
    preds = pd.DataFrame({
        'EQUITY': np.random.normal(0.08, 0.05, 120),
        'BONDS': np.random.normal(0.04, 0.02, 120),
        'GOLD': np.random.normal(0.05, 0.04, 120)
    }, index=dates)
    
    lower_ci = preds - 0.05
    regime = pd.Series(np.random.normal(1.0, 1.0, 120), index=dates)
    
    bt = StrategyBacktester(prices, preds, lower_ci, regime)
    
    # Test Buy & Hold
    res_bh = bt.run_strategy("Buy & Hold", weights_dict={'EQUITY': 0.6, 'BONDS': 0.3, 'GOLD': 0.1})
    print("Buy & Hold Metrics:", res_bh['metrics'])
    
    # Test Max Return
    res_mr = bt.run_strategy("Max Return", max_weight=0.8)
    print("Max Return Metrics:", res_mr['metrics'])
    
    # Test Min Vol
    res_mv = bt.run_strategy("Min Volatility", cov_lookback=36)
    print("Min Volatility Metrics:", res_mv['metrics'])

if __name__ == "__main__":
    test_backtester()
