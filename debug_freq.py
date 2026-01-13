
import pandas as pd
import numpy as np
from backtester import StrategyBacktester

# Create dummy data with high drift
dates = pd.date_range('2000-01-01', periods=24, freq='MS')
prices = pd.DataFrame({
    'EQUITY': [100 * (1.05**i) for i in range(24)], # High growth
    'BONDS': [100 * (0.98**i) for i in range(24)],  # High decay
    'GOLD': [100 for _ in range(24)]
}, index=dates)

preds = pd.DataFrame(0.1, index=dates, columns=['EQUITY', 'BONDS', 'GOLD'])
lower = pd.DataFrame(0.05, index=dates, columns=['EQUITY', 'BONDS', 'GOLD'])
regime = pd.Series(0.0, index=dates)

bt = StrategyBacktester(prices, preds, lower, regime)

# Run with diff frequencies (High trading cost to make it visible)
res1 = bt.run_strategy("Buy & Hold", weights_dict={'EQUITY': 0.5, 'BONDS': 0.5, 'GOLD': 0.0}, rebalance_freq=1, trading_cost_bps=100)
res12 = bt.run_strategy("Buy & Hold", weights_dict={'EQUITY': 0.5, 'BONDS': 0.5, 'GOLD': 0.0}, rebalance_freq=12, trading_cost_bps=100)

print(f"Final NAV (Freq 1): {res1['equity_curve'].iloc[-1]:.2f}")
print(f"Final NAV (Freq 12): {res12['equity_curve'].iloc[-1]:.2f}")
print(f"Turnover (Freq 1): {res1['metrics']['Turnover']:.4f}")
print(f"Turnover (Freq 12): {res12['metrics']['Turnover']:.4f}")

if res1['equity_curve'].iloc[-1] != res12['equity_curve'].iloc[-1]:
    print("SUCCESS: Frequencies produce different results.")
else:
    print("FAILURE: No difference detected.")
