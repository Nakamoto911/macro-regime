
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from benchmarking_engine import run_benchmarking_engine

def test_hit_rate():
    print("Setting up dummy data...")
    # Create dummy data
    dates = pd.date_range(start='2000-01-01', periods=100, freq='M')
    X = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=['F1', 'F2', 'F3', 'F4', 'F5'])
    y = pd.Series(np.random.randn(100), index=dates, name='Target')
    
    print("Running short benchmark...")
    # Run benchmark for a few steps
    # start_idx + horizon must be < len(X)
    # 50 + 12 = 62. Length is 100.
    # step = 12.
    # loops: 50, 62, 74, 86...
    try:
        results, _ = run_benchmarking_engine(X, y, start_idx=50, step=12, horizon=12)
        
        print("\nBenchmark Results:")
        print(results)
        
        if "OOS Hit Rate" in results.columns:
            print("\nSUCCESS: 'OOS Hit Rate' found in results.")
            # Check if values are valid (0 to 1)
            hit_rates = results["OOS Hit Rate"]
            if hit_rates.between(0, 1).all():
                print("SUCCESS: Hit Rate values are between 0 and 1.")
            else:
                print(f"FAILURE: Hit Rate values out of range: {hit_rates}")
        else:
            print("\nFAILURE: 'OOS Hit Rate' NOT found in results.")
            
    except Exception as e:
        print(f"\nFAILURE: Exception during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hit_rate()
