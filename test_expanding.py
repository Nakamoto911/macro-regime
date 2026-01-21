import pandas as pd
import numpy as np
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

def test_expanding_features():
    df = pd.DataFrame(np.random.randn(10, 2), columns=['A', 'B'])
    expanding = ExpandingWindowFeatures(
        functions=["mean", "min", "max"],
        variables=['A', 'B']
    )
    res = expanding.fit_transform(df)
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Result columns: {res.columns.tolist()}")
    print(f"Result preview:\n{res.head()}")

if __name__ == "__main__":
    test_expanding_features()
