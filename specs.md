# Strategic Memo: Macro-Driven Strategic Asset Allocation System

## Investment Horizon: 12 Months (Rolling)

**Target Assets:** US Equities (S&P 500) / US Bonds (10Y Treasury) / Gold
**Methodology:** Direct Forward Return Prediction via Asset-Specific Machine Learning & Stability Selection

---

## 1. Objective of the Approach

The primary objective is to generate superior risk-adjusted returns by predicting **annualized 12-month forward returns** for major asset classes. The system utilizes a systematic, data-driven framework to map current macroeconomic states directly to future asset performance ($X_t \rightarrow Y_{t+12}$), prioritizing statistical stability and capital preservation over complexity.

---

## 2. Asset-Specific Model Architectures

The system employs "Best-in-Class" modeling architectures tailored to the specific statistical properties and economic drivers of each asset class. These models were selected via a rigorous **Benchmarking Framework** (see benchmarking_engine.py) that evaluates candidates based on Out-of-Sample (OOS) Information Coefficient (IC) and RMSE across historical cycles.

### A. Selection Summary

| Asset Class | Model Architecture | Rationale |
|---|---|---|
| **US Equities** | **Random Forest Regressor** | **Non-Linear Ensemble.** Best suited for equities due to its ability to capture complex, regime-dependent interactions (e.g., asymmetric reactions to rate hikes). Configured with `max_depth=3` to ensure generalization and prevent overfitting to market noise. |
| **US Bonds** | **ElasticNetCV** | **Regularized Linear.** Bonds are driven by a high-dimensional set of overlapping macroeconomic indicators. ElasticNet handles this collinearity via L1/L2 regularization, identifying the most robust fundamental drivers while maintaining linear interpretability. |
| **Gold** | **Linear Regression (OLS)** | **Parsimonious Linear.** Gold's relationship with real rates and currency drivers is historically stable and transparent. A simpler OLS architecture provides the highest parameter stability and prevents the discovery of spurious non-linear patterns. |

### B. Benchmarking Protocol
The `benchmarking_engine.py` runs a continuous "leaderboard" test:
1.  **Candidate Pool:** OLS, Huber (Robust), ElasticNet, Random Forest, and Regime-Switching models.
2.  **Selection Metric:** Highest recursive OOS Spearman Correlation (IC) over the last 20 years.
3.  **Automatic Drift Detection:** If a model's stability metrics (Sign Consistency or Magnitude Stability) degrade, the benchmarking suite flags it for re-evaluation.

---

## 3. Feature Engineering & State Space

The model ingests raw macroeconomic data and transforms it into a "Current State" matrix ($X_t$) using stationary statistical transformations:

**Transformation Pipeline:**
1.  **Trend:** 12-month and 60-month moving averages for medium and long-term cycles.
2.  **Cyclical Position (Z-Score):** Deviation from the 5-year mean, normalized by 5-year volatility.
    * *Formula:* $Z_t = \frac{X_t - \mu_{60}}{\sigma_{60}}$
3.  **Historical Rank:** Rolling 10-year percentile rank (0 to 1) to contextualize current values.
4.  **Momentum (Slope):** Normalized rate of change over 12 and 60 months.

**Data Hygiene:**
* **Winsorization:** All input features are capped at **±3.0 Standard Deviations** (fit on training data) to mitigate outliers.

---

## 4. Stability Selection (Feature Selection)

To ensure trading signals are based on persistent economic drivers rather than noise, the system employs a **Stability Selection** phase prior to final estimation:

1.  **Bootstrapping:** Executes 50 iterations (30 in backtests) of ElasticNet regression ($l1\_ratio=0.5$) on random 70% subsets of the data.
2.  **Persistence Scoring:** Calculates the frequency (%) each variable is selected with a non-zero coefficient.
3.  **Filtering:** Only features exceeding a **Persistence Threshold** (default 0.6) are retained.

---

## 5. Risk Management: Regime Overlay

A "Nowcasting" module monitors stress to adjust portfolio beta dynamically.

### A. Stress Inputs (Rolling 60-Month Z-Scores)
1.  **Credit Stress:** BAA-AAA Corporate Bond Spread.
2.  **Yield Curve Stress:** 10Y Treasury minus Fed Funds Rate (Inverted).

### B. Regime Logic
* **Composite Stress Score:** $0.5 \cdot Z_{Credit} + 0.5 \cdot Z_{Curve}$
* **Operational States:**
    * **CALM:** Stress Score < 1.2 (Full Target Exposure)
    * **WARNING:** 1.2 < Stress Score < 2.0 (Defensive Tilt: -25% Equity Beta)
    * **ALERT:** Stress Score > 2.0 (High Defensive: -50% Equity Beta)

---

## 6. Strategic Allocation Logic

Portfolio weights are derived dynamically based on expected returns and market regime.

* **Base Weights:** Equity 60% / Bonds 30% / Gold 10%.
* **Return Tilt:** Weights are adjusted based on predicted excess returns: 
  $Target = Base \cdot RegimeMult \cdot (1 + (PredictedReturn - RF) \cdot 5)$
* **Hard Constraints:**
    * Equity: 20% - 80% | Bonds: 20% - 50% | Gold: 5% - 25%
* **Normalization:** Final weights are scaled to sum to 100%.

---

## 7. Data Universe

* **Macroeconomic Data:** FRED-MD Database (approx. 128 monthly series).
* **Asset Pricing:**
    * **Equities:** S&P 500 Index.
    * **Bonds:** Synthetic Total Return Index from GS10 assumes constant duration ($D=7.5$).
    * **Gold:** PPI for Gold (WPU1022) as long-term proxy.

---

## 8. Validation Protocol

The strategy is validated using a **Purged Walk-Forward Backtest**:

* **Training Window:** Expanding window with a minimum of 240 months.
* **Leakage Prevention (Purging):** Excludes 12 months immediately preceding the test date.
* **Overlap Adjustment (HAC):** 
    * Forecast errors are adjusted for 12-month overlap by scaling standard errors by $\sqrt{12}$.
    * **Effective Degrees of Freedom:** Adjusted for autocorrelation ($N_{eff} = N / 12$).
* **Performance Metrics:** OOS Information Coefficient (IC) and RMSE.

---

## 9. Point-in-Time (PIT) Prediction Engine

At any specific point $t$ in the backtest (or live operations), the predictive equation for an asset is constructed using a "Selection-Inside-Loop" protocol to eliminate Look-Ahead Bias:

### A. Equation Construction Step-by-Step
1.  **Data Purging:** To predict the $t \rightarrow t+12$ return, the training set is restricted to data available at time $t-12$. This 12-month "purge gap" prevents the model from learning from overlapping return windows that haven't yet completed.
2.  **Dynamic Feature Selection:** A fresh **Stability Selection** (ElasticNet bootstrapping) is executed using only the purged training set. This determines which macro drivers (e.g., Inflation, Credit Spreads) are statistically persistent *at that moment*.
3.  **Local Training:** The chosen model architecture (RF, ElasticNet, or OLS) is fit on the selected features.
4.  **Feature Normalization:** Live macro features at time $t$ are **Winsorized** (capped at ±3.0 SD) using the means and standard deviations calculated from the training set.

### B. Generating the Prediction ($X_t \rightarrow \hat{Y}_{t+12}$)
The finalized equation is applied to the live macro state $X_t$ to generate a single-point estimate of the annualized return:
*   **Clipping:** Predicted returns are capped at **±30%** to prevent extreme signals from outlier macro readings.
*   **Uncertainty Quantification:** Confidence intervals are generated using **HAC-Adjusted Standard Errors**. Because $Y$ is a 12-month overlapping return, the standard error is inflated by $\sqrt{12}$ and degrees of freedom are reduced by a factor of 12 to reflect the "Effective Sample Size."

### C. Signal Interpretation
The resulting value $\hat{Y}_{t+12}$ represents the expected return over the *next* 12 months. This signal is then passed to the strategic allocation logic (Section 6) to determine the portfolio tilt.

---

## 10. Data Architecture & PIT Vintage Building

The system employs a dual-path data architecture to ensure backtest integrity while providing the most accurate live predictions.

### A. Dual-Path Architecture
1.  **Backtest Engine:** Exclusively uses `PIT_Macro_Features.csv`. This ensures the model only "sees" data as it would have appeared to a researcher at any historical point $t$.
2.  **Live Prediction Terminal:** Uses the latest available FRED-MD snapshot (e.g., `2025-11-MD.csv`). This provides the most refined recent data for the current month's signal.

### B. PIT Vintage Construction (Diagonalization)
The historical vintage matrix is built via a "Diagonalization" process implemented in [pit_builder.py](pit_builder.py):

1.  **Sourcing:** The system scrapes historical monthly CSV vintages (dating back to the 1960s) from the St. Louis Fed research database.
2.  **Nowcast Extraction:** For every month $v$ in history, the builder loads the specific vintage file released during that month. It extracts the **last valid row** (the "Nowcast") from that snapshot.
3.  **Feature Harmonization:** Each extracted nowcast is passed through the standard transformation pipeline (Section 3).
4.  **Diagonal Assembly:** These individual nowcasts are stitched together by their **Vintage Date** (release date), creating a master matrix where each row $t$ contains the exact information set available at that time $t$.

### C. Rationale: Information Set Preservation
Standard macro databases (like the current FRED-MD snapshot) are "Backfilled" with revisions. If a backtest used the 2025 snapshot to test a 2008 strategy, it would be using GDP and Employment figures that were revised *years after* the fact, inducing severe **Look-Ahead Bias**.

The PIT architecture preserves the "Information Set" as it actually existed, ensuring that backtested performance is a realistic proxy for future live execution.