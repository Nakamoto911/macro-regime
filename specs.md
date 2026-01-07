# Strategic Memo: High-Dimensional Adaptive Sparse Elastic Net VECM

## Investment Horizon: 5 to 10 years

**Target Assets:** US Equities / Bonds / Gold

**Methodology:** Adaptive Sparse VECM with Elastic Net, Kernel Dictionary, and High-Frequency Nowcasting Overlay

---

## 1. Objective of the Approach

The primary objective of this approach is capital preservation while targeting systematically superior returns to the risk-free rate. The strategy is calibrated to generate target returns between 5% and 10% annually, with rigorous management constraints:

**Light Management and Minimal Rotation:** The strategy favors a dynamic "buy-and-hold" approach. Portfolio rebalancing occurs on a quarterly basis to limit transaction costs and tax impact.

**Flexibility in Crisis Situations:** As an exception to the quarterly rule, the model can trigger immediate rotation if the "Nowcasting" module detects a structural break or imminent systemic risk.

**Minimal Volatility and Limited Drawdowns:** Optimization of allocation to smooth the return curve and early identification of reversal risks to minimize maximum losses.

The success of this management relies on identifying the economic cycle, defined by the interaction between long-term movements (levels) and short-term movements (variations).

---

## 2. Architecture of the Temporal Kernel Dictionary

To capture cycles without rigidity, the model uses Temporal Kernels:

**Lags 1 to 6:** Dense structure for immediate reactivity (essential for crisis detection).

**Anchors 12, 24, 36, 48, 60:** Gaussian-weighted averages to capture cyclical shock waves without information "black holes."

---

## 3. Complementary Module: High-Frequency Nowcasting Overlay

This module acts as a tactical emergency brake. It monitors daily market stress to cut risk exposure before VECM macroeconomic data confirms the crisis.

### A. Stress Indicators (HF Inputs)

**Credit (HY-IG Spread):** Rate spread between risky and solid companies. A sharp widening warns of increased default risk.

**Fear (VIX Structure):** VIX / VIX3M ratio. A shift into Backwardation (VIX > VIX3M) signals immediate panic.

**Liquidity (DXY & 10-Year Rates):** Simultaneous increases in the Dollar and real rates signal global liquidity tightening.

### B. Composite Stress Index (CSI)

Variables are normalized into Z-Scores ($Z_t$) over 12 months.

$$CSI_t = 0.4 \cdot Z_{Credit} + 0.4 \cdot Z_{Vol} + 0.2 \cdot Z_{Liquidity}$$

### C. Decision Matrix (Overlay)

| VECM Signal (Macro) | Nowcast Signal (HF Stress) | Strategic Action |
|---|---|---|
| Bullish (Bull) | Calm | 100% Target Exposure. |
| Bullish (Bull) | High Stress | Tactical Hedge: Reduce exposure by 50%. |
| Bearish (Bear) | Calm | Progressive Exit: Sell on bounces. |
| Bearish (Bear) | High Stress | Full Defense: 100% Cash / Gold / Short-term Bonds. |

---

## 4. Algorithm Steps

**Step 1: Ingestion and Preprocessing**

Load FRED-MD (monthly) and market data (daily). Apply log-level and difference transformations.

Output: Cleaned dataset, separated into levels ($y_t$) and variations ($\Delta y_t$).

**Step 2: Daily Nowcasting Check**

Calculate the Composite Stress Index (CSI) daily.

Output: "Sentinel" status (Alert or Calm) dictating maintenance or reduction of exposure.

**Step 3: Cointegration Rank Identification**

Weighted Johansen test on level variables to identify long-term equilibrium.

Output: Rank $r$ and cointegration vectors $\beta$ (definition of equilibrium value).

**Step 4: Estimation via Adaptive Elastic Net**

Generate final predictive equations by applying dual penalty (L1/L2) on variations filtered by kernels.

Output: Stable equations and Heatmap of coefficients $\Gamma$ (isolates current active drivers).

**Step 5: Extraction of Error Correction Term (ECT)**

Calculate $ECT_t = \beta' y_{t-1}$. Measure of deviation from macroeconomic fair value.

Output: Imbalance score (Error Term) indicating the strength of pull back to equilibrium.

**Step 6: Signal Generation and Rebalancing**

Quarterly synthesis (or immediate in case of Nowcast alert) to adjust allocation.

Output: List of orders and new target portfolio weights.

---

## 5. Stability and Storytelling (Group Effect)

The use of Elastic Net ensures that the economic "narrative" remains coherent:

**Temporal Coherence:** Avoids jumping from one variable to another month to month, reducing unnecessary turnover.

**Grouped Variables:** Signal confirmation through coherent blocks (e.g., the "Labor Market" block is selected in its entirety).

---

## 6. Cycle Signatures (Example of Algorithm Output)

| Block / Module | Grouped Variables Selected | Status / Regime | Strategic Interpretation |
|---|---|---|---|
| Nowcast (HF) | HY-IG Spread & VIX | Calm | No immediate liquidity stress. |
| Labor (Macro) | PAYEMS & USPRIV (Employment) | Plateauing | Late-cycle expansion detected. |
| Output (Macro) | INDPRO & IPFINAL (Production) | Slowdown | Convergence of production indicators. |
| Prices (Macro) | CPI & PPI (Inflation) | "Sticky" | Persistent inflationary pressures. |

---

## 7. Validation Protocol and Resilience

**Hyperparameter Optimization:** Choice of L1/L2 ratio to favor selection stability (Ridge-heavy) over extreme sparsity.

**Cointegration Monitoring:** In case of major $\beta$ instability (structural break), the model switches to "Preservation" mode (Gold/Cash) until the next stabilization cycle.

---

## 8. Data Universe & Statistics

### A. Macroeconomic Data (FRED-MD)

* **Source:** St. Louis Fed FRED-MD Database.
* **Universe:** 128 monthly macroeconomic variables categorized into 8 blocks (Output, Labor, Housing, Consumption, Money, Interest Rates, Prices, Stock Market).
* **Coverage:** Jan 1959 to Oct 2025 (latest available in `2025-11-MD.csv`).
* **Preprocessing:** Log-level transformation and differencing based on McCracken & Ng (2016) codes.

### B. Strategic Asset Data (Spliced History)

* **US Equities (EQUITY):** S&P 500 Index. Sourced from FRED-MD (long history since 1959).
* **Gold (GOLD):** Composite history splicing Precious Metals PPI (FRED: WPU057301, 1960), Gold Import Price Index (FRED: IR14420, 1993), and GLD ETF (Yahoo Finance, 2004).
* **US Bonds (BONDS):** Total Return 10-Year Treasuries. Sourced from FRED (GS10) and Yahoo Finance (IEF). Synthetic history is generated by approximating total returns from Constant Maturity yields: $R_{t} \approx -D \cdot \Delta y_{t} + y_{t-1}/12$ (with duration $D=7.5$), spliced with IEF ETF returns since 2002.

---

**Final Note:** The model prioritizes *stability* over *frenetic trading*. Every signal is cross-validated through the cointegration vector ($\beta$) and the short-term dynamics ($\Gamma$).