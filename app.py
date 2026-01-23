""" 
Macro-Driven Strategic Asset Allocation System - V3
Refactored & Modularized
"""
import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Macro-Driven Strategic Asset Allocation System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modular Imports
from app_utils import (
    load_engine_state, save_engine_state, get_series_descriptions,
    load_full_fred_md_raw, load_fred_appendix, create_theme
)
from data_utils import (
    load_fred_md_data, load_asset_data, prepare_macro_features, compute_forward_returns,
    load_precomputed_features
)
from models import get_live_model_signals_v4, evaluate_regime, compute_allocation
from models_optimized import run_all_assets_backtest, BacktestConfig
from ui_components import (
    render_custom_css, render_allocation_tab, render_drivers_tab,
    render_series_tab, render_prediction_tab, render_diagnostics_tab,
    render_strategy_lab
)

def main():
    if 'theme' not in st.session_state: st.session_state.theme = 'dark'
    if 'sync_triggered' not in st.session_state: st.session_state.sync_triggered = False
    if 'engine_results' not in st.session_state: st.session_state.engine_results = None

    # Persistent Auto-Load
    if not st.session_state.sync_triggered:
        persisted = load_engine_state()
        if persisted and all(k in persisted for k in ['expected_returns', 'confidence_intervals']):
            st.session_state.engine_results = persisted
            st.session_state.sync_triggered = True

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div style="color: #ff6b35; font-family: monospace;">CONFIG & THEME</div>', unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        if t_col1.button("🌙 Dark", width='stretch'): st.session_state.theme = 'dark'; st.rerun()
        if t_col2.button("☀ Light", width='stretch'): st.session_state.theme = 'light'; st.rerun()
        st.markdown("---")
        if st.button("🔄 Clean Cache & Re-Sync", width='stretch'):
            if os.path.exists('engine_state.pkl'): os.remove('engine_state.pkl')
            if os.path.exists('pit_macro_cache.pkl'): os.remove('pit_macro_cache.pkl')
            st.session_state.sync_triggered = False; st.session_state.engine_results = None
            st.cache_data.clear(); st.rerun()
        
        horizon_months = st.slider("Horizon (Months)", 3, 36, 12)
        l1_ratio = st.slider("L1 Ratio", 0.1, 0.9, 0.5)
        min_persistence = st.slider("Min Persistence", 0.3, 0.9, 0.6)
        confidence_level = st.slider("Confidence Level", 0.80, 0.95, 0.90)
        est_window = st.slider("Estimation Window (Years)", 15, 35, 25)
        inf_method = st.selectbox("Inference Method", ["Hodrick (1992)", "Non-Overlapping", "HAC (Legacy)"])
        st.session_state.inference_method = inf_method
        intercept_mode = st.selectbox("Intercept Mode", ["standard", "hybrid"], index=1, help="Hybrid: De-trended Physics + Recalibrated Bias")
        calib_window = st.slider("Calibration Window (Months)", 12, 120, 60)
        alert_thresh = st.slider("Alert Threshold", 1.0, 3.0, 2.0)
        risk_free = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100

    # Theme CSS Injection
    themes = {
        'light': {'bg_primary': "#ffffff", 'bg_secondary': "#f8f9fa", 'bg_tertiary': "#f1f3f5", 'border_color': "#dee2e6", 'text_primary': "#1a1a1a", 'text_secondary': "#4a4a4a", 'text_muted': "#666666", 'header_gradient': "linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%)"},
        'dark': {'bg_primary': "#0a0a0a", 'bg_secondary': "#111111", 'bg_tertiary': "#1a1a1a", 'border_color': "#2a2a2a", 'text_primary': "#e8e8e8", 'text_secondary': "#888888", 'text_muted': "#555555", 'header_gradient': "linear-gradient(180deg, #111111 0%, #0a0a0a 100%)"}
    }[st.session_state.theme]
    render_custom_css(themes)

    st.markdown(f'<div class="header-container"><p class="header-title">◈ MACRO-DRIVEN STRATEGIC ASSET ALLOCATION SYSTEM</p><p class="header-subtitle">FORWARD RETURN PREDICTION · {horizon_months}-MONTH HORIZON</p></div>', unsafe_allow_html=True)
    
    asset_prices = load_asset_data()
    descriptions = get_series_descriptions()
    macro_data_current = load_fred_md_data()
    
    if macro_data_current.empty or asset_prices.empty:
        st.error("Failed to load required data.")
        return
        
    # CRITICAL FIX: Load RAW macro data for VSER reconstruction (Need Levels, not Differences)
    raw_macro, _ = load_full_fred_md_raw()
    # ALIGN INDEX: FRED-MD raw data is often Month Start, but asset data is Month End.
    raw_macro.index = raw_macro.index + pd.offsets.MonthEnd(0)
    
    # Create a specialized dataframe for the VSER logic that guarantees FEDFUNDS is a Level
    vser_macro_data = macro_data_current.copy()
    if 'FEDFUNDS' in raw_macro.columns:
        # Reindex to ensure alignment
        vser_macro_data['FEDFUNDS'] = raw_macro['FEDFUNDS'].reindex(vser_macro_data.index).ffill()
        
    # Sanity check printed for audit
    if 'FEDFUNDS' in vser_macro_data.columns:
        last_ff = vser_macro_data['FEDFUNDS'].iloc[-1]
        # print(f"AUDIT | Current FEDFUNDS Level: {last_ff}")
    
    # Optimized Data Path: Use Precomputed Features
    X_precomputed = load_precomputed_features()
    if X_precomputed is None:
        st.error("Run `python precompute_pit.py` first")
        st.stop()
    
    y_vser = compute_forward_returns(asset_prices, horizon_months=horizon_months, macro_data=vser_macro_data)
    y_nominal = compute_forward_returns(asset_prices, horizon_months=horizon_months, vol_scale=False, excess_return=False)
    
    # Align for live prediction & backtest
    valid_idx = X_precomputed.index.intersection(y_vser.index)
    X_live, y_live = X_precomputed.loc[valid_idx], y_vser.loc[valid_idx]
    
    # Legacy compatibility for Lab tab
    X_backtest, y_backtest = X_live, y_live
    
    if not st.session_state.sync_triggered:
        st.warning("ALPHA ENGINE OFFLINE. Synchronize to generate insights.")
        if st.button("🚀 START ALPHA ENGINE & RUN BACKTEST", width='stretch', type="primary"):
            st.session_state.sync_triggered = True; st.rerun()
        return

    if st.session_state.engine_results is None:
        with st.status("🚀 Running Optimized Alpha Engine...", expanded=True) as status:
            st.write("Generating live signals...")
            expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats = get_live_model_signals_v4(
                y_live, X_live, l1_ratio, min_persistence, est_window, horizon_months, 
                confidence_level=confidence_level, prices=asset_prices, macro_data=vser_macro_data,
                intercept_mode=intercept_mode, calibration_window=calib_window
            )
            
            st.write("Executing optimized backtest...")
            config = BacktestConfig(min_train_months=240, horizon_months=horizon_months)
            results, selections, coverage = run_all_assets_backtest(
                y_vser, X_precomputed, config,
                y_nominal_all=y_nominal, prices_all=asset_prices, macro_data=vser_macro_data
            )
            
            st.session_state.engine_results = {
                'expected_returns': expected_returns, 'confidence_intervals': confidence_intervals,
                'driver_attributions': driver_attributions, 'stability_results_map': stability_results_map,
                'model_stats': model_stats, 
                'prediction_results': results, 
                'prediction_selection': selections, 
                'coverage_stats': coverage,
                'backtest_results': results, 
                'backtest_selection': selections
            }
            save_engine_state(st.session_state.engine_results)
            status.update(label="✅ Engine Core & Backtest Ready!", state="complete")

    # Unpack State
    res = st.session_state.engine_results
    expected_returns, confidence_intervals = res['expected_returns'], res['confidence_intervals']
    driver_attributions, stability_results_map = res['driver_attributions'], res['stability_results_map']
    model_stats = res['model_stats']
    
    regime_status, stress_score, stress_indicators = evaluate_regime(macro_data_current, alert_threshold=alert_thresh)
    target_weights = compute_allocation(expected_returns, regime_status, risk_free_rate=risk_free)
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ALLOCATION", "STABLE DRIVERS", "SERIES", "PREDICTION", "DIAGNOSTIC", "BACKTEST"])
    with tab1: render_allocation_tab(horizon_months, expected_returns, confidence_intervals, y_live, target_weights, regime_status, driver_attributions, confidence_level)
    with tab2: render_drivers_tab(min_persistence, model_stats, driver_attributions, stability_results_map, X_live, y_vser, descriptions, horizon_months)
    with tab3: 
        df_full, transform_codes = load_full_fred_md_raw()
        appendix = load_fred_appendix()
        render_series_tab(df_full, transform_codes, appendix, driver_attributions, y_live, descriptions)
    with tab4: render_prediction_tab(res['prediction_results'], y_live, horizon_months, min_persistence, l1_ratio, confidence_level, X_live, asset_prices, macro_data_current, alert_thresh, y_nominal_live=y_nominal, model_stats=res['model_stats'])
    with tab5: render_diagnostics_tab(stress_indicators, stability_results_map, res['prediction_selection'], res['coverage_stats'], descriptions)
    with tab6: render_strategy_lab(asset_prices, res['prediction_results'], y_backtest, X_backtest, horizon_months, min_persistence, l1_ratio, confidence_level, macro_data_current, risk_free)

if __name__ == "__main__":
    main()