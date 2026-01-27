import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from app_utils import create_theme, NBER_RECESSIONS, TRANSFORMATION_LABELS, save_engine_state
from viz_utils import (
    plot_allocation, plot_assets, plot_ect, plot_fred_series,
    plot_feature_heatmap, plot_stability_boxplot, plot_trend_bars,
    plot_driver_vs_asset, plot_driver_scatter, plot_rolling_correlation,
    plot_quintile_analysis, plot_combined_driver_analysis,
    plot_variable_survival, plot_backtest, plot_rolling_correlation as plot_rolling_ic_chart
)
from models import get_historical_backtest, get_historical_stress
from data_utils import apply_transformation
from backtester import StrategyBacktester
from prediction_metrics import compute_all_metrics, get_quality_rating, format_metric_value, compute_rolling_ic, compute_calibration_data, compute_quintile_analysis, compute_ic_by_regime, generate_llm_report, construct_model_summary

def render_custom_css(themes):
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    :root {{
        --bg-primary: {themes['bg_primary']};
        --bg-secondary: {themes['bg_secondary']};
        --bg-tertiary: {themes['bg_tertiary']};
        --border-color: {themes['border_color']};
        --text-primary: {themes['text_primary']};
        --text-secondary: {themes['text_secondary']};
        --text-muted: {themes['text_muted']};
        --header-gradient: {themes['header_gradient']};
        --accent-orange: #ff6b35;
        --accent-green: #00d26a;
        --accent-red: #ff4757;
        --accent-blue: #4da6ff;
        --accent-gold: #ffd700;
    }}
    
    .stApp {{ background-color: var(--bg-primary); font-family: 'IBM Plex Sans', sans-serif; }}
    .main .block-container {{ padding: 1rem 2rem; max-width: 100%; }}
    .header-container {{ background: var(--header-gradient); border-bottom: 1px solid var(--border-color); padding: 1rem 0; margin-bottom: 1.5rem; }}
    .header-title {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: var(--accent-orange); letter-spacing: 0.5px; margin: 0; }}
    .header-subtitle {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-secondary); letter-spacing: 1px; margin-top: 0.25rem; }}
    .panel-header {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600; color: var(--text-secondary); letter-spacing: 1.5px; text-transform: uppercase; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-bottom: 0.75rem; }}
    .metric-card {{ background-color: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 2px; padding: 0.75rem; text-align: center; }}
    .metric-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; }}
    .metric-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 600; color: var(--text-primary); margin-top: 0.25rem; }}
    .metric-value.positive {{ color: var(--accent-green); }}
    .metric-value.negative {{ color: var(--accent-red); }}
    .metric-value.warning {{ color: var(--accent-orange); }}
    .metric-value.excellent {{ color: var(--accent-green); }}
    .metric-value.good {{ color: var(--accent-blue); }}
    .metric-value.marginal {{ color: var(--accent-orange); }}
    .metric-value.poor {{ color: var(--accent-red); }}
    .metric-quality {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; margin-top: 0.25rem; font-weight: 500; }}
    .metric-quality.excellent {{ color: var(--accent-green); }}
    .metric-quality.good {{ color: var(--accent-blue); }}
    .metric-quality.marginal {{ color: var(--accent-orange); }}
    .metric-quality.poor {{ color: var(--accent-red); }}
    .metric-quality.neutral {{ color: var(--text-muted); }}
    input {{ color: var(--text-primary) !important; background-color: var(--bg-tertiary) !important; }}
</style>
""", unsafe_allow_html=True)

def generate_narrative(expected_returns, driver_attributions, regime_status):
    narratives = []
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        if asset not in expected_returns or asset not in driver_attributions: continue
        exp_ret = expected_returns[asset]; attr = driver_attributions[asset]
        tailwinds = attr[attr['Impact'] > 0.005].sort_values('Impact', ascending=False).head(2)
        headwinds = attr[attr['Impact'] < -0.005].sort_values('Impact', ascending=True).head(2)
        tw = ', '.join([f.split('_')[0] for f in tailwinds['feature'].tolist()]) or 'none'
        hw = ', '.join([f.split('_')[0] for f in headwinds['feature'].tolist()]) or 'none'
        outlook = "bullish" if exp_ret > 0.05 else "cautious" if exp_ret < -0.02 else "neutral"
        narratives.append(f"**{asset}**: Outlook is **{outlook}** ({exp_ret:.1%}). Tailwinds: *{tw}*. Headwinds: *{hw}*.")
    regime_note = {'CALM': 'Stable regime.', 'WARNING': 'Elevated stress.', 'ALERT': 'High stress positioning.'}.get(regime_status, '')
    return '  \n'.join(narratives) + f'  \n\n{regime_note}'



def render_allocation_tab(horizon_months, expected_returns, confidence_intervals, y_live, target_weights, regime_status, driver_attributions, confidence_level):
    st.markdown(f'<div class="panel-header">EXPECTED {horizon_months}M RETURNS & STRATEGIC POSITIONING</div>', unsafe_allow_html=True)
    summary_data = []
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        exp = expected_returns.get(asset, 0); ci = confidence_intervals.get(asset, [0,0])
        avg_ret = y_live[asset].mean() if asset in y_live.columns else 0
        diff = exp - avg_ret
        rec = "OVERWEIGHT" if diff > 0.01 else "UNDERWEIGHT" if diff < -0.01 else "NEUTRAL"
        summary_data.append({'Asset': asset, 'Expected Return': f"{exp:.1%}", f'{int(confidence_level*100)}% CI': f"[{ci[0]:.1%}, {ci[1]:.1%}]", 'vs Historical': f"{diff:+.1%}", 'Recommendation': rec, 'Target Weight': f"{target_weights.get(asset, 0):.0%}"})
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, width='stretch')
    c1, c2 = st.columns([1, 2])
    with c1: st.plotly_chart(plot_allocation(target_weights), width='stretch', key="alloc_pie")
    with c2:
        st.markdown('<div class="panel-header">STRATEGIC RATIONALE</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:var(--bg-secondary); border:1px solid var(--border-color); padding:1rem; border-radius:2px; font-size:0.85rem;">{generate_narrative(expected_returns, driver_attributions, regime_status)}</div>', unsafe_allow_html=True)

def render_drivers_tab(min_persistence, model_stats, driver_attributions, stability_results_map, X_live, y_forward, descriptions, horizon_months):
    st.markdown(f'<div class="panel-header">STABLE MACRO DRIVERS (PERSISTENCE > {int(min_persistence*100)}%)</div>', unsafe_allow_html=True)
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        with st.expander(f"Drivers for {asset}", expanded=(asset=='EQUITY')):
            st.markdown(construct_model_summary(asset, model_stats))
            attr = driver_attributions.get(asset, pd.DataFrame())
            if not attr.empty:
                stable_feats = stability_results_map.get(asset, {}).get('stable_features', [])
                df_disp = attr[attr['feature'].isin(stable_feats)].copy()
                
                # Add Description column
                df_disp['Description'] = df_disp['feature'].apply(lambda x: descriptions.get(x.split('_')[0], x.split('_')[0]))
                
                sel = st.dataframe(df_disp[['feature', 'Description', 'Signal', 'Impact', 'Weight', 'State', 'Link']], hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key=f"sel_{asset}")
                if sel.get('selection', {}).get('rows'):
                    feat = df_disp.iloc[sel['selection']['rows'][0]]['feature']
                    st.plotly_chart(plot_combined_driver_analysis(X_live, y_forward, feat, asset, descriptions, horizon_months=horizon_months), width='stretch', key=f"driver_analysis_{asset}")

def render_series_tab(df_full, transform_codes, appendix, driver_attributions, y_live, descriptions):
    sel_col1, sel_col2, sel_col3, sel_col4, sel_col5, sel_col6 = st.columns([1.5, 1, 1, 1, 2, 3])
    with sel_col1: st.markdown("**Impact on:**")
    f_equity = sel_col2.checkbox("Equity", value=True)
    f_bonds = sel_col3.checkbox("Bonds", value=False)
    f_gold = sel_col4.checkbox("Gold", value=False)
    with sel_col5: st.markdown("**Display Mode:**")
    mode = sel_col6.radio("Mode", ["Raw", "Transformed", "Normalized"], horizontal=True, label_visibility="collapsed")
    
    selected_assets = [a for a in ['EQUITY', 'BONDS', 'GOLD'] if (f_equity if a=='EQUITY' else f_bonds if a=='BONDS' else f_gold)]
    active_series = {}
    if selected_assets:
        for a in selected_assets:
            attr = driver_attributions.get(a, pd.DataFrame())
            for _, row in attr.iterrows():
                active_series[row['feature'].split('_')[0]] = max(active_series.get(row['feature'].split('_')[0], 0), abs(row['Weight']))
        sorted_series = sorted(active_series.items(), key=lambda x: x[1], reverse=True)[:15]
        series_to_plot = [s[0] for s in sorted_series]
    else: series_to_plot = sorted(df_full.columns)[:15]

    for col in series_to_plot:
        if col in df_full.columns:
            st.plotly_chart(plot_fred_series(df_full[col], col, descriptions.get(col, ""), is_transformed=(mode!="Raw")), width='stretch', key=f"fred_{col}")

def render_prediction_tab(prediction_results, y_live, horizon_months, min_persistence, l1_ratio, confidence_level, X_live, asset_prices=None, macro_data_current=None, alert_threshold=2.0, y_nominal_live=None, model_stats=None):
    if prediction_results is None:
        if st.button("📊 Run Prediction Model Validation", type="primary"):
            res, sel, cov = get_historical_backtest(y_live, X_live, 240, horizon_months, 12, min_persistence, l1_ratio, confidence_level, prices=asset_prices, macro_data=macro_data_current)
            st.session_state.engine_results.update({'prediction_results': res, 'prediction_selection': sel, 'coverage_stats': cov})
            save_engine_state(st.session_state.engine_results); st.rerun()
    else:
        st.markdown('<div class="panel-header">PREDICTION ACCURACY & CALIBRATION</div>', unsafe_allow_html=True)
        
        # LLM Export Section
        with st.sidebar:
            st.markdown("---")
            st.markdown('<div style="color: #00d26a; font-family: monospace; font-size: 0.8rem;">SHARE TO LLM</div>', unsafe_allow_html=True)
            if st.button("📋 Copy Prediction Report", width='stretch', type="secondary"):
                report_md = generate_llm_report(prediction_results, y_live, confidence_level, model_stats, y_nominal=y_nominal_live)
                st.session_state.llm_report = report_md
                st.toast("Report generated! See below.")

        if 'llm_report' in st.session_state:
            with st.expander("📝 LLM-FRIENDLY REPORT", expanded=True):
                st.markdown("Copy the content below to share with your LLM:")
                st.code(st.session_state.llm_report, language="markdown")
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=st.session_state.llm_report,
                    file_name="macro_model_report.md",
                    mime="text/markdown"
                )
                if st.button("Close Report"):
                    del st.session_state.llm_report
                    st.rerun()

        asset = st.selectbox("Asset", ['EQUITY', 'BONDS', 'GOLD'])
        oos = prediction_results.get(asset, pd.DataFrame())
        
        if not oos.empty:
            # CRITICAL FIX: Use Nominal Returns for "Actual" plotting if provided
            if y_nominal_live is not None and asset in y_nominal_live.columns:
                actual = y_nominal_live[asset].loc[oos.index]
            else:
                actual = y_live[asset].loc[oos.index]
            metrics = compute_all_metrics(
                actual=actual,
                predicted=oos['predicted_return'],
                lower_ci=oos['lower_ci'],
                upper_ci=oos['upper_ci'],
                nominal_coverage=confidence_level
            )
            
            # Primary Metrics Grid
            col1, col2, col3, col4 = st.columns(4)
            
            primary_metrics = [
                ('oos_r2', 'OOS R²', 'oos_r2'),
                ('ic', 'Info. Coefficient', 'ic'),
                ('hit_rate', 'Hit Rate', 'hit_rate'),
                ('coverage', f'Coverage ({confidence_level:.0%})', 'coverage')
            ]
            
            for col, (attr, label, metric_key) in zip([col1, col2, col3, col4], primary_metrics):
                val = getattr(metrics, attr)
                qual_label, css = get_quality_rating(metric_key, val, confidence_level)
                col.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value {css}">{format_metric_value(metric_key, val)}</div>
                        <div class="metric-quality {css}">● {qual_label}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Main Backtest Chart
            st.plotly_chart(plot_backtest(actual, oos['predicted_return'], oos['lower_ci'], oos['upper_ci'], confidence_level), width='stretch', key=f"bt_{asset}")
            
            # Secondary Metrics Expander
            with st.expander("📊 Error Metrics & Significance"):
                s1, s2, s3, s4, s5 = st.columns(5)
                sec_metrics = [
                    ('rmse', 'RMSE', 'rmse'),
                    ('mae', 'MAE', 'mae'),
                    ('bias', 'Bias', 'bias'),
                    ('interval_width', 'Avg Width', 'interval_width'),
                    ('ic_tstat', 'IC t-stat', 'ic_tstat')
                ]
                for col, (attr, label, m_key) in zip([s1, s2, s3, s4, s5], sec_metrics):
                    val = getattr(metrics, attr)
                    qual, css = get_quality_rating(m_key, val)
                    col.metric(label, format_metric_value(m_key, val), delta=qual if m_key=='ic_tstat' else None, delta_color="normal")
            
            # Advanced Diagnostics Expander
            with st.expander("🔬 Model Diagnostics"):
                d1, d2 = st.columns(2)
                
                with d1:
                    # Rolling IC
                    roll_ic = compute_rolling_ic(actual, oos['predicted_return'])
                    fig_ic = go.Figure()
                    fig_ic.add_trace(go.Scatter(x=roll_ic.index, y=roll_ic, name="Rolling IC (36M)", line=dict(color='#4da6ff')))
                    fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_ic.update_layout(title="Rolling Information Coefficient (36M)", height=350, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
                    st.plotly_chart(fig_ic, width='stretch', key=f"roll_ic_{asset}")
                    
                    # Quintile Spread
                    quintiles = compute_quintile_analysis(actual, oos['predicted_return'])
                    fig_q = go.Figure(go.Bar(x=quintiles.index, y=quintiles['mean_return'], marker_color='#ff6b35'))
                    fig_q.update_layout(title="Mean Return by Prediction Quintile", height=350, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
                    st.plotly_chart(fig_q, width='stretch', key=f"quintile_{asset}")

                with d2:
                    # Calibration Plot
                    calib = compute_calibration_data(actual, oos['predicted_return'])
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=calib['predicted'], y=calib['actual'], mode='markers+lines', name="Actual", marker=dict(color='#00d26a')))
                    # Add 45 degree line
                    min_val = min(calib['predicted'].min(), calib['actual'].min())
                    max_val = max(calib['predicted'].max(), calib['actual'].max())
                    fig_cal.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name="Ideal", line=dict(dash='dash', color='gray')))
                    fig_cal.update_layout(title="Calibration: Predicted vs Actual", height=350, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
                    st.plotly_chart(fig_cal, width='stretch', key=f"calib_{asset}")
                    
                    # IC by Regime
                    stress_scores = get_historical_stress(X_live.loc[actual.index])
                    regimes = stress_scores.apply(lambda s: "ALERT" if s > alert_threshold else "WARNING" if s > alert_threshold * 0.6 else "CALM")
                    ic_regime = compute_ic_by_regime(actual, oos['predicted_return'], regimes)
                    if not ic_regime.empty:
                        fig_reg = go.Figure(go.Bar(x=ic_regime['regime'], y=ic_regime['ic'], marker_color='#4da6ff'))
                        fig_reg.update_layout(title="IC by Market Regime", height=350, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
                        st.plotly_chart(fig_reg, width='stretch', key=f"ic_regime_{asset}")
                    else:
                        st.info("Insufficient data for regime-specific IC analysis.")

def render_diagnostics_tab(stress_indicators, stability_results_map, prediction_selection, coverage_stats, descriptions):
    st.markdown("**Regime Indicators**")
    st.dataframe(pd.DataFrame([{'Indicator': k, 'Value': v} for k,v in stress_indicators.items()]), hide_index=True)
    if prediction_selection:
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            with st.expander(f"Diagnostics for {asset}"):
                sel_df = prediction_selection.get(asset, pd.DataFrame())
                if not sel_df.empty: 
                    st.plotly_chart(plot_feature_heatmap(sel_df, descriptions), width='stretch', key=f"heat_{asset}")
                st.plotly_chart(plot_stability_boxplot(stability_results_map, asset, descriptions), width='stretch', key=f"box_{asset}")
                st.plotly_chart(plot_variable_survival(stability_results_map, asset, descriptions), width='stretch', key=f"surv_{asset}")

def render_strategy_lab(asset_prices, prediction_results, y_backtest, X_backtest, horizon_months, min_persistence, l1_ratio, confidence_level, macro_data_current, risk_free_rate):
    with st.form("strategy_config"):
        strat = st.radio("Strategy", ["Max Return", "Min Volatility", "Min Drawdown", "Min Loss"], horizontal=True)
        submitted = st.form_submit_button("🚀 RUN SIMULATION")
        if submitted:
            if st.session_state.engine_results.get('backtest_results') is None:
                res, sel, cov = get_historical_backtest(y_backtest, X_backtest, 240, horizon_months, 12, min_persistence, l1_ratio, confidence_level, prices=asset_prices, macro_data=macro_data_current)
                st.session_state.engine_results.update({'backtest_results': res, 'backtest_selection': sel, 'coverage_stats': cov})
                save_engine_state(st.session_state.engine_results); st.rerun()
            results = st.session_state.engine_results['backtest_results']
            preds = pd.DataFrame({k: v['predicted_return'] for k, v in results.items()}).dropna()
            lower = pd.DataFrame({k: v['lower_ci'] for k, v in results.items()}).dropna()
            bt = StrategyBacktester(asset_prices, preds, lower, get_historical_stress(macro_data_current))
            st.session_state.lab_results = bt.run_strategy(strat)
            st.rerun()
    if 'lab_results' in st.session_state:
        st.line_chart(st.session_state.lab_results['equity_curve'])
