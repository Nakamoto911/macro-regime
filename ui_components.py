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
    plot_variable_survival, plot_backtest
)
from models import get_historical_backtest, get_historical_stress
from data_utils import apply_transformation
from backtester import StrategyBacktester

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

def construct_model_summary(asset: str, model_stats: dict) -> str:
    if asset not in model_stats: return "Model details not available."
    m_info = model_stats[asset]; model = m_info.get('model')
    if asset == 'EQUITY' or "XGB" in str(type(model)):
        importance = m_info.get('importance', pd.Series())
        if importance.empty: return "Non-linear Ensemble (XGBoost)."
        summary = "**Architecture: XGBoost**\n\nTop Drivers:\n"
        for feat, imp in importance.sort_values(ascending=False).head(5).items(): summary += f"- {feat}: `{imp:.4f}`\n"
        return summary
    else:
        coefs = m_info.get('coefficients', pd.Series())
        if coefs.empty: return "Linear Model."
        arch = "ElasticNet" if asset == 'BONDS' else "OLS"
        res = f"**Architecture: {arch}**\n\nEq: `{m_info.get('intercept', 0):.4f}`"
        for feat, val in coefs[coefs.abs() > 1e-6].items():
            if feat == 'const': continue
            res += f" {'+' if val >= 0 else '-'} (`{abs(val):.4f}` * {feat})"
        return res

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
    with c1: st.plotly_chart(plot_allocation(target_weights), width='stretch')
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
                sel = st.dataframe(df_disp[['feature', 'Signal', 'Impact', 'Weight', 'State', 'Link']], hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key=f"sel_{asset}")
                if sel.get('selection', {}).get('rows'):
                    feat = df_disp.iloc[sel['selection']['rows'][0]]['feature']
                    st.plotly_chart(plot_combined_driver_analysis(X_live, y_forward, feat, asset, descriptions, horizon_months=horizon_months), width='stretch')

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
            st.plotly_chart(plot_fred_series(df_full[col], col, descriptions.get(col, ""), is_transformed=(mode!="Raw")), width='stretch')

def render_prediction_tab(prediction_results, y_live, horizon_months, min_persistence, l1_ratio, confidence_level, X_live):
    if prediction_results is None:
        if st.button("📊 Run Prediction Model Validation", type="primary"):
            res, sel, cov = get_historical_backtest(y_live, X_live, 240, horizon_months, 12, min_persistence, l1_ratio, confidence_level)
            st.session_state.engine_results.update({'prediction_results': res, 'prediction_selection': sel, 'coverage_stats': cov})
            save_engine_state(st.session_state.engine_results); st.rerun()
    else:
        asset = st.selectbox("Asset", ['EQUITY', 'BONDS', 'GOLD'])
        oos = prediction_results.get(asset, pd.DataFrame())
        if not oos.empty:
            st.plotly_chart(plot_backtest(y_live[asset].loc[oos.index], oos['predicted_return'], oos['lower_ci'], oos['upper_ci'], confidence_level), width='stretch')

def render_diagnostics_tab(stress_indicators, stability_results_map, prediction_selection, coverage_stats, descriptions):
    st.markdown("**Regime Indicators**")
    st.dataframe(pd.DataFrame([{'Indicator': k, 'Value': v} for k,v in stress_indicators.items()]), hide_index=True)
    if prediction_selection:
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            with st.expander(f"Diagnostics for {asset}"):
                sel_df = prediction_selection.get(asset, pd.DataFrame())
                if not sel_df.empty: st.plotly_chart(plot_feature_heatmap(sel_df, descriptions), width='stretch')
                st.plotly_chart(plot_stability_boxplot(stability_results_map, asset, descriptions), width='stretch')
                st.plotly_chart(plot_variable_survival(stability_results_map, asset, descriptions), width='stretch')

def render_strategy_lab(asset_prices, prediction_results, y_backtest, X_backtest, horizon_months, min_persistence, l1_ratio, confidence_level, macro_data_current, risk_free_rate):
    with st.form("strategy_config"):
        strat = st.radio("Strategy", ["Max Return", "Min Volatility", "Min Drawdown", "Min Loss"], horizontal=True)
        submitted = st.form_submit_button("🚀 RUN SIMULATION")
        if submitted:
            if st.session_state.engine_results.get('backtest_results') is None:
                res, sel, cov = get_historical_backtest(y_backtest, X_backtest, 240, horizon_months, 12, min_persistence, l1_ratio, confidence_level)
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
