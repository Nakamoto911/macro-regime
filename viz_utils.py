import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from app_utils import create_theme, NBER_RECESSIONS

def plot_fred_series(data: pd.Series, title: str, subtitle: str, is_transformed: bool = False) -> go.Figure:
    """Create a detailed plot for a FRED-MD series with stats and recession bands."""
    theme = create_theme()
    color = '#ff4757' if is_transformed else '#4da6ff'
    
    # Calculate statistics
    stats = {
        'Mean': data.mean(),
        'Std': data.std(),
        'Q1': data.quantile(0.25),
        'Med': data.median(),
        'Q3': data.quantile(0.75)
    }
    stats_str = ", ".join([f"{k}: {v:.2e}" if abs(v) < 0.01 and v != 0 else f"{k}: {v:.2f}" for k, v in stats.items()])
    
    fig = go.Figure()
    
    # Add Recession Bands
    for start, end in NBER_RECESSIONS:
        if pd.to_datetime(start) <= data.index[-1] and pd.to_datetime(end) >= data.index[0]:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=theme['recession_color'], opacity=0.07,
                layer="below", line_width=0
            )
            
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        line=dict(color=color, width=1.3),
        hovertemplate='%{x|%b %Y}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 10px; color: {theme['text_secondary']};'>{subtitle}</span><br><span style='font-size: 9px; color: {theme['text_muted']};'>{stats_str}</span>",
            font=dict(family='IBM Plex Mono', size=13, color=theme['font']['color']),
            x=0.05, y=0.92
        ),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=75, b=40),
        height=280,
        xaxis=dict(**theme['xaxis']),
        yaxis=dict(**theme['yaxis'])
    )
    
    return fig

def plot_allocation(weights: dict) -> go.Figure:
    theme = create_theme()
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.65,
        marker=dict(colors=[colors[k] for k in weights.keys()], line=dict(color=theme['paper_bgcolor'], width=2)),
        textinfo='label+percent',
        textfont=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])
    )])
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        annotations=[dict(text='<b>TARGET</b>', x=0.5, y=0.5,
                         font=dict(family='IBM Plex Mono', size=12, color=theme['label_color']), showarrow=False)]
    )
    return fig

def plot_assets(prices: pd.DataFrame) -> go.Figure:
    theme = create_theme()
    normalized = prices / prices.iloc[0] * 100
    
    fig = go.Figure()
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col], name=col,
            mode='lines', line=dict(color=colors.get(col, '#888'), width=1.5)
        ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=280,
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Indexed (100)', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_ect(ect: pd.Series) -> go.Figure:
    theme = create_theme()
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'])
    
    std = ect.std()
    fig.add_hrect(y0=-std, y1=std, fillcolor="rgba(77, 166, 255, 0.05)", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=ect.index, y=ect.values, mode='lines',
        line=dict(color='#4da6ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(77, 166, 255, 0.1)'
    ))
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=10, b=40),
        height=150,
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='ECT', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_feature_heatmap(selection_history: pd.DataFrame, descriptions: dict) -> go.Figure:
    """Visualize stability selection probabilities over time."""
    if selection_history is None or (isinstance(selection_history, pd.DataFrame) and selection_history.empty) or (isinstance(selection_history, list) and not selection_history):
        return go.Figure()
        
    theme = create_theme()
    
    if isinstance(selection_history, list):
        rows = []
        for entry in selection_history:
            row = {'date': entry['date']}
            for feat in entry.get('selected', []):
                row[feat] = 1
            rows.append(row)
        df_plot = pd.DataFrame(rows).fillna(0).set_index('date').T
    else:
        df_plot = selection_history.T
    
    avg_probs = df_plot.mean(axis=1).sort_values(ascending=True)
    df_plot = df_plot.loc[avg_probs.index]
    y_labels = [f"{col} ({descriptions.get(col.split('_')[0], 'Macro Variable')})" for col in df_plot.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=df_plot.values,
        x=df_plot.columns,
        y=y_labels,
        colorscale=[[0, theme['plot_bgcolor']], [0.5, 'rgba(77, 166, 255, 0.2)'], [1, '#4da6ff']],
        showscale=True,
        colorbar=dict(title=dict(text='Selection Prob', side='right', font=dict(color=theme['label_color'])), thickness=15, len=0.7, tickfont=dict(color=theme['label_color']))
    ))
    
    fig.update_layout(
        title=dict(text='STABILITY SELECTION: FEATURE PERSISTENCE OVER TIME', font=dict(size=12, color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        xaxis=dict(title=dict(text='Estimation Date', font=dict(color=theme['label_color'])), gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(size=9, color=theme['label_color'])),
        margin=dict(l=10, r=10, t=40, b=40),
        height=500,
        font=theme['font']
    )
    return fig

def plot_stability_boxplot(results: dict, asset: str, descriptions: dict = None) -> go.Figure:
    """Show coefficient distribution across windows."""
    theme = create_theme()
    if asset not in results or 'all_coefficients' not in results[asset]:
        return go.Figure()
    
    coef_df = results[asset]['all_coefficients']
    stable_feats = results[asset].get('stable_features', [])[:8]
    if not stable_feats:
        return go.Figure()
    
    fig = go.Figure()
    colors = ['#ff6b35', '#4da6ff', '#ffd700', '#00d26a', '#ff4757', '#9b59b6', '#1abc9c', '#e74c3c']
    for i, feat in enumerate(stable_feats):
        if feat in coef_df.columns:
            base_var = feat.split('_')[0]
            desc = descriptions.get(base_var, base_var) if descriptions else base_var
            fig.add_trace(go.Box(
                y=coef_df[feat], 
                name=feat[:12],
                marker_color=colors[i % len(colors)],
                boxmean=True,
                hovertext=desc,
                hoverinfo='y+text+name'
            ))
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        showlegend=False,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Coefficient', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        font=theme['font']
    )
    return fig

def plot_trend_bars(trend_df: pd.DataFrame, variables: list, descriptions: dict = None) -> go.Figure:
    """Plot 5Y trend comparison."""
    theme = create_theme()
    filtered = trend_df[trend_df['Variable'].isin(variables)]
    if filtered.empty:
        return go.Figure()
    
    colors = ['#00d26a' if x > 0.05 else '#ff4757' if x < -0.05 else '#888888' for x in filtered['Slope_5Y']]
    hovers = [descriptions.get(var, var) if descriptions else var for var in filtered['Variable']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered['Variable'],
        y=filtered['Slope_5Y'],
        marker_color=colors,
        text=[f"{x:.2f}" for x in filtered['Slope_5Y']],
        textposition='outside',
        hovertext=hovers,
        hoverinfo='y+text+x'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'])
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'], margin=dict(l=40, r=20, t=20, b=60), height=250,
        xaxis=dict(tickangle=-45, tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='5Y Slope (Annualized)', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
    )
    return fig

def plot_driver_vs_asset(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, feat_name: str, asset: str, descriptions: dict = None) -> go.Figure:
    theme = create_theme()
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
    macro_vals, asset_vals = combined[feat_name], combined[asset]
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined.index, y=macro_vals, name=feat_name, mode='lines', line=dict(color='#00d26a', width=1.5), hovertemplate="<b>" + feat_name + "</b>: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=combined.index, y=asset_vals, name=asset, mode='lines', line=dict(color='#4da6ff', width=1.5), yaxis='y2', hovertemplate="<b>" + asset + " Return</b>: %{y:.2%}<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"{feat_name} ({desc}) vs {asset} Forward Return", font=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'], margin=dict(l=50, r=50, t=40, b=40), height=350,
        showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])), hovermode='x unified',
        xaxis=dict(gridcolor=theme['gridcolor'], showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=1, spikecolor=theme['text_muted'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], side='left', title=dict(text=f'Macro: {feat_name}', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis2=dict(gridcolor=theme['gridcolor'], overlaying='y', side='right', title=dict(text='Forward Return (Annualized)', font=dict(color=theme['label_color'])), tickformat='.1%', tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_driver_scatter(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, feat_name: str, asset: str, descriptions: dict = None) -> go.Figure:
    theme = create_theme()
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
    combined['Decade'] = (combined.index.year // 10 * 10).astype(str) + "s"
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    fig = px.scatter(combined, x=feat_name, y=asset, color='Decade', trendline="ols",
                     title=f"Correlation Density: {feat_name} ({desc}) vs {asset}",
                     labels={feat_name: f"{feat_name}", asset: f"{asset} Fwd Return"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        title=dict(font=dict(color=theme['font']['color'])), paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'], margin=dict(l=50, r=20, t=40, b=40), height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])),
        xaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=feat_name, font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f"{asset} Return", font=dict(color=theme['label_color'])), tickformat='.1%', tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_rolling_correlation(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, feat_name: str, asset: str, window: int = 60) -> go.Figure:
    theme = create_theme()
    if feat_name not in feat_data.columns or asset not in asset_returns.columns:
        return go.Figure()
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if len(combined) < window: return go.Figure()
    rolling_corr = combined[feat_name].rolling(window).corr(combined[asset])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', line=dict(color='#ff6b35', width=1.5), fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)', name='Rolling Correlation'))
    fig.update_layout(
        title=dict(text=f"Rolling {window}M Correlation: {feat_name} vs {asset}", font=dict(color=theme['font']['color'])),
        paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'], font=theme['font'], margin=dict(l=50, r=20, t=40, b=40), height=300,
        xaxis=dict(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Correlation', font=dict(color=theme['label_color'])), range=[-1, 1], tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_quintile_analysis(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, feat_name: str, asset: str, horizon_months: int = 12) -> go.Figure:
    theme = create_theme()
    if feat_name not in feat_data.columns or asset not in asset_returns.columns: return go.Figure()
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty: return go.Figure()
    combined['Quintile'] = pd.qcut(combined[feat_name], 5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
    quintile_avg = combined.groupby('Quintile')[asset].mean().reset_index()
    colors = ['#ff4757', '#ffa502', '#ced6e0', '#2ed573', '#1e90ff']
    fig = px.bar(quintile_avg, x='Quintile', y=asset, title=f"Quintile Analysis: {asset} {horizon_months}M Return by {feat_name} Bucket", color='Quintile', color_discrete_sequence=colors)
    fig.update_layout(
        title=dict(font=dict(color=theme['font']['color'])), paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'], margin=dict(l=50, r=20, t=40, b=40), height=350, showlegend=False,
        xaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f"{feat_name} Quintiles", font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])),
        yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text=f'Avg {horizon_months}M Forward Return', font=dict(color=theme['label_color'])), tickformat='.1%', tickfont=dict(color=theme['label_color']))
    )
    return fig

def plot_combined_driver_analysis(feat_data: pd.DataFrame, asset_returns: pd.DataFrame, feat_name: str, asset: str, descriptions: dict = None, window: int = 60, horizon_months: int = 12) -> go.Figure:
    theme = create_theme()
    if feat_name not in feat_data.columns or asset not in asset_returns.columns: return go.Figure()
    combined = pd.concat([feat_data[feat_name], asset_returns[asset]], axis=1).dropna()
    if combined.empty: return go.Figure()
    macro_vals, asset_vals = combined[feat_name], combined[asset]
    rolling_corr = macro_vals.rolling(window).corr(asset_vals)
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=combined.index, y=macro_vals, name=feat_name, mode='lines', line=dict(color='#00d26a', width=1.5), hovertemplate="<b>" + feat_name + "</b>: %{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=combined.index, y=asset_vals, name=asset, mode='lines', line=dict(color='#4da6ff', width=1.5), hovertemplate="<b>" + asset + f" {horizon_months}M Return</b>: %{{y:.2%}}<extra></extra>"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', line=dict(color='#ff6b35', width=1.5), fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)', name=f'{window}M Rolling Correlation', hovertemplate="<b>Correlation</b>: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=theme['label_color'], row=2, col=1)
    fig.update_layout(title=dict(text=f"{feat_name} ({desc}) Analysis", font=dict(family='IBM Plex Mono', size=12, color=theme['label_color'])), paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'], margin=dict(l=50, r=50, t=60, b=40), height=550, showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(color=theme['label_color'])), hovermode='x unified')
    fig.update_xaxes(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), row=1, col=1)
    fig.update_xaxes(gridcolor=theme['gridcolor'], tickfont=dict(color=theme['label_color']), row=2, col=1)
    fig.update_yaxes(title_text=f"Macro: {feat_name}", gridcolor=theme['gridcolor'], row=1, col=1, secondary_y=False, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    fig.update_yaxes(title_text=f"{horizon_months}M Fwd Return", gridcolor=theme['gridcolor'], tickformat='.0%', row=1, col=1, secondary_y=True, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    fig.update_yaxes(title_text=f"{window}M Correlation", gridcolor=theme['gridcolor'], range=[-1, 1], row=2, col=1, tickfont=dict(color=theme['label_color']), title_font=dict(color=theme['label_color']))
    return fig

def plot_variable_survival(stability_results_map: dict, asset: str, descriptions: dict = None) -> go.Figure:
    theme = create_theme()
    if asset not in stability_results_map: return go.Figure().update_layout(title="No stability data available", **theme)
    if isinstance(stability_results_map.get(asset), pd.DataFrame): coef_df = stability_results_map[asset]
    else: coef_df = stability_results_map[asset].get('all_coefficients', pd.DataFrame())
    if coef_df.empty: return go.Figure().update_layout(title="No selection history available", **theme)
    feature_cols = [c for c in coef_df.columns if c != 'const']
    counts = (coef_df[feature_cols].fillna(0) != 0).sum().sort_values(ascending=True)
    counts = counts[counts > 0]
    if counts.empty: return go.Figure().update_layout(title="No drivers survived the stability test", **theme)
    labels = [f"<b>{feat}</b><br><span style='font-size:9px; color:{theme['text_muted']};'>{descriptions.get(feat.split('_')[0], feat) if descriptions else feat}</span>" for feat in counts.index]
    fig = go.Figure(go.Bar(x=counts.values, y=labels, orientation='h', marker=dict(color=counts.values, colorscale='Oranges', line=dict(color=theme['paper_bgcolor'], width=1)), text=counts.values, textposition='auto'))
    layout_args = theme.copy()
    layout_args.update({'title': dict(text=f"VARIABLE SURVIVAL LEADERBOARD - {asset}", font=dict(size=14, color='#ff6b35')), 'xaxis_title': "Number of Windows Selected", 'margin': dict(l=20, r=20, t=60, b=40), 'height': 400 + (len(counts) * 15)})
    if 'yaxis' in layout_args: layout_args['yaxis'] = {**layout_args['yaxis'], 'showgrid': False}
    else: layout_args['yaxis'] = dict(showgrid=False)
    fig.update_layout(**layout_args)
    return fig

def plot_backtest(actual_returns: pd.Series, predicted_returns: pd.Series, confidence_lower: pd.Series, confidence_upper: pd.Series, confidence_level: float = 0.90) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predicted_returns.index, y=confidence_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=predicted_returns.index, y=confidence_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(77, 166, 255, 0.2)', name=f'{int(confidence_level*100)}% CI', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=predicted_returns.index, y=predicted_returns, mode='lines', line=dict(color='#4da6ff', width=2), name='Predicted', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=actual_returns.index, y=actual_returns, mode='lines', line=dict(color='#ff6b35', width=2), name='Actual', hoverinfo='skip'))
    hover_y = 0.05
    fig.add_trace(go.Scatter(x=predicted_returns.index, y=[hover_y] * len(predicted_returns), yaxis='y2', name='Pred', mode='markers', marker=dict(size=0, opacity=0), showlegend=False, hovertemplate="<b>Pred</b>: %{customdata:.1%}<extra></extra>", customdata=predicted_returns))
    fig.add_trace(go.Scatter(x=actual_returns.index, y=[hover_y] * len(actual_returns), yaxis='y2', name='Act', mode='markers', marker=dict(size=0, opacity=0), showlegend=False, hovertemplate="<b>Act</b>: %{customdata:.1%}<extra></extra>", customdata=actual_returns))
    theme = create_theme()
    fig.update_layout(paper_bgcolor=theme['paper_bgcolor'], plot_bgcolor=theme['plot_bgcolor'], margin=dict(l=50, r=20, t=30, b=40), height=350, hovermode='x', hoverlabel=dict(bgcolor='rgba(0,0,0,0.6)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)', font=dict(family='IBM Plex Mono', size=11, color=theme['font']['color'])), xaxis=dict(gridcolor=theme['gridcolor'], showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=1, spikecolor=theme['text_muted'], tickfont=dict(color=theme['label_color'])), yaxis=dict(gridcolor=theme['gridcolor'], title=dict(text='Annualized Return', font=dict(color=theme['label_color'])), tickfont=dict(color=theme['label_color'])), yaxis2=dict(range=[0, 1], overlaying='y', visible=False, fixedrange=True), legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    return fig
