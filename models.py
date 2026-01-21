import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import t as t_dist
from xgboost import XGBRegressor
import time
from joblib import Parallel, delayed

from benchmarking_engine import (
    Winsorizer,
    FactorStripper,
    PointInTimeFactorStripper,
    select_features_elastic_net,
    run_benchmarking_engine
)
from data_utils import (
    prepare_macro_features, 
    compute_forward_returns,
    MacroFeatureExpander
)
from inference import HodrickInference, NonOverlappingEstimator
from feature_selection import AdaptiveFeatureSelector
from prediction_intervals import BootstrapPredictionInterval, CoverageValidator

def estimate_with_hac(y: pd.Series, X: pd.DataFrame, lag: int = 11) -> dict:
    """OLS estimation with Newey-West HAC standard errors."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_stats': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'resid': model.resid
    }

def estimate_with_corrected_inference(y, X, config):
    """Dispatch to appropriate inference method."""
    method = config.get('inference_method', 'Hodrick (1992)')
    horizon = config.get('horizon', 12)
    
    if method == 'Hodrick (1992)':
        estimator = HodrickInference(horizon=horizon)
        X_const = sm.add_constant(X)
        estimator.fit(y, X_const)
        return {
            'coefficients': pd.Series(estimator.coefficients_, index=X_const.columns),
            'std_errors': pd.Series(estimator.hodrick_se_, index=X_const.columns),
            't_stats': pd.Series(estimator.t_stats_, index=X_const.columns),
            'p_values': pd.Series(estimator.p_values_, index=X_const.columns),
            'r_squared': estimator.r_squared_
        }
    elif method == 'Non-Overlapping':
        estimator = NonOverlappingEstimator(horizon=horizon)
        estimator.fit(y, X)
        return {
            'coefficients': pd.Series(estimator.coefficients_, index=X.columns),
            'intercept': estimator.intercept_,
            'std_errors': pd.Series(estimator.coef_std_, index=X.columns),
            'r_squared': None
        }
    else:
        return estimate_with_hac(y, X, lag=horizon + 4)

def estimate_robust(y: pd.Series, X: pd.DataFrame) -> dict:
    """Robust regression (Huber) for stability."""
    X_clipped = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    model = HuberRegressor(epsilon=1.35, max_iter=200)
    model.fit(X_clipped, y)
    return {
        'model': model,
        'coefficients': pd.Series(model.coef_, index=X.columns),
        'intercept': model.intercept_,
        'fitted_values': model.predict(X_clipped)
    }

def _perform_feature_selection_step(current_date, y, X_expanded_global, horizon_months, asset_class):
    """Helper for parallel feature selection."""
    train_limit = current_date - pd.DateOffset(months=horizon_months)
    mask_train = X_expanded_global.index <= train_limit
    
    X_train_prep = X_expanded_global.loc[mask_train]
    y_train_prep = y.loc[mask_train].dropna()
    
    common_train = X_train_prep.index.intersection(y_train_prep.index)
    X_train_prep = X_train_prep.loc[common_train]
    y_train_prep = y_train_prep.loc[common_train]
    
    if len(y_train_prep) < 60:
        return current_date, [], 0.5, 0.5
        
    selector = AdaptiveFeatureSelector(asset_class=asset_class)
    # Reduce bootstrap slightly for speed in parallel mode
    selector.fit(y_train_prep, X_train_prep, n_bootstrap=10) 
    cached_features = selector.get_selected_features()
    if not cached_features: 
        cached_features = X_train_prep.columns[:10].tolist()
        
    return current_date, cached_features, selector.selector.best_alpha_, selector.selector.best_l1_ratio_

def _perform_prediction_step(current_date, predict_idx, y, X_expanded_global, horizon_months, 
                           stable_feats, best_alpha, best_l1, asset_class, confidence_level):
    """Helper for parallel prediction."""
    train_limit = current_date - pd.DateOffset(months=horizon_months)
    mask_train = X_expanded_global.index <= train_limit
    
    X_train_prep = X_expanded_global.loc[mask_train]
    y_train_prep = y.loc[mask_train].dropna()
    
    common_train = X_train_prep.index.intersection(y_train_prep.index)
    X_train_prep = X_train_prep.loc[common_train]
    y_train_prep = y_train_prep.loc[common_train]
    
    X_test_expanded = X_expanded_global.loc[X_expanded_global.index.intersection(predict_idx)]
    
    X_train_sel = X_train_prep[stable_feats]
    X_test_sel = X_test_expanded[stable_feats]
    
    win = Winsorizer(threshold=3.0)
    X_train_final = win.fit_transform(X_train_sel).fillna(0)
    X_test_final = win.transform(X_test_sel).fillna(0)
    
    if X_test_final.empty or X_train_final.empty:
        raw_preds = np.zeros(len(predict_idx))
        lower_ci, upper_ci = raw_preds, raw_preds
    else:
        if asset_class in ['BONDS', 'GOLD']:
            if asset_class == 'BONDS':
                model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=1000)
            else:
                model = LinearRegression()
            
            model.fit(X_train_final, y_train_prep)
            raw_preds = model.predict(X_test_final)
            residuals = y_train_prep - model.predict(X_train_final)
            rse = np.std(residuals)
            dof = len(y_train_prep) - X_train_final.shape[1] - 1
            t_crit = t_dist.ppf((1 + confidence_level) / 2, df=max(1, dof))
            margin = t_crit * rse
            lower_ci = raw_preds - margin
            upper_ci = raw_preds + margin
        else:
            model = XGBRegressor(n_estimators=25, max_depth=3, learning_rate=0.08, random_state=42, n_jobs=1)
            model.fit(X_train_final, y_train_prep)
            raw_preds = model.predict(X_test_final)
            bt_interval = BootstrapPredictionInterval(confidence_level=confidence_level, n_bootstrap=20)
            bt_interval.fit(model, X_train_final, y_train_prep)
            _, lower_ci, upper_ci = bt_interval.predict_interval(X_test_final)

        raw_preds = np.clip(raw_preds, -0.50, 0.50)
        
    return current_date, X_test_final.index, raw_preds, lower_ci, upper_ci

def run_walk_forward_backtest(y: pd.Series, X: pd.DataFrame, 
                              min_train_months: int = 240, 
                              horizon_months: int = 12,
                              rebalance_freq: int = 12,
                              asset_class: str = 'EQUITY',
                              selection_threshold: float = 0.6,
                              l1_ratio: float = 0.5,
                              confidence_level: float = 0.90,
                              progress_cb=None,
                              X_precomputed: pd.DataFrame = None) -> tuple:
    """Researcher-Grade Optimized Backtester with Parallel Execution."""
    results = []
    selection_history = []
    coverage_validator = CoverageValidator(nominal_level=confidence_level)
    
    X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
    y = y.apply(pd.to_numeric, errors='coerce').astype('float32')
    
    common_global = X.index.intersection(y.index)
    X = X.loc[common_global]
    y = y.loc[common_global]
    dates = X.index
    
    start_idx = min_train_months + horizon_months
    if start_idx >= len(dates):
        return pd.DataFrame(columns=['predicted_return', 'lower_ci', 'upper_ci']), pd.DataFrame(), {}

    if X_precomputed is not None:
        X_expanded_global = X_precomputed
    else:
        pit_stripper = PointInTimeFactorStripper(drivers=['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS'], min_history=60)
        X_ortho = pit_stripper.fit_transform_pit(X)
        expander = MacroFeatureExpander()
        X_expanded_global = expander.transform(X_ortho).astype('float32').fillna(0)
    
    common_calc = X_expanded_global.index.intersection(y.index)
    X_expanded_global = X_expanded_global.loc[common_calc]
    y = y.loc[common_calc]
    dates = X_expanded_global.index

    # 1. Plan Execution Steps
    steps = list(range(start_idx, len(dates), rebalance_freq))
    selection_dates = []
    step_map = [] # (current_date, selection_date_to_use)
    
    last_sel_date = None
    
    # Determine which steps trigger new feature selection
    for i in steps:
        current_date = dates[i]
        # Logic: Update features if it's the first step, or if it's January (and freq < 12), or if freq >= 12
        needs_update = (last_sel_date is None) or \
                       (current_date.month == 1 and rebalance_freq < 12) or \
                       (rebalance_freq >= 12)
        
        if needs_update:
            last_sel_date = current_date
            selection_dates.append(current_date)
            
        predict_idx = dates[i : min(i + rebalance_freq, len(dates))]
        step_map.append({
            'current_date': current_date,
            'predict_idx': predict_idx,
            'sel_date': last_sel_date
        })

    # 2. Parallel Feature Selection
    # Run heavy selection tasks in parallel
    if progress_cb: progress_cb(0.1, dates[steps[0]])
    
    sel_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_perform_feature_selection_step)(d, y, X_expanded_global, horizon_months, asset_class)
        for d in selection_dates
    )
    
    # Map results for O(1) lookup
    feature_cache = {
        res[0]: {'features': res[1], 'alpha': res[2], 'l1': res[3]} 
        for res in sel_results
    }
    
    # Store history
    for d in selection_dates:
        selection_history.append({'selected': feature_cache[d]['features'], 'date': d})

    # 3. Parallel Prediction
    if progress_cb: progress_cb(0.5, dates[steps[len(steps)//2]])
    
    pred_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_perform_prediction_step)(
            step['current_date'], 
            step['predict_idx'],
            y, X_expanded_global, horizon_months,
            feature_cache[step['sel_date']]['features'],
            feature_cache[step['sel_date']]['alpha'],
            feature_cache[step['sel_date']]['l1'],
            asset_class, confidence_level
        )
        for step in step_map
    )

    # 4. Aggregate Results
    if progress_cb: progress_cb(0.9, dates[-1])
    
    for _, idxs, raw_preds, l_ci, u_ci in pred_results:
        pred_vals = pd.Series(raw_preds, index=idxs)
        for i, date in enumerate(idxs):
            if date in y.index:
                actual_ret = y.loc[date]
                val = pred_vals.loc[date]
                l = l_ci[i] if isinstance(l_ci, (list, np.ndarray)) else l_ci
                u = u_ci[i] if isinstance(u_ci, (list, np.ndarray)) else u_ci
                
                coverage_validator.record(actual_ret, l, u)
                results.append({'date': date, 'predicted_return': val, 'lower_ci': l, 'upper_ci': u})
            
    if not results:
        return pd.DataFrame(columns=['predicted_return', 'lower_ci', 'upper_ci']), pd.DataFrame(), {}
        
    oos_df = pd.DataFrame(results).set_index('date').sort_index()
    selection_df = pd.DataFrame(selection_history).set_index('date').sort_index()
    
    return oos_df, selection_df, coverage_validator.compute_statistics()

@st.cache_data(show_spinner=False)
def cached_walk_forward(y, X, min_train_months=240, horizon_months=12, rebalance_freq=12, asset_class='EQUITY', confidence_level=0.90):
    return run_walk_forward_backtest(y, X, min_train_months, horizon_months, rebalance_freq, asset_class, confidence_level=confidence_level)

def _process_stability_window(start, end, y, X, horizon_months, inference_method):
    """Helper for parallel stability analysis."""
    y_window = y.iloc[start:end]
    X_window = X.iloc[start:end]
    y_valid = y_window.dropna()
    X_valid = X_window.loc[y_valid.index].dropna(axis=1)
    
    if len(y_valid) < 120 or X_valid.empty:
        return None
        
    selected_features, selection_probs = select_features_elastic_net(y_valid, X_valid)
    
    if selected_features:
        estimation_config = {'inference_method': inference_method, 'horizon': horizon_months}
        inf_res = estimate_with_corrected_inference(y_valid, X_valid[selected_features], estimation_config)
        coef_dict = inf_res['coefficients'].to_dict()
    else:
        coef_dict = {col: 0.0 for col in X_valid.columns}
        
    window_corrs = X_valid.corrwith(y_valid)
    
    return {
        'start_date': y.index[start], 
        'end_date': y.index[end - 1],
        'selected_features': selected_features, 
        'coefficients': coef_dict,
        'correlations': window_corrs.to_dict(), 
        'n_selected': len(selected_features)
    }

def stability_analysis(y: pd.Series, X: pd.DataFrame, horizon_months: int = 12, window_years: int = 25, step_years: int = 5) -> list:
    common = y.index.intersection(X.index)
    y = y.loc[common]
    X = X.loc[common]
    window_months = window_years * 12
    step_months = step_years * 12
    
    tasks = []
    for start in range(0, len(y) - window_months, step_months):
        end = start + window_months
        tasks.append((start, end))
        
    inference_method = st.session_state.get('inference_method', 'Hodrick (1992)')
    
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_process_stability_window)(start, end, y, X, horizon_months, inference_method)
        for start, end in tasks
    )
    
    return [r for r in results if r is not None]

def compute_stability_metrics(stability_results: list, feature_names: list) -> pd.DataFrame:
    n_windows = len(stability_results)
    if n_windows == 0:
        return pd.DataFrame(columns=['feature', 'persistence', 'sign_consistency', 'magnitude_stability', 'mean_coefficient', 'correlation'])
    metrics = []
    for feat in feature_names:
        coefs = []
        corrs = []
        for result in stability_results:
            if feat in result.get('coefficients', {}): coefs.append(result['coefficients'][feat])
            if feat in result.get('correlations', {}): corrs.append(result['correlations'][feat])
        coefs = np.array(coefs); non_zero = coefs[coefs != 0]
        persistence = len(non_zero) / n_windows
        avg_corr = np.mean(corrs) if corrs else 0.0
        if len(non_zero) > 1:
            sign_consistency = max((non_zero > 0).sum() / len(non_zero), (non_zero < 0).sum() / len(non_zero))
            cv = np.std(non_zero) / np.abs(np.mean(non_zero)) if np.mean(non_zero) != 0 else 999
            magnitude_stability = 1 / (1 + cv); mean_coef = np.mean(non_zero)
        elif len(non_zero) == 1:
            sign_consistency = 1.0; magnitude_stability = 1.0; mean_coef = non_zero[0]
        else:
            sign_consistency = 0.0; magnitude_stability = 0.0; mean_coef = 0.0
        metrics.append({
            'feature': feat, 'persistence': persistence, 'sign_consistency': sign_consistency,
            'magnitude_stability': magnitude_stability, 'mean_coefficient': mean_coef, 'correlation': avg_corr
        })
    return pd.DataFrame(metrics)

def compute_expected_returns(macro_features_current: pd.Series, stable_coefficients: pd.Series, intercept: float) -> float:
    common_features = stable_coefficients.index.intersection(macro_features_current.index)
    return intercept + (macro_features_current[common_features] * stable_coefficients[common_features]).sum()

def compute_driver_attribution(macro_features_current: pd.Series, stable_coefficients: pd.Series, feature_means: pd.Series) -> pd.DataFrame:
    attributions = []
    for feat in stable_coefficients.index:
        if feat not in macro_features_current.index: continue
        current_val = macro_features_current[feat]; mean_val = feature_means.get(feat, 0); coef = stable_coefficients[feat]
        contribution = coef * (current_val - mean_val)
        direction = 'TAILWIND' if contribution > 0.005 else 'HEADWIND' if contribution < -0.005 else 'NEUTRAL'
        attributions.append({
            'feature': feat, 'coefficient': coef, 'current_value': current_val,
            'historical_mean': mean_val, 'deviation': current_val - mean_val,
            'contribution': contribution, 'direction': direction
        })
    return pd.DataFrame(attributions).sort_values('contribution', key=abs, ascending=False)

def evaluate_regime(macro_data: pd.DataFrame, alert_threshold: float = 2.0) -> tuple:
    recent = macro_data.tail(60); indicators = {}
    if 'BAA_AAA' in recent.columns:
        indicators['credit'] = (recent['BAA_AAA'].iloc[-1] - recent['BAA_AAA'].mean()) / recent['BAA_AAA'].std()
    else: indicators['credit'] = 0
    if 'SPREAD' in recent.columns:
        indicators['curve'] = -(recent['SPREAD'].iloc[-1] - recent['SPREAD'].mean()) / recent['SPREAD'].std()
    else: indicators['curve'] = 0
    stress_score = 0.5 * indicators['credit'] + 0.5 * indicators['curve']
    status = "ALERT" if stress_score > alert_threshold else "WARNING" if stress_score > alert_threshold * 0.6 else "CALM"
    return status, stress_score, indicators

def get_historical_stress(macro_data):
    history = pd.DataFrame(index=macro_data.index)
    for col, sign in [('BAA_AAA', 1), ('SPREAD', -1)]:
        if col in macro_data.columns:
            rolled_mean = macro_data[col].rolling(window=60).mean()
            rolled_std = macro_data[col].rolling(window=60).std()
            history[col] = sign * (macro_data[col] - rolled_mean) / rolled_std
        else: history[col] = 0
    return 0.5 * history['BAA_AAA'].fillna(0) + 0.5 * history['SPREAD'].fillna(0)

def compute_allocation(expected_returns: dict, regime_status: str, risk_free_rate: float = 0.04) -> dict:
    base = {'EQUITY': 0.60, 'BONDS': 0.30, 'GOLD': 0.10}
    min_w = {'EQUITY': 0.20, 'BONDS': 0.20, 'GOLD': 0.05}
    max_w = {'EQUITY': 0.80, 'BONDS': 0.50, 'GOLD': 0.25}
    regime_mult = {'ALERT': {'EQUITY': 0.5, 'BONDS': 1.2, 'GOLD': 1.5}, 'WARNING': {'EQUITY': 0.75, 'BONDS': 1.1, 'GOLD': 1.25}, 'CALM': {'EQUITY': 1.0, 'BONDS': 1.0, 'GOLD': 1.0}}.get(regime_status, {'EQUITY': 1.0, 'BONDS': 1.0, 'GOLD': 1.0})
    weights = {}
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        weights[asset] = np.clip(base[asset] * regime_mult[asset] * (1.0 + (expected_returns.get(asset, 0.05) - risk_free_rate) * 5), min_w[asset], max_w[asset])
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

MANUAL_MACRO_CACHE = {}
def get_precomputed_macro_data(X: pd.DataFrame, drivers: list, min_history: int = 60, progress_cb=None):
    cache_key = hash(tuple(X.index))
    if cache_key in MANUAL_MACRO_CACHE: return MANUAL_MACRO_CACHE[cache_key]
    pit_stripper = PointInTimeFactorStripper(drivers=drivers, min_history=min_history, update_frequency=12)
    X_ortho = pit_stripper.fit_transform_pit(X, progress_cb=progress_cb)
    X_expanded = MacroFeatureExpander().transform(X_ortho).astype('float32').fillna(0)
    MANUAL_MACRO_CACHE[cache_key] = X_expanded
    return X_expanded

def get_historical_backtest(y, X, min_train_months, horizon_months, rebalance_freq, selection_threshold, l1_ratio, confidence_level=0.90):
    results, heatmaps, coverage = {}, {}, {}
    start_time = time.time()
    with st.status("🚀 Engine Processing...", expanded=True) as status:
        status_msg, progress_bar = st.empty(), st.progress(0)
        def update_pit_progress(pct, current_date):
            progress_bar.progress(min(0.15, 0.05 + pct * 0.10))
            elapsed = time.time() - start_time
            status_msg.markdown(f"**Step 1/4**: PIT Ortho | {current_date.strftime('%Y')} | {int(elapsed)}s")
        X_precomputed = get_precomputed_macro_data(X, ['CPIAUCSL', 'INDPRO', 'M2SL', 'FEDFUNDS'], min_history=60, progress_cb=update_pit_progress)
        for a_idx, asset in enumerate(['EQUITY', 'BONDS', 'GOLD']):
            def update_progress(pct_inner, current_date):
                progress_bar.progress(min(0.99, 0.15 + (a_idx + pct_inner) / 3 * 0.85))
                elapsed = time.time() - start_time
                status_msg.markdown(f"**Step {a_idx+2}/4**: {asset} | {current_date.strftime('%b %Y')} | {int(elapsed)}s")
            oos_df, sel_df, coverage_stats = run_walk_forward_backtest(y[asset], X, min_train_months, horizon_months, rebalance_freq, asset, selection_threshold, l1_ratio, confidence_level, update_progress, X_precomputed)
            results[asset], heatmaps[asset], coverage[asset] = oos_df, sel_df, coverage_stats
        status.update(label="✅ Complete", state="complete", expanded=False)
    return results, heatmaps, coverage

def get_live_model_signals_v4(y, X, l1_ratio, selection_threshold, window_years, horizon_months, confidence_level=0.90):
    expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats = {}, {}, {}, {}, {}
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        stab_results = stability_analysis(y[asset], X, horizon_months, window_years)
        metrics = compute_stability_metrics(stab_results, X.columns.tolist())
        stable_features = metrics[metrics['persistence'] >= selection_threshold].sort_values('persistence', ascending=False)['feature'].tolist()
        if not stable_features: stable_features = X.columns[:5].tolist()
        y_valid, X_valid = y[asset].loc[X.index].dropna(), X.loc[y[asset].dropna().index]
        X_sel = X_valid[stable_features]
        if asset in ['BONDS', 'GOLD']:
            hac_results = estimate_with_corrected_inference(y_valid, X_sel, {'inference_method': st.session_state.get('inference_method', 'Hodrick (1992)'), 'horizon': horizon_months})
            exp_ret = compute_expected_returns(X_sel.iloc[-1], hac_results['coefficients'], hac_results.get('intercept', 0))
            lower, upper = exp_ret - 1.645 * hac_results['std_errors'].mean(), exp_ret + 1.645 * hac_results['std_errors'].mean()
        else:
            m = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05); m.fit(X_sel, y_valid)
            exp_ret = m.predict(X_sel.iloc[[-1]])[0]
            bt = BootstrapPredictionInterval(confidence_level=confidence_level, n_bootstrap=20); bt.fit(m, X_sel, y_valid)
            _, lower, upper = bt.predict_interval(X_sel.iloc[[-1]])
            hac_results = {'model': m, 'importance': pd.Series(m.feature_importances_, index=stable_features)}
        expected_returns[asset], confidence_intervals[asset] = exp_ret, [lower[0] if isinstance(lower, np.ndarray) else lower, upper[0] if isinstance(upper, np.ndarray) else upper]
        driver_attributions[asset] = compute_driver_attribution(X_sel.iloc[-1], hac_results.get('coefficients', hac_results.get('importance', pd.Series())), X_sel.mean())
        # Add Impact and other columns for UI compatibility
        attr = driver_attributions[asset]
        attr['Impact'] = attr['contribution']
        attr['Weight'] = attr['coefficient']
        attr['State'] = attr['deviation']
        attr['Link'] = attr['contribution'] # Simplified for now
        attr['Signal'] = attr['direction']
        stability_results_map[asset] = {'metrics': metrics, 'stable_features': stable_features, 'hac_results': hac_results, 'all_coefficients': pd.DataFrame([res['coefficients'] for res in stab_results])}
        model_stats[asset] = hac_results
    return expected_returns, confidence_intervals, driver_attributions, stability_results_map, model_stats
