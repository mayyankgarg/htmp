# hull_full_optuna_model_level_enhanced.py
"""
Enhanced Hull competition pipeline with:
- volatility targeting (rolling realized vol scaling of allocations)
- winsorization (MAD-based) + outlier "no-trade" rule
- regime detection using HMM (if available) + per-regime mapping
- rolling feature creation (rolling slopes, vol, PCA)
- additional meta-stackers: L1 (Lasso), BayesianRidge, Quantile regression
- behavioral/meta tricks: quantize small prediction steps, rolling mean, lagging
- vectorized operations and parallelized Optuna/model tuning
- tuned Optuna search spaces with meaningful ranges

Notes:
- Requires: scikit-learn, optuna, numpy, pandas, joblib. Optional: lightgbm, xgboost, catboost, hmmlearn
- Designed to run on a multi-core machine (n_jobs configurable). If running on Kaggle, set n_workers appropriately.

Usage:
    python hull_full_optuna_model_level_enhanced.py --train /mnt/data/train.csv --test /mnt/data/test.csv

"""
import os
import json
import pickle

import argparse
import numpy as np
from typing import Optional
import pandas as pd
import pandas.api.types
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import joblib
import warnings
import math
import time
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Optional boosters
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import catboost as cb
except Exception:
    cb = None
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


# Configs
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS = 252
DEFAULT_N_JOBS = max(1, (joblib.cpu_count() or 4) - 2)

class ParticipantVisibleError(Exception):
    pass

# --------------------------
# Smart preprocessing / imputation utilities
# --------------------------

def compute_missingness_stats(df, feature_cols, early_frac=0.25):
    """Return missing fraction overall and missing fraction in early period."""
    n = len(df)
    early_cut = int(n * early_frac)
    stats = {}
    for c in feature_cols:
        col = df[c]
        overall = float(col.isna().mean())
        early = float(col.iloc[:early_cut].isna().mean()) if early_cut > 0 else 0.0
        stats[c] = {"overall_missing": overall, "early_missing": early}
    return stats

def add_missing_indicators(df, feature_cols):
    """Add binary is_missing flag for each feature (suffix: __miss)."""
    for c in feature_cols:
        df[f"{c}__miss"] = df[c].isna().astype(int)
    return df

def mark_introduced_late(df, feature_cols, early_frac=0.25, overall_thresh=0.4, early_thresh=0.6):
    """
    Mark features that were mostly absent in the early period but present later.
    Returns dict col -> bool
    """
    n = len(df)
    early_cut = int(n * early_frac)
    introduced = {}
    for c in feature_cols:
        overall = float(df[c].isna().mean())
        early = float(df[c].iloc[:early_cut].isna().mean()) if early_cut>0 else 0.0
        introduced[c] = (overall > overall_thresh and early > early_thresh)
        if introduced[c]:
            df[f"{c}__introduced_late"] = 1
        else:
            df[f"{c}__introduced_late"] = 0
    return df, introduced

def rolling_median_impute(df, feature_cols, window=252, min_periods=5):
    """
    Replace NaNs with rolling median computed on past values (causal).
    For the first rows where rolling median is undefined, fallback to global median of available training data.
    This is causal-friendly: median at t uses data up to t (including t if non-null).
    """
    out = df.copy()
    global_medians = df[feature_cols].median()
    for c in feature_cols:
        ser = df[c]
        # rolling median with center=False uses past window including current index
        rolling_med = ser.rolling(window=window, min_periods=min_periods).median()
        # fill NaNs in rolling_med with global median (to handle start)
        rolling_med_filled = rolling_med.fillna(global_medians[c])
        # now fill original NaNs with rolling_med_filled values
        mask_na = ser.isna()
        filled = ser.copy()
        filled[mask_na] = rolling_med_filled[mask_na]
        # final fallback: if still NaN (very start), fill with global median
        filled = filled.fillna(global_medians[c])
        out[c] = filled
    return out

def sentinel_impute(df, feature_cols, sentinel=-999.0):
    """Fill NaNs with sentinel large value; keep missing-indicators to signal sentinel to model."""
    out = df.copy()
    out[feature_cols] = out[feature_cols].fillna(sentinel)
    return out

def prepare_datasets(train_df, test_df=None, exclude_cols=None, early_frac=0.25, rolling_window=252):
    """
    Returns a dict with:
      - feature_cols: list of feature columns used
      - X_for_trees: DataFrame (leave NaNs intact; missing indicators present)
      - X_for_linear: DataFrame (rolling-median-imputed numeric features + missing indicators + introduced flag)
      - optionally X_test_for_trees / X_test_for_linear if test_df provided (test imputation uses training medians/rolling approach)
    Usage:
      call prepare_datasets(...) after loading train (and test) and after deciding EXCLUDE_COLS
    """
    assert exclude_cols is not None
    # select features
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    # copy to avoid accidental mutation
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True) if test_df is not None else None

    # compute missing stats and mark introduced-late features
    stats = compute_missingness_stats(train, feature_cols, early_frac=early_frac)
    train = add_missing_indicators(train, feature_cols)
    train, introduced = mark_introduced_late(train, feature_cols, early_frac=early_frac)

    # Build X_for_trees: keep original numeric columns (NaNs allowed) + missing indicators + introduced flag
    cats_to_keep = [f"{c}__miss" for c in feature_cols] + [f"{c}__introduced_late" for c in feature_cols]
    X_for_trees = pd.concat([train[feature_cols], train[cats_to_keep]], axis=1)

    # Build X_for_linear: rolling median imputation for numeric features + missing indicators + introduced flag
    train_imputed = rolling_median_impute(train, feature_cols, window=rolling_window)
    X_for_linear = pd.concat([train_imputed[feature_cols], train[[*cats_to_keep]]], axis=1)

    # If test provided, impute test using training rolling approach:
    X_test_for_trees = None
    X_test_for_linear = None
    if test is not None:
        # add indicators / introduced flag for test using same logic (introduced flags from train)
        test = test.copy().reset_index(drop=True)
        for c in feature_cols:
            test[f"{c}__miss"] = test[c].isna().astype(int)
            # introduced_late flag uses train-defined introduced (structural)
            test[f"{c}__introduced_late"] = int(introduced.get(c, False))
        X_test_for_trees = pd.concat([test[feature_cols], test[cats_to_keep]], axis=1)
        # For linear: perform rolling median imputation on test using train global medians as fallback.
        # A simple approach: concatenate train+test, perform rolling impute on the concat (so test uses prior train history)
        combined = pd.concat([train[feature_cols], test[feature_cols]], axis=0).reset_index(drop=True)
        combined_imputed = rolling_median_impute(combined, feature_cols, window=rolling_window)
        # split back
        train_imputed_combined = combined_imputed.iloc[:len(train), :].reset_index(drop=True)
        test_imputed_combined = combined_imputed.iloc[len(train):, :].reset_index(drop=True)
        X_test_for_linear = pd.concat([test_imputed_combined.reset_index(drop=True), test[[*cats_to_keep]].reset_index(drop=True)], axis=1)

    result = {
        "feature_cols": feature_cols,
        "X_for_trees": X_for_trees,
        "X_for_linear": X_for_linear,
        "X_test_for_trees": X_test_for_trees,
        "X_test_for_linear": X_test_for_linear,
        "missing_stats": stats,
        "introduced_flags": introduced
    }
    return result


# --------------------------
# Scoring (unchanged but clearer units)
# --------------------------
def hull_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "date_id") -> float:
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')
    sol = solution.copy().reset_index(drop=True)
    sol['position'] = submission['prediction'].values
    if sol['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {sol["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if sol['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {sol["position"].min()} below minimum of {MIN_INVESTMENT}')

    # strategy returns = risk_free*(1-position) + position * forward_returns
    sol['strategy_returns'] = sol['risk_free_rate'] * (1 - sol['position']) + sol['position'] * sol['forward_returns']
    strategy_excess_returns = sol['strategy_returns'] - sol['risk_free_rate']

    # geometric mean annualized
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    T = len(sol)
    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / T) - 1
    strategy_std = sol['strategy_returns'].std()
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(TRADING_DAYS)

    strategy_volatility = float(strategy_std * np.sqrt(TRADING_DAYS) * 100)

    market_excess_returns = sol['forward_returns'] - sol['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = market_excess_cumulative ** (1 / T) - 1
    market_std = sol['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(TRADING_DAYS) * 100)
    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * TRADING_DAYS)
    return_penalty = 1 + (return_gap ** 2) / 100
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

# --------------------------
# Feature engineering: rolling slopes, vol, PCA
# --------------------------

def add_rolling_features(df, price_col='forward_returns', windows=[5, 10, 20, 60, 120, 252], slope_windows=[5,21,60]):
    df = df.copy()
    for w in windows:
        df[f'ret_mean_{w}'] = df[price_col].rolling(window=w, min_periods=1).mean()
        df[f'ret_std_{w}'] = df[price_col].rolling(window=w, min_periods=1).std().fillna(0)
        df[f'ret_mad_{w}'] = df[price_col].rolling(window=w, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))) if len(x)>0 else 0)
    for w in slope_windows:
        df[f'ret_slope_{w}'] = df[price_col].rolling(window=w, min_periods=2).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>1 else 0)
    # Fill NaNs forward/back
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df


def add_pca_features(df, feature_cols, n_components=3):
    p = PCA(n_components=n_components)
    X = df[feature_cols].fillna(0.0).values
    comps = p.fit_transform(X)
    for i in range(comps.shape[1]):
        df[f'pca_{i}'] = comps[:, i]
    return df, p

# --------------------------
# Winsorization / MAD clipping + tiny quantization + rolling mean + lag
# --------------------------

def winsorize_mad(a, n_mad=3.0):
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad < 1e-9:
        return a
    low = med - n_mad * mad
    high = med + n_mad * mad
    return np.clip(a, low, high)


def quantize_array(a, step=1e-5):
    return np.round(a / step) * step

# 5-20 day rolling mean
def rolling_mean_array(a, window=5):
    s = pd.Series(a)
    return s.rolling(window, min_periods=1).mean().values

# lagging
def lag_array(a, lag=1):
    if lag <= 0:
        return a
    out = np.empty_like(a)
    out[:lag] = a[:lag]
    out[lag:] = a[:-lag]
    return out

# --------------------------
# Volatility targeting (vectorized)
# --------------------------

def apply_vol_targeting_vectorized(alloc, forward_returns, window=60, target_multiplier=1.2, market_window=60):
    # vectorized rolling std using pandas
    strat_ret = alloc * forward_returns
    strat_vol = pd.Series(strat_ret).rolling(window, min_periods=2).std().fillna(method='bfill').values
    market_vol = pd.Series(forward_returns).rolling(market_window, min_periods=2).std().fillna(method='bfill').values
    scale = np.minimum(1.0, (target_multiplier * market_vol) / (strat_vol + 1e-9))
    return alloc * scale

# --------------------------
# Regime detection using HMM (or KMeans fallback)
# --------------------------

def infer_regimes_hmm(features_matrix, n_states=2):
    if GaussianHMM is None:
        km = KMeans(n_clusters=n_states, random_state=0).fit(features_matrix)
        return km, km.labels_
    else:
        model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=0)
        model.fit(features_matrix)
        states = model.predict(features_matrix)
        return model, states

# --------------------------
# Model templates
# --------------------------

def base_model_templates(random_state=42):
    templates = {}

    templates["ridge"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=(trial.params.get('ridge_alpha') if (trial and 'ridge_alpha' in getattr(trial,'params',{})) else 1.0), random_state=random_state))
    ])

    templates["enet"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=(trial.params.get('enet_alpha') if (trial and 'enet_alpha' in getattr(trial,'params',{})) else 1e-3), l1_ratio=(trial.params.get('enet_l1') if (trial and 'enet_l1' in getattr(trial,'params',{})) else 0.5), random_state=random_state))
    ])

    templates["lasso"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=(trial.params.get('lasso_alpha') if (trial and 'lasso_alpha' in getattr(trial,'params',{})) else 1e-3), random_state=random_state))
    ])

    templates["bayesridge"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", BayesianRidge())
    ])

    templates["hgb"] = lambda trial=None: Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_depth=(trial.params.get('hgb_max_depth') if (trial and 'hgb_max_depth' in getattr(trial,'params',{})) else 3),
            learning_rate=(trial.params.get('hgb_lr') if (trial and 'hgb_lr' in getattr(trial,'params',{})) else 0.05),
            max_iter=(trial.params.get('hgb_iter') if (trial and 'hgb_iter' in getattr(trial,'params',{})) else 300),
            random_state=random_state
        ))
    ])

    templates["rf"] = lambda trial=None: Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=(trial.params.get('rf_n') if (trial and 'rf_n' in getattr(trial,'params',{})) else 200),
            max_depth=(trial.params.get('rf_depth') if (trial and 'rf_depth' in getattr(trial,'params',{})) else 6),
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    templates["et"] = lambda trial=None: Pipeline([
        ("model", ExtraTreesRegressor(
            n_estimators=(trial.params.get('et_n') if (trial and 'et_n' in getattr(trial,'params',{})) else 200),
            max_depth=(trial.params.get('et_depth') if (trial and 'et_depth' in getattr(trial,'params',{})) else 6),
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    templates["knn"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=(trial.params.get('knn_k') if (trial and 'knn_k' in getattr(trial,'params',{})) else 10), n_jobs=-1))
    ])

    templates["mlp"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(trial.params.get('mlp_hidden') if (trial and 'mlp_hidden' in getattr(trial,'params',{})) else (64,32)),
            alpha=(trial.params.get('mlp_alpha') if (trial and 'mlp_alpha' in getattr(trial,'params',{})) else 1e-4),
            max_iter=(trial.params.get('mlp_iter') if (trial and 'mlp_iter' in getattr(trial,'params',{})) else 800),
            random_state=random_state
        ))
    ])

    if lgb is not None:
        templates["lgb"] = lambda trial=None: Pipeline([
            ("model", lgb.LGBMRegressor(
                n_estimators=(trial.params.get('lgb_n') if (trial and 'lgb_n' in getattr(trial,'params',{})) else 500),
                num_leaves=(trial.params.get('lgb_leaves') if (trial and 'lgb_leaves' in getattr(trial,'params',{})) else 31),
                learning_rate=(trial.params.get('lgb_lr') if (trial and 'lgb_lr' in getattr(trial,'params',{})) else 0.05),
                random_state=random_state,
                n_jobs=-1,
            ))
        ])

    if xgb is not None:
        templates["xgb"] = lambda trial=None: Pipeline([
            ("model", xgb.XGBRegressor(
                n_estimators=(trial.params.get('xgb_n') if (trial and 'xgb_n' in getattr(trial,'params',{})) else 400),
                max_depth=(trial.params.get('xgb_depth') if (trial and 'xgb_depth' in getattr(trial,'params',{})) else 4),
                learning_rate=(trial.params.get('xgb_lr') if (trial and 'xgb_lr' in getattr(trial,'params',{})) else 0.05),
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ))
        ])

    if cb is not None:
        templates["cat"] = lambda trial=None: Pipeline([
            ("model", cb.CatBoostRegressor(
                iterations=(trial.params.get('cb_iter') if (trial and 'cb_iter' in getattr(trial,'params',{})) else 400),
                depth=(trial.params.get('cb_depth') if (trial and 'cb_depth' in getattr(trial,'params',{})) else 4),
                learning_rate=(trial.params.get('cb_lr') if (trial and 'cb_lr' in getattr(trial,'params',{})) else 0.04),
                verbose=False,
                random_seed=random_state,
            ))
        ])

    # also include stacked meta-model templates (constructed later) but keep reachable names
    templates["quantile"] = lambda trial=None: Pipeline([
        ("scaler", StandardScaler()),
        ("model", QuantileRegressor(alpha=trial.params.get('quant_alpha', 0.1) if (trial and 'quant_alpha' in getattr(trial,'params',{})) else 0.1, quantile=0.5))
    ])

    return templates

# --------------------------
# Time-series OOF (vectorized where possible)
# --------------------------

def timeseries_oof_for_model(build_pipeline_fn, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n = X.shape[0]
    oof = np.zeros(n, dtype=float)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val = X.iloc[val_idx]
        mdl = build_pipeline_fn(trial=None)
        try:
            mdl.fit(X_tr, y_tr)
            oof[val_idx] = mdl.predict(X_val)
        except Exception:
            oof[val_idx] = 0.0
        # optional: print per-fold
    return oof

# --------------------------
# Per-model Optuna tuning: tuned search spaces narrowed to meaningful ranges
# and objective uses a small internal mapping test
# --------------------------

def tune_single_model(model_key, build_fn_template, X, y, forward_returns, risk_free, n_splits=4, trials=40, random_state=42, n_jobs=1):
    def objective(trial):
        if model_key == "ridge":
            trial.suggest_float("ridge_alpha", 1e-5, 10.0, log=True)
            # add small jitter to promote diversity
            trial.suggest_float("ridge_jitter", 0.0, 0.05)
        elif model_key == "enet":
            trial.suggest_float("enet_alpha", 1e-6, 1.0, log=True)
            trial.suggest_float("enet_l1", 0.0, 1.0)
            trial.suggest_float("enet_jitter", 0.0, 0.05)
        elif model_key == "lasso":
            trial.suggest_float("lasso_alpha", 1e-6, 1.0, log=True)
            trial.suggest_float("lasso_jitter", 0.0, 0.05)
        elif model_key == "hgb":
            trial.suggest_int("hgb_max_depth", 2, 6)
            trial.suggest_float("hgb_lr", 0.01, 0.2)
            trial.suggest_int("hgb_iter", 50, 600)
            # promote decorrelation: feature subsampling & min_samples_leaf
            trial.suggest_float("hgb_max_features", 0.4, 1.0)
            trial.suggest_int("hgb_min_samples_leaf", 10, 200)
        elif model_key == "rf":
            trial.suggest_int("rf_n", 50, 600)
            trial.suggest_int("rf_depth", 3, 12)
            trial.suggest_float("rf_max_features", 0.2, 0.9)
            trial.suggest_int("rf_min_samples_leaf", 5, 100)
            trial.suggest_int("rf_bootstrap", 0, 1)
        elif model_key == "et":
            trial.suggest_int("et_n", 50, 600)
            trial.suggest_int("et_depth", 3, 12)
            trial.suggest_float("et_max_features", 0.2, 0.9)
            trial.suggest_int("et_min_samples_leaf", 5, 100)
        elif model_key == "knn":
            trial.suggest_int("knn_k", 3, 50)
            trial.suggest_categorical("knn_weights", ["uniform", "distance"])
            trial.suggest_categorical("knn_metric", ["minkowski", "chebyshev"])
        elif model_key == "mlp":
            h1 = trial.suggest_int("mlp_h1", 16, 256)
            h2 = trial.suggest_int("mlp_h2", 0, 128)
            hidden = (h1, h2) if h2>0 else (h1,)
            trial.set_user_attr("mlp_hidden_tuple", hidden)
            trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True)
            trial.suggest_int("mlp_iter", 200, 1500)
            trial.suggest_float("mlp_dropout", 0.0, 0.4)
            trial.suggest_float("mlp_lr_init", 1e-4, 1e-2, log=True)
        elif model_key == "lgb" and lgb is not None:
            trial.suggest_int("lgb_n", 50, 1000)
            trial.suggest_int("lgb_leaves", 4, 32)
            trial.suggest_int("lgb_depth", 2, 5)
            trial.suggest_float("lgb_lr", 0.001, 0.1)
            trial.suggest_float("lgb_feat_fr", 0.1, 0.8)
            trial.suggest_float("lgb_colsample_bytree", 0.4, 1.0)
            trial.suggest_float("lgb_subsample", 0.5, 1.0)
            trial.suggest_float("lgb_lambda_l2", 0.0, 10.0)
        elif model_key == "xgb" and xgb is not None:
            trial.suggest_int("xgb_n", 50, 800)
            trial.suggest_int("xgb_depth", 2, 5)
            trial.suggest_float("xgb_lr", 0.01, 0.2)
            trial.suggest_float("xgb_subsample", 0.5, 1.0)
            trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0)
            trial.suggest_float("xgb_lambda", 0.0, 10.0)
        elif model_key == "cat" and cb is not None:
            trial.suggest_int("cb_iter", 50, 800)
            trial.suggest_int("cb_depth", 2, 5)
            trial.suggest_float("cb_lr", 0.01, 0.2)
            trial.suggest_float("cb_subsample", 0.5, 1.0)
            trial.suggest_float("cb_l2_leaf_reg", 0.0, 10.0)

        class DummyTrial:
            def __init__(self, params, user_attrs):
                self.params = params
                self.user_attrs = user_attrs
            def set_user_attr(self, k, v):
                self.user_attrs[k] = v

        user_attrs = {}
        trial_obj = DummyTrial(trial.params, user_attrs)
        if user_attrs.get("mlp_hidden_tuple"):
            trial.params["mlp_hidden"] = user_attrs["mlp_hidden_tuple"]

        model_pipeline = build_fn_template(trial_obj)
        # Apply suggested hyperparameters to underlying estimator when available
        try:
            est = None
            if hasattr(model_pipeline, 'steps'):
                for name, step in model_pipeline.steps:
                    if hasattr(step, 'fit') and name != 'scaler':
                        est = step
                        break
            if est is not None:
                if model_key == 'hgb':
                    if hasattr(est, 'set_params'):
                        est.set_params(max_features=trial.params.get('hgb_max_features', None),
                                       min_samples_leaf=trial.params.get('hgb_min_samples_leaf', None))
                elif model_key == 'rf':
                    est.set_params(max_features=trial.params.get('rf_max_features', 'auto'),
                                   min_samples_leaf=trial.params.get('rf_min_samples_leaf', 1),
                                   bootstrap=bool(trial.params.get('rf_bootstrap', 1)))
                elif model_key == 'et':
                    est.set_params(max_features=trial.params.get('et_max_features', 'auto'),
                                   min_samples_leaf=trial.params.get('et_min_samples_leaf', 1))
                elif model_key == 'knn':
                    est.set_params(weights=trial.params.get('knn_weights', 'uniform'),
                                   metric=trial.params.get('knn_metric', 'minkowski'))
                elif model_key == 'mlp':
                    est.set_params(learning_rate_init=trial.params.get('mlp_lr_init', 1e-3))
                elif model_key == 'lgb' and lgb is not None:
                    est.set_params(feature_fraction=trial.params.get('lgb_colsample_bytree', 0.8),
                                   subsample=trial.params.get('lgb_subsample', 1.0),
                                   reg_lambda=trial.params.get('lgb_lambda_l2', 0.0))
                elif model_key == 'xgb' and xgb is not None:
                    est.set_params(subsample=trial.params.get('xgb_subsample', 1.0),
                                   colsample_bytree=trial.params.get('xgb_colsample_bytree', 1.0),
                                   reg_lambda=trial.params.get('xgb_lambda', 0.0))
                elif model_key == 'cat' and cb is not None:
                    est.set_params(subsample=trial.params.get('cb_subsample', 1.0),
                                   l2_leaf_reg=trial.params.get('cb_l2_leaf_reg', 0.0))
        except Exception:
            pass

        # OOF preds for model
        oof = np.zeros(X.shape[0], dtype=float)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val = X.iloc[val_idx]
            try:
                model_pipeline.fit(X_tr, y_tr)
                oof[val_idx] = model_pipeline.predict(X_val)
            except Exception:
                return -1e9

        # simple mapping candidates (small grid) for objective
        std_oof = oof.std() if oof.std() != 0 else 1.0
        z = oof / std_oof
        best_local = -1e9
        for k_try in [0.3, 0.6, 1.0, 1.5]:
            for scale_try in [0.5, 1.0, 2.0]:
                alloc_raw = 1.0 + k_try * np.tanh(z / scale_try)
                alloc = np.clip(alloc_raw, 0.0, 2.0)
                sol = pd.DataFrame({"forward_returns": forward_returns, "risk_free_rate": risk_free})
                sub = pd.DataFrame({"prediction": alloc})
                try:
                    score_val = hull_score(sol, sub)
                except ParticipantVisibleError:
                    score_val = -1e9
                if score_val > best_local:
                    best_local = score_val
        return best_local

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)

    best_trial = study.best_trial
    class DummyTrial2:
        def __init__(self, params, user_attrs):
            self.params = params
            self.user_attrs = user_attrs
        def set_user_attr(self, k, v):
            self.user_attrs[k] = v

    dt2 = DummyTrial2(best_trial.params, {})
    if "mlp_h1" in best_trial.params:
        h1 = best_trial.params.get("mlp_h1")
        h2 = best_trial.params.get("mlp_h2", 0)
        dt2.params["mlp_hidden"] = (h1, h2) if h2>0 else (h1,)
    best_pipeline = build_fn_template(dt2)
    return study, best_pipeline

# --------------------------
# Blending + mapping (improved: per-regime mapping support + more params)
# --------------------------

def build_alloc_from_preds_array(preds, params, pred_std=None, forward_returns=None):
    preds = np.asarray(preds, dtype=float)
    if pred_std is None:
        pred_std = preds.std() if preds.std() != 0 else 1.0
    z = preds / (pred_std + 1e-8)

    # optional winsorize on z
    if params.get('winsorize_pred', False):
        z = winsorize_mad(z, n_mad=params.get('winsor_nmad', 3.0))

    # optional rolling mean smoothing on z
    rm = params.get('pred_roll', 0)
    if rm > 1:
        z = rolling_mean_array(z, window=rm)

    # pred ema
    alpha_p = params.get("pred_ema_alpha", 0.0)
    if alpha_p > 0:
        z_smoothed = np.zeros_like(z)
        z_smoothed[0] = z[0]
        for i in range(1, len(z)):
            z_smoothed[i] = alpha_p * z[i] + (1 - alpha_p) * z_smoothed[i-1]
    else:
        z_smoothed = z

    k = params.get("k", 1.0)
    scale = params.get("scale", 1.0)
    max_leverage = params.get("max_leverage", 2.0)
    min_leverage = params.get("min_leverage", 0.0)

    # piecewise option
    if params.get('use_pwl', False):
        # simple 3-segment PWL mapping: param a,b breakpoints
        a = params.get('pwl_a', 0.5)
        b = params.get('pwl_b', -0.5)
        alloc_raw = 1.0 + k * np.where(z_smoothed> a, (z_smoothed - a), np.where(z_smoothed < b, (z_smoothed - b), 0.0))
    else:
        alloc_raw = 1.0 + k * np.tanh(z_smoothed / scale)

    alloc_raw = np.clip(alloc_raw, min_leverage, max_leverage)

    # allocation EMA smoothing
    alpha_a = params.get("alloc_ema_alpha", 0.0)
    if alpha_a > 0:
        alloc = np.zeros_like(alloc_raw)
        alloc[0] = alloc_raw[0]
        for i in range(1, len(alloc_raw)):
            alloc[i] = alpha_a * alloc_raw[i] + (1 - alpha_a) * alloc[i-1]
    else:
        alloc = alloc_raw

    # vol targeting (if forward_returns provided)
    if params.get('vol_target', False) and forward_returns is not None:
        alloc = apply_vol_targeting_vectorized(alloc, forward_returns, window=params.get('vol_window', 60), target_multiplier=params.get('vol_mult',1.2), market_window=params.get('market_window',60))

    # tiny quantization
    if params.get('quantize_step', None) is not None:
        alloc = quantize_array(alloc, step=params.get('quantize_step', 1e-5))

    # winsorize allocations
    if params.get('winsorize_alloc', False):
        alloc = winsorize_mad(alloc, n_mad=params.get('winsor_alloc_nmad', 3.0))

    # outlier days no-trade: based on forward_returns magnitude
    if params.get('outlier_no_trade', False) and forward_returns is not None:
        fr = np.asarray(forward_returns)
        thr = params.get('outlier_thr', np.nanpercentile(np.abs(fr), 98))
        mask = np.abs(fr) > thr
        # set those days to previous allocation (lag)
        for i in range(len(alloc)):
            if mask[i] and i>0:
                alloc[i] = alloc[i-1]

    # lag predictions to reduce microstructure
    lag = params.get('lag', 0)
    if lag > 0:
        alloc = lag_array(alloc, lag=lag)

    return alloc


def run_optuna_blend_and_map(oof_df: pd.DataFrame, forward_returns: np.ndarray, risk_free: np.ndarray, n_trials=120, n_jobs=1, random_state=42, regimes=None):
    model_names = list(oof_df.columns)
    X_oof = oof_df.values
    n_models = X_oof.shape[1]

    def objective(trial):
        # Learn weights. If regimes provided, learn separate weights per regime
        if regimes is not None:
            regs_arr = np.asarray(regimes)
            uniq_regs = np.unique(regs_arr)
            blended = np.zeros(X_oof.shape[0], dtype=float)
            per_reg_weights = {}
            for r in uniq_regs:
                raw_r = np.array([trial.suggest_float(f"w_r{int(r)}_{i}", 0.0, 1.0) for i in range(n_models)])
                if raw_r.sum() == 0:
                    weights_r = np.ones(n_models)/n_models
                else:
                    weights_r = raw_r / raw_r.sum()
                per_reg_weights[int(r)] = weights_r
                mask = regs_arr == r
                if mask.any():
                    blended[mask] = X_oof[mask].dot(weights_r)
            # Use global std as a fallback for normalization inside mapping
            pred_std = blended.std() if blended.std() != 0 else 1.0
        else:
            raw = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)])
            if raw.sum() == 0:
                weights = np.ones(n_models)/n_models
            else:
                weights = raw / raw.sum()
            blended = X_oof.dot(weights)
            pred_std = blended.std() if blended.std() != 0 else 1.0

        # Regime-specific mapping params
        if regimes is not None:
            regs = np.asarray(regimes)
            uniq_regs = np.unique(regs)
            alloc = np.zeros_like(blended)
            for r in uniq_regs:
                # Suggest parameters per regime r
                k = trial.suggest_float(f"r{int(r)}_k", 0.2, 3.0)
                scale = trial.suggest_float(f"r{int(r)}_scale", 0.2, 3.0)
                pred_ema_alpha = trial.suggest_float(f"r{int(r)}_pred_ema_alpha", 0.0, 0.3)
                alloc_ema_alpha = trial.suggest_float(f"r{int(r)}_alloc_ema_alpha", 0.0, 0.3)
                max_leverage = trial.suggest_float(f"r{int(r)}_max_leverage", 1.0, 2.0)
                pred_roll = trial.suggest_int(f"r{int(r)}_pred_roll", 0, 20)
                winsorize_pred = trial.suggest_categorical(f"r{int(r)}_winsorize_pred", [False, True])
                winsor_nmad = trial.suggest_float(f"r{int(r)}_winsor_nmad", 2.0, 4.0)
                vol_target = trial.suggest_categorical(f"r{int(r)}_vol_target", [False, True])
                vol_window = trial.suggest_int(f"r{int(r)}_vol_window", 30, 90)
                vol_mult = trial.suggest_float(f"r{int(r)}_vol_mult", 0.8, 1.6)
                quant_step = trial.suggest_categorical(f"r{int(r)}_quant_step", [None, 1e-5, 5e-5, 1e-4])
                outlier_no_trade = trial.suggest_categorical(f"r{int(r)}_outlier_no_trade", [False, True])
                outlier_thr = trial.suggest_float(f"r{int(r)}_outlier_thr_pctl", 90.0, 99.9)
                lag = trial.suggest_int(f"r{int(r)}_lag", 0, 3)

                params_r = dict(k=k, scale=scale, pred_ema_alpha=pred_ema_alpha, alloc_ema_alpha=alloc_ema_alpha,
                                 max_leverage=max_leverage, min_leverage=0.0,
                                 pred_roll=pred_roll, winsorize_pred=winsorize_pred, winsor_nmad=winsor_nmad,
                                 vol_target=vol_target, vol_window=vol_window, vol_mult=vol_mult, market_window=vol_window,
                                 quantize_step=quant_step, outlier_no_trade=outlier_no_trade,
                                 outlier_thr=np.nanpercentile(np.abs(forward_returns), outlier_thr), lag=lag)
                mask = regs == r
                if mask.any():
                    alloc_r = build_alloc_from_preds_array(blended[mask], params_r, pred_std=pred_std, forward_returns=np.asarray(forward_returns)[mask])
                    alloc[mask] = alloc_r
        else:
            # Global mapping if no regimes provided
            k = trial.suggest_float("k", 0.2, 3.0)
            scale = trial.suggest_float("scale", 0.2, 3.0)
            pred_ema_alpha = trial.suggest_float("pred_ema_alpha", 0.0, 0.3)
            alloc_ema_alpha = trial.suggest_float("alloc_ema_alpha", 0.0, 0.3)
            max_leverage = trial.suggest_float("max_leverage", 1.0, 2.0)
            pred_roll = trial.suggest_int("pred_roll", 0, 20)
            winsorize_pred = trial.suggest_categorical('winsorize_pred', [False, True])
            winsor_nmad = trial.suggest_float('winsor_nmad', 2.0, 4.0)
            vol_target = trial.suggest_categorical('vol_target', [False, True])
            vol_window = trial.suggest_int('vol_window', 30, 90)
            vol_mult = trial.suggest_float('vol_mult', 0.8, 1.6)
            quant_step = trial.suggest_categorical('quant_step', [None, 1e-5, 5e-5, 1e-4])
            outlier_no_trade = trial.suggest_categorical('outlier_no_trade', [False, True])
            outlier_thr = trial.suggest_float('outlier_thr_pctl', 90.0, 99.9)
            lag = trial.suggest_int('lag', 0, 3)

            params = dict(k=k, scale=scale, pred_ema_alpha=pred_ema_alpha, alloc_ema_alpha=alloc_ema_alpha, max_leverage=max_leverage, min_leverage=0.0,
                          pred_roll=pred_roll, winsorize_pred=winsorize_pred, winsor_nmad=winsor_nmad,
                          vol_target=vol_target, vol_window=vol_window, vol_mult=vol_mult, market_window=vol_window,
                          quantize_step=quant_step, outlier_no_trade=outlier_no_trade, outlier_thr=np.nanpercentile(np.abs(forward_returns), outlier_thr),
                          lag=lag)
            alloc = build_alloc_from_preds_array(blended, params, pred_std=pred_std, forward_returns=forward_returns)

        solution = pd.DataFrame({"forward_returns": forward_returns, "risk_free_rate": risk_free})
        submission = pd.DataFrame({"prediction": alloc})
        try:
            score_val = hull_score(solution, submission)
        except ParticipantVisibleError:
            score_val = -1e9
        return score_val

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    # Extract best weights (global or per-regime)
    if regimes is not None:
        regs_arr = np.asarray(regimes)
        uniq_regs = np.unique(regs_arr)
        per_regime_weights = {}
        for r in uniq_regs:
            raw_r = np.array([study.best_trial.params.get(f"w_r{int(r)}_{i}", 0.0) for i in range(n_models)])
            if raw_r.sum() == 0:
                weights_r = np.ones(n_models)/n_models
            else:
                weights_r = raw_r / raw_r.sum()
            per_regime_weights[int(r)] = weights_r
        best_weights = {"per_regime": per_regime_weights}
    else:
        best_raw = np.array([study.best_trial.params.get(f"w_{i}", 0.0) for i in range(n_models)])
        if best_raw.sum()==0:
            best_weights = np.ones(n_models)/n_models
        else:
            best_weights = best_raw / best_raw.sum()
    # Extract best mapping(s)
    all_params = {k:v for k,v in study.best_trial.params.items() if not k.startswith("w_")}
    if regimes is not None:
        regs = np.unique(np.asarray(regimes))
        per_regime = {}
        for r in regs:
            key_prefix = f"r{int(r)}_"
            params_r = {k[len(key_prefix):]: v for k, v in all_params.items() if k.startswith(key_prefix)}
            if params_r:
                per_regime[int(r)] = params_r
        best_mapping = {"per_regime": per_regime}
    else:
        best_mapping = all_params
    return study, model_names, best_weights, best_mapping

# --------------------------
# Utilities to fit and predict
# --------------------------

def fit_full_models(models_dict, X, y, n_jobs=DEFAULT_N_JOBS):
    fitted = {}

    def fit_one(args):
        name, pipeline = args
        mdl = clone(pipeline)
        mdl.fit(X[name], y)
        return name, mdl

    items = list(models_dict.items())

    # Use processes (true parallelism, like joblib)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(fit_one, item) for item in items]

        for fut in as_completed(futures):
            name, mdl = fut.result()
            fitted[name] = mdl

    return fitted



def ensemble_predict_from_fitted(fitted_models, weights, X, regimes: Optional[np.ndarray] = None):
    preds_list = []
    names = list(fitted_models.keys())
    for n in names:
        preds_list.append(fitted_models[n].predict(X[n]))
    preds = np.column_stack(preds_list)
    # Support global weights (ndarray) or per-regime dict
    if isinstance(weights, dict) and "per_regime" in weights and regimes is not None:
        regs_arr = np.asarray(regimes)
        blended = np.zeros(preds.shape[0], dtype=float)
        for r, w in weights["per_regime"].items():
            mask = regs_arr == int(r)
            if mask.any():
                blended[mask] = preds[mask].dot(np.asarray(w))
        return blended
    else:
        return preds.dot(np.asarray(weights))

# --------------------------
# New: meta stacker comparison
# --------------------------

def stacker_compare(oof_df, y, forward_returns, risk_free):
    # Train meta models on OOFs (simple CV)
    X = oof_df.values
    results = {}
    # Lasso (L1)
    try:
        lasso = Lasso(alpha=1e-3)
        lasso.fit(X, y)
        preds = X.dot(lasso.coef_) + lasso.intercept_
        p_arr = np.asarray(preds, dtype=float)
        p_std = float(np.std(p_arr)) if float(np.std(p_arr)) != 0 else 1.0
        alloc = build_alloc_from_preds_array(p_arr, {'k':1.0,'scale':1.0}, pred_std=p_std, forward_returns=forward_returns)
        results['lasso'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['lasso'] = -1e9
    # BayesianRidge
    try:
        br = BayesianRidge()
        br.fit(X, y)
        preds = br.predict(X)
        p_arr = np.asarray(preds, dtype=float)
        p_std = float(np.std(p_arr)) if float(np.std(p_arr)) != 0 else 1.0
        alloc = build_alloc_from_preds_array(p_arr, {'k':1.0,'scale':1.0}, pred_std=p_std, forward_returns=forward_returns)
        results['bayesridge'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['bayesridge'] = -1e9
    # QuantileRegressor (median)
    try:
        qr = QuantileRegressor(quantile=0.5, alpha=0.1)
        qr.fit(X, y)
        preds = qr.predict(X)
        p_arr = np.asarray(preds, dtype=float)
        p_std = float(np.std(p_arr)) if float(np.std(p_arr)) != 0 else 1.0
        alloc = build_alloc_from_preds_array(p_arr, {'k':1.0,'scale':1.0}, pred_std=p_std, forward_returns=forward_returns)
        results['quantile'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['quantile'] = -1e9
    return results


def save_model(model, model_name, path):
    os.makedirs(path, exist_ok=True)

    if hasattr(model, "save_model"):  
        # XGBoost
        model.save_model(os.path.join(path, f"{model_name}.json"))

    elif model.__class__.__module__.startswith("lightgbm"):
        model.booster_.save_model(os.path.join(path, f"{model_name}.txt"))

    elif model.__class__.__module__.startswith("catboost"):
        model.save_model(os.path.join(path, f"{model_name}.cbm"))

    else:
        # sklearn / MLP / linear / tree
        with open(os.path.join(path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

def load_model(model_name, path, model_type):
    if model_type == "xgb":
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(path, f"{model_name}.json"))
        return model

    if model_type == "lgb":
        return lgb.Booster(model_file=os.path.join(path, f"{model_name}.txt"))

    if model_type == "cat":
        model = cb.CatBoostRegressor()
        model.load_model(os.path.join(path, f"{model_name}.cbm"))
        return model

    with open(os.path.join(path, f"{model_name}.pkl"), "rb") as f:
        return pickle.load(f)

# --------------------------
# Main flow (integrated changes)
# --------------------------

def main(train_path,
         n_splits=4, model_tuning_trials=40, optuna_trials=120, random_state=42, n_jobs=DEFAULT_N_JOBS):
    train = pd.read_csv(train_path)
    train = train.sort_values("date_id").reset_index(drop=True)
    
    print(train.columns)
    # feature engineering: add rolling features (train only)
    train = add_rolling_features(train, price_col='forward_returns')
    print('------------- Added Rolling Features -----------------')
    print(train.columns)
    
    # choose feature columns (exclude known targets/ids)
    EXCLUDE_COLS = {
        "date_id",
        "market_forward_excess_returns",
        "forward_returns",
        "risk_free_rate",
        "lagged_forward_returns",
        "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns",
        "is_scored",
    }
    pre = prepare_datasets(train, None, exclude_cols=EXCLUDE_COLS, early_frac=0.25, rolling_window=252)
    feature_cols = pre["feature_cols"]
    X_for_trees = pre["X_for_trees"].reset_index(drop=True)
    X_for_linear = pre["X_for_linear"].reset_index(drop=True)
    print('------------- Created data arrays for tree/linear models -----------------')
    print('TREE ->', X_for_trees.columns)
    print('LINR ->', X_for_linear.columns)

    # add PCA features to both prepped datasets
    train_pca_df = train.copy()
    train_pca_df, pca_model = add_pca_features(train_pca_df, feature_cols=feature_cols, n_components=3)
    for i in range(3):
        X_for_trees[f'pca_{i}'] = train_pca_df[f'pca_{i}'].values
        X_for_linear[f'pca_{i}'] = train_pca_df[f'pca_{i}'].values
    print('------------- Added PCA Features -----------------')

    y = train["market_forward_excess_returns"].values
    fwd = train["forward_returns"].values
    rf = train["risk_free_rate"].values

    # HMM regime features (train-only): assemble volatility, rates, macro subsets causally
    regime_feature_cols = []
    regime_feature_cols += [c for c in train.columns if c.startswith('V')]
    regime_feature_cols += [c for c in train.columns if c.startswith('I')]
    regime_feature_cols += [c for c in train.columns if c.startswith('E')]
    # include internal realized vol features from rolling block
    regime_feature_cols += [c for c in train.columns if c.startswith('ret_std_') or c.startswith('ret_mad_') or c.startswith('ret_mean_')]
    regime_feature_cols = sorted(list(dict.fromkeys(regime_feature_cols)))
    regime_X = train[regime_feature_cols].fillna(0.0).values
    scaler_regime = StandardScaler()
    regime_X_std = scaler_regime.fit_transform(regime_X)
    hmm_model, regimes = infer_regimes_hmm(regime_X_std, n_states=2)

    # build model templates
    templates = base_model_templates(random_state=random_state)
    model_keys = list(templates.keys())

    # split models into tree vs non-tree for dataset selection
    tree_models = set([k for k in model_keys if k in {"hgb","rf","et","lgb","xgb","cat"}])

    # Stage 1: per-model tuning (parallelized)
    tuned_pipelines = {}
    tuning_results = {}
    print("=== Starting per-model tuning (parallel) ===")

    # run tuning concurrently using ThreadPoolExecutor (avoids pickling ML objects)
    with ThreadPoolExecutor(max_workers=min(len(model_keys), n_jobs)) as exe:
        futures = {}
        for key in model_keys:
            X_use = X_for_trees if key in tree_models else X_for_linear
            fut = exe.submit(tune_single_model, key, templates[key], X_use, y, fwd, rf, n_splits, model_tuning_trials, random_state, 1)
            futures[fut] = (key, X_use)
        for fut in as_completed(futures):
            key, X_used = futures[fut]
            try:
                study, best_pipeline = fut.result()
                tuned_pipelines[key] = best_pipeline
                tuning_results[key] = study
                print(f" Finished tuning {key}. Best value: {study.best_value}")
            except Exception as e:
                print(f" Tuning failed for {key}: {e}")

    # Stage 2: generate OOF preds for all tuned models (cache)
    print("\n=== Generating OOF preds for tuned models ===")
    oof_dict = {}
    for key, pipe in tuned_pipelines.items():
        print(f" OOF for {key} ...")
        X_use = X_for_trees if key in tree_models else X_for_linear
        oof = timeseries_oof_for_model(lambda trial=None, pipe=pipe: pipe, X=X_use, y=y, n_splits=n_splits)        # apply winsorize to OOFs to reduce extremes
        oof = winsorize_mad(oof, n_mad=3.0)
        oof_dict[key] = oof
    oof_df = pd.DataFrame(oof_dict)

    # Stage 3: meta-stacker comparison
    print("\n=== Meta-stacker comparison on OOFs ===")
    stack_res = stacker_compare(oof_df, y, fwd, rf)
    print(stack_res)

    # Stage 4: blend + mapping optuna on OOFs (vectorized)
    print("\n=== Running Optuna to find blend weights + mapping ===")
    study_blend, model_names, best_weights, best_mapping = run_optuna_blend_and_map(oof_df, np.asarray(fwd, dtype=float), np.asarray(rf, dtype=float), n_trials=optuna_trials, n_jobs=n_jobs, random_state=random_state, regimes=regimes)
    print("Best mapping:", best_mapping)
    print("Best weights:", best_weights)

    # Stage 5: fit tuned pipelines on full data (parallel)
    print("\n=== Fitting tuned models on full training data ===")
    X_by_model = {}
    for name in tuned_pipelines.keys():
        X_by_model[name] = X_for_trees if name in tree_models else X_for_linear
    
    print(tuned_pipelines)
    print(X_by_model)
    print(tuned_pipelines.keys())
    print(X_by_model.keys())
    
    fitted = fit_full_models(tuned_pipelines, X_by_model, y, n_jobs=n_jobs)

    if isinstance(best_weights, dict) and 'per_regime' in best_weights:
        regs_arr = np.asarray(regimes)
        blended_oof = np.zeros(oof_df.shape[0], dtype=float)
        for r, w in best_weights['per_regime'].items():
            mask = regs_arr == int(r)
            if mask.any():
                blended_oof[mask] = oof_df.values[mask].dot(np.asarray(w))
    else:
        blended_oof = oof_df.values.dot(np.asarray(best_weights))
    pred_std = blended_oof.std() if blended_oof.std() != 0 else 1.0


    artifacts = {
        "fitted_models": fitted,
        "best_weights": best_weights,
        "best_mapping": best_mapping,
        "pred_std": pred_std,
        "per_model_studies": tuning_results,
        "blend_study": study_blend,
        "pca_model": pca_model,
        "preproc": pre,
        "regime_feature_cols": regime_feature_cols,
        "regime_scaler": scaler_regime,
        "hmm_model": hmm_model
    }
    
    BASE_PATH = "artifacts"
    MODEL_PATH = os.path.join(BASE_PATH, "models")
    META_PATH = os.path.join(BASE_PATH, "metadata")
    PREP_PATH = os.path.join(BASE_PATH, "preprocessors")
    STUDY_PATH = os.path.join(BASE_PATH, "studies")

    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(META_PATH, exist_ok=True)
    os.makedirs(PREP_PATH, exist_ok=True)
    os.makedirs(STUDY_PATH, exist_ok=True)

    # ---- Save models ----
    for name, mdl in fitted.items():
        final_estimator = mdl.steps[-1][1] if hasattr(mdl, "steps") else mdl
        save_model(final_estimator, name, MODEL_PATH)

    # ---- Save metadata ----
    with open(os.path.join(META_PATH, "best_weights.json"), "w") as f:
        json.dump(best_weights, f)

    with open(os.path.join(META_PATH, "best_mapping.json"), "w") as f:
        json.dump(best_mapping, f)

    np.save(os.path.join(META_PATH, "pred_std.npy"), np.array(pred_std))

    with open(os.path.join(META_PATH, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(META_PATH, "regime_feature_cols.json"), "w") as f:
        json.dump(regime_feature_cols, f)

    # ---- Save preprocessors ----
    with open(os.path.join(PREP_PATH, "pca.pkl"), "wb") as f:
        pickle.dump(pca_model, f)

    with open(os.path.join(PREP_PATH, "regime_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_regime, f)

    # ---- Save optuna studies ----
    with open(os.path.join(STUDY_PATH, "per_model_studies.pkl"), "wb") as f:
        pickle.dump(tuning_results, f)

    with open(os.path.join(STUDY_PATH, "blend_study.pkl"), "wb") as f:
        pickle.dump(study_blend, f)


    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/train.csv')

    # parser.add_argument('--n_splits', type=int, default=2)
    # parser.add_argument('--model_tuning_trials', type=int, default=4)
    # parser.add_argument('--optuna_trials', type=int, default=10)

    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--model_tuning_trials', type=int, default=100)
    parser.add_argument('--optuna_trials', type=int, default=500)
    
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--random_state', type=int, default=18)
    
    args = parser.parse_args()


    artifacts = main(
        args.train, 
        n_splits=args.n_splits, 
        model_tuning_trials=args.model_tuning_trials, 
        optuna_trials=args.optuna_trials, 
        random_state=args.random_state, 
        n_jobs=args.n_jobs
    )
    print('Done. Artifacts saved to artifacts_enhanced_with_preproc.joblib')

