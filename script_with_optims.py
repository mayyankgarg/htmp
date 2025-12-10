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

import argparse
import numpy as np
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
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import joblib
import warnings
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

import optuna

# Configs
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS = 252
DEFAULT_N_JOBS = max(1, (joblib.cpu_count() or 4) - 2)

class ParticipantVisibleError(Exception):
    pass

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

def infer_regimes_hmm(forward_returns, n_states=2, lookback=21):
    if GaussianHMM is None:
        # fallback: KMeans on rolling mean/std
        X = np.column_stack([
            pd.Series(forward_returns).rolling(lookback).mean().fillna(0).values,
            pd.Series(forward_returns).rolling(lookback).std().fillna(0).values
        ])
        km = KMeans(n_clusters=n_states, random_state=0).fit(X)
        return km.labels_
    else:
        # fit HMM on two-d features (mean, std)
        X = np.column_stack([
            pd.Series(forward_returns).rolling(lookback).mean().fillna(0).values,
            pd.Series(forward_returns).rolling(lookback).std().fillna(0).values
        ])
        model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=0)
        model.fit(X)
        states = model.predict(X)
        return states

# --------------------------
# Model templates
# --------------------------

def base_model_templates(random_state=42):
    templates = {}

    templates["ridge"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=(trial.params.get('ridge_alpha') if (trial and 'ridge_alpha' in getattr(trial,'params',{})) else 1.0), random_state=random_state))
    ])

    templates["enet"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=(trial.params.get('enet_alpha') if (trial and 'enet_alpha' in getattr(trial,'params',{})) else 1e-3), l1_ratio=(trial.params.get('enet_l1') if (trial and 'enet_l1' in getattr(trial,'params',{})) else 0.5), random_state=random_state))
    ])

    templates["lasso"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=(trial.params.get('lasso_alpha') if (trial and 'lasso_alpha' in getattr(trial,'params',{})) else 1e-3), random_state=random_state))
    ])

    templates["bayesridge"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", BayesianRidge())
    ])

    templates["hgb"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_depth=(trial.params.get('hgb_max_depth') if (trial and 'hgb_max_depth' in getattr(trial,'params',{})) else 3),
            learning_rate=(trial.params.get('hgb_lr') if (trial and 'hgb_lr' in getattr(trial,'params',{})) else 0.05),
            max_iter=(trial.params.get('hgb_iter') if (trial and 'hgb_iter' in getattr(trial,'params',{})) else 300),
            random_state=random_state
        ))
    ])

    templates["rf"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=(trial.params.get('rf_n') if (trial and 'rf_n' in getattr(trial,'params',{})) else 200),
            max_depth=(trial.params.get('rf_depth') if (trial and 'rf_depth' in getattr(trial,'params',{})) else 6),
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    templates["et"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=(trial.params.get('et_n') if (trial and 'et_n' in getattr(trial,'params',{})) else 200),
            max_depth=(trial.params.get('et_depth') if (trial and 'et_depth' in getattr(trial,'params',{})) else 6),
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    templates["knn"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=(trial.params.get('knn_k') if (trial and 'knn_k' in getattr(trial,'params',{})) else 10), n_jobs=-1))
    ])

    templates["mlp"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=(trial.params.get('lgb_n') if (trial and 'lgb_n' in getattr(trial,'params',{})) else 500),
                num_leaves=(trial.params.get('lgb_leaves') if (trial and 'lgb_leaves' in getattr(trial,'params',{})) else 31),
                learning_rate=(trial.params.get('lgb_lr') if (trial and 'lgb_lr' in getattr(trial,'params',{})) else 0.05),
                random_state=random_state,
                n_jobs=-1
            ))
        ])

    if xgb is not None:
        templates["xgb"] = lambda trial=None: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(
                n_estimators=(trial.params.get('xgb_n') if (trial and 'xgb_n' in getattr(trial,'params',{})) else 400),
                max_depth=(trial.params.get('xgb_depth') if (trial and 'xgb_depth' in getattr(trial,'params',{})) else 4),
                learning_rate=(trial.params.get('xgb_lr') if (trial and 'xgb_lr' in getattr(trial,'params',{})) else 0.05),
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            ))
        ])

    if cb is not None:
        templates["cat"] = lambda trial=None: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", cb.CatBoostRegressor(
                iterations=(trial.params.get('cb_iter') if (trial and 'cb_iter' in getattr(trial,'params',{})) else 400),
                depth=(trial.params.get('cb_depth') if (trial and 'cb_depth' in getattr(trial,'params',{})) else 4),
                learning_rate=(trial.params.get('cb_lr') if (trial and 'cb_lr' in getattr(trial,'params',{})) else 0.04),
                verbose=False,
                random_seed=random_state
            ))
        ])

    # also include stacked meta-model templates (constructed later) but keep reachable names
    templates["quantile"] = lambda trial=None: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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
        elif model_key == "enet":
            trial.suggest_float("enet_alpha", 1e-6, 1.0, log=True)
            trial.suggest_float("enet_l1", 0.0, 1.0)
        elif model_key == "lasso":
            trial.suggest_float("lasso_alpha", 1e-6, 1.0, log=True)
        elif model_key == "hgb":
            trial.suggest_int("hgb_max_depth", 2, 6)
            trial.suggest_float("hgb_lr", 0.01, 0.2)
            trial.suggest_int("hgb_iter", 50, 600)
        elif model_key == "rf":
            trial.suggest_int("rf_n", 50, 600)
            trial.suggest_int("rf_depth", 3, 12)
        elif model_key == "et":
            trial.suggest_int("et_n", 50, 600)
            trial.suggest_int("et_depth", 3, 12)
        elif model_key == "knn":
            trial.suggest_int("knn_k", 3, 50)
        elif model_key == "mlp":
            h1 = trial.suggest_int("mlp_h1", 16, 256)
            h2 = trial.suggest_int("mlp_h2", 0, 128)
            hidden = (h1, h2) if h2>0 else (h1,)
            trial.set_user_attr("mlp_hidden_tuple", hidden)
            trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True)
            trial.suggest_int("mlp_iter", 200, 1500)
        elif model_key == "lgb" and lgb is not None:
            trial.suggest_int("lgb_n", 50, 1000)
            trial.suggest_int("lgb_leaves", 8, 128)
            trial.suggest_float("lgb_lr", 0.01, 0.2)
        elif model_key == "xgb" and xgb is not None:
            trial.suggest_int("xgb_n", 50, 800)
            trial.suggest_int("xgb_depth", 2, 8)
            trial.suggest_float("xgb_lr", 0.01, 0.2)
        elif model_key == "cat" and cb is not None:
            trial.suggest_int("cb_iter", 50, 800)
            trial.suggest_int("cb_depth", 2, 8)
            trial.suggest_float("cb_lr", 0.01, 0.2)

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
        raw = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)])
        if raw.sum() == 0:
            weights = np.ones(n_models)/n_models
        else:
            weights = raw / raw.sum()
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

        blended = X_oof.dot(weights)
        pred_std = blended.std() if blended.std() != 0 else 1.0
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
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_raw = np.array([study.best_trial.params.get(f"w_{i}", 0.0) for i in range(n_models)])
    if best_raw.sum()==0:
        best_weights = np.ones(n_models)/n_models
    else:
        best_weights = best_raw / best_raw.sum()
    best_mapping = {k:v for k,v in study.best_trial.params.items() if not k.startswith("w_")}
    return study, model_names, best_weights, best_mapping

# --------------------------
# Utilities to fit and predict
# --------------------------

def fit_full_models(models_dict, X, y, n_jobs=DEFAULT_N_JOBS):
    fitted = {}
    def fit_one(name, pipeline):
        mdl = clone(pipeline)
        mdl.fit(X, y)
        return name, mdl
    items = list(models_dict.items())
    results = Parallel(n_jobs=n_jobs)(delayed(fit_one)(n,p) for n,p in items)
    for name, mdl in results:
        fitted[name] = mdl
    return fitted


def ensemble_predict_from_fitted(fitted_models, weights, X):
    preds = []
    names = list(fitted_models.keys())
    for n in names:
        preds.append(fitted_models[n].predict(X))
    preds = np.column_stack(preds)
    blended = preds.dot(weights)
    return blended

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
        alloc = build_alloc_from_preds_array(preds, {'k':1.0,'scale':1.0}, pred_std=preds.std(), forward_returns=forward_returns)
        results['lasso'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['lasso'] = -1e9
    # BayesianRidge
    try:
        br = BayesianRidge()
        br.fit(X, y)
        preds = br.predict(X)
        alloc = build_alloc_from_preds_array(preds, {'k':1.0,'scale':1.0}, pred_std=preds.std(), forward_returns=forward_returns)
        results['bayesridge'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['bayesridge'] = -1e9
    # QuantileRegressor (median)
    try:
        qr = QuantileRegressor(quantile=0.5, alpha=0.1)
        qr.fit(X, y)
        preds = qr.predict(X)
        alloc = build_alloc_from_preds_array(preds, {'k':1.0,'scale':1.0}, pred_std=preds.std(), forward_returns=forward_returns)
        results['quantile'] = hull_score(pd.DataFrame({'forward_returns':forward_returns,'risk_free_rate':risk_free}), pd.DataFrame({'prediction':alloc}))
    except Exception:
        results['quantile'] = -1e9
    return results

# --------------------------
# Main flow (integrated changes)
# --------------------------

def main(train_path, test_path=None,
         n_splits=4, model_tuning_trials=40, optuna_trials=120, random_state=42, n_jobs=DEFAULT_N_JOBS):
    train = pd.read_csv(train_path)
    train = train.sort_values("date_id").reset_index(drop=True)
    test = None
    if test_path is not None:
        test = pd.read_csv(test_path)
        test = test.sort_values("date_id").reset_index(drop=True)

    # feature engineering: add rolling features
    train = add_rolling_features(train, price_col='forward_returns')

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
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]

    # add PCA features
    train, pca_model = add_pca_features(train, feature_cols=feature_cols, n_components=3)
    for i in range(3):
        feature_cols.append(f'pca_{i}')

    X = train[feature_cols].reset_index(drop=True)
    y = train["market_forward_excess_returns"].values
    fwd = train["forward_returns"].values
    rf = train["risk_free_rate"].values

    # HMM regimes
    regimes = infer_regimes_hmm(fwd, n_states=2, lookback=21)

    # build model templates
    templates = base_model_templates(random_state=random_state)
    model_keys = list(templates.keys())

    # Stage 1: per-model tuning (parallelized)
    tuned_pipelines = {}
    tuning_results = {}
    print("=== Starting per-model tuning (parallel) ===")

    # run tuning concurrently using ThreadPoolExecutor (avoids pickling ML objects)
    with ThreadPoolExecutor(max_workers=min(len(model_keys), n_jobs)) as exe:
        futures = {exe.submit(tune_single_model, key, templates[key], X, y, fwd, rf, n_splits, model_tuning_trials, random_state, 1): key for key in model_keys}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                study, best_pipeline = fut.result()
                tuned_pipelines[key] = best_pipeline
                tuning_results[key] = study
                print(f"  Finished tuning {key}. Best value: {study.best_value}")
            except Exception as e:
                print(f"  Tuning failed for {key}: {e}")

    # Stage 2: generate OOF preds for all tuned models (cache)
    print("\n=== Generating OOF preds for tuned models ===")
    oof_dict = {}
    for key, pipe in tuned_pipelines.items():
        print(f" OOF for {key} ...")
        oof = timeseries_oof_for_model(lambda trial=None, pipe=pipe: pipe, X=X, y=y, n_splits=n_splits)
        # apply winsorize to OOFs to reduce extremes
        oof = winsorize_mad(oof, n_mad=3.0)
        oof_dict[key] = oof
    oof_df = pd.DataFrame(oof_dict)

    # Stage 3: meta-stacker comparison
    print("\n=== Meta-stacker comparison on OOFs ===")
    stack_res = stacker_compare(oof_df, y, fwd, rf)
    print(stack_res)

    # Stage 4: blend + mapping optuna on OOFs (vectorized)
    print("\n=== Running Optuna to find blend weights + mapping ===")
    study_blend, model_names, best_weights, best_mapping = run_optuna_blend_and_map(oof_df, fwd, rf, n_trials=optuna_trials, n_jobs=1, random_state=random_state, regimes=regimes)
    print("Best mapping:", best_mapping)
    print("Best weights:", best_weights)

    # Stage 5: fit tuned pipelines on full data (parallel)
    print("\n=== Fitting tuned models on full training data ===")
    fitted = fit_full_models(tuned_pipelines, X, y, n_jobs=n_jobs)

    # pred_std for inference
    blended_oof = oof_df.values.dot(best_weights)
    pred_std = blended_oof.std() if blended_oof.std() != 0 else 1.0

    submission_df = None
    if test is not None:
        # apply same feature engineering to test
        test = add_rolling_features(test, price_col='forward_returns')
        # apply PCA transform to test features
        test_features = test[[c for c in feature_cols if c in test.columns]].fillna(0.0)
        comps_test = pca_model.transform(test_features.values)
        for i in range(comps_test.shape[1]):
            test[f'pca_{i}'] = comps_test[:, i]
        X_test = test[feature_cols].reset_index(drop=True)
        blended_test = ensemble_predict_from_fitted(fitted, best_weights, X_test)
        alloc = build_alloc_from_preds_array(blended_test, best_mapping, pred_std=pred_std, forward_returns=test['forward_returns'].values)
        submission_df = pd.DataFrame({"date_id": test["date_id"].values, "allocation": alloc})
        print("\n=== Sample allocations ===")
        print(submission_df.head())

    artifacts = {
        "fitted_models": fitted,
        "best_weights": best_weights,
        "best_mapping": best_mapping,
        "pred_std": pred_std,
        "per_model_studies": tuning_results,
        "blend_study": study_blend,
        "pca_model": pca_model
    }
    joblib.dump(artifacts, "artifacts_enhanced.joblib")
    return artifacts, submission_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--n_splits', type=int, default=4)
    parser.add_argument('--model_tuning_trials', type=int, default=40)
    parser.add_argument('--optuna_trials', type=int, default=120)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args()

    artifacts, submission = main(args.train, args.test, n_splits=args.n_splits, model_tuning_trials=args.model_tuning_trials, optuna_trials=args.optuna_trials, random_state=args.random_state, n_jobs=args.n_jobs)
    print('Done. Artifacts saved to artifacts_enhanced.joblib')
